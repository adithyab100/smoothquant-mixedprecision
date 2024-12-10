import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from smoothquant.fake_quant import quantize_model

class SaliencyEvaluator:
    def __init__(self, model_path, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", 
                 split="test", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        
        # Load and preprocess dataset
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        self.input_ids = self.tokenizer(
            "\n\n".join(dataset["text"]), 
            return_tensors="pt"
        ).input_ids.to(device)
        
        # Initialize loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def load_model(self, dtype=None):
        """Load model with specified dtype"""
        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device
        )
    
    def evaluate_batch(self, model, input_ids, chunk_size=2048):
        """Evaluate a single batch"""
        with torch.no_grad():
            outputs = model(input_ids[:, :chunk_size])
            logits = outputs.logits[:, :-1].contiguous().float()
            labels = input_ids[:, 1:chunk_size].contiguous()
            
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
        return loss.item() * (chunk_size - 1)
    
    def evaluate_perplexity(self, model, num_batches=10, chunk_size=2048):
        """Calculate perplexity over multiple batches"""
        model.eval()
        total_nll = 0
        total_tokens = 0
        
        for i in range(num_batches):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            
            if end_idx > self.input_ids.size(1):
                break
                
            batch_nll = self.evaluate_batch(
                model, 
                self.input_ids[:, start_idx:end_idx],
                chunk_size
            )
            
            total_nll += batch_nll
            total_tokens += (chunk_size - 1)
            
        return torch.exp(torch.tensor(total_nll / total_tokens))

def get_weight_saliency(model):
    """Calculate weight saliency for each linear layer"""
    saliency_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Calculate weight magnitudes
            weight_magnitudes = torch.abs(module.weight.data)
            saliency_dict[name] = weight_magnitudes.flatten()
    
    return saliency_dict

def apply_mixed_precision(model, salient_proportion, group_size=128):
    """Apply mixed precision based on weight saliency"""
    # Get weight saliency for all layers
    saliency_dict = get_weight_saliency(model)
    
    # Calculate global threshold based on salient proportion
    all_magnitudes = torch.cat(list(saliency_dict.values()))
    threshold = torch.quantile(all_magnitudes, 1 - salient_proportion)
    
    # Quantize model with different precisions based on saliency
    model = quantize_model(
        model,
        group_size=group_size,
        act_quant="per_token",
        weight_threshold=threshold
    )
    
    return model

def analyze_saliency_proportions(model_path, proportions, group_size=128, num_batches=10, save_dir="results"):
    """Analyze model performance with different salient weight proportions"""
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    # Initialize evaluator
    evaluator = SaliencyEvaluator(model_path)
    
    # Test each proportion
    for proportion in tqdm(proportions, desc="Testing salient proportions"):
        print(f"\nEvaluating salient proportion: {proportion:.2f}")
        
        # Load fresh model
        model = evaluator.load_model()
        
        # Apply mixed precision quantization
        model = apply_mixed_precision(model, proportion, group_size)
        
        # Evaluate perplexity
        perplexity = evaluator.evaluate_perplexity(model, num_batches)
        results.append((proportion, perplexity.item()))
        
        print(f"Salient proportion {proportion:.2f}: Perplexity = {perplexity:.2f}")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def plot_results(results, save_dir="results"):
    """Plot and save results"""
    proportions, perplexities = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(proportions, perplexities, 'bo-', linewidth=2, markersize=8)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.xlabel('Salient Weights Proportion', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Impact of Salient Weights Proportion on Model Perplexity', fontsize=14)
    
    # Add value labels
    for x, y in zip(proportions, perplexities):
        plt.annotate(f'{y:.1f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'saliency_perplexity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open(os.path.join(save_dir, 'saliency_results.txt'), 'w') as f:
        f.write("Salient Proportion\tPerplexity\n")
        for prop, ppl in results:
            f.write(f"{prop:.2f}\t{ppl:.2f}\n")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "NousResearch/Llama-2-7b-hf"
    PROPORTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Different proportions of salient weights
    GROUP_SIZE = 128  # Fixed group size for this analysis
    NUM_BATCHES = 10
    SAVE_DIR = "saliency_analysis_results"
    
    # Run analysis
    results = analyze_saliency_proportions(
        model_path=MODEL_PATH,
        proportions=PROPORTIONS,
        group_size=GROUP_SIZE,
        num_batches=NUM_BATCHES,
        save_dir=SAVE_DIR
    )
    
    # Plot and save results
    plot_results(results, save_dir=SAVE_DIR)
