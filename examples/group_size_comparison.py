import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from smoothquant.fake_quant import quantize_model

class PerplexityEvaluator:
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

def analyze_group_sizes(model_path, group_sizes, num_batches=10, save_dir="results"):
    """Analyze model performance with different group sizes"""
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    # Initialize evaluator
    evaluator = PerplexityEvaluator(model_path)
    
    # Test each group size
    for group_size in tqdm(group_sizes, desc="Testing group sizes"):
        print(f"\nEvaluating group size: {group_size}")
        
        # Load fresh model
        model = evaluator.load_model()
        
        # Quantize model
        model = quantize_model(
            model,
            group_size=group_size,
            act_quant="per_token"
        )
        
        # Evaluate perplexity
        perplexity = evaluator.evaluate_perplexity(model, num_batches)
        results.append((group_size, perplexity.item()))
        
        print(f"Group size {group_size}: Perplexity = {perplexity:.2f}")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def plot_results(results, save_dir="results"):
    """Plot and save results"""
    group_sizes, perplexities = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(group_sizes, perplexities, 'bo-', linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.xlabel('Group Size', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Impact of Group Size on Model Perplexity', fontsize=14)
    
    # Add value labels
    for x, y in zip(group_sizes, perplexities):
        plt.annotate(f'{y:.1f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'group_size_perplexity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open(os.path.join(save_dir, 'group_size_results.txt'), 'w') as f:
        f.write("Group Size\tPerplexity\n")
        for size, ppl in results:
            f.write(f"{size}\t{ppl:.2f}\n")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "NousResearch/Llama-2-7b-hf"
    GROUP_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
    NUM_BATCHES = 10
    SAVE_DIR = "group_size_analysis_results"
    
    # Run analysis
    results = analyze_group_sizes(
        model_path=MODEL_PATH,
        group_sizes=GROUP_SIZES,
        num_batches=NUM_BATCHES,
        save_dir=SAVE_DIR
    )
    
    # Plot and save results
    plot_results(results, save_dir=SAVE_DIR)
