import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import tqdm

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=10):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

def evaluate_group_size(model_path, group_sizes, device="cpu"):
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, device, n_samples=10)  # Reduced samples for faster testing
    
    perplexities = []
    
    # Load activation scales
    try:
        act_scales = torch.load("../act_scales/llama-2-7b.pt")  # Adjust path as needed
    except:
        print("No activation scales found, proceeding without smoothing")
        act_scales = None
    
    for group_size in group_sizes:
        print(f"\nTesting group size: {group_size}")
        
        # Load model fresh each time to avoid contamination
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"  # Use CPU
        )
        
        # Apply smoothing if scales are available
        if act_scales is not None:
            smooth_lm(model, act_scales, 0.85)
        
        # Quantize model with current group size
        model = quantize_model(
            model,
            weight_quant="per_group",
            act_quant="per_token",
            quantize_bmm_input=True,
            group_size=group_size,
            input_feat=None  # No input features available
        )
        
        # Evaluate
        ppl = evaluator.evaluate(model)
        perplexities.append(ppl)
        print(f"Perplexity for group size {group_size}: {ppl}")
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
    
    return perplexities

def plot_results(group_sizes, perplexities, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(group_sizes, perplexities, 'bo-')
    plt.xlabel('Group Size')
    plt.ylabel('Perplexity')
    plt.title('Model Perplexity vs. Group Size')
    plt.grid(True)
    plt.yscale('log')  # Log scale for perplexity
    plt.xscale('log')  # Log scale for group size
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Configuration
    model_path = "facebook/opt-125m"  # Using OPT-125M which is publicly available
    group_sizes = [8, 16, 32, 64, 128]  # Different group sizes to test
    
    # Run evaluation
    perplexities = evaluate_group_size(model_path, group_sizes)
    
    # Plot results
    plot_results(group_sizes, perplexities, "group_size_perplexity.png")
    
    # Print final results
    print("\nFinal Results:")
    for size, ppl in zip(group_sizes, perplexities):
        print(f"Group Size: {size}, Perplexity: {ppl}")
