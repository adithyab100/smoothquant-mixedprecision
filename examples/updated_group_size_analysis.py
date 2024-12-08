import os
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import tqdm
import gc
from functools import partial
from smoothquant.fake_quant import quantize_model

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

def evaluate_group_size(model_path, group_sizes, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, device, n_samples=10)
    
    perplexities = []
    
    for group_size in group_sizes:
        print(f"\nTesting group size: {group_size}")
        
        # Load model fresh each time to avoid contamination
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device
        )
        
        # Quantize model with current group size
        model = quantize_model(
            model,
            group_size=group_size,
            act_quant="per_token"
        )
        
        # Evaluate
        ppl = evaluator.evaluate(model)
        perplexities.append(ppl)
        print(f"Perplexity for group size {group_size}: {ppl}")
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
    
    return perplexities

def plot_results(group_sizes, perplexities, output_path="figures/group_size_perplexity.png"):
    os.makedirs("figures", exist_ok=True)
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
    model_path = "NousResearch/Llama-2-7b-hf"  # Using Llama-2-7b
    group_sizes = [32, 64, 128, 256, 512, 1024, 2048]  # Updated group sizes
    
    # Run evaluation
    perplexities = evaluate_group_size(model_path, group_sizes)
    
    # Plot results
    plot_results(group_sizes, perplexities)
    
    # Print final results
    print("\nFinal Results:")
    for size, ppl in zip(group_sizes, perplexities):
        print(f"Group Size: {size}, Perplexity: {ppl}")
