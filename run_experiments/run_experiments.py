# Add these imports if not already present
import matplotlib.pyplot as plt
import itertools
from datasets import load_dataset
import gc
import os
import torch
import torch.nn as nn
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import itertools
import tqdm
import gc
from functools import partial
from smoothquant.fake_quant import quantize_llama_like, quantize_opt
from smoothquant.model_size import get_model_size

# Constants for size calculations
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=10, batch_size=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(
            1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * self.batch_size): ((i + 1) * self.batch_size)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * self.batch_size): ((i + 1) * self.batch_size)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)
                                  ), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * self.batch_size
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * self.batch_size))

def get_calib_feat(model, tokenizer, n_samples=128, block_size=512):
    """Get calibration features for the model using a sample dataset."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    
    # Process and pad samples to the same length
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        # Pad sequence to block_size
        if len(line_encoded) < block_size:
            line_encoded = line_encoded + [tokenizer.pad_token_id] * (block_size - len(line_encoded))
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run >= n_samples:
            break
    
    if len(samples) == 0:
        raise ValueError("No valid samples found for calibration")
    
    # Stack all samples into a single batch
    samples = torch.cat(samples, dim=0)
    
    # Prepare input features
    model.eval()
    input_feat = {}
    
    with torch.no_grad():
        # Process the entire batch at once
        samples = samples.to(model.device)
        out = model(samples, output_hidden_states=True)
        hidden_states = out.hidden_states
        
        # Store each hidden state
        for idx, hidden in enumerate(hidden_states):
            input_feat[idx] = hidden
    
    return input_feat

def run_experiments(model_name, group_sizes=[4, 8, 16, 32, 64, 128, 256, 512, 1024], 
                   salient_props=[0, 0.01, 0.05, 0.1], device="cuda"):
    """
    Run experiments for different group sizes and salient proportions.
    Returns perplexity and model size results.
    """
    results = {
        'perplexity': {},
        'model_size': {}
    }
    
    for salient_prop in salient_props:
        print(f"\nTesting salient proportion: {salient_prop}")
        perplexities = []
        model_sizes = []
        
        for group_size in group_sizes:
            print(f"\nGroup size: {group_size}")
            
            # Load model
            if 'llama' in model_name.lower():
                model_fp16 = LlamaForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto"
                )
                tokenizer = LlamaTokenizer.from_pretrained(model_name)
                quantize_fn = quantize_llama_like
            else:  # OPT model
                model_fp16 = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                quantize_fn = quantize_opt
            
            # Get calibration features
            input_feat = get_calib_feat(model_fp16, tokenizer)
            
            # Quantize model
            model_w4a4 = quantize_fn(
                model_fp16, 
                weight_quant="per_group",
                act_quant="per_group",
                input_feat=input_feat,
                salient_prop=salient_prop,
                quant_bits=4,
                group_size=group_size
            )
            
            # Evaluate perplexity
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            evaluator = Evaluator(dataset, tokenizer, device)
            ppl = evaluator.evaluate(model_w4a4)
            perplexities.append(ppl.item())
            
            # Calculate model size
            model_size = get_model_size(model_w4a4, data_width=4, 
                                      salient_prop=salient_prop, 
                                      group_size=group_size) / MiB
            model_sizes.append(model_size)
            
            # Clear memory
            del model_fp16
            del model_w4a4
            gc.collect()
            torch.cuda.empty_cache()
            
        results['perplexity'][salient_prop] = perplexities
        results['model_size'][salient_prop] = model_sizes
    
    return results, group_sizes

def plot_results(results, group_sizes, model_name, output_folder="./figures"):
    """Plot perplexity and model size results."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot settings
    markers = itertools.cycle(['o', 's', '^', 'd'])
    colors = itertools.cycle(['darkblue', 'mediumseagreen', 'orange', 'darkviolet'])
    linestyles = itertools.cycle(['-', '--', '-.', ':'])
    
    # Plot perplexity
    plt.figure(figsize=(10, 6))
    for salient_prop, perplexities in results['perplexity'].items():
        plt.plot(
            group_sizes,
            perplexities,
            marker=next(markers),
            label=f'Salient Prop = {salient_prop}',
            color=next(colors),
            linestyle=next(linestyles),
            markersize=8,
            linewidth=2
        )
    
    plt.xscale('log')
    plt.xlabel('Group Size', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(f'Model ({model_name}) Perplexity vs. Group Size for Different Salient Proportions', 
              fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title="Salient Proportion", loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{model_name.split('/')[-1]}_ppl_vs_group_size.png")
    plt.close()
    
    # Plot model size
    plt.figure(figsize=(10, 6))
    for salient_prop, sizes in results['model_size'].items():
        plt.plot(
            group_sizes,
            sizes,
            marker=next(markers),
            label=f'Salient Prop = {salient_prop}',
            color=next(colors),
            linestyle=next(linestyles),
            markersize=8,
            linewidth=2
        )
    
    plt.xscale('log')
    plt.xlabel('Group Size', fontsize=12)
    plt.ylabel('Model Size (MiB)', fontsize=12)
    plt.title(f'Model ({model_name}) Size vs. Group Size for Different Salient Proportions', 
              fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title="Salient Proportion", loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{model_name.split('/')[-1]}_size_vs_group_size.png")
    plt.close()

# Run experiments for both models
if __name__ == "__main__":
    # Test configurations
    models = [
        "facebook/opt-1.3b",
        "NousResearch/Llama-2-7b-hf"
    ]
    group_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    salient_props = [0, 0.01, 0.05, 0.1]
    
    for model_name in models:
        print(f"\nRunning experiments for {model_name}")
        results, group_sizes = run_experiments(
            model_name=model_name,
            group_sizes=group_sizes,
            salient_props=salient_props
        )
        plot_results(results, group_sizes, model_name)