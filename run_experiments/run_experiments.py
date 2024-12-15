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

def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

def get_calib_feat(model, tokenizer):
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return input_dict

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
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * self.batch_size
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * self.batch_size))

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
                    model_name, device_map="auto"
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
                group_size=group_size
            )
            
            # Evaluate perplexity
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            evaluator = Evaluator(dataset, tokenizer, device)
            ppl = evaluator.evaluate(model_w4a4)
            perplexities.append(ppl.item())
            
            # Calculate model size
            model_size = get_model_size(model_w4a4, data_width=4, group_size=group_size)
            model_sizes.append(model_size / MiB)  # Convert to MiB
            
            # Clear memory
            del model_fp16
            del model_w4a4
            gc.collect()
            torch.cuda.empty_cache()
            
        results['perplexity'][salient_prop] = perplexities
        results['model_size'][salient_prop] = model_sizes
    
    return results, group_sizes

def plot_results(results, group_sizes, model_name, output_folder="figures"):
    os.makedirs(output_folder, exist_ok=True)
    
    markers = itertools.cycle(['o', 's', '^', 'D'])
    colors = itertools.cycle(['blue', 'green', 'orange', 'purple'])
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