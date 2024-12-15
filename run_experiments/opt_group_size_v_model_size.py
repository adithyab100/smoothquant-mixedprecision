import os
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import itertools
import tqdm
import gc
from functools import partial
from smoothquant.fake_quant import quantize_llama_like, quantize_opt
from smoothquant.model_size import get_model_size

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


@torch.no_grad()
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


def evaluate_opt_group_model_size(model_path, salient_prop, group_sizes, device="cuda" if torch.cuda.is_available() else "cpu"):
    # load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, device, n_samples=40)
    input_feat = get_calib_feat(model_fp16, tokenizer)
    model_size = []

    for group_size in group_sizes:
        print(f"\nTesting group size: {group_size}")

        model_fp16 = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto")
        model_w4a4 = quantize_opt(model_fp16, weight_quant="per_group", act_quant="per_group",
                                  input_feat=input_feat, salient_prop=salient_prop, quant_bits=4, group_size=group_size)

        # compute model size
        model_sz = get_model_size(
            model_w4a4, data_width=4, salient_prop=salient_prop, group_size=group_size)
        model_size.append(model_sz / MiB)  # can change depending on unit
        print(f"Model size for group size {group_size}: {model_sz}")

        # Clear memory
        del model_fp16
        del model_w4a4
        gc.collect()
        torch.cuda.empty_cache()

    return model_size


def plot_results(group_sizes, size_series, output_path="group_size_model_size_pretty.png"):
    """
    args:
        group_sizes (list): group sizes (x-axis values).
        size_series (dict): dictionary where keys are salient proportions (labels) and values are size lists.
        output_path (str): file path to save the plot.
    """

    markers = itertools.cycle(['o', 's', '^', 'd', 'p', '*'])
    colors = itertools.cycle(
        ['darkblue', 'mediumseagreen', 'orange', 'darkviolet', 'crimson', 'gold'])
    linestyles = itertools.cycle(['-', '--', '-.', ':', '-'])

    plt.figure(figsize=(10, 6))

    for salient_prop, sizes in size_series.items():
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

    plt.xlabel('Group Size', fontsize=12)
    plt.ylabel('Model Size (in MiB)', fontsize=12)
    plt.title(
        'Model (OPT-1.3B) Model Size vs. Group Size for Different Salient Proportions', fontsize=14)

    # only log scale for x and not y
    plt.xscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend(title="Salient Proportion", loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    model_path = "facebook/opt-1.3b"  # use opt
    output_folder = "./figures"
    group_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    salient_props = [0, 0.01, 0.05, 0.1]

    model_size_by_salient_props = {}
    for salient_prop in salient_props:
        print(f"\nRunning for salient proportion: {salient_prop}")
        model_size_by_salient_props.update(
            {salient_prop: evaluate_opt_group_model_size(model_path, salient_prop, group_sizes)})

    # plot
    print(f"\nPlotting results...")
    plot_results(group_sizes, model_size_by_salient_props,
                 output_path=output_folder + "/opt_model_size_v_group_size.png")
