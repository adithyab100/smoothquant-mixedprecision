import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
from smoothquant.fake_quant import quantize_model
import torch.nn as nn

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
        for i in range(n_samples):
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

def evaluate_group_sizes(model_path, group_sizes, device="cpu"):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, device, n_samples=10)  # Reduced samples for faster testing
    perplexities = []

    for group_size in group_sizes:
        print(f"\nTesting group size: {group_size}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_quantized = quantize_model(model, group_size=group_size, act_quant="per_token")
        ppl = evaluator.evaluate(model_quantized)
        perplexities.append(ppl)
        print(f"Group size {group_size} perplexity: {ppl:.2f}")

    return group_sizes, perplexities

def plot_results(group_sizes, perplexities):
    plt.figure(figsize=(10, 6))
    plt.plot(group_sizes, perplexities, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Group Size')
    plt.ylabel('Perplexity')
    plt.title('Impact of Group Size on Model Perplexity')
    plt.grid(True)
    plt.savefig('group_size_analysis.png')
    plt.show()

if __name__ == "__main__":
    model_path = "NousResearch/Llama-2-7b-hf"  # Updated model path
    group_sizes = [32, 64, 128, 256, 512, 1024, 2048]  # Different group sizes to test
    group_sizes, perplexities = evaluate_group_sizes(model_path, group_sizes)
    plot_results(group_sizes, perplexities)
