## Mixed Precision and Group Quantization for LLM
Adithya Balachandran, Andi Liu, Ningshan Ma, Amber Wang, Jonathan Zhou

This is the final project for class 6.5940 (Fa 24).

# Introduction

## Background
Large language models (LLMs) demand
substantial computational and memory resources. In
recent years, the model size of LLMs is developing at a
faster pace than GPU memory, increasing the need for
effective model compression techniques.

## Current Solutions:
1. Activation-aware weight quantization: W4A16 quan-
tization by preserving salient weights
•Higher precision activations still remain a bottleneck
for computational performance
2. SmoothQuant: pre-smooths activations before quan-
tization to achieve W8A8 quantization
•Loses 4-bit weight quantization of AWQ
We propose W4A4 SmoothQuant, a quantization
method that performs 4-bit quantization for both
weights and activations.

## Main Contributions:
• Preserve salient weights and activations (mixed-pre-
cision approach) to maintain accuracy, which will be
measured by perplexity.
• Leverage group quantization to preserve more infor-
mation in the 4-bit weights/activations.
•Achieve a perplexity that within 0.5 the original
perplexity of the FP16 model.


## Experimental Results
