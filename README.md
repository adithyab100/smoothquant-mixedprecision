# W4A4 MixedQuant: Mixed Precision and Group Quantization for LLM Compression

This repository implements W4A4 MixedQuant, a novel quantization approach that enables 4-bit quantization for both weights and activations while maintaining model accuracy. Our method combines mixed-precision preservation with sorted group quantization to achieve significant model compression with minimal performance degradation.

## Quick Start

### Requirements

### Running Experiments

Two options:

1. Run all experiments: `./run.sh`
2. Interactive notebooks:
    - `examples/smoothquant_llama_demo.ipynb`: Main demo notebook

Requires high-memory GPU (A100 recommended, T4 insufficient for LLaMA models)

## Method Overview

Our approach addresses limitations of existing methods through three key innovations:

1. **Mixed-Precision Preservation**

    - Identifies and preserves 5-10% of salient channels in FP16
    - Quantizes remaining channels to INT4
    - Maintains critical information while reducing overall size

2. **Group Quantization**

    - Divides channels into groups (typically 64-128)
    - Applies quantization independently to each group
    - Balances compression and accuracy

3. **Channel Sorting Strategies**
    - Maximum value sorting
    - Statistical sorting (mean + 3\*STD)
    - Position-based sorting (argmax)

## Key Results

### OPT-1.3B

-   Maintains perplexity below 20 for all tested group sizes with 5-10% salient weights
-   Significant size reduction while preserving model performance

### Llama-2-7B

-   Model size reduced from ~12852 MiB to ~8193 MiB with 10% salient proportion
-   Perplexity increase of only 0.07 (5.8919 vs 5.823 FP16)
-   With 5% salient proportion and group size 64: halved model size with <0.3 perplexity increase

### Channel Sorting Impact

-   Max value and mean+3STD sorting significantly improve perplexity for large group sizes
-   Example: At group size 1024, perplexity improved from 59.98 (unsorted) to 19.56 (max sorting)

## Repository Structure

```python
smoothquant-mixedprecision/
├── run.sh
├── examples/                  # Demo notebooks
├── smoothquant/              # Core implementation
│   ├── fake_quant.py         # Quantization implementations
├── run_experiments/          # Experiment scripts
└── figures/                  # Generated plots and results
```

## Citation

```bibtex
@article{balachandran2024mixedquant,
  title={MixedQuant: Mixed Precision and Group Quantization for LLM Compression},
  author={Balachandran, Adithya and Liu, Andi and Ma, Ningshan and Wang, Amber and Zhou, Jonathan},
  year={2024}
}
```

## Acknowledgements

This project was supported by the MIT Han Lab and MIT 6.5940: TinyML and Efficient Deep Learning Computing.

## License

MIT License
