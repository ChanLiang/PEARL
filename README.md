# PEARL

Implementation for our ICLR 2025 paper [PEARL: Towards Permutation-Resilient LLMs](https://openreview.net/forum?id=txoJvjfI9w).

**PEARL** is an instruction tuning method that helps LLMs better handle set-structured inputs with order-independent elements — making them more robust in tasks such as in-context learning (ICL) and retrieval-augmented generation (RAG).

## Getting Started

### 1. Environment Setup

Set up the environment using Conda and the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

If you encounter missing dependencies, please install them manually.

### 2. Data and Resources

- Experimental datasets are located in the `niv2_data/` directory.
- Models should be downloaded and placed in the same paths as specified in the scripts.

### 3. Instruction Tuning

Run the following scripts to perform instruction tuning for PEARL and the baseline:

```bash
bash script/niv2_exp_pearl.sh
bash script/niv2_exp_baseline.sh
```

### 4. Inference on All Permutations

You can also run inference directly during the training experiments.

```bash
bash script/eval.sh
bash script/eval_manyshot.sh
```

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{chen2025pearl,
  title={{PEARL}: Towards Permutation-Resilient {LLM}s},
  author={Liang Chen and Li Shen and Yang Deng and Xiaoyan Zhao and Bin Liang and Kam-Fai Wong},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=txoJvjfI9w}
}
```
