# Representational Geometry and Grokking in Image Classification

This repository contains code for analyzing delayed generalization (grokking) in image classification tasks, focusing on the relationship between Neural Tangent Kernel (NTK) changes and representational geometry metrics.

The work is accepted at the 2024 Unireps workshop ([link](https://openreview.net/forum?id=1ae108kHk2&noteId=1ae108kHk2)).


We implement recipes to induce grokking and includes scripts for:
1. Training models that exhibit grokking
2. Measuring manifold capacity of layer activations (cloned from [Cohen et al., 2020](https://www.nature.com/articles/s41467-020-14578-5))
3. Computing kernel distance metrics


### Setup

Two conda environments are required:

```bash
# For model training, and kernel distance analysis
conda env create -f model_training_environment.yml

# For capacity measurements
conda env create -f manifold_environment_simple.yml
```

Setting up network module:
```bash
source setup_env.sh 
```

### Example training Models
```bash
python scripts/example_model_train.py 
```


## Citation
```bibtex
[paper's citation]
```


