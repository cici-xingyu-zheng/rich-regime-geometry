# Representational Geometry and Grokking in Image Classification

This repository contains code for analyzing delayed generalization (grokking) in image classification tasks, focusing on the relationship between Neural Tangent Kernel (NTK) changes and representational geometry metrics.

This work has been accepted at the 2024 Unireps workshop ([link](https://openreview.net/forum?id=1ae108kHk2&noteId=1ae108kHk2)).

The repository implements methods to induce grokking and includes scripts for:
1. Training models that exhibit grokking
2. Measuring manifold capacity of layer activations (based on [Cohen et al., 2020](https://www.nature.com/articles/s41467-020-14578-5))
3. Computing kernel distance metrics

### Setup

Two conda environments are required:

```bash
# For model training and kernel distance analysis
conda env create -f model_training_environment.yml

# For capacity measurements
conda env create -f manifold_environment_simple.yml
```

To set up the `neuronet` module:
```bash
source setup_env.sh
```

### Training Models

To train example models:
```bash
python scripts/example_model_train.py
```

The main scripts are located in `/scripts`, capacity measurement code in `/replicaMFT`, and MNIST-related code in `/MNIST_results`.

### Citation
```bibtex
@inproceedings{
    zheng2024delays,
    title={Delays in generalization match delayed changes in representational geometry},
    author={Xingyu Zheng and Kyle Daruwalla and Ari S Benjamin and David Klindt},
    booktitle={UniReps: 2nd Edition of the Workshop on Unifying Representations in Neural Models},
    year={2024},
    url={https://openreview.net/forum?id=1ae108kHk2}
}
```
