We will use README as log before we make it public.

## Summary of results so far:

### Induced Grokking (although not perfectly in 3 ways:

- Sample size
- Weight decay rate
- Output scaling 

#### Drawbacks: 
- All models start with the same weight scaling to start with, so models need to be based on this premise; and result shown not using the same optimizers
- Results look very variable, that our Figure 4 a) result with different seeds, some shows grokking some not
train performance also looks sometimes with delays 

### The progress measure we use:
- manifold capacity measurements
    - can track improvement of training performance.
- Kernel related:
    - Kernel distance (need to revisit, I might have not implemented correctly)
    - weight jacobian SVD

### Some unplesent inconvenience that we will improve in this repo:
Multiple folders (cluster and local) for different part of this project; will group things together in this attempt.

***

## Log for progress

### 09/10/24

**TO-DO:**
1. check if model and if capacity measures run successfully locally
2. remove checkpoints that are are no longer needed to free up space

