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

from now on to next week, I can train 42 models with 10000 epochs :') if I don't consider replicating this folder to Alex's cluster.

### 09/10/24
- model can train okay in this project directory, might need some linting and output organization.
- manifold environment successfully set up as well yayy
- removed some checkpoints that are are no longer needed to free up space, and also good news is that we seem only need to retrain for 2 networks! Already doing it [Harvest end of Wed.]


**TO-DO:**
3. need to remove some activation as well; locally they take up 50+ G... but also need to figure out how much capacity measurement we need to re-run
4. move the MNIST example here as well?
5. discuss with Ari and Kyle the CKA measurement
6. fix the kernel distance analysis

## Appendix planning

1. all 3 types, 3 examples each
2. variability 
3. training example capacity and test example capacity 
