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
- fixed the kernel distance analysis
- how to access the aignment: the CKA measurement?

### 09/11/24
- finished running the last networks (hopefully) and 
- organized them in a nicer way!!!
- note to self that all but `output_scale = 0.1` the random seed is 10, that one we used the seed 314159 --- alright I am re-training this one for consistency
- got all activations 
- start running capacity measurement locally and on the cluster (!) 130 hrs in total... so will only be able to harvest result on Sundzy :') if everything goes well. oh my...
- informed both Saket and Alex about this endeavor, they are okay, but assigned me stuff meanwhile fine..

Where are my network checkpoints and activations:

```
data/Geometry/new/checkpoints
data/Geometry/new/activations
```

### 09/12/24
-  `output_scale = 0.1` hasn't been been fed into the pipeline for capacity measure; will do that soon as local or cluster finish; should be local
- create the wrong folder; now capacity measure results is in `~/output/capacity_measures` and will move them over 
- for outputs now let's start using output relative to the script's dir, so that it's not dependent on where we run things.
- Jacobian running and should be done very quickly


**TO-DO:**
1. move over other measures: 
    1. input and weight jacobian svds, kernel distance
    2. out of distribution generalization 
2. implement CKA
3. need to remove some activation as well; locally they take up 50+ G... but also need to figure out how much capacity measurement we need to re-run
4. move the MNIST example here, make the format conform
5. code to reproduce the example networks 
6. weight norm over checkpoints (maybe)




## Appendix planning

1. all 3 types, 3 examples each
2. variability 
3. training example capacity and test example capacity 
