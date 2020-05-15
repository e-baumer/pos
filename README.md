# POS -- Particle Optimizing Swarm
A gradient free optimization routine which combines Particle Swarm Optimization with a local optimization for each particle


## Overview
In real world applications such as hyper-parameter tuning of machine learning models, the function that we are trying to optimize is noisy with
many local minimum and maximum values. The local noise can skew the search over the global structure. Particle Swarm Optimization (PSO) 
moves candidate solutions (or particles) across the feature space according to the last known best solution of all the particles as a whole and of
the individual particles. Where the particle lands on the local variability will help determine the future state of that individual particle as well
as all the other particles. The local variability with mask the overall structure of the objective function and either significanly increase the 
necessary number of iterations necessary to explore the overall structure or make impossible to explore. 

Here we propose a solution to overcome the inconsistencies of the local variability. The search over the feature space is divided into two steps. 
In the first step we use Particle Swarm Optimization to update a candidate solution within the feature space. After each update we perform a local
optimization within a bounded region of the updated particle's position. This local optimization ensures that are searching across the space
of the best local solutions. We repeat these two steps for a user defined number of iterations.

Currently, there are two options for the local optimization scheme. The first option is to perform a [Broyden-Fletcher-Goldfarb-Shannon (BFGS)](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
numerical optimization. The second option is to randomly sample within a bounded region of the candidate solution and take the minimum of the objective
function within that sample. On the final overall iteration, a BFGS optimization is performed.

The random sampling scheme for the local optimization is computationaly cheaper; however, if the most resource intensive step is in evaluation of
the objective function one should definitely use the BFGS optimization.


## Installation
To install pos from Github use the following command:
```shell
$ pip install git+git://github.com/e-baumer/pos.git@master
```

## Example Use



