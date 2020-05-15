# POS - Particle Optimizing Swarm
A gradient free optimization routine which combines Particle Swarm Optimization with a local optimization for each candidate solution.


## Overview
In real-world applications such as hyper-parameter tuning of machine learning models, the function that we are trying to optimize is noisy with
many local minimum and maximum values. The local noise can skew the search over the global structure. Particle Swarm Optimization (PSO) 
moves candidate solutions (or particles) across the feature space according to the last known best solution of all the particles as a whole and of
the individual particles. Where the particle lands on the local variability will help determine the future state of that individual particle as well
as all the other particles. The local variability will mask the overall structure of the objective function and either significantly increase the 
necessary number of iterations necessary to explore the overall structure or make impossible to explore. 

Here we propose a solution to overcome the inconsistencies of local variability. The search over the feature space is divided into two steps. 
In the first step, we use Particle Swarm Optimization to update a candidate solution within the feature space. After each update, we perform a local
optimization within a bounded region of the updated particle's position. This local optimization ensures that we are searching across the space
of the best local solutions. We repeat these two steps for a user-defined number of iterations.

Currently, there are two options for the local optimization scheme. The first option is to perform a [Broyden-Fletcher-Goldfarb-Shannon (BFGS)](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
numerical optimization. The second option is to randomly sample within a bounded region of the candidate solution and take the minimum of the objective
function within that sample. On the final overall iteration, a BFGS optimization is performed.

The random sampling scheme for the local optimization is computationally cheaper; however, if the most resource-intensive step is in evaluation of
the objective function one should definitely use the BFGS optimization.


## Installation
To install pos from Github use the following command:
```shell
$ pip install git+git://github.com/e-baumer/pos.git@master
```

## Example Use
```python
import numpy as np
from pos.swarm import Swarm

# Define the objective function
f = lambda x: np.exp(np.sin(50*x[0])) + np.sin(60*np.exp(x[0])) + np.sin(70*np.sin(x[1]))

# Number of particles in swarm
nparticles = 100

# Number of dimensions in feature space
ndims = 2

# The bounds of each dimension in feature space
bounds = [(0, 10), (2, 12)]

# Parameters for PSO and local optimization
params = {'c1':0.5, 'c2':0.3, 'w':0.6, 'beta':0.3, 'maxfun':25, 'dim_scale':[0.02]}

# Initialize Swarm and specify local optimization method
pso = Swarm(nparticles, ndims, bounds, params, f, method='minimize')

# Run two step optimization
func_min, position = pso.optimize(200)
```

Swarm Parameters

        Parameters
        ----------
        n_particles : int
            Number of particles to be included in the swarm.

        dimensions : int
            Number of dimensions to search function for optimal

        bounds : a list of tuples
            A list of bounds of each dimension (lower bound, upper bound). For example bounds for a
            two dimensional search :code: `[(0, 10), (-3, 2)]`.

        params : dict with keys :code: `{'c1', 'c2', 'w', 'beta', 'maxfun', 'dim_scale'}`
            Dictionary containing parameters for particle swarm optimization as well as the local
            optimization.
                * c1 : float
                    Particle Swarm Optimization cognitive coefficient
                * c2 : float
                    Particle Swarm Optimization social coefficient
                * w : float
                    Particle Swarm Optimization inertial coefficient
                * beta : float (<1)
                    Velocity reduction parameter for particles that reflect from boundary
                * maxfun : int
                    Maximum number of function evaluations for local minimization method
                    or Maximum number of samples to evaluate for random local search.
                * dim_scale : float (0 < x <=1)
                    This scale is applied to the length of each dimension bound to determine
                    the radius for either local minimization or local stochastic search.

        costfunc : callable
            The objective function to be minimized
                `` fun(x) -> float``
            where x is an 1-D array with shape (n,).

        method : string ('minimze' or 'stochastic'), default is 'minimize'
            Method to use for local search. If 'minimize', Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            numerical algorithm is used to find the local optimal value. If 'stochastic', random
            points are searched within a radius of the updated position to look for local optimum.

        verbose : int, optional
           The verbosity level: if non zero, progress messages are printed. Above 50, the output
           is sent to stdout. The frequency of the messages increases with the verbosity level.
           If it more than 10, all iterations are reported.



