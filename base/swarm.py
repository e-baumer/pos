import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from particle import Particle


class Swarm():
    def __init__(
        self,
        n_particles,
        dimensions,
        bounds,
        params,
        costfunc,
        method='L-BFGS-B',
        verbose=0
    ):
        '''
        Class to create particle swarm.

        Parameters
        ----------
        n_particles : int
            Number of particles to be included in the swarm.

        dimensions : int
            Number of dimensions to search function for optimal

        bounds : a list of tuples
            A list of bounds of each dimension (lower bound, upper bound). For example bounds for a
            two dimensional search :code: `[(0, 10), (-3, 2)]`.

        params : dict with keys :code: `{'c1', 'c2', 'w', 'beta', 'maxfun'}`
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

        costfunc : callable
            The objective function to be minimized
                `` fun(x) -> float``
            where x is an 1-D array with shape (n,).

        method : string
            Solver used for minimization

        verbose : int, optional
           The verbosity level: if non zero, progress messages are printed. Above 50, the output
           is sent to stdout. The frequency of the messages increases with the verbosity level.
           If it more than 10, all iterations are reported.
        '''
        self.n_particles = n_particles
        self.dim = dimensions
        self.bnds = bounds
        self.params = params
        self.costfunc = costfunc
        self.method = method
        self.verbose = verbose
        self.swarm = None
        self.global_best_position = None
        self.global_best_funcval = None

    def optimize(self, niters, n_jobs=None, x0=None):
        '''
        Optimize the swarm based on the number of iterations specified.

        Parameters
        ----------
        niters : int
            Number of iterations to optimize with.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        x0 : array-like or None, Optional
            Initial position of particle. If None the position of particle is assigned by random
            uniform distribution.

        Returns
        -------
        global_best_funcval : float
            The best (minimum) cost function value from the swarm
        global_best_position : float
            The position of the particle with the most optimal (out of the swarm) cost function
            value
        '''
        if x0 is None:
            self.swarm = [Particle(i, self.bnds, self.params) for i in range(self.n_particles)]
        else:
            self.swarm = [
                Particle(i, self.bnds, self.params, x0=x0[i]) for i in range(self.n_particles)
            ]

        if self.global_best_position is None:
            swarm_cost = np.array(
                [self.swarm[i].evaluate(self.costfunc) for i in range(self.n_particles)]
            )
            self.global_best_position = self.swarm[np.argmin(swarm_cost)].position
            self.global_best_funcval = np.min(swarm_cost)

        for i in tqdm(range(niters), desc='Swarm Iteration'):
            self._find_local_best(n_jobs=n_jobs)

            swarm_cost = [self.swarm[i].err for i in range(self.n_particles)]

            self.global_best_position = self.swarm[np.argmin(swarm_cost)].position
            self.global_best_funcval = np.min(swarm_cost)
        return (self.global_best_funcval.copy(), self.global_best_position.copy())

    def _find_local_best(self, n_jobs):
        '''
        Iterate through each particle to update position and velocity. Allow for parallelization.

        Parameters
        ----------
        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        '''
        parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose)

        def evaluate_particle(particle):
            particle.update(self.global_best_position, self.costfunc, method=self.method)
            return particle

        self.swarm = parallel(delayed(evaluate_particle)(p) for p in self.swarm)
