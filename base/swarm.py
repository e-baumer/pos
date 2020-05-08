import numpy as np
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

        for i in range(niters):
            self.find_local_best(n_jobs=n_jobs)

            swarm_cost = [self.swarm[i].err for i in range(self.n_particles)]

            self.global_best_position = self.swarm[np.argmin(swarm_cost)].position
            self.global_best_funcval = np.min(swarm_cost)
        return (self.global_best_funcval.copy(), self.global_best_position.copy())

    def find_local_best(self, n_jobs):
        parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose)

        def evaluate_particle(particle):
            particle.update(self.global_best_position, self.costfunc, method=self.method)
            return particle

        with parallel:
           self.swarm = parallel(delayed(evaluate_particle)(p) for p in self.swarm)


if __name__ == "__main__":

    f = lambda x: np.exp(np.sin(50*x)) + np.sin(60*np.exp(x)) + np.sin(70*np.sin(x)) + np.sin(np.power(x, 2))
    x = np.atleast_2d(np.linspace(0, 10, 500)).T

    nparticles = 10
    ndims = 1
    bounds = [(0, 10)]
    params = {'c1':0.5, 'c2':0.3, 'w':0.6, 'beta':0.3, 'maxfun':20}

    pso = Swarm(nparticles, ndims, bounds, params, f)
    pso.optimize(10)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
