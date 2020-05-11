import random
import numpy as np

from scipy.optimize import minimize


class Particle():
    def __init__(self, particle_id, bounds, params, x0=None):
        '''
        This class represents an individual particle. It is initialized with either a specific
        position and velocity or given a random value.

        Parameters
        ----------
        particle_id : int
            Integer id of particle.

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

        x0 : array-like or None, Optional
            Initial position of particle. If None the position of particle is assigned by random
            uniform distribution.
        '''
        self.id = particle_id
        ndims = len(bounds)

        if x0 is None:
            self.position = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds])
        else:
            self.position = x0

        self.velocity = np.random.uniform(low=-1.0, high=1.0, size=(ndims,))
        self.bounds = bounds
        self.params = params
        self.ndims = ndims
        self.pos_best = self.position
        self.err_best = None
        self.err = None
        self.err_hist = []
        self.pos_hist = []
        self.vel_hist = []

    def _storestate(self):
        '''
        Stores the current state of the particle.
        '''
        self.err_hist.append(self.err)
        self.pos_hist.append(self.position)
        self.vel_hist.append(self.velocity)

    def evaluate(self, costfunc):
        '''
        Evaluate the cost or objective function for the current location of the particle.

        Parameters
        ----------
        costfunc : callable
            Cost or objective function to be evaluated

        Returns
        -------
        err: float
            The value of the cost or objective function for the particles current position.
        '''
        err = costfunc(self.position)
        return err

    def update(self, global_best_position, costfunc, method='L-BFGS-B'):
        '''
        Runs one iteration of updating particle position and local minimization search based on
        new position.

        Parameters
        ----------
        global_best_position : float
            From all possible particles in swarm the global best position (based on the cost
            function) before updating.

        costfunc : callable
            Cost or objective function to be evaluated

        Returns
        -------
        local_opt : float
            The local minimized value of the cost function after updating particle position and
            searching for local optimal.
        '''
        self._storestate()
        self.update_velocity(global_best_position)
        self.update_position()
        local_opt = self.local_optimal(costfunc, method)
        return local_opt

    def local_optimal(self, costfunc, method):
        '''
        Determine the local optimal (minimum) of the cost function based on the particles current
        position.

        Parameters
        ----------
        costfunc : callable
            Cost or objective function to be evaluated

        method : string
            Solver used for minimization

        Returns
        -------
        err: float
            Local optimal value of cost function
        '''
        res = minimize(
            costfunc,
            self.position,
            method=method,
            bounds=self.bounds,
            options={'maxfun':self.params['maxfun']}
        )

        self.position = res.x
        self.err = res.fun

        if self.err_best is None or (self.err < self.err_best):
            self.pos_best = self.position
            self.err_best = self.err

        return self.err.copy()

    def update_velocity(self, global_best_position):
        '''
        Update particle velocity based particle and swarm best position.

        Parameters
        ----------
        global_best_position : float
            From all possible particles in swarm the global best position (based on the cost
            function) before updating.

        '''
        cognitive = (
            self.params['c1'] *
            np.random.uniform(0, 1, self.ndims) *
            (self.pos_best - self.position)
        )

        social = (
            self.params['c2'] *
            np.random.uniform(0, 1, self.ndims) *
            (global_best_position - self.position)
        )

        self.velocity = (self.params['w'] * self.velocity) + cognitive + social

    def update_position(self):
        '''
        Update the particle position based on its updated velocity.
        '''
        position = self.position.copy()
        position += self.velocity
        self._check_bounds(position)

    def _check_bounds(self, position):
        '''
        Check whether the current update to the particle position puts the particle outside of
        any bounds. If so, the particle will be reflected from the boundary. The resulting velocity
        will be damped by parameter beta.

        Parameters
        ----------
        position : array-like, float
        '''
        lwr_bnds = np.array([b[0] for b in self.bounds])
        upr_bnds = np.array([b[1] for b in self.bounds])

        lwr_mask = position <= lwr_bnds
        upr_mask = position >= upr_bnds

        self.position = position.copy()
        self.position[lwr_mask] = lwr_bnds[lwr_mask]
        self.position[upr_mask] = upr_bnds[upr_mask]

        all_mask = lwr_mask + upr_mask
        self.velocity[all_mask] = -self.velocity[all_mask] * self.params['beta']
