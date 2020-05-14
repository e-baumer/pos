import numpy as np

from unittest import TestCase
from pos.swarm import Swarm


class TestSwarm(TestCase):
    def setUp(self):
        self.f = lambda x: np.exp(np.sin(50*x)) + np.sin(60*np.exp(x)) + np.sin(70*np.sin(x))

    def test_position_bounds(self):
        nparticles = 10
        ndims = 1
        bounds = [(0, 10)]
        params = {'c1':0.5, 'c2':0.3, 'w':0.6, 'beta':0.3, 'maxfun':20, 'dim_scale':[0.02]}

        pso = Swarm(nparticles, ndims, bounds, params, self.f, method='stochastic')
        func_min, position = pso.optimize(10)
        self.assertTrue(position >= bounds[0][0] and position <= bounds[0][1])
