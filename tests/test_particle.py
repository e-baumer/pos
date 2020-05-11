import numpy as np

from unitest import TestCase
from base.particle import Particle


class TestParticle(TestCase):

    def test_position_bounds(self):
        particle_id = 1
        bounds = [(0, 10)]
        params = {'c1':0.5, 'c2':0.3, 'w':0.6, 'beta':0.3, 'maxfun':20}

        p1 = Particle(particle_id, bounds, params)
        p1._check_bounds(np.array([12.0]))
        self.assertEqual(p1.position, bounds[0][1])

    def test_velocity_bounds(self):
        particle_id = 1
        bounds = [(0, 10)]
        params = {'c1':0.5, 'c2':0.3, 'w':0.6, 'beta':0.3, 'maxfun':20}

        p1 = Particle(particle_id, bounds, params)
        vel = p1.velocity.copy()
        p1._check_bounds(np.array([12.0]))
        self.assertEqual(p1.velocity, -vel*params['beta'])



