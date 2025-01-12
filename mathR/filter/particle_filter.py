import numpy as np
from mathR.filter.ekf import State2D
from mathR.utilities.math_tools import *


class ParticleFilter:
    def __init__(self, num_particles, initial_state, motion_model, measurement_model):
        self.num_particles = num_particles
        self.particles = np.array([initial_state.to_array()
                                  for _ in range(num_particles)])
        self.weights = np.ones(num_particles) / num_particles
        self.motion_model = motion_model
        self.measurement_model = measurement_model

    def predict(self, u, dt):
        for i in range(self.num_particles):
            state = State2D(self.particles[i][:2], expSO2(
                self.particles[i][2]), self.particles[i][3:])
            predicted_state, _, _ = self.motion_model.predict(
                state, u, dt, noise=True)
            self.particles[i] = predicted_state.to_array()

    def correct(self, z):
        for i in range(self.num_particles):
            state = State2D(self.particles[i][:2], expSO2(
                self.particles[i][2]), self.particles[i][3:])
            predicted_measurement, _ = self.measurement_model.measure(state)
            self.weights[i] = self.gaussian_likelihood(
                z, predicted_measurement, self.measurement_model.R)
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= np.sum(self.weights)  # normalize

    def resample(self):
        indices = np.random.choice(
            range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def gaussian_likelihood(self, z, z_pred, R):
        diff = z - z_pred
        return np.exp(-0.5 * diff.T @ np.linalg.inv(R) @ diff) / np.sqrt((2 * np.pi) ** len(z) * np.linalg.det(R))

    def get_estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        cov = np.cov(self.particles.T, aweights=self.weights)
        return mean, cov
