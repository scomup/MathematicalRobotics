import numpy as np
from mathR.utilities.math_tools import *


class State2D:
    def __init__(self, pos, R, vel):
        self.pos = np.array(pos)
        self.R = R
        self.vel = np.array(vel)

    def to_array(self):
        return np.array([self.pos[0], self.pos[1], logSO2(self.R), self.vel[0], self.vel[1]])

    def __add__(self, delta):
        p = self.pos + delta[0:2]
        R = self.R @ expSO2(delta[2])
        v = self.vel + delta[3:]
        return State2D(p, R, v)

    def __sub__(self, other):
        p = self.pos - other.pos
        R = other.R.T @ self.R
        v = self.vel - other.vel
        return np.array([p[0], p[1], logSO2(R), v[0], v[1]])

    def __len__(self):
        return 5

    def theta(self):
        return logSO2(self.R)


class Odometry2DModel:
    def __init__(self, process_noise):
        # Motion model for 2D odometry
        self.Q = process_noise

    def predict(self, state, u, dt, noise=False):
        if noise:
            u = u + np.random.multivariate_normal(np.zeros(3), self.Q)
        # Position update: pos = pos + velocity * dt
        pos = state.pos + state.vel * dt
        # Orientation update: R = R @ expSO2(omega * dt)
        R = state.R @ expSO2(u[2] * dt)
        # Velocity update: vel = R @ control velocity
        vel = R @ u[0:2]
        predicted_state = State2D(pos, R, vel)
        # Compute Jacobian with respect to the state (F)
        Fx = np.zeros([5, 5])
        Fx[0:2, 0:2] = np.eye(2)
        Fx[0:2, 3:5] = np.eye(2) * dt
        Fx[2, 2] = 1
        Fx[3:5, 2] = state.R @ (hat2d(u[0:2]))

        Fu = np.zeros([5, 3])
        Fu[2, 2] = -dt
        Fu[3:5, 0:2] = -state.R
        return predicted_state, Fx, Fu

class GPSModel:
    def __init__(self, measurement_noise):
        # MeasurementModel for 2D GPS measurement
        self.R = measurement_noise

    def measure(self, state):
        H = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
        return state.pos, H

class EKF:
    def __init__(self, initial_mean, initial_cov, motion_model, measurement_model):
        self.x = initial_mean
        self.P = initial_cov
        self.f = motion_model
        self.h = measurement_model

    def predict(self, u, dt):
        Q = self.f.Q
        self.x, Fx, Fu = self.f.predict(self.x, u, dt)
        self.P = Fx @ self.P @ Fx.T + Fu @ Q @ Fu.T

    def correct(self, z):
        z_pred, H = self.h.measure(self.x)
        R = self.h.R
        y = z - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    def get_estimate(self):
        return self.x, self.P


if __name__ == "__main__":
    # Initial state
    initial_pos = [0, 0]
    initial_R = np.eye(2)
    initial_vel = [1, 0]
    initial_state = State2D(initial_pos, initial_R, initial_vel)

    # Initial covariance
    initial_cov = np.eye(5) * 0.1

    # Process noise
    process_noise = np.eye(3) * 0.1

    # Measurement noise
    measurement_noise = np.eye(2) * 0.1

    # Models
    motion_model = Odometry2DModel(process_noise)
    measurement_model = GPSModel(measurement_noise)

    # EKF
    ekf = EKF(initial_state, initial_cov, motion_model, measurement_model)

    # Control input (velocity in x, y and angular velocity)
    u = np.array([1, 0, 0.1])

    # Time step
    dt = 1.0

    # Predict step
    ekf.predict(u, dt)

    # Measurement (GPS position)
    z = np.array([1.1, 0.1])

    # Update step
    ekf.correct(z)

    # Get the updated state and covariance
    mean, cov = ekf.get_estimate()

    print("Updated state:", mean.to_array())
    print("Updated covariance:", cov)


