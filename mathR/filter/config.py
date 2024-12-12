
import numpy as np
from ekf import State2D

# Simulation parameters
dt = 0.1
total_time = 1000
steps = int(total_time / dt)

# Initial state for PF and EKF
initial_state = State2D([0, 0], np.eye(2), [1, 0])
initial_cov = np.eye(5) * 0.1
odometry_noise = np.eye(3)
odometry_noise[0, 0] =  0.1 # odometry x velocity covariance
odometry_noise[1, 1] =  0.1 # odometry y velocity covariance
odometry_noise[2, 2] = 0.1 # odometry arngular velocity covariance
measurement_noise = np.eye(2) * 1

# Arrow configurations
arrow_size = 0.1
mean_arrow_size = 0.2

mean_arrow_config = {
    'width': mean_arrow_size / 4,
    'head_width': mean_arrow_size,
    'head_length': mean_arrow_size / 2,
    'length_includes_head': True,
    'zorder': 10
}

arrow_config = {
    'width': arrow_size / 4,
    'head_width': arrow_size,
    'head_length': arrow_size / 2,
    'length_includes_head': True,
    'zorder': 10
}