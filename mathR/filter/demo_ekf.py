import numpy as np
import matplotlib.pyplot as plt
from mathR.filter.ekf import State2D, Odometry2DModel, GPSModel, EKF
from mathR.filter.draw_cov import draw_confidence_ellipses
from matplotlib.patches import FancyArrow
from matplotlib.widgets import Button
from mathR.filter.plt_tools import remove_history_cov, remove_history_arrows
from mathR.filter.config import dt, total_time, steps, initial_state, initial_cov, odometry_noise, measurement_noise, arrow_size, mean_arrow_size, mean_arrow_config, arrow_config

def main():
    # Ground truth state
    true_pos = np.array([0, 0])
    true_R = np.eye(2)
    true_vel = np.array([1, 0])
    true_state = State2D(true_pos, true_R, true_vel)

    # Models
    motion_model = Odometry2DModel(odometry_noise)
    measurement_model = GPSModel(measurement_noise)
    ekf = EKF(initial_state, initial_cov, motion_model, measurement_model)
    ekf_no_correct = EKF(initial_state, initial_cov, motion_model, measurement_model)

    # Storage for plotting
    true_trajectory = []
    estimated_trajectory = []
    odom_trajectory = []

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the graph
    plt.axis('equal')
    true_plot, = ax.plot([], [], 'r--', label='True Trajectory')
    gps_plot = ax.scatter([], [], facecolor='r', edgecolor='w', label='GPS', s=20)
    ekf_arrow_handle = FancyArrow(0, 0, 0.1, 0, edgecolor='w', facecolor='darkblue', label='EKF Updated Pose')
    ax.add_patch(ekf_arrow_handle)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10+6, 10+6)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('EKF Demo: Robot Moving in a Circle')
    ax.legend(handles=[true_plot, gps_plot, ekf_arrow_handle], loc='upper right')  # Fix legend in top right
    ax.grid(True)  # Add grid to the figure

    for step in range(steps):
        # Ground truth update (robot moving in a circle)
        omega = 0.1  # Angular velocity
        true_vel = np.array([1, 0])
        true_u = np.array([0.5, 0, omega])
        true_state = motion_model.predict(true_state, true_u, dt)[0]
        true_trajectory.append(true_state.pos.copy())

        # Control input with noise
        u = true_u + np.random.multivariate_normal(np.zeros(3), motion_model.Q)

        # EKF predict step
        ekf.predict(u, dt)
        ekf_no_correct.predict(u, dt)

        # GPS measurement with noise
        gps_measurement = true_state.pos + np.random.multivariate_normal(np.zeros(2), measurement_model.R)

        # EKF correct step
        ekf.correct(gps_measurement)

        # Store estimated state
        estimated_trajectory.append(ekf.x.pos.copy())
        odom_trajectory.append(ekf_no_correct.x.pos.copy())

        # Update plot
        true_plot.set_data(np.array(true_trajectory)[:, 0], np.array(true_trajectory)[:, 1])
        gps_plot.set_offsets([gps_measurement])

        # Remove historical ellipses and arrows
        remove_history_cov(ax)
        remove_history_arrows(ax)

        # Draw EKF confidence ellipses
        draw_confidence_ellipses(ax, ekf.P[:2, :2], ekf.x.pos, cmap='Blues')
        draw_confidence_ellipses(ax, measurement_model.R, gps_measurement, cmap='Reds', alpha=0.1)

        # Draw EKF pose arrow
        arrow = FancyArrow(ekf.x.pos[0], ekf.x.pos[1], mean_arrow_size * np.cos(ekf.x.theta()), mean_arrow_size * np.sin(ekf.x.theta()), edgecolor='w', facecolor='darkblue', **mean_arrow_config)
        ax.add_patch(arrow)

        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
