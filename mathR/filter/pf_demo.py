import numpy as np
import matplotlib.pyplot as plt
from ekf import State2D, Odometry2DModel, GPSModel, EKF
from particle_filter import ParticleFilter
from draw_cov import draw_confidence_ellipses
from matplotlib.patches import FancyArrow
from matplotlib.collections import PatchCollection
from mathR.filter.plt_tools import remove_history_cov, remove_history_arrows, remove_history_particles
from config import dt, total_time, steps, initial_state, initial_cov, odometry_noise, measurement_noise, arrow_size, mean_arrow_size, mean_arrow_config, arrow_config

gps_update = False  # Configuration switch for GPS updates

def main():
    global running, gps_update

    # Ground truth state
    true_pos = np.array([0, 0])
    true_R = np.eye(2)
    true_vel = np.array([1, 0])
    true_state = State2D(true_pos, true_R, true_vel)

    # Models
    motion_model = Odometry2DModel(odometry_noise)
    measurement_model = GPSModel(measurement_noise)
    pf = ParticleFilter(100, initial_state, motion_model, measurement_model)
    ekf = EKF(initial_state, initial_cov, motion_model, measurement_model)

    # Storage for plotting
    true_trajectory = []
    pf_trajectory = []
    ekf_trajectory = []

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the size of the graph
    plt.axis('equal')
    true_plot, = ax.plot([], [], 'r--', label='True Trajectory')
    gps_plot = ax.scatter([], [], facecolor='r', edgecolor='w', label='GPS', s=20) if gps_update else None
    pf_arrow_handle = FancyArrow(0, 0, 0.1, 0, edgecolor='w', facecolor='darkgreen', label='PF Updated Pose')
    ekf_arrow_handle = FancyArrow(0, 0, 0.1, 0, edgecolor='w', facecolor='darkblue', label='EKF Predicted Pose')
    ax.add_patch(pf_arrow_handle)
    ax.add_patch(ekf_arrow_handle)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10+6, 10+6)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Particle Filter and EKF Demo: Robot Moving in a Circle')
    ax.legend(handles=[true_plot, gps_plot, pf_arrow_handle, ekf_arrow_handle] if gps_update else [true_plot, pf_arrow_handle, ekf_arrow_handle], loc='upper right')  # Fix legend in top right
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

        # PF predict step
        pf.predict(u, dt)

        # EKF predict step
        ekf.predict(u, dt)

        if gps_update:
            # GPS measurement with noise
            gps_measurement = true_state.pos + np.random.multivariate_normal(np.zeros(2), measurement_model.R)

            # PF and ekf update step
            ekf.correct(gps_measurement)
            pf.correct(gps_measurement)
            pf.resample()

            # Update plot with GPS data
            gps_plot.set_offsets([gps_measurement])
            draw_confidence_ellipses(ax, measurement_model.R, gps_measurement, cmap='Reds', alpha=0.2)

        # Store estimated state
        pf_mean, _ = pf.get_estimate()
        ekf_trajectory.append(ekf.x.pos.copy())
        pf_trajectory.append(pf_mean[:2])

        # Update plot
        true_plot.set_data(np.array(true_trajectory)[:, 0], np.array(true_trajectory)[:, 1])

        # Remove historical ellipses, arrows, and particles
        remove_history_cov(ax)
        remove_history_arrows(ax)
        remove_history_particles(ax)

        # Draw EKF confidence ellipses
        draw_confidence_ellipses(ax, ekf.P[:2, :2], ekf.x.pos, cmap='Blues', alpha=0.1)

        # Draw PF and EKF pose arrows
        arrows = []
        for particle in pf.particles:
            arrows.append(FancyArrow(particle[0], particle[1], arrow_size * np.cos(particle[2]), arrow_size * np.sin(particle[2]), color='darkgreen', **arrow_config))
        pf_arrow = FancyArrow(pf_mean[0], pf_mean[1], mean_arrow_size * np.cos(pf_mean[2]), mean_arrow_size * np.sin(pf_mean[2]), edgecolor='w', facecolor='darkgreen', **mean_arrow_config)
        ekf_arrow = FancyArrow(ekf.x.pos[0], ekf.x.pos[1], mean_arrow_size * np.cos(ekf.x.theta()), mean_arrow_size * np.sin(ekf.x.theta()), edgecolor='w', facecolor='darkblue', **mean_arrow_config)
        # arrows.append(pf_arrow)
        arrows.append(ekf_arrow)
        arrow_collection = PatchCollection(arrows, match_original=True)
        ax.add_collection(arrow_collection)

        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()