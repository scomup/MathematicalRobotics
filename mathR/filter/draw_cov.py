import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

def draw_confidence_ellipses(ax, cov, mean, cmap='Blues', facecolor='none', alpha=0.3, label=None):
    # Ensure covariance is a 2x2 matrix and mean is a 2-element vector
    assert cov.shape == (2, 2), "Covariance matrix must be 2x2."
    assert len(mean) == 2, "Mean must be a 2-element vector."
    
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort the eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    # Define the number of standard deviations for the ellipse (e.g., 1, 2, 3)
    n_std_devs = np.arange(0.3, 3, 0.3)
    
    # Get the colormap
    colormap = cm.get_cmap(cmap, len(n_std_devs))
    
    # Invert the colormap
    colormap = colormap.reversed()
    
    # Plot each ellipse
    for i, n_std in enumerate(n_std_devs):
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse_label = label if i == 0 else None
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor='none', fc=colormap(i, alpha=alpha), lw=2, zorder=-n_std, label=ellipse_label)
        ax.add_patch(ellipse)
    
    # Set limits for the plot
    # ax.set_xlim(mean[0] - 3*np.sqrt(cov[0, 0]), mean[0] + 3*np.sqrt(cov[0, 0]))
    # ax.set_ylim(mean[1] - 3*np.sqrt(cov[1, 1]), mean[1] + 3*np.sqrt(cov[1, 1]))
    # ax.set_aspect('equal', 'box')
    if label is not None:
        ax.legend()

if __name__ == "__main__":
    # Example usage
    fig, ax = plt.subplots()
    cov_matrix = np.array([[3, 1], [1, 2]])
    mean_vector = np.array([0, 0])
    draw_confidence_ellipses(ax, cov_matrix, mean_vector, alpha=0.2, label='Custom Label')
    plt.show()