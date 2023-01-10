import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
radius = 1.
x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

roll_slider = Slider(
    ax=ax,
    label="Amplitude",
    valmin=0,
    valmax=np.pi*2,
    valinit=0,
    orientation="vertical"
)

ax.plot_surface(x, y, z,color="lightgreen",alpha=0.3, antialiased=True)
quivers = ax.quiver([0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1])
ax.set_aspect('equal')
plt.show()

