from matplotlib.patches import Ellipse, FancyArrow
from matplotlib.collections import PatchCollection

def remove_history_cov(ax):
    for artist in ax.get_children():
        if isinstance(artist, Ellipse):
            artist.remove()

def remove_history_arrows(ax):
    for artist in ax.get_children():
        if isinstance(artist, FancyArrow):
            artist.remove()

def remove_history_particles(ax):
    for artist in ax.get_children():
        if isinstance(artist, PatchCollection):
            artist.remove()
