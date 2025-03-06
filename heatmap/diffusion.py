from pathlib import Path
import matplotlib.collections
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def diffuse_heatgrid(grid, timesteps: int = 100):
    """
    Given a 2D heatmap, perform diffusion according to laplace equation.

    :param grid_path:
    :param timesteps:
    :return:
    """

    rows, cols = grid.shape

    new_grid = grid.copy()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            new_grid[i, j] = (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[
                i, j - 1]) / 4
    grid = new_grid

    return grid


def plot_diffusion(grid: np.array) -> (
        plt.Figure, plt.Axes, matplotlib.collections.QuadMesh):
    """

    :param grid:
    :return:
    """
    fig, ax = plt.subplots()

    im = ax.pcolormesh(grid, cmap="hot", vmin=25)
    fig.colorbar(im, ax=ax, label="Temperature (C)")
    return fig, ax, im


def animate_diffusion(grid: np.array, path: str):
    """
    Generate an animated GIF of the diffusion process
    :param grid:
    :param: path:
    :return:
    """

    def update(frame):
        global final_grid
        final_grid = diffuse_heatgrid(final_grid)
        im.set_array(final_grid.ravel())
        ax.set_title(f"Frame {frame}")
        return im,

    if animate:
        ani = animation.FuncAnimation(fig, update, frames=500, interval=250, blit=False)
        writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('diffusion.gif', writer=writer)


def split_grid(grid: np.array):
    """

    :param grid:
    :return:
    """


def gather_splits(splits: list):
    """

    :param splits:
    :return:
    """


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run diffuse simulations")
    parser.add_argument("input", help="input heatmap data")
    parser.add_argument("--timesteps", default=25)
    parser.add_argument("--parallel", action='store_true', default=False)
    parser.add_argument("output", help="The output file name")

    plot_dir = "fig/"
    args = parser.parse_args()
    timesteps = args.timesteps
    final_grid = np.genfromtxt(args.input, delimiter=',')

    fig, ax, im = plot_diffusion(final_grid)
    if args.parallel:
        grid_splits = np.split(final_grid, [10, 10])

        post_splits = []
        for g in grid_splits:
            for i in range(timesteps):
                g = diffuse_heatgrid(g)

        post_splits.append(g)

        final_grid = np.vstack(post_splits)

    else:
        for i in range(timesteps):
            final_grid = diffuse_heatgrid(final_grid)
        print(np.mean(final_grid))

        fig2, ax2 = plt.subplots()
        im2 = ax2.pcolormesh(final_grid, cmap="hot", vmin=25, shading='auto', vmax=100)
        fig2.colorbar(im2, ax=ax2, label="Temperature (C)")
        plt.savefig("diffusion.png")
