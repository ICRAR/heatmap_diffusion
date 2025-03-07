import argparse
import time
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

def diffuse_heatgrid(grid: np.array):
    """
    Given a 2D heatmap, perform diffusion according to laplace equation.

    :param grid:
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


def plot_diffusion(grid: np.array, filename: str) -> (
        plt.Figure, plt.Axes, matplotlib.collections.QuadMesh):
    """
    :param grid:
    :return:
    """
    fig, ax = plt.subplots()
    im = ax.pcolormesh(grid, cmap="hot", vmin=25, shading='auto', vmax=100)
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    plt.savefig(filename)


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

    ani = animation.FuncAnimation(fig, update, frames=500, interval=250, blit=False)
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(path, writer=writer)


def diffuse_heatgrid_with_timesteps(grid, timesteps):

    padded_grid = np.pad(grid, ((5, 5), (0, 0)), mode='edge')

    for i in range(timesteps):
        padded_grid = diffuse_heatgrid(padded_grid)

    return padded_grid[5:-5, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run diffuse simulations")
    parser.add_argument("input", help="input heatmap data")
    parser.add_argument("output", help="The output file name")
    parser.add_argument("--timesteps", default=25, type=int)
    parser.add_argument("--parallel", action='store_true', default=False)
    parser.add_argument("-V", "--visualise", action="store_true", default=False)

    plot_dir = "fig/"
    args = parser.parse_args()
    timesteps = args.timesteps
    final_grid = np.genfromtxt(args.input, delimiter=',')
    print("Initial max (°C): ", np.max(final_grid))

    st = time.time()
    if args.parallel:
        grid_splits = [(split, timesteps) for split in np.array_split(final_grid, 10)]

        post_splits = []
        from multiprocessing import Pool
        with Pool(processes=4) as pool:
            post_splits = (
                pool.starmap(diffuse_heatgrid_with_timesteps, grid_splits)
            )

        final_grid = np.vstack(post_splits)

    else:
        for i in range(timesteps):
            final_grid = diffuse_heatgrid(final_grid)
    ft = time.time()
    print("Diffuse max (°C): ", np.max(final_grid))
    print(f"Runtime: {ft-st}")

    if args.visualise:
        fig, ax = plt.subplots()
        im = ax.pcolormesh(final_grid, cmap="hot", vmin=25, shading='auto', vmax=100)
        fig.colorbar(im, ax=ax, label="Temperature (C)")
        plt.savefig("output/diffusion.png")
