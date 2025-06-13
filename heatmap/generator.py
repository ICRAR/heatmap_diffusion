import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import colorbar

from scipy.stats import multivariate_normal


def random_mean_from_range(lower=10, upper=100):
    """

    :param lower:
    :param upper:
    :return:
    """
    rng = np.random.default_rng()
    return rng.integers(lower, upper), rng.integers(lower, upper)


def random_covariance_from_range(lower=1000, upper=100000):
    """
    Create
    :param lower:
    :param upper:
    :return:
    """
    rng = np.random.default_rng()
    return rng.integers(lower, upper)


def generate_sources(num_sources: int = 1, mean=(10, 100), cov=(100, 1000)):
    """
    :param num_sources:
    :param cov:
    :param mean:

    :return:
    """
    sources = []
    # Scale sources according to number of sources?
    mu_lower, mu_upper = mean
    cov_lower, cov_upper = cov
    for i in range(num_sources):
        mean_a, mean_b, = random_mean_from_range(mu_lower, mu_upper)
        cov = random_covariance_from_range(cov_lower, cov_upper)
        cov_x = (cov_upper - cov)
        cov_y = (cov_lower + cov)
        if i%2 == 0:
            sources.append(([mean_a, mean_b], [[cov_x,cov_y], [0, cov]]))
        else:
            sources.append(([mean_b, mean_a], [[cov,0], [0, cov]]))
    return sources


def generate_heatmap(sources: list, rows: int = 100, cols: int = 100,
                     bounds: tuple[int, int] = (25, 100)):
    """
    :param sources: mean and covariance params the distribution
    :param rows: The number of rows
    :param cols: The number of colums
    :param bounds: Temperature boundaries of the heatmap
    :return: grid: of heat map
    """

    grid = np.meshgrid(np.linspace(0, rows - 1, rows), np.linspace(0, cols - 1, cols))
    grid_coordinates = np.dstack(grid)
    heat = np.zeros((rows, cols))
    for mean, cov in sources:
        heat += multivariate_normal(mean, cov).pdf(grid_coordinates)
    lower, upper = bounds
    return lower + (heat / np.max(heat)) * (upper-lower)

def heatmap_to_csv(heatmap, path: str):
    """
    Produce a csv file with the heatmap
    :param heatmap:
    :param path:
    """
    np.savetxt(path, heatmap, delimiter=',')


def heatmap_to_png(heatmap, path: str):
    """
    Produce a png of the heatmap
    :param heatmap:
    :param path:
    """
    fig, ax = plt.subplots()

    im = ax.pcolormesh(heatmap, cmap="hot", vmin=25, shading='auto', vmax=100)
    fig.colorbar(im, ax=ax, label="Temperature (C)")

    plt.savefig(path)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate set of heatmaps within boundary "
                                     "parameters.")
    for i in range(25):
        num_sources = random.randint(1, 5)
        map = generate_heatmap(generate_sources(num_sources, mean=(5, 85),
                                                cov=(100,500)),
                                                rows=100,
                                               cols=100)
        heatmap_to_csv(map, f"heatmap/data/heatmap_small_{i}.csv")
        heatmap_to_png(map, f"heatmap/data/heatmap_small_{i}.png")
