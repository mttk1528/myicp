"""
This module implements Poisson Disk Sampling on a set of 3D points.

The PoissonDiskSampler class offers the following functionalities:
    - Initializing the Poisson Disk Sampler with an array of points
        and a specified minimum distance between sampled points.
    - Performing Poisson Disk Sampling on the initialized points.
        The sampling process takes into account the minimum distance between
        points to ensure the randomness and uniformity of the sampled points.
"""
import numpy as np
from scipy.spatial import cKDTree


class PoissonDiskSampler:
    """
    Perform Poisson Disk Sampling on a given point set in a 3D space.
    """
    def __init__(self, points, min_dist):
        """
        Initialize the PoissonDiskSampler with a set of points
        and a minimum distance.

        Parameters
        ----------
        points : np.ndarray
            The array of points to sample from.
        min_dist : float
            The minimum distance between sampled points.
        """
        self.points = points
        self.min_dist = min_dist
        self.tree = cKDTree(points)

    def sample(self, num_samples):
        """
        Perform Poisson Disk Sampling on the given points.

        Parameters
        ----------
        num_samples : int
            The number of points to sample.

        Returns
        -------
        np.ndarray
            The indices of the sampled points.
        """
        indices = []
        active_list = list(range(self.points.shape[0]))

        init_index = np.random.choice(active_list)
        indices.append(init_index)
        active_list.remove(init_index)
        while len(indices) < num_samples and active_list:
            index = np.random.choice(active_list)
            point = self.points[index]
            points_in_range = self.tree.query_ball_point(
                point, 2 * self.min_dist)

            for point_index in points_in_range:
                if point_index not in indices and np.min(
                    cKDTree(self.points[indices]).query(
                        self.points[point_index])[0]) > self.min_dist:
                    indices.append(point_index)
                    active_list.remove(point_index)
                    break
        return np.array(indices)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    plt.style.use('dark_background')

    points = np.random.uniform(0, 1000, (1000, 3))
    pds = PoissonDiskSampler(points, 10)
    sampled_indices = pds.sample(50)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        color='b', alpha=0.5, s=10, label='Original Points')
    ax.scatter(
        points[sampled_indices, 0],
        points[sampled_indices, 1],
        points[sampled_indices, 2],
        color='r', alpha=0.7, s=80, label='Sampled Points')
    plt.legend()
    plt.show()
