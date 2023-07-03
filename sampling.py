import numpy as np
from scipy.spatial import cKDTree
# from . import kdtree

# 重すぎて諦めました


class PoissonDiskSampler:
    """Perform Poisson Disk Sampling on a given point set in a 3D space."""
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

        # Initial random sample
        init_index = np.random.choice(active_list)
        indices.append(init_index)
        active_list.remove(init_index)

        # Loop until we get the required number of samples
        while len(indices) < num_samples and active_list:
            # Randomly choose from active list
            index = np.random.choice(active_list)
            point = self.points[index]

            # Get points within the range from the chosen point
            points_in_range = self.tree.query_ball_point(
                point, 2 * self.min_dist)

            # Choose a point that maintains the minimum distance
            for point_index in points_in_range:
                if point_index not in indices and np.min(
                    cKDTree(self.points[indices]).query(
                        self.points[point_index])[0]) > self.min_dist:
                    indices.append(point_index)
                    active_list.remove(point_index)
                    break
            else:
                active_list.remove(index)

        if len(indices) < num_samples:
            print("Warning: could not sample the desired number of points. \
                Consider decreasing min_dist.")

        return np.array(indices)
