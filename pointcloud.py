"""pointcloud class"""
import numpy as np
from scipy.spatial import cKDTree
from .kdtree import KDTree

# import time
# import sys
# from tqdm import tqdm


class PointCloud():
    """
    PointCloud class to manage and operate on a cloud of points.
    """

    def __init__(
        self,
        pcd: np.ndarray,
        sample_points_num: int,
        is_fixed: bool = False,
        fast: bool = False
    ) -> None:
        """
        Initializes the PointCloud object.

        Parameters
        ----------
        pcd : np.ndarray
            An array of 3D points.
        sample_points_num : int
            Number of points to sample.
        is_fixed : bool, optional
            Flag to decide if the PointCloud is fixed. Defaults to False.
        fast : bool, optional
            Flag to decide if to use the faster KDTree implementation, cKDTree.
            Defaults to False.
        """
        self.points_num = pcd.shape[0]
        self.sample_points_num = sample_points_num
        self.pcd_RGB = pcd[:, 3:]
        self.projective_coordinate = pcd[:, :3]
        self.sample_indices = np.random.choice(
            self.points_num,
            sample_points_num,
            replace=False
        )

        if is_fixed:
            self._build_kdtree(fast=fast)

    def _build_kdtree(self, fast: bool) -> None:
        """
        Builds the KDTree for efficient nearest neighbor searches.

        Parameters
        ----------
        fast : bool
            Use the faster cKDTree if True, otherwise use custom KDTree.
        """
        # 自作の方は重い というかscipy.spatial.cKDTreeが速すぎる
        if fast:
            self.kdtree = cKDTree(self.projective_coordinate)
        else:
            self.kdtree = KDTree(self.projective_coordinate)
        return None

    def select_sample_points(self) -> None:
        """
        Selects sample points from the PointCloud.
        """
        self.sample_indices = np.random.choice(
            self.points_num,
            self.sample_points_num,
            replace=False
        )
        return None

    def homogenous_transformation(self, H: np.ndarray) -> None:
        """
        Transforms the PointCloud with a homogeneous transformation matrix.

        Parameters
        ----------
        H : np.ndarray
            The transformation matrix.
        """
        pcd_h = np.column_stack((
            self.projective_coordinate,
            np.ones(self.points_num))
        ).T  # homogeneous coordinate
        pcd_h = np.dot(H, pcd_h)
        self.projective_coordinate = pcd_h[:-1, :].T  # projective coordinate
        return None

    def calc_correspondence(self, source_points: np.ndarray) -> np.ndarray:
        """
        Calculates the correspondence between the source points
        and the PointCloud.

        Parameters
        ----------
        source_points : np.ndarray
            The source points.

        Returns
        -------
        np.ndarray
            The correspondence indices in the PointCloud.
        """
        correspondence_index_list = np.zeros(source_points.shape[0])
        for i, source_point in enumerate(source_points):
            _, correspondence_indices = self.kdtree.query(
                x=source_point,
                k=1
            )
            correspondence_index_list[i] = correspondence_indices
        correspondence_index_list = np.array(
            list(map(int, correspondence_index_list)))
        return correspondence_index_list

    def calc_normal_vectors(
        self,
        correspondence_idx: np.ndarray,
        neibour_points_num: int
    ) -> np.ndarray:
        """
        Calculates the normal vectors for the points at the given indices.

        Parameters
        ----------
        correspondence_idx : np.ndarray
            The indices of the points.
        neibour_points_num : int
            The number of neighbor points to consider when calculating
            the normal vector.
        """
        # calculate normal vectors of each correspondence points
        correspondence_normal_vectors = np.zeros(
            (correspondence_idx.shape[0], 3))
        correspondence_idx = np.array(list(map(int, correspondence_idx)))

        for i, idx in enumerate(correspondence_idx):
            correspondence = self.projective_coordinate[idx]
            _, nearest_neighbor_idx_lst = self.kdtree.query(
                x=correspondence,
                k=neibour_points_num + 1  # nearest neighbor is itself
            )
            nearest_neighbor_lst = self.projective_coordinate[
                nearest_neighbor_idx_lst[1:]]
            C = np.cov(nearest_neighbor_lst.T, bias=False)
            eigenvalues, eigenvectors = np.linalg.eig(C)
            normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
            correspondence_normal_vectors[i] = normal_vector[:3]
        return correspondence_normal_vectors
