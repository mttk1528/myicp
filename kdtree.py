import numpy as np
import heapq
from typing import List, Tuple, Optional, Union


class Node:
    """
    A class representing a node in the k-dimensional tree (KDTree).

    Attributes
    ----------
    point : np.ndarray
        The point in k-dimensional space that this node represents.
    index : int
        The index of this point in the original dataset.
    split_dim : int
        The dimension on which this node splits the space.
    left : Node or None
        The left child node, representing all points less than this node along
        split_dim.
    right : Node or None
        The right child node, representing all points greater than this node
        along split_dim.
    """
    def __init__(
        self,
        point: np.ndarray,
        index: int,
        split_dim: int,
        left: Optional['Node'],
        right: Optional['Node']
    ) -> None:
        self.point = point
        self.index = index
        self.split_dim = split_dim
        self.left = left
        self.right = right


class KDTree:
    """
    A class for k-dimensional tree (KDTree). A KDTree is a space-partitioning
    data structure for organizing points in a k-dimensional space.

    Attributes
    ----------
    points : np.ndarray
        The points to build the KDTree from.
    indices : np.ndarray
        The indices of the points in the original dataset.
    root : Node or None
        The root node of the KDTree.

    Methods
    -------
    query(query_point: np.ndarray, k: int):
        Query the KDTree for k nearest neighbors to a given point. Returns the
        distances and indices of the k nearest neighbors.
    """
    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.indices = np.arange(points.shape[0])
        self.root = self._build(points, np.arange(points.shape[0]), 0)

    def _build(
            self, points: np.ndarray, indices: np.ndarray, split_dim: int
            ) -> Optional[Node]:
        if len(points) == 0:
            return None
        argsort_indices = points[:, split_dim].argsort()
        points = points[argsort_indices]
        indices = indices[argsort_indices]
        median = len(points) // 2
        return Node(
            points[median],
            indices[median],
            split_dim,
            self._build(points[:median],
                        indices[:median], (split_dim + 1) % 3),
            self._build(points[median + 1:],
                        indices[median + 1:], (split_dim + 1) % 3)
        )

    def _distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    def _k_nearest_neighbors(
        self,
        node: Optional[Node],
        query_point: np.ndarray,
        k: int,
        so_far: List[Tuple[float, np.ndarray]]
    ) -> List[Tuple[float, np.ndarray]]:
        if node is None:
            return so_far
        node_distance = self._distance(node.point, query_point)
        axis = node.split_dim
        diff = query_point[axis] - node.point[axis]

        if len(so_far) < k or node_distance < -so_far[0][0]:
            heapq.heappush(so_far, (-node_distance, node.index, node.point))
            if len(so_far) > k:
                heapq.heappop(so_far)

        if diff <= 0:
            close, away = node.left, node.right
        else:
            close, away = node.right, node.left
        so_far = self._k_nearest_neighbors(close, query_point, k, so_far)

        if not away or len(so_far) >= k and diff > -so_far[0][0]:
            return so_far

        return self._k_nearest_neighbors(away, query_point, k, so_far)

    def query(
            self, x: np.ndarray, k: int) -> Tuple[
                Union[np.ndarray, float], Union[np.ndarray, int]]:
        """
        Query the KDTree for k nearest neighbors to a given point.

        Parameters:
        -----------
        x : np.ndarray
            The query point.
        k : int
            The number of nearest neighbors to return.
            If k = 0, the function returns the nearest neighbor.
            If k > 0, the function returns k nearest neighbors.

        Returns
        -------
        tuple
            If k = 0, returns a tuple containing distance and index of the
            nearest neighbor.
            If k > 0, returns a tuple containing arrays of distances and
            indices of the k nearest neighbors.
        """
        result = self._k_nearest_neighbors(self.root, x, k, [])
        sorted_result = np.array(
            [(-i[0], i[1]) for i in heapq.nlargest(len(result), result)],
            dtype=object)
        distances, indices = map(np.array, zip(*sorted_result))
        if k == 0:
            return distances[0], indices[0]
        else:
            return distances, indices


if __name__ == "__main__":
    from scipy.spatial import cKDTree

    def test_kdtree(n_points=500, n_dim=3, n_query=100, k=10):

        points = np.random.rand(n_points, n_dim)
        query_points = np.random.rand(n_query, n_dim)
        tree_scipy = cKDTree(points)
        tree_custom = KDTree(points)
        for query in query_points:
            dists_scipy, idxs_scipy = tree_scipy.query(query, k)
            dists_custom, idxs_custom = tree_custom.query(query, k)
            assert set(idxs_scipy) == set(idxs_custom),\
                "Mismatch in nearest neighbor indices."
            for i_scipy, i_custom in zip(idxs_scipy, idxs_custom):
                np.testing.assert_almost_equal(
                    dists_scipy[list(idxs_scipy).index(i_scipy)],
                    dists_custom[list(idxs_custom).index(i_custom)],
                    decimal=5
                )

        print("All tests passed.")

    test_kdtree()
