"""
This module implements a KDTree class with Node class,
a space-partitioning data structure for organizing points in a k-dimensional
space.

The KDTree class provides the following functionalities:
    - Building a KDTree from a set of points in k-dimensional space.
    - Querying the KDTree for k nearest neighbors to a given point.
"""
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
        """
        Construct a KDTree instance.

        Parameters
        ----------
        points : np.ndarray
            The points in k-dimensional space to build the KDTree from.
        """
        self.points = points
        self.indices = np.arange(points.shape[0])
        self.root = self._build(points, np.arange(points.shape[0]), 0)

    def _build(
            self, points: np.ndarray, indices: np.ndarray, split_dim: int
            ) -> Optional[Node]:
        """
        Recursive method to build the KDTree.

        Parameters
        ----------
        points : np.ndarray
            The points in k-dimensional space.
        indices : np.ndarray
            The indices of the points in the original dataset.
        split_dim : int
            The dimension along which to split the points.

        Returns
        -------
        Node or None
            The root of the built (sub-)tree.
        """
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
        """
        Compute the Euclidean distance between two points.

        Parameters
        ----------
        point1, point2 : np.ndarray
            The points between which to compute the distance.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """
        return np.linalg.norm(point1 - point2)

    def _k_nearest_neighbors(
        self,
        node: Optional[Node],
        query_point: np.ndarray,
        k: int,
        so_far: List[Tuple[float, np.ndarray]]
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Recursive method to find the k nearest neighbors of a given point.

        Parameters
        ----------
        node : Node or None
            The current node.
        query_point : np.ndarray
            The point to find the nearest neighbors of.
        k : int
            The number of nearest neighbors to find.
        so_far : List[Tuple[float, np.ndarray]]
            The current list of nearest neighbors.

        Returns
        -------
        List[Tuple[float, np.ndarray]]
            The updated list of nearest neighbors.
        """
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

    def _query_ball_point(
        self,
        node: Optional[Node],
        query_point: np.ndarray,
        r: float,
        indices_in_range: List[int]
    ) -> List[int]:
        """
        A private method to find all points in the KDTree within a radius r
        of the given query point. This is a recursive function that traverses
        the KDTree in a depth-first manner.

        Parameters
        ----------
        node : Optional[Node]
            The current node being inspected. This node will be None when
            the function has traversed all relevant branches of the tree.
        query_point : np.ndarray
            The point from which to measure the distance r.
        r : float
            The radius within which to find all points in the KDTree.

        Returns
        -------
        List[int]
            A list of indices of all points within the KDTree that are within
            a radius r of the query point.
        """
        if node is None:
            return indices_in_range
        node_distance = self._distance(node.point, query_point)
        axis = node.split_dim
        diff = query_point[axis] - node.point[axis]

        if node_distance <= r:
            indices_in_range.append(node.index)

        if diff <= 0:
            close, away = node.left, node.right
        else:
            close, away = node.right, node.left

        indices_in_range = self._query_ball_point(close, query_point,
                                                  r, indices_in_range)

        if not away or diff > r:
            return indices_in_range

        return self._query_ball_point(away, query_point, r, indices_in_range)

    def query_ball_point(self, x: np.ndarray, r: float) -> np.ndarray:
        """
        Query the KDTree for all points within a specified distance r to
        a given point.

        Parameters:
        -----------
        x : np.ndarray
            The query point.
        r : float
            The distance within which to return all points.

        Returns
        -------
        np.ndarray
            The indices of all points within a distance r to the query point.
        """
        indices_in_range = self._query_ball_point(self.root, x, r, [])
        return np.array(indices_in_range)

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

    def test_kdtree(n_points=500, n_dim=3, n_query=100, k=10, r=0.5):

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
            # test for query_ball_point
            idxs_scipy = tree_scipy.query_ball_point(query, r)
            idxs_custom = tree_custom.query_ball_point(query, r)
            assert set(idxs_scipy) == set(idxs_custom),\
                "Mismatch in query_ball_point indices."

        print("All tests passed.")

    test_kdtree()
