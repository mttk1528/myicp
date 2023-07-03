"""
utility functions
"""
import numpy as np
from typing import Tuple


def euler_angles_to_rotation_matrix(
        alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Creating 3d rotation matrix from euler angles
    """
    return np.array([
        [np.cos(beta) * np.cos(gamma),
         -np.cos(beta) * np.sin(gamma), np.sin(beta)],
        [np.cos(alpha) * np.sin(gamma) +
            np.sin(alpha) * np.sin(beta) * np.cos(gamma),
            np.cos(alpha) * np.cos(gamma)
            - np.sin(alpha) * np.sin(beta) * np.sin(gamma),
            - np.sin(alpha) * np.cos(beta)],
        [np.sin(alpha) * np.sin(gamma) -
         np.cos(alpha) * np.sin(beta) * np.cos(gamma),
            np.sin(alpha) * np.cos(gamma) +
            np.cos(alpha) * np.sin(beta) * np.sin(gamma),
            np.cos(alpha) * np.cos(beta)]
    ])


def homogenous_transformation_matrix(
    R: np.ndarray,
    t: np.ndarray
) -> np.array:
    """
    Creating 3d homogenous transformation matrix
    """
    return np.array([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [0., 0., 0., 1.]
    ])


def rotation_matrix_to_euler_angles(
        R: np.ndarray) -> Tuple[float, float, float]:

    alpha = np.arctan2(-R[1, 2], R[2, 2])
    beta = np.arcsin(R[0, 2])
    gamma = np.arctan2(-R[0, 1], R[0, 0])

    return (alpha, beta, gamma)


def homogenous_params_to_homogenous_transformation_matrix(
    homogenous_params: Tuple[float] = (0., 0., 0., 0., 0., 0.)
) -> np.ndarray:
    """
    Creating 3d homogenous transformation matrix from homogenous parameters
    """
    return homogenous_transformation_matrix(
        euler_angles_to_rotation_matrix(
            homogenous_params[0],
            homogenous_params[1],
            homogenous_params[2]
        ),
        homogenous_params[3:]
    )


def axis_angle_to_rotation_matrix(axis: np.ndarray, theta: float):
    """
    Args:
      axis(ndarray): rotation axis
      theta(float): rotation angle
    """

    warp_matrix = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    rot = np.identity(3) + (np.sin(theta) * warp_matrix) + \
        ((1 - np.cos(theta)) * np.dot(warp_matrix, warp_matrix))

    return rot


def quaternion_to_rotation_matrix(q: np.ndarray):
    return np.array([
        [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
            2.0 * (q[1] * q[2] - q[0] * q[3]),
            2.0 * (q[1] * q[3] + q[0] * q[2])],
        [2.0 * (q[1] * q[2] + q[0] * q[3]),
            q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2,
            2.0 * (q[2] * q[3] - q[0] * q[1])],
        [2.0 * (q[1] * q[3] - q[0] * q[2]),
            2.0 * (q[2] * q[3] + q[0] * q[1]),
            q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2]
    ])
