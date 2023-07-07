"""
This module offers utility functionalities for 3D transformations including:
    - converting Euler angles to a rotation matrix
    - converting a rotation matrix to Euler angles
    - converting axis-angle representation to a rotation matrix
    - converting a quaternion to a rotation matrix
    - creating a homogeneous transformation matrix from homogeneous parameters
    - creating a homogeneous transformation matrix from a rotation matrix and
        a translation vector
    - creating a homogeneous transformation matrix from a quaternion and a
        translation vector
"""
import numpy as np
from typing import Tuple


def euler_angles_to_rotation_matrix(
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """
    Converts Euler angles to a 3D rotation matrix.

    Parameters
    ----------
    alpha : float
        The rotation angle around the x-axis.
    beta : float
        The rotation angle around the y-axis.
    gamma : float
        The rotation angle around the z-axis.

    Returns
    -------
    np.ndarray
        The 3D rotation matrix.
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


def rotation_matrix_to_euler_angles(
    R: np.ndarray
) -> Tuple[float, float, float]:
    """
    Converts a 3D rotation matrix to Euler angles.

    Parameters
    ----------
    R : np.ndarray
        The 3D rotation matrix.

    Returns
    -------
    Tuple[float, float, float]
        The rotation angles around the x, y, and z axes (alpha, beta, gamma).
    """
    alpha = np.arctan2(-R[1, 2], R[2, 2])
    beta = np.arcsin(R[0, 2])
    gamma = np.arctan2(-R[0, 1], R[0, 0])

    return (alpha, beta, gamma)


def axis_angle_to_rotation_matrix(
    axis: np.ndarray, theta: float
) -> np.ndarray:
    """
    Converts axis-angle representation to a rotation matrix.

    Parameters
    ----------
    axis : np.ndarray
        The rotation axis.
    theta : float
        The rotation angle.

    Returns
    -------
    np.ndarray
        The corresponding rotation matrix.
    """
    warp_matrix = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    rot = np.identity(3) + (np.sin(theta) * warp_matrix) + \
        ((1 - np.cos(theta)) * np.dot(warp_matrix, warp_matrix))

    return rot


def quaternion_to_rotation_matrix(
    q: np.ndarray
) -> np.ndarray:
    """
    Converts a quaternion representation to a rotation matrix.

    Parameters
    ----------
    q : np.ndarray
        The quaternion.

    Returns
    -------
    np.ndarray
        The corresponding rotation matrix.
    """
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


def homogenous_transformation_matrix(
    R: np.ndarray,
    t: np.ndarray
) -> np.array:
    """
    Creates a 3D homogeneous transformation matrix.

    Parameters
    ----------
    R : np.ndarray
        The 3D rotation matrix.
    t : np.ndarray
        The translation vector.

    Returns
    -------
    np.array
        The 3D homogeneous transformation matrix.
    """
    return np.array([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [0., 0., 0., 1.]
    ])


def homogenous_params_to_homogenous_transformation_matrix(
    homogenous_params: Tuple[float] = (0., 0., 0., 0., 0., 0.)
) -> np.ndarray:
    """
    Creates a 3D homogeneous transformation matrix from homogeneous parameters.

    Parameters
    ----------
    homogenous_params : Tuple[float]
        The homogeneous parameters, including three rotation angles and
        three translation distances.

    Returns
    -------
    np.ndarray
        The 3D homogeneous transformation matrix.
    """
    return homogenous_transformation_matrix(
        euler_angles_to_rotation_matrix(
            homogenous_params[0],
            homogenous_params[1],
            homogenous_params[2]
        ),
        homogenous_params[3:]
    )
