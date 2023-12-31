"""
This module implements an optimization class for the ICP algorithm,
used for aligning 3D point clouds.

The Optimize class provides the following functionalities:
    - Initializing the optimization process with various parameters, such as
        the optimization method, fixed and moving (source) point clouds,
        number of sample points, and initial homogeneous parameters.
    - Updating the transformation matrix and moving point cloud according to
        the selected optimization method, which can be one of "point to point",
        "point to plane", or "color".
    - Implementing different types of ICP algorithms as private methods:
        _point_to_point_icp, _point_to_plane_icp and _color_icp. These methods
        calculate the rotation matrix and translation vector (or homogenous
        transformation matrix), and residual that minimize the distance
        between the sampled points from the moving point cloud and their
        corresponding points in the fixed point cloud.
"""
import numpy as np

from typing import Tuple
import yaml
import os

from . import utils, pointcloud

_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")


class Optimize:
    """
    A class to implement the optimization methods used in the
    Iterative Closest Point (ICP) algorithm.
    """
    def __init__(
        self,
        optimization_method: str,
        pcd_fix: pointcloud.PointCloud,
        pcd_mov: pointcloud.PointCloud,
        sample_points_num: int,
        init_homogeneous_params: Tuple[float],
    ) -> None:
        """
        Initializes the optimization methods.

        Parameters
        ----------
        optimization_method : str
            The optimization method to use.
        pcd_fix : pointcloud.PointCloud
            The fixed (target) point cloud.
        pcd_mov : pointcloud.PointCloud
            The moving (source) point cloud.
        sample_points_num : int
            The number of points to sample from the point clouds.
        init_homogeneous_params : Tuple[float]
            The initial homogenous parameters (alpha, beta, gamma, tx, ty, tz).
        """
        self.optimization_method = optimization_method
        self.pcd_fix = pcd_fix
        self.pcd_mov = pcd_mov
        self.pcd_mov_orig = pcd_mov
        self.sample_points_num = sample_points_num
        self.H = utils.homogenous_params_to_homogenous_transformation_matrix(
            init_homogeneous_params)
        self.H_ret = \
            utils.homogenous_params_to_homogenous_transformation_matrix(
                init_homogeneous_params)
        self.residual = 0.
        self.residual_dist = 0.
        self.residual_dist_init = 0.

    def update(
        self,
        step: int,
        neighbor_points_num: int
    ) -> None:
        """
        Updates the transformation matrix and moving point cloud based on
        the selected optimization method.

        Parameters
        ----------
        step : int
            The current iteration step.
        neighbor_points_num : int
            The number of neighbor points to consider
            when calculating the normal vector.
        """
        # sample points from moving point cloud
        indices = self.pcd_mov.sample_indices
        sample_points = self.pcd_mov.projective_coordinate[indices]
        sample_points_RGB = self.pcd_mov.pcd_RGB[indices]

        # calculate correspondence
        correspondence_index_list = self.pcd_fix.calc_correspondence(
            source_points=sample_points,
        )
        correspondence = \
            self.pcd_fix.projective_coordinate[correspondence_index_list]
        correspondence_RGB = self.pcd_fix.pcd_RGB[correspondence_index_list]
        correspondence_normal_vectors = None

        # calculate normal vectors of each correspondence points
        if self.optimization_method in ["point to plane", "color"]:
            correspondence_normal_vectors = self.pcd_fix.calc_normal_vectors(
                correspondence_index_list,
                neighbor_points_num
            )

        # calculate initial residual
        if step == 0:
            for i in range(self.sample_points_num):
                self.residual_dist_init += np.linalg.norm(
                    correspondence[i] - sample_points[i]
                )
            self.residual_dist_init /= self.sample_points_num

        # update homogenous transformation matrix H
        H_new = self.H

        if self.optimization_method == "point to point":
            rot, trans, self.residual\
                = self._point_to_point_icp(
                 sample_points=sample_points,
                 correspondence=correspondence,
                )
            H_new = utils.homogenous_transformation_matrix(rot, trans)

        if self.optimization_method == "point to plane":
            rot, trans, self.residual\
                = self._point_to_plane_icp(
                 sample_points=sample_points,
                 correspondence=correspondence,
                 correspondence_normal_vectors=correspondence_normal_vectors,
                )
            H_new = utils.homogenous_transformation_matrix(rot, trans)

        if self.optimization_method == "color":
            H_new, self.residual\
                = self._color_icp(
                 sample_points=sample_points,
                 sample_points_RGB=sample_points_RGB,
                 correspondence=correspondence,
                 correspondence_normal_vectors=correspondence_normal_vectors,
                 correspondence_RGB=correspondence_RGB
                )

        self.H = H_new  # update H
        self.pcd_mov.homogenous_transformation(H_new)  # transform pcd_mov
        self.H_ret = np.dot(H_new, self.H_ret)  # update final H

        self.residual_dist = 0
        for i in range(self.sample_points_num):
            self.residual_dist += np.linalg.norm(
                correspondence[i] -
                self.pcd_mov.projective_coordinate[indices[i]]
            )
        self.residual_dist /= self.sample_points_num

        return None

    # This is implementation of following paper
    # P.J. Besl, Neil D. McKay,
    # A Method for Registration of 3-D Shapes, IEEE 1992.
    def _point_to_point_icp(
        self,
        sample_points: np.array,
        correspondence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Implements the point-to-point ICP algorithm.

        Parameters
        ----------
        sample_points : np.array
            The points sampled from the moving point cloud.
        correspondence : np.ndarray
            The corresponding points in the fixed point cloud.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            The rotation matrix and translation vector that
            minimizes the distance between the sample_points
            and their correspondence, and the computed residual.
        """
        # calculate R that maximizes \sum_i y_i \cdot R p_i in local coordinate
        mu_sample = np.mean(sample_points, axis=0)
        mu_corr = np.mean(correspondence, axis=0)

        C = np.zeros((3, 3))  # covariance matrix
        for i in range(self.sample_points_num):
            C += np.dot(
                sample_points[i].reshape(3, 1),
                correspondence[i].reshape(1, 3))
        C /= self.sample_points_num
        C -= np.dot(mu_sample.reshape(3, 1), mu_corr.reshape(1, 3))

        A = C - C.T
        delta = np.array([A[1, 2], A[2, 0], A[0, 1]])
        tr_C = np.trace(C)

        N = np.zeros((4, 4))  # move to fix
        N[0, 0] = tr_C
        N[0, 1:] = delta
        N[1:, 0] = delta
        N[1:, 1:] = C + C.T - tr_C * np.identity(3)

        w, v = np.linalg.eig(N)
        rot = utils.quaternion_to_rotation_matrix(v[:, np.argmax(w)])
        trans = mu_corr - np.dot(rot, mu_sample)

        residual = 0.
        for i in range(self.sample_points_num):
            x = correspondence[i] - np.dot(
                rot, sample_points[i].reshape(3, 1)).reshape(1, 3) - trans
            residual += float(np.dot(x, x.T))
        residual /= self.sample_points_num

        return rot, trans, residual

    # This is implementation of following paper
    # Kok-Lim Low,
    # Linear Least-Squares Optimization for
    # Point-to-Plane ICP Surface Registration, IEEE 2004.
    def _point_to_plane_icp(
        self,
        sample_points: np.array,
        correspondence: np.ndarray,
        correspondence_normal_vectors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Implements the point-to-plane ICP algorithm.

        Parameters
        ----------
        sample_points : np.array
            The points sampled from the moving point cloud.
        correspondence : np.ndarray
            The corresponding points in the fixed point cloud.
        correspondence_normal_vectors : np.ndarray
            The normal vectors at the corresponding points
            in the fixed point cloud.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            The rotation matrix and translation vector that
            minimizes the distance between the sample_points
            and their correspondence, and the computed residual.
        """
        # https://github.com/3d-point-cloud-processing/3dpcp_book_codes/blob/
        # master/section_registration/ICP-point-to-plane.ipynb
        # solve a linear minization problem (eq.8)
        A = np.zeros((6, 6))
        b = np.zeros((6, 1))
        # cst = 0.
        residual = 0

        for i in range(self.sample_points_num):
            xn = np.cross(sample_points[i], correspondence_normal_vectors[i])
            xn_n = np.hstack(
                (xn, correspondence_normal_vectors[i])).reshape(6, 1)

            n = correspondence_normal_vectors[i]
            nT = n.reshape(1, 3)
            p_x = (correspondence[i] - sample_points[i]).reshape(3, 1)

            A += np.dot(xn_n, xn_n.T)
            b += xn_n.dot(nT).dot(p_x)
            # cst += p_x.T.dot(nT.T).dot(nT).dot(p_x)
            residual += float(np.dot(p_x.T, n) ** 2)

        residual /= self.sample_points_num

        try:
            u_opt = np.dot(np.linalg.inv(A), b)
        except np.linalg.LinAlgError:
            u_opt = np.dot(np.linalg.pinv(A), b)
        theta = np.linalg.norm(u_opt[:3])
        w = (u_opt[:3] / theta).reshape(-1)

        rot = utils.axis_angle_to_rotation_matrix(w, theta)
        trans = np.array(list(map(float, u_opt[3:])))

        return rot, trans, residual

    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017.
    def _color_icp(
        self,
        sample_points: np.array,
        sample_points_RGB: np.ndarray,
        correspondence: np.ndarray,
        correspondence_normal_vectors: np.ndarray,
        correspondence_RGB: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Implements the color ICP algorithm.

        Parameters
        ----------
        sample_points : np.array
            The points sampled from the moving point cloud.
        sample_points_RGB : np.ndarray
            The RGB colors of the points sampled from the moving point cloud.
        correspondence : np.ndarray
            The corresponding points in the fixed point cloud.
        correspondence_normal_vectors : np.ndarray
            The normal vectors at the corresponding points
            in the fixed point cloud.
        correspondence_RGB : np.ndarray
            The RGB colors of the corresponding points
            in the fixed point cloud.

        Returns
        -------
        Tuple[np.ndarray, float]
            The homogenous transformation matrix that minimizes the distance
            between the sample_points and their correspondence
            and the difference in color, and the computed residual.
        """
        with open(_config_path) as f:
            _params = yaml.safe_load(f)
        # \in [0, 1], weight for joint optimization objective
        delta = _params["color_icp_delta"]
        # 0 for sRGB (and Rec709), 1 for YIQ, YUV and NTSC
        rgb_to_intensity_weight = \
            np.array(_params["rgb_to_intensity_weight"][0])

        # calculate color gradient for each correspondence
        target_color_gradient = np.zeros((self.sample_points_num, 3))
        for i in range(self.sample_points_num):
            p = correspondence[i]
            n_p = correspondence_normal_vectors[i]
            c_p = correspondence_RGB[i]
            i_p = float(
                np.dot(c_p.reshape(1, 3),
                       rgb_to_intensity_weight.reshape(3, 1))
                ) / 255.
            nn_num = 10  # num of nearest neighbor, temp value
            _, nn_idxs = self.pcd_fix.kdtree.query(p, k=nn_num)
            # solve a linear least square problem, (eq.10)
            A = np.zeros((3, nn_num))
            b = np.zeros(nn_num)
            for i, idx in enumerate(nn_idxs):
                if i == 0:
                    continue
                pp = self.pcd_fix.projective_coordinate[idx]
                c_pp = self.pcd_fix.pcd_RGB[idx]
                i_pp = float(
                    np.dot(c_pp.reshape(1, 3),
                           rgb_to_intensity_weight.reshape(3, 1))
                    ) / 255.
                # project pp (neighbor point) onto the plane
                pp_proj = pp - np.dot(pp - p, n_p) * n_p

            A[:, i - 1] = pp_proj - p
            b[i - 1] = i_pp - i_p
            # add orthogonal constraint np.dot(dp, n_p) = 0
            # with weight (nn_size-1)
            A[:, -1] = (nn_num - 1) * n_p
            b[-1] = 0

            try:
                X = np.dot(
                    np.linalg.inv(np.dot(A, A.T)),
                    np.dot(A, b).reshape(3, 1)
                )
            except np.linalg.LinAlgError:
                X = np.dot(
                    np.linalg.pinv(np.dot(A, A.T)),
                    np.dot(A, b).reshape(3, 1)
                )
            target_color_gradient[i] = X.flatten()

        # Gauss Newton
        J_geo = np.zeros((6, self.sample_points_num))
        r_geo = np.zeros(self.sample_points_num)
        J_col = np.zeros((6, self.sample_points_num))
        r_col = np.zeros(self.sample_points_num)

        for i in range(self.sample_points_num):
            dp = target_color_gradient[i]
            q = sample_points[i]
            c_q = sample_points_RGB[i]
            i_q = float(
                np.dot(c_q.reshape(1, 3),
                       rgb_to_intensity_weight.reshape(3, 1))
                ) / 255.

            p = correspondence[i]
            n_p = correspondence_normal_vectors[i]
            c_p = correspondence_RGB[i]
            i_p = float(
                np.dot(c_p.reshape(1, 3),
                       rgb_to_intensity_weight.reshape(3, 1))
                ) / 255.

            r_geo[i] = np.dot(q - p, n_p)  # eq.19
            J_geo[:, i] = np.hstack((np.cross(q, n_p), n_p))  # eq.30

            q_proj = q - np.dot(q - p, n_p) * n_p  # project q onto the plane
            i_q_proj = i_p + np.dot(dp, q_proj - p)

            M = np.identity(3) - np.dot(n_p, n_p.T)
            dp_M = np.dot(dp.T, M)

            r_col[i] = i_q_proj - i_q  # eq. 18
            J_col[:, i] = np.hstack((np.cross(q, dp_M), dp_M))  # eq.28-29

        # eq.21-17
        JTJ_geo = np.dot(J_geo, J_geo.T)
        JTr_geo = np.dot(J_geo, r_geo)
        JTJ_col = np.dot(J_col, J_col.T)
        JTr_col = np.dot(J_col, r_col)

        JTJ = np.sqrt(delta) * JTJ_geo + np.sqrt(1 - delta) * JTJ_col
        JTr = np.sqrt(delta) * JTr_geo + np.sqrt(1 - delta) * JTr_col

        # solve eq.21 to obtain homogenous parameters
        try:
            X = np.dot(np.linalg.inv(JTJ), -JTr)
        except np.linalg.LinAlgError:
            X = np.dot(np.linalg.pinv(JTJ), -JTr)
        H = utils.homogenous_params_to_homogenous_transformation_matrix(X)

        residual = delta * float(np.dot(r_geo, r_geo.T)) +\
            (1 - delta) * float(np.dot(r_col, r_col.T))  # eq.17
        residual /= self.sample_points_num

        return H, residual
