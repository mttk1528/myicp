import numpy as np

from typing import Tuple

from . import pointcloud, optimize

from tqdm import tqdm

import time


class ICP:
    """
    A class that implements the Iterative Closest Point (ICP) algorithm for
    aligning point clouds.
    """
    def __init__(
        self, pcd_fix: np.ndarray, pcd_mov: np.ndarray,
        sample_points_num: int = 1000, fast=False
    ) -> None:
        """
        Initializes the ICP algorithm with the fixed and moving point clouds.

        Parameters
        ----------
        pcd_fix : np.ndarray
            The fixed (target) point cloud.
        pcd_mov : np.ndarray
            The moving (source) point cloud.
        pcd_mov_orig : np.ndarray
            The original moving (source) point cloud.
        sample_points_num : int, optional
            The number of points to sample from the point clouds.
            Defaults to 1000.  # 自作の方だと重すぎる 頑張っても50あたりまで
        fast : int, optional
            Use the faster cKDTree if True, otherwise use custom KDTree.
        """
        self.pcd_fix = pointcloud.PointCloud(
            pcd_fix, sample_points_num, is_fixed=True, fast=fast)
        self.pcd_mov = pointcloud.PointCloud(pcd_mov, sample_points_num)
        self.sample_points_num = sample_points_num

    def exec(
        self,
        optimization_method: str,
        max_iter: int = 10,
        neighbor_points_num: int = 10,
        init_homogeneous_params: Tuple[float] = (0., 0., 0., 0., 0., 0.),
        convergence_condition: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes the ICP algorithm.

        Parameters
        ----------
        optimization_method : str
            The optimization method to use. Can be "point to point",
            "point to plane", or "color".
        max_iter : int
            The maximum number of iterations to perform.
        neighbor_points_num : int, optional
            The number of neighbor points to consider
            when calculating the normal vector. Defaults to 10.
        init_homogeneous_params : Tuple[float], optional
            The initial homogenous parameters (alpha, beta, gamma, tx, ty, tz).
            Alpha, beta, and gamma are radians.
            Defaults to (0., 0., 0., 0., 0., 0.).
        convergence_condition : float, optional
            The convergence condition for the ICP algorithm. Defaults to 0.01.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The final transformation matrix and the transformed
            moving point cloud and the residual list, and
            the point to point residual list.
        """
        time_start = time.time()

        self.pcd_mov.projective_coordinate = \
            self.pcd_mov.projective_coordinate_orig.copy()

        method_list = ["point to point", "point to plane", "color"]
        if optimization_method not in method_list:
            raise ICPException(
                f"optimization_method must be one of {method_list}")

        if max_iter == 0:
            np.savetxt("ICP_result.txt", self.pcd_mov.projective_coordinate,
                       delimiter=" ")
            return None, self.pcd_mov.projective_coordinate

        print(f'---Executing "{optimization_method}" ICP---')
        optim = optimize.Optimize(
            optimization_method=optimization_method,
            pcd_fix=self.pcd_fix,
            pcd_mov=self.pcd_mov,
            sample_points_num=self.sample_points_num,
            init_homogeneous_params=init_homogeneous_params,
        )
        residual_list = np.zeros(max_iter)
        residual_ptop_list = np.zeros(max_iter + 1)

        # iterate optimization
        for step in tqdm(range(max_iter), bar_format="{l_bar}{bar:30}{r_bar}"):
            optim.update(step, neighbor_points_num)
            residual_list[step] = optim.residual
            residual_ptop_list[step + 1] = optim.residual_ptop
            # if _check_convergence(step, convergence_condition):
            #     pass
            if step == max_iter - 1:
                print("Reached max iteration!  ", end="")
        residual_ptop_list[0] = optim.residual_ptop_init
        self._print_result(step, residual_list, residual_ptop_list)
        H_result = optim.H_ret
        transformed_pcd_mov = np.hstack((
            self.pcd_mov.projective_coordinate,
            self.pcd_mov.pcd_RGB
        ))
        # save as txt
        np.savetxt("ICP_result.txt", transformed_pcd_mov, delimiter=" ")

        time_end = time.time()
        print(f"Total execution time: {(time_end - time_start):.5f}s")
        return H_result, transformed_pcd_mov, residual_list, residual_ptop_list

    # def _check_convergence(
    #         self, step: int, convergence_condition: float = 0.01):
    #     residual_diff = np.linalg.norm(
    #         self.residual_list[step + 1]
    #         - self.residual_list[step]) / self.residual_list[step]
    #     if step >= 1:
    #         print("Residual diff = {}".format(residual_diff))
    #     if residual_diff < convergence_condition:
    #         return True
    #     else:
    #         return False

    def _print_result(
        self,
        max_step: int,
        residual_list: np.ndarray,
        residual_ptop_list: np.ndarray
    ) -> None:
        """
        Prints the status of the current iteration of the ICP algorithm.

        Parameters
        ----------
        max_step : int
            The maximum number of iterations.
        residual_list : float
            The list of the residual.
        residual_ptop_list : float
            The list of the point to point residual.
        """
        print("Showing progress log:")
        for i in range(max_step + 1):
            if i == 0:
                print("------------------------------------------------")
                print(
                    f"| {'iteration':^9s}   "
                    f"{'residuals':^12s} "
                    f"  {'residuals(ptop)':^17s} | "
                )
            print(
                f"| {i:^9d}   "
                f"{residual_list[i]:^12.10f} "
                f"  {residual_ptop_list[i]:^17.10f} |"
            )
            if i == max_step:
                print("------------------------------------------------")
                pass
        print("Final residual                  = {:.10f}"
              .format(residual_list[max_step]))
        print("Final residual (point to point) = {:.10f}"
              .format(residual_ptop_list[max_step]))

        return None


class ICPException(Exception):
    """
    Custom exception for ICP algorithm related errors.
    """
    pass
