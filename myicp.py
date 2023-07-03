import numpy as np

from typing import Tuple

from . import pointcloud, optimize


class ICP:
    """
    A class that implements the Iterative Closest Point (ICP) algorithm for
    aligning point clouds.
    """
    def __init__(
        self, pcd_fix: np.ndarray, pcd_mov: np.ndarray,
        sample_points_num: int = 1000,
    ) -> None:
        """
        Initializes the ICP algorithm with the fixed and moving point clouds.

        Parameters
        ----------
        pcd_fix : np.ndarray
            The fixed (target) point cloud.
        pcd_mov : np.ndarray
            The moving (source) point cloud.
        sample_points_num : int, optional
            The number of points to sample from the point clouds.
            Defaults to 1000.  # 自作の方だと重すぎる 頑張っても50あたりまで
        """
        self.pcd_fix = pointcloud.PointCloud(
            pcd_fix, sample_points_num, is_fixed=True, fast=True)
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
        convergence_condition : float, optional けすとおもう！！！！！！！！！！！！！！！！！！
            The convergence condition for the ICP algorithm. Defaults to 0.01.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The final transformation matrix and the transformed
            moving point cloud, and the residual list.
        """
        method_list = ["point to point", "point to plane", "color"]
        if optimization_method not in method_list:
            raise ICPException(
                f"optimization_method must be one of {method_list}")

        if max_iter == 0:
            np.savetxt(
                'output2.txt',
                self.pcd_mov.projective_coordinate, delimiter=' ')
            return None, self.pcd_mov.projective_coordinate

        print(f'Executing "{optimization_method}" pointcloud registration')
        optim = optimize.Optimize(
            optimization_method=optimization_method,
            pcd_fix=self.pcd_fix,
            pcd_mov=self.pcd_mov,
            sample_points_num=self.sample_points_num,
            init_homogeneous_params=init_homogeneous_params,
        )
        # print("residual = {:.10f}".format(optim.residual))
        residual_list = np.zeros(max_iter)

        print("Starting optimization...")
        # iterate optimization
        for step in range(max_iter):
            optim.update(neighbor_points_num)
            residual_list[step] = optim.residual
            self._print_status(step, max_iter, optim.residual)
            if step == max_iter - 1:
                print("Reached max iteration")
        H_result = optim.H_ret
        transformed_pcd_mov = self.pcd_mov.projective_coordinate
        # save as txt
        np.savetxt('output2.txt', transformed_pcd_mov, delimiter=' ')
        return H_result, transformed_pcd_mov, residual_list

    def _check_convergence(
            self, step: int, convergence_condition: float = 0.01):
        residual_diff = np.linalg.norm(
            self.residual_list[step + 1]
            - self.residual_list[step]) / self.residual_list[step]
        if step >= 1:
            print("Residual diff = {}".format(residual_diff))
        if residual_diff < convergence_condition:
            return True
        else:
            return False

    def _print_status(self, step: int, max_iter, residual: float):
        """
        Prints the status of the current iteration of the ICP algorithm.

        Parameters
        ----------
        step : int
            The current iteration step.
        max_iter : int
            The maximum number of iterations.
        residual : float
            The current value of the residual.
        """
        if step == 0:
            print(
                f"| {'iteration':^9s} | "
                f"{'residuals':^15s} | "
            )
            print("-------------------------------")
        print(
            f"| {step:^9d} | "
            f"{residual:^15.10f} |"
        )
        if step == max_iter - 1:
            print("-------------------------------")
        return None


class ICPException(Exception):
    """
    Custom exception for ICP algorithm related errors.
    """
    pass
