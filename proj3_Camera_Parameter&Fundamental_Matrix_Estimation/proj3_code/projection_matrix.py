import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    # ensure points are Nx4 homogeneous
    pts = np.asarray(points_3d)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] == 3:
        ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
        pts_h = np.hstack([pts, ones])
    elif pts.shape[1] == 4:
        pts_h = pts.copy()
    else:
        raise ValueError("points_3d must be shape (N,3) or (N,4)")

    # project
    proj = (P @ pts_h.T)  # shape (3, N)
    proj = proj.T  # shape (N, 3)

    # avoid division by zero
    w = proj[:, 2:3]
    if np.any(np.isclose(w, 0.0)):
        # small epsilon to avoid divide by zero; points at infinity remain large
        w = w + 1e-12

    projected_points_2d = proj[:, 0:2] / w
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return projected_points_2d


def objective_func(x, **kwargs):
    """
        Calculates the difference in image (pixel coordinates) and returns
        it as a 2*n_points vector

        Args:
        -        x: numpy array of 11 parameters of P in vector form
                    (remember you will have to fix P_34=1) to estimate the reprojection error
        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                   	retrieve these 2D (using the key ‘pts2d’) and 3D(using the key ‘pts3d’) points and then
		            use them to compute the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences between
                    projected and actual 2D points. (the difference between all the x
                    and all the y coordinates)

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    pts2d = kwargs.get('pts2d')
    pts3d = kwargs.get('pts3d')
    if pts2d is None or pts3d is None:
        raise ValueError("objective_func requires pts2d and pts3d in the second argument")

    # reconstruct full 3x4 P from 11-vector x, fixing last element to 1
    x = np.asarray(x).ravel()
    if x.size != 11:
        raise ValueError("Expected vector of length 11 to represent P with last element fixed to 1")

    p = np.zeros(12, dtype=x.dtype)
    p[:11] = x
    p[11] = 1.0
    P = p.reshape(3, 4)

    projected = projection(P, pts3d)  # (N,2)
    pts2d_arr = np.asarray(pts2d)
    if pts2d_arr.ndim == 1:
        pts2d_arr = pts2d_arr.reshape(1, -1)
    if projected.shape != pts2d_arr.shape:
        raise ValueError("Projected points and pts2d must have same shape")

    diff = (projected - pts2d_arr).ravel()
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return diff


def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1)
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters.

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables
                          for the objective function
    '''

    start_time = time.time()

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    P0_flat = initial_guess.flatten()
    x0 = np.hstack([P0_flat[0:11]])
    
    kwargs = {
        'pts2d': pts2d,
        'pts3d': pts3d
    }
    
    res = least_squares(
        fun=objective_func,
        x0=x0,
        method='lm',
        max_nfev=50000,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
        verbose=2,
        kwargs=kwargs
    )
    x = res.x
    P = np.concatenate([x[:11], [1.0]]).reshape((3,4))
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    print("Time since optimization start", time.time() - start_time)

    return P

def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    M = P[:, :3]

    # Perform RQ decomposition
    K, R = rq(M)
    
    # Ensure positive diagonal elements in K
    signs = np.sign(np.diag(K))
    signs[signs == 0] = 1.0
    S = np.diag(signs)
    
    # Adjust K and R using the sign matrix
    K = K @ S
    R = S @ R
    
    # Handle possible reflection (determinant negative)
    if np.linalg.det(R) < 0:
        flip = np.diag([1, 1, -1])
        K = K @ flip
        R = flip @ R
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return K, R

def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (3,) representing the camera center
            location in world coordinates
    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    KRt = np.linalg.inv(K) @ P
    tr = KRt[:, 3]
    cc = - R.T @ tr # Needs to be negated
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return cc
