"""Fundamental matrix utilities."""

import math
import numpy as np

def point_line_distance(line, point):
    """Calculate line-point distance according to the formula
    from the project webpage.

    d(l, x) = (au + bv + c) / sqrt(a^2 + b^2)

    Arguments:
        line {3-vector} -- Line coordinates a, b, c
        point {3-vector} -- homogeneous point u, v, w

        Note that we don't really use w because w = 1 for the
        homogeneous coordinates

    Returns:
        float -- signed distance between line and point
    """

    a, b, c = line
    u, v, w = point
    error = 0

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    line = np.asarray(line, dtype=float).ravel()
    point = np.asarray(point, dtype=float).ravel()

    if line.size != 3:
        raise ValueError("line must be a 3-vector (a, b, c)")
    if point.size != 3:
        raise ValueError("point must be a 3-vector (u, v, w)")

    a, b, c = line
    u, v, w = point

    denom = np.hypot(a, b)  # for taking stable sqrt
    if np.isclose(denom, 0.0):
        raise ValueError("a and b are both zero, invalid")

    numer = a * u + b * v + c * w
    error =  float(numer / denom)

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return error

def signed_point_line_errors(x_0s, F, x_1s):
    """Calculate all signed line-to-point distances. Takes as input
    the list of x_0 and x_1 points, as well as the current estimate
    for F, and calculates errors for every pair of points and
    returns it as a list of floats.

    You'll want to call point_line_distance() to get the error
    between line and point.

    Keep in mind that this is a symmetric line-to-point error,
    meaning we calculate the line-to-point distance between Fx_1 and
    x_0, as well as F'x_0 and x_1, where F' is F transposed. You'll
    also have to append the errors to the errors list in that order,
    d(Fx_1,x_0) first then d(F'x_0,x_1) for every pair of points.

    Helpful functions: np.dot()

    Arguments:
        x_0s {Nx3 list} -- points in image 1
        F {3x3 array} -- Fundamental matrix
        x_1s {Nx3 list} -- points in image 2

    Returns:
        [float] {2N} -- list of d(Fx_1,x_0) and d(F'x_0,x_1) for each
        pair of points, because SciPy takes care of squaring and
        summing
    """
    assert F.shape == (3, 3)
    assert len(x_0s) == len(x_1s)
    errors = []

    #######################################################################
    # YOUR CODE HERE  |  Distance Computation                             #
    #######################################################################
    # Ensure arrays for safe indexing and dot products
    x0_arr = np.asarray(x_0s, dtype=float)
    x1_arr = np.asarray(x_1s, dtype=float)

    if x0_arr.ndim == 1:
        x0_arr = x0_arr.reshape(1, -1)
    if x1_arr.ndim == 1:
        x1_arr = x1_arr.reshape(1, -1)

    if x0_arr.shape[1] != 3 or x1_arr.shape[1] != 3:
        raise ValueError("Input points must be homogeneous 3-vectors (u,v,w)")

    for x0, x1 in zip(x0_arr, x1_arr):
        # line in image0 corresponding to x1
        l0 = F @ x1
        d0 = point_line_distance(l0, x0)
        errors.append(d0)

        # line in image1 corresponding to x0 (use F^T)
        l1 = F.T @ x0
        d1 = point_line_distance(l1, x1)
        errors.append(d1)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return errors

def skew(x, y, z):
    """Return the 3x3 skew-symmetric matrix for a vector (x, y, z)."""
    x, y, z = float(x), float(y), float(z)
    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ], dtype=float)


def create_F(K, R, t):
    """Create F from calibration and pose R,t between two views.
    Used in unit tests

    Arguments:
        K {3x3 matrix} -- Calibration matrix
        R {3x3 matrix} -- wRc, rotation from second camera to first (world)
        t {3-vector} -- wtc, position of camera in first (world)
    """
    x, y, z = t
    T = skew(x, y, z)
    Kinv = np.linalg.inv(K)
    F = np.dot(Kinv.T, T).dot(R).dot(Kinv)
    return F / np.linalg.norm(F)
