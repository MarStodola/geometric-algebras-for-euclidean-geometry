"""CGA functions"""

# imports
import numpy as np
from clifford.g3c import *
from pyganja import *

# init pseudoscalars
I_CGA = e1 ^ e2 ^ e3 ^ einf ^ eo
I_CRA = e1 ^ e2 ^ einf ^ eo
I_G3 = e1 ^ e2 ^ e3

# init planeXY
planeXY = e3

# basic CGA functions


def norm_plane(plane):
    """norm IPNS plane to standard representation"""
    return plane / (plane | plane).value[0] ** 0.5


def norm_line(line):
    """norm IPNS line to standard representation"""
    return line / (-(line | line).value[0]) ** 0.5


def norm_point(point):
    """norm CGA OPNS point to standard representation"""
    return -point / (point | einf).value[0]


def norm_sphere(sphere):
    """norm IPNS sphere to standard representation"""
    return norm_point(sphere)


def norm_flat_point(flat_point):
    """norm IPNS flat point to standard representation"""
    return - flat_point / ((I_CGA * (flat_point ^ einf)) | eo).value[0]


def norm_flat_point_opns(flat_point):
    """norm OPNS CGA flat point to standard representation"""
    return - flat_point / ((flat_point | einf) | eo).value[0]


def motor_between_two_planes(plane1, plane2):
    pl_1 = norm_plane(plane1)
    pl_2 = norm_plane(plane2)
    theta_m = np.arccos((pl_1 | pl_2).value[0])
    if 0 < theta_m < np.pi:
        line_m = norm_line(pl_1 ^ pl_2)
        return np.exp(- 0.5 * theta_m * line_m)
    else:
        return np.exp(-0.5 * pl_1 * (pl_2 - pl_1))


def duality_pga_partial_1(element):
    """pga partial duality $hat{D}_{P3}$"""
    return I_CGA * (element ^ einf)


def duality_pga_partial_2(element):
    """pga partial duality $overline{D}_{P3}$"""
    return -I_G3 * (element|eo)


def duality_pga(element):
    """pga duality $D_{P3}$"""
    return duality_pga_partial_1(element) + duality_pga_partial_2(element)


def clear_cga_object(cga_object):
    """The function sets values of cga element that are less than tolerance = 1e-12 to 0
    —helper function for drawing objects"""
    return np.dot(cga_object.value * (np.abs(cga_object.value) > 1e-12),
                  np.array(list(blades.values())))


def flat_point_to_point(flat_point):
    """The function converts IPNS flat point to OPNS CGA point"""
    pf = norm_flat_point(flat_point)
    return up(-duality_pga_partial_2(pf))


def log_motor(motor, n):
    """The inversion of exp(bivector). The function returns the bivector of the motor.
    It is only an approximation of the logarithm function by a sequence.
    n -  number of sequence members"""
    result = 0
    for i in range(1, n+1):
        result += 2 * ((motor -1) / (motor + 1)) ** (2 * i - 1) / (2 * i - 1)
    return result


def translator_to_point(point):
    """The function returns the translator from the origin to the CGA OPNS point"""
    return np.exp(-0.5 * ((norm_point(point) - eo) ^ einf))


def orientation(vector, point1, point2, point3):
    """orientation: sign of inner product of plane, where vector is a normal vector of the plane, and other plane
    given by point1, point2 and point3 in this order"""
    return np.sign((vector | (I_CGA * (point1 ^ point2 ^ point3 ^ einf))).value[0])
