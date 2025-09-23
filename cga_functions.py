"""CGA functions"""

# imports
import numpy as np
from clifford.g3c import *  # tools for cga computation
from pyganja import *  # tools for visualizations


# init pseudoscalars
I_CGA = e1 ^ e2 ^ e3 ^ einf ^ eo
I_CRA = e1 ^ e2 ^ einf ^ eo
I_G3 = e1 ^ e2 ^ e3

# init planeXY
planeXY = e3

# basic CGA functions


def np_to_cga(point_np):
    """convert point as np array to cga point"""
    return up(point_np[0] * e1 + point_np[1] * e2 + point_np[2] * e3)


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


def points_of_point_pair(point_pair):
    """return two cga points of cga point-pair """
    return (- np.sqrt((point_pair|point_pair).value[0]) + point_pair) / (-einf|point_pair),\
           (np.sqrt((point_pair|point_pair).value[0]) + point_pair) / (-einf|point_pair)


def sphere(point, radius):
    """return cga sphere given by cga point and radius"""
    return point - 0.5 * radius ** 2 * einf


def motor_between_two_planes(plane1, plane2):
    """motor that converts plane1 to plane2"""
    pl_1 = norm_plane(plane1)
    pl_2 = norm_plane(plane2)
    theta_m = np.arccos((pl_1 | pl_2).value[0])
    if 0 < theta_m < np.pi:
        line_m = norm_line(pl_1 ^ pl_2)
        return np.exp(- 0.5 * theta_m * line_m)
    else:
        return np.exp(-0.5 * (pl_1 ^ pl_2))


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


def orientation_with_line(vector, point, line):
    """orientation: sign of inner product of plane, where vector is a normal vector of the plane, and other plane
    given by point1, point2 and point3 in this order"""
    return np.sign((vector | (I_CGA * (point ^ line))).value[0])


def position_of_p1_by_cga(variant, position, len1, len2, I_C2_pl):
    """position of middle joint P1 of basic (2-link) mechanism using GA
    variant 0 - 2D mechanism
    variant 1 - 3D sliding mechanism
    variant 2 - 3D rotating mechanism"""
    if variant == 1:
        P0 = up(position.value[3] * e3)
    else:
        P0 = eo
    P2 = position
    big_return = False
    if I_C2_pl is None:
        big_return = True
        if variant == 0:
            plane = e3
        elif variant == 1:
            plane = e3 + position.value[3] * einf
        else:
            plane = -I_CGA * (eo ^ up(e2) ^ P2 ^ einf)
        motor_to_plane = motor_between_two_planes(planeXY, plane)
        I_C2_pl = motor_to_plane * I_CRA * ~motor_to_plane

    sphere0 = sphere(P0, len1)
    sphere1 = sphere(P2, len2)
    point_pair = I_C2_pl * (sphere0 ^ sphere1)
    P1a, P1B = points_of_point_pair(point_pair)
    if big_return:
        return P0, norm_point(P1a), norm_point(P1B), plane, I_C2_pl
    return P0, norm_point(P1a), norm_point(P1B)


def motor_between_start_and_end_basic(p_start : list, p_end : list, I_pl_end):
    """motor between two positions of basic (2-link) mechanism"""
    plane_start = I_CGA * (p_start[0] ^ p_start[1] ^ p_start[2] ^ einf)
    plane_end = I_CGA * (p_end[0] ^ p_end[1] ^ p_end[2] ^ einf)
    motor0 = motor_between_two_planes(plane_start, plane_end)
    axis0 = motor0 * (e1 ^ e2) * ~motor0

    p_tr = []
    for point in p_start:
        p_tr.append(norm_point(motor0 * point * ~motor0))

    line01_shifted = norm_line(I_CGA * (p_tr[0] ^ p_tr[1] ^ einf))
    line01_end = norm_line(I_CGA * (p_tr[0] ^ p_end[1] ^ einf))

    line12_shifted = norm_line(I_CGA * (p_tr[1] ^ p_tr[2] ^ einf))
    line12_end = norm_line(I_CGA * (p_end[1] ^ p_end[2] ^ einf))

    theta0 = orientation(e3, p_tr[0],
                         p_tr[1], p_end[1]) * np.arccos(-(line01_shifted |
                                                                   line01_end).value[0])

    motor1 = np.exp(-0.5 * theta0 * axis0) * motor0

    theta1 = orientation(e3, p_tr[1], p_tr[2],
                         p_end[2]) * np.arccos(-(line12_shifted |
                                                   line12_end).value[0]) - theta0

    axis1_end = norm_line(-I_pl_end * (p_end[1] ^ einf))

    motor = np.exp(-0.5 * theta1 * axis1_end) * motor1
    return motor0, motor1, motor

