import numpy as np
"""
Functions for calculating forward and inverse kinematics of basic 2-link mechanisms using the classical approach
"""


def forward_kinematics(variant, thetas, len1, len2):
    """forvard kinematics of basic mechanism
    variant 0 - 2D mechanism
    variant 1 - 3D sliding mechanism
    variant 2 - 3D rotating mechanism"""
    if variant == 0:
        return forward_kinematics_v0(thetas[0], thetas[1], len1, len2)
    elif variant == 1:
        return forward_kinematics_v1(thetas[0], thetas[1], len1, len2, thetas[2])
    else:
        return forward_kinematics_v2(thetas[0], thetas[1], len1, len2, thetas[2])


def forward_kinematics_v0(theta0, theta1, len1, len2):
    """forvard kinematics of basic 2D mechanism"""
    return np.array([len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1),
                     len1 * np.sin(theta0) + len2 * np.sin(theta0 + theta1)])


def forward_kinematics_v1(theta0, theta1, len1, len2, d):
    """forvard kinematics of basic 3D sliding mechanism"""
    return np.array([len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1),
                     len1 * np.sin(theta0) + len2 * np.sin(theta0 + theta1),
                     d])


def forward_kinematics_v2(theta0, theta1, len1, len2, psi):
    """forvard kinematics of basic 3D rotating mechanism"""
    return np.array([(len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1)) * np.cos(psi),
                     len1 * np.sin(theta0) + len2 * np.sin(theta0 + theta1),
                     (len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1)) * np.sin(psi)])


def jacobi_matrix(variant, thetas, len1, len2):
    """jacobi matrix of forward kinematics of basic mechanism
    variant 0 - 2D mechanism
    variant 1 - 3D sliding mechanism
    variant 2 - 3D rotating mechanism"""
    if variant == 0:
        return jacobi_matrix_v0(thetas[0], thetas[1], len1, len2)
    elif variant == 1:
        return jacobi_matrix_v1(thetas[0], thetas[1], len1, len2)
    else:
        return jacobi_matrix_v2(thetas[0], thetas[1], len1, len2, thetas[2])


def jacobi_matrix_v0(theta0, theta1, len1, len2):
    """jacobi matrix of basic 2D mechanism"""
    return np.array([[-len1 * np.sin(theta0) - len2 * np.sin(theta0 + theta1), - len2 * np.sin(theta0 + theta1)],
                     [len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1), len2 * np.cos(theta0 + theta1)]])


def jacobi_matrix_v1(theta0, theta1, len1, len2):
    """jacobi matrix of basic 3D sliding mechanism"""
    return np.array([[-len1 * np.sin(theta0) - len2 * np.sin(theta0 + theta1), - len2 * np.sin(theta0 + theta1), 0],
                     [len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1), len2 * np.cos(theta0 + theta1), 0],
                     [0, 0, 1]])


def jacobi_matrix_v2(theta0, theta1, len1, len2, psi):
    """jacobi matrix of basic 3D rotating mechanism"""
    return np.array([[(-len1 * np.sin(theta0) - len2 * np.sin(theta0 + theta1)) * np.cos(psi),
                      - len2 * np.sin(theta0 + theta1) * np.cos(psi),
                      -(len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1)) * np.sin(psi)],
                     [len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1), len2 * np.cos(theta0 + theta1), 0],
                     [(-len1 * np.sin(theta0) - len2 * np.sin(theta0 + theta1)) * np.sin(psi),
                      - len2 * np.sin(theta0 + theta1) * np.sin(psi),
                      (len1 * np.cos(theta0) + len2 * np.cos(theta0 + theta1)) * np.cos(psi)]])


def angles_of_inverse_kinematics(variant, position, parameters_initial, len1, len2, eps):
    """angles of inverse kinematics of basic mechanism
    variant 0 - 2D mechanism
    variant 1 - 3D sliding mechanism
    variant 2 - 3D rotating mechanism"""
    index = 3
    if variant == 0:
        thetas = np.array([parameters_initial[0], parameters_initial[1]])
        pos_end = np.array([position[0], position[1]])
    elif variant == 1:
        thetas = np.array([parameters_initial[0], parameters_initial[1], parameters_initial[2]])
        pos_end = position
    else:
        thetas = np.array([parameters_initial[0], parameters_initial[1], parameters_initial[3]])
        pos_end = position
        index = 4
    pos = forward_kinematics(variant, thetas, len1, len2)
    diff = np.abs(pos_end - pos)
    while np.max(diff) > eps:
        jacobi = jacobi_matrix(variant, thetas, len1, len2)
        thetas += np.matmul(np.linalg.inv(jacobi), diff)
        thetas[:index] = thetas[:index] % (2 * np.pi) - (thetas[:index] < 0) * 2 * np.pi
        thetas[:index][thetas[:index] > np.pi] -= 2 * np.pi
        thetas[:index][thetas[:index] <= -np.pi] += 2 * np.pi

        pos = forward_kinematics(variant, thetas, len1, len2)
        diff = np.abs(pos_end - pos)

    return thetas


def angles_of_inverse_kinematics_v0(x_end, y_end, theta0_initial, theta1_initial, len1, len2, eps):
    """angles of inverse kinematics of basic 2D mechanism"""
    thetas = np.array([theta0_initial, theta1_initial])
    pos_end = np.array([x_end, y_end])
    pos = forward_kinematics_v0(thetas[0], thetas[1], len1, len2)
    diff = np.abs(pos_end - pos)
    while np.max(diff) > eps:
        jacobi = jacobi_matrix_v0(thetas[0], thetas[1], len1, len2)
        thetas += np.matmul(np.linalg.inv(jacobi), diff)
        pos = forward_kinematics_v0(thetas[0], thetas[1], len1, len2)
        diff = np.abs(pos_end - pos)

    return thetas


def angles_of_inverse_kinematics_v1(x_end, y_end, z_end, theta0_initial, theta1_initial, d_initial, len1, len2, eps):
    """angles of inverse kinematics of basic 3D sliding mechanism"""
    thetas = np.array([theta0_initial, theta1_initial, d_initial])
    pos_end = np.array([x_end, y_end, z_end])
    pos = forward_kinematics_v1(thetas[0], thetas[1], len1, len2, thetas[2])
    diff = np.abs(pos_end - pos)
    while np.max(diff) > eps:
        jacobi = jacobi_matrix_v1(thetas[0], thetas[1], len1, len2)
        thetas += np.matmul(np.linalg.inv(jacobi), diff)
        pos = forward_kinematics_v1(thetas[0], thetas[1], len1, len2, thetas[2])
        diff = np.abs(pos_end - pos)

    return thetas


def angles_of_inverse_kinematics_v2(x_end, y_end, z_end, theta0_initial, theta1_initial, psi_initial, len1, len2, eps):
    """angles of inverse kinematics of basic 3D rotating mechanism"""
    thetas = np.array([theta0_initial, theta1_initial, psi_initial])
    pos_end = np.array([x_end, y_end, z_end])
    pos = forward_kinematics_v2(thetas[0], thetas[1], len1, len2, thetas[2])
    diff = np.abs(pos_end - pos)
    while np.max(diff) > eps:
        jacobi = jacobi_matrix_v2(thetas[0], thetas[1], len1, len2, thetas[2])
        thetas += np.matmul(np.linalg.inv(jacobi), diff)
        pos = forward_kinematics_v2(thetas[0], thetas[1], len1, len2, thetas[2])
        diff = np.abs(pos_end - pos)

    return thetas


def position_of_p1(variant, position, parameters_initial, len1, len2, eps):
    """position of middle joint P1 of basic mechanism using classical approach
    variant 0 - 2D mechanism
    variant 1 - 3D sliding mechanism
    variant 2 - 3D rotating mechanism"""
    thetas = angles_of_inverse_kinematics(variant, position, parameters_initial, len1, len2, eps)
    return forward_kinematics(variant, thetas, len1, 0)





