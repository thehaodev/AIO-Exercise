import numpy as np


def length_of_vector(vector):
    return np.linalg.norm(vector)


def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)


def matrix_multi_vector(matrix, vec):
    return np.dot(matrix, vec)


def matrix_multi_matrix(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def matrix_inverse(matrix):
    det_matrix = np.linalg.det(matrix)
    if det_matrix != 0:
        return np.linalg.inv(matrix)
    else:
        return "This matrix can not be inverse"
