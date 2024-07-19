from numpy import linalg as la


def compute_eigenvalues_eigenvectors(matrix):
    eigenvalues, eigenvectors = la.eig(matrix)
    return eigenvalues, eigenvectors
