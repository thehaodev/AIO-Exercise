import numpy as np


def compute_cosine(v1, v2):
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine = np.dot(v1, v2) / (magnitude_v1 * magnitude_v2)
    return cosine
