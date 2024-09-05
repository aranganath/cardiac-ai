import numpy as np
from scipy.spatial.transform import Rotation


def affine_transform(coords):
    angle = np.radians((np.random.rand(3) * 2.0 - 1.0) * 20) # +/- 20
    t_mat = (np.random.rand(3) * 2.0 - 1.0) * 10  # +/- 10 angstrom
    r_mat = Rotation.from_euler('xyz', angle).as_matrix()
    return np.matmul(coords, r_mat) + t_mat
