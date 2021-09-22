import numpy as np


### MATRICES ###

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def getx(R):
    return R[:, 0].reshape((3, -1))


def gety(R):
    return R[:, 1].reshape((3, -1))


def getz(R):
    return R[:, 2].reshape((3, -1))


def rotx(th):
    return np.array([[1,          0,           0],
                     [0, np.cos(th), -np.sin(th)],
                     [0, np.sin(th),  np.cos(th)]])


def roty(th):
    return np.array([[np.cos(th),  0, np.sin(th)],
                     [          0, 1,          0],
                     [-np.sin(th), 0, np.cos(th)]])


def rotz(th):
    return np.array([[np.cos(th), -np.sin(th), 0],
                     [np.sin(th),  np.cos(th), 0],
                     [         0,           0, 1]])


# vectorized version of `tf.transformation.rotation_matrix()`
def rotations(angles, direction):
    M = (direction.reshape((3, 1)) * direction.T)[:, :, np.newaxis] * (1 - np.cos(angles)) \
        + skew(direction)[:, :, np.newaxis] * np.sin(angles) \
        + np.eye(3)[:, :, np.newaxis] * np.cos(angles)
    M /= np.linalg.norm(M, axis=0)
    return M
