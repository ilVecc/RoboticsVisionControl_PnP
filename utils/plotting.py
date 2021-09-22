import numpy as np
import quaternion
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tf.transformations import quaternion_matrix
from utils.quaternions import posedt


def plot_poses(poses, keep_aspect_ratio=True):
    _ = plt.figure()
    ax = plt.axes(projection="3d")
    for pose in poses:
        p, o = pose['p'], pose['o'][()]
        R = quaternion.as_rotation_matrix(o)
        ax.quiver(p[0], p[1], p[2], R[0, 0], R[1, 0], R[2, 0], color='r', pivot='tail')
        ax.quiver(p[0], p[1], p[2], R[0, 1], R[1, 1], R[2, 1], color='g', pivot='tail')
        ax.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2], color='b', pivot='tail')
    min_ = np.min(poses['p'], axis=0) - 1
    max_ = np.max(poses['p'], axis=0) + 1
    lim = np.stack([min_, max_], axis=1)
    if keep_aspect_ratio:
        ranges = lim[:, 1] - lim[:, 0]
        max_range = np.max(ranges)
        lim = lim / ranges[:, np.newaxis] * max_range  # scale ranges
    ax.set_xlim(lim[0, :])
    ax.set_ylim(lim[1, :])
    ax.set_zlim(lim[2, :])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()