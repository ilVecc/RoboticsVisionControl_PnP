import numpy as np
import quaternion
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_inverse, unit_vector, quaternion_about_axis

posedt = np.dtype([('p', np.float64, (3,)), ('o', np.quaternion, 1)])


# TODO not ready
class position(np.ndarray):

    def __init__(self, object):
        super(position, self).__init__(object)
    
    def __repr__(self):
        return "position(%s)" % (self.data)
    
    def __str__(self):
        return self.__repr__()


# TODO not ready
class Pose(np.ndarray):

    def __init__(self, p, o):
        if len(p) != 3:
            assert False, "position must be a 3-value array"
        if len(o) != 4:
            assert False, "orientation must be a 4-value array"
        self.pos = np.array(p)
        self.ori = np.quaternion(o)
        super(Pose, self).__init__(p + o)
    
    def __add__(self, pose):
        return Pose(pose.pos + self.pos, pose.ori * self.ori)
    
    def __sub__(self, pose):
        return Pose(self.pos - pose.pos, pose.ori / self.ori)
    
    # def __repr__(self):
    #     return "(%s,%s)" % (self.p, quaternion.as_float_array(self.o))
    
    # def __str__(self):
    #     return self.__repr__()


# FIXME workaround for the lack of a native NumPy C++ API "pose" dtype
def pose(p, o):
    if not isinstance(o, np.ndarray):
        return np.array((p, o), dtype=posedt)
    return np.array(zip(p, o), dtype=posedt)


def quaternion_convert(q):
    if isinstance(q, list):
        q = np.array(q)
    if isinstance(q, np.quaternion) or q.ndim <= 1:
        return np.roll(quaternion.as_float_array(q), -1)
    return np.roll(quaternion.as_float_array(q), -1, axis=1)


def quaternion_convert_from(q):
    if isinstance(q, list):
        q = np.array(q)
    if isinstance(q, np.quaternion) or q.ndim == 1:
        return quaternion.from_float_array(np.roll(q, 1))
    return quaternion.from_float_array(np.roll(q, 1, axis=1))


def quaternion_average(Q):
    # the average quaternion is the strongest eigenvector (i.e. with maximum eigenvalue)
    # https://core.ac.uk/download/pdf/10536576.pdf
    # (averaging is performed over the last axis, meaning that K time series N quaternions must be
    # passed as a NxK quaternion matrix)
    AT = quaternion.as_float_array(Q)  # obtain a NxKx4 matrix
    A = np.swapaxes(AT, -1, -2)  # swap the last two axes, obtaining a Nx4xK matrix
    M = np.matmul(A, AT)  # dot product over the first axis, to obtain a Nx4x4 matrix
    l, v = np.linalg.eig(M)  # find eigenvalues/vectors, for each of the N 4x4 matrices
    l_max_idx = np.argsort(l)[..., -1]  # index of biggest eigenvalue, for each of the N matrices
    if l_max_idx.ndim == 0:  # eigenvectors (already normalized) with biggest eigenvalues, for each of the N matrices
        v_max = v[:, l_max_idx]  # here we are 2D and  l_max_idx.shape == ()
    else:
        v_max = v[np.arange(l_max_idx.shape[0]), :, l_max_idx]
    q_avg = quaternion.from_float_array(v_max)
    return q_avg


###################################
#            DEPRECATED           #
###################################


def quaternions_angle(q0, q1):
    z = relative_quaternion(q0, q1)
    angle, _ = about_axis_from_quaternion(z)
    return angle


def relative_quaternion(q0, q1):
    q0 = unit_vector(q0)
    q1 = unit_vector(q1)
    # since we have unit quaternions, we could be using the conjugate
    z = unit_vector(quaternion_divide(q1, q0))
    return z


def about_axis_from_quaternion(q):
    q = unit_vector(q)
    a = 2 * np.arctan2(np.linalg.norm(q[0:3]), q[3])
    sin = np.sin(a / 2)
    if sin < 1e-10:
        # This is a 2pi rotation, thus a non-rotation, 
        # and the axis could be any vector.
        # We just return a standard z vector in this case
        v = np.array([0, 0, 1])
    else:
        v = q[0:3] / sin
        v = v / np.linalg.norm(v)
    return a, v


def quaternion_power(q, x):
    a, v = about_axis_from_quaternion(q)
    return quaternion_about_axis(a * x, v)


def quaternion_divide(q0, q1):
    return quaternion_multiply(q0, quaternion_inverse(q1))


def quaternion_rotate(q, v):
    return np.dot(quaternion_matrix(q)[0:3, 0:3], v)


def _batch_quaternion_operation(Q0, Q1, operation):
    # Q0 and Q1 must be (n,4)
    if Q0.shape != Q1.shape:
        raise RuntimeError('Matrices must have the same (n,4) shape')
    return np.array([operation(Q0[i, :], Q1[i, :]) for i in range(Q0.shape[0])])


def batch_quaternion_multiply(Q0, Q1):
    return _batch_quaternion_operation(Q0, Q1, quaternion_multiply)


def batch_quaternion_divide(Q0, Q1):
    return _batch_quaternion_operation(Q0, Q1, quaternion_divide)


if __name__ == '__main__':
    q = quaternion_about_axis(0.0, (1, 1, 0))
    a, v = about_axis_from_quaternion(q)
    print(a)
    print(v)

    p1 = Pose([0, 0, 0], [0, 0.707, 0, 0.707])
    p2 = Pose([2, 2, 2], [0, 0.707, 0, 0.707])

    print(p1 + p2)