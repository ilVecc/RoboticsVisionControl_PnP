import numpy as np
from tf.transformations import quaternion_multiply, quaternion_inverse, unit_vector, quaternion_about_axis

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
        # We just return a zero vector to point out this case
        v = np.zeros(3)
    else:
        v = q[0:3] / sin
        v = v / np.linalg.norm(v)
    return a, v


def quaternion_power(q, x):
    a, v = about_axis_from_quaternion(q)
    return quaternion_about_axis(a * x, v)


def quaternion_divide(q0, q1):
    return quaternion_multiply(q0, quaternion_inverse(q1))


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