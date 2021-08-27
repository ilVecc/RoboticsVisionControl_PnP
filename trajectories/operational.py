import numpy as np
from tf.transformations import quaternion_matrix, quaternion_slerp, quaternion_multiply
from utils.filters import numpy_from_pose, numpy_from_quaternion
from utils.matrices import skew, rotations
from utils.quaternions import relative_quaternion, about_axis_from_quaternion, batch_quaternion_multiply
from trajectories.profiles import Unit_Space_Scaler, Space_Scaler


class Trajectory(object):

    def __init__(self, time_law):
        self.time_law = time_law

    def __call__(self, n_samples):
        return self.sample(n_samples)

    def sample(self, n_samples):
        t = np.linspace(0.0, self.time_law.DT, num=n_samples, endpoint=True)
        return t, self.get(t)

    def __getitem__(self, T):
        return self.get(T)

    def get(self, T):
        raise NotImplementedError('No motion law specified for this trajectory')


# TODO add support to jerk
class Linear_Trajectory(Trajectory):

    def __init__(self, time_law, pose_init, pose_end):
        super(Linear_Trajectory, self).__init__(time_law)

        self.pi, self.oi = pose_init[0:3].reshape([1, 3]), pose_init[3:7]
        self.pf, self.of = pose_end[0:3].reshape([1, 3]), pose_end[3:7]

        self.time_law = Unit_Space_Scaler(time_law)
        r = relative_quaternion(self.oi, self.of)
        r_a, r_v = about_axis_from_quaternion(r)  # r_a === theta_final
        self.w = np.append(r_a * r_v, [0])

    def get(self, T):
        DP = (self.pf - self.pi)
        u, uD, uDD = self.time_law[T]
        u = u[:, np.newaxis]
        uD = uD[:, np.newaxis]
        uDD = uDD[:, np.newaxis]

        # POSE
        p = self.pi + u * DP
        # o = np.array([quaternion_multiply(quaternion_power(self.r, ui), self.oi) for ui in u])
        o = np.array([quaternion_slerp(self.oi, self.of, ui) for ui in u])
        pose = np.concatenate([p, o], axis=1)

        # VELOCITY
        # https://gamedev.stackexchange.com/questions/108920/applying-angular-velocity-to-quaternion
        # https://fgiesen.wordpress.com/2012/08/24/quaternion-differentiation/
        pD = uD * DP
        w = self.w * uD  # angular velocity over time as a quaternion vector
        oD = batch_quaternion_multiply(w, o) / 2
        poseD = np.concatenate([pD, oD], axis=1)

        # ACCELERATION
        pDD = uDD * DP
        wD = self.w * uDD  # angular acceleration over time as a quaternion vector
        oDD = (batch_quaternion_multiply(wD, o) + batch_quaternion_multiply(w, oD)) / 2
        poseDD = np.concatenate([pDD, oDD], axis=1)

        return pose, poseD, poseDD


# TODO could be re-written using quaternions
class Frenet(object):
    def __init__(self, ori_start, ori_end):

        Ri = quaternion_matrix(numpy_from_quaternion(ori_start))[0:3, 0:3]  # from Eb to Ei w.r.t. Eb
        Rf = quaternion_matrix(numpy_from_quaternion(ori_end))[0:3, 0:3]  # from Eb to Ef w.r.t. Eb

        Rif = Ri.T * Rf  # from Ei to Ef w.r.t. Ei
        Rif /= np.linalg.norm(Rif, axis=0)

        # angle from Ei to Ef w.r.t. Ei
        uf = np.arccos((Rif.diagonal().sum() - 1) / 2)
        ax = np.array([Rif[2, 1] - Rif[1, 2],
                       Rif[0, 2] - Rif[2, 0],
                       Rif[1, 0] - Rif[0, 1]]).T / (2 * np.sin(uf))  # axis between Ei and Ef w.r.t. Ei
        self.Ri = Ri
        self.ax = ax

    def get(self, theta):
        return self.Ri[:, :, np.newaxis] * rotations(theta, self.ax)


if __name__ == '__main__':

    from utils.filters import numpy_pose, numpy_poses
    from trajectories.profiles import Polynomial_Linear_Profile

    # linear time law with DT = 4 seconds
    tl = Polynomial_Linear_Profile(4.0, 0, 1)
    Pi = numpy_pose(np.array([0, 0,   0,     0, 0,     0, 1]))
    Pf = numpy_pose(np.array([0, 0, 0.5, 0.707, 0, 0.707, 0]))
    traj = Linear_Trajectory(tl, Pi, Pf)

    # extract trajectory (20 samples)
    t, (np_poses, _, _) = traj(20)
    poses = numpy_poses(np_poses)
