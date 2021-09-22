import numpy as np
import quaternion
from utils.quaternions import pose, posedt, quaternion_average
from trajectories.profiles import Polynomial_Linear_Profile, Unit_Space_Scaler, Space_Scaler, Time_Scaler


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

    def __add__(self, other):
        return Composite_Trajectory(self.time_law.DT, *[self, other])


# TODO add support to jerk
class Linear_Trajectory(Trajectory):

    def __init__(self, time_law, pose_init, pose_end):
        super(Linear_Trajectory, self).__init__(time_law)

        self.pi, self.oi = pose_init['p'], pose_init['o']
        self.pf, self.of = pose_end['p'], pose_end['o']

        r = quaternion.as_rotation_vector(self.of / self.oi)  # r_a (angle of r, here its norm) === theta_final
        # angular velocity is  w = r_a * r_v  but r is already in that representation (rotation vector)
        self.w = np.quaternion(0, *r.tolist())
        
        self.time_law = Unit_Space_Scaler(time_law)

    def get(self, T):
        DP = (self.pf - self.pi)
        u, uD, uDD = self.time_law[T]

        # POSE
        p = self.pi + u[:, np.newaxis] * DP
        # o = np.array([quaternion_multiply(quaternion_power(self.r, ui), self.oi) for ui in u])
        o = np.slerp_vectorized(self.oi, self.of, u)

        # VELOCITY
        # https://gamedev.stackexchange.com/questions/108920/applying-angular-velocity-to-quaternion
        # https://fgiesen.wordpress.com/2012/08/24/quaternion-differentiation/
        pD = uD[:, np.newaxis] * DP
        w = self.w * uD  # angular velocity over time as a quaternion vector
        oD = w * o / 2

        # ACCELERATION
        pDD = uDD[:, np.newaxis] * DP
        wD = self.w * uDD  # angular acceleration over time as a quaternion vector
        oDD = (wD * o + w * oD) / 2

        return pose(p, o), pose(pD, oD), pose(pDD, oDD)


# TODO test end_pose
class Arc_Trajectory(Trajectory):

    def __init__(self, time_law, pose_init, pose_end=None, angle=None, radius=None):
        super(Arc_Trajectory, self).__init__(time_law)

        self.pi, self.oi = pose_init['p'], pose_init['o']
        oi_matrix = quaternion.as_rotation_matrix(self.oi)
        oi_x_axis = oi_matrix[:, 0]  # tangent direction
        oi_y_axis = oi_matrix[:, 1]  # radius direction
        oi_z_axis = oi_matrix[:, 2]  # axis direction

        if angle is None:
            if pose_end is None:
                raise RuntimeError("Either `pose_end` or (`angle`, `radius`) must be specified")
            else:
                # poses must share the same xy-plane
                pf, self.of = pose_end['p'], pose_end['o']
                pi_to_pf = pf - self.pi
                expected_rotation_axis = np.cross(oi_x_axis, pi_to_pf, axis=0)  # TODO problem when pi = pf
                expected_rotation_axis /= np.linalg.norm(expected_rotation_axis)
                if not np.allclose(expected_rotation_axis, oi_z_axis):
                    raise RuntimeError("Poses are not lying on the same xy-plane")
                
                # TODO double check this
                # find center of circumference
                of_x_axis = quaternion.as_rotation_matrix(self.of)[:, 0]
                M = np.concatenate([oi_x_axis, of_x_axis, oi_z_axis], axis=1).T
                P = np.concatenate([self.pi, pf, pf], axis=1)
                self.c = np.dot(np.linalg.inv(M), np.dot(M, P).diagonal())

                # radii must be the same
                expected_radius = np.linalg.norm(self.pi - self.c)
                if not np.allclose(expected_radius, np.linalg.norm(pf - self.c)):
                    raise RuntimeError("Poses are not on a circle")
                
                self.radius = expected_radius
                r = quaternion.as_rotation_vector(self.of / self.oi)
                angle = np.linalg.norm(r)
        else:
            if radius is None:
                raise RuntimeError("Either `pose_end` or (`angle`, `radius`) must be specified")    
            self.radius = radius
            self.c = self.pi + self.radius * oi_y_axis
            rzi = quaternion.from_rotation_vector(-np.pi/2 * oi_z_axis)  # fixed relative rotation from initial pose rotation to initial circumference rotation
            self.ri = rzi * self.oi  # circumference initial rotation
            r = quaternion.from_rotation_vector(1 * oi_z_axis)  # relative rotation from initial to final
            self.of = r * self.oi
        
        self.w = np.quaternion(0, *(angle * oi_z_axis).tolist())

        self.time_law = Space_Scaler(time_law, 0, angle)

    def _C(self, u):
        return np.array([np.cos(u), np.sin(u), np.zeros(u.shape)]).T
    
    def _T(self, u):
        return np.array([-np.sin(u), np.cos(u), np.zeros(u.shape)]).T

    def get(self, T):
        u, uD, uDD = self.time_law[T]

        # POSE
        p = self.radius * quaternion.rotate_vectors(self.ri, self._C(u)) + self.c
        o = np.slerp_vectorized(self.oi, self.of, u)

        # VELOCITY
        pD = self.radius * quaternion.rotate_vectors(self.ri, self._T(u) * uD[:, np.newaxis])
        w = self.w * uD  # angular velocity over time as a quaternion vector
        oD = w * o / 2

        # ACCELERATION
        pDD = self.radius * quaternion.rotate_vectors(self.ri, self._T(u) * uDD[:, np.newaxis] + self._C(u) * uD[:, np.newaxis] ** 2)
        wD = self.w * uDD  # angular acceleration over time as a quaternion vector
        oDD = (wD * o + w * oD) / 2

        return pose(p, o), pose(pD, oD), pose(pDD, oDD)


# FIXME quaternion_average doesn't seem to act properly
# TODO initial pose must be the same
class Composite_Trajectory(Trajectory):

    def __init__(self, DT, *trajectories):
        assert len(trajectories) >= 2, "At least two trajectories are needed"
        super(Composite_Trajectory, self).__init__(Polynomial_Linear_Profile(DT, 0, 1))
        self.trajectories = trajectories

    def get(self, T):
        # collect trajectories
        t = T.shape[0]
        n = len(self.trajectories)
        poses = np.empty(shape=(t, n), dtype=posedt)
        posesD = np.empty(shape=(t, n), dtype=posedt)
        posesDD = np.empty(shape=(t, n), dtype=posedt)
        for i, traj in enumerate(self.trajectories):
            poses[:, i], posesD[:, i], posesDD[:, i] = traj.get(T / self.time_law.DT * traj.time_law.DT)
        # sum position, slerp orientation
        pi = poses[0, 0]['p']
        p = np.sum(poses['p'] - pi, axis=1) + pi
        o = quaternion_average(poses['o'])
        pDi = posesD[0, 0]['p']
        pD = np.mean(poses['p'] - pDi, axis=1) + pDi
        oD = quaternion_average(poses['o'])
        pDDi = posesDD[0, 0]['p']
        pDD = np.mean(poses['p'] - pDDi, axis=1) + pDDi
        oDD = quaternion_average(poses['o'])
        return pose(p, o), pose(pD, oD), pose(pDD, oDD)



# TODO could be re-written using quaternions
# TODO could also be completely deleted...
class Frenet(object):
    def __init__(self, ori_start, ori_end):

        from utils.matrices import rotations
        
        Ri = quaternion.as_rotation_matrix(ori_start)  # from Eb to Ei w.r.t. Eb
        Rf = quaternion.as_rotation_matrix(ori_end)    # from Eb to Ef w.r.t. Eb

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
