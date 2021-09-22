import numpy as np
import rospy
import PyKDL as KDL
from kdl_parser_py import urdf as kdl_parser
from status import JointStateFilter
from utils.quaternions import quaternion_convert, quaternion_convert_from, pose

JOINT_EFFORT = [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
JOINT_POS_MIN = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]  # * 2
JOINT_POS_MAX = [ np.pi,  np.pi,  np.pi,  np.pi,  np.pi,  np.pi]  # * 2
JOINT_VEL_MIN = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
JOINT_VEL_MAX = [ np.pi,  np.pi,  np.pi,  np.pi,  np.pi,  np.pi]

# http://wiki.ros.org/tf2_kdl
class RobotArm(object):

    def __init__(self, robot_prefix, base_name, ee_name):
        tree_ok, tree = kdl_parser.treeFromParam('%s/robot_description' % (robot_prefix))

        if not tree_ok:
            err = "Failed to construct kdl tree"
            rospy.logerr(err)
            raise RuntimeError(err)

        base_name = base_name[1:] if base_name.startswith('/') else base_name
        ee_name = ee_name[1:] if ee_name.startswith('/') else ee_name
        # root_seg = tree.getRootSegment()  # KDL.SegmentMap.const_iterator
        self.chain = tree.getChain(base_name, ee_name)

        self.n = self.chain.getNrOfJoints()
        self.joint_names = [str(self.chain.getSegment(i).getJoint().getName())
                            for i in range(self.chain.getNrOfSegments()) 
                            if self.chain.getSegment(i).getJoint().getType() != KDL.Joint.Fixed]
        self.joint_pos = KDL.JntArray(self.n)
        self.joint_vel = KDL.JntArray(self.n)
        self.joint_eff = KDL.JntArray(self.n)

        self.fk_pos_solver = KDL.ChainFkSolverPos_recursive(self.chain)
        # self.fk_vel_solver = KDL.ChainFkSolverVel_recursive(self.chain)
        self.ik_pos_solver = KDL.ChainIkSolverPos_LMA(self.chain)
        # self.ik_vel_solver = KDL.ChainIkSolverVel_pinv(self.chain)
        self.jac_solver = KDL.ChainJntToJacSolver(self.chain)
        self.jac_dot_solver = KDL.ChainJntToJacDotSolver(self.chain)

        # current pose
        self.js_filter = JointStateFilter(robot_prefix, self.joint_names, self._js_cb)

    def _js_cb(self, data):
        data = zip(data.name, data.position, data.velocity, data.effort)
        for i, (_, p, v, e) in enumerate(data):
            self.joint_pos[i] = p
            self.joint_vel[i] = v
            self.joint_eff[i] = e

    @staticmethod
    def _to_JntArray(array):
        if isinstance(array, KDL.JntArray):
            return array
        # convert
        n = len(array)
        jnt_array = KDL.JntArray(n)
        for i in range(n):
            jnt_array[i] = array[i]
        return jnt_array

    @staticmethod
    def _from_JntArray(jnt_array):
        return np.array([el for el in jnt_array])

    @staticmethod
    def _to_Frame(pose):
        return KDL.Frame(
            KDL.Rotation.Quaternion(*quaternion_convert(pose['o']).tolist()),
            KDL.Vector(*pose['p'].tolist())
        )

    @staticmethod
    def _from_Frame(frame):
        pos = np.array(RobotArm._from_JntArray(frame.p))
        ori = quaternion_convert_from(list(frame.M.GetQuaternion(*tuple())))
        return pose(pos, ori)

    @staticmethod
    def _from_Twist(twist):
        vel = RobotArm._from_JntArray(twist.vel)
        rot = RobotArm._from_JntArray(twist.rot)
        return np.append(vel, rot)

    @staticmethod
    def _from_Jacobian(jac):
        jac_numpy = np.zeros(shape=[jac.rows(), jac.columns()])
        for i in range(jac.rows()):
            for j in range(jac.columns()):
                jac_numpy[i, j] = jac[i, j]
        return jac_numpy

    def FK(self, q):
        q_arr = self._to_JntArray(q)
        target_frame = KDL.Frame()
        error = self.fk_pos_solver.JntToCart(q_arr, target_frame)
        return self._from_Frame(target_frame) if not error else None

    def IK(self, xd, q_guess=None):
        q_arr = self._to_JntArray(
            q_guess) if q_guess is not None else KDL.JntArray(self.n)
        x_frame = self._to_Frame(xd)
        target_joints = KDL.JntArray(self.n)
        error = self.ik_pos_solver.CartToJnt(q_arr, x_frame, target_joints)
        return self._from_JntArray(target_joints) if not error or error == -100 or error == -101 else None

    def J(self, q):
        q_arr = self._to_JntArray(q)
        jac = KDL.Jacobian(self.chain.getNrOfJoints())
        self.jac_solver.JntToJac(q_arr, jac)
        return self._from_Jacobian(jac)

    def JD(self, q, qD):
        qD_arr = KDL.JntArrayVel(self._to_JntArray(q), self._to_JntArray(qD))
        jac_dot = KDL.Jacobian(self.n)
        self.jac_dot_solver.JntToJacDot(qD_arr, jac_dot)
        return self._from_Jacobian(jac_dot)

    def JDqD(self, q, qD):
        qD_arr = KDL.JntArrayVel(self._to_JntArray(q), self._to_JntArray(qD))
        twist = KDL.Twist()
        self.jac_dot_solver.JntToJacDot(qD_arr, twist)
        return self._from_Twist(twist)

    def get_current_pose(self):
        return self.FK(self.joint_pos)

    def _get_joint_dynamics(self, x, xD, xDD, q_guess=None):

        # position
        q = self.IK(x, q_guess)
        if q is None:
            return None
        Jpinv = np.linalg.pinv(self.J(q))

        # velocity
        v = pose(
            xD['p'],
            2*xD['o'] / x['o']
        )
        v_vec = np.concatenate([v['p'], v['o'][()].vec])
        qD = np.dot(Jpinv, v_vec)

        # acceleration
        vD = pose(
            xDD['p'],
            (2*xDD['o'] - v['o'] * xD['o']) / x['o']
        )
        vD_vec = np.concatenate([vD['p'], vD['o'][()].vec])
        qDD = np.dot(Jpinv, vD_vec - self.JDqD(q, qD))

        return q, qD, qDD

    def get_joint_dynamics(self, X, XD, XDD, q_guess=None):
        assert X.shape[0] == XD.shape[0] and XD.shape[0] == XDD.shape[0], "Sequences must be of same length"
        N = X.shape[0]
        Q = np.zeros(shape=[N + 1, self.n])
        QD = np.zeros(shape=[N, self.n])
        QDD = np.zeros(shape=[N, self.n])
        # set initial conditions
        if q_guess is not None:
            Q[0, :] = q_guess
        for i in range(N):
            res = self._get_joint_dynamics(X[i], XD[i], XDD[i], q_guess=Q[i, :])
            if res is not None:
                Q[i+1, :], QD[i, :], QDD[i, :] = res[0], res[1], res[2]
            else:
                if i > 0:
                    Q[i+1, :], QD[i, :], QDD[i, :] = Q[i, :], QD[i-1, :], QDD[i-1, :]
                else:
                    raise RuntimeError("Unable to begin the inverse kinematics trajectory")
        return Q[1:, :], QD, QDD
