import rospy
import PyKDL as KDL
import math as m
import numpy as np
from sensor_msgs.msg import JointState
from tf.transformations import euler_matrix, quaternion_multiply
from kdl_parser_py import urdf as kdl_parser
from utils.quaternions import quaternion_divide
from utils.filters import filter_joint_states

JOINT_EFFORT  = [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
JOINT_POS_MIN = [-m.pi, -m.pi, -m.pi, -m.pi, -m.pi, -m.pi]  # * 2
JOINT_POS_MAX = [ m.pi,  m.pi,  m.pi,  m.pi,  m.pi,  m.pi]  # * 2
JOINT_VEL_MIN = [-m.pi, -m.pi, -m.pi, -m.pi, -m.pi, -m.pi]
JOINT_VEL_MAX = [ m.pi,  m.pi,  m.pi,  m.pi,  m.pi,  m.pi]

JOINT_NAMES = [
    'robot_arm_shoulder_pan_joint',
    'robot_arm_shoulder_lift_joint',
    'robot_arm_elbow_joint',
    'robot_arm_wrist_1_joint',
    'robot_arm_wrist_2_joint',
    'robot_arm_wrist_3_joint'
]

class RobotArm(object):

    def __init__(self, robot_prefix, base_name, ee_name):
        tree_ok, tree = kdl_parser.treeFromParam('%s/robot_description' % (robot_prefix))

        if not tree_ok:
            err = "Failed to construct kdl tree"
            rospy.logerr(err)
            raise RuntimeError(err)
        
        # root_seg = tree.getRootSegment()  # KDL.SegmentMap.const_iterator
        self.chain = tree.getChain(base_name, ee_name)

        self.n = self.chain.getNrOfJoints()
        self.joint_names = [str(self.chain.getSegment(i).getJoint().getName()) for i in range(self.n)]
        self.joint_names = JOINT_NAMES
        self.joint_pos = KDL.JntArray(self.n)
        self.joint_vel = KDL.JntArray(self.n)
        self.joint_acc = KDL.JntArray(self.n)
        self.joint_eff = KDL.JntArray(self.n)
        
        self.fk_pos_solver = KDL.ChainFkSolverPos_recursive(self.chain)
        self.fk_vel_solver = KDL.ChainFkSolverVel_recursive(self.chain)
        self.ik_pos_solver = KDL.ChainIkSolverPos_LMA(self.chain)
        self.ik_vel_solver = KDL.ChainIkSolverVel_pinv(self.chain)
        self.jac_solver = KDL.ChainJntToJacSolver(self.chain)
        self.jac_dot_solver = KDL.ChainJntToJacDotSolver(self.chain)

        # current pose
        self.js_sub = rospy.Subscriber('%s/joint_states' % (robot_prefix), JointState, self._js_cb, queue_size=1)

    def _js_cb(self, data):
        # TODO filter joints
        data = sorted(filter_joint_states(data, JOINT_NAMES), key=lambda e: JOINT_NAMES.index(e[0]))
        for i, (_, p, v, a, e) in enumerate(data):
            self.joint_pos[i] = p
            self.joint_vel[i] = v
            self.joint_acc[i] = a
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
    def _to_Frame(array):
        return KDL.Frame(
            KDL.Rotation.Quaternion(array[3], array[4], array[5], array[6]), 
            KDL.Vector(array[0], array[1], array[2])
        )

    @staticmethod
    def _from_Frame(frame):
        pos = np.array(RobotArm._from_JntArray(frame.p))
        ori = np.array(frame.M.GetQuaternion(*tuple()))
        return np.append(pos, ori)

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
        q_arr = self._to_JntArray(q_guess) if q_guess is not None else KDL.JntArray(self.n)
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

    def get_joint_dynamics(self, x, xD, xDD, q_guess=None):
        
        # position
        q = self.IK(x, q_guess)
        Jpinv = np.linalg.pinv(self.J(q))

        # velocity
        v = np.zeros(7)
        v[0:3] = xD[0:3]
        v[3:7] = quaternion_divide(2*xD[3:7], x[3:7])
        qD = np.matmul(Jpinv, v[0:6])
        
        # acceleration
        vD = np.zeros(7)
        vD[0:3] = xDD[0:3]
        vD[3:7] = quaternion_divide(2*xDD[3:7] - quaternion_multiply(v[3:7], xD[3:7]), x[3:7])
        qDD = np.matmul(Jpinv, vD[0:6] - self.JDqD(q, qD))

        return q, qD, qDD

    def batch_get_joint_dynamics(self, X, XD, XDD, q_guess=None):
        assert X.shape[0] == XD.shape[0] and XD.shape[0] == XDD.shape[0], "Sequences must be of same length"
        N = X.shape[0]
        Q = np.zeros(shape=[N + 1, self.n])
        if q_guess is not None:
            Q[0, :] = q_guess
        QD = np.zeros(shape=[N, self.n])
        QDD = np.zeros(shape=[N, self.n])
        for i in range(N):
            Q[i+1, :], QD[i, :], QDD[i, :] = self.get_joint_dynamics(X[i, :], XD[i, :], XDD[i, :], q_guess=Q[i, :])
        return Q[1:, :], QD, QDD

if __name__ == '__main__':

    # example of use
    import numpy as np
    q = np.random.rand(6) * 2*m.pi - m.pi
    qD = np.random.rand(6) * 2*m.pi - m.pi
    # TODO change to quaternion
    xd = [0.8, 0.2, 0.1, 0, 0, 0, -1]

    ra = RobotArm("/robot", "robot_arm_base", "robot_arm_tool0")

    pose = ra.FK(q)
    print("result frame fk_p \n%s" % (pose))

    target_joints = ra.IK(xd)
    print("result joints ik_p \n%s" % (target_joints))

    jacobian = ra.J(q)
    print("result jacobian \n%s" % (jacobian))

    jacobian_dot = ra.JD(q, qD)
    print("result jacobian dot \n%s" % (jacobian_dot))

    twist = ra.JDqD(q, qD)
    print("result twist \n%s" % (twist))
