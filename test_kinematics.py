import numpy as np
import quaternion
from kinematics.chain_solver import RobotArm
from utils.quaternions import pose

if __name__ == '__main__':

    # example of use
    q = np.random.rand(6) * 2*np.pi - np.pi
    qD = np.random.rand(6) * 2*np.pi - np.pi
    xd = pose(np.array([0.8, 0.2, 0.1]), np.quaternion(-1, 0, 0, 0))

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
