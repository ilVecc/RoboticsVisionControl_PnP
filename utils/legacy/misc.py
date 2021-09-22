import sys
import numpy as np
import sensor_msgs.msg
import geometry_msgs.msg


def binom_coeff(order):
    bin = np.zeros(order + 1)
    bin[0] = 1
    for _ in range(order):
        bin[1:] = bin[1:] + bin[:-1]
    return bin



def make_joint_state_msg(positions):
    return sensor_msgs.msg.JointState(
        name=JOINT_NAMES,
        position=positions
    )


def make_pose_stamped_msg(pose):
    ps = geometry_msgs.msg.PoseStamped()
    ps.header.frame_id = EE_NAME
    ps.pose = pose
    return ps


tmp = None


def brutal_stdout_kill():
    global tmp
    tmp = sys.stdout
    sys.stdout = None


def brutal_stdout_revive():
    global tmp
    sys.stdout = tmp


if __name__ == '__main__':
    print(binom_coeff(10))