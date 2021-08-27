#!/usr/bin/env python

# Python 2/3 compatibility imports
from __future__ import print_function

import numpy as np
import rospy
from six.moves import input
from kinematics.orocos_kdl import RobotArm, JOINT_NAMES
from trajectories.mover import Mover

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


PREFIX = '/robot'
MOVE_GROUP = 'arm'
BASE_NAME = 'robot_arm_base'
EE_NAME = 'robot_arm_tool0'

READY_CFG = np.array([0.0, -pi/2, -pi/2, -pi, -pi/2, pi/2])
REST_CFG = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

ERROR_CODES = {
    '0': 'SUCCESSFUL',
    '-1': 'INVALID_GOAL',
    '-2': 'INVALID_JOINTS',
    '-3': 'OLD_HEADER_TIMESTAMP',
    '-4': 'PATH_TOLERANCE_VIOLATED',
    '-5': 'GOAL_TOLERANCE_VIOLATED'
}

def LOG(title, error_code):
    print("%s: %s" % (title, ERROR_CODES[str(error_code)]))


def main():
    try:
        print("Init ROS node")
        rospy.init_node("rvc_controller", anonymous=False)

        # prepare forward and inverse kinematics solver ...
        ra = RobotArm(PREFIX, BASE_NAME, EE_NAME)
        # ... and invoke high-level commander
        mv = Mover(ra)

        # prepare camera recorder
        


        res = mv.j_move_to(READY_CFG, 3.0)
        LOG('Move to ready', res)

        res = mv.move_vertical(-0.400, 3.0)
        LOG('Down', res)

        # horizontal scan
        res = mv.move_horizontal(+0.400, 3.0)
        LOG('Left', res)
        res = mv.move_horizontal(-0.800, 3.0)
        LOG('Right', res)

        # res = mv.move_approach(+0.200, 3.0)
        # LOG('Approach', res)


    except rospy.ROSInterruptException as ex:
        return print("ROS died! ", ex)
    except KeyboardInterrupt as ex:
        return print("Manually interrupted!")


if __name__ == '__main__':
    main()
