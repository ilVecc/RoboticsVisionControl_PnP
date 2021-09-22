#!/usr/bin/env python

# Python 2/3 compatibility imports
from __future__ import print_function

import numpy as np
import quaternion
import rospy
from kinematics.chain_solver import RobotArm
from trajectories.mover import Mover
from vision.camera import CameraStreamer
from vision.feature_detector import ArucoSceneBuilder
from utils.quaternions import quaternion_convert

import matplotlib.pyplot as plt


PREFIX = '/robot'
MOVE_GROUP = 'arm'
BASE_FRAME = '/robot_arm_base'
EE_FRAME = '/robot_arm_tool0'
CAMERA_COLOR_NAME = '/wrist_rgbd/color'
CAMERA_COLOR_IMAGE_FRAME = '/robot_wrist_rgbd_color_optical_frame'

GRIPPER_SAFE_OFFSET   = 0.020  # m
GRIPPER_FINGER_LENGTH = 0.085  # m
GRIPPER_HAND_LENGTH   = 0.085  # m
GRIPPER_GRABBING_OFFSET = GRIPPER_HAND_LENGTH + GRIPPER_FINGER_LENGTH + GRIPPER_SAFE_OFFSET

BOX_ARUCO_FRONT = 5

READY_CFG = np.array([0.0, -np.pi/2, -np.pi/2, -np.pi, -np.pi/2, np.pi/2])
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
        ra = RobotArm(PREFIX, BASE_FRAME, EE_FRAME)
        # ... and invoke high-level commander
        mv = Mover(ra)
        # prepare image processing class
        asb = ArucoSceneBuilder(CAMERA_COLOR_NAME, BASE_FRAME, CAMERA_COLOR_IMAGE_FRAME, debug_mode=True)


        ##############
        #   HOMING   #
        ##############

        mv.set_gripper_length(GRIPPER_HAND_LENGTH)
        mv.gripper_close()
        mv.set_home_cfg(READY_CFG)
        mv.go_home(time=2.0)


        ################
        #   SCANNING   #
        ################

        # move to scan init
        mv.move_global([0, +0.100, -0.450], time=1.0)

        # horizontal scan
        mv.scan_horizontal(step_time=1.0, scan_width=1.000, scan_steps=8, callback=asb.record_aruco_poses, sleep_time=1.0)
        mv.go_home(time=1.5)

        # show the poses (will last 10 seconds by default in rviz)
        wrt_frame = BASE_FRAME
        boxes_poses = [(aruco_id, pose) for aruco_id, poses in asb.get_recorded_aruco_poses(wrt_frame).items() for pose in poses]
        import tf
        br = tf.TransformBroadcaster()
        send_tf = 10
        while send_tf > 0:
            for i, (aruco_id, pose_aruco) in enumerate(boxes_poses):
                t, r = pose_aruco['p'], quaternion_convert(pose_aruco['o'][()])
                br.sendTransform(t, r, rospy.Time.now(), "id=%d (%d)" % (aruco_id, i), wrt_frame)
            rospy.sleep(0.5)
            send_tf -= 1


        ######################
        #   PICK AND PLACE   #
        ######################
        
        # filter the poses
        boxes_poses = asb.get_recorded_aruco_poses(BASE_FRAME)[BOX_ARUCO_FRONT]

        # FIRST BOX: frontal grab, vertical rotation
        pos = mv.helper.offset_position(boxes_poses[0], GRIPPER_GRABBING_OFFSET, 'z')['p']
        mv.go_to(pos, time=1.0)  # move in front of the cube
        mv.grasp_approach(time=1.0)
        mv.move_z(0.050, time=1.0)  # lift the cube
        mv.rotate_vertical(np.pi/2, time=1.0)  # rotate it
        mv.move_z(-0.050, time=1.0)  # place the cube
        mv.release_approach(time=1.0)
        
        mv.go_home(time=1.5)

        # SECOND BOX: frontal grab, horizontal rotation
        pos = mv.helper.offset_position(boxes_poses[1], GRIPPER_GRABBING_OFFSET, 'z')['p']
        mv.go_to(pos, time=1.0)  # move in front of the cube
        mv.grasp_approach(time=1.0)
        mv.move_z(0.050, time=1.0)  # lift the cube
        mv.rotate_horizontal(np.pi/2, time=1.0)  # rotate it
        mv.move_z(-0.050, time=1.0)  # place the cube
        mv.release_approach(time=1.0)
        
        mv.go_home(time=1.5)
        
        # THIRD BOX: frontal grab, horizontal rotation
        pos = mv.helper.offset_position(boxes_poses[2], GRIPPER_GRABBING_OFFSET, 'z')['p']
        mv.go_to(pos, time=1.0)  # move in front of the cube
        mv.grasp_approach(time=1.0)
        mv.move_z(0.050, time=1.0)  # lift the cube
        mv.rotate_approach(np.pi, time=1.0)  # rotate it
        mv.move_z(-0.050, time=1.0)  # place the cube
        mv.release_approach(time=1.0)
        
        mv.go_home(time=1.5)


        # mv.focus_yaw(0.100, np.pi/2, 3.0, samples=200)
        # mv.focus_yaw(0.100, -np.pi/2, 3.0, samples=200)

    except rospy.ROSInterruptException as ex:
        return print("ROS died! ", ex)
    except KeyboardInterrupt as ex:
        return print("Manually interrupted!")
    finally:
        if asb is not None:
            asb.quit()


if __name__ == '__main__':
    main()
