from geometry_msgs.msg import Pose
import numpy as np


def numpy_from_position(pos):
    return np.array([pos.x, pos.y, pos.z])


def numpy_from_quaternion(quat):
    return np.array([quat.x, quat.y, quat.z, quat.w])


def numpy_from_pose(pose):
    # type: (Pose) -> np.ndarray
    p = numpy_from_position(pose.position)
    o = numpy_from_quaternion(pose.orientation)
    return np.concatenate([p, o])


def numpy_pose(np_pose):
    pose = Pose()
    pose.position.x = np_pose[0]
    pose.position.y = np_pose[1]
    pose.position.z = np_pose[2]
    pose.orientation.x = np_pose[3]
    pose.orientation.y = np_pose[4]
    pose.orientation.z = np_pose[5]
    pose.orientation.w = np_pose[6]
    return pose


def numpy_poses(np_poses):
    return [numpy_pose(np_poses[row, :]) for row in range(np_poses.shape[0])]
