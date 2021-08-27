import control_msgs.msg
import geometry_msgs.msg
import numpy as np


def filter_joint_states(msg, allowed_joint_names):
    if isinstance(msg, control_msgs.msg.JointTrajectoryControllerState):
        n = len(msg.joint_names)
        names = msg.joint_names
        pos = msg.positions if msg.positions else [0] * n
        vel = msg.velocities if msg.velocities else [0] * n
        acc = msg.accelerations if msg.accelerations else [0] * n
        eff = msg.efforts if msg.efforts else [0] * n
    else:
        n = len(msg.name)
        names = msg.name
        pos = msg.position if msg.position else [0] * n
        vel = msg.velocity if msg.velocity else [0] * n
        acc = [0] * n
        eff = msg.effort if msg.effort else [0] * n
    return [(n, p, v, a, e) for n, p, v, a, e in zip(names, pos, vel, acc, eff) if n in allowed_joint_names]


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
    pose = geometry_msgs.msg.Pose()
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
