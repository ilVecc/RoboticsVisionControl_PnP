import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

class JointStateFilter(object):

    def __init__(self, robot_prefix, joint_names, callback):
        self.js_sub = rospy.Subscriber('%s/joint_states' % (robot_prefix), JointState, self._js_cb, queue_size=1)
        self.joint_names = joint_names
        self.callback = callback

    def _js_cb(self, data):
        n = len(data.name)
        names = data.name
        pos = data.position if data.position else [0] * n
        vel = data.velocity if data.velocity else [0] * n
        eff = data.effort if data.effort else [0] * n
        data = [j for j in zip(names, pos, vel, eff) if j[0] in self.joint_names]
        data = sorted(data, key=lambda j: self.joint_names.index(j[0]))
        names, pos, vel, eff = zip(*data)
        data = JointState(
            name=names,
            position=pos,
            velocity=vel,
            effort=eff
        )
        return self.callback(data)

    @staticmethod
    def filter_joint_states(msg, allowed_joint_names):
        if isinstance(msg, JointTrajectoryControllerState):
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
