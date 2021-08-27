
# Python 2/3 compatibility imports
from __future__ import print_function

import sys
import time
import actionlib
import rospy
import genpy
import numpy as np
import control_msgs.msg
import trajectory_msgs.msg
from profiles import Polynomial_Linear_Profile
from operational import Linear_Trajectory

#
# usefull video: https://www.youtube.com/watch?v=K09E-_2M-vQ
#
# about the ros_control/ros_controllers packages: 
#   http://wiki.ros.org/ros_control?distro=kinetic
#   http://wiki.ros.org/ros_controllers?distro=kinetic
#
# the arm can be found under /robot/arm, in particular:
#  - /robot/arm/pos_traj_controller  is a `ros_control` position_controllers/JointTrajectoryController,
#    - /command                  performs the trajectory send-and-forget, without tolerances
#    - /follow_joint_trajectory  is the action interface, with monitoring features and tolerances
#    - /state                    gives the current joints state
#

class Mover(object):

    def __init__(self, robot_arm):
        super(Mover, self).__init__()
        self.ra = robot_arm
        
        self.client = actionlib.SimpleActionClient('/robot/arm/pos_traj_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
        started = self.client.wait_for_server(timeout=rospy.Duration(5.0))
        if not started:
            raise RuntimeError("Cannot contact position trajectory controller Action Server")

    def execute_trajectory(self, t, q, qD, qDD):
        goal = control_msgs.msg.FollowJointTrajectoryGoal(
            trajectory=trajectory_msgs.msg.JointTrajectory(
                joint_names=self.ra.joint_names,
                points=[
                    trajectory_msgs.msg.JointTrajectoryPoint(
                        positions=q[i, :].tolist(),
                        velocities=qD[i, :].tolist(),
                        accelerations=qDD[i, :].tolist(),
                        time_from_start=genpy.Duration(secs=t[i])
                    )
                    for i in range(q.shape[0])
                ]
            ),
            # path_tolerance=[
            #     control_msgs.msg.JointTolerance(
            #         name=joint_name,
            #         position=0,
            #         velocity=0,
            #         acceleration=0
            #     )
            #     for joint_name in joint_names
            # ],
            # goal_tolerance=[
            #     control_msgs.msg.JointTolerance(
            #         name=joint_name,
            #         position=0,
            #         velocity=0,
            #         acceleration=0
            #     )
            #     for joint_name in joint_names
            # ],
            # goal_time_tolerance=genpy.Duration(secs=0.05)
        )
        self.client.send_goal_and_wait(goal)
        # self.client.send_goal(goal)
        # self.client.wait_for_result()

        res = self.client.get_result()
        self.client.cancel_all_goals()
        return res.error_code

    def move_approach(self, displacement, time, samples=100):
        pose = self.ra.get_current_pose()
        pose[0] += displacement
        return self.o_move_straigth_to(pose, time, samples)
    
    def move_horizontal(self, displacement, time, samples=100):
        pose = self.ra.get_current_pose()
        pose[1] += displacement
        return self.o_move_straigth_to(pose, time, samples)

    def move_vertical(self, displacement, time, samples=100):
        pose = self.ra.get_current_pose()
        pose[2] += displacement
        return self.o_move_straigth_to(pose, time, samples)

    def o_move_straigth_to(self, pose, time, samples=100):
        Pi = self.ra.get_current_pose()
        Pf = pose
        if np.allclose(Pi, Pf, atol=1e-3):
            return 0

        tl = Polynomial_Linear_Profile(time, 0, 1)
        traj = Linear_Trajectory(tl, Pi, Pf)

        js_init = self.ra._from_JntArray(self.ra.joint_pos)
        t, (x, xD, xDD) = traj(samples)
        q, qD, qDD = self.ra.batch_get_joint_dynamics(x, xD, xDD, q_guess=js_init)
        
        return self.execute_trajectory(t, q, qD, qDD)
    
    def j_move_straigth_to(self, js, time, samples=100):
        pose = self.ra.FK(js)
        return self.o_move_straigth_to(pose, time, samples)
    
    def j_move_to(self, js, time, samples=100):
        js_init = self.ra._from_JntArray(self.ra.joint_pos)
        if np.allclose(js_init, js, atol=1e-3):
            return 0

        q = np.zeros(shape=[samples, self.ra.n])
        qD = np.zeros(shape=[samples, self.ra.n])
        qDD = np.zeros(shape=[samples, self.ra.n])
        for i in range(self.ra.n):
            profile = Polynomial_Linear_Profile(time, js_init[i], js[i])
            t, (q[:, i], qD[:, i], qDD[:, i]) = profile.sample(samples)
        
        return self.execute_trajectory(t, q, qD, qDD)
