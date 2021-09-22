import actionlib
import rospy
import numpy as np
import quaternion
import control_msgs.msg
import trajectory_msgs.msg
import wsg_50_common.srv
from profiles import Polynomial_Linear_Profile
from operational import Linear_Trajectory, Arc_Trajectory

#
# useful video: https://www.youtube.com/watch?v=K09E-_2M-vQ
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

class PoseHelper(object):

    def __init__(self):
        super(PoseHelper, self).__init__()
    
    @staticmethod
    def offset_position(pose, offset, axis='z'):
        if axis == 'z':
            axis = 2
        elif axis == 'y':
            axis = 1
        elif axis == 'x':
            axis = 0
        else:
            assert False, "Invalid axis provided (['x', 'y', 'z'] expected, %s provided)" % (axis)
        pose['p'] += offset * quaternion.as_rotation_matrix(pose['o'][()])[:, axis]
        return pose

class Mover(object):

    def __init__(self, robot_arm):
        super(Mover, self).__init__()
        self.ra = robot_arm
        
        # TODO shouldn't be hardcoded
        self.arm_controller = actionlib.SimpleActionClient('/robot/arm/pos_traj_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
        started = self.arm_controller.wait_for_server(timeout=rospy.Duration(5.0))
        if not started:
            raise RuntimeError("Cannot contact position trajectory controller Action Server")
        
        # TODO shouldn't be hardcoded
        self.gripper_max_width = 110.0  # mm
        self.gripper_controller = rospy.ServiceProxy("/robot/wsg_50/move", wsg_50_common.srv.Move)
        try:
            self.gripper_controller.wait_for_service(timeout=3.0)
        except:
            raise RuntimeError("Cannot contact gripper controller Service")

        # utils
        self.helper = PoseHelper()
        self.home = None
        self.gripper_length = None

    def set_home_cfg(self, cfg):
        self.home = cfg

    def set_gripper_length(self, length):
        self.gripper_length = length


    ###########
    #   ARM   #
    ###########

    def _execute_trajectory(self, t, q, qD, qDD):
        goal = control_msgs.msg.FollowJointTrajectoryGoal(
            trajectory=trajectory_msgs.msg.JointTrajectory(
                joint_names=self.ra.joint_names,
                points=[
                    trajectory_msgs.msg.JointTrajectoryPoint(
                        positions=q[i, :].tolist(),
                        velocities=qD[i, :].tolist(),
                        accelerations=qDD[i, :].tolist(),
                        time_from_start=rospy.Duration(secs=t[i])
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
            # goal_time_tolerance=rospy.Duration(secs=0.05)
        )
        self.arm_controller.send_goal_and_wait(goal)
        # self.client.send_goal(goal)
        # self.client.wait_for_result()

        res = self.arm_controller.get_result()
        self.arm_controller.cancel_all_goals()
        return res.error_code

    def o_move_straigth_to(self, pose, time, samples=100):
        Pi = self.ra.get_current_pose()
        Pf = pose
        if np.allclose(Pi['p'], Pf['p'], atol=1e-3) and np.allclose(Pi['o'], Pf['o'], atol=1e-3):
            rospy.sleep(time)
            return 0

        tl = Polynomial_Linear_Profile(time, 0, 1)
        traj = Linear_Trajectory(tl, Pi, Pf)

        js_init = self.ra._from_JntArray(self.ra.joint_pos)
        t, (x, xD, xDD) = traj(samples)
        q, qD, qDD = self.ra.get_joint_dynamics(x, xD, xDD, q_guess=js_init)
        
        return self._execute_trajectory(t, q, qD, qDD)
    
    def j_move_straigth_to(self, js, time, samples=100):
        pose = self.ra.FK(js)
        return self.o_move_straigth_to(pose, time, samples)
    
    def j_move_to(self, js, time, samples=100):
        js_init = self.ra._from_JntArray(self.ra.joint_pos)
        if np.allclose(js_init, js, atol=1e-3):
            rospy.sleep(time)
            return 0

        q = np.zeros(shape=[samples, self.ra.n])
        qD = np.zeros(shape=[samples, self.ra.n])
        qDD = np.zeros(shape=[samples, self.ra.n])
        for i in range(self.ra.n):
            profile = Polynomial_Linear_Profile(time, js_init[i], js[i])
            t, (q[:, i], qD[:, i], qDD[:, i]) = profile.sample(samples)
        
        return self._execute_trajectory(t, q, qD, qDD)


    def go_to(self, position, time, samples=100):
        if isinstance(position, list):
            position = np.array(position)
        pose = self.ra.get_current_pose()
        pose['p'] = position
        return self.o_move_straigth_to(pose, time, samples)

    def go_home(self, time):
        if self.home is None:
            assert False, "Can't perform homing: no home configuration defined!"
        self.j_move_to(self.home, time)  # move to home


    def move_global(self, displacement_vec, time, samples=100):
        if isinstance(displacement_vec, list):
            displacement_vec = np.array(displacement_vec)
        pose = self.ra.get_current_pose()
        position = pose['p'] + displacement_vec
        return self.go_to(position, time, samples)

    def move_x(self, displacement, time, samples=100):
        return self.move_global(np.array([displacement, 0, 0]), time, samples)
    
    def move_y(self, displacement, time, samples=100):
        return self.move_global(np.array([0, displacement, 0]), time, samples)
    
    def move_z(self, displacement, time, samples=100):
        return self.move_global(np.array([0, 0, displacement]), time, samples)


    def move_relative(self, displacement_vec, time, samples=100):
        if isinstance(displacement_vec, list):
            displacement_vec = np.array(displacement_vec)
        pose = self.ra.get_current_pose()
        R_base_ee = quaternion.as_rotation_matrix(pose['o'])
        disp = np.dot(R_base_ee, displacement_vec)
        return self.move_global(disp, time, samples)

    def move_approach(self, displacement, time, samples=100):
        return self.move_relative(np.array([0, 0, displacement]), time, samples)
    
    def move_horizontal(self, displacement, time, samples=100):
        return self.move_relative(np.array([0, displacement, 0]), time, samples)

    def move_vertical(self, displacement, time, samples=100):
        return self.move_relative(np.array([displacement, 0, 0]), time, samples)


    def rotate_quaternion(self, quat, time, samples=100):
        pose = self.ra.get_current_pose()
        pose['o'] = quat * pose['o']
        return self.o_move_straigth_to(pose, time, samples)

    def rotate_x(self, angle, time, samples=100):
        r = quaternion.from_rotation_vector(angle * np.array([1, 0, 0]))
        return self.rotate_quaternion(r, time, samples)

    def rotate_y(self, angle, time, samples=100):
        r = quaternion.from_rotation_vector(angle * np.array([0, 1, 0]))
        return self.rotate_quaternion(r, time, samples)
    
    def rotate_z(self, angle, time, samples=100):
        r = quaternion.from_rotation_vector(angle * np.array([0, 0, 1]))
        return self.rotate_quaternion(r, time, samples)

    def rotate_approach(self, angle, time, samples=100):
        pose = self.ra.get_current_pose()
        z_axis = quaternion.as_rotation_matrix(pose['o'])[:, 2]
        r = quaternion.from_rotation_vector(angle * z_axis)
        return self.rotate_quaternion(r, time, samples)

    def rotate_horizontal(self, angle, time, samples=100):
        pose = self.ra.get_current_pose()
        y_axis = quaternion.as_rotation_matrix(pose['o'])[:, 1]
        r = quaternion.from_rotation_vector(angle * y_axis)
        return self.rotate_quaternion(r, time, samples)
    
    def rotate_vertical(self, angle, time, samples=100):
        pose = self.ra.get_current_pose()
        x_axis = quaternion.as_rotation_matrix(pose['o'])[:, 0]
        r = quaternion.from_rotation_vector(angle * x_axis)
        return self.rotate_quaternion(r, time, samples)

    #
    # fancy movements
    #

    def o_focus(self, Pi, distance, angle, time, samples=100):
        if np.allclose(angle, 0, atol=1e-6) or np.allclose(distance, 0, atol=1e-6):
            rospy.sleep(time)
            return 0

        tl = Polynomial_Linear_Profile(time, 0, 1)
        traj = Arc_Trajectory(tl, Pi, angle=angle, radius=distance)

        js_init = self.ra._from_JntArray(self.ra.joint_pos)
        t, (x, xD, xDD) = traj(samples)
        q, qD, qDD = self.ra.get_joint_dynamics(x, xD, xDD, q_guess=js_init)
        
        return self._execute_trajectory(t, q, qD, qDD)

    def focus_roll(self, distance, angle, time, samples=100):
        Pi = self.ra.get_current_pose()
        return self.o_focus(Pi, distance, angle, time, 100)
    
    # TODO broken rotation
    def focus_pitch(self, distance, angle, time, samples=100):
        Pi = self.ra.get_current_pose()
        Pi['o'] = \
            quaternion.from_rotation_matrix(
                np.roll(
                    quaternion.as_rotation_matrix(Pi['o'][()]), 
                    -1, axis=1
                    )
                )
        return self.o_focus(Pi, distance, angle, time, 100)
    
    # TODO broken rotation
    def focus_yaw(self, distance, angle, time, samples=100):
        Pi = self.ra.get_current_pose()
        Pi['o'] = \
            quaternion.from_rotation_matrix(
                np.roll(
                    quaternion.as_rotation_matrix(Pi['o'][()]), 
                    1, axis=1
                    )
                )
        return self.o_focus(Pi, distance, angle, time, 100)


    def scan_horizontal(self, step_time, scan_steps, scan_width=None, scan_step_size=None, callback=None, callback_args=[], sleep_time=0):
        if scan_width is None:
            scan_width = scan_step_size * scan_steps
        elif scan_step_size is None:
            scan_step_size = scan_width / scan_steps
        else:
            assert False, "Either 'scan_width' or 'scan_step_size' must be provided"

        if callback is None:
            callback = lambda *args: None
        action = lambda *args: callback(*args)

        self.move_y(+scan_width/2, step_time)
        rospy.sleep(sleep_time)
        action(*callback_args)
        rospy.sleep(sleep_time)
        for _ in range(scan_steps):
            self.move_y(-scan_step_size, step_time)
            rospy.sleep(sleep_time)
            action(*callback_args)
            rospy.sleep(sleep_time)
        self.move_y(+scan_width/2, step_time)
        rospy.sleep(sleep_time)


    ###############
    #   GRIPPER   #
    ###############
    
    def _execute_gripper(self, width):
        res = self.gripper_controller(width=width, speed=0.0)  # speed is ignored here
        return res.error
    
    def gripper_close(self):
        return self._execute_gripper(0.0)

    def gripper_open(self):
        return self._execute_gripper(self.gripper_max_width)
    
    def grasp_approach(self, time, distance=None, samples=100):
        if distance is None:
            if self.gripper_length is None:
                assert False, "Can't approach goal: no gripper length defined!"
            distance = self.gripper_length
        self.gripper_open()
        self.move_approach(distance, time, samples)
        self.gripper_close()
    
    def release_approach(self, time, distance=None, samples=100):
        if distance is None:
            if self.gripper_length is None:
                assert False, "Can't approach goal: no gripper length defined!"
            distance = self.gripper_length
        self.grasp_approach(-distance, time, samples)