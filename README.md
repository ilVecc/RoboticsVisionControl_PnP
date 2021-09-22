# RVC UR5 PnP control
"Robotic, Vision and Control" course final project, focused on the Pick&amp;Place task on a UR5 robot using a RealSense camera.

In order to obtain the requested result, we provide an interface for the construction of the various time laws studied during the course and another one for the end effector trajectory planning. Leveraging these two tools, we can request specific motions in the operational space and traslate them in the joint space using simple inverse kinematics formulae.

The forward/inverse kinematics layer is provided by the Orocos KDL library, which is essential when dealing with the Jacobian and its time derivative.



## Clarifications

The RB-KAIROS (from Robotnik) is a prefab robot, a combination of a SUMMIT-XL STEEL robot mobile base (from Robotnik), a UR<> robot arm (from Universal Robots) and an end-effector tool.
The UR<> robot can be any of the family (UR3, UR3e, UR5, UR5e, UR10, UR10e).
The tool can vary, but this simulation bundle includes a VGC10 vacuum gripper (from OnRobot), a WSG-50 clamp gripper (from Schunk), a EGH clamp gripper (from Schunk), or a combination of one of these on a support and a camera.


## Initial Bugs

The ArUco cubes textures were vertically flipped (seemingly for no reason at all), and the cv2.aruco module of course couldn't find any marker.



Some info:
http://gazebosim.org/tutorials/?tut=ros_control
http://wiki.ros.org/rqt_joint_trajectory_controller
http://wiki.ros.org/tf2_kdl

