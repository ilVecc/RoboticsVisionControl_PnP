# RVC UR5 PnP control
"Robotic, Vision and Control" course final project, focused on the Pick&amp;Place task on a UR5 robot using a RealSense camera.

In order to obtain the requested result, we provide an interface for the construction of the various time laws studied during the course and another one for the end effector trajectory planning. Leveraging these two tools, we can request specific motions in the operational space and traslate them in the joint space using simple inverse kinematics formulae.

The inverse kinematics layer is provided by the Orocos KDL library, which is essential when dealing with the Analytical Jacobian and its time derivative.
