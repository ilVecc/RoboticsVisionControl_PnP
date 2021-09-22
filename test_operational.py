import numpy as np
from trajectories.profiles import Polynomial_Linear_Profile
from trajectories.operational import Linear_Trajectory, Arc_Trajectory
from utils.quaternions import pose
from utils.plotting import plot_poses


def get_line_traj():
    # linear time law with DT = 4 seconds
    tl = Polynomial_Linear_Profile(4.0, 0, 1)
    Pi = pose(np.array([0, 0, 0]), np.quaternion(1,     0, 0,     0))
    Pf = pose(np.array([2, 2, 2]), np.quaternion(0, 0.707, 0, 0.707))
    traj = Linear_Trajectory(tl, Pi, Pf)
    # extract trajectory (20 samples)
    _, (np_poses, _, _) = traj(20)
    return np_poses


def get_arc_traj():
    # linear time law with DT = 4 seconds
    tl = Polynomial_Linear_Profile(4.0, 0, 1)
    Pi = pose(np.array([0, 0, 0]), np.quaternion(0, 0.707, 0, 0.707))
    traj = Arc_Trajectory(tl, Pi, angle=4*np.pi/3, radius=5)
    # extract trajectory (100 samples)
    _, (np_poses, _, _) = traj(100)
    return np_poses


def get_composite_traj():
    # linear time law with DT = 4 seconds
    tl = Polynomial_Linear_Profile(4.0, 0, 1)
    # trajectory 1
    Pi = pose(np.array([0, 0, 0]), np.quaternion(1,     0, 0,     0))
    Pf = pose(np.array([0, 0, 10]), np.quaternion(1,     0, 0,     0))
    # Pf = pose(np.array([2, 2, 2]), np.quaternion(0, 0.707, 0, 0.707))
    traj_1 = Linear_Trajectory(tl, Pi, Pf)
    # trajectory 2
    traj_2 = Arc_Trajectory(tl, Pi, angle=4*np.pi/3, radius=5)
    # composition
    traj = traj_1 + traj_2
    # extract trajectory (100 samples)
    _, (np_poses, _, _) = traj(100)
    return np_poses


if __name__ == '__main__':

    # np_poses = get_arc_traj()
    np_poses = get_composite_traj()
    plot_poses(np_poses)
