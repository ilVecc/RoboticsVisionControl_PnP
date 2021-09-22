import numpy as np
import quaternion

class Pose(np.dtype):

    def __init__(self, p, o):
        self.p = p
        self.o = o
    
    def __add__(self, pose):
        return (pose.p + self.p, pose.o * self.o)
    
    def __sub__(self, pose):
        return Pose(self.p - pose.p, pose.o / self.o)
    
    def __repr__(self):
        return "(%s,%s)" % (self.p, quaternion.as_float_array(self.o))
    
    def __str__(self):
        return self.__repr__()
    
if __name__ == '__main__':
    Pi = Pose(np.array([0, 0, 0]), np.quaternion(1,     0, 0,     0))
    Pf = Pose(np.array([2, 2, 2]), np.quaternion(0, 0.707, 0, 0.707))
    print(Pf - Pi)

    p = np.array([Pi, Pf])
    print(p)
