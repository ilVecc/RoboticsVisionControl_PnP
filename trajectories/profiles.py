import numpy as np

class Profile(object):

    def __init__(self, DT):
        """
        Represents the evolution of a trajectory on its path in range [0,DT].
        """
        self.DT = DT

    def _normalize(self, T):
        return self._clip(T) / self.DT
    
    def _clip(self, T):
        return np.clip(T, 0, self.DT)

    def __call__(self, n_samples):
        return self.sample(n_samples)

    def sample(self, n_samples):
        t = np.linspace(0.0, self.DT, num=n_samples, endpoint=True)
        return t, self.get(t)

    def __getitem__(self, T):
        return self.get(T)
    
    def get(self, T):
        # type: (np.ndarray) -> np.ndarray
        raise NotImplementedError("you must first implement this function")


class Polynomial_Profile(Profile):

    def __init__(self, DT, *V):
        super(Polynomial_Profile, self).__init__(DT)

        if len(V) == 1:
            if isinstance(V[0], list) or isinstance(V[0], tuple):
                V = np.array(V[0])
            elif isinstance(V[0], np.ndarray):
                V = V[0]


        n = len(V)        
        assert n > 0, "Must specify at least a couple of initial/final conditions (vi, vf)"
        assert n % 2 == 0, "Initial/Final conditions (vi, vf, vDi, vDf, ...) must come in pairs"

        # dynamically add attributes 'vi', 'vf', 'vDi', ... for consistency
        for i in range(n / 2):
            setattr(self, 'v%si' % ('D' * i), V[i*2])
            setattr(self, 'v%sf' % ('D' * i), V[i*2+1])

        # obtain polynomial coefficients
        H = np.zeros([n, n])
        P = np.ones([n])
        for i in range(n / 2):
            H[i*2, n-1-i] = P[-1]
            H[i*2+1, 0:n-i] = P
            P = np.polyder(P)
        
        self.n = n
        self.a = np.linalg.solve(H, V)
    
    def get(self, T, order=3):
        # type: (np.ndarray, int) -> np.ndarray

        if order is None or order < 0:
            order = self.n

        T = self._normalize(T)

        res = []
        a_der = self.a.copy()
        for _ in range(order):
            res.append(np.polyval(a_der, T))
            a_der = np.polyder(a_der)
        return tuple(res)

class Polynomial_Linear_Profile(Polynomial_Profile):

    def __init__(self, DT, vi, vf):
        super(Polynomial_Linear_Profile, self).__init__(DT, [vi, vf])

class Polynomial_Cubic_Profile(Polynomial_Profile):

    def __init__(self, DT, vi, vf, vDi, vDf):
        super(Polynomial_Cubic_Profile, self).__init__(DT, [vi, vf, vDi, vDf])

class Polynomial_Quintic_Profile(Polynomial_Profile):

    def __init__(self, DT, vi, vf, vDi, vDf, vDDi, vDDf):
        super(Polynomial_Quintic_Profile, self).__init__(DT, [vi, vf, vDi, vDf, vDDi, vDDf])


class Trapezoidal_Generic_Profile(Profile):
    
    def __init__(self, ta, td, DT, vi, vf, vDi, vDc, vDf):
        super(Trapezoidal_Generic_Profile, self).__init__()

        assert 0 < ta, 'Acceleration time must be greater than zero'
        assert 0 < td, 'Deceleration time must be greater than zero'
        assert ta + td < DT, 'Total time must be greater than %.3f' % (ta + td)

        self.ta, self.td = ta, td
        self.DT = DT
        self.vi, self.vf = vi, vf
        self.vDi, self.vDf = vDi, vDf
        self.vDc = vDc
        self.vDDc = (vDc - vDi)/ta  # or (vcD - vDf)/td

        self.map = np.vectorize(self._get_vect)
    
    def _get_vect(self, t):
        if t <= self.ta:
            return self.vi + self.vDi*t + self.vDDc/2*t**2, \
                   self.vDi + self.vDDc*t, \
                   self.vDDc
        elif self.ta < t and t <= (self.DT-self.td):
            return self.vi + self.vDi*self.ta/2 + self.vDc*(t-self.ta/2), \
                   self.vDc, \
                   0.0
        else:
            return  self.vf + self.vDf*(t-self.DT) - self.vDDc/2*(t-self.DT)**2, \
                    self.vDf - self.vDDc*(t-self.DT), \
                   -self.vDDc

    def get(self, T):
        T = self._clip(T)
        return self.map(T)

class Trapezoidal_Simmetric_Profile(Trapezoidal_Generic_Profile):
    
    def __init__(self, tc, DT, vi, vf):

        assert 0 < tc and tc <= DT/2, 'Cut speed must be between %.3f and %.3f' % (0, DT/2)

        DV = vf - vi
        vDc = DV/(DT-tc)
        super(Trapezoidal_Simmetric_Profile, self).__init__(tc, tc, DT, vi, vf, 0, vDc, 0)

class Trapezoidal_tc_Profile(Trapezoidal_Simmetric_Profile):
    
    def __init__(self, tc, DT, vi, vf):
        super(Trapezoidal_tc_Profile, self).__init__(tc, DT, vi, vf)

class Trapezoidal_vDc_Profile(Trapezoidal_tc_Profile):
    
    def __init__(self, DT, vi, vf, vDc):

        DV = float(vf - vi)

        min_vDc = DV/DT
        max_vDc = 2*DV/DT
        assert min_vDc < vDc and vDc <= max_vDc, 'Cut speed must be between %.3f and %.3f' % (min_vDc, max_vDc)

        tc = DT - DV/vDc
        super(Trapezoidal_vDc_Profile, self).__init__(tc, DT, vi, vf)


class DoubleS_Profile(Profile):

    def __init__(self, vi, vf, vDi, vDf, vD_max, vDD_max, vDDD_max):
        # vD_max = -vD_min, vDD_max = -vDD_min, vDDD_max = -vDDD_min

        assert vD_max >= vDi and vD_max >= vDf, 'Max speed cannot be smaller than initial/final speed'

        DV = vf - vi
        DDV = vDf - vDi

        tj_star = min(np.sqrt(abs(DDV)/vDDD_max), vDD_max/vDDD_max)

        reaches_vDmax = tj_star < vDD_max/vDDD_max

        # case vD_lim = vD_max (... or is it? :) )
        if reaches_vDmax:
            assert DV > tj_star*(vDi+vDf), 'Realization constraint not satisfied: DV > tj_star*(vDi+vDf)'
            
            # assume vD_lim = vD_max
            # acceleration phase
            if (vD_max-vDi)*vDDD_max < vDD_max**2:
                # vDD_max not reached
                tj1 = np.sqrt((vD_max-vDi)/vDDD_max)
                ta = 2*tj1
            else:
                # vDD_max reached
                tj1 = vDD_max/vDDD_max
                ta = tj1+(vD_max-vDi)/vDD_max
            
            # deceleration phase
            if (vD_max-vDf)*vDDD_max < vDD_max**2:
                # vDD_min not reached
                tj2 = np.sqrt((vD_max-vDf)/vDDD_max)
                td = 2*tj2
            else:
                # vDD_min reached
                tj2 = vDD_max/vDDD_max
                td = tj2+(vD_max-vDf)/vDD_max
            
            # check feasability again
            tv = DV/vD_max - ta/2*(1+vDi/vD_max) - td/2*(1+vDf/vD_max)
            if tv < 0:
                # everything above is wrong because actually vD_lim < vD_max
                reaches_vDmax = False

        # case vD_lim < vD_max
        if not reaches_vDmax:
            tj_star = vDD_max/vDDD_max
            
            assert DV > (vDi+vDf)/2*(tj_star+abs(DDV)/vDDD_max), 'Realization constraint not satisfied: DV > (vDi+vDf)/2*(tj_star+abs(DDV)/vDDD_max)'
            
            tj1 = tj_star
            tj2 = tj_star
            D = tj_star**2*vDD_max**2+(4*DV-2*tj_star*(vDi+vDf))*vDD_max+2*(vDi**2+vDf**2)
            ta = (tj_star*vDD_max-2*vDi+np.sqrt(D))/(2*vDD_max)
            td = (tj_star*vDD_max-2*vDf+np.sqrt(D))/(2*vDD_max)
            tv = 0

        if ta < 0:
            # no acceleration phase
            ta = 0 # ?

        if td < 0:
            # no deceleration phase
            td = 0 # ?

        vDDa_lim = vDDD_max*tj1
        vDDd_lim = -vDDD_max*tj2
        vD_lim = vDi+(ta-tj1)*vDDa_lim # or = vDf-(Td-Tj2)*vDDd_lim

        DT = ta + tv + td

        self.vi = vi
        self.vf = vf
        self.vDi = vDi
        self.vDf = vDf
        self.vD_lim = vD_lim
        self.vDDa_lim = vDDa_lim
        self.vDDd_lim = vDDd_lim
        self.vDDD_max = vDDD_max
        self.tj1 = tj1
        self.ta = ta
        self.tj2 = tj2
        self.td = td
        self.DT = DT

        self.map = np.vectorize(self._get_vect)

    def _get_vect(self, t):
        # acceleration time
        if t <= self.tj1:
            return \
                self.vi + self.vDi*t + self.vDDD_max/6*t**3, \
                self.vDi + self.vDDD_max/2*t**2, \
                self.vDDD_max*t, \
                self.vDDD_max
        elif self.tj1 < t and t <= (self.ta-self.tj1):
            return \
                self.vi + self.vDi*t + self.vDDa_lim/6*(3*t**2 - 3*self.tj1*t + self.tj1**2), \
                self.vDi + self.vDDa_lim*(t - self.tj1/2), \
                self.vDDa_lim, \
                0
        elif (self.ta-self.tj1) < t and t <= self.ta:
            return \
                self.vi + (self.vD_lim+self.vDi)*self.ta/2 + self.vD_lim*(t-self.ta) - self.vDDD_max/6*(t-self.ta)**3, \
                self.vD_lim - self.vDDD_max/2*(t-self.ta)**2, \
                -self.vDDD_max*(t-self.ta), \
                -self.vDDD_max
        # constant velocity time
        elif self.ta < t and t <= (self.DT-self.td):
            return \
                self.vi + (self.vD_lim+self.vDi)*self.ta/2 + self.vD_lim*(t-self.ta), \
                self.vD_lim, \
                0, \
                0
        # deceleration time
        elif (self.DT-self.td) < t and t <= (self.DT-self.td+self.tj2):
            return \
                self.vf - (self.vD_lim+self.vDf)*self.td/2 + self.vD_lim*(t-self.DT+self.td) - self.vDDD_max/6*(t-self.DT+self.td)**3, \
                self.vD_lim - self.vDDD_max/2*(t-self.DT+self.td)**2, \
                -self.vDDD_max*(t-self.DT+self.td), \
                -self.vDDD_max
        elif (self.DT-self.td+self.tj2) < t and t <= (self.DT-self.tj2):
            return \
                self.vf - (self.vD_lim+self.vDf)*self.td/2 + self.vD_lim*(t-self.DT+self.td) + self.vDDd_lim/6*(3*(t-self.DT+self.td)**2 - 3*self.tj2*(t-self.DT+self.td) + self.tj2**2), \
                self.vD_lim + self.vDDd_lim*((t-self.DT+self.td) - self.tj2/2), \
                self.vDDd_lim, \
                0
        else:
            return \
                self.vf + self.vDf*(t-self.DT) + self.vDDD_max/6*(t-self.DT)**3, \
                self.vDf + self.vDDD_max/2*(t-self.DT)**2, \
                self.vDDD_max*(t-self.DT), \
                self.vDDD_max

    def get(self, T):
        T = self._clip(T)
        return self.map(T)

class DoubleS_DT_Profile(DoubleS_Profile):

    def __init__(self, DT, a, b, vi, vf):
        # symmetric time periods
        # ta = td = a*DT, tj1 = tj2 = b*ta
        assert 0 < a and a <= 0.5, "'a' coefficient must be between 0 and 1/2"
        assert 0 < b and b <= 0.5, "'b' coefficient must be between 0 and 1/2"
        DV = vf - vi
        vDc_max = DV/((1-a)*DT)
        vDDc_max = vDc_max/(a*(1-b)*DT)
        vDDDc_max = vDDc_max/(a*b*DT)
        super(DoubleS_DT_Profile, self).__init__(vi, vf, 0, 0, vDc_max, vDDc_max, vDDDc_max)



class Profile_Scaler(Profile):

    def __init__(self, profile):
        # type: (Profile) -> Profile_Scaler
        super(Profile_Scaler, self).__init__(profile.DT)
        self.profile = profile
    
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            return object.__getattribute__(self.profile, name)

    def get(self, T):
        raise NotImplementedError("No scaling formula defined")

    @staticmethod
    def scale(profile, new_DT, T):
        raise NotImplementedError("No scaling formula defined")


class Time_Scaler(Profile_Scaler):

    def __init__(self, profile, new_DT):
        super(Time_Scaler, self).__init__(profile)
        self.DT = new_DT
    
    def get(self, T):
        return self.scale(self.profile, self.DT, T)

    @staticmethod
    def scale(profile, new_DT, T):
        s = (new_DT / profile.DT)
        results = profile.get(T / s)
        scaled_results = [results[i] / s ** i for i in range(len(results))]
        return tuple(scaled_results)
    
class Space_Scaler(Profile_Scaler):

    def __init__(self, profile, new_vi, new_vf):
        super(Space_Scaler, self).__init__(profile)
        self.vi = new_vi
        self.vf = new_vf
    
    def get(self, T):
        return self.scale(self.profile, self.vi, self.vf, T)
    
    @staticmethod
    def scale(profile, new_vi, new_vf, T):
        results = list(profile.get(T))
        results[0] -= profile.vi  # just for position
        scaled_results = [(new_vf - new_vi) * results[i] / (profile.vf - profile.vi) for i in range(len(results))]
        scaled_results[0] += new_vi  # just for position
        return tuple(scaled_results)

class Full_Scaler(Time_Scaler, Space_Scaler):

    def __init__(self, profile, new_DT, new_vi, new_vf):
        self.profile = profile
        self.DT = new_DT
        self.vi = new_vi
        self.vf = new_vf
    
    def get(self, T):
        return self.scale(self.profile, self.DT, self.vi, self.vf, T)
    
    @staticmethod
    def scale(profile, new_DT, new_vi, new_vf, T):
        return Space_Scaler(Time_Scaler(profile, new_DT), new_vi, new_vf).get(T)

class Unit_Time_Scaler(Time_Scaler):

    def __init__(self, profile):
        super(Unit_Time_Scaler, self).__init__(profile, 1)

class Unit_Space_Scaler(Space_Scaler):

    def __init__(self, profile):
        super(Unit_Space_Scaler, self).__init__(profile, 0, 1)

class Unit_Full_Scaler(Full_Scaler):

    def __init__(self, profile):
        super(Unit_Full_Scaler, self).__init__(profile, 1, 0, 1)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    def plot():
        fig, axs = plt.subplots(4, sharex=True)
        axs[0].plot(t, v)
        axs[0].set(ylabel='position')
        axs[1].plot(t, vD)
        axs[1].set(ylabel='velocity')
        axs[2].plot(t, vDD)
        axs[2].set(ylabel='acceleration')
        axs[3].plot(t, vDDD)
        axs[3].set(ylabel='jerk')
        plt.show()

    t = np.linspace(-1, 6, num=1000)

    # example of Double-S trajectory with imposed DT
    tl = DoubleS_DT_Profile(5, 0.25, 0.25, 0, 3)
    v, vD, vDD, vDDD = tl[t]
    # as we can see, in [-1,0] and [5,6] the last value is maintained
    plot()
    
    # example of a scaling operation on the previous profile
    stl = Full_Scaler(tl, 8, 1, 4)
    v, vD, vDD, vDDD = stl[t]
    plot()
