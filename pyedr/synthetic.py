
import numpy as np
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
from fastcache import clru_cache
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple, ChainMap

@clru_cache(maxsize=256)
def sr2sqdt(sampling_rate):
    return np.sqrt(1.0/np.float64(sampling_rate))

__all__ = ['SyntheticECG']

def random_walk(N, sampling_rate):
    """ generate random walk
    Args:
        time (array): times (must be equidistant a la np.linspace)
    Returns:
        array of len like time
    """
    return np.sr2sqdt(sampling_rate) * np.random.randn(N).cumsum()


def get_respiratory_phase(num_samples, sampling_rate, frequency=15.0/60.0, stdev=1.0/60.0):
    w  = 2.0 * np.pi * frequency
    dw = 2.0 * np.pi * stdev
    dt = 1.0/np.float64(sampling_rate)
    sqdt = np.sqrt(dt)
    time = dt * np.arange(num_samples)
    phi_init = 2.0 * np.pi * np.random.rand() 
    phase = phi_init + time*w + dw*sqdt*np.random.randn(num_samples).cumsum()
    return phase


class SyntheticECGGenerator:
    """ Generate synthetic ECG Signals

    >>> get_signal = synthetic.SyntheticECG()
    >>> signal = get_signal()
    >>> time, input_, target = signal

    Paper: P. McSharry el. al.,
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 50, NO. 3, MARCH 2003

    Args:
        max_time: length of sequence in sec (default 10)
        sampling_rate: samples per second (default 250)
        heart_rate: heartrate in beats per second  (default 60/60)
        respiration_rate: ensemble average of the respiration rate
            in beats per second (default 15/60)
        respiration_rate_stdev: the standard deviation of the
            respiration rate (default 1/60) for both the ensemble fluctuations
            and the insample fluctuations
        esk_strength: ESK coupling stength, the ecg signal varies with
            1 + esk_strength * respiration   (default 0.1)
        rsa_strength: strength of the respiratory sinus arrhythmia (default 0.5)
        heart_noise_stength: noise added to ecg signal (default 0.05)
        heart_rate_fluctuations: relative heat rate fluctuations (default 0.05)
        respiration_noise_stength: noise added to respiration signal
            (default 0.05)
        esk_spot_width (None or float): if None then the esk acts uniform if float
            than only in a spot centered at the R-peak with the given width
            (see also show_single_trajectory)
        rsa_spot_width (None or float): if None then the esk acts uniform if float
            than it acts everywhere except in the spot centered at the R-peak 
            with the given width (see also show_single_trajectory)
    """
    defaults = {
        'sampling_rate': 250,
        'heart_noise_strength': 0.05,
        'respiration_noise_strength': 0.05,
        'esk_spot_width': 0.15,
        'rsa_spot_width': 0.25,
        'heart_rate_fluctuations': 0.1,
        'heart_rate': 60.0/60.0,
        'respiration_rate': 15.0/60.0,
        'respiration_rate_stdev': 1/60,
        'respiration_fluctuation': 1/60,
        'esk_strength': 0.1,
        'rsa_strength': 0.5,
        'num_samples': 1024,
        'seed': 42
    }

    Signal = namedtuple("Signal", ["time", "input", "target"])
    State = namedtuple("EKGState", "theta z")

    WaveParameter = namedtuple("Parameter", "a b theta")
    WAVE_PARAMETERS = {
        "P": WaveParameter(a=1.2,  b=.25, theta=-np.pi/3),
        "Q": WaveParameter(a=-5.0, b=.1,  theta=-np.pi/12),
        "R": WaveParameter(a=30.0, b=.1,  theta=0),
        "S": WaveParameter(a=-7.5, b=.1,  theta=np.pi/12),
        "T": WaveParameter(a=.75,  b=.4,  theta=np.pi/2)
    }

    A = 0.0
    ESK_SPOT_WIDTH_QRS = 2 * np.pi / 12
    RSA_SPOT_WIDTH_QRS = 2 * np.pi / 3

    def __init__(self, *args, **kwargs):
        self.__dict__.update(**ChainMap(kwargs, self.defaults))

        np.random.seed(self.seed)
        self.omega_heart_mean = 2.0 * np.pi * self.heart_rate


    def _random_process(self, coefficients, time):
        """TODO(dv): ideally this should be a Ornstein-Uhlenbeck process, 
                  how is its spectral decomposition?
        """
        return sum(c * np.sin((k + 1) * np.pi * time / self.max_time / 2) / (k + 1) 
                   for k, c in enumerate(coefficients)) / np.pi

    def derivs(self, state, resp_state):
        theta, EKG = state
        tht_dot = self.omega_heart_mean + self.rsa_strength * (1.0 + np.cos(theta)) * resp_state
        z_dot = -EKG + self.A * resp_state
        for a_i, b_i, theta_i in self.WAVE_PARAMETERS.values():
            delta_theta = (theta - theta_i + np.pi) % (2 * np.pi) - np.pi
            z_dot += -a_i * delta_theta * np.exp(-delta_theta**2 / (2 * b_i**2))

        return np.array([tht_dot, z_dot])

    def show_single_trajectory(self, show=False):
        trajectory = self._heartbeat_trajectory(num_samples=1024)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        tht = trajectory[:, 0]
        z   = trajectory[:, 1]
        x, y = np.sin(tht), np.cos(tht)
        ax.plot(x, y, z)
        if show: plt.show()

    @staticmethod
    def _noise(signal, stdev):
        signal += np.random.normal(0, stdev, len(signal))

    def get_resp_phase(self, num_samples):
        return get_respiratory_phase(self.num_samples, self.sampling_rate, self.respiration_rate, self.respiration_rate_stdev)

    def _coupled_via_esk(self, heartbeat_trajectory, respiration):
        heartbeat = heartbeat_trajectory[:, 2]
        if self.esk_spot_width is None:
            heartbeat[:] = heartbeat * (1 + self._esk_strength * respiration)
        else:
            x = heartbeat_trajectory[:, 0]
            y = heartbeat_trajectory[:, 1]
            theta = np.arctan2(y, x)
            heartbeat[:] *= (1 + np.exp(- theta**2 / self.esk_spot_width**2) * 
                                 self._esk_strength * respiration)

    def _heartbeat(self, respiration):
        return self._heartbeat_trajectory(respiration)[:, 1]

    def _heartbeat_trajectory(self, num_samples):
        dt = 1./np.float64(self.sampling_rate)
        derivs = self.derivs
        resp_phase = self.get_resp_phase(num_samples)
        resp_states = np.cos(resp_phase)
        trajectory = np.zeros((num_samples, 2), np.float64)
        for n in range(1, num_samples):
            trajectory[n] = trajectory[n-1] + dt * derivs(trajectory[n-1], resp_states[n-1])
        trajectory[:, 1] *= 20
        #np.vstack((trajectory, resp_states[None,:]))
        return trajectory

    def __call__(self):
        heartbeat_trajectory = self._heartbeat_trajectory(self.num_samples)
        heartbeat = heartbeat_trajectory[:, 1]
        respiration = heartbeat_trajectory[:, 2]
        self._coupled_via_esk(heartbeat_trajectory, respiration)
        self._noise(heartbeat, self._heart_noise_stength)
        self._noise(respiration, self._respiration_noise_stength)
        return self.Signal(time=self.time, input=heartbeat, target=respiration)
