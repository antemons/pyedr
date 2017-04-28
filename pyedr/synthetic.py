
import numpy as np
from numpy.random import normal
from collections import namedtuple, ChainMap

pi2 = 2*np.pi

__all__ = ['SyntheticECG']

def get_respiratory_phase(num_samples, sampling_rate, frequency=15.0/60.0, stdev=1.0/60.0):
    """ 
    Returns:
        array[num_samples]: the phase (as func of time)
    """
    w  = pi2 * frequency
    dw = pi2 * stdev
    dt = 1/np.float64(sampling_rate)
    sqdt = np.sqrt(dt)
    t = dt * np.arange(num_samples)
    phi_init = pi2 * np.random.rand() 
    phase = phi_init + t*w + dw*sqdt*np.random.randn(num_samples).cumsum()
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
    """
    defaults = {
        'settling_time': 5.0,
        'sampling_rate': 250,
        'heart_noise_strength': 0.05,
        'heart_fluctuation_strength': 1,  # is relative
        'respiration_noise_strength': 0.05,
        'heart_rate_fluctuations': 0.1,
        'heart_rate': 60.0/60.0,
        'respiration_rate': 15.0/60.0,
        'respiration_rate_stdev': 1/60,
        'respiration_fluctuation': 1/60,
        'esk_strength': 0.1,
        'rsa_strength': 0.5,
        'rsa_width_shift': 0.0,
        'rsa_dispersion': 0.1,
        'num_samples': 1024,
        'seed': None
    }

    Signal = namedtuple("SyntheticEKG", ["input", "target"])

    WaveParameter = namedtuple(
        "Parameter", ["a", "b", "theta", "esk_factor", 
                      "a_sigma", "b_sigma", "theta_sigma"])
    WAVE_PARAMETERS = {
        "P": WaveParameter(a= .20, b=.25, theta=-np.pi/3,  esk_factor= .0,
                           a_sigma=0.01, b_sigma=0.02, theta_sigma=0),
        "Q": WaveParameter(a=-.20, b=.1,  theta=-np.pi/12, esk_factor= .0,
                           a_sigma=0.01, b_sigma=0.02, theta_sigma=0),
        "R": WaveParameter(a=3.00, b=.1,  theta=0,         esk_factor= .2,
                           a_sigma=.10, b_sigma=0.02, theta_sigma=0),
        "S": WaveParameter(a=-.35, b=.1,  theta=np.pi/12,  esk_factor=-.1,
                           a_sigma=0.01, b_sigma=0.02, theta_sigma=0),
        "T": WaveParameter(a= .55, b=.3,  theta=np.pi/2,   esk_factor= .0,
                           a_sigma=0.01, b_sigma=0.02, theta_sigma=0)
    }

    def __init__(self, **kwargs):
        self.__dict__.update(**ChainMap(kwargs, self.defaults))
        np.random.seed(self.seed)
        self.omega_heart_mean = pi2 * self.heart_rate

    def phase_deriv(self, theta, resp_state):
        """Derivative of the heartbeat phase
        Args:
            theta: heartbeat phase
            resp_state: state of the respiratory cycle (-1, 1).
                Negative values decelerate, and positive values
                accelerate the heart beat.
        General form:
            tht' = w + Q(tht, R)
            where R is the respiratory oscillation.
        Coupling function Q
            Q(tht, R) = strength R(t) / (1+exp((cos(tht)+shift)/width))
        """
        Q = self.rsa_strength/(1+np.exp((np.cos(theta)+self.rsa_width_shift)/self.rsa_dispersion)) * resp_state
        return self.omega_heart_mean + Q

    def EKG_from_phase(self, phase, RESP=None):
        """Computes EKG from a heartbeat phase timeseries
        Args:
            phase: numpy.ndarray, heartbeat phase.
            RESP: numpy.ndarray, respiratory oscillation in (-1, 1).

        RESP modulates the amplitude of each EKG wave with an absolute
        strength self.esk_strength, and a wave-specific contribution esk_factor.
        """
        if RESP is None:
            RESP = np.zeros_like(phase, dtype=np.float64)
        assert phase.size == RESP.size
        EKG = np.zeros_like(phase, dtype=np.float64)
        for peak_idx in range(int(min(phase) / pi2) - 10, int(max(phase) / pi2) + 10):
            for a_i, b_i, tht_i, esk_fac, a_sigma, b_sigma, theta_sigma in self.WAVE_PARAMETERS.values():
                a   = normal(a_i,   self.heart_fluctuation_strength * a_sigma)
                b   = normal(b_i,   self.heart_fluctuation_strength * b_sigma)
                tht = normal(tht_i, self.heart_fluctuation_strength * theta_sigma)
                dtht = phase - tht - peak_idx * pi2
                EKG += (1+self.esk_strength*esk_fac*RESP) * a * np.exp(-dtht**2 / (2*b**2))
        return EKG

    def show_single_trajectory(self, show=False):
        import matplotlib.pyplot as plt
        trajectory = self.heartbeat_trajectory()
        heart_phase  = trajectory[:, 0]
        EKG  = trajectory[:, 1]
        RESP = trajectory[:, 2]
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(211)
        plt.plot(EKG)
        plt.subplot(212, sharex=ax)
        plt.plot(RESP)
        if show: plt.show()

    def get_resp_phase(self, num_samples):
        return get_respiratory_phase(num_samples, self.sampling_rate, self.respiration_rate, self.respiration_rate_stdev)

    def heartbeat_trajectory(self):
        dt    = 1./np.float64(self.sampling_rate)
        deriv = self.phase_deriv
        num_pre = int(self.settling_time*self.sampling_rate)
        num_int = num_pre + self.num_samples
        initial_state = 2.0*np.pi*np.random.rand()
        resp_phase = self.get_resp_phase(num_int)
        resp_states = np.cos(resp_phase)
        heart_phase = np.zeros((num_int), np.float64)
        heart_phase[0] = np.array(initial_state)
        for n in range(1, num_int):
            heart_phase[n] = heart_phase[n-1] + dt * deriv(heart_phase[n-1], resp_states[n-1])
        EKG  = self.EKG_from_phase(heart_phase, resp_states)
        trajectory = np.transpose(np.vstack((heart_phase, EKG, resp_states)))
        return trajectory[num_pre:]

    def __call__(self):
        heartbeat_trajectory = self.heartbeat_trajectory()
        EKG  = heartbeat_trajectory[:, 1]
        RESP = heartbeat_trajectory[:, 2]
        EKG  += normal(0.0, self.heart_noise_strength,       size=EKG.size)
        RESP += normal(0.0, self.respiration_noise_strength, size=RESP.size)
        return self.Signal(input=EKG, target=RESP)


if __name__ == "__main__":
    N = 20 * 250
    gen = SyntheticECGGenerator(sampling_rate=250, num_samples=N, rsa_strength=1)
    gen.show_single_trajectory(show=True)
