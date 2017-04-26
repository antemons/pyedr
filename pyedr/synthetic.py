
import numpy as np
from collections import namedtuple, ChainMap

pi2 = 2*np.pi

__all__ = ['SyntheticECG']

def get_respiratory_phase(num_samples, sampling_rate, frequency=15.0/60.0, stdev=1.0/60.0):
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
        esk_spot_width (None or float): if None then the esk acts uniform if float
            than only in a spot centered at the R-peak with the given width
            (see also show_single_trajectory)
        rsa_spot_width (None or float): if None then the esk acts uniform if float
            than it acts everywhere except in the spot centered at the R-peak 
            with the given width (see also show_single_trajectory)
    """
    defaults = {
        'settling_time': 5.0,
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
        'rsa_width_shift': 0.0,
        'rsa_dispersion': 0.1,
        'num_samples': 1024,
        'seed': 42
    }

    Signal = namedtuple("SyntheticEKG", ["input", "target"])

    WaveParameter = namedtuple("Parameter", "a b theta esk_factor")
    WAVE_PARAMETERS = {
        "P": WaveParameter(a= 2.0, b=.25, theta=-np.pi/3,  esk_factor= .0),
        "Q": WaveParameter(a=-2.0, b=.1,  theta=-np.pi/12, esk_factor= .0),
        "R": WaveParameter(a=30.0, b=.1,  theta=0,         esk_factor= .2),
        "S": WaveParameter(a=-3.5, b=.1,  theta=np.pi/12,  esk_factor=-.1),
        "T": WaveParameter(a= 5.5, b=.3,  theta=np.pi/2,   esk_factor= .0)
    }

    def __init__(self, *args, **kwargs):
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
        return self.omega_heart_mean + self.rsa_strength/(1+np.exp((np.cos(theta)+self.rsa_width_shift)/self.rsa_dispersion)) * resp_state

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
        else:
            assert phase.size == RESP.size
        EKG = np.zeros_like(phase, dtype=np.float64)
        for a_i, b_i, theta_i, esk_fac in self.WAVE_PARAMETERS.values():
            dtht = phase-theta_i
            EKG += (1+self.esk_strength*esk_fac*RESP)*a_i * np.exp((np.cos(dtht)-1) / (2*b_i**2))
        return EKG

    def show_single_trajectory(self, show=False):
        import matplotlib.pyplot as plt
        trajectory = self._heartbeat_trajectory()
        heart_phase  = trajectory[:, 0]
        EKG  = trajectory[:, 1]
        RESP = trajectory[:, 2]
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(211)
        plt.plot(EKG)
        plt.subplot(212, sharex=ax)
        plt.plot(RESP)
        if show: plt.show()

    @staticmethod
    def _noise(signal, stdev):
        signal += np.random.normal(0, stdev, len(signal))

    def get_resp_phase(self, num_samples):
        return get_respiratory_phase(num_samples, self.sampling_rate, self.respiration_rate, self.respiration_rate_stdev)

    def _heartbeat(self):
        return self._heartbeat_trajectory()[:, 1]

    def _heartbeat_trajectory(self):
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
        heartbeat_trajectory = self._heartbeat_trajectory()
        EKG  = heartbeat_trajectory[:, 1]
        RESP = heartbeat_trajectory[:, 2]
        self._noise(EKG, self._heart_noise_stength)
        self._noise(RESP, self._respiration_noise_stength)
        return self.Signal(input=EKG, target=RESP)


if __name__ == "__main__":
    gen = SyntheticECGGenerator(sampling_rate=250, num_samples=N, rsa_strength=1)
    gen.show_single_trajectory(show=True)
