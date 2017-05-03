
import pyedr
from pyedr.ekg import Ekg

def test_synthetic():
    fs = 250
    num_samples = 128
    num_of_segments = 1
    f_HR = 1.2
    q = 0.0
    ds = pyedr.Dataset(subject_ids=['synthetic'], sampling_rate=fs,
            num_of_segments=num_of_segments, heart_rate=f_HR, hr_stdev_factor=q,
            rsa_strength=0.0, esk_strength=0.0,
            EKG_noise_strength=0.0, EKG_fluctuation_strength=0.0,
            num_samples=num_samples)
    output = ds.get_data(normalize=None)
    assert type(output) == list
    assert len(output) == num_of_segments
    assert len(output[0]) == 2
    assert len(output[0][0]) == num_samples


test_synthetic()
