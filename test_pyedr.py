
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


def test_ekg():
    expected_result = 4853
    seed = 42
    num_seg = 2
    fs = 250
    ds = pyedr.Dataset(subject_ids=['synthetic'], num_of_segments=num_seg,
            hr_stdev_factor=0.0, EKG_fluctuation_strength=0.0, esk_strength=0.0, rsa_strength=0.0,
            seed=seed, sampling_rate=fs)
    data = ds.get_data(normalize=None)
    ecg = [d[0] for d in data]
    ekg = Ekg(ecg, sampling_rate=fs)
    ekg.get_all_R_peaks()
    assert ekg.R_peaks[0][-1] == expected_result
    assert len(ekg.R_peaks) == 2
    assert len(ekg.R_peaks[0]) == 20


tests = [test_synthetic, test_ekg]
for test in tests:
    try:
        test()
        print("{} : successful".format(test.__name__))
    except:
        print("{} : failed".format(test.__name__))

