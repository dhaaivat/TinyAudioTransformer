from torch_audiomentations import Compose, AddBackgroundNoise, Gain, PolarityInversion, Shift, ApplyImpulseResponse
import torch

def get_waveform_augmentations(sample_rate: int = 16000):
    augment = Compose(
        transforms=[
            PolarityInversion(p=0.5),
            Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5) # time shift
        ],
        p=0.5,
        sample_rate=sample_rate,
        # device="cuda" will automatically use GPU if available
    )
    return augment
