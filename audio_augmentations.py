from torch_audiomentations import Compose, AddBackgroundNoise, Gain, PolarityInversion, Shift, ApplyImpulseResponse
import torch

def get_waveform_augmentations(sample_rate: int = 16000):
    augment = Compose(
        transforms=[
            PolarityInversion(p=0.5),
            Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5),  # time shift
            AddBackgroundNoise(
                background_paths=None,  # or path to your background noise dir
                min_snr_in_db=5,
                max_snr_in_db=20,
                p=0.5
            ),
            # ApplyImpulseResponse(impulse_response_paths=..., p=0.3), # Optional
        ],
        p=0.5,
        sample_rate=sample_rate,
        # device="cuda" will automatically use GPU if available
    )
    return augment
