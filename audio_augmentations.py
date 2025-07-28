from torch_audiomentations import Compose, Gain, PolarityInversion, Shift, AddBackgroundNoise
background_path="/content/background_noise"

def get_waveform_augmentations(sample_rate: int = 16000, noise_dir=background_path):
    augment = Compose(
        transforms=[
            PolarityInversion(p=0.3),  # changed p from 0.5 to 0.3
            Gain(min_gain_in_db=-3.0, max_gain_in_db=3.0, p=0.5),  # narrowed gain range
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5),  # ±10% shift instead of ±50%
            AddBackgroundNoise(background_paths=noise_dir, p=0.3, sample_rate=sample_rate),  # p=0.3
        ]
    )
    return augment
