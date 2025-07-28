from torch_audiomentations import Compose, Gain, PolarityInversion, Shift

def get_waveform_augmentations(sample_rate: int = 16000):
    augment = Compose(
        transforms=[
            PolarityInversion(p=0.5),
            Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ]
    )
    return augment
