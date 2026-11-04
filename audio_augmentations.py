# audio_augmentations.py
# Not wired into training yet (Phase 2). Background noise must come from real recordings,
# e.g. <speech_commands_root>/_background_noise_; torch_audiomentations fails with background_paths=None.

from torch_audiomentations import Compose, AddBackgroundNoise, Gain, PolarityInversion, Shift


def get_waveform_augmentations(background_noise_dir, sample_rate: int = 16000):
    augment = Compose(
        transforms=[
            PolarityInversion(p=0.5),
            Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5),  # ±100 ms; ±0.5 s could shift a 1 s keyword out of the clip
            AddBackgroundNoise(
                background_paths=background_noise_dir,
                min_snr_in_db=0,
                max_snr_in_db=20,
                p=0.5
            ),
        ],
        p=1.0,
        sample_rate=sample_rate,
    )
    return augment
