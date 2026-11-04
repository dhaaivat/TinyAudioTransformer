# TinyAudioTransformer vs. reference implementations

References, read from source rather than paper prose:

- **AST code**: `YuanGongND/ast`, `src/dataloader.py` and `egs/speechcommands/run_sc.sh`
- **AST port**: HF `transformers` `modeling_audio_spectrogram_transformer.py` and `feature_extraction_audio_spectrogram_transformer.py`
- **Checkpoint**: `MIT/ast-finetuned-speech-commands-v2`, including `config.json` and `preprocessor_config.json`
- **KWT**: `ARM-software/keyword-transformer` README

"Verified by" names the test that would fail if we diverged.

## Data and features

| | Original code | Reference | Now | Verified by |
|---|---|---|---|---|
| Dataset | `mini_speech_commands`, 8 words, no official split | Speech Commands v2, 35 words | SC v2, 35 words | `test_splits` |
| Split | label-sorted list sliced 80/10/10, so **test contained only the last label(s)**; same speakers in train and test | `validation_list.txt` / `testing_list.txt` (speaker-disjoint) | official lists: 84,843 / 9,981 / 11,005 | `test_speaker_disjoint`, `test_every_split_has_every_class` |
| Features | `MelSpectrogram(n_fft=1024, hop=256)` → dB, 128 mels | Kaldi fbank: 25 ms window, 10 ms shift, Hanning, 128 bins, pre-emphasis 0.97, per-frame DC removal | own Kaldi fbank in plain torch (`features.py`) | `test_matches_torchaudio_kaldi`, `test_matches_hf_ast_feature_extractor` |
| Paper vs code | paper text says "Hamming window" | **code uses `hanning`** | follow the code (the released weights were trained with it) | HF extractor parity |
| Length | `[:, :128, :64]` crop; a 1 s clip gives **63** frames | 98 frames per 1 s, zero-padded to 128 | zero-pad/crop to 128 | `test_fixed_output_shape` |
| Normalisation | none (dB values ≈ −100…+40) | `(x − mean) / (2·std)`, mean −6.846, std 5.565 | same formula, train-split stats, stored as model buffers | parity tests |

## Model

| | Original code | AST | Now (tiny default) | Verified by |
|---|---|---|---|---|
| Input layout | `[B, 1, 128, 64]` | `[B, T, F]` → `[B, 1, F, T]` | same as AST | parity |
| Patches | 16×16, stride 16, count hard-coded from `input_shape` → **crash** with real input (24 patches vs 32 expected) | 16×16, stride 10 (overlap 6) → 12×12 = 144 | stride configurable (default 10), count derived from the conv formula; wrong input shape raises a clear error | `test_patch_count_matches_conv_output`, `test_rejects_wrong_input_shape` |
| Prefix tokens | `[CLS]` | `[CLS]` + `[DIST]` (DeiT), pooled as their mean | configurable: 1 for tiny, 2 for parity | parity |
| Token/pos init | `randn` (std **1.0**) | trunc-normal std 0.02 (ViT/DeiT); AST then loads ImageNet weights | trunc-normal 0.02, linears too | `test_init_scale` |
| Block | pre-norm; `norm1` computed 3× (correct but wasteful); no dropout on the attention branch | pre-norm ViT | `norm1` once; dropout on both residual branches | parity |
| MLP ratio | 2 | 4 (3072/768) | 4, configurable (KWT-1 uses 2) | parity |
| LayerNorm eps | 1e-5 | 1e-12 (HF config) | 1e-6; 1e-12 in parity mode | parity |
| Head | LN → Linear on `[CLS]` | final LN → mean(`[CLS]`, `[DIST]`) → LN → Linear | `head_norm` flag | parity |
| Size | 128 dim / 4 deep / 4 heads | 768 / 12 / 12, ~87M params | 128 / 4 / 4, ~0.85M params | — |

**Parity result:** with the authors' fine-tuned weights loaded into these modules, logits match HF AST to within 1e-3. This holds on random inputs and end to end from waveform (our fbank + our model vs. their extractor + their model). So the architecture and features compute the same function as the reference. Checked with transformers 4.57.6 and 5.16.1 (5.x renamed the AST modules; `ast_reference.py` normalises both spellings and fails if any reference tensor goes unused).

## Training (expected to differ; listed so the differences are deliberate)

| | Original code | AST SC recipe | Now (Phase 1) |
|---|---|---|---|
| Init | scratch | **ImageNet-pretrained DeiT** | scratch |
| Loss | CE | BCE | CE |
| Optimiser | Adam 1e-4, constant | Adam 2.5e-4, batch 128, 30 epochs, ×0.85/epoch from epoch 5 | AdamW 1e-3, wd 0.05 (not on norms/tokens), 2-epoch warmup + cosine, batch 256, 30 epochs, AMP, grad clip 1.0 |
| Augmentation | defined but unused; `AddBackgroundNoise(background_paths=None)` errors | SpecAugment (freq 48, time 48), mixup 0.6 on waveforms, fbank noise + ±10-frame roll | Phase 2 ablation (`augment.py`): ±100 ms shift, waveform mixup 0.6 / Beta(10,10), real background noise at 0–20 dB SNR, SpecAugment 24/24 with time masks kept off the padding, label smoothing 0.1 |
| Model selection | best val; test never evaluated; `get_dataloaders` didn't exist | best val | best val, then test evaluated **once**; per-class accuracy and confusion matrix saved |
| Seeds | none | — | seeded |

## Published targets (35-word SC v2)

- **AST: 98.11 ± 0.05%.** That's 87M params with ImageNet pretraining, so it's an upper reference, not a target for a 0.85M from-scratch model.
- **KWT** (from scratch): the README reports **12-label** accuracy (KWT-1 97.7, KWT-2 98.2, KWT-3 98.7). Its 35-label numbers need to be read from the paper's table before Phase 2 sets a target.
- **Phase 1 bar:** the transformer should beat the CNN baseline on the same features. Gaps to the published numbers are expected until augmentation (Phase 2).
