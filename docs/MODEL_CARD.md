# Model card: TinyAudioTransformer (keyword spotting, 35 words)

## Summary
A 0.85M-parameter audio transformer that recognises one of 35 English command words in a 1-second clip. It was trained from scratch on Speech Commands v2. The exported model takes a raw 16 kHz waveform and returns 35 logits, with the log-mel feature extraction built into the graph. It runs on-device in the browser demo (`web/`).

| | |
|---|---|
| File | `web/model/tat.onnx` (4.7 MB, ONNX opset 20) |
| Checkpoint | `results/phase2b/runs_phase2b/tat_shift_specaug_noise_s0/best.pt` (epoch 28) |
| Input | `waveform`: float32 `[batch, 16000]`, mono 16 kHz, values in [−1, 1] |
| Outputs | `logits` `[batch, 35]` and `fbank` `[batch, 128, 128]` (the log-mel features, for display) |
| Words | backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero |

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture_dark.png">
  <img src="assets/architecture_light.png" alt="Architecture diagram. Front-end: waveform [B,16000] → framing [B,98,400] → DC removal, pre-emphasis, Hann window → FFT power [B,98,257] → 128-filter mel filterbank [B,98,128] → log and pad [B,128,128]. Tokens: normalise → [B,1,128,128] → Conv2d 16×16 stride 10 patch embedding [B,128,12,12] → 144 tokens [B,144,128] → add [CLS] and positions [B,145,128]. Encoder: 4 pre-norm blocks of LayerNorm, 4-head self-attention, residual, LayerNorm, MLP 128→512→128, residual. Head: final LayerNorm, take [CLS] [B,128], linear → logits [B,35], softmax." width="100%">
</picture>

*Shapes and parameter counts were measured by running the real modules, not typed in by hand.*

- **Features:** Kaldi log-mel filterbank with 25 ms Hann windows every 10 ms, 128 mel bins, pre-emphasis 0.97. That gives 98 frames per second, zero-padded to 128, normalised as (x − mean)/(2·std) with training-set statistics.
- **Model:** AST-style. The spectrogram is cut into overlapping 16×16 patches (stride 10, 144 patches). Then [CLS] + learned positions, 4 pre-norm transformer blocks (dim 128, 4 heads, MLP ×4), LayerNorm, and a linear head.
- **Verified against the reference:** with the AST authors' fine-tuned weights loaded, the same modules reproduce Hugging Face AST's logits to within 1e-3 (`tests/test_parity_ast.py`). The ONNX graph reproduces the PyTorch training pipeline to within 3e-5 in logits, with identical predictions on 8,000 real clips (`export_onnx.py`, `tests/test_export.py`).

## Training
- **Data:** Speech Commands v2, official split (84,843 train / 9,981 val / 11,005 test). The split is speaker-disjoint: no test speaker was heard in training.
- **Recipe:** AdamW (lr 1e-3, weight decay 0.05), 2-epoch warmup + cosine, 30 epochs, batch 256, mixed precision. The checkpoint was selected on val.
- **Augmentation:** random time shift ±100 ms, SpecAugment (one ≤24-bin frequency mask and one ≤24-frame time mask), and real background noise from the dataset's `_background_noise_` recordings (first 80% of each file) at 0–20 dB SNR on 80% of clips.
- **Why this recipe:** a 27-run ablation (Phase 2) and an 18-run follow-up (Phase 2b), 3 seeds each, paired by seed. Shift and SpecAugment helped on every seed. Mixup and label smoothing made the model underconfident and didn't help once shift and SpecAugment were in. Noise cost 2.4 points on clean audio but bought +24 points at 0 dB. Full reasoning is in `docs/LEARNING_LOG.md`, Sessions 4 and 6.
- **Seed choice:** seed 0 of 3, chosen because it had the best **val** accuracy at every noise level. Test numbers were not used to choose.

## Results (official test set, 11,005 clips, 250 unseen speakers)

| Condition | This model (seed 0) | Same recipe, 3 seeds |
|---|---|---|
| Clean | **93.35%** | 92.35 ± 0.87 |
| + held-out noise, 20 dB SNR | 92.62% | 91.75 ± 0.82 |
| 10 dB | 90.60% | 89.87 ± 0.69 |
| 5 dB | 88.12% | 87.29 ± 0.87 |
| 0 dB (noise as loud as speech) | 83.06% | 82.00 ± 1.07 |

- **Calibration:** ECE 0.80%, NLL 0.22. Mean confidence is close to accuracy, so the confidence shown in the demo is meaningful.
- **Weakest words:** forward 79.4%, follow 82.0%, tree 85.0%, go 88.3%, bed 88.9%.
- **Most frequent confusions:** tree→three (22), three→tree (20), go→no (20), forward→four (17), four→forward (14), two→go (14).
- **Context:** a no-augmentation baseline of the same model gets 93.5% clean but 53.7% at 0 dB. A 0.25M-parameter CNN on the same features gets 92.4% clean. AST (87M parameters, ImageNet-pretrained) reports 98.1%.

### Real-microphone test (demo "Test it" mode)

| Date | Speakers | Browser / mic | Result | 95% CI |
|---|---|---|---|---|
| 2026-11-04 | 1 (the author), quiet room | Safari 18.6, MacBook built-in mic | **34 / 35 (97.1%)**, right word in top 5: 35 / 35 | 85.1–99.9% (exact binomial) |

The one error was tree → three (75% confident), the test set's most frequent confusion. The other low-confidence answers were also on the known confusable pairs: no vs go (63%), dog vs no (71%), three vs tree (72%), go vs no (72%), four vs go (77%). Raw results are in `results/phase3/real_user/`. One speaker in one session is evidence that the browser pipeline works, not an accuracy estimate for other people.

## Intended use
- A learning project and a demonstration of keyword spotting: say one of the 35 words, see what the model hears.
- Collecting real-microphone results with the demo's **Test it** mode, as an evaluation the training data can't provide.

## Not suitable for
- Anything safety-relevant or access-controlling (voice commands for devices, accessibility tools people depend on, authentication).
- Words outside the 35, continuous speech, or open-vocabulary recognition.
- Always-on detection. **There is no "unknown word" or "silence" class**, so any sound is mapped to one of the 35 words. The demo only partly works around this, with a loudness gate and a "low confidence" message.

## Known limitations and risks
- **Accent and speaker coverage:** Speech Commands was crowd-sourced from volunteers, mostly English speakers with unrecorded demographics. Accuracy for accents, ages or voices that are under-represented is **unknown and likely lower**. Per-group results can't be computed because the dataset has no such metadata.
- **Noise coverage:** the "held-out" noise is unseen *segments* of the same 6 recordings used in training (dishes, cat, exercise bike, pink and white noise, running tap). Unfamiliar noise types, reverberant rooms, other people talking and music were not tested.
- **Microphones:** dataset clips came from many consumer devices, at low quality, via the browser. Studio mics, phone noise suppression or Bluetooth headsets may differ. The demo switches off the browser's echo cancellation, noise suppression and auto-gain to stay close to training conditions.
- **Clip framing:** the model expects a word roughly centred in 1 s. The demo records 2 s and centres a 1 s window on the loudest part. Two words, long words, or a cough next to the word can confuse that step.
- **Confusable pairs** (tree/three, go/no, four/forward) are the main source of errors, even in clean conditions.

## Privacy
The demo runs fully in the browser with onnxruntime-web: audio is never uploaded. In **Test it** mode, only the predictions are kept (target, prediction, confidence, input level, time), in the browser's local storage. They're downloadable as JSON, and nothing is sent anywhere.

## How to reproduce
Training outputs (`results/…`) and `mini_speech_commands/` aren't in the git repo, since they're too large. The notebook below recreates them.

1. `kaggle/tiny_audio_transformer_phase2b.ipynb` on Kaggle (2×T4) trains all Phase 2b runs and writes the checkpoint.
2. Export and check: `python export_onnx.py <best.pt> web/model --real-audio mini_speech_commands`.
3. Serve the demo: `cd web && python -m http.server`, then open http://localhost:8000.
