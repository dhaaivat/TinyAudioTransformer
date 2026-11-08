# Learning log: TinyAudioTransformer

The goal is to build a keyword-spotting model from scratch in PyTorch and learn audio ML properly along the way: how it works, why each piece is there, and how to know it's right. This log records every step with its **what**, **why**, **result** and **lesson**. Concepts are explained in the [glossary](#glossary) at the end.

**The plan** (each phase has to pass its checks before the next starts):

| Phase | Question it answers | Status |
|---|---|---|
| 0 Ground truth | Is the data split honest? | ✅ done |
| 1 Correctness | Does the code compute what the paper's authors' code computes? | ✅ done |
| 2 Training | Which training tricks actually help, and by how much (beyond seed noise)? | ✅ done: deployment candidate shift + specaug + noise (Session 6) |
| 3 Deployment | Does it work in a browser, on real people's microphones? | ✅ live at dhaaivat.github.io/TinyAudioTransformer (Session 9) |

---

## Session 1 (2026-10-31): audit, then rebuild for correctness

### Step 1: read the code before changing anything
**What:** read all six original files end to end and traced one clip through the pipeline by hand.
**Found:**
- **The test set held only the last word or two.** Files were listed folder by folder (`down/…, go/…, …, yes/…`) and then sliced 80/10/10, so "test" was mostly `yes`.
- **The same speakers appeared in train and test.** Even with a shuffled split, one person's voice could be in both, so the model can score well by recognising voices instead of words.
- **Training would have crashed.** A 1 s clip with `hop_length=256` gives 16000/256 ≈ **63** frames, not 64. That's 3 time patches instead of 4, so 24 patches + CLS against a positional embedding sized for 32 + CLS.
- **The features were never normalised.** dB values span roughly −100 to +40.
- **The positional and CLS embeddings started at std 1.0.** That's as large as the patch embeddings, so position would drown out content at the start of training. ViT and DeiT use 0.02.
- **Parts were never connected.** `get_dataloaders` didn't exist, the augmentations were never called, and `AddBackgroundNoise(background_paths=None)` errors.

**Lesson:** shape bugs and split bugs never show up as an error message in the metric. A crash is the *best* case; a wrong split gives a great-looking number that means nothing.

### Step 2: use the authors' *code* as the reference, not the paper's prose
**What:** downloaded AST's real config files and source: `run_sc.sh`, `dataloader.py`, the HF port, and the fine-tuned checkpoint's `config.json` and `preprocessor_config.json`.
**Why:** papers compress details; the code doesn't. Example: **the AST paper says "Hamming window", but its code uses `hanning`.** We follow the code, because the released weights were trained with it.
**Lesson:** "the rawest material" means the authors' code, configs and data, not the paper's summary of them.

### Step 3: switch to the full Speech Commands v2 and its official split
**What:** 35 words, ~106k clips, with the official `validation_list.txt` and `testing_list.txt`.
**Why:** the official split is **speaker-disjoint** (the speaker id is the part of the filename before `_nohash_`). It's also what every published number uses, so our numbers become comparable to theirs.
**Result:** 84,843 / 9,981 / 11,005 clips from 2,112 / 256 / 250 speakers, with no overlap (asserted in code).

### Step 4: write our own Kaldi filterbank in plain torch (`features.py`)
**What:** reimplemented `torchaudio.compliance.kaldi.fbank` step by step: frame → remove DC → pre-emphasis → Hann window → FFT → power → mel triangles → log.
**Why:**
1. Writing it ourselves is how you actually learn what a spectrogram is.
2. torchaudio is no longer released in step with torch, which is a risk on Kaggle.
3. It runs batched on the GPU, which Phase 2 needs.
4. The browser demo (Phase 3) will need a JavaScript port, and a short torch reference is easy to port.

**Verified by:** tests compare it with torchaudio and with HF's AST feature extractor (max difference < 1e-3).

### Step 5: make the model general enough to load the authors' weights
**What:** added overlapping patches (stride 10), configurable prefix tokens (`[CLS]` + `[DIST]`), an optional LayerNorm in the head, and configurable LayerNorm eps.
**Why:** if our modules can load AST's fine-tuned weights and produce **the same logits**, then our patch ordering, attention layout, norm placement and pooling all compute the same function as theirs. This is a **parity test**, and it's far stronger than "looks right to me".
**Result:** logits match within 1e-3, on random inputs and end to end from audio. With their weights, our code classified 74 of 74 real clips correctly.

### Step 6: tests that each guard one specific mistake
- `test_splits`: speakers don't overlap, every word appears in every split, sizes are the official ones.
- `test_model`: patch arithmetic, a clear error for wrong shapes (the 63-frame bug), init scale, **overfit one batch**.
- `test_features` and `test_parity_ast`: agreement with the reference implementations.

**Lesson:** a good test is named after the bug it prevents.

---

## Session 2 (2026-11-01): first Kaggle run (Phase 0–1)

### Results (seed 0, no augmentation, 30 epochs)
| Model | Params | Val | Test | ECE | Time |
|---|---|---|---|---|---|
| TinyAudioTransformer | 0.85M | 93.81 | **92.93** | 3.2% | 12 min |
| CNN baseline | 0.25M | 93.05 | 92.33 | 1.2% | 16 min |
| *AST (paper)* | *87M, ImageNet-pretrained* | | *98.11* | | |

### What we learned from it
- **The transformer beats the CNN, and not by chance.** On the same 11,005 test clips it got 417 clips right that the CNN got wrong, against 351 the other way round. McNemar p = 0.019, 95% CI for the gap [+0.09, +1.09] points. But this is **one seed**, and seed-to-seed variation could be about that large. That's why Phase 2 uses 3 seeds.
- **It's overfitting:** 97.8% train accuracy (even with dropout on) against a 93.8% val plateau from about epoch 21.
- **It's overconfident:** mean confidence 96% at 93% accuracy (ECE 3.2%). This matters for a demo that shows confidence.
- **Its mistakes sound like human mistakes:** tree↔three, go→no, four/forward/follow, on↔off, left→learn. The weakest words (forward, learn, follow, tree) have the least training data, because they were only added in v2. Sensible confusions are a good sign the pipeline is healthy.
- **A cross-check:** our training-set feature stats (mean −6.79, std 5.56) almost exactly match the AST authors' (−6.85, 5.57), an independent sign our features are on the same scale as theirs.

### The bug the run caught: the parity test failed on Kaggle
**What happened:** Kaggle had `transformers` 5.16.1 and I had tested on 4.57. Version 5 renamed AST's internal modules (`encoder.layer.N.attention.attention.query` → `layers.N.attention.q_proj`), so the weight mapping hit a `KeyError`.
**Fix:** rewrite old names to new ones before mapping, and **fail if any reference weight goes unused**. Verified on both versions.
**Lessons:**
1. The test environment is part of what you're testing. Kaggle isn't your laptop.
2. The test did its job: it **failed loudly** instead of passing silently. A weight-loading check that ignored unknown keys would have "passed" while comparing nothing.
3. Read the actual outputs, not the summary. I first reported Phase 1 as done before seeing the notebook, and the notebook showed 2 errors.

---

## Session 3 (2026-11-01): designing Phase 2

### The question
Which augmentations help a *tiny, from-scratch* transformer on this data, and is each effect bigger than seed-to-seed noise?

### Design decisions and why

**1. Cache waveforms, not spectrograms.**
Shift, mixup and background noise are *physical* operations: two sounds playing together add as **waveforms**. Adding two *log*-spectrograms isn't the same thing, since log(a+b) ≠ log a + log b. So the cache now holds lossless int16 waveforms (still ~2.7 GB), and fbank runs on the GPU for every batch, after the waveform augmentations. Owning `features.py` (Step 4) is what makes this cheap.
*Side effect:* clips shorter than 1 s are now zero-padded as audio (silence) before fbank, instead of having their spectrogram padded afterwards. So Phase 2's "no augmentation" run is the new baseline, not Phase 1's number.

**2. One change at a time (an ablation), and 3 seeds.**
Configs: none, shift, noise, mixup, SpecAugment, label smoothing, all. Each is compared with "none" **using the same seed**, which is a paired comparison. A real effect should help for *every* seed and exceed the baseline's seed spread. With one seed you can't separate a trick that works from a lucky initialisation.

**3. Augmentation settings, and where they come from:**

| Augmentation | Setting | Source / reasoning |
|---|---|---|
| Time shift | ±100 ms | Google's Speech Commands tutorial default; the word's position in the window shouldn't matter |
| Background noise | p = 0.8, SNR 0–20 dB, real recordings from `_background_noise_` | the dataset ships these files for this purpose; 0 dB = noise as loud as speech |
| Mixup | p = 0.6, λ ~ Beta(10, 10) | AST's speech-commands recipe; Beta(10, 10) keeps λ near 0.5, so a genuinely mixed signal |
| SpecAugment | 1 freq mask ≤ 24 bins, 1 time mask ≤ 24 frames | AST uses 48/48, but it's a huge pretrained model; halved for a 0.85M model (a choice to revisit) |
| Label smoothing | ε = 0.1 | standard value; aimed directly at the overconfidence we measured |

**4. Two things we only noticed by looking at the spectrograms** (the "listen and look" cell):
- *Time masks were landing in the padding.* A 1 s clip has 98 real frames, padded to 128. Masks drawn over all 128 frames partly hide nothing about a quarter of the time. **Fix:** time masks stay within the real frames. This is a deliberate deviation from AST.
- *Padding is brighter than silence.* Zero in a log-spectrogram is log(1), medium-loud energy, while true digital silence is log(ε) ≈ −15.9. AST pads with zero, so we keep it for now: changing it mid-ablation would confound the comparison, and the parity tests rely on it. → [Open question 1](#open-questions).

**5. Measure more than accuracy.**
Every run now records val and test **ECE** (calibration) and **NLL**. Label smoothing and mixup are supposed to fix calibration even when accuracy barely moves, and we'd miss that if we only tracked accuracy.

**6. Use both GPUs well.**
A 0.85M-parameter model doesn't keep a T4 busy, so `sweep.py` runs 2 jobs per GPU (4 at once). It also resumes: finished runs are skipped if the session dies.

**Verified locally before Kaggle:** 38 tests pass. New tests check that noise lands at exactly the requested SNR, that mixup is a convex combination whose labels sum to 1, that SpecAugment masks are contiguous and only cover real frames, that the soft cross-entropy matches PyTorch's, and that ECE and McNemar behave on hand-computed cases. A full sweep and analysis also ran on a small fake dataset.

---

## Session 4 (2026-11-01): Phase 2 results

All 27 runs finished (about 13 min each, 4 at a time). Raw outputs are in `results/phase2/runs_phase2/`, and re-running `analyze.py` locally reproduces Kaggle's `summary.md` exactly.

### Results (mean ± std over 3 seeds; Δ paired by seed against `none`)

| Config | Test acc | Δ vs none | Seeds better | Test ECE | Mean confidence − accuracy |
|---|---|---|---|---|---|
| tat_none | 93.39 ± 0.10 | — | — | 3.01 | **+3.0** (overconfident) |
| tat_shift | 94.40 ± 0.13 | **+1.01** | 3/3, all p < 1e-6 | 1.77 | +1.7 |
| tat_specaug | **94.45 ± 0.09** | **+1.06** | 3/3, all p < 2e-5 | **0.82** | +0.8 |
| tat_mixup | 93.82 ± 0.42 | +0.43 | 3/3, p = 0.8 / 0.02 / 0.0005 | 6.21 | −6.1 (underconfident) |
| tat_ls | 93.58 ± 0.19 | +0.19 | 3/3, none significant | 5.85 | −4.6 (underconfident) |
| tat_noise | 92.08 ± 0.72 | **−1.31** | 0/3 | 1.71 | +1.7 |
| tat_all | 88.49 ± **2.34** | **−4.90** | 0/3 | 15.26 | **−15.2** (underconfident) |
| cnn_none | 92.42 ± 0.07 | — | — | 1.29 | −1.2 |
| cnn_all | 90.53 ± 0.02 | −1.89 | 0/3 | 23.26 | **−23.3** (underconfident) |

### What we learned

**1. Two augmentations clearly work: time shift and SpecAugment.** Each adds about +1 point, on every seed, with a margin ten times the baseline's seed spread (std 0.10). Both also *improve* calibration. They share one property: the label stays hard. They change what the model hears, not what it's told the answer is.

**2. Label smoothing and mixup "fixed" overconfidence by overshooting.** The baseline was 3 points overconfident. Both turned it into 5–6 points of *under*confidence, which is why their ECE went up. That's what training on soft targets does: with mixup at Beta(10, 10), 60% of batches have targets close to 50/50, so the model learns to hedge. Accuracy gains were small (mixup) or within noise (label smoothing).
*Lesson:* ECE is a distance with no sign. Always check the direction (mean confidence − accuracy) before calling calibration better or worse.

**3. Background noise hurt clean-test accuracy (−1.3 on every seed).** It's not a bug: the SNR test passes, and the model still trains (val 93.1). The likely reason is that the test set is clean recordings, so noise at 0–20 dB on 80% of clips trains for a harder world than the one we score in. This experiment can't show the noise model's benefit: robustness to real noise. That needs a *noisy* test set, which matters for Phase 3 because real microphones aren't clean.

**4. "All together" is the worst config, and it's unstable.** −4.9 points, and test accuracy across seeds ranges 86.0 to 90.7. Val was still climbing at epoch 30 (best epochs 26, 29, 30), so the stacked regularisation is too much for a 0.85M model in 30 epochs. The CNN shows the same pattern with even more underconfidence. This answers [open question 2](#open-questions): with all augmentations on, 30 epochs is not enough. It also shows that augmentations don't simply add up. They interact, so the good ones have to be combined and tested again.

**5. The transformer beats the CNN, and the margin grew.** Unaugmented, TAT is 0.97 points ahead with 3 seeds. With shift or SpecAugment it's 2 points ahead.

**6. "Train accuracy" with mixup isn't comparable.** `tat_all` shows 48% train accuracy against 90% val. That's not a bug: the training argmax is scored against blended labels. Only runs without mixup have a meaningful train/val gap. For `tat_none` that gap is 98.0 vs 94.0, so it still overfits.

### Next (Phase 2b, proposed)
- **Combine the winners:** `shift + specaug`, 3 seeds. If the gains add up, that's about 95.4.
- **Then add the soft-target tricks back one at a time** on top of it, to see whether mixup helps once the hard-target augmentations are in place.
- **Noisy test set:** evaluate the existing `none` and `noise` checkpoints on test clips with noise added at fixed SNRs (20, 10, 5, 0 dB). This is evaluation only, no training. The noise has to come from segments held out from training, or the test leaks.
- **Gentler noise** (p 0.5, SNR 10–30 dB) if the noisy test shows noise training matters.

---

## Session 5 (2026-11-03): building Phase 2b

### The questions
1. Do the two winners add up? (`shift + specaug`)
2. Once they're in, does anything else still help? (`+ mixup`, `+ ls`, `+ noise`, each compared with `shift + specaug`, not only with `none`)
3. What does noise training actually buy? (score every checkpoint on noisy audio)

### Design decisions and why

**1. Hold out noise, not only speakers.** The background recordings are now split in time: the first 80% of each file is for training augmentation, and the last 20% is only for evaluation. If the noisy test reused segments the noise-trained model had heard, it would partly measure memorisation of those exact sounds. It's the same principle as the speaker-disjoint split, applied to the noise.
*Side effect:* the training noise bank is 20% smaller than in Phase 2, so `tat_noise` is re-run instead of reused.

**2. Paired noisy evaluation.** Each clip gets one fixed noise segment (a seeded generator per split). Every model and every SNR sees exactly that segment, so differences between models come from the models, not from luckier noise draws. Mixing goes through one shared function, `augment.mix_at_snr`, for both training and evaluation, so the SNR means the same thing in both. There's no clipping, so the SNR stays exact.

**3. Decide on val, report on test.** Choosing a noise setting by looking at noisy *test* scores would quietly tune to the test set, so `eval_noisy.py` scores both splits.

**4. Re-run the baseline in the same session.** `tat_none` runs again, so every comparison inside 2b is same-session. It also gives a free reproducibility check: does 93.39 ± 0.10 come back?

**5. A self-check inside the evaluation.** `eval_noisy.py` re-scores the clean test set and warns if any checkpoint doesn't reproduce its reported `test_acc` (to within 0.05). That catches a broken checkpoint load or a feature mismatch before the noisy numbers can be wrong without anyone noticing.

**6. Analysis improvements from Session 4's lessons.** `analyze.py` now shows **confidence − accuracy**, which gives the calibration error a sign, and a **Δ vs parent** comparison. The parent is the config a run extends, so `shift_specaug_mixup` is compared with `shift_specaug`.

**Verified locally:** 40 tests pass. The new ones check that the train and test noise banks never overlap, that the SNR is exact, that noisy clips are identical across calls, that checkpoints load and reproduce their outputs, and that the parent lookup works. The full sweep, analysis and noisy-eval pipeline also ran on the fake dataset, and the clean-score self-check passed.

---

## Session 6 (2026-11-03): Phase 2b results

All 18 runs finished. Raw outputs are in `results/phase2b/runs_phase2b/`. Checks: `analyze.py` reproduces Kaggle's summary exactly, and all 18 checkpoints reproduce their reported clean test score in the noisy evaluation, so the noisy numbers come from the right models.

**Reproducibility:** `tat_none` came back at 93.52 ± 0.31 (Phase 2: 93.39 ± 0.10), and `tat_noise` at 92.06 (Phase 2: 92.08). Same code, new session, same answer within seed noise.

### Clean test (mean ± std over 3 seeds)

| Config | Test acc | Δ vs none | Δ vs parent (`shift_specaug`) | Test ECE | Confidence − accuracy |
|---|---|---|---|---|---|
| none | 93.52 ± 0.31 | — | — | 2.88 | +2.9 |
| **shift + specaug** | **94.77 ± 0.21** | **+1.25**, 3/3 seeds | — | **0.59** | **+0.3** |
| + label smoothing | 94.80 ± 0.36 | +1.29 | **+0.03** (no effect) | 6.48 | −6.4 |
| + mixup | 93.40 ± 0.04 | −0.11 | **−1.37**, 0/3 seeds | 6.45 | −6.4 |
| + noise | 92.35 ± 0.87 | −1.17 | **−2.42** on clean audio | 0.88 | −0.2 |

### Test accuracy with held-out noise

| Config | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| none | 93.52 | 89.64 | 81.26 | 70.91 | 53.65 |
| shift + specaug | **94.77** | 91.49 | 83.39 | 74.24 | 57.91 |
| noise | 92.06 | 91.56 | 89.35 | 86.25 | 80.55 |
| **shift + specaug + noise** | 92.35 | **91.75** | **89.87** | **87.29** | **82.00** |

(Val shows the same ordering at every level, and decisions are based on val.)

### What we learned

**1. The winners combine, but don't simply add.** Shift + SpecAugment gives +1.25. Each alone gave about +1.0, so a sum would be about +2. Both teach "the word can be partly hidden or moved", so they overlap. It's still the best clean model, and the best calibrated one we've trained (ECE 0.59, almost exactly honest).

**2. Once the winners are in, the soft-target tricks don't help.** Label smoothing adds nothing to accuracy and makes the model 6 points underconfident. Mixup actively *hurts* (−1.37 on every seed), even though alone it gave +0.43 in Phase 2. With shift and SpecAugment already regularising, mixup tips the model into underfitting. That's the interaction effect from Session 4, now confirmed directly.

**3. The clean test was hiding what noise training buys.** Without noise training, accuracy collapses as noise rises: 93.5 → 53.7 at 0 dB. With it, accuracy falls gently: 92.4 → 82.0. That's **+24 points at 0 dB and +13 at 5 dB**, compared with the same model trained without noise. The cost is −2.4 points on perfectly clean audio. Already at 20 dB (faint noise) the noise-trained model is ahead. Phase 2 called noise "harmful" only because it measured in a world with no noise.
*Lesson:* the test set defines what "better" means. Choose it to match where the model will be used.

**4. The noise models hadn't finished learning.** Their val accuracy was still rising between epochs 25 and 30 (+0.3–0.4), while the clean models had flattened. More epochs, or slightly gentler noise, may recover part of the clean cost.

**5. The deployment candidate is `shift + specaug + noise`.** Phase 3 means real people on real microphones, in rooms with fans, traffic and other voices. A model that loses 2 points on studio-clean audio but keeps 82–90% under noise is the right trade for that. `shift + specaug` stays as the "clean audio" reference.

---

## Session 7 (2026-11-03): Phase 3, from checkpoint to a website

### Step 1: choose the final model on val, not test
Three seeds of `shift + specaug + noise`. Seed 0 has the best **val** accuracy at *every* noise level (clean 94.26, 0 dB 84.98), so it was chosen before looking at its test numbers. Its test results (93.35 clean, 83.06 at 0 dB) are then reported as they are.

### Step 2: put the feature extraction *inside* the exported model
**What:** `export_onnx.py` wraps the model so the ONNX graph takes the raw **waveform** and computes the log-mel filterbank itself.
**Why:** the most common way a deployed audio model fails without any error is the website computing features slightly differently from training: another FFT library, window or mel formula, or log base. If the browser only passes waveform samples, there's nothing to get wrong.
**How:** two operations were rewritten in a form every ONNX runtime supports, with the same maths:
- *Framing:* `Tensor.unfold` → indexing with a fixed `[98, 400]` table of sample positions.
- *FFT:* `torch.fft.rfft` → **a matrix multiply with precomputed cos/sin tables**. A DFT *is* a matrix multiply: X[k] = Σₙ x[n]·e^(−2πikn/N). Because the frame is zero-padded from 400 to 512, only the first 400 rows of the table matter, and the Hann window (a per-sample scale) can be folded into the table.

The graph also outputs the features, so the demo can show the spectrogram the model actually saw.

### Step 3: check the export against the *training* pipeline
ONNX Runtime was compared with the original PyTorch path (`features.fbank` with unfold + rfft, then the model), so one check covers both the rewrite and the export:
- random audio from loud to near-silent: max |Δlogit| 4.7e-6
- **8,000 real speech clips** (`mini_speech_commands`): max |Δlogit| 2.7e-5, and **100% identical predictions**

These clips aren't an accuracy benchmark, since their speakers may overlap with training. Their job is to check agreement. `tests/test_export.py` guards both the frontend rewrite and the round trip.

### Step 4: the website (`web/`)
Static files: `index.html`, `app.js`, `style.css`, and `model/tat.onnx` (4.7 MB). onnxruntime-web runs the model on the device's CPU through WebAssembly, so it needs no server and no upload. The pipeline:
1. **Microphone at whatever rate the device uses** (usually 48 kHz), with the browser's echo cancellation, noise suppression and auto-gain **off**, since training never saw their processing.
2. **Resample to 16 kHz** with the browser's own resampler (`OfflineAudioContext`).
3. **Record 2 s and centre a 1 s window on the word**, using the energy centroid of loud 10 ms frames. Training clips have the word roughly centred, and the ±100 ms shift augmentation covers small offsets. Outside the recording counts as silence, so a word spoken late is still centred. That bug was caught by a JavaScriptCore unit test, written before any browser test.
4. **Loudness gate:** below about −50 dBFS → "didn't hear anything". The model has no silence class, so without this gate it would confidently "hear" a word in silence.
5. Show top-5 probabilities, the spectrogram (padding marked), and a button to play the exact second the model heard.

**"Test it" mode** is the real-user evaluation from the original plan. It shows each of the 35 words once in random order. You say it, and the page logs target, prediction, confidence, the rank of the right answer and the input level, but no audio. The results download as JSON. That provides a measurement the dataset can't: our own voices, microphones and rooms.

### Step 5: the model card (`docs/MODEL_CARD.md`)
It records what the model is for and not for, the numbers (clean and noisy, per-seed context, weakest words, confusions), and the limitations the numbers don't show. Those are: no unknown/silence class, unmeasured accent coverage, noise from only 6 recordings, and sensitivity to clip framing.

### Lessons
- **Make the deployment boundary as thin as possible.** Waveform in, logits out. Every piece of preprocessing left outside the model is a place for train/serve skew.
- **Test the export against the training code, not against itself.** Comparing ONNX with the rewritten frontend would only show that the export works. Comparing it with `features.fbank` shows the deployed model computes what was trained and evaluated.
- **A classifier always answers.** A closed-set model has no way to say "none of these". The loudness gate and the confidence threshold are workarounds. The real fix (an `_unknown_`/`_silence_` class, as in the 12-class benchmark) is an open question.

---

## Session 8 (2026-11-04): the first real-microphone test

**What:** one full "Test it" round in Safari 18.6 on a MacBook's built-in mic. All 35 words were said once, in random order, in a quiet room. Input levels ranged from −35 to −27 dBFS, comfortably above the −50 dBFS silence gate. The raw file is `results/phase3/real_user/`.

**Result: 34 / 35 correct (97.1%)**, and the right word was in the top 5 every time.

**What it does and doesn't tell us:**
- **The deployment pipeline works end to end in a real browser:** mic, resampling, centring, ONNX fbank and model. If any step were subtly wrong (wrong sample rate, mis-centred word, different features), accuracy would collapse, not land at 97%. This is the check that couldn't be done offline.
- **The mistakes are the ones the test set predicted.** The only error was tree → three, the most frequent confusion on the official test set. The least confident correct answers were all on known confusable pairs: no/go 63%, dog/no 71%, three/tree 72%, go/no 72%, four/go 77%. So the model's behaviour carried over from the dataset to a new voice and microphone, including *where* it's unsure. It was also less confident when wrong (75%) than on average when right (93%), so the confidence shown is a useful hint.
- **It is not an accuracy estimate.** With 35 trials, the exact 95% interval for 34/35 is **85.1–99.9%**. That's consistent with the test set's 93.4% and says little more. One speaker in one quiet session doesn't cover accents, noise or other microphones.

**Lesson: count the trials before celebrating a number.** 97.1% sounds better than the test set's 93.4%, but with n = 35 the two can't be told apart. The value of this run is *qualitative* (the pipeline works, and the errors are the expected ones). Getting a *quantitative* real-world number needs more people, rooms and devices.

---

## Session 9 (2026-11-05): going live

- **Repo:** the rebuilt project was pushed as one commit on top of the original July 2025 history, so nothing old is lost. Training outputs, datasets and zips stay out of git, and the site only needs `web/`. New: `README.md` (for both non-technical and technical readers, with figures from `docs/make_figures.py`) and an MIT `LICENSE`.
- **Caught before publishing:** the ONNX exporter embeds a Python stack trace in every graph node, including local file paths (username, folder layout). `export_onnx.py` now strips that debug info. The model shrank from 5.1 to 4.7 MB with identical outputs (re-verified on 8,000 clips).
  *Lesson:* look inside the artifacts you publish (`strings model.onnx | grep Users/`), not just at your source code.
- **Hosting:** GitHub Pages, deployed by a GitHub Actions workflow from `web/`. The first deploy failed with "Get Pages site failed / Not Found" because Pages has to be switched on once in the repo (Settings → Pages → Source: GitHub Actions). After that, a manual re-run deployed it. Every push that touches `web/` now redeploys automatically, and the test suite runs on every push.
- **Live:** https://dhaaivat.github.io/TinyAudioTransformer/

---

## Open questions
1. **Pad with zero or with the log floor?** Does padding with log-silence (−15.9) instead of 0 change accuracy? It's a one-line change, to test after Phase 2 against a 3-seed baseline.
2. ~~**Is 30 epochs enough with augmentation?**~~ **Answered (Session 4):** not for `all`; val was still rising at epoch 30. Single augmentations peak around epochs 27–29, so 30 is borderline but fine for them.
3. **SpecAugment strength:** 24 vs AST's 48. Is the tiny model under-regularised or over-regularised?
4. **KWT's 35-word numbers:** the KWT README only reports 12-word results. Read the paper's table to get a fair from-scratch target.
5. **Can the noise model's clean cost be recovered?** (Session 6) Try longer training (val was still rising at epoch 30) or gentler noise (p 0.5, SNR 5–30 dB), and check clean *and* 0–10 dB together.
6. **Is our held-out noise representative?** It comes from the same 6 recordings as the training noise (different segments). Real microphones will bring noise types the model has never heard. Phase 3's real-user evaluation is the true test.
7. **Add "unknown" and "silence" classes?** (Session 7) The model always picks one of 35 words. Training with the dataset's background noise as `_silence_` plus held-out words as `_unknown_` would let it say "none of these". That's how the standard 12-class benchmark is set up.

---

## Glossary

### Audio
- **Waveform:** the raw signal, air pressure sampled 16,000 times per second (16 kHz). A 1 s clip is 16,000 numbers in [−1, 1]. Stored as int16 (−32768…32767), which is lossless.
- **Frame / hop:** a spectrogram looks at short overlapping windows. Kaldi uses 25 ms windows (400 samples) every 10 ms (160 samples), which gives 98 frames per second.
- **Pre-emphasis:** `x[t] − 0.97·x[t−1]` boosts high frequencies, which are weak in speech but carry consonants.
- **Window (Hann/Hamming):** a bell-shaped taper applied to each frame so its edges don't create fake frequencies (spectral leakage).
- **FFT and power spectrum:** splits each frame into frequency components. Power = |FFT|². A 512-point FFT gives 257 frequency bins.
- **Mel scale:** human hearing resolves low pitches finely and high pitches coarsely. Mel filters are triangles packed densely at low frequencies and sparsely at high ones (mel = 1127·ln(1 + f/700)).
- **Log-mel filterbank (fbank):** the log of the energy in each mel band per frame, a 2D "image" of [time × 128 bands]. Log because loudness perception is roughly logarithmic, and it compresses a huge dynamic range.
- **SNR (signal-to-noise ratio):** 10·log10(P_signal / P_noise) in dB. +20 dB means noise at 1% of the signal's power; 0 dB means equal power.

### Model
- **Patch embedding:** cut the spectrogram into 16×16 tiles and project each to a vector with a strided Conv2d. Stride 10 < 16 means tiles overlap by 6, giving 12×12 = 144 patches.
- **[CLS] token:** an extra learned vector placed in front of the patches. After attention it has gathered information from all of them, and the classifier reads it. AST adds a second one, [DIST], from DeiT distillation.
- **Positional embedding:** attention has no built-in notion of order, so a learned vector per position tells the model *where* each patch is.
- **Pre-norm block:** `x + Attn(LN(x))`, then `x + MLP(LN(x))`. Normalising *before* each sublayer keeps the residual path clean and makes deep transformers train stably.
- **Init std 0.02:** small initial weights keep early activations and gradients in a sane range. Std 1.0 for position embeddings would swamp the content signal.
- **Feature normalisation `(x − mean)/(2·std)`:** puts inputs roughly in [−1, 1]. The factor 2 is AST's convention. We keep the stats inside the model as buffers, so they ship with the checkpoint.

### Training
- **AdamW plus weight decay:** Adam with decoupled L2 shrinkage. Biases, norms and embeddings are excluded, because shrinking them toward 0 has no regularising meaning.
- **Warmup plus cosine schedule:** start with a tiny learning rate (Adam's statistics are noisy at first), ramp up, then decay smoothly to 0.
- **AMP (mixed precision):** run matmuls in fp16 for speed, with loss scaling to avoid gradient underflow. The fbank is kept in fp32, since power spectra span many orders of magnitude.
- **Gradient clipping:** caps the gradient norm, so one bad batch can't throw the weights off.
- **Overfit-one-batch test:** a healthy model must be able to memorise a handful of examples. If it can't, there's a bug in the wiring or gradients, and no hyperparameter will fix it.
- **Mixup:** train on blends of two clips with blended labels. It discourages over-confident, sharp decision boundaries.
- **SpecAugment:** blank random bands of frequency and time, so the model learns to use redundant cues.
- **Label smoothing:** a target of 0.9 instead of 1.0 for the true class. The model is never pushed to infinite confidence.

### Evaluation
- **Speaker-disjoint split:** no person appears in more than one split, so the test measures recognising *words from new voices*, which is what deployment needs.
- **Model selection on val, test once:** choosing the checkpoint by test accuracy would quietly tune to the test set. Val is for decisions; test is reported once.
- **Parity test:** load a reference implementation's weights into your code and check the outputs match numerically. This is the strongest correctness check available for "I implemented the paper".
- **McNemar test:** compares two classifiers *on the same examples*, counting only the clips where they disagree. Much more sensitive than comparing two accuracy numbers.
- **Seeds and paired deltas:** run each config with several random seeds. Compare config vs baseline *with the same seed*, and trust an effect when it's positive for every seed and larger than the spread between seeds.
- **ECE (expected calibration error):** group predictions by confidence and measure how far confidence is from actual accuracy in each group. 0 = when the model says 90%, it's right 90% of the time.
- **Ablation:** change one thing at a time against a fixed baseline, so each effect can be attributed to its cause.
- **Over- vs underconfidence:** mean confidence minus accuracy. Positive means the model claims more than it delivers; negative means it hedges. ECE alone can't tell the two apart.
- **Interaction effect:** when two changes together don't give the sum of their separate effects. Stacking regularisers is the classic case: each alone helps a little, all together underfit.
- **Train/serve skew:** the deployed system computes something slightly different from what was trained, e.g. features made with another library. It causes no error, only worse predictions. The cure is to ship preprocessing inside the model and test the export against the training code.
- **ONNX / onnxruntime-web:** ONNX is a portable file format for a computation graph. onnxruntime-web runs it in the browser via WebAssembly, so the model runs on the user's device.
- **Confidence interval for an accuracy:** a range that would contain the true accuracy in 95% of repeated experiments. For k correct out of n, the exact (Clopper–Pearson) interval comes from the binomial distribution. It's wide when n is small: 34/35 → 85–100%.
- **Closed-set classifier:** it must pick one of its known classes, even for inputs that are none of them. Confidence thresholds and extra "unknown" classes are the standard workarounds.
- **Train/test distribution mismatch:** training on harder data (heavy noise) than you evaluate on can lower the score on the clean test even when the model is more robust. To measure robustness, you need a test set that has the condition in it.
