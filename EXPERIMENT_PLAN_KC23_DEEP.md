# Plan: kill-critic 23 September, deep-model stages (KC-D0 to KC-D5)

**Phase 0/1 status header (23 September 2026, this session; plan text below unedited):**
- **KC-D0:** done. D0.2 (chanoffset, globalgain), D0.3 (permutation.csv, embed_probes.csv) and D0.5 (`run_adv_align_loso.py`, `--coral-normalize` on `run_deep_coral_align_loso.py`) all landed. Both mandatory D0.4 inertness assertions PASS (CPU, since GPU training is cross-process nondeterministic here via `cudnn.benchmark=True`). See `KC23_PHASE1_REPORT.md`.
- **KC-D1 to KC-D6:** not run. Queued in `kc23_jobs_gpu.csv` / `kc23_jobs_cpu.csv` (KC-D6 Stage 2, item 11c, deliberately left ungenerated until Stage 1's manipulation gate is known, per the plan's own staging).

**Status:** ready to run. Written 23 September 2026 from `04_Reviews_and_QA/KILL_CRITIC_2026-09-23.md`.
**Execution:** local GPU (RTX 4050 Laptop, 6 GB), one job at a time, through Claude Code. Dispatcher: `RUN_ORDER_KC23.md`.
**Cost:** about 75 to 85 GPU hours for KC-D1 to KC-D5, plus up to about 72 GPU hours for KC-D6 (added 23 September at Enam's request; staged, so a failed manipulation check stops its later seeds early). All serial. The measured references are 25.6 min for a no-augmentation `resnet` 40-fold run and 52.5 min for a channel-dropout one (`_run_logs/w2_g1.log`). Every arm records its wall-clock, so the estimates below are replaced by measurements as they come in.
**Owner decision points:** D1 Section 1.6 (headline gate) and D1 Section 1.7 (family replacement, deferred to KC-F1). Everything else carries a pre-registered rule.

| Stage | Kill-critic item | Question |
|---|---|---|
| KC-D0 | all | Code changes (instrumentation, two augmentation modes) and their inertness proofs |
| KC-D1 | M1 | What is the real run-to-run variance, and which deep contrasts survive it? |
| KC-D2 | M2 | Does channel dropout reduce *reliance*, or only sensitivity to the zeroing it was trained on? (analysis of D1's instrumented runs) |
| KC-D3 | moderate 1 | Is the active ingredient per-channel multiplicative variance, or variance at the right magnitude? |
| KC-D4 | M6 | On the channel axis, does *subject* invariance rise with dose while accuracy follows class information? |
| KC-D5 | M4 | Do the deep mechanism findings replicate on ENABL3S, and where does ResNet-SE+CD stand against the SVM there across seeds? |
| KC-D6 | M6, and the Section 5.3 limitation on Deep CORAL | With alignment knobs that provably move invariance (a subject adversary and a scale-free Deep CORAL), does a learned, in-network alignment axis show the through-line's shape, and does class-conditional alignment escape the fall? |

---

## 0. Read this first

1. **Interpreter:** `06_Code/.venv/Scripts/python.exe`.
2. **Do not edit `01_Thesis/`. Do not touch the v9 family. Do not change the deep model of record** (`resnet_se` with channel dropout p = 0.2, 250 ms, per-subject normalization).
3. **Every arm writes to a new `results_kc23_d*` directory**, logs its command line and writes `run_config.json`.
4. **cuDNN determinism stays off**, as in every published run, so new realizations are comparable with the published ones. Seeds change initialization, validation-subject choice and batch order. That is exactly the training-realization variance being measured.
5. **Run seed by seed, not arm by arm.** Finish every Tier A arm at one seed before starting the next seed, so an interruption leaves a balanced partial design.
6. **Check every `--resume` for duplicated subject rows** before trusting a summary.
7. **Two parameters are called `p`.** `--aug-chandrop-p` is the augmentation rate; `p = 0.1` in `ResBlock.__init__` is architectural dropout and must not be touched.
8. **The SE gates weight feature-map channels (32 to 128), not the nine electrodes.**
9. **No model weights are saved anywhere**, so every measurement on a trained model is taken by `--instrument` during the run. That is why D0 comes first and why every D1 arm is instrumented.
10. **Escalation protocol** as in the classical plan: write `06_Code/KC23_HALT.md`, finish the running arm, start no dependent arm.

---

## KC-D0. Code changes and inertness. No GPU runs until D0 passes.

### D0.1 Capture the "before" state first

Before editing anything, run a capture script (`kc23_d0_capture_before.py`) that saves:

- for every existing augmentation mode (none, gaussian, chandrop, timemask, combined, gainjitter, subset, mpchandrop), in **both** `train_cnn_loso.py` and `run_cnn_arch_loso.py`: the output tensor bytes of `augment_batch` on a fixed seeded batch, and the post-call torch RNG state;
- one instrumented smoke run (`--heldout 1 --epochs 2`, arch `resnet_se`, chandrop): its `occlusion.csv`, `attenuation.csv`, `se_gates.csv`, final F1 and post-run RNG state.

### D0.2 Change 1: two augmentation modes (shared `augment_batch`)

- **`chanoffset`:** per sample and per channel, add a constant offset drawn from N(0, σ²), with σ = `--aug-gain-sd`. Tests per-channel *additive* perturbation.
- **`globalgain`:** per sample, one multiplicative gain shared by all channels, drawn uniformly with the same SD construction as `gainjitter`. Tests whether *per-channel independence* matters.

Additive branches only. Existing modes' code paths are untouched.

### D0.3 Change 2: instrumentation (`_instrument` in `run_cnn_arch_loso.py`)

Keep the existing occlusion, attenuation and SE-gate blocks byte-identical, and add:

1. **Permutation reliance (`permutation.csv`).** For each channel, permute that channel's time series across the held-out subject's windows, R = 5 times, then record the mean F1 drop and its SD. This is the standard permutation-importance construction for model reliance. It keeps the channel's marginal distribution and removes its class information, so it does not feed the network the flat-line input that channel dropout trains on. Use a **separate** generator, `np.random.default_rng(10_000 * seed + subject)`, never the global RNG.
2. **Unseen-subject embedding probes (`embed_probes.csv`).** Extract penultimate embeddings (after global average pooling, before head dropout) for the held-out subject and the fold's validation subjects, all of them never trained on. Pass the validation-subject IDs into `_instrument`. Compute three things:
   - a linear subject-identity probe across these subjects: 5-fold stratified CV logistic regression, balanced accuracy, with chance = 1 / (number of subjects) recorded;
   - class silhouette on the held-out subject's embeddings;
   - a linear class probe on the held-out subject's embeddings (5-fold stratified CV, balanced accuracy).

   Use a fixed window cap per subject and class (`--probe-cap 100`) with a seeded subsample.

All additions stay inside the existing RNG snapshot and restore.

### D0.4 Inertness assertions (`kc23_d0_inertness.py`). Both mandatory.

1. For every existing augmentation mode in both scripts: identical output bytes and identical post-call RNG state before and after Change 1.
2. The smoke run after Change 2 reproduces the before-capture's `occlusion.csv`, `attenuation.csv` and `se_gates.csv` byte for byte, the same final F1, and the same post-run RNG state. (One smoke run is compared with another under the same process conditions. If cuDNN nondeterminism makes even the before-capture unrepeatable, run the smoke on CPU for this assertion and say so.)

**If either fails, stop.** Every prior run becomes incomparable.

### D0.5 Change 3, for KC-D6 only

These changes are specified in KC-D6 Section 6.3 and follow the same rule: capture the before state, then make an additive change, then assert byte-identity.

- **The new script `run_adv_align_loso.py`.** It is new code, so no inertness assertion applies to it, but it is covered by the KC-D6 sanity gate in Section 6.5.
- **The `--coral-normalize {none,l2}` flag on `run_deep_coral_align_loso.py`.** The `none` path must be byte-identical to today. Assert it on a CPU smoke run (`--heldout 1 --epochs 2`): identical per-step cross-entropy and CORAL loss values, identical `alignment_metrics` row, and identical post-run RNG state.

---

## KC-D1. The replicate programme (M1). The long pole.

### D1.1 Problem

Section 3.6 sets a 0.5-point band from three repeats (SD 0.47 on the forty-fold mean). With 2 degrees of freedom, the 95% interval for that SD runs from about 0.24 to 2.95 points. The thesis reports cross-run disagreements of 1.5 points (AdaBN pre-pass 0.787 against a 0.772 mean; V7 checks the configurations match) and 1.47 points (the Deep CORAL weight-100 contrast in two runs). Subject-resampled intervals cannot see training-realization variance, because each arm is one training.

### D1.2 Realizations

- **Tier A:** four new realizations at seeds **42 (re-run), 7, 123 and 1001**, plus the published seed-42 run as a fifth where one exists. Every analysis is also reported with the published run excluded, as a sensitivity.
- **Tier B:** three realizations at seeds **42 (re-run), 7 and 123**.

### D1.3 Arms

All arms: SIAT-LLMD, 250 ms unless stated, per-subject normalization unless stated. `run_cnn_arch_loso.py` arms run with `--instrument`.

| Arm | Script and configuration | Tier | Feeds | Est. per run |
|---|---|---|---|---|
| R1 | `run_cnn_arch_loso.py --arch resnet_se --augmentation none` | A | C1, C5, C6, D2, D3 | 30 min |
| R2 | `--arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --save-proba ... --model-tag RESNET_SE_CD` | A | C1, C2, C9, C11 to C14, D2 | 55 min |
| R3 | `--arch resnet_se --augmentation gainjitter --aug-gain-sd 0.40` | A | C2, D2, D3 | 55 min |
| R4 | `--arch resnet --augmentation none` | A | C5 to C8, D2 | 26 min |
| R5 | `--arch resnet --augmentation chandrop` | A | C6, C7, D2 | 52 min |
| R6 | `--arch resnet_nores --augmentation none` | A | C7, C8 | 25 min |
| R7 | `--arch resnet_nores --augmentation chandrop` | A | C7 | 50 min |
| R8 | SimpleEMGCNN, none (the exact published script and flags, from `REPRODUCE.md`) | A | C15 | 10 min |
| R9 | SimpleEMGCNN, chandrop | A | C15 | 12 min |
| R10 | `run_adabn_cnn_loso.py --arch resnet_se --augmentation chandrop --epochs 40` (pre and post from one training) | A | C9, C10 | 55 min |
| R11 | `run_deep_coral_align_loso.py --arch resnet_se --augmentation chandrop --coral-lambda 30 --batch 256` (the published best-weight configuration; confirm every flag from `REPRODUCE.md` and the D2c plan) | A | C10, C11 | 80 min |
| R12 | R2's configuration at **400 ms** (`windows_..._w400_...npz`) | A | C14 | 50 min |
| R13 | `resnet_se` chandrop p = 0.1 | B | C3, C16 | 55 min |
| R14 | `resnet_se` gainjitter SD 0.30 | B | C3 | 55 min |
| R15 | `resnet_se` chandrop p = 0.5 | B | C4, C16 | 55 min |
| R16 | `resnet_se` gainjitter SD 0.50 | B | C4 | 55 min |
| R17 | `resnet_se` chandrop p = 0.3 | B | C16 | 55 min |

Per-seed cost: Tier A about 8.3 h, Tier B about 4.6 h. Total: Tier A 4 × 8.3 = 33 h; Tier B 3 × 4.6 = 14 h.

**Per-seed ensemble:** for each R2 realization, recompute the soft vote with the (deterministic) SVM probabilities using `ensemble_v2_combine.py`, and record its F1.

### D1.4 Reproduction gate, before any other seed runs

After the Tier A arms at seed 42 (re-run) finish:

- R2's 40-fold mean must lie within ±1.5 pts of the published 0.8395, and R1's within ±1.5 of 0.782.
- R10's pre-adaptation mean must lie within ±1.5 of the published 0.787 or the 0.772 mean.

If any fails, **ESCALATE: code drift since publication.** Compare `run_config.json` against `REPRODUCE.md` first; most drift is a flag.

### D1.5 Registered contrasts and the verdict rule

Analysis script: `kc23_d1_replicate_stats.py`, with the rule encoded.

| ID | Contrast | Thesis location |
|---|---|---|
| C1 | R2 − R1: channel-dropout gain on ResNet-SE | 4.3.1 |
| C2 | R3 − R2: gain jitter SD 0.40 against channel dropout p = 0.2 | 4.3.3, 4.7 |
| C3 | R14 − R13: matched point SD 0.30 | 4.3.3 |
| C4 | R16 − R15: matched point SD 0.50 | 4.3.3, 4.3.4 |
| C5 | R1 − R4: the SE block without augmentation | 4.3.1 |
| C6 | (R2 − R1) − (R5 − R4): SE × channel dropout interaction | 4.3.1 |
| C7 | (R5 − R4) − (R7 − R6): skip × channel dropout interaction | 4.3.1 |
| C8 | R4 − R6: the skip connections without augmentation | 4.3.1 |
| C9 | R2 − R10(post): per-subject normalization against AdaBN | 4.2.2 |
| C10 | R11 − R10(post): Deep CORAL against AdaBN | 4.2.2 |
| C11 | R2 − R11: per-subject normalization against Deep CORAL (batch differs; state it) | 4.2.2 |
| C12 | R2 − SVM (fixed, 77.7): the deep lead | 4.3.1 |
| C13 | per-seed ensemble − R2: the ensemble's gain over its deep member | 4.4.1 |
| C14 | R12 − R2: 400 ms against 250 ms | 4.4.6 |
| C15 | R9 − R8: channel dropout on SimpleEMGCNN (the null) | 4.3.1 |
| C16 | R13, R17 against R2 (plateau), R15 against R2 (fall) | 4.3.4 |
| C17 | occlusion reduction factor, R2 against R1 and R5 against R4, with its across-realization SD | 4.3.2 |

For each contrast, compute:

1. **Realization-averaged paired test:** per subject, average each arm over its realizations, then Wilcoxon (n = 40), dz, BCa 95% interval, subjects improved.
2. **Seed-level consistency:** the difference of forty-fold means within each shared seed, as mean ± SD, and the count of seeds with the same sign.
3. **Secondary model:** `F1 ~ arm + (1 | subject) + (1 | seed)` via `statsmodels` MixedLM, fixed effect and interval.

**Verdict per contrast:**

- **ESTABLISHED:** the realization-averaged Wilcoxon survives BH within the registered KC-D1 family, **and** the same sign holds in at least 4 of 5 realizations (Tier A) or 3 of 3 (Tier B).
- **AMBIGUOUS:** one of the two conditions holds.
- **NOT ESTABLISHED:** neither holds.

For contrasts the thesis states as nulls (C15, the C16 plateau pairs), report equivalence instead: TOST with bounds of ±1.0 pt on the realization-averaged differences.

**Run-variance deliverable (replaces the 0.5-point band):** the SD of the forty-fold mean for each arm across its realizations; the pooled SD across arms with its degrees of freedom and a chi-square 95% interval; and the per-fold SD distribution.

### D1.6 Headline gate. Owner decision point.

For R2 (0.8395 published), the per-seed ensemble (0.858), R12 (0.860) and the global ResNet-SE+CD baseline (0.772):

| Letter | Condition | Reading |
|---|---|---|
| **H1** | The published value lies within the realization mean ± 2 SD | The headline stays as the version of record; the realization mean ± SD is added in Section 3.6 and Appendix C.7 |
| **H2** | Outside ± 2 SD | **ESCALATE.** Whether to report the realization mean as the headline is Enam's decision. It cascades through about 100 restatements |

### D1.7 Stop conditions and reporting

- C1 or C12 NOT ESTABLISHED: **ESCALATE**, because Finding C's core changes.
- Any other contrast the thesis states as established that lands on AMBIGUOUS or NOT ESTABLISHED: **report, do not halt.** List the affected passages; the wording change belongs to text block TB2.
- Whether the realization-averaged tests **replace** their single-run counterparts in the statistical family is deferred to KC-F1, which computes both versions for Enam.

**Output:** `results_kc23_d1_<arm>_s<seed>/` per run, `results_kc23_d1_stats/` and `D1_VERDICT.md`.

---

## KC-D2. Reliance against trained-in robustness (M2). Analysis only, on D1's instrumented runs.

### D2.1 Problem

Occlusion sets a z-scored channel to zero, a flat line at the subject's mean. That is the input channel dropout produces in training, and an out-of-distribution input for the un-augmented model. A smaller occlusion cost after training on zeroed channels is therefore largely expected by construction.

### D2.2 The discriminating comparison

Use R1 (none), R2 (chandrop), R3 (gainjitter, never zero in training), R4 and R5 (no-SE backbone), across their realizations.

| Measure | What it probes |
|---|---|
| Zeroing occlusion (published measure) | tolerance to the training perturbation of channel dropout |
| Attenuation at α = 0.5 | tolerance to a gain change, the training perturbation of gain jitter |
| **Permutation reliance** | information use, with the marginals kept |

Primary quantities:

- reduction factor of permutation reliance, R2 against R1;
- reduction factor of zeroing occlusion, **R3 against R1**, the key cross-check;
- reduction factor of attenuation, R2 against R1.

Each is computed per realization, with the across-realization mean and SD, plus the paired Wilcoxon on realization-averaged per-subject sums.

### D2.3 Outcomes

| Letter | Condition | Reading |
|---|---|---|
| **O-R** | Zeroing occlusion falls ≥ 3-fold under gain jitter, **and** permutation reliance falls ≥ 2-fold under channel dropout | "Reduced reliance" is supported by two measures that do not share channel dropout's training perturbation. Section 4.3.2 keeps its meaning, with the measures named |
| **O-T** | Zeroing occlusion falls < 1.5-fold under gain jitter, **and** permutation reliance falls < 1.5-fold under channel dropout | Occlusion measured trained-in robustness. Section 4.3.2 is rewritten as sensitivity to electrode loss, "draws on what was there all along" goes, and the abstract's "six-fold" is reworded |
| **O-M** | Anything else | Report per measure. The text uses permutation reliance as the reliance measure and occlusion as the robustness measure |

Residual caveat, to record in the verdict: permuting one channel breaks cross-channel coherence, so it is not perfectly in-distribution either. It is the closest standard probe.

**Output:** `results_kc23_d2_reliance/` and `D2_VERDICT.md`.

---

## KC-D3. Axis against magnitude of the augmentation (moderate item 1). About 11 h.

### D3.1 Problem

The Gaussian-noise null (Section 4.3.1) used SD 0.1 on z-scored envelopes. Channel dropout at p = 0.2 injects a multiplicative SD of 0.40. Section 4.7's "Gaussian noise perturbs along axes that do not match how subjects differ" confounds axis with magnitude. "Multiplicative" is also untested against "per-channel additive", and "per-channel" is untested against "shared across channels".

### D3.2 Arms

ResNet-SE, per-subject normalization, instrumented, at 3 realizations (42 re-run, 7, 123):

| Arm | Mode | Magnitude |
|---|---|---|
| X1 | gaussian | σ = 0.10 (the published setting, now replicated) |
| X2 | gaussian | σ = 0.40 |
| X3 | chanoffset | SD 0.40 |
| X4 | globalgain | SD 0.40 |

Comparators: R1 (none) and R3 (gainjitter SD 0.40), from the same seeds.

### D3.3 Outcomes (realization-averaged; "≥ GJ − 1" means within 1 pt of R3 or above)

| Letter | Condition | Reading |
|---|---|---|
| **A1** | X2 ≤ R1 + 1, and X3 and X4 both < R3 − 1.5 | Per-channel multiplicative is the active axis. Section 4.7's rule stands, now with magnitude-matched evidence |
| **A2** | X2 ≥ R3 − 1 | Noise helps at matched magnitude. The "wrong axis" claim is withdrawn and the claim becomes variance injection in general |
| **A3** | X3 ≥ R3 − 1 | "Multiplicative" is unsupported; the claim becomes per-channel perturbation |
| **A4** | X4 ≥ R3 − 1 | Per-channel independence is not needed |

A2 to A4 can co-occur; report every letter that fires. None of them halts the programme.

**Output:** `results_kc23_d3_*` and `D3_VERDICT.md`.

---

## KC-D4. The channel axis measured in subject-invariance terms (M6). About 19 h.

### D4.1 Problem

On the alignment axis, invariance means subject invariance (probes, MMD). On the channel axis, the thesis measures tolerance to electrode zeroing, and on a different operator family (plain channel dropout) from the one that collapses (mean-preserving). "The same shape on two axes" is therefore a parallel between two different quantities.

### D4.2 Arms

All on ResNet-SE, per-subject normalization, instrumented (occlusion, permutation, embedding probes), at 3 realizations (42 re-run, 7, 123):

- `mpchandrop` at SD {0.40, 0.50, 0.60, 0.80, 1.00};
- `gainjitter` at SD {0.80, 1.00}. Section 4.7 recommends the continuous operator for new work, so whether it has a boundary is a direct question. SD 0.30, 0.40 and 0.50 come from D1.

Reference points: R1 (none) and R13, R2, R17, R15 (plain channel dropout 0.1 to 0.5), all instrumented in D1.

### D4.3 Primary quantities per arm, averaged over folds and then realizations

- **Invariance:** the unseen-subject identity probe (lower means more invariant), and permutation reliance.
- **Class information:** held-out class silhouette and the held-out class probe.
- **Accuracy:** macro-F1.

### D4.4 Outcomes

| Letter | Condition | Reading |
|---|---|---|
| **T1** | Subject-probe accuracy falls with dose along `mpchandrop` (Page trend, Holm < 0.05), F1 peaks and then falls, and within-fold F1 tracks class silhouette (mean Spearman > 0, sign test) but not the subject probe | The channel axis now measures the same kind of invariance as the alignment axis. The through-line may be stated on two axes with a mechanism measured on both |
| **T2** | The subject probe does not fall with dose | The channel axis does not move subject invariance. The through-line is restricted to the alignment axis, and the channel boundary is reported as an augmentation dose-response. Abstract, Sections 4.6 and 5.2 change (text block TB8) |
| **T3** | T1's first two conditions hold but F1 does not track class information | The shape is shared; the mechanism stays "proposed" on the channel axis |

Also report whether `gainjitter` shows a boundary by SD 1.00 (a one-line answer for Section 4.7).

**Output:** `results_kc23_d4_*` and `D4_VERDICT.md`.

---

## KC-D5. ENABL3S deep mechanism replication (M4). About 2 to 3 h.

Resolve the exact published ENABL3S ResNet-SE command from `REPRODUCE.md` and the P-3 plan (`p3_external_cd_stats.py`). Then run, at 5 realizations (42 re-run, 7, 123, 1001, 2026), per-subject normalization, instrumented:

- E1: ResNet-SE none
- E2: ResNet-SE chandrop 0.2, with `--save-proba` for the per-seed ensemble against the ENABL3S SVM probabilities
- E3: ResNet-SE gainjitter 0.40

Record for each: the channel-dropout gain, gain jitter against channel dropout, the occlusion and permutation reduction, **ResNet-SE+CD against the SVM (65.7)** across realizations, and the ensemble against the SVM.

| Letter | Condition | Reading |
|---|---|---|
| **E-R** | A finding's direction holds in the realization average, with ≥ 7 of 10 subjects agreeing | It may be listed as replicated |
| **E-N** | Otherwise | Listed as not replicated |

ResNet-SE+CD against the SVM is reported as a number with its across-seed SD, not as a letter. It feeds the abstract's replication sentence directly.

With n = 10, significance is claimed only for what survives KC-F1.

**Output:** `results_kc23_d5_*` and `D5_VERDICT.md`.

---

## KC-D6. A learned alignment axis that actually moves invariance (M6; the Deep CORAL limitation). Staged, up to about 72 h.

### 6.1 Problem

The through-line is measured on the alignment ladder and only proposed for the channel axis, which KC-D4 addresses. The one in-network, learned alignment the thesis tried, Deep CORAL, never tested the pattern.

A thousandfold weight moved the source-target probe by 0.89 points (Section 4.2.2), and the loss fell mainly because the embedding shrank (mean norm 9.35 to 6.65). The reason is structural:

- the CORAL loss compares covariances, which grow with the square of the feature scale;
- its squared Frobenius form therefore grows with the fourth power;
- so shrinking the features is the cheapest way to lower it.

A test needs a knob that provably moves invariance. The literature offers two.

1. **Subject-adversarial training** (DANN: Ganin et al., 2016). A gradient-reversal adversary predicts the subject from the embedding, and its weight is the knob. It has been used for cross-user EMG (Côté-Allard et al., 2020; Campbell et al., 2021) and for subject invariance in EEG (Özdenizci et al., 2020), and it is the setting Zhao et al. (2019) analyse.
2. **Scale-free Deep CORAL.** The same loss on L2-normalized embeddings, so the shrinking shortcut is gone. This repairs the axis the thesis already has.

The mechanism test comes from Tachet des Combes et al. (2020): conditional (class-wise) alignment avoids the error floor that marginal alignment hits when class proportions differ. The through-line predicts the same, as the deep counterpart of the oracle rung `4o` in KC-C2.

### 6.2 Arms

Common harness for every arm, identical to the D2c Deep CORAL runs so Table 4.6 stays comparable:

- `resnet_se`, channel dropout 0.2, **global normalization** (every learned adaptation in the thesis starts from global), batch 256, epochs 40, patience 7;
- the held-out subject's unlabeled windows forwarded every step (`--target-pass train`);
- all runs instrumented with the D2c `alignment_metrics` **plus** the D0.3 embedding probes. The D0.3 probes use the fold's validation subjects, which neither the classifier nor the adversary ever sees, so the unseen-subject probe measures invariance that generalizes.

| Family | Arms | Knob grid |
|---|---|---|
| **ADV (primary)** | 40-way subject adversary: the 33 training subjects with labels, plus the held-out subject as its own unlabeled class. MLP head 128 → 64 → n_subjects on the penultimate embedding, through a gradient-reversal layer. Adversary learning rate equals the classifier's. Ganin warm-up: λ(p) = λ_max · (2 / (1 + e^(−10p)) − 1), p = training progress | λ_max ∈ {0, 0.03, 0.1, 0.3, 1, 3, 10}. The λ_max = 0 arm keeps the adversary head and the target forward pass; it is the batch-norm-only control that D2e showed is necessary |
| **SFC** | Deep CORAL with `--coral-normalize l2` (the CORAL term computed on unit-norm embeddings) | weight ∈ {0.1, 1, 10, 100, 1000} |
| **ADV-PS** | ADV on the **per-subject-normalized** harness (batch 256): does learned invariance on top of z-scoring help or hurt? | λ_max ∈ {0.1, 1, 10} |
| **ADV-C (mechanism)** | Class-conditional subject adversary: one adversary head per movement class, each fed only windows of its class. Source uses true labels. **The target uses its true labels (oracle, diagnostic only, never deployable)**; every output row carries `oracle=True` | λ_max at the ADV collapse point: the smallest λ_max whose realization-mean F1 is ≥ 2 pts below the ADV peak, plus the next grid value up. If ADV has no collapse, ADV-C is not run and this is reported |
| **ADV-CDAN (optional, deployable)** | Conditional adversary on the multilinear map of embedding and predicted class probabilities (Long et al., 2018); no target labels | same λ_max as ADV-C. Run only if ADV-C lands C-M1 |

### 6.3 Code

- **New script `run_adv_align_loso.py`.**
  - Built by importing from `run_deep_coral_align_loso.py` (`embed`, `alignment_metrics`, `cv_bacc`, loaders, `LABELS`) and `train_cnn_loso.py` (`augment_batch`, `choose_val_subjects`, `class_weights_from_y`).
  - The training loop is copied from `train_deep_coral_logged`, with the CORAL term replaced by the adversary's cross-entropy through gradient reversal.
  - Per-epoch logging: classifier cross-entropy, adversary cross-entropy, adversary training accuracy, and source validation loss.
  - Flags: `--adv-lambda`, `--adv-mode {marginal,classcond,cdan}`, `--oracle-target-labels` (required with `classcond`, refused otherwise), and `--grad-clip`.
- **Scale-free CORAL.** The `--coral-normalize` flag of D0.5.
- **Divergence rule, fixed now.** An arm has diverged if the source validation loss at its best epoch exceeds twice that of the same seed's λ_max = 0 arm, or any loss is NaN. A diverged arm is retried **once** with `--grad-clip 5.0`. If it diverges again, it is recorded as diverged (a result: training destroyed at that strength) and never dropped silently.

### 6.4 Staging

- **Stage 1** runs every family at seed 42 (re-run): about 24 h.
- The **manipulation gate (6.5)** is then evaluated per family on Stage 1 alone. It is a check on the knob, not on the outcome, so reading it early does not bias the outcome test.
- **Stage 2** runs seeds 7 and 123 **only for families that pass**: about 16 h per passing family pair, up to about 48 h.

### 6.5 Gates

**Sanity gate (after Stage 1, ADV at λ_max = 0):** F1 within ±1.5 pts of the D2e weight-0 target-pass arm (83.0%). If it fails, **ESCALATE**, because the harness does not match the published Deep CORAL runs.

**Manipulation gate, per family (the check Deep CORAL never passed):**

| Letter | Condition, across the knob grid (40 folds, seed 42 re-run) | Action |
|---|---|---|
| **G-PASS** | The source-target domain probe falls by ≥ 10 points from the lowest to the highest knob value, **and** the unseen-subject probe falls with a significant Page trend (Holm < 0.05) | Run Stage 2 for this family |
| **G-WEAK** | A fall between 2 and 10 points, or only one of the two meters moves | Run Stage 2, and flag every conclusion from this family as a weak manipulation |
| **G-FAIL** | A fall < 2 points (the D2c threshold) | No Stage 2. Report that this knob, like Deep CORAL's, does not move invariance |

For SFC, also report the embedding norm across weights: it must stay flat by construction. If it does not, the normalization is broken, so stop the family and report.

### 6.6 Primary outcomes (ADV family; realization-averaged)

| Letter | Condition | Reading |
|---|---|---|
| **X1** | Invariance rises monotonically with λ_max (both meters, Page trend), **and** F1 has an interior peak, with the largest non-diverged λ_max ≥ 2 pts below the peak (paired, significant), **and** within-fold F1 tracks the class probe and silhouette but not the invariance meters (the D4 T1 criterion) | A third axis, learned and inside the network, is **measured**. The through-line can be stated on three axes with the mechanism measured on each that passes |
| **X2** | Invariance rises, and F1 is flat or rising up to the largest λ_max | No falling limb within the sweep. On the thesis's own reading, the sweep has not reached the point where class structure goes. Report it. The through-line is not supported on this axis, and Section 4.6 says so |
| **X3** | Invariance rises and F1 falls from the first step, with no rising limb | The falling limb only. Report it as consistent with the pattern's second half |
| **X4** | The shape appears but F1 does not track class information | The shape is shared; the mechanism is not supported on this axis |

**SFC** is read on the same grid: every letter that fires for SFC is reported beside ADV. SFC also answers directly whether the Section 5.3 limitation was only the shrinking shortcut.

**Mechanism test (ADV-C against ADV at the collapse λ_max, realization-averaged, paired):**

| Letter | Condition | Reading |
|---|---|---|
| **C-M1** | ADV-C ≥ ADV peak − 1 pt, and ADV-C − ADV ≥ 2 pts at the collapse λ_max, while ADV-C's within-class subject invariance is at least as high as ADV's | Marginal alignment damages class structure and conditional alignment does not. The mechanism is measured on the learned axis |
| **C-M2** | ADV-C collapses like ADV | The damage is not about class-marginal mixing. Report; the mechanism wording changes |
| **C-M3** | In between | Report the numbers |

**Secondary contrasts (they update Table 4.6 and the Section 5.3 limitation that adversarial objectives were not tested):**

- ADV at its best λ_max against per-subject normalization (R2 of D1), against AdaBN (R10) and against Deep CORAL (R11);
- ADV-PS at each λ_max against the per-subject harness without an adversary. This reads directly on whether z-scoring already sits at or near the turn.

### 6.7 Halting

- Only the sanity gate halts.
- X2 to X4, C-M2 and C-M3 are claim-level results: report them and continue.
- A G-FAIL on both ADV and SFC is itself the result that this axis cannot be tested with these knobs. Report it, and the Section 5.3 limitation is restated with the new evidence.

**Output:** `results_kc23_d6_{adv,sfc,advps,advc,cdan}_<knob>_s<seed>/`, `results_kc23_d6_stats/` and `D6_VERDICT.md`. The verdict ends with a **combined through-line matrix**: axis (ladder from C2, channel from D4, learned from D6) × {invariance measured, shape shown, mechanism measured, replicated on ENABL3S where run}.

---

## Outputs and report-back

Stage verdict files open with their letter. Every new paired test is appended to `kc23_new_tests.csv`. Add a status header to this file per stage, leaving the plan unedited.

Report: the letters, the gates (D0 inertness, D1 reproduction, D1 headline), every ESCALATE, the measured wall-clock per arm, and confirmation that no thesis file was edited and the model of record and the v9 family were untouched.
