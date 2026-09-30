# Experiment plan: is it removal, or is it per-channel variation? (W-4)

**Status:** ready to run. Written 2 September 2026.
**Cost:** one 40-fold run, about 50 minutes GPU. One conditional second run.
**Why it exists:** channel dropout has never been tested against a real rival.

---

## 0. Read this first

1. **This is the missing control, not another augmentation sweep.** Section 4.8 tested channel
   dropout against Gaussian noise and time masking. Section 5.7 then argues, correctly, that those
   two "perturb the signal along axes that do not correspond to how subjects actually differ."
   By the thesis's own reasoning they are not rivals. Nothing channel-structured has ever been put
   against channel dropout.
2. **Run W-2 Stage G3 before making the code change below.** G3's comparison is against the
   existing `results_cd_resnet_nose_chandrop`, and running it on unchanged code means that
   comparison needs no inertness argument at all.
3. **Do not edit `MSc Thesis.docx` or any chapter file.**

---

## 1. The question

Section 5.7 makes a specific claim about why channel dropout works, and it names three failure
modes: electrodes re-sited between users so a channel does not overlie the same muscle belly,
per-channel gain varying with skin preparation and electrode-skin impedance, and channels
degrading or failing outright through liftoff or sweat.

Only the third is *removal*. The first two are per-channel **gain** variation, and channel dropout
simulates them only in the degenerate limit of gain zero. So the paragraph's argument covers a
broader class of corruption than the augmentation actually applies, and the experiment cannot
currently tell which half is doing the work.

**Gain jitter separates them.** It multiplies each channel of each training sample by a random
gain that is never zero. If it recovers most of channel dropout's +6.51 pp, the mechanism is
channel-structured multiplicative perturbation in general and §5.7's liftoff-and-re-siting framing
is too specific. If it recovers little, then removal is doing the work, and §5.7 can say so in its
sharpest form: it is not enough to vary a channel, the network has to learn to proceed without it.

Either answer improves the section. There is no outcome in which the current text is the right text.

---

## 2. The code change

In `train_cnn_loso.py`'s `augment_batch`, and identically in `run_cnn_arch_loso.py` (which carries
its own copy), add one branch **after** the existing ones, touching nothing above it:

```python
    if mode == "gainjitter":
        # Per sample, per channel multiplicative gain. Uniform(1-a, 1+a) has
        # SD = a/sqrt(3), so a = gain_sd*sqrt(3) matches a target SD exactly.
        a = gain_sd * (3.0 ** 0.5)
        g = 1.0 + (torch.rand(N, C, 1, device=Xb.device) * 2.0 - 1.0) * a
        Xb = Xb * g
```

Add `gain_sd: float` to the signature with default 0.4, `"gainjitter"` to both `--augment` /
`--augmentation` choice lists, and `--aug-gain-sd` (default 0.4) threaded through to the call.

### 2.1 Strength, pre-registered before any run

**Match the injected per-channel multiplicative variance.** Channel dropout multiplies a channel by
Bernoulli(1 - p) with p = 0.2, whose SD is sqrt(p(1-p)) = **0.4**. Gain jitter is therefore run at
`gain_sd = 0.4`, giving Uniform(0.307, 1.693): the same second moment, and never zero.

Two properties of this choice must be stated in the write-up rather than left implicit. The means
differ deliberately — channel dropout attenuates on average (mean multiplier 0.8) while gain jitter
does not (mean 1.0) — because a matched mean would require systematic attenuation, which
BatchNorm partly absorbs and which would confound the comparison with a scale effect. And the
distributions differ in shape by construction, since that difference *is* the hypothesis: one
reaches zero and the other cannot.

### 2.2 Inertness assertion, mandatory before any run

The change must be a no-op for every existing mode. Reproducing a number is not evidence, the
run-to-run SD being 0.47 pp. Assert on state:

```python
import torch
from run_cnn_arch_loso import augment_batch   # and again from train_cnn_loso

for mode in ["none", "gaussian", "chandrop", "timemask", "combined"]:
    torch.manual_seed(42)
    X = torch.randn(8, 9, 500)
    torch.manual_seed(7)
    out = augment_batch(X.clone(), mode=mode, sigma=0.1, chandrop_p=0.2, mask_frac=0.15)
    state = torch.get_rng_state().numpy().tobytes()
    # compare `out` bytes and `state` against the values captured on the pre-change code
```

Both the output tensor bytes and the post-call RNG state must be identical before and after the
edit, for all five existing modes, in **both** files. If they are not, the new branch is consuming
randomness it should not and every prior run becomes incomparable.

---

## 3. The run

Matched to `results_cd_resnet_noaug_repro` and `results_cd_resnet_nose_chandrop` flag for flag
except `--augmentation`:

```
"<PY>" -u run_cnn_arch_loso.py --npz <as G1> --meta <as G1> \
  --arch resnet --augmentation gainjitter --aug-gain-sd 0.4 \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 \
  --norm-mode per_subject --resume --out results_w4_gainjitter
```

`--arch resnet`, SE-free, matching G1. Not `resnet_se`: SE has a main effect of +2.23 pp that
would sit inside the contrast for no reason.

---

## 4. The decision rule, pre-registered

Per subject, against the shared `results_cd_resnet_noaug_repro` baseline:

```
delta_cd   = f1(chandrop)   - f1(no aug)      # measured: +6.51 pp
delta_gain = f1(gainjitter) - f1(no aug)      # this experiment
contrast   = delta_cd - delta_gain            # paired across the 40
```

Paired Wilcoxon, paired Cohen's d, BCa 95% CI at 10,000 resamples, seed 42. Family of one.
The 2.0 and 4.0 pp thresholds are the same grid used by W-2 §1.4 and W-3 §5, deliberately.

| Outcome | Condition | What it means | For the thesis |
|---|---|---|---|
| **P. Perturbation** | `delta_gain` >= 4.0 pp **and** contrast not significant | Any channel-structured multiplicative perturbation does the work; zeroing is not special | §5.7 must be reworded. The recommendation broadens usefully: match the augmentation to the sensor's *variability*, not specifically to its failures |
| **M. Mixed** | `delta_gain` 2.0 to 4.0 pp | Both contribute | §5.7 apportions. **Run the conditional second arm** at `gain_sd = 0.2` to see whether gain jitter has a dose-response of its own |
| **R. Removal** | `delta_gain` < 2.0 pp **and** contrast significant and positive | Varying a channel is not enough; the network has to learn to proceed without one | §5.7 gets its strongest form, and the sensor-failure framing is earned rather than asserted |

**Outcome P is not a bad result and must not be argued away.** It would broaden §5.7's
recommendation from a claim about electrode failure to a claim about sensor variability, which is
more useful to a reader with a different modality, and it would still leave channel dropout as the
operating point on grounds of simplicity.

---

## 5. Reproduction gates

1. The inertness assertion of §2.2 passed for all five existing modes in both files.
2. `results_w4_gainjitter` reports 40 distinct subjects, no duplicates from a resume.
3. Its subject index matches `results_cd_resnet_noaug_repro` exactly.
4. The printed `[aug]` line confirms `augmentation=gainjitter` and `gain_sd=0.4`.

If gate 1 fails, nothing else here is interpretable and the prior runs are contaminated too.

---

## 6. A by-product worth recording

After G3, three no-augmentation `--arch resnet` runs will exist at identical settings:
`results_cd_resnet_noaug_repro`, `results_g3_noaug_instr`, and any resume-fresh repeat. R-1
measured run-to-run SD on `resnet_se` at 250 ms under *global* normalization; these give the same
quantity on `resnet` under per-subject normalization, free. Report their mean, SD and range.

---

## 7. What to report

1. The outcome letter, P, M or R, in the first line.
2. The four gates of §5, before any result.
3. `delta_gain` with CI, p and d; the contrast with CI, p and d; `delta_cd` re-read from disk.
4. The three no-aug run means from §6, with their SD.
5. Wall-clock, against G1's 52.5 min for the equivalent channel-dropout run.
6. One sentence on what this does **not** establish.
