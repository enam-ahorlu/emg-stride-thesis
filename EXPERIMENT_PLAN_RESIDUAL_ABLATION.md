# Experiment plan: is the skip connection what makes channel dropout work? (W-3)

> ## STATUS, 2 September 2026: RAN. BOUNDARY CASE. NO PRE-REGISTERED LETTER APPLIES.
>
> §4 trainability check **PASSED**: no-skip no-aug 0.7718 against plain residual 0.7600, +1.18 pp,
> **p = 0.24, not significant.** The degradation problem did not appear at this depth and the two
> baselines are interchangeable, so the comparison is clean.
>
> | | Δ | 95% BCa | p | d |
> |---|---|---|---|---|
> | CD on plain residual | +6.51 pp | [+5.00, +7.98] | 2.3e-9 | +1.35 |
> | CD on no-skip | +4.55 pp | [+3.29, +5.95] | 5.1e-8 | +1.04 |
> | interaction | +1.96 pp | [+0.22, +3.49] | 0.011 | +0.36 |
>
> **Why no letter, and the defect is in the grid, not the data.** §5 wrote outcome D as "gain ≥ 4.0 pp
> **and** interaction not significant", which silently assumed those co-occur. They did not: the gain
> clears 4.0 pp (so not M, capped there) while the interaction is significant (so not D), and it is far
> above 2.0 pp (so not K). The grid had no cell for "not necessary, but contributory". That is a
> pre-registration error to record, not a result to force.
>
> **Resolution taken, D-leaning, framed as a decomposition.** What D asserts, that channel dropout
> works without skip connections, is established at d = 1.04, p = 5e-8. What M asserts rests on the
> least robust number in the set: d = 0.36 against d > 1.0 for the effects it decomposes, p = 0.011
> unadjusted, a CI lower bound of +0.22, and it attenuates to +1.29 pp with a CI of [-0.42, +2.49]
> under a post-hoc headroom correction (the rank test still reads p = 0.005; the thesis reports the
> disagreement rather than choosing). Skip connections are reported as **helpful but not necessary**,
> a minority contributor rather than an established component.
>
> **The decomposition is the deliverable.** Across three backbones, per subject:
> depth and capacity contribute **+4.24 pp** (CI [+3.07, +5.50], p = 4.5e-8, d = 1.07, **68%**) and
> skip connections **+1.96 pp** (**32%**) of the +6.21 pp architecture-dependence. This is the
> augmentation's analogue of §4.13.1's location-versus-spread decomposition of the between-subject shift.
>
> **Still confounded:** depth against capacity, and both against kernel width and the stem. §5.13 now
> proposes the separating experiment (a width sweep at fixed depth, or a depth sweep at fixed parameter
> count) in place of the skip ablation, which is done.
>
> §4.8.1 and §§5.7, 5.13 are written. `verify_section_4_8_1.py` regenerates every figure above.
>
> **Everything below is the plan as written on 2 September, kept unedited as the pre-registration.**


**Status:** ready to run. Written 2 September 2026, after W-2 Stage G1 returned Outcome R.
**Cost:** 2 runs, about 1.3 hours GPU, serial. This is measured, not estimated: the two
G1 runs on the same hardware took 25.6 min (no augmentation) and 52.5 min (channel dropout),
per `_run_logs/w2_g1.log`. Channel dropout roughly doubles wall-clock because it delays
overfitting and so pushes early stopping later.
**Blocks:** nothing. Section 4.8.1 of the thesis is already written and correct without this;
this experiment can only make its final claim narrower and stronger.

---

## 0. Read this first

1. **G1 has already falsified the squeeze-and-excitation hypothesis.** Do not re-test it.
   Channel dropout is worth +6.51 pp on the plain residual network and +5.73 pp with the SE
   block present; the per-subject interaction contrast is -0.78 pp, p = 0.221. SE is not the
   mechanism. This experiment tests the next candidate, not that one.
2. **No model weights are saved anywhere.** Both runs below are fresh 40-fold LOSO runs.
3. **The comparator is `results_cd_resnet_nose_chandrop` and `results_cd_resnet_noaug_repro`,
   both generated during G1 from the current code state.** Do not compare against the published
   0.8395 or 0.7563: those are from an earlier code era, and mixing eras lets drift masquerade
   as an architecture effect.
4. **Do not edit `MSc Thesis.docx` or any chapter file.** Section 4.8.1 already states the
   confound this experiment addresses, and it states it correctly. The wording of any update
   is Enam's once the numbers are in.

---

## 1. The question

Section 4.8.1 of the thesis currently ends on an honest confound:

> What still separates the two backbones is not one difference but three. SimpleEMGCNN and the
> residual network differ in skip connections, in depth, and in capacity by a factor of nine
> (59k against 546k parameters), and the ablation above separates none of these from the others.

Channel dropout is worth **+0.30 pp** on SimpleEMGCNN (p = 0.82, a null) and **+6.51 pp** on the
plain residual network (p < 10^-8). Something about the deeper backbone lets the augmentation
work. Three candidates remain and they are fully entangled.

| Candidate | Isolated by | Status |
|---|---|---|
| Squeeze-and-excitation | W-2 G1 | **Ruled out.** Interaction -0.78 pp, p = 0.221 |
| Skip connections | **this experiment** | untested |
| Depth (6 residual blocks vs 3 conv blocks) | not isolated here | untested |
| Capacity (546k vs 59k parameters) | not isolated here | untested |

This experiment removes the identity addition while holding depth, width, kernel size, training
schedule, seed and augmentation parameters fixed. It therefore separates **one** of the three,
and the honest write-up of a null result names the two that remain.

---

## 2. Stage R0 — the code change

In `cnn_architectures.py`:

```python
class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1, use_se=True, p=0.1, use_residual=True):
        ...
        self.use_residual = use_residual
        self.down = None
        if use_residual and (stride != 1 or cin != cout):
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if not self.use_residual:
            return self.drop(torch.relu(out))
        idn = x if self.down is None else self.down(x)
        return self.drop(torch.relu(out + idn))
```

Thread `use_residual` through `EMGResNet1D.__init__` and add to `build_model`:

```python
if arch == "resnet_nores":
    return EMGResNet1D(in_ch, n_classes, use_se=False, use_residual=False)
```

Add `resnet_nores` to `run_cnn_arch_loso.py`'s `--arch` choices. Nothing else changes.

### 2.1 Inertness assertion (mandatory, before any run)

The change must be a no-op when `use_residual=True`. Reproducing a number is not sufficient
evidence of that, because the pipeline's run-to-run SD is 0.47 pp. Assert on state instead:

```python
import torch, numpy as np, random
from cnn_architectures import build_model

def fingerprint(arch, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = build_model(arch, 9, 4)
    st = torch.get_rng_state().numpy().tobytes()
    ps = b"".join(p.detach().cpu().numpy().tobytes() for p in m.parameters())
    return st, ps, sum(p.numel() for p in m.parameters())

# run once on the pre-change code, save; run again after the change; compare
```

Both the post-construction torch RNG state **and** the concatenated initial parameter bytes must
be identical before and after the edit, for `arch="resnet"` and for `arch="resnet_se"`. If they
are not, the edit has consumed or reordered random draws and every downstream comparison is
contaminated. Fix it before running anything.

### 2.2 Report the parameter counts

Print and record `count_params` for `resnet`, `resnet_se` and `resnet_nores`. Removing the
identity path also removes the two 1x1 projection convolutions that exist only to make the shapes
match for the addition, worth roughly 10.6k parameters, so the no-skip network is about **1.9%
smaller** than the plain residual network. State the exact figures in the report. That 1.9% is
immaterial against the ninefold gap this experiment is trying to decompose, but it should be on
the record rather than discovered by a reader.

---

## 3. Stage R1 — the two runs

Both use `--arch resnet_nores`, which is SE-free, matching the G1 arm exactly.

```
"<PY>" -u run_cnn_arch_loso.py --npz <the same npz G1 used> --meta <the same meta G1 used> \
  --arch resnet_nores --augmentation none \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 \
  --out results_w3_nores_noaug --resume

"<PY>" -u run_cnn_arch_loso.py --npz <same> --meta <same> \
  --arch resnet_nores --augmentation chandrop --aug-chandrop-p 0.2 \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 \
  --out results_w3_nores_chandrop --resume
```

**Take the npz and meta paths from the G1 invocation, not from this file.** Every flag other than
`--arch` must match `results_cd_resnet_noaug_repro` and `results_cd_resnet_nose_chandrop`
respectively. Echo the four commands side by side in the report so the match can be checked.

Run serially. The GPU does not benefit from concurrency here. Use `--resume`; if a run is
interrupted, check `cnn_arch_subjectwise.csv` for duplicated subject rows before trusting it,
which is the failure mode a resume produces.

---

## 4. The trainability check, which comes first

**Report the no-augmentation baseline before anything else, and stop if it has collapsed.**

Six stacked convolutional blocks without skip connections is precisely the regime where the
degradation problem that motivated ResNets appears. If `results_w3_nores_noaug` lands far below
the plain residual network's 0.7600, the experiment has not compared two healthy networks; it has
compared a healthy one against one that failed to optimize.

| No-aug baseline | Reading | What to do |
|---|---|---|
| within 3.0 pp of 0.7600 | both networks trained | proceed to §5, the comparison is clean |
| 3.0 to 8.0 pp below | partial degradation | proceed, but the §5 verdict is **conditional**: a larger channel-dropout gain on a weaker baseline is partly headroom, and must be reported as such |
| more than 8.0 pp below, or below 0.68 | the no-skip network did not train | **stop.** The result says nothing about channel dropout. Report the collapse, which is itself a clean finding about the architecture, and do not run the interaction test |

The 3.0 pp figure is the same threshold W-2 §4.3 uses, set at roughly six times the measured
run-to-run SD of 0.47 pp from R-1 rather than at a round number.

---

## 5. The decision rule, pre-registered

Let, per subject:

```
delta_res   = f1(resnet + chandrop)        - f1(resnet, no aug)          # measured: +6.51 pp
delta_nores = f1(resnet_nores + chandrop)  - f1(resnet_nores, no aug)    # this experiment
interaction = delta_res - delta_nores                                     # paired across 40
```

Test `interaction` with a paired Wilcoxon signed-rank, report paired Cohen's d and a BCa
bootstrap 95% CI (10,000 resamples, seed 42). This is a family of one; no Holm correction applies.

| Outcome | Condition | What it means | For the thesis |
|---|---|---|---|
| **K. Skip-dependent** | mean `delta_nores` < 2.0 pp **and** interaction significant and positive | The identity path is what lets channel dropout work | §4.8.1's final paragraph narrows from three candidates to one. The transferable recommendation becomes "use channel dropout on a residual backbone" |
| **M. Mixed** | mean `delta_nores` between 2.0 and 4.0 pp | Skip connections contribute but do not account for the effect | §4.8.1 apportions: skips carry part of it, depth and capacity carry the rest |
| **D. Not the skips** | mean `delta_nores` >= 4.0 pp **and** interaction not significant | Skip connections are not the enabler either | §4.8.1's confound narrows from three candidates to two, depth and capacity, and the recommendation becomes "use channel dropout on a network with the capacity to exploit it". This is a real result, not a failure |

**Outcome D is the most likely of the three and is worth having.** Two of three candidates ruled
out by direct ablation is a stronger position than the current one, and it is honest about what
remains. Do not treat a null interaction as a wasted 1.3 hours.

Under every outcome, the verdict inherits the §4 caveat if the baseline degraded.

---

## 6. Statistics

Match the thesis convention exactly:

- Paired Wilcoxon signed-rank on the 40 held-out subjects.
- Paired Cohen's d, computed as mean difference over the SD of the differences.
- BCa bootstrap 95% CI, 10,000 resamples, seed 42.
- Differences of differences formed **per subject** before testing, never as a difference of means.
- Holm correction within families only. This experiment is one family of one test.

`window_ablation_stats.py` already implements all four correctly and its BCa, d and Holm were
verified by hand during W-1. Reuse it rather than rewriting the estimators.

---

## 7. Reproduction gates

Before the comparison is read:

1. Both new runs report 40 distinct subjects in `cnn_arch_subjectwise.csv`, no duplicates.
2. The subject index sets of all four runs are identical.
3. The two G1 runs are re-read from disk, not from any number in this file or in the thesis.
4. The inertness assertion of §2.1 passed on both `resnet` and `resnet_se`.

If gate 4 fails, nothing else in this document is interpretable.

---

## 8. Downstream impact

**This experiment changes no existing number in the thesis.** It adds one or two rows to the
architecture comparison and rewrites the final paragraph of §4.8.1 and one sentence of §5.13. The
deep model of record, the ensemble, the causal analysis, the external validation and the headline
85.8% are all untouched under every outcome, because no configuration tested here is a candidate
to replace ResNet-SE+CD. A no-skip network that happened to win would be a finding about
mechanism, not a new model of record, and promoting it would require redoing eight downstream
result sets for no accuracy gain.

---

## 9. What to report

1. The outcome letter, K, M or D, in the first line.
2. The §4 trainability check: the no-augmentation baseline, its distance from 0.7600, and which
   row of that table applies. This comes before the interaction result, not after it.
3. The four run means side by side, with the four full commands, so the flag match is checkable.
4. `delta_nores` with CI, p and d; the interaction contrast with CI, p and d.
5. The three parameter counts from §2.2.
6. Wall-clock per run, against the G1 measurements of 25.6 min and 52.5 min. A materially
   longer no-skip run is itself informative: it would mean the network is still improving at the
   patience horizon, which bears on the §4 trainability question.
7. One sentence on what remains confounded after this experiment.
