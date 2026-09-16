# emg-stride-thesis

Code, results and reproduction path for an MSc thesis on **cross-subject generalization in
surface electromyography (sEMG) based lower-limb movement classification**.

**Author:** Enam Ahorlu, MSc Computer Science, University of Ghana
**Dataset:** SIAT-LLMD (40 subjects, 9 channels, 1920 Hz), externally replicated on ENABL3S (10 subjects, 7 channels, 1000 Hz)
**Reproduction:** every number, table and figure in the thesis is mapped to its script and output in [`REPRODUCE.md`](REPRODUCE.md)

---

## The question

A model trained on one group of people typically loses 10 to 25 percentage points of accuracy on a
new user, and the conventional fix is a labeled calibration session from every user. This work asks
how far a **data-centric** pipeline can close that gap **without learned domain adaptation and
without labeled per-user calibration**, under leave-one-subject-out (LOSO) evaluation on four
movement classes: level walking (WAK), ascending stairs (UPS), descending stairs (DNS) and
sit-to-stand (STDUP).

The answer is that the cheapest correction available is also the best one, and that the reason it
wins is the reason more thorough corrections lose.

---

## What the work found

### 1. Per-subject normalization dominates, and it is a location shift

Per-subject z-scoring, computed from the held-out subject's own **unlabeled** windows inside the
LOSO loop, improves macro-F1 by 5 to 7 pp across every model family on SIAT and by 10 to 17 pp on
ENABL3S. It outperforms CORAL and adaptive batch normalization, and is not separable from Deep
CORAL, whose alignment weight was swept across three orders of magnitude without improving it on
this backbone. Because it draws on the
target subject's own data it is transductive, and is best read not as an alternative to domain
adaptation but as its simplest label-free member.

An alignment ladder explains why. Each rung is a stronger alignment operator carried through the
identical LOSO protocol:

| Rung | Operator | Discrepancy removed (MMD) | Subject probe | Class silhouette | SVM LOSO F1 |
|---|---|---|---|---|---|
| 0 | Global z-score (baseline) | 0 | 0.777 | 0.008 | 0.709 |
| 1 | Mean centering only | 42.4% | 0.043 | 0.019 | 0.748 |
| 2 | Scale only | -26.7% | 0.909 | 0.008 | 0.719 |
| 3 | **Mean and scale (per-subject z)** | 54.4% | **0.024** | **0.023** | **0.777** |
| 4 | Full covariance whitening | 73.0% | 0.012 | -0.006 | 0.675 |

Mean-centering alone drives the subject-identity probe from 0.777 to 0.043 against a chance floor of
0.025, so the subject structure a **linear** probe can decode is overwhelmingly a difference in
**location**. That qualifier carries weight. Nonlinear probes fit on the same rung outputs still
identify the subject at 0.9999, and a linear probe reaches 0.608 once restricted to a single
movement, so what the operation removes is the component of the subject shift common across
movements rather than subject identity itself. Full whitening
removes more subject variance still, and is 10.1 pp **worse**, because it strips class-discriminative
structure along with the nuisance. The ordering replicates at a 400 ms window, where the
over-alignment cost is 12.1 pp.

### 2. Channel dropout is what makes the deep model competitive

Zeroing a random subset of electrode channels during training is worth **+5.7 pp**, taking a
squeeze-and-excitation residual network to 84.0% LOSO macro-F1. It works through per-channel
**multiplicative variance** rather than through removal: a random gain of matched variance does the
same work, and swept in a mean-preserving form it too runs flat and then collapses.

### 3. The through-line

Both interventions induce invariance. Both help to a point, then destroy the structure the classifier
depends on. **More invariance is not better**, demonstrated twice by mechanisms that share no
machinery.

---

## Headline results

SIAT-LLMD, LOSO macro-F1, n = 40:

| Model | Global norm | Per-subject norm | Δ (pp) |
|---|---|---|---|
| SVM (RBF, Freq-72) | 0.708 | 0.777 | +6.9 |
| Random Forest | 0.722 | 0.773 | +5.1 |
| SimpleEMGCNN | 0.682 | 0.754 | +7.2 |
| ResNet-SE + channel dropout | 0.772 | 0.840 | +6.7 |
| **Soft-vote ensemble (SVM + ResNet-SE+CD)** | - | **0.858** | - |

ENABL3S, LOSO macro-F1, n = 10, pipeline re-run unchanged:

| Model | Global norm | Per-subject norm | Δ (pp) |
|---|---|---|---|
| SVM | 0.554 | 0.657 | +10.3 |
| Random Forest | 0.525 | 0.636 | +11.1 |
| CNN | 0.387 | 0.556 | +16.9 |

**The deployable number is 81.7%, not 85.8%.** Per-subject normalization needs the subject's whole
session, so 85.8% is an offline upper bound. Measured under a strictly causal calibration buffer
covering the four movements, the same locked pipeline retains 81.7% at roughly 4 ms of compute per
window. That is the figure a real-time system should be judged against.

Two further results worth knowing before reading the code:

- **Label budget.** Where labeled data can be collected, a subject-specific model does not match the
  label-free pipeline until roughly 50 labeled windows per class, and how they are gathered matters
  as much as how many.
- **Window length.** 250 ms is the operating point. A 400 ms window is genuinely more accurate
  (+2.5 pp SVM, +2.0 pp deep model) but lifts both normalization conditions in parallel, so it buys
  signal quality rather than cross-subject transfer, and costs 150 ms of decision delay against
  ~4 ms of compute. The choice is documented rather than inherited.

---

## Statistical protocol

Every comparison is a subject-paired Wilcoxon signed-rank test with a paired Cohen's d and a BCa
confidence interval, at n = 40 (SIAT) or n = 10 (ENABL3S).

Every paired comparison the thesis reports is pooled into **one Benjamini-Hochberg family of 229
tests**, recomputed from source each time it grew rather than patched. 154 survive at a 5%
false-discovery rate. The 75 that do not are, with two exceptions the thesis names, results already
reported as null or as honest negatives. `recompute_unified_fdr_v8.py` builds the family of record;
every earlier version is kept so the growth of the family is inspectable, and the two contrasts that
lost significance as it grew are reported rather than absorbed.

---

## Datasets

**SIAT-LLMD** (primary). 40 healthy adults, 9 sEMG channels at 1920 Hz. The published feature
matrices were extracted with `extract_features.py`'s 2000 Hz default left in place, which changes
nothing: the rate enters as one multiplicative constant on the two spectral features that both
normalization schemes divide out, and a rebuild at 1920 Hz reproduces the SVM figure to every
digit stored.
Wei, W., Tan, F., Zhang, H., Mao, H., Fu, M., Samuel, O. W., & Li, G. (2023). Surface electromyogram,
kinematic, and kinetic dataset of lower limb walking for movement intent recognition. *Scientific
Data*, 10, Article 358. https://doi.org/10.1038/s41597-023-02263-3

**ENABL3S** (external replication). 10 able-bodied adults, 7 right-leg channels at 1000 Hz, different
laboratory and different hardware.
Hu, B., Rouse, E., & Hargrove, L. (2018). Benchmark datasets for bilateral lower-limb neuromechanical
signals from wearable sensors during unassisted locomotion in able-bodied individuals. *Frontiers in
Robotics and AI*, 5, 14. https://doi.org/10.3389/frobt.2018.00014

Neither raw deposit is redistributed here. Both are gitignored and must be obtained from the original
authors. The **derived feature matrices are tracked**, so most of the pipeline can be re-run without
the raw recordings.

---

## Pipeline

```
preprocess_emg.py     bandpass 20-450 Hz (4th-order Butterworth, zero-phase) -> rectify
                      -> 50 ms moving-average envelope -> 250 ms windows at 50% overlap
                      -> 60% label-purity rule -> windowed .npz
extract_features.py   Base-36 / Extended-54 / Freq-72 / Combined-81 feature matrices
                      Freq-72 (primary) = MAV, RMS, WL, ZC, WAMP, MNF, MDF, spectral power x 9 channels
train_classical_loso.py   SVM / RF nested LOSO, inner 5-fold GroupKFold, GridSearchCV on macro-F1
run_cnn_arch_loso.py      SimpleEMGCNN / ResNet-SE LOSO, --augmentation {none,gaussian,chandrop,timemask}
ensemble_v2_combine.py    soft, weighted-soft, hard and stacked combiners from saved probabilities
```

Normalization is applied **inside** the LOSO loop and never uses held-out labels, which is what makes
the transductive framing honest rather than a leak.

Fixed seed 42 throughout (NumPy, scikit-learn, PyTorch); CNN validation splits are seeded per fold as
`seed + held-out-subject-id`. Seed stability was checked by re-running the full pipeline under
additional seeds.

---

## Repository layout

| Path | Contents |
|---|---|
| `*.py` at root | preprocessing, feature extraction, trainers, ablation drivers, statistics, figure scripts |
| `features_out/`, `features_out_ext/` | derived feature matrices, tracked (SIAT and ENABL3S) |
| `results_*/` | one directory per experimental arm, each with per-subject metrics and saved predictions |
| `report_figs/` | publication figures and CSV summary tables |
| `EXPERIMENT_PLAN_*.md` | pre-registered protocols, written before the runs they govern |
| `*_REPORT.md`, `RUN_QUEUE*.md` | outcome reports and run logs |
| `REPRODUCE.md` | **the reproduction map**: every thesis number to its script and output |
| `_STRUCTURE.md` | fuller directory index |

Several experiments were pre-registered: the protocol, the endpoints and the decision thresholds were
written into an `EXPERIMENT_PLAN_*.md` before any run began. Where a pre-registered rule produced an
uncomfortable answer, the rule was followed and the discomfort reported.

---

## Reproducing

Prerequisites: Python 3.10+, `numpy scipy pandas scikit-learn torch matplotlib seaborn`
(see `requirements.txt`).

```bash
python preprocess_emg.py            # raw recordings -> windowed .npz
python extract_features.py          # windowed .npz -> feature matrices
python train_classical_loso.py --norm-mode per_subject   # SVM / RF LOSO
python run_cnn_arch_loso.py --arch resnet_se --augmentation chandrop --norm-mode per_subject
python ensemble_v2_combine.py       # combiner comparison
python recompute_unified_fdr_v8.py  # whole-thesis correction family
```

`REPRODUCE.md` is the authority and carries the exact invocation for every reported result,
including the ablations, the external replication, the causal deployability arms and the mechanism
programme. Start there rather than here.

---

## Archive and citation

The repository is archived on Zenodo. The concept identifier
https://doi.org/10.5281/zenodo.22179742 always resolves to the most recent version and is the
stable thing to cite. The current version of record is **v1.2.0** (16 September 2026) at
https://doi.org/10.5281/zenodo.22801920. Two snapshots precede it: v1.1.0 (10 September 2026)
at https://doi.org/10.5281/zenodo.22684107 and v1.0.0 (30 August 2026) at
https://doi.org/10.5281/zenodo.22179743. Both predate the September remediation programme and
therefore carry an earlier correction family than the one the thesis reports.

```
Ahorlu, E. (2026). Cross-subject generalization in surface electromyography-based
lower-limb movement classification. MSc thesis, University of Ghana.
```

---

## Scope and limits

Both cohorts are able-bodied, the external replication is small at n = 10, and every recording is
laboratory-collected under supervision. Clinical generalization to amputee or impaired users is
untested, and the cross-subject gap is **reduced rather than closed**. The results here are a
benchmark and a set of mechanisms, not a deployed controller.
