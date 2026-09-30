# B1: run manifests and the canonical-numbers cross-check

Built 3 September 2026 by `build_run_manifest.py`. Output: `RUN_MANIFEST.csv`
(143 directories). No thesis file edited; Section 4.17 not touched.

## The manifest

One row per `results_*` directory, with per-field provenance recorded in
`source_of_config` under the three states of section 1.3:

- **verified** - read from the data itself: `backbone` from `cnn_arch_summary.csv`'s
  `arch` column or the `SVM/RF/LDA` token in a classical subjectwise filename;
  `window_ms` and `feature_set` from the `_w250_ov50_conf60` / `features_ext`
  tokens embedded in filenames the driver wrote; `dataset` from the subject-ID
  range (1..40 SIAT, 156 and 185..194 ENABL3S); `n_subjects` and
  `subject_id_set_hash` from the subjectwise file.
- **documented** - a single citable line in a plan file, `REPRODUCE.md`, or a
  committed `run_*.sh` that names the directory literally AND carries the config
  token on the same line. The file and line are recorded. A nearby mention is not
  documentation and was scored unknown.
- **unknown** - written `UNKNOWN`. Never inferred from the directory name.

### Counts

| | directories |
|---|---|
| total | 143 |
| all six core fields `verified` | **0** |
| weakest field only `documented`, none unknown | 1 |
| carry at least one `UNKNOWN` core field | **142** |

**That 142 is the B1 finding, not a script failure.** No run in the repository
ever wrote a configuration file. `cnn_arch_summary.csv` records `arch` and nothing
else; `train_classical_loso.py` records nothing. `seed` is on a citable line for
almost no directory, `augmentation` for few, `norm_mode` for some. A third party
reproducing from the public repo has to infer the rest from folder names and the
plan markdown, which is exactly the reasoning that produced the backbone defect.

## The cross-check (section 1.4)

Every source directory named in the handoff's Section 4 canonical-numbers table,
checked against what the thesis says about that number.

### The one mismatch

**`results_g3_noaug_instr` and `results_cd_resnet_nose_chandrop`** back the
"84.2 pp to 15.0 pp, 5.6x" occlusion result. The manifest **verifies** both are
`--arch resnet` (SE-free), read from `cnn_arch_summary.csv`. Sections 4.8.2, 5.7
and 6.2 state the number without naming the backbone, and 5.7's sentence
adjacency implies `resnet_se`, the model of record. This is the known defect.

P-9 (four arms, both backbones) settled the substance by measurement: backbone
outcome **S**, the occlusion reduction is 6.12x on `resnet_se` against 6.48x on
`resnet`, a difference well inside the A1/A2 re-run spread. **The number does not
have to move; the three passages need the backbone named.** That wording change
is in the write-up queue, not this plan.

### Is that the only one? Yes, for backbone/dataset contradictions.

No other canonical number sits on a directory whose *verifiable* config
contradicts the thesis. `results_cnn_aug_resnet_se_chandrop` verifies as
`resnet_se`; `results_w4_gainjitter` as `resnet` (matching the thesis, which
sites gain jitter on the SE-free backbone); `results_cd_rate_p0.1..0.5` as
`resnet_se`; the classical dirs as classical. Datasets all verify as SIAT.

### Unverifiable (not a contradiction) on canonical dirs

These back canonical numbers but record no `arch` anywhere, so the backbone
cannot be machine-verified. The directory name and the thesis are mutually
consistent; nobody can confirm it from data:

- `results_ensemble_v2_chandrop` (85.8% headline) - also has the B4.4 defect: its
  `ensemble_v2_subjectwise.csv` has no subject column, so its 40 rows are
  positional and `n_subjects` reads as 0.
- `results_causal_ensemble` (81.7%)
- `results_variance_decomposition` (alignment ladder, subject probe, silhouette, ICC)
- `results_within_subject` (label-budget crossover)
- `results_window_ablation` (W-1)
- `results_cnn_calibration_chandrop_resnet_se` (calibration +3.6 pp)

And nearly every canonical dir has `UNKNOWN` for at least one of `norm_mode`,
`augmentation`, `seed`. Same root cause; section 1.5 fixes it forward.

## Section 1.5: stop future recurrence

`run_config_dump.py` added. `run_cnn_arch_loso.py` and `train_classical_loso.py`
now call `dump_run_config()` right after the output directory is created, writing
`run_config.json` with `vars(args)`, resolved input paths, the git commit, and
the Python / torch / cuDNN / numpy / scipy / sklearn / pandas versions.

Follows the `gainjitter` precedent: additive, guarded (never raises into the
run), and RNG-inert with an inline assertion (torch RNG state and numpy RNG state
both asserted unchanged across the call). Verified by a unit test and a 2-epoch
CNN smoke; `p5p6_inertness.py --check` still passes on all six augment modes and
on resnet / resnet_se initialisation.
