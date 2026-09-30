# Post-28-August figures: PLACED, 4 September 2026

All five generated, inspected and inserted. This file is now a record, not a plan.

Theme: the thesis PLOT theme, not the diagram theme. rcParams
`{"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200}` and the established palette
(SVM `#e07b39`, RF `#f0a860`, ResNet-SE / per-subject / primary `#2e6f9e`, CNN / secondary `#8fb8de`, ensemble and
improvement `#3a923a`, baseline and global `#9e9e9e` / `#bbbbbb`, negative `#c44e52`), black bar edges at lw 0.3,
y-grid at alpha 0.25, `set_axisbelow(True)`. Same as `make_deeptier_figs_cd.py` and `make_norm_replication_fig.py`.

Generators: `make_new_result_figs.py` (4.8 and 4.6), `make_new_result_figs2.py` (4.3 and A.22),
`make_new_result_figs3.py` (A.23). All read from `results_*` on disk. The only hardcoded values are the published
per-class figures quoted from Table 4.5 and the published subject-dependent figures quoted from Table 4.7, both
marked as such in the source.

## Where they went

| file | number | section | width |
|---|---|---|---|
| `b8_pooled_vs_blocked.png` | **Figure 4.3** | 4.1.3 | 6.2 in |
| `s1_active_only_per_class.png` | **Figure 4.6** | 4.3.1 | 6.2 in |
| `p10_dose_response.png` | **Figure 4.8** | 4.8.2 | 5.4 in |
| `p9_backbone_reduction.png` | **Figure A.22** | appendix, after A.21 | 6.2 in |
| `b3_per_subject_silhouette.png` | **Figure A.23** | appendix, after A.22 | 6.2 in |

Renumbering was done once, as a two-phase token swap so no cascade could occur: old 4.3 to 4.9 became 4.4, 4.5,
4.7, 4.9, 4.10, 4.11 and 4.12. Figures 4.1 and 4.2 were unaffected, as were the Figure 5.x and Appendix
sequences. In-text cross-references were added in 4.1.3, 4.3.1, 4.8.2 (twice, one to A.22) and 4.13.2, so no new
figure is orphaned. Final sequence: Figure 4.1 to 4.12 and Figure A.1 to A.23, no gaps.

## Defect found during inspection, and fixed

**Figure 4.3's right panel was wrong on its first draft and would have contradicted the section it illustrates.**
It compared the re-run's own pooled-random numbers against its movement-blocked numbers. In that harness the
pooled SVM is 93.7, so the CNN never leads and no ranking flip is visible, while the section's whole claim is that
the CNN leads in Table 4.7 and does not on a common protocol. The panel now plots the published Table 4.7 figures
(SVM 87.4, RF 84.3, CNN 90.4) against the movement-blocked ones (88.8, 88.0, 85.9), labels the two harnesses
explicitly, annotates "CNN first" and "CNN last", and carries the reproduction caveat as an axis label. This is
also the honest way to present that caveat rather than hiding it.

Two smaller fixes: annotation collisions on Figure 4.8 and a "-0" label on Figure 4.6.

## Considered and deliberately not made

- **W-5, depth versus capacity (outcome N).** A null result the thesis reports in one sentence does not earn an
  exhibit; 4.8.1 already carries the two-thirds / one-third split in prose.
- **S-2, sampling-rate invariance.** The result is that 40 values are byte-identical. A plot of that is a plot of
  nothing.
- **W-1, window length.** Already carried by Table 4.19.
- **P-8, channel informativeness.** Feeds P-9's argument rather than standing alone. If it ever needs an exhibit
  it should become a third panel of Figure A.22, not a sixth figure.
