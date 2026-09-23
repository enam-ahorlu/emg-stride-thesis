# KC-C1 verdict: nested selection audit

**Outcome letter: E**

Selection nested inside the outer LOSO loop, over the existing predictions (no
retraining). Caveat, recorded not fixed: the other folds' models were trained on
data that includes subject s, so this nests the SELECTION but not the TRAINING.
Fully nested retraining would need 40x39 trainings per candidate and is out of
scope.

## Level E (24 ensemble configurations, Table A.9)

- Published: `SVM+RESNET_SE [soft]` = 0.8580
- **Note:** the published config is NOT the row-max of the 24 in this table --
  `SVM+RESNET_SE [stacking]` scores 0.8604 higher.
  The soft-vote headline was evidently chosen for reasons beyond raw
  mean-F1 maximization (deployability / avoiding a learned stacking
  meta-classifier); this is reported, not resolved, here.
- Nested estimate: 0.8582 (sd 0.0637)
- Optimism: -0.017 pt (bootstrap 95% CI [-2.069, +1.816] pt)
- Published configuration chosen in 0/40 folds
- **What the nested procedure actually picks:** a stacking combiner in all 40/40 folds
  (`SVM+RESNET_SE [stacking]` in 24, `SVM+RF+CNN+RESNET_SE [stacking]` in 16) -- never the
  published soft-vote. The escalation therefore fires on the "published configuration chosen
  in fewer than 20 of 40 folds" clause, not on optimism size (optimism is ~0, even slightly
  negative: a truly nested procedure would have reported marginally HIGHER than 85.8%, not
  lower). This is a defensible-headline / indefensible-combiner-choice result, not evidence
  that 85.8% itself is inflated by selection.
- Letter: **E**

## Level D (deep single-model choice)

- Candidates (10): resnet_se+none, resnet_se+gaussian, resnet_se+timemask, resnet_se+combined, resnet_se+chandrop (model of record), resnet+none, resnet_nose+chandrop, resnet_nores+none, resnet_nores+chandrop, simple+none
- Excluded (UNKNOWN backbone/augmentation or missing outputs): none
- Published: model of record (resnet_se+chandrop) = 0.8395
- Nested estimate: 0.8395 (sd 0.0671)
- Optimism: +0.000 pt (bootstrap 95% CI [-1.914, +2.401] pt)
- Published configuration chosen in 40/40 folds
- Letter: **N**

## Level J (joint: deep choice, then ensemble choice)

Level J nests the deep-architecture choice first; where that nested choice is not the chandrop model of record, no ensemble proba exists for the alternative so the deep candidate's own solo F1 is substituted (conservative, not optimistic).

- Published: 0.8580
- Nested estimate: 0.8582 (sd 0.0637)
- Optimism: -0.017 pt
- Full joint configuration (deep=chandrop AND ensemble=soft-vote) chosen in 0/40 folds
- Letter: **E**

## Outcome grid (fixed before the numbers were seen, EXPERIMENT_PLAN_KC23_CLASSICAL.md C1.3)

| Letter | Condition | Reading |
|---|---|---|
| N | Optimism < 0.3 pt at Level J, and published config chosen >= 35/40 | Headline stands; nested figure goes beside it |
| O | Optimism 0.3 to 1.0 pt | Report nested figure alongside 85.8% wherever stated |
| E | Optimism >= 1.0 pt, or published config chosen < 20/40 | ESCALATE: headline framing is Enam's decision |

**Overall letter across E, D, J (worst case, most conservative): E**
