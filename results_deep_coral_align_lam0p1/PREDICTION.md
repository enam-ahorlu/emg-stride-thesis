# D2c prediction, written before any run started

Timestamp: 2026-09-19T13:56:14Z (UTC), before the smoke test or Stage A.

Source: `docs/EXPERIMENT_PLAN_DEEPCORAL.md`, section "D2c AMENDMENT, 19 September 2026".

## The prediction

If the through-line's mechanism is right and the CORAL weight is a real invariance knob,
lambda = 100 should leave the target less separable from the source than lambda = 0.1, and
the class probe on the target should hold or fall; accuracy should not rise. If the weight
is inert on invariance, the flat F1 in D2 says nothing about the shape, and the thesis says
so with numbers.

## The outcome table

| Outcome | Condition | Meaning for the thesis | Action |
|---|---|---|---|
| 1, inert | alignment did not move | the weight is not an invariance knob here; the flat sweep tests nothing about the shape | run Stage B; body states it with numbers |
| 2, instance | moved; class information fell; accuracy did not rise | consistent with the shape on a third axis | report; framing is Enam's call |
| 3, not better, no measured cost | moved; class information held; accuracy did not rise | supports "not better"; does not show the cost on this axis | report; framing is Enam's call |
| 4, counter-example | moved; accuracy rose | contradicts the through-line on this axis | **hard stop, escalate** |
| 5, reverse | domain probe **higher** by at least 2.0 pp, significant | the higher weight left the embedding more separable | stop and escalate |

## Decision rules, fixed now (for reference)

- **Alignment moved:** domain probe lower by at least 2.0 pp, Holm p < 0.05.
- **Class information fell:** target class probe lower by at least 1.0 pp, Holm p < 0.05.
- **Accuracy rose:** F1 higher by more than 0.5 pp (the nondeterminism band), Holm p < 0.05.

## Reproduction gates (per arm)

Mean macro-F1 within 1.5 pp of the D2 arm: lambda 0.1 -> 0.832075; lambda 100 -> 0.834530.
A miss stops the run; report it.
