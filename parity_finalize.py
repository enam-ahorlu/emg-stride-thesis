#!/usr/bin/env python3
"""
parity_finalize.py
==================
Consolidates the channel-dropout parity programme. Reads the per-stage outcome
JSONs in results_parity/, checks the section 6 gates on any GPU arm that has
completed, writes results_parity/parity_outcome.json and PARITY_REPORT.md, and
prints the FDR-family contribution table (section 6.3). No thesis file is
touched and section 4.17 is not edited.

  python parity_finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_parity"

# comparator arms whose published means must reproduce before their contrasts
# are trusted (section 6, "reproduce a known number before trusting a new one")
COMPARATORS = {
    "results_cnn_aug_resnet_se_chandrop": 0.8395,
    "results_cnn_aug_resnet_se_none": 0.7822,
}
NEW_ARMS = [
    "results_p5_subset_resnet_se",
    "results_p6_gainjitter_resnet_se_sd0.40",
    "results_p6_mpchandrop_resnet_se_sd0.40",
    "results_p6_gainjitter_resnet_se_sd0.30",
    "results_p6_gainjitter_resnet_se_sd0.50",
]


def load_json(name: str):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None


def arm_check(dirname: str) -> dict:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        return {"dir": dirname, "status": "absent"}
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    n_rows = len(df)
    n_uniq = df[sc].nunique()
    dups = n_rows != n_uniq
    mean = float(df["f1_macro"].mean())
    return {"dir": dirname, "status": "present", "rows": n_rows, "unique_subjects": n_uniq,
            "duplicate_rows": dups, "mean_f1": round(mean, 4), "complete_40": n_uniq == 40}


def main() -> int:
    print("=" * 78)
    print("CHANNEL-DROPOUT PARITY PROGRAMME: consolidation")
    print("=" * 78)

    # ---- section 6 gates on comparator + new arms -----------------------
    print("\n--- gate: comparator arms reproduce their published means ---")
    gate_ok = True
    for d, ref in COMPARATORS.items():
        c = arm_check(d)
        if c["status"] != "present":
            print(f"  {d}: ABSENT (expected on disk already)"); gate_ok = False; continue
        delta = (c["mean_f1"] - ref) * 100
        ok = abs(delta) <= 1.5 and not c["duplicate_rows"] and c["complete_40"]
        gate_ok &= ok
        print(f"  {d}: mean {c['mean_f1']:.4f} vs published {ref:.4f}  ({delta:+.2f} pp)  "
              f"rows {c['rows']}/uniq {c['unique_subjects']}  {'OK' if ok else 'CHECK'}")

    print("\n--- gate: new GPU arms, exactly 40 unique subjects, no duplicate rows ---")
    arm_states = {}
    for d in NEW_ARMS:
        c = arm_check(d)
        arm_states[d] = c
        if c["status"] == "absent":
            print(f"  {d}: not yet on disk")
        else:
            flag = "OK" if (c["complete_40"] and not c["duplicate_rows"]) else "CHECK"
            print(f"  {d}: rows {c['rows']} / uniq {c['unique_subjects']}  "
                  f"mean {c['mean_f1']:.4f}  dup={c['duplicate_rows']}  {flag}")

    # ---- per-stage outcomes -------------------------------------------
    stages = {
        "P0": {"outcome": "done", "file": "p0_literature.md"},
        "P1": load_json("p1_outcome.json"),
        "P2": load_json("p2_outcome.json"),
        "P3": load_json("p3_outcome.json"),
        "P5": load_json("p5_outcome.json"),
        "P6": load_json("p6_outcome.json"),
        "P7": load_json("p7_outcome.json"),
    }
    letters = {}
    print("\n--- per-stage outcome letters ---")
    for k, v in stages.items():
        if k == "P0":
            letters[k] = "done"
            print(f"  {k}: literature grounding recorded (p0_literature.md)")
        elif v is None:
            letters[k] = "pending"
            print(f"  {k}: pending (run not finished / stats not run)")
        else:
            letters[k] = v.get("outcome", "?")
            print(f"  {k}: {letters[k]}  -  {v.get('outcome_meaning', '')[:90]}")

    # ---- FDR family contribution (section 6.3) -----------------------
    print("\n--- new paired Wilcoxon tests for the whole-thesis BH family (section 6.3) ---")
    fdr_rows = []
    p1 = stages["P1"]
    if p1:
        for r in p1.get("p1a", {}).get("adjacent", []):
            fdr_rows.append(("P-1a " + r["contrast"], r["p_raw"]))
    p2 = stages["P2"]
    if p2:
        fdr_rows.append(("P-2 per-subject rank consistency, CD vs baseline",
                         p2["paired_wilcoxon"]["p_raw"]))
    p3 = stages["P3"]
    if p3:
        fdr_rows.append(("P-3 external CD vs no aug, per-subject norm", p3["primary"]["p_raw"]))
        fdr_rows.append(("P-3 external CD vs no aug, global norm", p3["secondary"]["p_raw"]))
    p5 = stages["P5"]
    if p5:
        fdr_rows.append(("P-5 subset vs channel dropout", p5["primary"]["p_raw"]))
        fdr_rows.append(("P-5 subset vs no augmentation", p5["secondary"]["p_raw"]))
    p6 = stages["P6"]
    if p6:
        fdr_rows.append(("P-6 C1 channel dropout vs mean-preserving chandrop",
                         p6["C1_chandrop_vs_mpchandrop"]["p_raw"]))
        fdr_rows.append(("P-6 C2 mean-preserving chandrop vs gain jitter",
                         p6["C2_mpchandrop_vs_gainjitter"]["p_raw"]))
    p6e = load_json("p6_extension_outcome.json")
    stages["P6ext"] = p6e
    if p6e:
        for r in p6e["matched_points"]:
            tag = "(FDR family)" if abs(r["sd"] - 0.40) > 1e-6 else "(cross-check, SD 0.40 already in core)"
            fdr_rows.append((f"P-6 ext matched-variance SD {r['sd']:.2f} {tag}", r["p_raw"]))
    p7 = stages["P7"]
    if p7:
        fdr_rows.append(("P-7 D1 mean-preserving chandrop vs channel dropout p0.5",
                         p7["sd50"]["D1"]["p_raw"]))
        fdr_rows.append(("P-7 D2 gain jitter sd0.50 vs mean-preserving chandrop",
                         p7["sd50"]["D2"]["p_raw"]))
    for name, p in fdr_rows:
        print(f"    {name:<64} raw p = {p:.4g}")
    additive = [r for r in fdr_rows if "cross-check" not in r[0]]
    print(f"  new paired Wilcoxon tests recommended for the BH family: {len(additive)} "
          f"({len(fdr_rows)} listed; the SD 0.40 extension row is a cross-check, not additive)")
    print("  NOT folded in (reported separately, section 6.3): P-1 Spearman/Page correlations "
          "and their permutation nulls; P-2 randomization test and shuffle-null checks.")
    print("  Section 4.17 is NOT edited here; recompute_unified_fdr_v2.py runs it in one pass "
          "afterwards.")

    consolidated = {
        "programme": "channel-dropout parity",
        "outcome_letters": letters,
        "comparator_gate_ok": gate_ok,
        "new_arm_states": arm_states,
        "fdr_family_new_paired_tests": [{"test": n, "p_raw": p} for n, p in fdr_rows],
        "fdr_family_count_listed": len(fdr_rows),
        "fdr_family_count_additive": len(additive),
        "section_4_17_edited": False,
        "thesis_files_edited": False,
        "deep_model_of_record_changed": False,
        "p4_started": False,
    }
    json.dump(consolidated, open(OUT / "parity_outcome.json", "w"), indent=2)
    print(f"\nwrote {OUT/'parity_outcome.json'}")

    # ---- consolidated human report ----
    def g(stage, *path, default="n/a"):
        v = stages.get(stage)
        for k in path:
            if v is None:
                return default
            v = v.get(k) if isinstance(v, dict) else default
        return v

    lines = [
        "# Channel-dropout parity programme: consolidated outcome",
        "",
        "No thesis file was edited. Section 4.17 was not touched. "
        "The deep model of record (channel-dropout resnet_se, 0.840) is unchanged. "
        "Stage P-4 was not started.",
        "",
        "| Stage | Outcome | One line |",
        "|---|---|---|",
        f"| P-0 literature | done | 4 sources verified; Pereira et al. is ICASSP 2024, not a "
        "preprint; Srivastava Table 10 read in full. See p0_literature.md. |",
        f"| P-1 mechanism to outcome | {letters['P1']} | "
        f"{g('P1','outcome_meaning')[:120]} |",
        f"| P-2 between-subject transfer | {letters['P2']} | "
        f"{g('P2','outcome_meaning')[:120]} |",
        f"| P-3 external replication | {letters['P3']} | "
        f"{g('P3','outcome_meaning')[:120]} |",
        f"| P-5 published alternative | {letters['P5']} | "
        f"{g('P5','outcome_meaning')[:120]} |",
        f"| P-6 perturbation ladder | {letters['P6']} | "
        f"{g('P6','outcome_meaning')[:120]} |",
    ]
    if p6e:
        lines.append(f"| P-6 extension | {p6e['verdict']} | {p6e['verdict_text'][:120]} |")
    if stages.get("P7"):
        lines.append(f"| P-7 high-dose decomposition | {letters['P7']} | "
                     f"{g('P7','outcome_meaning')[:120]} |")
    lines += [
        "",
        "## New paired Wilcoxon tests for the whole-thesis Benjamini-Hochberg family (section 6.3)",
        "",
        "Section 4.17 is not edited here. recompute_unified_fdr_v2.py runs the family in one pass.",
        "",
        "| test | raw p |",
        "|---|---|",
    ]
    for name, p in fdr_rows:
        lines.append(f"| {name} | {p:.4g} |")
    lines += [
        "",
        f"Recommended additive count: {len(additive)} "
        f"({len(fdr_rows)} listed; the SD 0.40 extension row duplicates information already in the "
        "P-6 core arms and is a cross-check, not an additive test).",
        "",
        "Reported separately (not paired Wilcoxon, per section 6.3, as with the G3 randomization "
        "test): P-1a Spearman(rate, cost) and Page trend and their permutation context; P-1b and "
        "P-1c Spearman correlations with their 10,000-draw permutation nulls; P-2 subject-level "
        "randomization test and the two shuffle-null validity checks.",
        "",
        "## Files",
        "",
        "results_parity/: p0_literature.md, p1_*, p2_*, p3_*, p5_*, p6_*, p6_extension_*, "
        "parity_outcome.json, this report.",
    ]
    (OUT / "PARITY_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'PARITY_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
