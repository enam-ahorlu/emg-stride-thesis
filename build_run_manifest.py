#!/usr/bin/env python3
"""
build_run_manifest.py
=====================
B1 of EXPERIMENT_PLAN_AUDIT_REMEDIATION.md. Walk every 06_Code/results_* directory
and emit 06_Code/RUN_MANIFEST.csv with one row per directory:

  dir, dataset, backbone, norm_mode, augmentation, aug_param, window_ms, seed,
  n_subjects, n_rows, subject_id_set_hash, source_of_config, confidence,
  mtime_earliest, mtime_latest

Section 1.3 rule, enforced here: every config field carries exactly one of three
provenance states, recorded per field in `source_of_config`:

  verified   - read from the data itself (a CSV column, or a token embedded in a
               filename the driver wrote, or the subject-ID range)
  documented - stated on a citable line of a plan file / REPRODUCE.md; the file
               and line number are recorded
  unknown    - neither. Written UNKNOWN. NEVER inferred from the directory name.

`confidence` is the weakest state among the six config fields. A directory whose
config is mostly UNKNOWN is the finding, not a failure of the script.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "RUN_MANIFEST.csv"

DOC_FILES = ([
    "REPRODUCE.md", "README.md", "RUN_ORDER_CD_MECHANISM.md",
    "CD_MECHANISM_PROGRAMME_REPORT.md", "_STRUCTURE.md",
] + sorted(p.name for p in ROOT.glob("EXPERIMENT_PLAN_*.md"))
  + sorted(p.name for p in ROOT.glob("run_*.sh")) + ["_run_cnn_sequence.sh"])

SIAT_IDS = set(range(1, 41))
ENABL3S_IDS = {156, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194}

TOKENS = {
    "norm_mode": re.compile(r"--norm[-_]mode\s+(per_subject|global|robust|none)"
                            r"|\b(per[-_]subject)\b|\bglobal[- ]norm\b|\bglobnorm\b"),
    "augmentation": re.compile(r"--augment(?:ation)?\s+(none|gaussian|chandrop|timemask|combined|"
                               r"gainjitter|subset|mpchandrop)|--aug-chandrop-p\s+([\d.]+)"
                               r"|channel drop(?:out)?|--aug-gain-sd\s+([\d.]+)"),
    "window_ms": re.compile(r"\bw(150|250|400)\b|\b(150|250|400)\s?ms\b"),
    "seed": re.compile(r"--seed\s+(\d+)|\bseed\s+(\d+)\b"),
    "backbone": re.compile(r"--arch\s+(simple|resnet_se|resnet_nores|resnet)"
                           r"|\bresnet_se\b|\bResNet-SE\b|\bSimpleEMGCNN\b"),
}


# --------------------------------------------------------------------------- docs
def load_doc_index() -> list[tuple[str, int, str]]:
    idx = []
    for fn in DOC_FILES:
        p = ROOT / fn
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            idx.append((fn, i, line))
    return idx


DOC_INDEX = load_doc_index()


def documented(dirname: str, field: str) -> str | None:
    """Return 'documented(file:line)' only if ONE line both names this dir literally
    AND carries a token for `field` (a citable line, per section 1.3). A mere
    nearby mention is NOT documentation and returns None (-> unknown). The dir
    name's own semantics are never inspected."""
    pat = TOKENS[field]
    for fn, ln, line in DOC_INDEX:
        if dirname in line and pat.search(line):
            return f"documented({fn}:{ln})"
    return None


# ----------------------------------------------------------------- dir inspection
def find_subjectwise(d: Path) -> Path | None:
    for pat in ("cnn_arch_subjectwise.csv", "*_nested_loso_subjectwise.csv", "*subjectwise*.csv"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def read_subject_ids(csv_path: Path) -> list[int]:
    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", errors="replace")))
    except Exception:
        return []
    if not rows:
        return []
    for col in ("subject", "heldout_subject", "subject_id", "Subject", "subj"):
        if col in rows[0]:
            out = []
            for r in rows:
                try:
                    out.append(int(float(r[col])))
                except (ValueError, TypeError):
                    pass
            return out
    return []


def mtimes(d: Path) -> tuple[str, str]:
    ts = []
    for f in d.rglob("*"):
        if f.is_file():
            ts.append(f.stat().st_mtime)
    if not ts:
        return "", ""
    import datetime as dt
    return (dt.datetime.fromtimestamp(min(ts), dt.timezone.utc).isoformat(timespec="seconds"),
            dt.datetime.fromtimestamp(max(ts), dt.timezone.utc).isoformat(timespec="seconds"))


FNAME_WIN = re.compile(r"_w(150|250|400)_ov\d+_conf\d+")
FNAME_FEAT = re.compile(r"features_(ext|base|full)|combined_\w+_(full)")
FNAME_ENABL = re.compile(r"ENABL3S", re.I)
CLASSICAL_MODEL = re.compile(r"__?(SVM|RF|LDA)_")


def inspect(d: Path) -> dict:
    row = {k: "" for k in
           ("dir", "dataset", "backbone", "norm_mode", "augmentation", "aug_param",
            "window_ms", "seed", "n_subjects", "n_rows", "subject_id_set_hash",
            "source_of_config", "confidence", "mtime_earliest", "mtime_latest")}
    row["dir"] = d.name
    prov = {}   # field -> "verified(...)" / "documented(file:line)" / "unknown"

    sw = find_subjectwise(d)
    ids = read_subject_ids(sw) if sw else []
    uniq = sorted(set(ids))
    row["n_rows"] = str(len(ids))
    row["n_subjects"] = str(len(uniq))
    row["subject_id_set_hash"] = (hashlib.sha1(",".join(map(str, uniq)).encode()).hexdigest()[:12]
                                  if uniq else "")

    # dataset: verified from subject-ID range
    if uniq:
        s = set(uniq)
        if s & ENABL3S_IDS:
            row["dataset"], prov["dataset"] = "ENABL3S", "verified(subject-id range)"
        elif s <= SIAT_IDS:
            row["dataset"], prov["dataset"] = "SIAT", "verified(subject-id range)"
        else:
            row["dataset"], prov["dataset"] = f"mixed/other({min(uniq)}-{max(uniq)})", "verified(subject-id range)"
    else:
        prov["dataset"] = "unknown"

    # backbone: verified from cnn_arch_summary.csv, or classical model from filename
    arch_csv = d / "cnn_arch_summary.csv"
    if arch_csv.exists():
        try:
            r0 = next(csv.DictReader(arch_csv.open(encoding="utf-8")))
            if r0.get("arch"):
                row["backbone"], prov["backbone"] = r0["arch"], "verified(cnn_arch_summary.csv)"
        except Exception:
            pass
    if not row["backbone"] and sw:
        m = CLASSICAL_MODEL.search(sw.name)
        if m:
            row["backbone"], prov["backbone"] = f"classical:{m.group(1)}", "verified(subjectwise filename)"
    if not row["backbone"]:
        dd = documented(d.name, "backbone")
        if dd:
            row["backbone"], prov["backbone"] = "see source", dd
        else:
            row["backbone"], prov["backbone"] = "UNKNOWN", "unknown"

    # window_ms + feature set: verified from any embedded filename token in the dir
    win = feat = None
    src_file = None
    for f in list(d.glob("*")) + (list((d).rglob("*.csv"))[:200]):
        mw = FNAME_WIN.search(f.name)
        if mw and win is None:
            win, src_file = mw.group(1), f.name
        mf = FNAME_FEAT.search(f.name)
        if mf and feat is None:
            g = [x for x in mf.groups() if x]
            if g:
                feat = g[0]
    if win:
        row["window_ms"], prov["window_ms"] = win, f"verified(filename token: {src_file})"
    else:
        dd = documented(d.name, "window_ms")
        row["window_ms"], prov["window_ms"] = ("see source", dd) if dd else ("UNKNOWN", "unknown")

    # norm_mode: never from dir name. documented only.
    dd = documented(d.name, "norm_mode")
    row["norm_mode"], prov["norm_mode"] = ("see source", dd) if dd else ("UNKNOWN", "unknown")

    # augmentation + aug_param: documented only (CNN CSVs do not record it)
    dd = documented(d.name, "augmentation")
    row["augmentation"], prov["augmentation"] = ("see source", dd) if dd else ("UNKNOWN", "unknown")
    row["aug_param"] = "see source" if dd else "UNKNOWN"

    # seed: documented only
    dd = documented(d.name, "seed")
    row["seed"], prov["seed"] = ("see source", dd) if dd else ("UNKNOWN", "unknown")

    if feat:
        prov["feature_set"] = f"verified({feat})"

    row["mtime_earliest"], row["mtime_latest"] = mtimes(d)

    order = {"verified": 0, "documented": 1, "unknown": 2}
    def state(v: str) -> str:
        return "verified" if v.startswith("verified") else "documented" if v.startswith("documented") else "unknown"
    core = ["dataset", "backbone", "norm_mode", "augmentation", "window_ms", "seed"]
    row["confidence"] = max((state(prov[f]) for f in core), key=lambda s: order[s])
    row["source_of_config"] = "; ".join(f"{f}:{prov[f]}" for f in
                                        ("dataset", "backbone", "norm_mode", "augmentation",
                                         "window_ms", "seed") + (("feature_set",) if feat else ()))
    return row


def main() -> int:
    dirs = sorted(p for p in ROOT.glob("results_*") if p.is_dir())
    rows = [inspect(d) for d in dirs]
    cols = ["dir", "dataset", "backbone", "norm_mode", "augmentation", "aug_param",
            "window_ms", "seed", "n_subjects", "n_rows", "subject_id_set_hash",
            "source_of_config", "confidence", "mtime_earliest", "mtime_latest"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    fully_verified = sum(r["confidence"] == "verified" for r in rows)
    any_unknown = sum(r["confidence"] == "unknown" for r in rows)
    documented_lvl = sum(r["confidence"] == "documented" for r in rows)
    print(f"wrote {OUT}  ({n} directories)")
    print(f"  fully verified (all six core fields): {fully_verified}")
    print(f"  best-case documented (weakest field documented, none unknown): {documented_lvl}")
    print(f"  carry at least one UNKNOWN core field: {any_unknown}")
    print(f"  scanned doc/script files: {len([f for f in DOC_FILES if (ROOT / f).exists()])}")

    # ------------------------------------------------------------------ 1.4 cross-check
    # (dir, canonical quantity, thesis-implied backbone, thesis-implied dataset)
    CANON = [
        ("results_ensemble_v2_chandrop", "headline soft vote 85.8%", None, "SIAT"),
        ("results_causal_ensemble", "causal ensemble 81.7%", None, "SIAT"),
        ("results_cnn_aug_resnet_se_chandrop", "deep model of record 84.0%", "resnet_se", "SIAT"),
        ("results_loso_freq_persubj", "per-subject SVM/RF 77.7/77.3", "classical", "SIAT"),
        ("results_loso_freq", "global baseline 70.8/72.2/68.2", "classical", "SIAT"),
        ("results_variance_decomposition", "alignment ladder / probe / silhouette / ICC", None, "SIAT"),
        ("results_latency", "RF single-window latency 30.5 ms", None, None),
        ("results_within_subject", "label-budget crossover", None, "SIAT"),
        ("results_g3_noaug_instr", "occlusion cost 84.2 pp (un-augmented half)", "resnet_se", "SIAT"),
        ("results_cd_resnet_nose_chandrop", "occlusion cost 15.0 pp (channel-dropout half)", "resnet_se", "SIAT"),
        ("results_w4_gainjitter", "gain jitter +7.31 pp", "resnet", "SIAT"),
        ("results_cd_rate_p0.1", "dropout-rate sweep 0.8311", "resnet_se", "SIAT"),
        ("results_cd_rate_p0.2", "dropout-rate sweep 0.8376", "resnet_se", "SIAT"),
        ("results_cd_rate_p0.3", "dropout-rate sweep 0.8373", "resnet_se", "SIAT"),
        ("results_cd_rate_p0.5", "dropout-rate sweep 0.8205", "resnet_se", "SIAT"),
        ("results_window_ablation", "W-1 window ablation", None, None),
        ("results_cnn_calibration_chandrop_resnet_se", "calibration +3.6 pp", "resnet_se", "SIAT"),
        ("results_alignment_ladder_loso", "Table 4.16 ladder means", "classical", "SIAT"),
    ]
    by = {r["dir"]: r for r in rows}
    print("\n" + "=" * 78)
    print("SECTION 1.4 CROSS-CHECK against the handoff Section 4 canonical numbers table")
    print("=" * 78)
    hits = []
    for d, quant, th_bb, th_ds in CANON:
        r = by.get(d)
        if r is None:
            print(f"  [{d}] {quant}: NO MANIFEST ROW (directory absent)")
            hits.append((d, "directory absent"))
            continue
        bb = r["backbone"]
        bb_state = ("verified" if "verified" in r["source_of_config"].split("backbone:")[1][:20]
                    else "documented" if "documented(" in r["source_of_config"].split("backbone:")[1][:40]
                    else "unknown")
        flags = []
        if th_bb and bb not in ("see source", "UNKNOWN") and th_bb not in bb:
            flags.append(f"BACKBONE MISMATCH: manifest {bb} ({bb_state}) vs thesis-implied {th_bb}")
        if th_bb and bb in ("UNKNOWN",):
            flags.append(f"BACKBONE UNKNOWN on a canonical dir (thesis implies {th_bb})")
        if th_ds and r["dataset"] not in ("", th_ds) and not r["dataset"].startswith(th_ds):
            flags.append(f"DATASET MISMATCH: manifest {r['dataset']} vs thesis {th_ds}")
        core_unknowns = [f for f in ("dataset", "backbone", "norm_mode", "augmentation", "window_ms", "seed")
                         if f"{f}:unknown" in r["source_of_config"]]
        tag = "  ".join(flags) if flags else f"ok (backbone {bb} [{bb_state}])"
        print(f"  [{d}] {quant}")
        print(f"      {tag}")
        if core_unknowns:
            print(f"      unresolved fields (no citable line): {', '.join(core_unknowns)}")
        if flags:
            hits.extend((d, f) for f in flags)

    occ_hits = [h for h in hits if h[0] in ("results_g3_noaug_instr", "results_cd_resnet_nose_chandrop")]
    other_hits = [h for h in hits if h not in occ_hits]
    print("\n" + "-" * 78)
    print(f"CROSS-CHECK RESULT: {len(occ_hits)} occlusion-backbone flag(s), {len(other_hits)} other "
          "config flag(s).")
    if not other_hits:
        print("  The occlusion-pair backbone (resnet, not resnet_se) is the ONLY backbone/dataset "
              "mismatch against the canonical numbers table. No other canonical number is attached "
              "to a directory whose verifiable config contradicts the thesis.")
    else:
        print("  Additional flags beyond the occlusion pair, same class of defect, each needs a note:")
        for d, f in other_hits:
            print(f"    - {d}: {f}")
    print("  Separately: many canonical dirs carry UNKNOWN norm_mode / augmentation / seed because "
          "no run wrote a config file. That is the B1 finding, not a per-number defect; §1.5 fixes "
          "it going forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
