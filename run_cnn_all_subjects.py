# run_cnn_all_subjects.py
# Wrapper to run train_cnn_subjectdep.py across all subjects and aggregate results.

from __future__ import annotations

import argparse
import re
import sys
import subprocess
from pathlib import Path

import pandas as pd


# -----------------------------
# Parsing helpers
# -----------------------------

_RE_BLOCK = re.compile(r"=== CNN RESULT ===(?P<body>.*?)(?:\n\s*\n|\Z)", re.DOTALL)

_RE_ACC = re.compile(r"Accuracy:\s*([0-9]*\.?[0-9]+)")
_RE_BAL = re.compile(r"Balanced\s*Acc:\s*([0-9]*\.?[0-9]+)")
_RE_F1 = re.compile(r"F1\s*macro:\s*([0-9]*\.?[0-9]+)")
_RE_TRAIN_TIME = re.compile(r"Train\s*time\s*\(s\):\s*([0-9]*\.?[0-9]+)")
_RE_LAT = re.compile(r"Inference\s*per\s*window\s*\(ms\):\s*([0-9]*\.?[0-9]+)")


def parse_cnn_result(stdout: str) -> dict:
    """
    Parse the final '=== CNN RESULT ===' block from train_cnn_subjectdep.py output.
    Returns dict with keys: acc, bal_acc, f1, train_time_s, latency_ms
    Raises ValueError if parsing fails.
    """
    m = _RE_BLOCK.search(stdout)
    if not m:
        raise ValueError("Could not find '=== CNN RESULT ===' block in output.")

    body = m.group("body")

    def grab(rx: re.Pattern, name: str) -> float:
        mm = rx.search(body)
        if not mm:
            raise ValueError(f"Could not parse {name} from CNN RESULT block.")
        return float(mm.group(1))

    return {
        "acc": grab(_RE_ACC, "Accuracy"),
        "bal_acc": grab(_RE_BAL, "Balanced Acc"),
        "f1": grab(_RE_F1, "F1 macro"),
        "train_time_s": grab(_RE_TRAIN_TIME, "Train time (s)"),
        "latency_ms": grab(_RE_LAT, "Inference per window (ms)"),
    }


# -----------------------------
# Runner
# -----------------------------

def run_one_subject(
    train_script: Path,
    npz: Path,
    meta: Path,
    subject: int,
    use: str,
    norm: str,
    epochs: int,
    batch: int,
    seed: int,
    timeout_s: int | None,
    extra_args: list[str],
    log_dir: Path,
) -> tuple[dict | None, str, str, int]:
    """
    Runs train_cnn_subjectdep.py for one subject.
    Returns: (metrics_or_None, stdout, stderr, returncode)
    """
    cmd = [
        sys.executable, str(train_script),
        "--npz", str(npz),
        "--meta", str(meta),
        "--subject", str(subject),
        "--use", use,
        "--norm", norm,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--seed", str(seed),
    ] + extra_args

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s if timeout_s and timeout_s > 0 else None,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[wrapper] TIMEOUT after {timeout_s}s"
        rc = 124

    # Save logs always
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"sub_{subject:03d}.out.txt").write_text(stdout, encoding="utf-8", errors="ignore")
    (log_dir / f"sub_{subject:03d}.err.txt").write_text(stderr, encoding="utf-8", errors="ignore")

    if rc != 0:
        return None, stdout, stderr, rc

    try:
        metrics = parse_cnn_result(stdout)
    except Exception:
        metrics = None

    return metrics, stdout, stderr, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--train-script", default="train_cnn_subjectdep.py", type=Path)
    ap.add_argument("--use", choices=["raw", "env"], default="env")
    ap.add_argument("--norm", choices=["none", "zscore"], default="zscore")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--log-dir", default="results_cnn_subjectdep/logs", type=Path)
    ap.add_argument("--timeout-s", type=int, default=0, help="0 disables timeout")
    ap.add_argument(
        "--subjects",
        default="all",
        help="Comma list like '1,2,3' or 'all' (default).",
    )
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to train_cnn_subjectdep.py after '--extra'. "
             "Example: --extra --dropout 0.3 --report",
    )
    args = ap.parse_args()

    meta_df = pd.read_csv(args.meta)
    if "subject" not in meta_df.columns:
        raise KeyError("Meta CSV must contain a 'subject' column.")

    if args.subjects.strip().lower() == "all":
        subjects = sorted(pd.unique(meta_df["subject"].dropna()).tolist())
        subjects = [int(s) for s in subjects]
    else:
        subjects = [int(x.strip()) for x in args.subjects.split(",") if x.strip()]

    # Resolve script path relative to this wrapper if needed
    train_script = args.train_script
    if not train_script.exists():
        # try next to this wrapper
        here = Path(__file__).resolve().parent
        cand = here / args.train_script
        if cand.exists():
            train_script = cand
        else:
            raise FileNotFoundError(f"Could not find train script: {args.train_script}")

    results = []
    failures = []

    for sub in subjects:
        metrics, stdout, stderr, rc = run_one_subject(
            train_script=train_script,
            npz=args.npz,
            meta=args.meta,
            subject=sub,
            use=args.use,
            norm=args.norm,
            epochs=args.epochs,
            batch=args.batch,
            seed=args.seed,
            timeout_s=args.timeout_s if args.timeout_s > 0 else None,
            extra_args=args.extra,
            log_dir=args.log_dir,
        )

        row = {
            "subject": sub,
            "acc": None,
            "bal_acc": None,
            "f1": None,
            "train_time_s": None,
            "latency_ms": None,
            "returncode": rc,
        }

        if metrics is not None:
            row.update(metrics)
        else:
            failures.append(sub)

        results.append(row)

        # Lightweight progress
        if metrics is not None:
            print(f"[OK] subject={sub} acc={row['acc']:.4f} bal={row['bal_acc']:.4f} f1={row['f1']:.4f}")
        else:
            print(f"[FAIL] subject={sub} (rc={rc}) - see logs in {args.log_dir}")

    out_df = pd.DataFrame(results).sort_values("subject")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}")
    if failures:
        print(f"Subjects failed ({len(failures)}): {failures}")
        print(f"Check per-subject logs in: {args.log_dir}")


if __name__ == "__main__":
    main()
