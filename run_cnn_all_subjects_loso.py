import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser("Run CNN LOSO over all subjects by calling train_cnn_loso.py repeatedly")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env")
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--out", default="results_cnn_loso")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=40)
    args = ap.parse_args()

    for heldout in range(args.start, args.end + 1):
        cmd = [
            sys.executable, "train_cnn_loso.py",
            "--npz", args.npz,
            "--meta", args.meta,
            "--xkey", args.xkey,
            "--label-col", args.label_col,
            "--out", args.out,
            "--heldout", str(heldout),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
        ]
        if args.amp:
            cmd.append("--amp")
        if args.resume:
            cmd.append("--resume")

        print("\n[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()