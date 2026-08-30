# measure_inference_latency.py
# ---------------------------------------------------------------------------
# Reviewer catch (Critique 2): the reported CNN inference latency (0.0116 ms/window,
# blank SD) is implausibly low for a single-window CPU forward pass and was almost
# certainly a batched/vectorised figure divided by the window count. This script
# re-measures per-window latency HONESTLY: batch size 1, warm-up, many timed reps,
# reporting mean / median / p95, on CPU (the deployment-relevant case for an
# embedded prosthesis) and optionally on GPU.
#
# SVM/RF latency depends on the fitted model (n support vectors / n trees), so
# they are fit with the headline hyperparameters (SVM C=1; RF n_estimators=500)
# on per-subject-normalised features. CNN latency depends only on the architecture
# (not the weights), so an initialised SimpleEMGCNN is timed as-is.
#
# Example:
#   python measure_inference_latency.py \
#       --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
#       --meta     features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --npz      windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive
from cnn_architectures import build_model


def timed(fn, x, warmup, reps):
    for _ in range(warmup):
        fn(x)
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter(); fn(x); ts[i] = (time.perf_counter() - t0) * 1000.0  # ms
    return ts


def summarise(name, ts, cadence_ms=125.0):
    return {"model": name, "mean_ms": round(ts.mean(), 4), "median_ms": round(np.median(ts), 4),
            "sd_ms": round(ts.std(ddof=1), 4), "p95_ms": round(np.percentile(ts, 95), 4),
            "min_ms": round(ts.min(), 4), "realtime_vs_125ms": bool(np.percentile(ts, 95) < cadence_ms)}


def main():
    ap = argparse.ArgumentParser("Honest per-window inference latency (batch=1).")
    ap.add_argument("--features", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--npz", required=True); ap.add_argument("--xkey", default="X_env")
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="results_latency")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- classical models on per-subject-normalised features ----
    X = load_features_npz(Path(args.features)).astype(np.float64)
    meta = pd.read_csv(args.meta)
    subj = meta[next(c for c in ["subject","subject_id","sid"] if c in meta.columns)].astype(int).to_numpy()
    y, _ = encode_labels(meta[next(c for c in ["movement","label","y"] if c in meta.columns)].astype(str).to_numpy())
    Xn = per_subject_transductive(X, subj, np.ones(len(y), bool))
    print("[fit] SVM (C=1) ...", flush=True)
    svm = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", cache_size=500).fit(Xn, y)
    print(f"      n_support={int(svm.n_support_.sum())}", flush=True)
    print("[fit] RF (n_estimators=500) ...", flush=True)
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=args.seed, n_jobs=-1).fit(Xn, y)
    # Time single-window inference SINGLE-THREADED (the realistic embedded case): with
    # n_jobs=-1 a 1-sample predict pays joblib's parallel-dispatch overhead (tens of ms),
    # which is itself a measurement artifact, not real compute.
    rf.set_params(n_jobs=1)

    x1 = Xn[:1]  # a single window's feature vector, shape (1, F)
    rows = []
    rows.append(summarise("SVM", timed(lambda a: svm.predict(a), x1, args.warmup, args.reps)))
    rows.append(summarise("RF",  timed(lambda a: rf.predict(a),  x1, args.warmup, args.reps)))

    # ---- CNN + ResNet-SE: single-window forward on CPU (and GPU if available) ----
    import torch
    torch.set_num_threads(1)  # single-core, matching the embedded-deployment scenario
    from train_cnn_loso import SimpleEMGCNN
    data = np.load(args.npz); Xw = data[args.xkey].astype(np.float32)
    in_ch = Xw.shape[1]
    xt_cpu = torch.from_numpy(Xw[:1]).float()  # (1, C, T)
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    resnet_se_nets = {}  # dev -> eval()'d net, reused below for the ensemble timing
    for dev in devices:
        device = torch.device(dev)
        net = SimpleEMGCNN(in_ch=in_ch, n_classes=4).to(device).eval()
        xt = xt_cpu.to(device)
        def fwd(a, net=net, dev=dev):
            with torch.no_grad():
                _ = net(a)
                if dev == "cuda":
                    torch.cuda.synchronize()
        rows.append(summarise(f"CNN ({dev})", timed(fwd, xt, args.warmup, args.reps)))

        rnet = build_model("resnet_se", in_ch, 4).to(device).eval()
        resnet_se_nets[dev] = rnet
        def fwd_rse(a, net=rnet, dev=dev):
            with torch.no_grad():
                _ = net(a)
                if dev == "cuda":
                    torch.cuda.synchronize()
        rows.append(summarise(f"ResNet-SE ({dev})", timed(fwd_rse, xt, args.warmup, args.reps)))

    # ---- Soft ensemble (SVM + RF + ResNet-SE): sum of member inference + probability-average op ----
    # Timed as ONE end-to-end call per rep (not summed percentiles from separate measurements),
    # so the reported percentiles are honest about the full sequential pipeline's variance.
    print("[fit] SVM (C=1, probability=True) for the ensemble's soft-vote path ...", flush=True)
    svm_proba = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced",
                     cache_size=500, probability=True, random_state=args.seed).fit(Xn, y)
    for dev in devices:
        rnet = resnet_se_nets[dev]
        xt = xt_cpu.to(torch.device(dev))
        def ensemble_fwd(_a, rnet=rnet, dev=dev, xt=xt):
            p_svm = svm_proba.predict_proba(x1)
            p_rf = rf.predict_proba(x1)
            with torch.no_grad():
                logits = rnet(xt)
                p_rse = torch.softmax(logits, dim=1).cpu().numpy()
                if dev == "cuda":
                    torch.cuda.synchronize()
            avg = (p_svm + p_rf + p_rse) / 3.0
            _ = avg.argmax(1)
        rows.append(summarise(f"Ensemble SVM+RF+ResNet-SE ({dev})", timed(ensemble_fwd, x1, args.warmup, args.reps)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "inference_latency_measured.csv", index=False)
    print("\n=== PER-WINDOW LATENCY (batch=1) ===")
    print(df.to_string(index=False))
    print("\nThesis previously reported: SVM 1.06 ms, RF 0.57 ms, CNN 0.0116 ms (the CNN figure is the one under review).")


if __name__ == "__main__":
    main()
