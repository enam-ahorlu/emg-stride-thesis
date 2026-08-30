#!/usr/bin/env python3
"""
verify_onnx_equivalence.py
==========================
Pays technical-debt item TD-05.

Asserts that the exported ONNX graphs reproduce the native scikit-learn and
PyTorch models to within 1e-4, so the model MyoLens serves is provably the
model the thesis measured.

RUN THIS IN THE PYTHON 3.12 VENV, NOT THE THESIS VENV. Two reasons:
  1. onnxruntime's cp314 support is unconfirmed, and the thesis venv is 3.14.
  2. More importantly, verifying an artefact in the runtime that will actually
     serve it is a stronger test than verifying it in the runtime that produced
     it. Version skew between export and serving is exactly the failure this
     test exists to catch.

Setup:
    py -3.12 -m venv .venv-verify
    .venv-verify\\Scripts\\python -m pip install onnxruntime numpy

Usage:
    .venv-verify\\Scripts\\python verify_onnx_equivalence.py --artifacts <dir>

Exit code 0 = pass. Non-zero = do not deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOL = 1e-4


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # Determinism: thread scheduling is the usual source of run-to-run drift.
    # The service pins these identically, so the test must too.
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def report(name: str, native: np.ndarray, onnx_out: np.ndarray) -> bool:
    diff = np.abs(native - onnx_out)
    max_abs = float(diff.max())
    agree = float((native.argmax(1) == onnx_out.argmax(1)).mean())
    ok = max_abs < TOL and agree == 1.0
    print(f"  {name:<16} max|Δ| {max_abs:.3e}   argmax agreement {agree:6.2%}   "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = int(diff.max(axis=1).argmax())
        print(f"    worst window {worst}: native {np.round(native[worst], 6)}")
        print(f"                        onnx   {np.round(onnx_out[worst], 6)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="TD-05: ONNX vs native equivalence.")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    art = Path(args.artifacts).resolve()
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    ref = np.load(art / "reference_outputs.npz")

    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__} · python {sys.version.split()[0]}")
    print(f"artefacts   {art}")
    print(f"tolerance   {args.tol:g}\n")

    results = []

    # ---- SVM -----------------------------------------------------------------
    sess = session(art / manifest["models"]["svm"]["file"])
    outs = sess.run(None, {"input": ref["features"]})
    # skl2onnx with zipmap=False emits [label, probabilities]; find the (N,4) one.
    proba = next((o for o in outs if getattr(o, "ndim", 0) == 2 and o.shape[1] == 4), None)
    if proba is None:
        print("FAIL: no (N,4) probability output from the SVM graph; "
              f"got shapes {[getattr(o, 'shape', None) for o in outs]}")
        return 2
    results.append(report("SVM proba", ref["svm_proba"], proba.astype(np.float64)))

    # ---- ResNet --------------------------------------------------------------
    if "resnet_se_cd" in manifest["models"]:
        sess = session(art / manifest["models"]["resnet_se_cd"]["file"])
        logits = sess.run(None, {"input": ref["envelopes"]})[0]
        results.append(report("ResNet proba", ref["resnet_proba"], softmax(logits.astype(np.float64))))

        # ---- ensemble, end to end -------------------------------------------
        ens_native = (ref["svm_proba"] + ref["resnet_proba"]) / 2.0
        ens_onnx = (proba.astype(np.float64) + softmax(logits.astype(np.float64))) / 2.0
        results.append(report("Soft vote", ens_native, ens_onnx))

    # ---- determinism ---------------------------------------------------------
    sess = session(art / manifest["models"]["svm"]["file"])
    a = sess.run(None, {"input": ref["features"][:32]})
    b = sess.run(None, {"input": ref["features"][:32]})
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    print(f"  {'Determinism':<16} repeat run byte-identical: {same}   "
          f"{'PASS' if same else 'FAIL'}")
    results.append(same)

    ok = all(results)
    print(f"\nTD-05 {'PAID — artefacts are faithful to the thesis models.' if ok else 'FAILED — do not deploy.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
