"""Synthetic tests for kc23_c3_merge_proba.py: the KC-C3 ensemble data-prep step."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c3_merge_proba import merge


def _write_npz_set(d: Path, model: str, n_subjects: int):
    d.mkdir(parents=True, exist_ok=True)
    for s in range(1, n_subjects + 1):
        np.savez(d / f"{model}_sub{s:02d}.npz", proba=np.zeros((5, 4)), y_true=np.zeros(5, dtype=int))


def test_merge_copies_new_svm_and_published_others(tmp_path):
    new_svm = tmp_path / "new_svm"
    published = tmp_path / "published"
    out = tmp_path / "merged"
    _write_npz_set(new_svm, "SVM", 40)
    for m in ["RF", "CNN", "RESNET_SE"]:
        _write_npz_set(published, m, 40)
    counts = merge(new_svm, published, out)
    assert counts == {"SVM": 40, "RF": 40, "CNN": 40, "RESNET_SE": 40}
    assert len(list(out.glob("SVM_sub*.npz"))) == 40
    assert len(list(out.glob("RESNET_SE_sub*.npz"))) == 40


def test_merge_does_not_delete_sources(tmp_path):
    new_svm = tmp_path / "new_svm"
    published = tmp_path / "published"
    out = tmp_path / "merged"
    _write_npz_set(new_svm, "SVM", 40)
    for m in ["RF", "CNN", "RESNET_SE"]:
        _write_npz_set(published, m, 40)
    merge(new_svm, published, out)
    assert len(list(new_svm.glob("SVM_sub*.npz"))) == 40
    assert len(list(published.glob("RF_sub*.npz"))) == 40


def test_merge_raises_on_missing_svm(tmp_path):
    new_svm = tmp_path / "empty"
    new_svm.mkdir()
    published = tmp_path / "published"
    for m in ["RF", "CNN", "RESNET_SE"]:
        _write_npz_set(published, m, 40)
    with pytest.raises(FileNotFoundError):
        merge(new_svm, published, tmp_path / "merged")


def test_merge_warns_on_mismatched_counts(tmp_path, capsys):
    import kc23_c3_merge_proba as mod
    new_svm = tmp_path / "new_svm"
    published = tmp_path / "published"
    _write_npz_set(new_svm, "SVM", 39)  # one short
    for m in ["RF", "CNN", "RESNET_SE"]:
        _write_npz_set(published, m, 40)
    counts = merge(new_svm, published, tmp_path / "merged")
    assert counts["SVM"] == 39 and counts["RF"] == 40
