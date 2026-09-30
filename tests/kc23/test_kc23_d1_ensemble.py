"""src/kc23_d1_ensemble.py: the per-seed soft vote and stacking (C13, C13b), built by running src/ensemble_v2_combine.py itself on
a merged probability directory. Real npz layout ({MODEL}_sub{K:02d}.npz with proba and y_true); tiny synthetic data."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
import kc23_d1_ensemble as ens

N_SUB, N_WIN = 40, 24


def _proba(rng, y, sharp):
    p = rng.dirichlet(np.ones(4), len(y)) * (1 - sharp)
    p[np.arange(len(y)), y] += sharp
    return p / p.sum(1, keepdims=True)


def build(root: Path, seed=42):
    rng = np.random.default_rng(seed)
    pub = root / "results/ensemble_v2" / "proba_aug_chandrop"
    r2 = root / f"results/kc23_d1_r2_s{seed}" / "proba"
    pub.mkdir(parents=True); r2.mkdir(parents=True)
    for s in range(1, N_SUB + 1):
        y = rng.integers(0, 4, N_WIN)
        for model, sharp in (("SVM", 0.15), ("RF", 0.12), ("CNN", 0.10), ("RESNET_SE", 0.05)):
            np.savez(pub / f"{model}_sub{s:02d}.npz", proba=_proba(rng, y, sharp), y_true=y.astype(np.int32))
        np.savez(r2 / f"RESNET_SE_CD_sub{s:02d}.npz", proba=_proba(rng, y, 0.6), y_true=y.astype(np.int32))


@pytest.fixture
def root(tmp_path):
    build(tmp_path)
    return tmp_path


def test_builds_soft_and_stacking_per_subject(root):
    out = root / "results/kc23_d1_ensemble_s42"
    assert ens.run(root, 42, out) == 0
    df = pd.read_csv(out / "ensemble_s42.csv")
    assert list(df.columns) == ["subject", "soft", "stacking"] and len(df) == 40
    assert df["soft"].between(0, 1).all() and df["stacking"].between(0, 1).all()


def test_the_seed_specific_resnet_is_what_is_combined_not_the_published_unaugmented_one(root, tmp_path_factory):
    out1 = root / "o1"
    assert ens.run(root, 42, out1) == 0
    a = pd.read_csv(out1 / "ensemble_s42.csv")["soft"].mean()
    # swap the R2 probabilities for the (weaker) published ones: the soft vote must change
    weak = root / "results/kc23_d1_r2_s42" / "proba"
    for f in weak.glob("*.npz"):
        pub = np.load(root / "results/ensemble_v2" / "proba_aug_chandrop" / f.name.replace("RESNET_SE_CD", "RESNET_SE"))
        np.savez(f, proba=pub["proba"], y_true=pub["y_true"])
    out2 = root / "o2"
    assert ens.run(root, 42, out2) == 0
    assert a > pd.read_csv(out2 / "ensemble_s42.csv")["soft"].mean() + 0.01


@pytest.mark.parametrize("victim", ["results/kc23_d1_r2_s42/proba/RESNET_SE_CD_sub07.npz",
                                    "results/ensemble_v2/proba_aug_chandrop/SVM_sub12.npz",
                                    "results/ensemble_v2/proba_aug_chandrop/RF_sub40.npz"])
def test_missing_input_fails_and_writes_nothing(root, victim):
    (root / victim).unlink()
    out = root / "o"
    assert ens.run(root, 42, out) == 1 and not out.exists()


def test_misaligned_rows_fail_closed(root):
    p = root / "results/kc23_d1_r2_s42" / "proba" / "RESNET_SE_CD_sub05.npz"
    z = np.load(p)
    np.savez(p, proba=z["proba"], y_true=(z["y_true"] + 1) % 4)
    assert ens.run(root, 42, root / "o") == 1


def test_wrong_window_count_fails_closed(root):
    p = root / "results/kc23_d1_r2_s42" / "proba" / "RESNET_SE_CD_sub05.npz"
    z = np.load(p)
    np.savez(p, proba=z["proba"][:-3], y_true=z["y_true"][:-3])
    assert ens.run(root, 42, root / "o") == 1


def test_the_published_directory_is_the_chandrop_one():
    assert ens.PUBLISHED_DIR.endswith("proba_aug_chandrop")
