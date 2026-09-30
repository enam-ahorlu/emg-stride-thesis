"""Synthetic tests for src/kc23_d3_axis_stats.py: A1/A2/A3/A4."""
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_d3_axis_stats import classify_a


def test_a1_clean_positive():
    # R1=0.70, R3=0.75 (gainjitter helps 5pt); X2 (gaussian sd0.4) no better than R1;
    # X3, X4 well below R3
    fired = classify_a(r1=0.70, r3=0.75, x1=0.705, x2=0.705, x3=0.70, x4=0.70)
    assert "A1" in fired, fired
    assert "A2" not in fired and "A3" not in fired and "A4" not in fired


def test_a2_noise_helps():
    fired = classify_a(r1=0.70, r3=0.75, x1=0.705, x2=0.748, x3=0.70, x4=0.70)
    assert "A2" in fired, fired
    assert "A1" not in fired  # X2 no longer <= R1+1


def test_a3_multiplicative_unsupported():
    fired = classify_a(r1=0.70, r3=0.75, x1=0.705, x2=0.705, x3=0.749, x4=0.70)
    assert "A3" in fired, fired


def test_a4_independence_not_needed():
    fired = classify_a(r1=0.70, r3=0.75, x1=0.705, x2=0.705, x3=0.70, x4=0.749)
    assert "A4" in fired, fired


def test_multiple_cooccur():
    fired = classify_a(r1=0.70, r3=0.75, x1=0.705, x2=0.748, x3=0.749, x4=0.749)
    assert set(fired) >= {"A2", "A3", "A4"}, fired


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
