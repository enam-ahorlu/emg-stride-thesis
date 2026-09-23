"""Unit tests for kc23_c4_extract_rich.py's TD-PSD moments on a synthetic
sinusoid of known amplitude/frequency, plus a structural test for Rich-126's
AR(4)/Hjorth blocks."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c4_extract_rich import tdpsd_one_channel, tdpsd54, hjorth, ar4_burg, rich126

FS = 1920.0
T = 480  # 250ms at 1920Hz


def sinusoid(freq_hz, amp=1.0, fs=FS, n=T):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t)


def test_m0_scales_with_amplitude():
    x1 = sinusoid(20.0, amp=1.0)
    x2 = sinusoid(20.0, amp=2.0)
    f1 = tdpsd_one_channel(x1)
    f2 = tdpsd_one_channel(x2)
    # f1 = slog(m0): m0 doubles with amplitude, so f1 (log-scale) should increase
    assert f2[0] > f1[0], (f1, f2)


def test_sparseness_amplitude_invariant():
    """Sparseness/IRF are ratio-based, so scaling the whole signal's amplitude
    should leave the log-sparseness (f4) and log-IRF (f5) features
    approximately unchanged -- a known property of TD-PSD's moment ratios,
    and a good internal-consistency check for a correct implementation."""
    x1 = sinusoid(20.0, amp=1.0)
    x2 = sinusoid(20.0, amp=5.0)
    f1 = tdpsd_one_channel(x1)
    f2 = tdpsd_one_channel(x2)
    assert abs(f1[3] - f2[3]) < 0.05, (f1[3], f2[3])  # sparseness
    assert abs(f1[4] - f2[4]) < 0.05, (f1[4], f2[4])  # IRF


def test_higher_frequency_raises_m2_relative_to_m0():
    """A higher-frequency sinusoid has a proportionally larger first
    difference (its derivative amplitude scales with frequency), so m2/m0
    should rise with frequency at fixed amplitude -- i.e. f2 = log(m0-m2)
    should FALL (m0-m2 shrinks or goes more negative) as frequency rises."""
    x_low = sinusoid(5.0, amp=1.0)
    x_high = sinusoid(60.0, amp=1.0)
    f_low = tdpsd_one_channel(x_low)
    f_high = tdpsd_one_channel(x_high)
    assert f_high[1] < f_low[1], (f_low, f_high)


def test_tdpsd54_shape():
    N, C = 5, 9
    X = np.stack([np.stack([sinusoid(10 + 3 * c, amp=1.0) for c in range(C)]) for _ in range(N)])
    feats = tdpsd54(X)
    assert feats.shape == (N, 54)
    assert np.isfinite(feats).all()


def test_hjorth_constant_signal_zero_mobility():
    x = np.ones(480) * 3.0
    mob, comp = hjorth(x)
    assert mob < 1e-6, mob  # zero variance derivative on a constant signal


def test_hjorth_higher_freq_higher_mobility():
    x_low = sinusoid(5.0)
    x_high = sinusoid(60.0)
    mob_low, _ = hjorth(x_low)
    mob_high, _ = hjorth(x_high)
    assert mob_high > mob_low, (mob_low, mob_high)


def test_ar4_burg_shape_and_stability():
    x = sinusoid(15.0) + np.random.default_rng(0).normal(0, 0.05, T)
    a = ar4_burg(x)
    assert a.shape == (4,)
    assert np.isfinite(a).all()


def test_rich126_shape():
    N, C = 4, 9
    X = np.stack([np.stack([sinusoid(10 + 3 * c, amp=1.0) for c in range(C)]) for _ in range(N)])
    freq72 = np.random.default_rng(0).normal(size=(N, 72))
    r = rich126(X, freq72)
    assert r.shape == (N, 126)
    assert np.array_equal(r[:, :72], freq72)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
