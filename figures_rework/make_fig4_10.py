"""Figure 4.10  Single-electrode occlusion cost, SE-free residual backbone.
Two panels: (a) per-channel macro-F1 lost to occluding each of the nine
electrodes, un-augmented against channel-dropout; (b) per-subject total
occlusion cost, the two arms paired by subject.

Black-and-white, square-cornered, no colour, to sit beside Figures 3.1 and 5.2.
Drawn from the RAW drop_pp values in the two instr/occlusion.csv files, not the
normalized g3_profiles_*.csv (Section 4.8.2 states the normalized measures are
scale artifacts). Point sizes are multiplied by FIG_W / PAGE_W so the numbers in
this source are the sizes seen at 6.15 in page width (see make_fig5_2.py).

Regenerate with:  python make_fig4_10.py
Produces fig4_10_new.png (panel b as connected lines) and, for the Section 1.4
decision, fig4_10_dotplot.png (panel b as a paired dot plot).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO = ROOT / "results_g3_noaug_instr" / "instr" / "occlusion.csv"
CD = ROOT / "results_cd_resnet_nose_chandrop" / "instr" / "occlusion.csv"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'DejaVu Sans', 'Arial'],
    'text.color': 'black', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 1.0,
})

FIG_W, FIG_H, PAGE_W = 12.4, 5.8, 6.15
S = FIG_W / PAGE_W
def pt(x): return x * S


def wide(path):
    df = pd.read_csv(path)
    return df.pivot(index='subject', columns='channel', values='drop_pp').sort_index()


wn, wc = wide(NO), wide(CD)
ch = np.arange(9)                              # channel indices 0..8, as stored
mn, sn = wn.mean(0).to_numpy(), wn.std(0, ddof=1).to_numpy()
mc, sc = wc.mean(0).to_numpy(), wc.std(0, ddof=1).to_numpy()
tot_n, tot_c = wn.sum(1).to_numpy(), wc.sum(1).to_numpy()


def panel_a(ax):
    w = 0.38
    ax.bar(ch - w / 2, mn, w, yerr=sn, facecolor='white', edgecolor='black',
           linewidth=1.3, error_kw=dict(ecolor='black', elinewidth=1.0, capsize=pt(2.2)),
           label='no augmentation', zorder=3)
    ax.bar(ch + w / 2, mc, w, yerr=sc, facecolor='black', edgecolor='black',
           linewidth=1.3, error_kw=dict(ecolor='black', elinewidth=1.0, capsize=pt(2.2)),
           label='channel dropout', zorder=3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(ch)
    ax.set_xlabel('EMG channel index', fontsize=pt(8))
    ax.set_ylabel('mean macro-F1 lost to occlusion (pp)', fontsize=pt(8))
    ax.set_title('(a) per-channel occlusion cost', fontsize=pt(9), fontweight='bold',
                 loc='left', pad=pt(4))
    ax.tick_params(labelsize=pt(7.5))
    ax.legend(fontsize=pt(7.2), frameon=True, edgecolor='black', handlelength=1.6, loc='upper left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def panel_b_lines(ax):
    x0, x1 = 0.0, 1.0
    for a, b in zip(tot_n, tot_c):
        ax.plot([x0, x1], [a, b], color='black', linewidth=0.6, solid_capstyle='butt', zorder=2)
    ax.plot(np.full_like(tot_n, x0), tot_n, 'o', mfc='white', mec='black', mew=1.0,
            ms=pt(3.2), zorder=3)
    ax.plot(np.full_like(tot_c, x1), tot_c, 'o', mfc='black', mec='black', mew=1.0,
            ms=pt(3.2), zorder=3)
    ax.plot([x0 - 0.16, x0 + 0.16], [tot_n.mean(), tot_n.mean()], color='black', linewidth=2.6, zorder=5)
    ax.plot([x1 - 0.16, x1 + 0.16], [tot_c.mean(), tot_c.mean()], color='black', linewidth=2.6, zorder=5)
    ax.text(x0 - 0.30, tot_n.mean(), f'mean {tot_n.mean():.1f}', ha='right', va='center', fontsize=pt(7.4))
    ax.text(x1 + 0.30, tot_c.mean(), f'mean {tot_c.mean():.1f}', ha='left', va='center', fontsize=pt(7.4))
    ax.axhline(0, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
    _finish_b(ax)


def panel_b_dots(ax):
    rng = np.random.default_rng(42)
    for x, tot in ((0.0, tot_n), (1.0, tot_c)):
        jit = x + rng.uniform(-0.12, 0.12, size=tot.size)
        mfc = 'white' if x == 0.0 else 'black'
        ax.plot(jit, tot, 'o', mfc=mfc, mec='black', mew=1.0, ms=pt(3.2), zorder=3)
        ax.plot([x - 0.24, x + 0.24], [tot.mean(), tot.mean()], color='black',
                linewidth=2.8, zorder=5)
        side = -0.36 if x == 0.0 else 0.36
        ha = 'right' if x == 0.0 else 'left'
        ax.text(x + side, tot.mean(), f'mean {tot.mean():.1f}', ha=ha, va='center',
                fontsize=pt(7.4))
    ax.axhline(0, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
    _finish_b(ax)


def _finish_b(ax):
    ax.set_xlim(-1.05, 2.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['no\naugmentation', 'channel\ndropout'])
    lo = min(tot_c.min(), tot_n.min()) - 12
    hi = max(tot_c.max(), tot_n.max()) + 14
    ax.set_ylim(lo, hi)                       # negative minimum is not clipped
    ax.set_ylabel('total macro-F1 lost across all nine channels (pp)', fontsize=pt(8))
    ax.set_title('(b) per-subject total occlusion cost', fontsize=pt(9),
                 fontweight='bold', loc='left', pad=pt(4))
    ax.tick_params(labelsize=pt(7.5))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def build(b_fn, out):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=300,
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    panel_a(axes[0])
    b_fn(axes[1])
    fig.tight_layout(pad=0.8, w_pad=2.0)
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    build(panel_b_lines, ROOT / 'figures_rework' / 'fig4_10_new.png')
    build(panel_b_dots, ROOT / 'figures_rework' / 'fig4_10_dotplot.png')
    print(f'panel (a) channel means noaug : {np.round(mn,2).tolist()}')
    print(f'panel (a) channel means CD    : {np.round(mc,2).tolist()}')
    print(f'panel (b) totals noaug  mean {tot_n.mean():.2f}  range [{tot_n.min():.2f}, {tot_n.max():.2f}]')
    print(f'panel (b) totals CD     mean {tot_c.mean():.2f}  range [{tot_c.min():.2f}, {tot_c.max():.2f}]')
    print(f'ratio {tot_n.mean()/tot_c.mean():.2f}x   negative-total subjects: {(tot_c<0).sum()}')
