"""Figure 5.2  The three adaptation regimes and where this thesis sits.
Black-and-white, square-cornered panels separated so the arrows sit between
them.  Type is sized so that the figure stays legible at 6.15 in on the page.
Regenerate with:  python make_fig5_2.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'DejaVu Sans', 'Arial'],
    'text.color': 'black',
})

FIG_W, FIG_H = 12.6, 7.8
PAGE_W = 6.15                     # width the figure is placed at in the thesis
S = FIG_W / PAGE_W                # point sizes are multiplied by this so that
                                  # the rendered size is what the numbers say
def pt(x):
    return x * S

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
# margin on the limits so the outermost panel borders are not clipped
ax.set_xlim(-1.5, 101.5); ax.set_ylim(-1.5, 101.5)
ax.axis('off')

BW, GAP = 31.0, 3.5
X0 = 0.5          # panel 3 right edge lands at 100.5, inside the limits above
BY, BH = 5.0, 87.0
LW = 1.2

PANELS = [
    dict(title='1. Inductive',
         sub='no target-subject data',
         body=('Model trained on the\n'
               'source subjects and\n'
               'applied unchanged.'),
         methods='global normalization',
         rows=[('SVM', '0.708'), ('RF', '0.722'), ('ResNet-SE+CD', '0.787')]),
    dict(title='2. Transductive',
         sub='unlabeled target-subject data',
         body=('The subject’s own\n'
               'unlabeled windows give\n'
               'the statistics. No labels.'),
         methods='per-subject normalization,\nAdaBN, CORAL',
         rows=[('SVM', '0.777'), ('ResNet-SE+CD', '0.840'), ('Ensemble', '0.858')]),
    dict(title='3. Supervised calibration',
         sub='labeled target-subject data',
         body=('A small labeled set from\n'
               'the subject fine-tunes the\n'
               'model. Costs user effort.'),
         methods='regularized fine-tuning\n(K windows per class)',
         rows=[('K = 5', '0.830'), ('K = 10', '0.843'), ('K = 20', '0.854')]),
]

def rule(x1, x2, y, lw=LW):
    ax.add_line(Line2D([x1, x2], [y, y], color='black', linewidth=lw, zorder=3))

for i, p in enumerate(PANELS):
    x = X0 + i * (BW + GAP)
    ax.add_patch(Rectangle((x, BY), BW, BH, facecolor='white', edgecolor='black',
                           linewidth=LW, joinstyle='miter', zorder=2))
    cx, lx = x + BW / 2, x + 1.8
    ax.text(cx, BY + BH - 7.0, p['title'], ha='center', va='center',
            fontsize=pt(9.8), fontweight='bold', zorder=3)
    ax.text(cx, BY + BH - 14.0, p['sub'], ha='center', va='center',
            fontsize=pt(7.8), style='italic', zorder=3)
    rule(x, x + BW, BY + BH - 19.0)

    ax.text(lx, BY + BH - 23.0, p['body'], ha='left', va='top',
            fontsize=pt(8.0), linespacing=1.5, zorder=3)

    ax.text(lx, BY + 40.0, 'METHODS IN THIS THESIS', ha='left', va='top',
            fontsize=pt(6.8), fontweight='bold', zorder=3)
    ax.text(lx, BY + 34.5, p['methods'], ha='left', va='top',
            fontsize=pt(8.0), linespacing=1.5, zorder=3)

    rule(x, x + BW, BY + 22.0)
    ax.text(lx, BY + 17.5, 'LOSO macro-F1', ha='left', va='center',
            fontsize=pt(7.8), fontweight='bold', zorder=3)
    for j, (lab, val) in enumerate(p['rows']):
        yy = BY + 12.0 - j * 3.8
        ax.text(lx, yy, lab, ha='left', va='center', fontsize=pt(8.0), zorder=3)
        ax.text(x + BW - 1.8, yy, val, ha='right', va='center',
                fontsize=pt(8.0), fontweight='bold', zorder=3)

for i in range(2):
    xa = X0 + (i + 1) * BW + i * GAP
    ax.add_patch(FancyArrowPatch((xa + 0.5, BY + BH / 2), (xa + GAP - 0.5, BY + BH / 2),
                                 arrowstyle='-|>', mutation_scale=pt(9), linewidth=1.3,
                                 color='black', shrinkA=0, shrinkB=0, zorder=4))

ax.text(50, 96.5, 'Increasing demand on the target subject',
        ha='center', va='center', fontsize=pt(8.2), style='italic', zorder=3)
ax.text(0.75, 1.6,
        'Under a strictly causal 100-window calibration buffer the ensemble '
        'retains 0.817 of its 0.858 offline figure.',
        ha='left', va='center', fontsize=pt(7.6), zorder=3)

fig.savefig('fig5_2_new.png', dpi=300, facecolor='white', bbox_inches='tight',
            pad_inches=0.04)
print('wrote fig5_2_new.png')
