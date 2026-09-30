"""Figure 3.1  Nested leave-one-subject-out protocol.
Black-and-white, square-cornered boxes, plain black arrows: drawn to look like a
Word-authored figure.  Regenerate with:  python make_fig3_1.py
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

fig = plt.figure(figsize=(10.5, 6.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis('off')

LW = 1.3
FS = 13.0     # body text in boxes
FSB = 14.0    # bold banner text

def box(x, y, w, h, text, fs=FS, weight='normal'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor='black',
                           linewidth=LW, joinstyle='miter', zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight=weight, color='black', zorder=3,
            linespacing=1.4)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=16, linewidth=1.3,
                                 color='black', shrinkA=0, shrinkB=0, zorder=4))

def line(x1, y1, x2, y2):
    ax.add_line(Line2D([x1, x2], [y1, y2], color='black', linewidth=1.3,
                       solid_capstyle='butt', zorder=4))

L_X, L_W = 3, 44          # left column
R_X, R_W = 53, 44         # right column
LC, RC = L_X + L_W / 2, R_X + R_W / 2

# ---------------------------------------------------------------- outer loop
box(3, 87, 94, 11,
    'OUTER LOOP (40 folds):  hold out subject k as TEST,  remaining 39 subjects = TRAIN',
    fs=FSB, weight='bold')

# fork down into the two columns
line(50, 87, 50, 83.5)
line(LC, 83.5, RC, 83.5)
arrow(LC, 83.5, LC, 80)
arrow(RC, 83.5, RC, 80)

# ------------------------------------------------------------ train / test row
box(L_X, 66, L_W, 14,
    'TRAIN: 39 subjects\nper-subject z-score from each\nsubject’s own windows')
box(R_X, 66, R_W, 14,
    'TEST: subject k\nz-scored by k’s OWN unlabeled\nstatistics, labels NEVER used')

# ---------------------------------------------------------------- inner loop
arrow(LC, 66, LC, 60)
box(L_X, 42, L_W, 18,
    'INNER LOOP (5-fold GroupKFold\non the 39 training subjects)\n'
    'GridSearchCV, scoring = macro-F1\n→ select hyperparameters')

# ------------------------------------------------------------------- refit row
arrow(LC, 42, LC, 36)
box(L_X, 23, L_W, 13, 'Refit best model on all 39\ntraining subjects')
box(R_X, 23, R_W, 13, 'Predict held-out subject k\n(once per fold)')

arrow(RC, 66, RC, 36)                    # test subject flows into prediction
arrow(L_X + L_W, 29.5, R_X, 29.5)        # fitted model flows into prediction

# ------------------------------------------------------------------ aggregate
arrow(RC, 23, RC, 16)
box(14, 2.5, 72, 13.5,
    'Aggregate 40 held-out predictions → per-subject macro-F1\n'
    'Wilcoxon signed-rank (Holm/BH-corrected), BCa bootstrap 95% CI',
    fs=FSB, weight='bold')

fig.savefig('fig3_1_new.png', dpi=300, facecolor='white', bbox_inches='tight',
            pad_inches=0.05)
print('wrote fig3_1_new.png')
