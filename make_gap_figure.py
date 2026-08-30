#!/usr/bin/env python3
"""
make_gap_figure.py — Results Figure 4.14 (generalization gap: baseline vs optimized).
Uses the thesis's canonical primary-config numbers (Table 4.5): baseline gaps
16.6/12.1/22.2 pp -> optimized 9.7/7.0/15.1 pp (per-subject normalization),
reductions 42%/42%/32%. This matches the Abstract/Conclusion/Discussion §5.2.
Output: report_figs/new_experiments/generalization_gap_comparison.png
"""
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).parent/"report_figs"/"new_experiments"; OUT.mkdir(parents=True,exist_ok=True)
models=["SVM","RF","CNN"]
baseline=[0.166,0.121,0.222]; optimized=[0.097,0.070,0.151]
delta=[round(b-o,3) for b,o in zip(baseline,optimized)]; red=[42,42,32]
x=np.arange(3); w=0.38
fig,ax=plt.subplots(figsize=(9,6))
ax.bar(x-w/2,baseline,w,label="Baseline (global norm)",color="#e8666e",edgecolor="black")
ax.bar(x+w/2,optimized,w,label="Optimized (per-subject norm)",color="#5cb85c",edgecolor="black")
for xi,v in zip(x-w/2,baseline): ax.text(xi,v+0.004,f"{v:.3f}",ha="center",fontweight="bold")
for xi,v in zip(x+w/2,optimized): ax.text(xi,v+0.004,f"{v:.3f}",ha="center",fontweight="bold")
for i in range(3):
    ax.text(x[i]+w/2,optimized[i]+0.028,f"-{delta[i]:.3f}\n({red[i]}%)",ha="center",color="#2c6fbb",fontweight="bold",fontsize=9)
ax.set_ylabel("Generalization Gap (SD F1 - LOSO F1)")
ax.set_title("Generalization Gap: Baseline vs Optimized\n(Freq-72, 40-Subject LOSO)")
ax.set_xticks(x); ax.set_xticklabels(models); ax.legend(loc="upper left"); ax.set_ylim(0,0.27); ax.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"generalization_gap_comparison.png",dpi=150,bbox_inches="tight"); print("saved",OUT/"generalization_gap_comparison.png")
