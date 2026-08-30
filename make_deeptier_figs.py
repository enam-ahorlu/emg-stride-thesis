"""Deep-tier figures for Results: model comparison, CNN adaptation, and ensemble combiners."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
OUT="report_figs/new_experiments"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})

# ---- Figure 4.20: overall model comparison (ensemble bar now the NEW best 0.815) ----
models=["LDA","SimpleEMG\nCNN","ResNet\n(no SE)","RF","SVM","ResNet-SE","Ensemble\n(best: soft)"]
f1=[0.687,0.754,0.756,0.773,0.777,0.782,0.815]
sd=[0.072,0.075,0.072,0.060,0.055,0.081,0.062]
colors=["#9e9e9e","#8fb8de","#8fb8de","#f0a860","#e07b39","#2e6f9e","#3a923a"]
order=np.argsort(f1)
fig,ax=plt.subplots(figsize=(8.4,4.3)); x=np.arange(len(models))
ax.bar(x,[f1[i] for i in order],yerr=[sd[i] for i in order],capsize=3,
       color=[colors[i] for i in order],edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.8})
ax.set_xticks(x); ax.set_xticklabels([models[i] for i in order])
ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.60,0.90)
ax.axhline(0.777,ls="--",lw=0.8,color="#666",zorder=0)
ax.text(0.02,0.779,"best classical (SVM 0.777)",fontsize=8,color="#555",transform=ax.get_yaxis_transform())
for i,idx in enumerate(order): ax.text(i,f1[idx]+sd[idx]+0.004,f"{f1[idx]:.3f}",ha="center",fontsize=8.5)
plt.title("Cross-subject (LOSO) macro-F1 by model, per-subject normalization",fontsize=11,pad=12)
plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison_loso.png"); plt.close(); print("wrote model_comparison_loso.png")

# ---- Figure 4.21: CNN-side adaptation ----
meth=["Global norm\n(no adaptation)","AdaBN\n(label-free)","Deep CORAL\n(learned)","Per-subject norm\n(label-free)"]
f1b=[0.703,0.743,0.764,0.782]; sdb=[None,0.061,0.072,0.081]; colb=["#9e9e9e","#8fb8de","#f0a860","#2e6f9e"]
fig,ax=plt.subplots(figsize=(7.4,4.3)); x=np.arange(len(meth))
ax.bar(x,f1b,yerr=[0 if s is None else s for s in sdb],capsize=3,color=colb,edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.8})
ax.set_xticks(x); ax.set_xticklabels(meth); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.60,0.90)
for i,(v,s) in enumerate(zip(f1b,sdb)): ax.text(i,v+(s or 0)+0.004,f"{v:.3f}",ha="center",fontsize=8.5)
plt.title("CNN-side adaptation on the ResNet-SE backbone (SIAT, LOSO, n=40)",fontsize=11,pad=12)
plt.tight_layout(); plt.savefig(f"{OUT}/cnn_adaptation_loso.png"); plt.close(); print("wrote cnn_adaptation_loso.png")

# ---- Figure 4.22: ensemble combiner progression ----
lab=["Best single\n(ResNet-SE)","Hard vote\n(SVM+RF+CNN)","Soft vote\n(SVM+RF+CNN)","Soft/wt-soft\n(SVM+RF+ResNet-SE)"]
v=[0.782,0.792,0.805,0.815]; s=[0.081,0.058,0.062,0.062]
col=["#2e6f9e","#9e9e9e","#8fb8de","#3a923a"]
fig,ax=plt.subplots(figsize=(7.6,4.3)); x=np.arange(len(lab))
ax.bar(x,v,yerr=s,capsize=3,color=col,edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.8})
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.70,0.88)
for i,(a,b) in enumerate(zip(v,s)): ax.text(i,a+b+0.003,f"{a:.3f}",ha="center",fontsize=9)
ax.annotate("+2.3 pp",xy=(3,0.815),xytext=(1.5,0.86),fontsize=9,color="#3a923a",
            arrowprops=dict(arrowstyle="->",color="#3a923a",lw=0.8))
plt.title("Ensemble combiner comparison (SIAT, LOSO, n=40)",fontsize=11,pad=12)
plt.tight_layout(); plt.savefig(f"{OUT}/ensemble_combiner_loso.png"); plt.close(); print("wrote ensemble_combiner_loso.png")
