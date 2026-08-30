"""Deep-tier figures (CD frontier): model comparison, CNN adaptation, ensemble combiners."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
OUT="report_figs/new_experiments"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})

# ---- Fig 4.3: model comparison (ResNet-SE+CD 0.840, Ensemble 0.858) ----
models=["LDA","SimpleEMG\nCNN","ResNet\n(no SE)","RF","SVM","ResNet-SE","Ensemble\n(best: soft)"]
f1=[0.687,0.754,0.756,0.773,0.777,0.782,0.858]
sd=[0.072,0.075,0.072,0.060,0.055,0.081,0.065]
colors=["#9e9e9e","#8fb8de","#8fb8de","#f0a860","#e07b39","#2e6f9e","#3a923a"]
order=np.argsort(f1)
fig,ax=plt.subplots(figsize=(8.4,4.3)); x=np.arange(len(models))
ax.bar(x,[f1[i] for i in order],yerr=[sd[i] for i in order],capsize=3,
       color=[colors[i] for i in order],edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.8})
ax.set_xticks(x); ax.set_xticklabels([models[i] for i in order])
ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.60,0.92)
ax.axhline(0.777,ls="--",lw=0.8,color="#666",zorder=0)
ax.text(0.02,0.779,"best classical (SVM 0.777)",fontsize=8,color="#555",transform=ax.get_yaxis_transform())
for i,idx in enumerate(order): ax.text(i,f1[idx]+sd[idx]+0.004,f"{f1[idx]:.3f}",ha="center",fontsize=8.5)
plt.title("Cross-subject (LOSO) macro-F1 by model, per-subject normalization",fontsize=11,pad=12)
plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison_loso.png"); plt.close(); print("wrote model_comparison_loso.png")

# ---- Fig 4.19: CNN-side adaptation on the ResNet-SE+CD backbone ----
meth=["Global norm\n(no adaptation)","AdaBN\n(label-free)","Deep CORAL\n(learned)","Per-subject norm\n(label-free)"]
f1b=[0.787,0.818,0.826,0.840]; sdb=[None,0.060,0.065,0.067]; colb=["#9e9e9e","#8fb8de","#f0a860","#2e6f9e"]
fig,ax=plt.subplots(figsize=(7.4,4.3)); x=np.arange(len(meth))
ax.bar(x,f1b,yerr=[0 if s is None else s for s in sdb],capsize=3,color=colb,edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.8})
ax.set_xticks(x); ax.set_xticklabels(meth); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.60,0.92)
for i,(v,s) in enumerate(zip(f1b,sdb)): ax.text(i,v+(s or 0)+0.004,f"{v:.3f}",ha="center",fontsize=8.5)
plt.title("CNN-side adaptation on the ResNet-SE+CD backbone (SIAT, LOSO, n=40)",fontsize=11,pad=12)
plt.tight_layout(); plt.savefig(f"{OUT}/cnn_adaptation_loso.png"); plt.close(); print("wrote cnn_adaptation_loso.png")

# ---- Fig 4.12: ensemble combiner / composition ----
lab=["Best single\n(ResNet-SE+CD)","Classical soft\n(SVM+RF)","3-model soft\n(+ResNet-SE+CD)","Best soft\n(SVM+ResNet-SE+CD)"]
v=[0.840,0.786,0.847,0.858]; s=[0.067,0.053,0.061,0.065]
col=["#2e6f9e","#9e9e9e","#8fb8de","#3a923a"]
fig,ax=plt.subplots(figsize=(7.8,4.3)); x=np.arange(len(lab))
ax.bar(x,v,yerr=s,capsize=3,color=col,edgecolor="black",linewidth=0.4,error_kw={"elinewidth":0.8})
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=8.5); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.70,0.90)
for i,(a,b) in enumerate(zip(v,s)): ax.text(i,a+b+0.003,f"{a:.3f}",ha="center",fontsize=9)
ax.annotate("+1.8 pp",xy=(3,0.858),xytext=(1.4,0.885),fontsize=9,color="#3a923a",
            arrowprops=dict(arrowstyle="->",color="#3a923a",lw=0.8))
plt.title("Ensemble composition (SIAT, LOSO, n=40): soft vote over SVM + ResNet-SE+CD",fontsize=10,pad=12)
plt.tight_layout(); plt.savefig(f"{OUT}/ensemble_combiner_loso.png"); plt.close(); print("wrote ensemble_combiner_loso.png")
print("DONE deeptier CD figs")
