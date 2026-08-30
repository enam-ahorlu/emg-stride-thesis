import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os
OUT="report_figs/new_experiments/frontier"; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})
models=["SVM","RF","CNN","ResNet-SE"]
SIAT_g=[0.708,0.722,0.682,0.703]; SIAT_p=[0.777,0.773,0.754,0.782]
EN_g  =[0.554,0.525,0.387,0.476]; EN_p  =[0.657,0.636,0.556,0.565]
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=False)
for ax,(g,p,ttl,n) in zip(axes,[(SIAT_g,SIAT_p,"SIAT",40),(EN_g,EN_p,"ENABL3S",10)]):
    x=np.arange(len(models)); w=0.38
    ax.bar(x-w/2,g,w,label="global norm",color="#bbbbbb",edgecolor="black",linewidth=0.3)
    ax.bar(x+w/2,p,w,label="per-subject norm",color="#2e6f9e",edgecolor="black",linewidth=0.3)
    for i in range(len(models)):
        ax.annotate(f"+{(p[i]-g[i])*100:.0f}",xy=(i,p[i]+0.008),ha="center",fontsize=8,color="#3a923a")
    ax.set_xticks(x); ax.set_xticklabels(models,fontsize=9); ax.set_title(f"{ttl} (n={n})",fontsize=10)
    ax.set_ylim(0.30 if ttl=="ENABL3S" else 0.60, 0.85)
    if ttl=="SIAT": ax.set_ylabel("LOSO macro-F1")
axes[0].legend(fontsize=8.5,frameon=False,loc="lower right")
fig.suptitle("Per-subject vs global normalization across model families, on both datasets",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/norm_replication_frontier.png"); plt.close(); print("wrote norm_replication_frontier.png")
