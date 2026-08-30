"""Frontier figures: models of record = SVM, RF, ResNet-SE, soft-vote ensemble."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, glob, os, csv
OUT="report_figs/new_experiments/frontier"; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})
C={"SVM":"#e07b39","RF":"#f0a860","ResNet-SE":"#2e6f9e","Ensemble":"#3a923a","classical":"#e07b39"}

# ---- 1. per-class frontier ----
classes=["DNS","STDUP","UPS","WAK"]
data={"SVM":[67.68,96.07,76.28,70.23],"RF":[66.19,95.95,75.40,70.94],
      "ResNet-SE":[66.6,95.4,74.4,70.1],"Ensemble":[72.6,96.7,79.6,76.1]}
fig,ax=plt.subplots(figsize=(8,4.3)); x=np.arange(4); w=0.2
for i,m in enumerate(["SVM","RF","ResNet-SE","Ensemble"]):
    ax.bar(x+(i-1.5)*w,data[m],w,label=m,color=C[m],edgecolor="black",linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(classes); ax.set_ylabel("LOSO F1 (%)"); ax.set_ylim(55,100)
ax.legend(ncol=4,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.5,1.12),frameon=False)
ax.set_title("Per-class LOSO F1 by model (per-subject normalization)",fontsize=10,pad=22)
plt.tight_layout(); plt.savefig(f"{OUT}/per_class_frontier.png"); plt.close(); print("1 per_class")

# ---- 2. subject difficulty frontier ----
rows=list(csv.DictReader(open("results_ensemble_v2/subject_difficulty_frontier.csv")))
ens=np.array([float(r["ENS"]) for r in rows]); order=np.argsort(ens)
fig,ax=plt.subplots(figsize=(9,4.3)); xi=np.arange(len(rows))
for m,key,col in [("Ensemble","ENS",C["Ensemble"]),("ResNet-SE","RESNET_SE",C["ResNet-SE"]),("SVM","SVM",C["SVM"]),("RF","RF",C["RF"])]:
    v=np.array([float(r[key]) for r in rows])[order]
    ax.plot(xi,v,marker="o",ms=3,lw=1,label=m,color=col)
ax.set_xlabel("subjects (sorted by ensemble F1)"); ax.set_ylabel("LOSO macro-F1")
ax.legend(fontsize=8.5,frameon=False,ncol=4,loc="lower right")
ax.set_title("Per-subject difficulty across the frontier models (n=40)",fontsize=10)
ax.text(0.02,0.04,"cross-model r: ResNet-SE–SVM 0.68, ResNet-SE–RF 0.75",transform=ax.transAxes,fontsize=8,color="#555")
plt.tight_layout(); plt.savefig(f"{OUT}/subject_difficulty_frontier.png"); plt.close(); print("2 subject_difficulty")

# ---- 3. external ENABL3S headline models ----
lab=["SVM","RF","ResNet-SE","Deep CORAL","AdaBN","Ensemble"]
ps=[0.657,0.636,0.565,0.524,0.542,0.675]; gl=[0.554,0.525,0.476,None,None,None]
fig,ax=plt.subplots(figsize=(8.4,4.3)); x=np.arange(len(lab))
ax.bar(x-0.2,ps,0.4,label="per-subject / method",color="#2e6f9e",edgecolor="black",linewidth=0.3)
ax.bar([i+0.2 for i in range(3)],[gl[i] for i in range(3)],0.4,label="global norm",color="#bbbbbb",edgecolor="black",linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=9); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.40,0.75)
for i,v in enumerate(ps): ax.text(i-0.2,v+0.005,f"{v:.3f}",ha="center",fontsize=7.5)
ax.axvspan(2.5,5.5,color="#f4f4f4",zorder=0)
ax.legend(fontsize=8.5,frameon=False)
ax.set_title("External validation on ENABL3S (n=10): headline models",fontsize=10)
ax.text(4,0.72,"deep models trail classical at n=10",fontsize=8,ha="center",color="#a33")
plt.tight_layout(); plt.savefig(f"{OUT}/external_headline.png"); plt.close(); print("3 external_headline")

# ---- 4. ResNet-SE calibration ----
K=[0,5,10,20]; f=[0.760,0.769,0.787,0.803]; sd=[0.070,0.066,0.067,0.067]
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.errorbar(K,f,yerr=sd,marker="o",capsize=3,color=C["ResNet-SE"],lw=1.5)
for k,v in zip(K,f): ax.text(k,v+0.012,f"{v:.3f}",ha="center",fontsize=8.5)
ax.set_xlabel("labeled calibration windows per class (K)"); ax.set_ylabel("LOSO macro-F1"); ax.set_xticks(K); ax.set_ylim(0.70,0.86)
ax.set_title("Supervised calibration on ResNet-SE (regularized, 5-draw)",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/calibration_resnet_se.png"); plt.close(); print("4 calibration")

# ---- 5. optimization journey to 81.5 ----
stg=["Global-norm\nbaseline","+ Per-subject\nnormalization","+ Hard-vote\nensemble","+ Soft vote\n& ResNet-SE"]
val=[0.708,0.777,0.792,0.815]
fig,ax=plt.subplots(figsize=(7.4,4.3)); x=np.arange(4)
ax.plot(x,val,marker="o",ms=8,lw=2,color="#3a923a")
for i,v in enumerate(val): ax.text(i,v+0.006,f"{v:.3f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(stg,fontsize=8.5); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.68,0.84)
ax.set_title("Optimization journey: global baseline to the 0.815 frontier",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/optimization_journey_815.png"); plt.close(); print("5 journey")

# ---- 6. deployability: classical + deep causal ----
fig,ax=plt.subplots(figsize=(8,4.3))
groups=["SVM (classical)","ResNet-SE / AdaBN (deep)"]
base=[0.708,0.703]; causal=[0.745,0.705]; upper=[0.777,0.782]
x=np.arange(2); w=0.25
ax.bar(x-w,base,w,label="global (no adaptation)",color="#bbbbbb",edgecolor="black",linewidth=0.3)
ax.bar(x,causal,w,label="causal / deployable",color="#8fb8de",edgecolor="black",linewidth=0.3)
ax.bar(x+w,upper,w,label="offline upper bound",color="#2e6f9e",edgecolor="black",linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.65,0.82)
ax.legend(fontsize=8,frameon=False)
ax.set_title("Deployable fraction under causal estimation (classical vs deep)",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/deployability_frontier.png"); plt.close(); print("6 deployability")

# ---- 7. confusion (ResNet-SE + ensemble) ----
def load(m):
    d={}
    for fp in glob.glob(f"results_ensemble_v2/proba/{m}_sub*.npz"):
        s=int(fp.split("_sub")[1][:2]); z=np.load(fp); d[s]=(z["y_true"].astype(int),z["proba"].astype(float))
    return d
P={m:load(m) for m in ["SVM","RF","RESNET_SE"]}; subs=sorted(P["RESNET_SE"])
def conf(predfn):
    Cm=np.zeros((4,4))
    for s in subs:
        n=min(len(P[m][s][0]) for m in P); yt=P["RESNET_SE"][s][0][:n]; yp=predfn(s,n)
        for t,p in zip(yt,yp): Cm[t,p]+=1
    return Cm/Cm.sum(1,keepdims=True)*100
cse=conf(lambda s,n:P["RESNET_SE"][s][1][:n].argmax(1))
cen=conf(lambda s,n:((P["SVM"][s][1][:n]+P["RF"][s][1][:n]+P["RESNET_SE"][s][1][:n])/3).argmax(1))
fig,axes=plt.subplots(1,2,figsize=(9,4.2))
for ax,Cm,ttl in zip(axes,[cse,cen],["ResNet-SE","Soft-vote ensemble"]):
    im=ax.imshow(Cm,cmap="Blues",vmin=0,vmax=100)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(ttl,fontsize=10)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{Cm[i,j]:.0f}",ha="center",va="center",fontsize=8,color="white" if Cm[i,j]>50 else "black")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_frontier.png"); plt.close(); print("7 confusion")
print("ALL FRONTIER FIGURES WRITTEN to",OUT)
