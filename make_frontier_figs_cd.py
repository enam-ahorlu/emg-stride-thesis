"""Frontier figures, ResNet-SE+CD (channel-dropout) as deep model of record.
Models of record = SVM, RF, ResNet-SE+CD, soft-vote ensemble (SVM+ResNet-SE+CD)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, glob, os, csv, re
OUT="report_figs/new_experiments/frontier"; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})
C={"SVM":"#e07b39","RF":"#f0a860","ResNet-SE+CD":"#2e6f9e","Ensemble":"#3a923a"}
classes=["DNS","STDUP","UPS","WAK"]
DPROB="results_ensemble_v2/proba_aug_chandrop"

def load(pref):
    d={}
    for fp in glob.glob(f"{DPROB}/{pref}_sub*.npz"):
        m=re.search(r'sub0*(\d+)',fp); z=np.load(fp)
        d[int(m.group(1))]=(z["y_true"].astype(int),z["proba"].astype(float))
    return d
P={m:load(m) for m in ["SVM","RF","RESNET_SE"]}
subs=sorted(set(P["SVM"])&set(P["RESNET_SE"]))
def align(s):
    n=min(len(P["SVM"][s][0]),len(P["RESNET_SE"][s][0]))
    return n,P["RESNET_SE"][s][0][:n]
def cdpred(s): n,_=align(s); return P["RESNET_SE"][s][1][:n].argmax(1)
def enpred(s): n,_=align(s); return ((P["SVM"][s][1][:n]+P["RESNET_SE"][s][1][:n])/2).argmax(1)
def svmpred(s): n,_=align(s); return P["SVM"][s][1][:n].argmax(1)
def rfpred(s): n,_=align(s); return P["RF"][s][1][:n].argmax(1)
def subjf1(predfn):
    out={}
    for s in subs:
        n,yt=align(s); yp=predfn(s); fs=[]
        for c in range(4):
            tp=np.sum((yp==c)&(yt==c));fp=np.sum((yp==c)&(yt!=c));fn=np.sum((yp!=c)&(yt==c))
            pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
            fs.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        out[s]=np.mean(fs)
    return out

# ---- 1. per-class frontier ----
data={"SVM":[67.68,96.07,76.28,70.23],"RF":[66.19,95.95,75.40,70.94],
      "ResNet-SE+CD":[80.0,97.1,80.0,77.8],"Ensemble":[81.1,97.4,83.3,80.0]}
fig,ax=plt.subplots(figsize=(8,4.3)); x=np.arange(4); w=0.2
for i,m in enumerate(["SVM","RF","ResNet-SE+CD","Ensemble"]):
    ax.bar(x+(i-1.5)*w,data[m],w,label=m,color=C[m],edgecolor="black",linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(classes); ax.set_ylabel("LOSO F1 (%)"); ax.set_ylim(55,100)
ax.legend(ncol=4,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.5,1.12),frameon=False)
ax.set_title("Per-class LOSO F1 by model (per-subject normalization)",fontsize=10,pad=22)
plt.tight_layout(); plt.savefig(f"{OUT}/per_class_frontier.png"); plt.close(); print("1 per_class")

# ---- 2. subject difficulty frontier (+ write CSV) ----
Fsvm=subjf1(svmpred); Frf=subjf1(rfpred); Fcd=subjf1(cdpred); Fen=subjf1(enpred)
with open("results_ensemble_v2/subject_difficulty_frontier_cd.csv","w",newline="") as fo:
    wr=csv.writer(fo); wr.writerow(["subject","SVM","RF","RESNET_SE_CD","ENS"])
    for s in subs: wr.writerow([s,Fsvm[s],Frf[s],Fcd[s],Fen[s]])
ens=np.array([Fen[s] for s in subs]); order=np.argsort(ens); xi=np.arange(len(subs))
rSVM=np.corrcoef([Fcd[s] for s in subs],[Fsvm[s] for s in subs])[0,1]
rRF=np.corrcoef([Fcd[s] for s in subs],[Frf[s] for s in subs])[0,1]
fig,ax=plt.subplots(figsize=(9,4.3))
for m,arr,col in [("Ensemble",Fen,C["Ensemble"]),("ResNet-SE+CD",Fcd,C["ResNet-SE+CD"]),("SVM",Fsvm,C["SVM"]),("RF",Frf,C["RF"])]:
    v=np.array([arr[s] for s in subs])[order]; ax.plot(xi,v,marker="o",ms=3,lw=1,label=m,color=col)
ax.set_xlabel("subjects (sorted by ensemble F1)"); ax.set_ylabel("LOSO macro-F1")
ax.legend(fontsize=8.5,frameon=False,ncol=4,loc="lower right")
ax.set_title("Per-subject difficulty across the frontier models (n=40)",fontsize=10)
ax.text(0.02,0.04,f"cross-model r: ResNet-SE+CD-SVM {rSVM:.2f}, ResNet-SE+CD-RF {rRF:.2f}",transform=ax.transAxes,fontsize=8,color="#555")
plt.tight_layout(); plt.savefig(f"{OUT}/subject_difficulty_frontier.png"); plt.close(); print("2 subject_difficulty")

# ---- 3. external ENABL3S headline models ----
lab=["SVM","RF","ResNet-SE+CD","Ensemble"]
ps=[0.657,0.636,0.637,0.706]; gl=[0.554,0.525,0.504,None]
fig,ax=plt.subplots(figsize=(7.6,4.3)); x=np.arange(len(lab))
ax.bar(x-0.2,ps,0.4,label="per-subject / method",color="#2e6f9e",edgecolor="black",linewidth=0.3)
ax.bar([i+0.2 for i in range(3)],[gl[i] for i in range(3)],0.4,label="global norm",color="#bbbbbb",edgecolor="black",linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=9); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.40,0.78)
for i,v in enumerate(ps): ax.text(i-0.2,v+0.006,f"{v:.3f}",ha="center",fontsize=7.5)
ax.legend(fontsize=8.5,frameon=False,loc="upper left")
ax.set_title("External validation on ENABL3S (n=10): headline models",fontsize=10)
ax.text(2.5,0.74,"deep model matches classical solo;\nensemble best at 0.706",fontsize=7.5,ha="center",color="#a33")
plt.tight_layout(); plt.savefig(f"{OUT}/external_headline.png"); plt.close(); print("3 external_headline")

# ---- 4. ResNet-SE+CD calibration ----
K=[0,5,10,20]; f=[0.818,0.830,0.843,0.854]; sd=[0.069,0.068,0.065,0.064]
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.errorbar(K,f,yerr=sd,marker="o",capsize=3,color=C["ResNet-SE+CD"],lw=1.5)
for k,v in zip(K,f): ax.text(k,v+0.012,f"{v:.3f}",ha="center",fontsize=8.5)
ax.set_xlabel("labeled calibration windows per class (K)"); ax.set_ylabel("LOSO macro-F1"); ax.set_xticks(K); ax.set_ylim(0.78,0.90)
ax.set_title("Supervised calibration on ResNet-SE+CD (regularized, 5-draw)",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/calibration_resnet_se.png"); plt.close(); print("4 calibration")

# ---- 5. optimization journey to 0.858 ----
stg=["Global-norm\nbaseline","+ Per-subject\nnormalization","+ Hard-vote\nensemble","+ Soft vote\n& ResNet-SE","+ Channel-dropout\nResNet-SE"]
val=[0.708,0.777,0.792,0.815,0.858]
fig,ax=plt.subplots(figsize=(8.2,4.3)); x=np.arange(len(val))
ax.plot(x,val,marker="o",ms=8,lw=2,color="#3a923a")
for i,v in enumerate(val): ax.text(i,v+0.006,f"{v:.3f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(stg,fontsize=8); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.68,0.88)
ax.set_title("Optimization journey: global baseline to the 0.858 frontier",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/optimization_journey_815.png"); plt.close(); print("5 journey")

# ---- 6. deployability: classical + deep causal (CD backbone) ----
fig,ax=plt.subplots(figsize=(8,4.3))
groups=["SVM (classical)","ResNet-SE+CD / AdaBN (deep)"]
base=[0.708,0.787]; causal=[0.745,0.768]; upper=[0.777,0.840]
x=np.arange(2); w=0.25
ax.bar(x-w,base,w,label="global (no adaptation)",color="#bbbbbb",edgecolor="black",linewidth=0.3)
ax.bar(x,causal,w,label="causal / deployable",color="#8fb8de",edgecolor="black",linewidth=0.3)
ax.bar(x+w,upper,w,label="offline upper bound",color="#2e6f9e",edgecolor="black",linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.65,0.87)
ax.legend(fontsize=8,frameon=False)
ax.set_title("Deployable fraction under causal estimation (classical vs deep)",fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/deployability_frontier.png"); plt.close(); print("6 deployability")

# ---- 7. confusion (ResNet-SE+CD + ensemble) ----
def conf(predfn):
    Cm=np.zeros((4,4))
    for s in subs:
        n,yt=align(s); yp=predfn(s)
        for t,p in zip(yt,yp): Cm[t,p]+=1
    return Cm/Cm.sum(1,keepdims=True)*100
cse=conf(cdpred); cen=conf(enpred)
fig,axes=plt.subplots(1,2,figsize=(9,4.2))
for ax,Cm,ttl in zip(axes,[cse,cen],["ResNet-SE+CD","Soft-vote ensemble (SVM+ResNet-SE+CD)"]):
    im=ax.imshow(Cm,cmap="Blues",vmin=0,vmax=100)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(ttl,fontsize=9)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{Cm[i,j]:.0f}",ha="center",va="center",fontsize=8,color="white" if Cm[i,j]>50 else "black")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_frontier.png"); plt.close(); print("7 confusion")

# ---- 8. NEW: augmentation ablation on ResNet-SE ----
aug=["none","Gaussian\nnoise","channel\ndropout","time\nmasking","combined"]
av=[0.782,0.778,0.840,0.781,0.833]
cols=["#bbbbbb","#bbbbbb","#2e6f9e","#bbbbbb","#8fb8de"]
fig,ax=plt.subplots(figsize=(7.2,4.3)); x=np.arange(len(aug))
ax.bar(x,av,0.6,color=cols,edgecolor="black",linewidth=0.3)
for i,v in enumerate(av): ax.text(i,v+0.004,f"{v:.3f}",ha="center",fontsize=8.5)
ax.axhline(0.782,ls="--",lw=0.8,color="#a33")
ax.set_xticks(x); ax.set_xticklabels(aug,fontsize=8.5); ax.set_ylabel("LOSO macro-F1"); ax.set_ylim(0.74,0.87)
ax.set_title("Data augmentation on ResNet-SE (LOSO, n=40): channel dropout wins",fontsize=10)
ax.text(2,0.855,"+5.7 pp",ha="center",fontsize=9,color="#2e6f9e",fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/augmentation_ablation.png"); plt.close(); print("8 augmentation")
print("ALL CD FRONTIER FIGURES WRITTEN to",OUT)
