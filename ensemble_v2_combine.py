#!/usr/bin/env python3
"""
ensemble_v2_combine.py
======================
Compare ensemble COMBINATION methods (hard vote, soft vote, weighted-soft,
stacking) across model subsets, INCLUDING ResNet-SE, from saved per-window
class probabilities. LOSO-safe throughout.

Prerequisite (see EXPERIMENT_PLAN_ENSEMBLE.md, Phase 1): per-window probabilities
saved as  results_ensemble_v2/proba/{MODEL}_sub{K:02d}.npz  (keys: proba [n,4]
in LABELS order = [DNS,STDUP,UPS,WAK], y_true [n]) for MODEL in
{SVM, RF, CNN, RESNET_SE}, per-subject normalization, all 40 held-out subjects.

Outputs: results_ensemble_v2/ensemble_v2_summary.csv
         results_ensemble_v2/ensemble_v2_subjectwise.csv
"""
import argparse
import re
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser("Ensemble combiner comparison (hard/soft/weighted-soft/stacking).")
ap.add_argument("--proba-dir", default="results_ensemble_v2/proba",
                help="Dir with {MODEL}_sub{K:02d}.npz files (default: SIAT, 40 subjects).")
ap.add_argument("--out", default="results_ensemble_v2", help="Output dir for summary/subjectwise CSVs.")
ap.add_argument("--ref", default="SVM+RF+CNN [hard]",
                help="Reference ensemble key for the p_vs_current column.")
args, _ = ap.parse_known_args()

PROBA = ROOT / args.proba_dir
OUT = ROOT / args.out; OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["DNS","STDUP","UPS","WAK"]
MODELS = ["SVM","RF","CNN","RESNET_SE"]

# Auto-detect the subject-id set from whatever npz files are actually present
# (SIAT uses sequential 1..40; ENABL3S uses real subject codes like 156, 185-194).
_SUB_RE = re.compile(r"^(?:" + "|".join(MODELS) + r")_sub(\d+)\.npz$")
_found = set()
if PROBA.exists():
    for f in PROBA.iterdir():
        m = _SUB_RE.match(f.name)
        if m:
            _found.add(int(m.group(1)))
SUBS = sorted(_found) if _found else list(range(1, 41))
N_SUBS = len(SUBS)

def load(model):
    d={}
    for s in SUBS:
        f = PROBA/f"{model}_sub{s:02d}.npz"
        if f.exists():
            z=np.load(f); d[s]=(z["y_true"].astype(int), z["proba"].astype(float))
    return d

P = {m:load(m) for m in MODELS}
avail = {m for m in MODELS if len(P[m])==N_SUBS}
print(f"subjects detected: {SUBS} (n={N_SUBS})")
print("models with full-subject probabilities:", sorted(avail))
# per-model overall F1 (for weighting + tie-break priority)
model_f1={}
for m in avail:
    fs=[f1_score(P[m][s][0], P[m][s][1].argmax(1), average="macro") for s in SUBS]
    model_f1[m]=float(np.mean(fs)); print(f"  {m}: {model_f1[m]:.4f}")

def subject_probs(subset, s):
    yt = P[subset[0]][s][0]
    n = min(len(P[m][s][0]) for m in subset)
    yt = yt[:n]
    stack = np.stack([P[m][s][1][:n] for m in subset])   # [n_models, n, 4]
    return yt, stack

def hard_vote(subset, s):
    yt, stack = subject_probs(subset, s)
    preds = stack.argmax(2)   # [n_models, n]
    order = sorted(subset, key=lambda m:-model_f1[m])     # tie-break priority
    out=np.empty(preds.shape[1], int)
    for i in range(preds.shape[1]):
        vals,cnts=np.unique(preds[:,i],return_counts=True)
        if cnts.max()>=2 or len(subset)==1: out[i]=vals[cnts.argmax()]
        else:
            # all disagree -> highest-F1 model's vote
            out[i]=P[order[0]][s][1][i].argmax()
    return yt, out

def soft_vote(subset, s, weights=None):
    yt, stack = subject_probs(subset, s)
    w = np.ones(len(subset)) if weights is None else np.array([weights[m] for m in subset])
    avg = np.tensordot(w, stack, axes=(0,0))/w.sum()
    return yt, avg.argmax(1)

def eval_method(fn):
    return [f1_score(*fn(s)) for s in SUBS] if False else None

def macro_per_subject(pred_fn):
    return np.array([f1_score(*pred_fn(s), average="macro") for s in SUBS])

def stacking(subset):
    # LOSO-safe: meta LogReg trained on other 39 subjects' probs
    feats={s:np.concatenate([P[m][s][1][:min(len(P[mm][s][0]) for mm in subset)] for m in subset],axis=1) for s in SUBS}
    ys={s:P[subset[0]][s][0][:feats[s].shape[0]] for s in SUBS}
    out=[]
    for s in SUBS:
        Xtr=np.vstack([feats[k] for k in SUBS if k!=s]); ytr=np.concatenate([ys[k] for k in SUBS if k!=s])
        clf=LogisticRegression(max_iter=2000, C=1.0)  # lbfgs (default) is multinomial for multi-class natively in sklearn>=1.7
        clf.fit(Xtr,ytr); out.append(f1_score(ys[s], clf.predict(feats[s]), average="macro"))
    return np.array(out)

rows=[]; subjectwise={}
subsets = [("SVM","RF","CNN"), ("SVM","RF","RESNET_SE"), ("SVM","RF","CNN","RESNET_SE"),
           ("SVM","RF"), ("SVM","RESNET_SE"), ("RF","RESNET_SE")]
ref_key=("SVM","RF","CNN","hard")   # current thesis ensemble
for subset in subsets:
    subset=tuple(m for m in subset if m in avail)
    if len(subset)<2: continue
    for name,fn in [("hard", lambda s,ss=subset: hard_vote(ss,s)),
                    ("soft", lambda s,ss=subset: (subject_probs(ss,s)[0], soft_vote(ss,s)[1])),
                    ("weighted_soft", lambda s,ss=subset: (subject_probs(ss,s)[0], soft_vote(ss,s,model_f1)[1]))]:
        vals=macro_per_subject(fn); key="+".join(subset)+f" [{name}]"
        subjectwise[key]=vals; rows.append(dict(ensemble=key, method=name, models="+".join(subset), f1_mean=vals.mean(), f1_sd=vals.std()))
    # stacking (probability meta-learner)
    try:
        vals=stacking(subset); key="+".join(subset)+" [stacking]"
        subjectwise[key]=vals; rows.append(dict(ensemble=key, method="stacking", models="+".join(subset), f1_mean=vals.mean(), f1_sd=vals.std()))
    except Exception as e:
        print("stacking failed for", subset, e)

df=pd.DataFrame(rows).sort_values("f1_mean", ascending=False)
# significance vs the reference ensemble (default: current SVM+RF+CNN hard vote)
ref=args.ref
if ref in subjectwise:
    for k in list(subjectwise):
        if k==ref: continue
        try:
            w,p=wilcoxon(subjectwise[k], subjectwise[ref]); 
        except Exception: p=np.nan
        df.loc[df.ensemble==k,"p_vs_current"]=p
df.to_csv(OUT/"ensemble_v2_summary.csv", index=False)
pd.DataFrame(subjectwise).to_csv(OUT/"ensemble_v2_subjectwise.csv", index=False)
print("\n=== ranked ensembles ===")
print(df.to_string(index=False))
print("\nwrote", OUT/"ensemble_v2_summary.csv")
