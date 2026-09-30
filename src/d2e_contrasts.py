import pandas as pd, numpy as np
from scipy.stats import wilcoxon, norm
def load(d):
    return pd.read_csv(d+"/deep_coral_subjectwise.csv").sort_values("subject").set_index("subject").f1_macro
arms={"lam0_train":"results/deep_coral_align_lam0","E1_notgt":"results/deep_coral_align_lam0_notgt",
      "E2_lam30_eval":"results/deep_coral_align_lam30_tgteval","lam0p1":"results/deep_coral_align_lam0p1","lam100":"results/deep_coral_align_lam100"}
S={k:load(v) for k,v in arms.items()}
for k,v in S.items(): print(k, len(v), round(v.mean(),4), round(v.std(ddof=1),4))
def boot(x,B=10000,seed=0):
    rng=np.random.default_rng(seed); n=len(x)
    bs=np.sort(x[rng.integers(0,n,(B,n))].mean(1)); th=x.mean()
    z0=norm.ppf(max((bs<th).mean(),1e-6)); jk=np.array([np.delete(x,i).mean() for i in range(n)]); jm=jk.mean()
    a=((jm-jk)**3).sum()/(6*(((jm-jk)**2).sum())**1.5); out=[]
    for q in (0.025,0.975):
        zq=norm.ppf(q); p=norm.cdf(z0+(z0+zq)/(1-a*(z0+zq))); out.append(np.quantile(bs,p))
    return out
def con(a,b):
    d=(S[a]-S[b]).dropna(); w=wilcoxon(S[a],S[b]); lo,hi=boot(d.values)
    print("%s - %s: %+.2f pp, dz %+.2f, BCa [%+.2f,%+.2f], lower %d/40, p %.2e" % (a,b,100*d.mean(),d.mean()/d.std(ddof=1),100*lo,100*hi,int((d<0).sum()),w.pvalue))
con("E1_notgt","lam0_train"); con("E2_lam30_eval","E1_notgt"); con("E2_lam30_eval","lam0_train")
con("lam100","lam0_train"); con("lam0p1","lam0_train")
print("ref", round(np.mean([0.7723,0.7667,0.7760]),4))
