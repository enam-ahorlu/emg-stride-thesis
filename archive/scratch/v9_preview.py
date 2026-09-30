import pandas as pd, numpy as np
a=pd.read_csv("unified_fdr_family_v8_A_all_reported.csv"); b=pd.read_csv("unified_fdr_family_v8_B_bearing.csv")
def bh(p):
    p=np.asarray(p); o=np.argsort(p); r=np.empty(len(p)); q=p[o]*len(p)/(np.arange(len(p))+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; r[o]=np.minimum(q,1); return r
new2=[("D2c domain probe, lambda 100 vs 0.1",0.010628069954691455),("D2c target class probe, lambda 100 vs 0.1",0.09449897358172166)]
new4=new2+[("D2e E1 (no target pass) vs lambda 0",3.49e-06),("D2e E2 (CORAL, BN shielded) vs E1",3.36e-03)]
for name,new in [("v9 (+2, D2c primaries)",new2),("v9 (+4, with D2e)",new4)]:
    for label,df in [("A all reported",a),("B bearing",b)]:
        p=list(df.p.values)+[x[1] for x in new]; q=bh(p)
        print(label, name, "tests", len(p), "survive", int((q<=0.05).sum()))
        old=bh(df.p.values); ch=((old<=0.05)!=(q[:len(old)]<=0.05)).sum(); print("   status changes among existing:", int(ch))
