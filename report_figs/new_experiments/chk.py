import pandas as pd
a=pd.read_csv("unified_fdr_family_v9_A_all_reported.csv")
for k in ["RF external per-subj","AdaBN post vs pre (resnet_se, ENABL3S)","ResNet-SE ENABL3S per-subject"]:
    r=a[a.comparison.str.contains(k,regex=False)]
    print(k, [(round(x,4),round(y,4)) for x,y in zip(r.p,r.p_BH)])
print("survivors", int(a.sig.sum()), "of", len(a))
