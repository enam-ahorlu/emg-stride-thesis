import pandas as pd, numpy as np
from scipy.stats import wilcoxon, norm
e3=pd.read_csv("results_repro_250global_b256/cnn_arch_subjectwise.csv").sort_values("subject").set_index("subject").f1_macro
e1=pd.read_csv("results_deep_coral_align_lam0_notgt/deep_coral_subjectwise.csv").sort_values("subject").set_index("subject").f1_macro
print("E3 n=%d mean=%.4f sd=%.4f" % (len(e3), e3.mean(), e3.std(ddof=1)))
print("E1 mean=%.4f" % e1.mean(), "| ref 0.7717")
d=(e3-e1).dropna(); print("E3-E1: %+.2f pp, dz %+.2f, lower %d/40, p %.3f" % (100*d.mean(), d.mean()/d.std(ddof=1), int((d<0).sum()), wilcoxon(e3,e1).pvalue))
x=d.values; rng=np.random.default_rng(0); bs=np.sort(x[rng.integers(0,len(x),(10000,len(x)))].mean(1))
print("BCa-ish percentile CI: [%+.2f, %+.2f] pp" % (100*np.quantile(bs,0.025), 100*np.quantile(bs,0.975)))
print("E3 - 0.7717 = %+.2f pp" % (100*(e3.mean()-0.7717)))
