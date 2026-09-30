import gc, sys, psutil, os, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedKFold as SKF

use_gc = "--gc" in sys.argv
proc = psutil.Process(os.getpid())

rng = np.random.default_rng(0)
n = 36420  # matches the SD script's per-fold training size
X = rng.standard_normal((n, 56)).astype(np.float64)
y = rng.integers(0, 4, n)

grid = {"n_estimators": [200, 400, 500], "max_depth": [None, 10]}

for fold in range(3):
    rss_before = proc.memory_info().rss / 1e9
    inner_cv = SKF(n_splits=3, shuffle=True, random_state=fold)
    rf = RandomForestClassifier(n_jobs=3, random_state=42, class_weight="balanced_subsample")
    gs = GridSearchCV(rf, grid, cv=inner_cv, scoring="f1_macro", n_jobs=1, refit=True)
    gs.fit(X, y)
    rss_after = proc.memory_info().rss / 1e9
    print(f"fold {fold+1}: rss_before={rss_before:.2f}GB rss_after={rss_after:.2f}GB best={gs.best_params_}", flush=True)
    del rf, gs
    if use_gc:
        gc.collect()
    rss_after_cleanup = proc.memory_info().rss / 1e9
    print(f"  after cleanup (gc={use_gc}): rss={rss_after_cleanup:.2f}GB", flush=True)
