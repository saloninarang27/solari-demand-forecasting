import time
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from features import (load_raw, base_features, add_hub_history, calendar_context,
                      add_last_year, FEATURES)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
t0 = time.time()
tr, te, hm = load_raw()
cal = calendar_context(tr, te)

# Full history now — nothing held back.
hist = tr
fit = hist[(hist.IsOpen == 1) & (hist.OrderVolume > 0)]


def build(df):
    x = base_features(df, hm)
    x = add_hub_history(x, hist)
    x = add_last_year(x, hist)
    x = x.merge(cal, on=["HubID", "Date"], how="left")
    return x


Xf, Xt = build(fit), build(te)
print("fit", Xf.shape, "test", Xt.shape, f"({time.time()-t0:.0f}s)")

dtrain = xgb.DMatrix(Xf[FEATURES].astype(np.float32),
                     label=np.log1p(Xf.OrderVolume.values))
dtest = xgb.DMatrix(Xt[FEATURES].astype(np.float32))

# Holdout picked 1565 rounds on ~40 fewer days of data; a modest bump keeps
# the effective amount of fitting the same now that the window is longer.
ROUNDS = 1700
base = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "min_child_weight": 6,
    "tree_method": "hist",
    "nthread": 1,
}

import sys, os, glob
seed = int(sys.argv[1])
m = xgb.train(dict(base, seed=seed), dtrain, num_boost_round=ROUNDS, verbose_eval=False)
np.save(os.path.join(PROJECT_ROOT, f"pred_{seed}.npy"), m.predict(dtest))
print(f"seed {seed} done ({time.time()-t0:.0f}s)")

# Average every seed run so far, in log space — the space the metric lives in.
files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "pred_*.npy")))
print("averaging:", files)
pred = np.expm1(np.mean([np.load(f) for f in files], axis=0)).clip(0)
pred[Xt.IsOpen.values == 0] = 0.0

sub = pd.DataFrame({"Id": Xt.Id.values, "OrderVolume": pred})
sub = sub.sort_values("Id")

ss = pd.read_csv(os.path.join(PROJECT_ROOT, "sample_submission.csv"))
assert len(sub) == len(ss) and set(sub.Id) == set(ss.Id), "submission Id mismatch"
assert sub.OrderVolume.notna().all(), "NaNs in submission"

sub.to_csv(os.path.join(PROJECT_ROOT, "submission.csv"), index=False)
print("\n", sub.OrderVolume.describe())
print("zero rows:", (sub.OrderVolume == 0).sum())
print(f"written ({time.time()-t0:.0f}s)")
