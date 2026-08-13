import numpy as np
import pandas as pd
import xgboost as xgb
from features import (load_raw, base_features, add_hub_history, calendar_context,
                      add_last_year, FEATURES, rmsle)

tr, te, hm = load_raw()
cal = calendar_context(tr, te)

CUT = tr.Date.max() - pd.Timedelta(days=41)
hist = tr[tr.Date < CUT]
val = tr[tr.Date >= CUT]

fit = hist[(hist.IsOpen == 1) & (hist.OrderVolume > 0)]


def build(df):
    x = base_features(df, hm)
    x = add_hub_history(x, hist)
    x = add_last_year(x, hist)
    x = x.merge(cal, on=["HubID", "Date"], how="left")
    return x


Xf, Xv = build(fit), build(val)
print("fit", Xf.shape, "val", Xv.shape)

dtrain = xgb.DMatrix(Xf[FEATURES].astype(np.float32),
                     label=np.log1p(Xf.OrderVolume.values))
Xvo = Xv[(Xv.IsOpen == 1) & (Xv.OrderVolume > 0)]
dval_open = xgb.DMatrix(Xvo[FEATURES].astype(np.float32),
                        label=np.log1p(Xvo.OrderVolume.values))
dval = xgb.DMatrix(Xv[FEATURES].astype(np.float32))

params = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "min_child_weight": 6,
    "tree_method": "hist",
    "nthread": 1,
    "seed": 42,
}

model = xgb.train(params, dtrain, num_boost_round=3000,
                  evals=[(dval_open, "val")], early_stopping_rounds=80,
                  verbose_eval=200)
print("best_iteration:", model.best_iteration)

pred = np.expm1(model.predict(dval, iteration_range=(0, model.best_iteration + 1))).clip(0)
pred[Xv.IsOpen.values == 0] = 0.0
actual = Xv.OrderVolume.values
om = Xv.IsOpen.values == 1
print("RMSLE all rows      :", round(rmsle(pred, actual), 5))
print("RMSLE open rows only:", round(rmsle(pred[om], actual[om]), 5))

for k, v in sorted(model.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:18]:
    print(f"  {k:26s} {v:,.0f}")
