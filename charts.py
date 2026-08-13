import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import load_raw

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "-", "axes.edgecolor": "#888888",
})
INK, ACC, WARN = "#1f3b57", "#c8622a", "#7a8b99"
tr, te, hm = load_raw()
op = tr[tr.IsOpen == 1]

# 1 — network demand over time, with the forecast window marked
d = tr[tr.IsOpen == 1].groupby("Date").OrderVolume.mean()
fig, ax = plt.subplots(figsize=(9, 2.9))
ax.plot(d.index, d.rolling(7, center=True).mean(), color=INK, lw=1.1)
ax.plot(d.index, d.values, color=INK, lw=0.35, alpha=0.25)
ax.axvspan(pd.Timestamp("2015-06-20"), pd.Timestamp("2015-07-31"),
           color=ACC, alpha=0.18)
ax.text(pd.Timestamp("2015-06-24"), d.max() * 0.97, "forecast window",
        color=ACC, fontsize=8, va="top")
ax.set_ylabel("mean orders per open hub"); ax.set_xlabel("")
ax.set_title("Daily network demand, 7-day rolling mean", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_trend.png"); plt.close(fig)

# 2 — weekday profile split by promotion
fig, ax = plt.subplots(figsize=(4.3, 2.7))
w = op.groupby(["Weekday", "PromoActive"]).OrderVolume.mean().unstack()
x = np.arange(7)
ax.bar(x - 0.19, w[0].values, 0.38, label="no promo", color=WARN)
if 1 in w.columns:
    ax.bar(x + 0.19, w[1].reindex(w.index).values, 0.38, label="promo", color=ACC)
ax.set_xticks(x); ax.set_xticklabels(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
ax.set_ylabel("mean orders"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Weekday and promotion effect", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_weekday.png"); plt.close(fig)

# 3 — hub size spread
fig, ax = plt.subplots(figsize=(4.3, 2.7))
hb = op.groupby("HubID").OrderVolume.mean()
ax.hist(hb, bins=60, color=INK, alpha=0.85)
ax.set_xlabel("mean daily orders per hub"); ax.set_ylabel("hubs")
ax.set_title(f"Hub size spread ({hb.min():,.0f} to {hb.max():,.0f})",
             loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_hubsize.png"); plt.close(fig)

# 4 — data coverage: 180 hubs vanish from the file for six months of 2014
fig, ax = plt.subplots(figsize=(9, 2.4))
cov = tr.groupby("Date").HubID.nunique()
ax.fill_between(cov.index, cov.values, color=ACC, alpha=0.6, lw=0)
ax.set_ylim(800, 1160)
ax.set_ylabel("hubs reporting"); ax.set_xlabel("")
ax.text(pd.Timestamp("2014-07-06"), 880,
        "180 hubs absent\n2014-07-01 to 2014-12-31", fontsize=7.5, color=INK)
ax.set_title("Data coverage — a six-month reporting gap, not a recorded closure",
             loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_closures.png"); plt.close(fig)

# 5 — feature importance
gain = json.load(open("gain.json"))
top = sorted(gain.items(), key=lambda kv: -kv[1])[:14][::-1]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.barh([k for k, _ in top], [v for _, v in top], color=INK, alpha=0.85)
ax.set_xlabel("gain"); ax.tick_params(labelsize=7.5)
ax.set_title("Feature importance by gain", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_importance.png"); plt.close(fig)

# 6 — holdout accuracy: predicted vs actual on open days
vp = pd.read_csv("val_pred.csv", parse_dates=["Date"])
o = vp[(vp.IsOpen == 1) & (vp.OrderVolume > 0)]
fig, ax = plt.subplots(figsize=(4.0, 3.4))
ax.scatter(o.OrderVolume, o.pred, s=1.2, alpha=0.10, color=INK, edgecolors="none")
lim = [0, np.percentile(o.OrderVolume, 99.7)]
ax.plot(lim, lim, color=ACC, lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("actual orders"); ax.set_ylabel("predicted orders")
ax.set_title("Holdout: predicted vs actual", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_scatter.png"); plt.close(fig)

# 7 — error by day across the holdout horizon
o = o.copy()
o["sqerr"] = (np.log1p(o.pred) - np.log1p(o.OrderVolume)) ** 2
byday = o.groupby("Date").sqerr.mean() ** 0.5
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(byday.index, byday.values, color=INK, lw=1.2, marker="o", ms=2.5)
ax.set_ylabel("RMSLE"); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=35, labelsize=7)
ax.set_title("Holdout error across the 42-day horizon", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_horizon.png"); plt.close(fig)

# 8 — score comparison
sc = json.load(open("valscore.json"))
fig, ax = plt.subplots(figsize=(4.0, 2.5))
names = ["Hub x weekday\nbaseline", "XGBoost\n(open rows)", "XGBoost\n(all rows)"]
vals = [sc["rmsle_base"], sc["rmsle_open"], sc["rmsle_all"]]
b = ax.bar(names, vals, color=[WARN, INK, INK], alpha=0.9)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.4f}",
            ha="center", fontsize=8.5)
ax.set_ylabel("RMSLE"); ax.set_ylim(0, max(vals) * 1.25)
ax.tick_params(labelsize=8)
ax.set_title("Holdout scores (lower is better)", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("fig_scores.png"); plt.close(fig)

print("figures written")
