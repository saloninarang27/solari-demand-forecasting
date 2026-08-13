import numpy as np
import pandas as pd

import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
U = os.path.join(PROJECT_ROOT, "")
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def load_raw():
    tr = pd.read_csv(os.path.join(PROJECT_ROOT, "orders_train.csv"), parse_dates=["Date"])
    te = pd.read_csv(os.path.join(PROJECT_ROOT, "orders_test.csv"), parse_dates=["Date"])
    hm = pd.read_csv(os.path.join(PROJECT_ROOT, "hub_metadata.csv"))
    return tr, te, hm


def base_features(df, hm):
    """Row-level features that need no history."""
    d = df.merge(hm, on="HubID", how="left")

    d["year"] = d.Date.dt.year
    d["month"] = d.Date.dt.month
    d["day"] = d.Date.dt.day
    d["weekofyear"] = d.Date.dt.isocalendar().week.astype(int)
    d["dayofyear"] = d.Date.dt.dayofyear
    # linear time index — lets the model extrapolate a global trend
    d["day_index"] = (d.Date - pd.Timestamp("2013-01-01")).dt.days

    # months since the nearest competitor opened (negative = not yet open)
    d["CompetitorOpenMonths"] = (
        12 * (d.year - d.CompetitorOpenSinceYear)
        + (d.month - d.CompetitorOpenSinceMonth)
    )
    d.loc[d.CompetitorOpenSinceYear.isna(), "CompetitorOpenMonths"] = np.nan
    d["CompetitorDistanceLog"] = np.log1p(d.CompetitorDistance)

    # weeks since the hub joined the loyalty programme
    d["LoyaltyWeeks"] = (
        52 * (d.year - d.LoyaltyProgramSinceYear)
        + (d.weekofyear - d.LoyaltyProgramSinceWeek)
    )
    d.loc[d.LoyaltyProgramSinceYear.isna(), "LoyaltyWeeks"] = np.nan
    d["LoyaltyWeeks"] = d.LoyaltyWeeks.clip(lower=0)

    # is this month one of the hub's loyalty push months?
    month_name = d.month.map(lambda m: MONTH_ABBR[m - 1])
    interval = d.LoyaltyProgramInterval.fillna("")
    d["InLoyaltyMonth"] = [
        1 if (iv and mn in iv) else 0 for iv, mn in zip(interval, month_name)
    ]
    # only counts once the hub is actually enrolled
    d.loc[d.LoyaltyWeeks.isna(), "InLoyaltyMonth"] = 0

    return d


def add_hub_history(d, hist):
    """Per-hub demand levels learned from `hist` (a training slice only).

    Everything here is a log-space mean, because the metric is RMSLE and the
    model is fitted on log1p(OrderVolume).
    """
    h = hist[(hist.IsOpen == 1) & (hist.OrderVolume > 0)].copy()
    h["logv"] = np.log1p(h.OrderVolume)

    g = h.groupby("HubID").logv
    hub = pd.DataFrame({
        "hub_log_mean": g.mean(),
        "hub_log_std": g.std(),
        "hub_log_med": g.median(),
        "hub_log_max": g.max(),
        "hub_log_min": g.min(),
        "hub_days": g.size(),
    })
    # recent level — last 90 open days per hub, catches trend/renovation shifts
    recent = (h.sort_values("Date").groupby("HubID").tail(90)
                .groupby("HubID").logv.mean().rename("hub_log_recent"))
    hub = hub.join(recent)
    hub["hub_recent_delta"] = hub.hub_log_recent - hub.hub_log_mean

    # weekday profile, as a deviation from the hub's own mean
    wd = h.groupby(["HubID", "Weekday"]).logv.mean().rename("hub_wd_mean").reset_index()
    wd = wd.merge(hub.hub_log_mean.rename("_m"), on="HubID")
    wd["hub_wd_delta"] = wd.hub_wd_mean - wd._m
    wd = wd.drop(columns=["_m"])

    # promo uplift, per hub
    pm = h.groupby(["HubID", "PromoActive"]).logv.mean().rename("hub_promo_mean").reset_index()
    pm = pm.merge(hub.hub_log_mean.rename("_m"), on="HubID")
    pm["hub_promo_delta"] = pm.hub_promo_mean - pm._m
    pm = pm.drop(columns=["_m"])

    # month-of-year seasonality, network-wide deviation
    h["month"] = h.Date.dt.month
    net_mean = h.logv.mean()
    mo = (h.groupby("month").logv.mean() - net_mean).rename("month_delta")

    d = d.merge(hub, on="HubID", how="left")
    d = d.merge(wd, on=["HubID", "Weekday"], how="left")
    d = d.merge(pm, on=["HubID", "PromoActive"], how="left")
    d = d.merge(mo, on="month", how="left")
    return d


def calendar_context(tr, te):
    """Lead/lag flags over the combined timeline.

    IsOpen and PromoActive are known in advance for the test window, so
    looking one day forward is legitimate here — it is scheduling data, not
    an outcome. Demand spikes the day before a closure and sags the day after.
    """
    cols = ["HubID", "Date", "IsOpen", "PromoActive"]
    cal = pd.concat([tr[cols], te[cols]], ignore_index=True)
    cal = cal.sort_values(["HubID", "Date"]).reset_index(drop=True)
    g = cal.groupby("HubID")

    cal["open_prev"] = g.IsOpen.shift(1).fillna(1)
    cal["open_next"] = g.IsOpen.shift(-1).fillna(1)
    cal["promo_prev"] = g.PromoActive.shift(1).fillna(0)
    cal["promo_next"] = g.PromoActive.shift(-1).fillna(0)
    cal["promo_first_day"] = ((cal.PromoActive == 1) & (cal.promo_prev == 0)).astype(int)

    # length of the current run of closed days the hub is about to come out of
    closed = (cal.IsOpen == 0).astype(int)
    grp = (closed != closed.groupby(cal.HubID).shift(1)).cumsum()
    run = closed.groupby([cal.HubID, grp]).cumsum()
    cal["closed_run_len"] = run
    # how long the hub had been shut before today (0 if it was open yesterday)
    cal["closed_days_before"] = cal.groupby("HubID").closed_run_len.shift(1).fillna(0)
    cal.loc[cal.IsOpen == 0, "closed_days_before"] = 0
    cal["reopening"] = ((cal.IsOpen == 1) & (cal.closed_days_before >= 5)).astype(int)

    return cal.drop(columns=["IsOpen", "PromoActive", "closed_run_len"])


def add_last_year(d, hist):
    """Hub's demand in the same calendar window one year earlier."""
    h = hist[(hist.IsOpen == 1) & (hist.OrderVolume > 0)].copy()
    h["logv"] = np.log1p(h.OrderVolume)
    h["ly_key"] = h.Date + pd.DateOffset(years=1)
    # +/- 7 day window around the matching date, per hub
    h["ly_week"] = h.ly_key.dt.isocalendar().week.astype(int)
    h["ly_year"] = h.ly_key.dt.year
    ly = (h.groupby(["HubID", "ly_year", "ly_week"]).logv.mean()
            .rename("hub_lastyear_logmean").reset_index()
            .rename(columns={"ly_year": "year", "ly_week": "weekofyear"}))
    d = d.merge(ly, on=["HubID", "year", "weekofyear"], how="left")
    d["lastyear_delta"] = d.hub_lastyear_logmean - d.hub_log_mean
    return d


FEATURES = [
    "HubID", "Weekday", "PromoActive", "SchoolClosureFlag", "RegionalHoliday",
    "year", "month", "day", "weekofyear", "dayofyear", "day_index",
    "HubFormat", "AssortmentTier",
    "CompetitorDistanceLog", "CompetitorOpenMonths",
    "LoyaltyProgram", "LoyaltyWeeks", "InLoyaltyMonth",
    "hub_log_mean", "hub_log_std", "hub_log_med", "hub_log_max", "hub_log_min",
    "hub_days", "hub_log_recent", "hub_recent_delta",
    "hub_wd_mean", "hub_wd_delta", "hub_promo_mean", "hub_promo_delta",
    "month_delta",
    "open_prev", "open_next", "promo_prev", "promo_next", "promo_first_day",
    "closed_days_before", "reopening",
    "hub_lastyear_logmean", "lastyear_delta",
]


def rmsle(pred, actual):
    return np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2))
