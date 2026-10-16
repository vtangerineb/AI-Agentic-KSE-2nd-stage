
# # Model B (rewritten): "Will Kyiv city have an air-raid alert onset tonight?"
# 
# **Forecast task.** Every day at `FORECAST_HOUR` (18:00 Kyiv time) we predict the probability that at least one
# alert *starts* in Kyiv city during the night window 22:00 -> 07:00 (next morning).
# 
# **Leakage rules enforced in this notebook**
# 1. Every feature for day D is computed only from events with `started_at < D 18:00`.
#    Alerts still running at 18:00 have their end time clipped to 18:00 (we don't know when they will end).
# 2. Features about previous nights use only nights that were fully over before D 18:00.
# 3. Walk-forward evaluation: the model used for day D is trained only on days < D (their targets are resolved).
# 4. `assert_point_in_time` re-computes features from data truncated at the cutoff and checks they are identical.

# In[1]:
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.calibration import calibration_curve

try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings("ignore", category=FutureWarning)

TZ = "Europe/Kyiv"
SOURCE_TZ = "UTC"            # timezone of started_at / finished_at in the CSV (verify with the hour histogram below!)
FORECAST_HOUR = 18           # forecast is issued at 18:00 local time
NIGHT_START_HOUR = 22        # target window start (same day)
NIGHT_END_HOUR = 7           # target window end (next day)
TRAIN_START = "2024-01-01"   # first day used for modelling (earlier data is still used as feature history)
EVAL_DAYS = 90               # most recent days used as the walk-forward test period
RETRAIN_EVERY = 7            # retrain cadence (days) inside the test period
MIN_TRAIN_DAYS = 120
SEED = 42

HOUR = 3600 * 10**9
DAY = 24 * HOUR

# ## 1. Load & clean data

# In[2]:
KYIV_CITY_NAMES = {"kyiv", "kyiv city", "kiev", "м. київ", "м.київ", "київ", "m. kyiv"}


def download_official_csv(out_path="data/raw_official_alerts.csv"):
    import requests
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    api = "https://api.github.com/repos/Vadimkin/ukrainian-air-raid-sirens-dataset/contents/datasets"
    files = requests.get(api, timeout=30).json()
    cands = [f for f in files if "official" in f["name"].lower() and f["name"].endswith(".csv")]
    if not cands:                                   # fail loudly instead of silently using a random file
        raise RuntimeError(f"No 'official' CSV found. Files: {[f['name'] for f in files]}")
    cands.sort(key=lambda f: ("_en" not in f["name"], f["name"]))
    with open(out_path, "wb") as fh:
        fh.write(requests.get(cands[0]["download_url"], timeout=120).content)
    print("downloaded:", cands[0]["name"])
    return out_path


def load_alerts(path):
    df = pd.read_csv(path)
    missing = {"oblast", "started_at", "finished_at"} - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=(SOURCE_TZ == "UTC"))
        if SOURCE_TZ != "UTC":
            df[c] = df[c].dt.tz_localize(SOURCE_TZ, ambiguous="NaT", nonexistent="NaT")
        df[c] = df[c].dt.tz_convert(TZ)
    n0 = len(df)
    bad_time = df[["started_at", "finished_at"]].isna().any(axis=1)
    df = df[~bad_time]
    neg = df["finished_at"] < df["started_at"]
    df = df[~neg]
    print(f"rows: {n0} | dropped unparsable/missing time: {bad_time.sum()} | dropped end<start: {neg.sum()}")
    return df.sort_values("started_at").reset_index(drop=True)


def split_kyiv_and_national(df):
    name = df["oblast"].astype(str).str.strip().str.lower()
    is_city = name.isin(KYIV_CITY_NAMES)
    print("oblast values that mention Kyiv:", sorted(df.loc[name.str.contains("kyiv|київ|kiev"), "oblast"].unique()))
    assert is_city.sum() > 0, "Kyiv city rows not found - adjust KYIV_CITY_NAMES"
    kyiv = df[is_city].sort_values("started_at").reset_index(drop=True)
    nat = df[~is_city]
    if "level" in nat.columns:
        nat = nat[nat["level"].astype(str).str.lower() == "oblast"]
    nat = nat.sort_values("started_at").reset_index(drop=True)
    print(f"Kyiv-city alerts: {len(kyiv)} | other oblast-level alerts: {len(nat)}")
    return kyiv, nat

# ## 2. Point-in-time feature builder

# In[3]:
def to_ns(s):
    """tz-aware datetime Series -> int64 UTC nanoseconds."""
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")


def local_ns(day, hour):
    """Local wall-clock `hour` on `day` -> UTC ns (DST-safe; hours used are never in a DST gap)."""
    return pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {hour:02d}:00", tz=TZ).value


def day_times(day):
    day = pd.Timestamp(day).normalize()
    nxt = day + pd.Timedelta(days=1)
    return dict(mid=local_ns(day, 0), cut=local_ns(day, FORECAST_HOUR),
                ws=local_ns(day, NIGHT_START_HOUR), we=local_ns(nxt, NIGHT_END_HOUR))


def night_counts(kyiv_start_ns, days):
    """Number of Kyiv alert onsets inside each day's night window (this is the TARGET definition)."""
    out = []
    for d in days:
        t = day_times(d)
        out.append(np.searchsorted(kyiv_start_ns, t["we"], "left") - np.searchsorted(kyiv_start_ns, t["ws"], "left"))
    return pd.Series(out, index=pd.DatetimeIndex(days), dtype=float)


def _overlap_min(st, en, lo, hi):
    return np.clip(np.minimum(en, hi) - np.maximum(st, lo), 0, None).sum() / 60e9


def build_features(kyiv, nat, days):
    """Features for forecast day D, using ONLY events with started_at < D FORECAST_HOUR:00."""
    days = pd.DatetimeIndex(pd.to_datetime(days)).normalize()
    k_st, k_en = to_ns(kyiv["started_at"]), to_ns(kyiv["finished_at"])
    n_st, n_en = to_ns(nat["started_at"]), to_ns(nat["finished_at"])
    n_code = pd.factorize(nat["oblast"])[0]

    # --- previous-night history (night D-1 ends at D 07:00 < D 18:00 -> fully known at forecast time)
    hist_days = pd.date_range(days.min() - pd.Timedelta(days=35), days.max(), freq="D")
    cnt = night_counts(k_st, hist_days)
    hit = (cnt > 0).astype(float)
    hist = pd.DataFrame(index=hist_days)
    hist["night_lag1"] = hit.shift(1)
    hist["night_lag2"] = hit.shift(2)
    hist["night_lag3"] = hit.shift(3)
    hist["onsets_lag1"] = cnt.shift(1)
    hist["hit_rate_7"] = hit.shift(1).rolling(7).mean()
    hist["hit_rate_14"] = hit.shift(1).rolling(14).mean()
    hist["hit_rate_30"] = hit.shift(1).rolling(30).mean()
    hist["mean_onsets_7"] = cnt.shift(1).rolling(7).mean()
    since, last = [], None                                   # nights since last night with an alert (strictly past nights)
    for i, h in enumerate(hit.shift(1).to_numpy()):
        if h == 1:
            last = i
        since.append(min(i - last, 30) if last is not None else 30)
    hist["nights_since_hit"] = since

    rows = []
    for d in days:
        T = day_times(d)
        c = T["cut"]
        # ---------- Kyiv alerts known at c
        hi = np.searchsorted(k_st, c, "left")                # started_at < c
        lo = np.searchsorted(k_st[:hi], c - 8 * DAY, "left")
        st, en = k_st[lo:hi], np.minimum(k_en[lo:hi], c)     # clip: unfinished alerts end "now"
        active = float((en >= c).any()) if len(en) else 0.0
        f = dict(
            k_active_now=active,
            k_hrs_since_start=min((c - k_st[hi - 1]) / HOUR, 720) if hi > 0 else 720,
            k_hrs_since_end=0.0 if active else (min((c - en.max()) / HOUR, 192) if len(en) else 192),
            k_starts_today=int((st >= T["mid"]).sum()),
            k_starts_24h=int((st >= c - DAY).sum()),
            k_starts_7d=int((st >= c - 7 * DAY).sum()),
            k_min_today=_overlap_min(st, en, T["mid"], c),
            k_min_24h=_overlap_min(st, en, c - DAY, c),
        )
        # ---------- national picture known at c
        nhi = np.searchsorted(n_st, c, "left")
        nlo = np.searchsorted(n_st[:nhi], c - 3 * DAY, "left")
        s, e, code = n_st[nlo:nhi], np.minimum(n_en[nlo:nhi], c), n_code[nlo:nhi]
        prev = day_times(d - pd.Timedelta(days=1))
        f.update(
            n_active_now=len(np.unique(code[e >= c])),
            n_obl_6h=len(np.unique(code[s >= c - 6 * HOUR])),
            n_obl_24h=len(np.unique(code[s >= c - DAY])),
            n_starts_24h=int((s >= c - DAY).sum()),
            n_hours_24h=_overlap_min(s, e, c - DAY, c) / 60,
            n_starts_prev_night=int(((s >= prev["ws"]) & (s < prev["we"])).sum()),
            dow=d.dayofweek,
            is_weekend=float(d.dayofweek >= 5),
        )
        rows.append(f)
    X = pd.DataFrame(rows, index=days)
    return pd.concat([X, hist.reindex(days)], axis=1)


def assert_point_in_time(kyiv, nat, X_full, n_checks=25, seed=SEED):
    """Leakage test: rebuild features from data truncated at the cutoff; they must match exactly."""
    rng = np.random.default_rng(seed)
    recent = X_full.iloc[-400:]
    # leaks hide in days where an alert is still RUNNING at the cutoff -> always include such days
    ongoing_k = recent.index[recent["k_active_now"] == 1]
    ongoing_n = recent.index[recent["n_active_now"] > 0]
    pick = lambda idx, k: list(rng.choice(idx, size=min(k, len(idx)), replace=False)) if len(idx) else []
    sample = list(dict.fromkeys(pick(recent.index, n_checks) + pick(ongoing_k, 20) + pick(ongoing_n, 20)))

    def trunc(df, c_ts):
        d = df[df["started_at"] < c_ts].copy()
        d["finished_at"] = d["finished_at"].clip(upper=c_ts)   # we don't know future end times
        return d

    for d in sample:
        c_ts = pd.Timestamp(day_times(d)["cut"], tz="UTC").tz_convert(TZ)
        Xt = build_features(trunc(kyiv, c_ts), trunc(nat, c_ts), [d])
        d = pd.Timestamp(d); a, b = X_full.loc[[d]], Xt.loc[[d], X_full.columns]
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)
    print(f"point-in-time test PASSED on {len(sample)} days incl. {len(pick(ongoing_k, 20))} with a Kyiv alert running at the cutoff (features identical with future data removed)")

# ## 3. Models, baselines, walk-forward evaluation

# In[4]:
LOG_COLS = ["k_hrs_since_start", "k_hrs_since_end", "k_starts_today", "k_starts_24h", "k_starts_7d",
            "k_min_today", "k_min_24h", "n_obl_6h", "n_obl_24h", "n_starts_24h", "n_hours_24h",
            "n_starts_prev_night", "onsets_lag1", "mean_onsets_7"]


class TrainRate:
    name = "baseline: train hit-rate"
    def fit(self, X, y): self.p = (y.sum() + 1) / (len(y) + 2); return self
    def predict_proba(self, X): return np.full(len(X), self.p)


class PrevNightConditional:
    name = "baseline: P(hit | last night)"
    def fit(self, X, y):
        self.p = {v: ((y[X["night_lag1"] == v].sum() + 1) / ((X["night_lag1"] == v).sum() + 2)) for v in (0.0, 1.0)}
        return self
    def predict_proba(self, X): return X["night_lag1"].map(self.p).to_numpy()


class Roll14:
    name = "baseline: hit-rate last 14 nights"
    def fit(self, X, y): return self
    def predict_proba(self, X): return np.clip(X["hit_rate_14"].to_numpy(), 0.03, 0.97)


class Sk:
    def __init__(self, name, est, cols): self.name, self.est, self.cols = name, est, cols
    def fit(self, X, y):
        self.const = None if y.nunique() > 1 else float(y.iloc[0])
        if self.const is None: self.est.fit(X[self.cols], y)
        return self
    def predict_proba(self, X):
        return np.full(len(X), np.clip(self.const, .03, .97)) if self.const is not None else self.est.predict_proba(X[self.cols])[:, 1]


def make_logreg(cols):
    log_cols = [c for c in cols if c in LOG_COLS]
    ct = ColumnTransformer([("log", FunctionTransformer(np.log1p), log_cols)], remainder="passthrough")
    return Sk("LogReg (L2)", make_pipeline(ct, StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)), cols)


def make_xgb(cols):
    return Sk("XGBoost (shallow)", xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8,
                                                      colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                                                      random_state=SEED, eval_metric="logloss"), cols)


def walk_forward(data, feature_cols, factories, eval_days=EVAL_DAYS, retrain_every=RETRAIN_EVERY):
    """For each block of test days, train on days strictly before the block's first day
    (their night windows ended before that day's 18:00 forecast time) and predict the block."""
    test_days = data.index[-eval_days:]
    preds = {}
    for i in range(0, len(test_days), retrain_every):
        block = test_days[i:i + retrain_every]
        train = data[data.index < block[0]]
        assert len(train) >= MIN_TRAIN_DAYS, "not enough training days"
        for fac in factories:
            m = fac(feature_cols).fit(train[feature_cols], train["y"])
            preds.setdefault(m.name, []).append(pd.Series(m.predict_proba(data.loc[block, feature_cols]), index=block))
    return pd.DataFrame({k: pd.concat(v) for k, v in preds.items()})


def metrics(y, p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return dict(
        PR_AUC_alert=average_precision_score(y, p),
        PR_AUC_quiet=average_precision_score(1 - y, 1 - p),
        ROC_AUC=roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
        Brier=brier_score_loss(y, p),
        LogLoss=log_loss(y, p, labels=[0, 1]),
    )


def block_bootstrap_brier_gain(y, p_model, p_ref, block=7, n=2000, seed=SEED):
    """Mean Brier improvement of model over reference (positive = better) with 95% moving-block bootstrap CI."""
    rng = np.random.default_rng(seed)
    d = ((p_ref - y) ** 2 - (p_model - y) ** 2).to_numpy()
    T = len(d); nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, [2.5, 97.5])

# ## 4. Run

# In[5]:
def run(csv_path):
    df = load_alerts(csv_path)
    kyiv, nat = split_kyiv_and_national(df)
    data_end = max(df["started_at"].max(), df["finished_at"].max())
    print("data covers up to:", data_end)

    hrs = kyiv["started_at"].dt.hour.value_counts(normalize=True).sort_index()
    print("Kyiv onsets by local hour (sanity-check timezone: should peak at night/early morning):")
    print((hrs * 100).round(1).to_dict())

    days = pd.date_range(TRAIN_START, data_end.tz_localize(None).normalize(), freq="D")
    cnt = night_counts(to_ns(kyiv["started_at"]), days)
    valid = np.array([day_times(d)["we"] <= data_end.value for d in days])     # target window must be complete
    days = days[valid]
    print(f"modelling days: {days[0].date()} -> {days[-1].date()} ({len(days)}); incomplete trailing days dropped: {(~valid).sum()}")

    X = build_features(kyiv, nat, days)
    assert_point_in_time(kyiv, nat, X)

    data = X.join((cnt.reindex(days) > 0).astype(int).rename("y"))
    n_before = len(data)
    data = data.dropna()
    print(f"dropped {n_before - len(data)} days with incomplete feature history | base rate of nights with an alert: {data['y'].mean():.3f}")

    feature_cols = [c for c in X.columns]
    factories = [lambda c: TrainRate(), lambda c: PrevNightConditional(), lambda c: Roll14(), make_logreg]
    if xgb is not None:
        factories.append(make_xgb)
    P = walk_forward(data, feature_cols, factories)
    y = data.loc[P.index, "y"]

    res = pd.DataFrame({k: metrics(y, P[k].to_numpy()) for k in P.columns}).T
    res.loc["baseline: train hit-rate", ["PR_AUC_alert", "PR_AUC_quiet", "ROC_AUC"]] = np.nan   # constant forecast: no ranking skill by definition
    res["Brier_skill_vs_trainrate"] = 1 - res["Brier"] / res.loc["baseline: train hit-rate", "Brier"]
    print(f"\nWalk-forward test: {P.index[0].date()} -> {P.index[-1].date()} ({len(y)} nights, base rate {y.mean():.3f})")
    print(res.round(4).to_string())

    best_base = res.loc[[c for c in res.index if c.startswith("baseline")], "Brier"].idxmin()
    print(f"\nBest baseline by Brier: {best_base}")
    for k in [c for c in P.columns if not c.startswith("baseline")]:
        for ref in ["baseline: train hit-rate", best_base]:
            g, ci = block_bootstrap_brier_gain(y, P[k], P[ref])
            print(f"  {k} vs {ref}: Brier gain {g:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    return data, P, y, res, feature_cols


def plot_results(data, P, y, res, feature_cols):
    models = [c for c in P.columns if not c.startswith("baseline")]
    best = res.loc[models, "Brier"].idxmin() if models else "baseline: train hit-rate"
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))

    ax[0].bar(range(len(res)), res["Brier"], color=["#999" if i.startswith("baseline") else "#2a7" for i in res.index])
    ax[0].set_xticks(range(len(res))); ax[0].set_xticklabels([i.replace("baseline: ", "") for i in res.index], rotation=30, ha="right")
    ax[0].set_ylabel("Brier score (lower is better)"); ax[0].set_title("Walk-forward Brier score")

    pb, pt = calibration_curve(y, P[best], n_bins=5, strategy="quantile")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].plot(pt, pb, "o-")
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Observed frequency"); ax[1].set_title(f"Calibration: {best}")

    ax[2].plot(P.index, P[best], label="P(alert tonight)")
    ax[2].scatter(y.index, y.values, c=np.where(y == 1, "#c33", "#38c"), s=12, label="outcome (1 = alert)")
    ax[2].set_title("Test period: forecast vs outcome"); ax[2].legend(); ax[2].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.show()

# In[6]:
if __name__ == "__main__":
    path = os.environ.get("ALERTS_CSV") or download_official_csv()
    out = run(path)
    plot_results(*out)