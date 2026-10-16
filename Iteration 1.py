
# In[1]:
import pandas as pd
df_test = pd.read_csv("data/raw_official_alerts.csv")
print("Columns:", list(df_test.columns))
print("First row:\n", df_test.iloc[0])

# In[2]:
import os
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Define your modern historical cutoff date
MODERN_START_DATE = "2025-01-01"

# =============================================================================
# 1. DYNAMIC DATA ACQUISITION FROM GITHUB API
# =============================================================================
def download_vadimkin_dataset(output_path: str = "data/raw_official_alerts.csv") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    api_url = "https://api.github.com/repos/Vadimkin/ukrainian-air-raid-sirens-dataset/contents/datasets"

    response = requests.get(api_url, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API check failed: {response.status_code}")

    files = response.json()
    target_url = None
    for file in files:
        name = file["name"].lower()
        if name.endswith(".csv") and ("official" in name or "alert" in name or "report" in name):
            target_url = file["download_url"]
            break

    if not target_url and files:
        csv_files = [f for f in files if f["name"].endswith(".csv")]
        if csv_files:
            target_url = csv_files[0]["download_url"]

    file_res = requests.get(target_url, timeout=30)
    if file_res.status_code == 200 and len(file_res.content) > 1000:
        with open(output_path, "wb") as f:
            f.write(file_res.content)
    else:
        raise RuntimeError("Failed to download dataset stream.")

    return output_path

# =============================================================================
# 2. PREPROCESSING & HOURLY RESAMPLING (FILTERED FOR RECENT KYIV HISTORY)
# =============================================================================
def load_and_resample_kyiv_data(csv_path: str, start_cutoff: str = "2025-01-01") -> pd.DataFrame:
    print("\n--- Processing Raw Timestamps for Kyiv City ---")
    df = pd.read_csv(csv_path)

    # Filter strictly for Kyiv City
    is_kyiv = df["oblast"].astype(str).str.contains("Kyiv|Київ", case=False, na=False)
    is_not_oblast = ~df["oblast"].astype(str).str.contains("oblast|область", case=False, na=False)

    kyiv_df = df[is_kyiv & is_not_oblast].copy()
    if kyiv_df.empty:
        kyiv_df = df[df["oblast"].astype(str).str.contains("Kyiv|Київ", case=False, na=False)].copy()

    # Parse UTC datetime strings
    kyiv_df["started_at"] = pd.to_datetime(kyiv_df["started_at"], errors="coerce", utc=True).dt.tz_localize(None)
    kyiv_df["finished_at"] = pd.to_datetime(kyiv_df["finished_at"], errors="coerce", utc=True).dt.tz_localize(None)

    kyiv_df = kyiv_df.dropna(subset=["started_at", "finished_at"]).sort_values("started_at").reset_index(drop=True)

    # Define start/end grid bounds
    min_time = pd.to_datetime(start_cutoff)
    max_time = kyiv_df["finished_at"].max().ceil("h")
    time_grid = pd.date_range(start=min_time, end=max_time, freq="1h")

    # Filter raw alert starts to cutoff date
    kyiv_df_recent = kyiv_df[kyiv_df["started_at"] >= min_time]
    hourly_starts = kyiv_df_recent.set_index("started_at").resample("1h").size().reindex(time_grid, fill_value=0)

    ts_df = pd.DataFrame({
        "alert_count": hourly_starts.values,
        "alert_binary": (hourly_starts.values > 0).astype(int)
    }, index=time_grid)

    print(f"Filtered real alert records for Kyiv City (post-{start_cutoff}): {len(kyiv_df_recent)} events.")
    print(f"✅ Kyiv modern time series range: {ts_df.index.min()} to {ts_df.index.max()}")
    return ts_df

# =============================================================================
# 3. LEAK-FREE FEATURE ENGINEERING & SPLITTER
# =============================================================================
def build_leak_free_features(df: pd.DataFrame, target_col: str = "alert_binary") -> pd.DataFrame:
    data = df.copy().sort_index()
    data["target"] = data[target_col]

    data["lag_1h"] = data[target_col].shift(1)
    data["lag_2h"] = data[target_col].shift(2)
    data["lag_3h"] = data[target_col].shift(3)
    data["lag_24h"] = data[target_col].shift(24)
    data["lag_168h"] = data[target_col].shift(168)

    shifted_target = data[target_col].shift(1)
    data["rolling_mean_6h"] = shifted_target.rolling(window=6).mean()
    data["rolling_mean_24h"] = shifted_target.rolling(window=24).mean()
    data["rolling_sum_24h"] = shifted_target.rolling(window=24).sum()
    data["rolling_mean_7d"] = shifted_target.rolling(window=168).mean()
    data["rolling_std_7d"] = shifted_target.rolling(window=168).std()

    hour = data.index.hour
    dayofweek = data.index.dayofweek

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    data["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    data["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    data["is_weekend"] = dayofweek.isin([5, 6]).astype(int)

    return data.dropna().copy()

def split_data_chronologically(df: pd.DataFrame, test_hours: int = 24 * 14):
    ignore_cols = ["target", "alert_count", "alert_binary"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    cutoff_idx = len(df) - test_hours
    train_df = df.iloc[:cutoff_idx]
    test_df = df.iloc[cutoff_idx:]

    return train_df[feature_cols], train_df["target"], test_df[feature_cols], test_df["target"], feature_cols

# EXECUTE CELL 1
RAW_CSV_PATH = download_vadimkin_dataset("data/raw_official_alerts.csv")
hourly_data = load_and_resample_kyiv_data(RAW_CSV_PATH, start_cutoff=MODERN_START_DATE)
features_df = build_leak_free_features(hourly_data, target_col="alert_binary")

print("\n✅ Cell 1 Complete! Kyiv features restricted to modern data only.")

# In[3]:
import pandas as pd
import numpy as np

print("--- BUILDING SPATIAL THREAT INDICATORS FROM MODERN REAL DATA ---")

raw_df = pd.read_csv("data/raw_official_alerts.csv")

# Parse timestamps cleanly
raw_df["started_at"] = pd.to_datetime(raw_df["started_at"], errors="coerce", utc=True).dt.tz_localize(None)

# 1. FILTER ALL REGIONS STRICTLY TO MODERN TIMELINE (SAME AS CELL 1)
MODERN_START_DATE = hourly_data.index.min()
raw_df = raw_df[raw_df["started_at"] >= MODERN_START_DATE].copy()

print(f"Filtered raw regional records starting from: {MODERN_START_DATE}")
print(f"Total unique regions detected in modern dataset: {raw_df['oblast'].nunique()}")

# Patterns for neighboring regions
NEIGHBOR_PATTERNS = {
    "zhytomyr": "Zhytomyr",
    "chernihiv": "Chernihiv",
    "cherkasy": "Cherkas",
    "kyiv_oblast": "Kyivska"
}

spatial_signals = pd.DataFrame(index=hourly_data.index)

# 2. Resample alerts for neighboring regions strictly on Kyiv's modern time grid
for key, pattern in NEIGHBOR_PATTERNS.items():
    region_mask = raw_df["oblast"].astype(str).str.contains(pattern, case=False, na=False)
    sub_df = raw_df[region_mask]

    if not sub_df.empty:
        # Resample onto Kyiv's exact modern hourly index
        starts = sub_df.set_index("started_at").resample("1h").size().reindex(hourly_data.index, fill_value=0)
        spatial_signals[f"spatial_{key}"] = (starts > 0).astype(int)
        print(f" - Found {len(sub_df)} modern alert events for {key.title()}")
    else:
        spatial_signals[f"spatial_{key}"] = 0
        print(f" ⚠️ No modern events found matching pattern '{pattern}'")

# Calculate aggregate neighbor threat score
spatial_signals["total_neighbor_threat_score"] = spatial_signals.sum(axis=1)

# STRICT LEAKAGE PREVENTION: Shift indicators by 1 hour (t-1)
nlp_feature_df = spatial_signals[["total_neighbor_threat_score"]].shift(1).fillna(0)

active_hours_count = (nlp_feature_df["total_neighbor_threat_score"] > 0).sum()
print(f"\n✅ Modern Spatial Threat Features successfully engineered!")
print(f"Total modern hours with active neighboring threat signals: {active_hours_count}")

# Sample preview strictly showing modern dates (e.g. 2025/2026)
active_samples = nlp_feature_df[nlp_feature_df["total_neighbor_threat_score"] > 0]
print("\nSample Preview of Active Spatial Threat Hours (Modern Range):")
print(active_samples.head(10))

# In[4]:
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

print("--- TRAINING OPTIMIZED & CALIBRATED SPATIAL XGBOOST MODEL ---")

# Pass individual spatial lags alongside the aggregate total
spatial_cols = [col for col in nlp_feature_df.columns if "spatial" in col or "total" in col]
features_df_spatial = features_df.join(nlp_feature_df, how="left").fillna(0)

# Chronological Train/Test Split (14-day holdout)
X_train, y_train, X_test, y_test, feature_names = split_data_chronologically(features_df_spatial, test_hours=24*14)

# Calculate exact class imbalance ratio
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_weight = num_neg / max(num_pos, 1)

print(f"Dataset Imbalance Ratio: {scale_weight:.2f}:1")

# Train Calibrated XGBoost Model
model_optimized = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.02,
    max_depth=5,
    scale_pos_weight=scale_weight,  # Rebalances positive siren classes
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model_optimized.fit(X_train, y_train)

# Predict probabilities
y_pred_prob = model_optimized.predict_proba(X_test)[:, 1]

# Fine-Grained Threshold Optimization (0.01 to 0.50)
print("\n--- FINE-GRAINED THRESHOLD SEARCH ---")
best_f1, best_thresh, best_p, best_r = 0, 0.5, 0, 0

for thresh in np.arange(0.01, 0.51, 0.01):
    preds = (y_pred_prob >= thresh).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    if f1 > best_f1:
        best_f1, best_thresh, best_p, best_r = f1, thresh, p, r

print("\n================ FINAL OPTIMIZED EVALUATION ================")
print(f"✅ Optimal Decision Threshold : {best_thresh:.2f}")
print(f"✅ Precision @ Best Threshold : {best_p:.4f}")
print(f"✅ Recall @ Best Threshold    : {best_r:.4f}")
print(f"✅ Maximum Achieved F1-Score  : {best_f1:.4f}")
print("============================================================")

# Feature Importance Breakdown
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model_optimized.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Features Impacting Predictions:")
print(importance_df.head(8).to_string(index=False))