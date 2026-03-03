import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')
# File paths
sand_path = "2d_sand_proportion.npy"
prod_logs_path = "Well_log_data_production_wells.csv"
preprod_logs_path = "Well_log_data_preproduction_wells.csv"
prod_hist_path = "Production_history_production_wells.csv"

# Load data
sand = np.load(sand_path)
prod = pd.read_csv(prod_logs_path)
preprod = pd.read_csv(preprod_logs_path)
hist = pd.read_csv(prod_hist_path)


# Get one (X, Y) per well
prod_xy = prod.groupby("Well_ID", as_index=False)[["X", "Y"]].mean()
preprod_xy = preprod.groupby("Well_ID", as_index=False)[["X", "Y"]].mean()


# Build well-level cumulative oil
hist.columns = [c.strip().lower() for c in hist.columns]

well_col = [c for c in hist.columns if "well" in c and "id" in c][0]
time_col = ([c for c in hist.columns if "date" in c] +
            [c for c in hist.columns if "month" in c] +
            [c for c in hist.columns if "time" in c])[0]
cum_oil_col = [c for c in hist.columns if "oil" in c and ("cum" in c or "cumulative" in c)][0]

# Parse dates
t = pd.to_datetime(hist[time_col], errors="coerce")
if t.notna().mean() > 0.8:
    hist[time_col] = t

hist[cum_oil_col] = pd.to_numeric(hist[cum_oil_col], errors="coerce")
hist = hist.dropna(subset=[well_col, time_col, cum_oil_col]).copy()
hist[well_col] = hist[well_col].astype(int)
hist = hist.sort_values([well_col, time_col])

# Fix cumulative dips
hist["cum_oil_fixed"] = hist.groupby(well_col)[cum_oil_col].cummax()
final_cum = hist.groupby(well_col, as_index=False)["cum_oil_fixed"].max()
final_cum = final_cum.rename(columns={well_col: "Well_ID",
                                      "cum_oil_fixed": "final_cum_oil"})

# Merge into XY table
prod_xy2 = prod_xy.merge(final_cum, on="Well_ID", how="left")

vmin, vmax = np.percentile(sand, [2, 98])

plt.figure(figsize=(9, 7))

# Background map
im = plt.imshow(sand, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
plt.colorbar(im, label="Sand proportion (2–98% clipped)")

# Production wells (circles)
sc = plt.scatter(
    prod_xy2["X"], prod_xy2["Y"],
    s=90, marker="o",
    c=prod_xy2["final_cum_oil"],
    cmap="viridis",
    edgecolors="black", linewidths=0.6,
    zorder=3
)
cb = plt.colorbar(sc)
cb.set_label("Final cumulative oil (bbl)")

# Preproduction wells (triangles)
plt.scatter(
    preprod_xy["X"], preprod_xy["Y"],
    s=110, marker="^",
    c="#ff9500",
    edgecolors="black", linewidths=0.7,
    label="Preproduction wells",
    zorder=4
)

# Label each well with Well ID
for _, r in prod_xy2.iterrows():
    plt.text(
        r["X"] + 0.8, r["Y"] + 0.8,
        str(int(r["Well_ID"])),
        color="red",
        fontsize=8,
        weight="bold",
        zorder=5
    )

for _, r in preprod_xy.iterrows():
    plt.text(
        r["X"] + 0.8, r["Y"] + 0.8,
        str(int(r["Well_ID"])),
        color="red",
        fontsize=9,
        weight="bold",
        zorder=6
    )

plt.title("Sand Proportion Map + Wells (Labeled by Well ID)")
plt.xlabel("X index")
plt.ylabel("Y index")
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.show()

# Import and preprocess the dataset
prod_history = pd.read_csv('Production_history_production_wells.csv')
train_logs = pd.read_csv('Well_log_data_production_wells.csv')
test_logs = pd.read_csv('Well_log_data_preproduction_wells.csv')

prod_history['Date'] = pd.to_datetime(prod_history['Date'])
targets = []
for well_id in prod_history['Well_ID'].unique():
    w_data = prod_history[prod_history['Well_ID'] == well_id].sort_values('Date').reset_index(drop=True)
    if len(w_data) > 36:
        cum_oil = w_data.loc[36, 'Cumulative Oil Production, BBL'] - w_data.loc[0, 'Cumulative Oil Production, BBL']
        targets.append({'Well_ID': well_id, 'Target_Oil_3yr': cum_oil})
targets_df = pd.DataFrame(targets)

cols_to_impute = [c for c in train_logs.columns if c != 'Well_ID']
imputer = KNNImputer(n_neighbors=3, weights='distance')

train_logs[cols_to_impute] = imputer.fit_transform(train_logs[cols_to_impute])
test_logs[cols_to_impute] = imputer.transform(test_logs[cols_to_impute])

# Feature engineering
def add_physics_features(df):
    df["dt_p"] = 1.0 / df["Vp"]
    rho = df["rho_b"] * 1000.0
    Vp2 = df["Vp"] ** 2
    Vs2 = df["Vs"] ** 2
    df["mu"] = rho * Vs2
    df["E"] = 2.0 * df["mu"] * (1.0 + ((Vp2 - 2*Vs2)/(2*(Vp2-Vs2))))

    df['Pore_Perm'] = df['phi'] * df['perm']
    df['RQI'] = np.sqrt(df['perm'] / (df['phi']+1e-6))
    df['Clean_Index'] = df['phi'] / (df['GR']+1e-6)
    return df

def remove_features(df):
    if "rho_m" in df.columns:
        df = df.drop(columns=["rho_m"])
    return df

train_logs = add_physics_features(train_logs)
test_logs = add_physics_features(test_logs)
train_logs = remove_features(train_logs)
test_logs = remove_features(test_logs)

all_features = [c for c in train_logs.columns if c not in ['Well_ID']]
X_train_agg = train_logs.groupby('Well_ID')[all_features].mean().reset_index()
X_test_agg  = test_logs.groupby('Well_ID')[all_features].mean().reset_index()

train_df = targets_df.merge(X_train_agg, on='Well_ID', how='inner')

X = train_df[all_features].values
y = train_df['Target_Oil_3yr'].values

X_predict = X_test_agg[all_features].values
submit_ids = X_test_agg['Well_ID'].values

y_log = np.log1p(y)

# ----------------------------
# Model selection (point prediction) with repeated CV
# ----------------------------
cv = RepeatedKFold(n_splits=min(5, len(X)), n_repeats=5, random_state=42)

candidates = {
    "ridge_poly": Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler2", StandardScaler()),
        ("regressor", Ridge(alpha=10.0)),
    ]),
    "elasticnet": Pipeline([
        ("scaler", StandardScaler()),
        ("scaler2", StandardScaler()),
        ("regressor", ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=42, max_iter=20000)),
    ]),
    "rf": RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    ),
    "gbr": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

scores = []
oof_store = {}
resid_store = {}

def repeated_oof_predictions(model, X, y_log, cv):
    """Return per-sample mean OOF predictions in log space and a pooled residual sample.

    RepeatedKFold does not form a single partition of the data, so we can't use
    cross_val_predict. Instead we collect many OOF predictions per sample and
    average them.
    """
    preds_by_index = [[] for _ in range(len(X))]
    resid_pool = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X), start=1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr = y_log[tr_idx]
        y_val = y_log[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        pred_log = m.predict(X_val)

        for j, idx in enumerate(val_idx):
            preds_by_index[idx].append(float(pred_log[j]))

        resid_pool.extend((y_val - pred_log).tolist())

    # Mean prediction per sample (each should have n_repeats predictions)
    oof_mean = np.array([np.mean(p) if len(p) else np.nan for p in preds_by_index], dtype=float)
    if np.isnan(oof_mean).any():
        raise RuntimeError("Some samples never received an OOF prediction. Check CV settings.")

    return oof_mean, np.array(resid_pool, dtype=float)

for name, model in candidates.items():
    oof_pred_log_mean, resid_pool = repeated_oof_predictions(model, X, y_log, cv)
    oof_pred = np.expm1(oof_pred_log_mean)

    mae = mean_absolute_error(y, oof_pred)
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    r2 = r2_score(y, oof_pred)

    scores.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    oof_store[name] = oof_pred_log_mean
    resid_store[name] = resid_pool

scores_df = pd.DataFrame(scores).sort_values(["MAE", "RMSE"], ascending=True)
print("\nOOF model comparison (repeated CV):")
print(scores_df.to_string(index=False))

best_name = scores_df.iloc[0]["model"]
best_model = candidates[best_name]
print(f"\nSelected model: {best_name}")

# ----------------------------
# Residual distribution for uncertainty (OOF residuals in log space)
# ----------------------------
best_oof_pred_log = oof_store[best_name]
oof_resid_log = resid_store[best_name]

# ----------------------------
# Fit final model on all training data
# ----------------------------
best_model.fit(X, y_log)

# ----------------------------
# Probabilistic submission via bootstrap + OOF residual sampling
# ----------------------------
n_iterations = 100
realizations = np.zeros((len(X_predict), n_iterations))

rng = np.random.default_rng(42)

for i in range(n_iterations):
    # Bootstrap wells (model uncertainty)
    boot_idx = rng.integers(0, len(X), size=len(X))
    X_boot = X[boot_idx]
    y_boot = y_log[boot_idx]

    model_i = clone(candidates[best_name])
    model_i.fit(X_boot, y_boot)

    pred_log = model_i.predict(X_predict)

    # Add sampled OOF residuals (aleatoric + remaining model misspec)
    noise = rng.choice(oof_resid_log, size=len(pred_log), replace=True)
    realizations[:, i] = np.expm1(pred_log + noise)

results_df = pd.DataFrame({
    "Well_ID": submit_ids,
    "Prediction_BBL": realizations.mean(axis=1),
})

# Realizations columns (if required by the competition)
for i in range(n_iterations):
    results_df[f"R{i+1}"] = realizations[:, i]

results_df.to_csv("solution.csv", index=False)
print("File 'solution.csv' saved.")