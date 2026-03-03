import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.utils import resample
from sklearn.impute import KNNImputer
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
plt.show()# Import and preprocess the dataset
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

# Estimate log-space RMSE with LOOCV (more stable for tiny N)
oof_pred_log = np.zeros_like(y_log)
for i in range(len(X)):
    tr_idx = np.arange(len(X)) != i
    model_i = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler2', StandardScaler()),
        ('regressor', Ridge(alpha=10.0))
    ])
    model_i.fit(X[tr_idx], y_log[tr_idx])
    oof_pred_log[i] = model_i.predict(X[i:i+1])[0]

log_rmse = np.sqrt(mean_squared_error(y_log, oof_pred_log))
print(f"LOOCV RMSE (log-space): {log_rmse:.4f}")

# Fit final base model on all data (for later bootstrap)
base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler2', StandardScaler()),
    ('regressor', Ridge(alpha=10.0))
])
base_model.fit(X, y_log)

# ----------------------------
# Folded evaluation (more stable than a single 80/20 split when N is tiny)
# ----------------------------
kf = KFold(n_splits=min(6, len(X)), shuffle=True, random_state=42)
oof_realizations = np.zeros((len(X), 100))

n_features = X.shape[1]
all_indices = np.arange(n_features)
n_iterations = 100

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold_log = y_log[train_idx]

    fold_preds = np.zeros((len(val_idx), n_iterations))

    for i in range(n_iterations):
        X_boot, y_boot_log = resample(X_train_fold, y_train_fold_log, replace=True, random_state=i)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler2', StandardScaler()),
            ('regressor', Ridge(alpha=10.0))
        ])
        model.fit(X_boot, y_boot_log)

        pred_log = model.predict(X_val_fold)

        # Residual bootstrap noise in log-space (more realistic than N(0, rmse) here)
        resid = y_boot_log - model.predict(X_boot)
        noise = np.random.choice(resid, size=len(pred_log), replace=True)

        pred_log_noisy = pred_log + noise
        fold_preds[:, i] = np.expm1(pred_log_noisy)

    oof_realizations[val_idx, :] = fold_preds
    print(f"Fold {fold} done")

y_pred_mean = oof_realizations.mean(axis=1)
r2 = r2_score(y, y_pred_mean)
mape = mean_absolute_percentage_error(y, y_pred_mean)

# Coverage / goodness (P10-P90)
p10 = np.percentile(oof_realizations, 10, axis=1)
p90 = np.percentile(oof_realizations, 90, axis=1)
inside = (y >= p10) & (y <= p90)
goodness = inside.mean()

print("Results (KFold OOF):")
print(f"R2 Score:       {r2:.4f}")
print(f"MAPE:           {mape:.1%}")
print(f"Goodness Score: {goodness:.2f}")

realizations = np.zeros((len(X_predict), n_iterations))

# Submission
for i in range(n_iterations):
    X_boot, y_boot_log = resample(X, y_log, replace=True, random_state=i)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler2', StandardScaler()),
        ('regressor', Ridge(alpha=10.0))
    ])
    model.fit(X_boot, y_boot_log)

    pred_log = model.predict(X_predict)
    noise = np.random.normal(0, log_rmse, size=len(pred_log))
    realizations[:, i] = np.expm1(pred_log + noise)

results_df = pd.DataFrame()
results_df['Well_ID'] = submit_ids
results_df['Prediction_BBL'] = realizations.mean(axis=1)

for i in range(n_iterations):
    results_df[f'R{i+1}'] = realizations[:, i]

results_df.to_csv('solution.csv', index=False)
print("File 'solution.csv' saved.")
# Cross Validation

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

# Base model pipeline definition
pipeline_steps = [
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler2', StandardScaler()),
    ('regressor', Ridge(alpha=10.0))
]

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=50)
oof_realizations = np.zeros((len(X), 100))
y_oof_true = np.zeros(len(X))

n_features = X.shape[1]
all_indices = np.arange(n_features)
n_iterations = 100

fold = 1
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]

    y_train_fold_log = y_log[train_idx]
    y_val_fold_real  = y[val_idx]
    y_oof_true[val_idx] = y_val_fold_real

    base_model = Pipeline(pipeline_steps)
    base_model.fit(X_train_fold, y_train_fold_log)

    fold_rmse_log = np.sqrt(mean_squared_error(y_train_fold_log, base_model.predict(X_train_fold)))

    fold_preds = np.zeros((len(X_val_fold), n_iterations))

    for i in range(n_iterations):
        X_boot, y_boot_log = resample(X_train_fold, y_train_fold_log, replace=True, random_state=i)

        model = Pipeline(pipeline_steps)
        model.fit(X_boot, y_boot_log)

        pred_log = model.predict(X_val_fold)

        noise = np.random.normal(0, fold_rmse_log, size=len(pred_log))

        fold_preds[:, i] = np.expm1(pred_log + noise)

    oof_realizations[val_idx, :] = fold_preds
    print(f"Fold {fold} Done. RMSE): {fold_rmse_log:.4f}")
    fold += 1

# Metrics (Calculated on Real Values)
y_pred_mean = oof_realizations.mean(axis=1)

r2 = r2_score(y, y_pred_mean)
mape = mean_absolute_percentage_error(y, y_pred_mean)

ensemble_rmse = np.sqrt(mean_squared_error(y, y_pred_mean))
ensemble_spread = np.std(oof_realizations, axis=1).mean()
spread_skill = ensemble_spread / ensemble_rmse

crps_estimates = []
for i in range(len(y)):
    forecasts = np.sort(oof_realizations[i, :])
    truth = y[i]
    crps_estimates.append(np.mean(np.abs(forecasts - truth)))
avg_crps = np.mean(crps_estimates)

print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.1%}")