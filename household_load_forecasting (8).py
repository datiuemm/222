

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

import xgboost as xgb

def mape_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_daily_agg(df):
    df = df.copy()
    if "Datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Date"] = df["Datetime"].dt.normalize()

    weather_cols_all = [
        "pressure_at_sea", "precip_dur_past10min", "wind_dir",
        "temp_dew", "pressure", "visib_mean_last10min", "temp_dry",
        "humidity", "cloud_cover", "visibility"
    ]
    weather_cols = [c for c in weather_cols_all if c in df.columns]

    agg = {"kWh": "sum"}
    agg.update({c: "mean" for c in weather_cols})

    out = df.groupby("Date").agg(agg)
    out.index = pd.to_datetime(out.index)
    out["is_weekend"] = (out.index.weekday >= 5).astype(int)
    out["month"] = out.index.month
    out["day_of_week"] = out.index.dayofweek
    out["month_sin"] = np.sin(2*np.pi*(out["month"]/12.0))
    out["month_cos"] = np.cos(2*np.pi*(out["month"]/12.0))

    years = range(out.index.year.min(), out.index.year.max() + 1)
    vn_holidays = holidays.Vietnam(years=years)
    vn_set = set(vn_holidays.keys())
    out["is_holiday"] = out.index.map(lambda d: int(d in vn_set))

    if "temp_dry" in out.columns:
        out["temp_dry_sq"] = out["temp_dry"]**2
        out["temp_dry_cub"] = out["temp_dry"]**3
    if "humidity" in out.columns:
        out["humidity_sq"] = out["humidity"]**2
    if ("temp_dry" in out.columns) and ("humidity" in out.columns):
        out["temp_humidity_interaction"] = out["temp_dry"] * out["humidity"]
    if ("temp_dry" in out.columns) and ("temp_dew" in out.columns):
        out["dew_spread"] = out["temp_dry"] - out["temp_dew"]

    return out


def make_time_features(index, fourier_order=2, weekly_seasonal=True):
    fourier = CalendarFourier(freq="YE", order=fourier_order)
    dp = DeterministicProcess(
        index=index,
        constant=True,
        order=1,                    # linear trend
        seasonal=weekly_seasonal,   # weekly dummies
        additional_terms=[fourier], # annual Fourier
        drop=True
    )
    return dp.in_sample()


def add_kwh_derivatives(df):
    df = df.copy()
    df["kWh_diff1"] = df["kWh"].diff(1)
    df["kWh_pct"] = df["kWh"].pct_change().replace([np.inf, -np.inf], np.nan)
    return df


def add_kwh_lags_rollings(df,
                          lags=(1,2,3,7,14,21,28),
                          windows=(7,14,30)):

    df = df.copy()
    for L in lags:
        df[f"kWh_t{L}"] = df["kWh"].shift(L)
    for W in windows:
        past = df["kWh"].shift(1)
        roll = past.rolling(W, min_periods=1)
        df[f"kWh_roll{W}_mean"]  = roll.mean()
        df[f"kWh_roll{W}_std"]   = roll.std()
        df[f"kWh_roll{W}_min"]   = roll.min()
        df[f"kWh_roll{W}_max"]   = roll.max()
        df[f"kWh_roll{W}_skew"]  = roll.skew()
    return df


def _sample_params(rng, n_train):
    if n_train < 180:
        n_estimators_hi = 900
        depth_hi = 6
    else:
        n_estimators_hi = 1600
        depth_hi = 7

    return {
        "n_estimators":     int(rng.integers(400, n_estimators_hi + 1)),
        "learning_rate":    float(rng.uniform(0.02, 0.08)),
        "max_depth":        int(rng.integers(4, depth_hi + 1)),
        "subsample":        float(rng.uniform(0.80, 0.95)),
        "colsample_bytree": float(rng.uniform(0.80, 0.95)),
        "reg_lambda":       float(rng.uniform(0.8, 2.0)),
        "min_child_weight": int(rng.integers(1, 8)),
        "gamma":            float(rng.uniform(0.0, 0.2)),
    }

def _to_train_params(p):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": p["learning_rate"],
        "max_depth": p["max_depth"],
        "subsample": p["subsample"],
        "colsample_bytree": p["colsample_bytree"],
        "lambda": p["reg_lambda"],
        "min_child_weight": p["min_child_weight"],
        "gamma": p["gamma"],
    }

from sklearn.metrics import mean_squared_error  # dùng cho RMSE trong CV

def tune_xgb_time_series(X, y, n_splits=3, n_iter=25, random_state=42):
    rng = np.random.default_rng(random_state)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best = None

    for _ in range(n_iter):
        params = _sample_params(rng, len(X))
        train_params = _to_train_params(params)
        rmses = []

        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)

            bst = xgb.train(
                params=train_params,
                dtrain=dtr,
                num_boost_round=params["n_estimators"],
                evals=[(dva, "valid")],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            pred_va = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
            rmse_va = np.sqrt(mean_squared_error(y_va, pred_va))
            rmses.append(rmse_va)

        score = float(np.mean(rmses))
        if (best is None) or (score < best["cv_rmse"]):
            best = {"params": params, "cv_rmse": score, "best_num_boost_round": int(bst.best_iteration + 1)}

    val_cut = int(len(X) * 0.8)
    val_cut = min(max(val_cut, 1), len(X) - 1)
    X_tr_sub, X_val = X.iloc[:val_cut], X.iloc[val_cut:]
    y_tr_sub, y_val = y.iloc[:val_cut], y.iloc[val_cut:]

    dtr = xgb.DMatrix(X_tr_sub, label=y_tr_sub)
    dva = xgb.DMatrix(X_val, label=y_val)

    final_bst = xgb.train(
        params=_to_train_params(best["params"]),
        dtrain=dtr,
        num_boost_round=best["params"]["n_estimators"],
        evals=[(dva, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    best["model"] = final_bst
    best["best_n_estimators"] = int(final_bst.best_iteration + 1)
    return best


def one_shot_forecast(
    input_csv,
    test_size=0.2,
    fourier_order=2,
    random_state=42,
    plot=False
):
    raw = pd.read_csv(input_csv)
    data_by_day = build_daily_agg(raw)


    X_time = make_time_features(data_by_day.index, fourier_order=fourier_order, weekly_seasonal=True)
    X_time["day_of_week"] = data_by_day["day_of_week"]
    X_time["is_holiday"] = data_by_day["is_holiday"]

    y = data_by_day["kWh"].astype(float)


    Xtr, Xte, ytr, yte = train_test_split(X_time, y, test_size=test_size, shuffle=False)


    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xtr, ytr)
    ytr_hat_lin = pd.Series(lr.predict(Xtr), index=Xtr.index)
    yte_hat_lin = pd.Series(lr.predict(Xte), index=yte.index)

    df_feat = data_by_day.copy()
    df_feat = add_kwh_derivatives(df_feat)
    df_feat = add_kwh_lags_rollings(df_feat)

    X1 = df_feat.drop(columns=["kWh"]).loc[y.index]
    X1_tr = X1.loc[ytr.index].copy()
    X1_te = X1.loc[yte.index].copy()

    valid_tr_idx = X1_tr.dropna().index
    X1_tr = X1_tr.loc[valid_tr_idx]
    ytr_adj = ytr.loc[valid_tr_idx]
    ytr_hat_lin_adj = ytr_hat_lin.loc[valid_tr_idx]

    valid_te_idx = X1_te.dropna().index
    X1_te = X1_te.loc[valid_te_idx]
    yte_adj = yte.loc[valid_te_idx]
    yte_hat_lin_adj = yte_hat_lin.loc[valid_te_idx]

    resid_tr = (ytr_adj - ytr_hat_lin_adj)

    best = tune_xgb_time_series(X1_tr, resid_tr, n_splits=3, n_iter=25, random_state=random_state)

    y_pred_boosted_tr = pd.Series(
        best["model"].predict(xgb.DMatrix(X1_tr), iteration_range=(0, best["best_n_estimators"])),
        index=valid_tr_idx
    ) + ytr_hat_lin_adj

    y_pred_boosted_te = pd.Series(
        best["model"].predict(xgb.DMatrix(X1_te), iteration_range=(0, best["best_n_estimators"])),
        index=valid_te_idx
    ) + yte_hat_lin_adj

    metrics_linear = {
        "MAE":  float(mean_absolute_error(yte_adj, yte_hat_lin_adj)),
        "RMSE": rmse(yte_adj, yte_hat_lin_adj),
        "MAPE(%)": mape_manual(yte_adj, yte_hat_lin_adj),
        "R2":   float(r2_score(yte_adj, yte_hat_lin_adj)),
    }
    metrics_hybrid = {
        "MAE":  float(mean_absolute_error(yte_adj, y_pred_boosted_te)),
        "RMSE": rmse(yte_adj, y_pred_boosted_te),
        "MAPE(%)": mape_manual(yte_adj, y_pred_boosted_te),
        "R2":   float(r2_score(yte_adj, y_pred_boosted_te)),
    }

    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(yte_adj.index, yte_adj.values, label="Actual (Test)", color="black")
        plt.plot(y_pred_boosted_te.index, y_pred_boosted_te.values, label="Hybrid (Test)", color="red")
        plt.title("Actual vs Hybrid (Test only)")
        plt.xlabel("Date"); plt.ylabel("kWh"); plt.legend(); plt.tight_layout(); plt.show()

    return {
        "test_index": yte_adj.index,
        "y_test": yte_adj,
        "y_pred_linear": yte_hat_lin_adj,
        "y_pred_hybrid": y_pred_boosted_te,
        "metrics_linear": metrics_linear,
        "metrics_hybrid": metrics_hybrid,
        "best_xgb_params": best["params"],
        "best_val_resid_rmse_cv": best["cv_rmse"],
        "best_n_estimators_after_es": best["best_n_estimators"]
    }


if __name__ == "__main__":
    INPUT_CSV = "household-load.csv"

    out = one_shot_forecast(
        INPUT_CSV,
        test_size=0.2,
        fourier_order=2,
        plot=True
    )

    print("\n== METRICS (v2 - AutoTune XGB) ==")
    print("Linear (time-only):", out["metrics_linear"])
    print("Hybrid (Linear + XGB residual):", out["metrics_hybrid"])
    print("Best XGB Params:", out["best_xgb_params"])
    print("Residual RMSE (CV):", round(out["best_val_resid_rmse_cv"], 4))
    print("Best n_estimators (after early stopping):", out["best_n_estimators_after_es"])