import importlib, sys, subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Auto-install missing packages ---
def _pip_install(pkg):
    # Sử dụng '-U' (Upgrade) để đảm bảo gói được cập nhật/cài đặt
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Cài đặt LightGBM
try:
    import lightgbm as lgb
except Exception:
    print("Installing lightgbm...")
    _pip_install("lightgbm")
    import lightgbm as lgb

# Cài đặt Optuna
try:
    import optuna
except Exception:
    print("Installing optuna...")
    _pip_install("optuna")
    import optuna

# Cài đặt optuna-integration[lightgbm] và import LightGBMPruningCallback
# Đây là phần quan trọng để khắc phục lỗi 'ModuleNotFoundError: No module named 'optuna.integration.lightgbm''
try:
    from optuna.integration import LightGBMPruningCallback
except Exception as e:
    if "ModuleNotFoundError" in str(e):
        print("Installing optuna-integration[lightgbm]...")
        _pip_install("optuna-integration[lightgbm]")
        # Phải invalidate caches để Python nhận ra gói mới được cài đặt
        importlib.invalidate_caches()
        from optuna.integration import LightGBMPruningCallback
    else:
        raise e

# Cài đặt holidays
try:
    import holidays
except Exception:
    print("Installing holidays...")
    _pip_install("holidays")
    import holidays

# --- Scikit-learn & Statsmodels imports ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from optuna.samplers import TPESampler

# -------------------------
# 1. Metrics & Helpers
# -------------------------
def mape_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Ngăn chia cho 0 bằng cách dùng np.clip
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def _eval_metrics(y_true, y_pred):
    mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    y_true = y_true.loc[mask]
    y_pred = y_pred.loc[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE(%)": np.nan, "R2": np.nan}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE(%)": float(mape_manual(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

# -------------------------
# 2. Feature Building
# -------------------------
def build_daily_agg(df):
    df = df.copy()
    if "Datetime" not in df.columns:
        # Nếu cột đầu tiên không phải Datetime, đổi tên cột đó
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
    try:
        # Sử dụng Vietnam holidays
        vn_holidays = holidays.Vietnam(years=years)
    except Exception:
        # Nếu không có data, dùng dict rỗng
        vn_holidays = {}
    vn_set = set(vn_holidays.keys())
    out["is_holiday"] = out.index.map(lambda d: int(d in vn_set))

    # Features tương tác và phi tuyến tính
    if "temp_dry" in out.columns:
        out["temp_dry_sq"] = out["temp_dry"]**2
        out["temp_dry_cub"] = out["temp_dry"]**3
    if "humidity" in out.columns:
        out["humidity_sq"] = out["humidity"]**2
    if ("temp_dry" in out.columns) and ("humidity" in out.columns):
        out["temp_humidity_interaction"] = out["temp_dry"] * out["humidity"]
    if ("temp_dry" in out.columns) and ("temp_dew" in out.columns):
        out["dew_spread"] = out["temp_dry"] - out["temp_dew"]

    out = out.rename_axis("Date")
    return out

def make_time_features(index, fourier_order=2, weekly_seasonal=True):
    # Sử dụng Fourier Terms cho chu kỳ năm (YE: Year End)
    fourier = CalendarFourier(freq="YE", order=fourier_order)
    dp = DeterministicProcess(
        index=index,
        constant=True, # Intercept
        order=1,       # Trend (linear)
        seasonal=weekly_seasonal, # Weekly seasonality (DayofWeek One-hot)
        additional_terms=[fourier],
        drop=True      # Bỏ các cột bị trùng lặp
    )
    return dp.in_sample()

def add_kwh_derivatives(df):
    df = df.copy()
    df["kWh_diff1"] = df["kWh"].diff(1)
    df["kWh_pct"] = df["kWh"].pct_change().replace([np.inf, -np.inf], np.nan)
    return df

def add_kwh_lags_rollings(df, lags=(1,2,3,7,14,21,28), windows=(7,14,30)):
    df = df.copy()
    # Lag features
    for L in lags:
        df[f"kWh_t{L}"] = df["kWh"].shift(L)
    # Rolling features
    for W in windows:
        past = df["kWh"].shift(1) # Chỉ dùng dữ liệu quá khứ (trước t-1)
        roll = past.rolling(W, min_periods=1)
        df[f"kWh_roll{W}_mean"]  = roll.mean()
        df[f"kWh_roll{W}_std"]   = roll.std()
        df[f"kWh_roll{W}_min"]   = roll.min()
        df[f"kWh_roll{W}_max"]   = roll.max()
        df[f"kWh_roll{W}_skew"]  = roll.skew()
    return df

# -------------------------
# 3. LightGBM + Optuna Tuning
# -------------------------
def _lgb_from_optuna_params(optuna_params):
    p = {}
    p["objective"] = "regression"
    p["metric"] = "rmse"
    p["verbosity"] = -1
    p["boosting_type"] = "gbdt"
    # Lấy các tham số được tối ưu
    p["learning_rate"] = float(optuna_params["learning_rate"])
    p["max_depth"] = int(optuna_params["max_depth"])
    # Thiết lập num_leaves nếu không được tối ưu riêng
    p["num_leaves"] = int(optuna_params.get("num_leaves", max(7, (2 ** p["max_depth"]) - 1)))
    p["bagging_fraction"] = float(optuna_params["bagging_fraction"])
    p["feature_fraction"] = float(optuna_params["feature_fraction"])
    p["lambda_l2"] = float(optuna_params["lambda_l2"])
    p["min_child_samples"] = int(optuna_params["min_child_samples"])
    p["min_split_gain"] = float(optuna_params["min_split_gain"])
    p["bagging_freq"] = 1
    return p

def _lgb_param_suggestor(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 7, 255),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.95),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 50),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }
    return params

def tune_lgb_time_series_optuna(X, y, n_splits=3, n_trials=30, random_state=42, early_stopping_rounds=100, verbose=False):
    # Sử dụng TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sampler = TPESampler(seed=random_state)

    def objective(trial):
        opt_params = _lgb_param_suggestor(trial)
        # Tối ưu số lượng estimators/rounds
        num_boost_round = trial.suggest_int("n_estimators", 200, 2000)
        lgb_params = _lgb_from_optuna_params(opt_params)
        rmses = []
        
        # Cross-Validation trên Time Series
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=False)
            
            # Callback cho Pruning
            pruning_callback = LightGBMPruningCallback(trial, "rmse")
            bst = lgb.train(
                params=lgb_params,
                train_set=dtr,
                num_boost_round=num_boost_round,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False), pruning_callback]
            )
            
            best_iter = int(bst.best_iteration) if bst.best_iteration is not None else num_boost_round
            pred_va = bst.predict(X_va, num_iteration=best_iter)
            rmses.append(float(np.sqrt(mean_squared_error(y_va, pred_va))))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_trial = study.best_trial
    raw_best_params = best_trial.params.copy()
    lgb_best_params = _lgb_from_optuna_params({k: v for k, v in raw_best_params.items() if k != "n_estimators"})
    n_estimators_best = int(raw_best_params.get("n_estimators", 1000))

    # Final fit trên tập huấn luyện (80% train -> 20% validation)
    val_cut = int(len(X) * 0.8)
    val_cut = min(max(val_cut, 1), len(X) - 1)
    X_tr_sub, X_val = X.iloc[:val_cut], X.iloc[val_cut:]
    y_tr_sub, y_val = y.iloc[:val_cut], y.iloc[val_cut:]
    dtr = lgb.Dataset(X_tr_sub, label=y_tr_sub, free_raw_data=False)
    dva = lgb.Dataset(X_val, label=y_val, reference=dtr, free_raw_data=False)
    final_bst = lgb.train(
        params=lgb_best_params,
        train_set=dtr,
        num_boost_round=n_estimators_best,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    )

    best = {
        "params": raw_best_params,
        "cv_rmse": float(study.best_value),
        "best_n_estimators": int(final_bst.best_iteration) if final_bst.best_iteration is not None else n_estimators_best,
        "model": final_bst,
        "study": study
    }
    return best

# -------------------------
# 4. Prediction Helpers (Logic Rolling/Recursive)
# -------------------------
def predict_next_day_rolling(model, X_test, lin_pred_test, num_iteration=None):
    # Dự báo h=1, tương đương với 1 bước của recursive.
    preds = []
    for i in range(len(X_test)):
        x_row = X_test.iloc[i:i+1]
        if num_iteration is None:
            resid_pred = model.predict(x_row)[0]
        else:
            resid_pred = model.predict(x_row, num_iteration=num_iteration)[0]
        final = resid_pred + lin_pred_test.iloc[i]
        preds.append(final)
    return pd.Series(preds, index=X_test.index)

def predict_horizon_rolling(model, df_daily, lin_pred_all, start_dates, horizon=3, num_iteration=None):
    results = {}
    idx_list = list(df_daily.index)

    # Loop qua từng ngày bắt đầu dự báo (start_dates)
    for start in start_dates:
        if start not in idx_list:
            continue
        pos = idx_list.index(start)
        preds_for_start = []
        ok = True

        # Copy dữ liệu gốc để thực hiện recursive
        current_df = df_daily.copy()

        # Loop qua từng bước thời gian trong horizon (h=1, h=2, ...)
        for h in range(horizon):
            target_pos = pos + h
            if target_pos >= len(idx_list):
                ok = False; break

            target_date = idx_list[target_pos]

            # CẬP NHẬT kWh BẰNG GIÁ TRỊ DỰ BÁO TRƯỚC ĐÓ (Recursive step)
            # Đây là logic chính của Rolling/Recursive
            for j, pred_val in enumerate(preds_for_start):
                date_to_update_kWh = idx_list[pos + j]
                if date_to_update_kWh in current_df.index:
                    # Gán giá trị dự báo làm giá trị 'actual' cho các lag/rollings trong lần dự báo tiếp theo
                    current_df.loc[date_to_update_kWh, "kWh"] = pred_val

            # Tính lại feature với dữ liệu giả lập/cập nhật
            df_feat_current = add_kwh_derivatives(current_df)
            df_feat_current = add_kwh_lags_rollings(df_feat_current)

            # Lấy dòng feature cho ngày cần dự báo
            x_row = df_feat_current.drop(columns=["kWh"], errors='ignore').loc[[target_date]]

            # Kiểm tra feature bị thiếu/NaN (thường do đầu chuỗi data, nhưng recursive cần kiểm tra)
            if x_row.empty or x_row.isna().any().any():
                ok = False; break

            # Dự báo Residuals
            if num_iteration is None:
                resid = float(model.predict(x_row)[0])
            else:
                resid = float(model.predict(x_row, num_iteration=num_iteration)[0])

            # Lấy Linear Prediction
            lin_val = float(lin_pred_all.loc[target_date])
            # Kết quả cuối cùng
            pred_kwh = resid + lin_val
            preds_for_start.append(pred_kwh)

        if ok and len(preds_for_start) == horizon:
            results[start] = preds_for_start

    if not results:
        return pd.DataFrame()
    cols = [f"h{h+1}" for h in range(horizon)]
    return pd.DataFrame.from_dict(results, orient="index", columns=cols)

# -------------------------
# 5. MAIN RUNNER (CHỈ CHẠY ROLLING/RECURSIVE)
# -------------------------
def run_compare_and_save(input_csv="household-load.csv", test_size=0.2,
                             fourier_order=2, n_trials=20, n_splits=3,
                             random_state=42, plot=False,
                             horizons=[3, 7, 30],
                             out_prefix="compare_out"):

    # 1. Đọc và tạo feature cơ bản
    raw = pd.read_csv(input_csv)
    data_by_day = build_daily_agg(raw)

    # --- Feature Engineering & Split ---
    # 2. Tạo feature thời gian cho Linear Model
    X_time = make_time_features(data_by_day.index, fourier_order=fourier_order, weekly_seasonal=True)
    X_time["day_of_week"] = data_by_day["day_of_week"]
    X_time["is_holiday"] = data_by_day["is_holiday"]
    y = data_by_day["kWh"].astype(float)

    # Chia train/test (TimeSeriesSplit)
    Xtr, Xte, ytr, yte = train_test_split(X_time, y, test_size=test_size, shuffle=False)

    # 3. Train Linear Model (Baseline/Level)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xtr, ytr)
    # Dự báo Linear cho toàn bộ tập dữ liệu (cần cho recursive)
    lin_full = pd.Series(lr.predict(X_time), index=X_time.index)
    yte_hat_lin = lin_full.loc[yte.index]
    ytr_hat_lin = lin_full.loc[ytr.index]


    # 4. Tạo Lag/Rolling features (cho LightGBM)
    df_feat = data_by_day.copy()
    df_feat = add_kwh_derivatives(df_feat)
    df_feat = add_kwh_lags_rollings(df_feat)
    X1 = df_feat.drop(columns=["kWh"]).loc[y.index] # X1 chứa cả lag features dựa trên actuals

    X1_tr = X1.loc[ytr.index].copy()
    X1_te = X1.loc[yte.index].copy()

    # Xử lý NaN do Lag/Rolling (chủ yếu ở đầu tập Train)
    valid_tr_idx = X1_tr.dropna().index
    X1_tr = X1_tr.loc[valid_tr_idx]
    ytr_adj = ytr.loc[valid_tr_idx]
    ytr_hat_lin_adj = ytr_hat_lin.loc[valid_tr_idx]

    valid_te_idx = X1_te.dropna().index # Chỉ lấy index của các điểm test có feature hợp lệ
    X1_te = X1_te.loc[valid_te_idx]
    yte_adj = yte.loc[valid_te_idx]
    yte_hat_lin_adj = yte_hat_lin.loc[valid_te_idx]

    resid_tr = (ytr_adj - ytr_hat_lin_adj)

    # --- Training LightGBM cho Residuals ---
    print("Tuning LightGBM (may take time)...")
    best = tune_lgb_time_series_optuna(X1_tr, resid_tr, n_splits=n_splits, n_trials=n_trials, random_state=random_state)
    model = best["model"]
    best_iter = best["best_n_estimators"]

    # Tạo folder output
    os.makedirs(out_prefix, exist_ok=True)
    summary_rows = []

    # -------------------------------------------------------
    # A. Chạy Baseline h=1 (Rolling/Recursive)
    # -------------------------------------------------------
    print("\n--- Running Horizon h=1 (Rolling) ---")
    y_pred_roll_h1 = predict_next_day_rolling(model, X1_te, yte_hat_lin_adj, num_iteration=best_iter)

    metrics_roll_h1 = _eval_metrics(yte_adj, y_pred_roll_h1)

    summary_rows.append({"Method": "Rolling (recursive)", "Horizon": 1, **metrics_roll_h1})

    # -------------------------------------------------------
    # B. Vòng lặp chạy các Horizons khác (3, 7, 30...)
    # -------------------------------------------------------
    for h in horizons:
        print(f"\n--- Running Horizon h={h} (Rolling) ---")

        # a. Dự báo Rolling (Recursive)
        # valid_te_idx là ngày đầu tiên của mỗi lần dự báo h-step
        df_roll_h = predict_horizon_rolling(model, data_by_day, lin_full, valid_te_idx, horizon=h, num_iteration=best_iter)

        # b. Lấy dữ liệu thực tế (Actuals) tương ứng
        actuals = {}
        idx_list = list(data_by_day.index)
        for start in df_roll_h.index:
            try:
                pos = idx_list.index(start)
                if pos + (h - 1) < len(idx_list):
                    vals = data_by_day["kWh"].iloc[pos: pos+h].tolist()
                    actuals[start] = vals
            except ValueError:
                continue

        if actuals:
            cols = [f"h{i+1}" for i in range(h)]
            df_actual_h = pd.DataFrame.from_dict(actuals, orient="index", columns=cols)
            df_actual_h.index.name = "Date"
        else:
            df_actual_h = pd.DataFrame()

        # c. Đánh giá Metrics (trên tất cả h-steps gộp lại)
        def get_all_metrics(df_pred, df_true):
            if df_pred.empty or df_true.empty:
                return {"MAE":np.nan,"RMSE":np.nan,"MAPE(%)":np.nan,"R2":np.nan}
            # Cần đảm bảo index và columns khớp nhau khi so sánh
            return _eval_metrics(df_true.stack(), df_pred.reindex(df_true.index)[df_true.columns].stack())

        roll_metrics = get_all_metrics(df_roll_h, df_actual_h)

        summary_rows.append({"Method": "Rolling (recursive)", "Horizon": h, **roll_metrics})

        # d. Lưu CSV
        if not df_roll_h.empty:
            df_roll_h.to_csv(os.path.join(out_prefix, f"roll_h{h}.csv"))
        if not df_actual_h.empty:
            df_actual_h.to_csv(os.path.join(out_prefix, f"actual_h{h}.csv"))

    # -------------------------------------------------------
    # C. Tổng hợp và In kết quả
    # -------------------------------------------------------
    comparison_all = pd.DataFrame(summary_rows)
    comparison_all = comparison_all.sort_values(by=["Horizon", "Method"])

    print("\n=== FINAL COMPARISON SUMMARY (All Horizons) ===")
    print(comparison_all)

    comparison_all.to_csv(os.path.join(out_prefix, "comparison_all_horizons.csv"), index=False)
    print(f"\nAll results saved to folder ./{out_prefix}/")

    return {
        "comparison_df": comparison_all,
        "model": model
    }

# -------------------------
# EXECUTION BLOCK
# -------------------------
if __name__ == "__main__":
    # GIẢ ĐỊNH file 'household-load.csv' tồn tại trong cùng thư mục
    # Nếu không có file này, code sẽ báo lỗi khi đọc CSV.
    # Để chạy, bạn cần có file 'household-load.csv'.
    try:
        out = run_compare_and_save(
            input_csv="household-load.csv",
            test_size=0.2,
            fourier_order=2,
            n_trials=30,
            n_splits=3,
            random_state=42,
            plot=False,
            horizons=[3, 7, 30], # Thử nghiệm với các horizon 3, 7, và 30 ngày
            out_prefix="compare_out_rolling"
        )
        print("\nDone. Check compare_out_rolling/ for CSV outputs.")
    except FileNotFoundError:
        print("\nERROR: The file 'household-load.csv' was not found.")
        print("Please ensure the CSV file is in the same directory as the script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
