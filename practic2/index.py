# -*- coding: utf-8 -*-
"""
Повний скрипт ARIMA/SARIMA для теми:
"Аналіз економічної ефективності Сонячних панелей"

Що робить код:
 1) Читає CSV (розділювач "~") з колонками: date, value.
 2) Перевіряє стаціонарність (ADF) та, за потреби, застосовує лог/різницювання d і сезонне різницювання D.
 3) Підбирає найкращу SARIMA за AIC на помірній решітці параметрів.
 4) Зберігає параметри моделі, діагностику та графіки (у тому числі 06_residuals.png, 07_resid_acf.png, 07_resid_pacf.png).
 5) Будує прогноз на 10 кроків уперед і записує його у forecast_10.csv.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
DATA_CSV_PATH = os.getenv("SOLAR_TS_PATH", "./solar_data.csv")  # шлях до даних (або змінна середовища)
CSV_SEP = "~"
DATE_COL = "date"
VALUE_COL = "value"
FREQ = "MS"               # місячні дані; за потреби: 'D','W','Q','A'
SEASONAL_PERIODS = 12     # сезонність для місячних даних
FORECAST_STEPS = 10
CONF_ALPHA = 0.05         # 95% ДІ

# Ґрати для перебору SARIMA
p_RANGE = range(0, 3)     # p
d_RANGE = [0, 1, 2]       # d
q_RANGE = range(0, 3)     # q
P_RANGE = range(0, 3)     # P
D_CANDIDATES = [0, 1]     # D
Q_RANGE = range(0, 3)     # Q

OUTPUT_DIR = "./outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def ensure_template_csv(path: str):
    """Створює демонстраційний шаблон CSV, якщо файл даних відсутній."""
    if os.path.exists(path):
        return
    idx = pd.date_range("2020-01-01", periods=36, freq=FREQ)
    trend = np.linspace(100, 160, len(idx))
    seasonal = 8 * np.sin(2 * np.pi * np.arange(len(idx)) / SEASONAL_PERIODS)
    noise = np.random.normal(0, 3, len(idx))
    values = trend + seasonal + noise
    df = pd.DataFrame({DATE_COL: idx.strftime("%Y-%m-%d"), VALUE_COL: values})
    df.to_csv(path, index=False, sep=CSV_SEP)

def adf_test(series: pd.Series) -> Dict[str, Any]:
    res = adfuller(series.dropna(), autolag='AIC')
    return {
        "adf_stat": res[0],
        "p_value": res[1],
        "lags_used": res[2],
        "n_obs": res[3],
        "crit_1%": res[4]['1%'],
        "crit_5%": res[4]['5%'],
        "crit_10%": res[4]['10%'],
        "is_stationary": bool(res[1] < 0.05)
    }

def save_kv_csv(path: str, kv: Dict[str, Any]):
    rows = pd.DataFrame([(k, v) for k, v in kv.items()], columns=["metric", "value"])
    rows.to_csv(path, index=False, sep=CSV_SEP)

def plot_series(y: pd.Series, title: str, filename: str, in_plots_dir: bool = False):
    plt.figure(figsize=(10, 4))
    y.plot()
    plt.title(title)
    plt.tight_layout()
    full_path = os.path.join(PLOTS_DIR if in_plots_dir else OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=150)
    plt.close()
    return full_path

def plot_stem(values, title, filename):
    plt.figure(figsize=(10, 4))
    plt.stem(range(len(values)), values)
    plt.title(title)
    plt.tight_layout()
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=150)
    plt.close()
    return full_path

@dataclass
class TransformInfo:
    log_applied: bool = False
    diff_d: int = 0
    seasonal_D: int = 0

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 0) Створимо шаблон, якщо даних немає
    ensure_template_csv(DATA_CSV_PATH)

    # 1) Завантаження і підготовка
    df = pd.read_csv(DATA_CSV_PATH, sep=CSV_SEP)
    if DATE_COL not in df.columns or VALUE_COL not in df.columns:
        raise ValueError(f"CSV має містити '{DATE_COL}' і '{VALUE_COL}'. Поточні: {list(df.columns)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    y = df.set_index(DATE_COL)[VALUE_COL]
    y = y.asfreq(FREQ)
    y = y.interpolate(limit_direction='both')

    plot_series(y, "Початковий ряд", "01_raw_series.png", in_plots_dir=True)

    # 2) Перевірка стаціонарності + можливе логарифмування
    adf_raw = adf_test(y)

    tinfo = TransformInfo()
    y_work = y.copy()

    adf_log_info = {"note": "логарифмування пропущено"}
    if (y_work > 0).all():
        y_log = np.log(y_work)
        adf_log_info = adf_test(y_log)
        # якщо лог дає кращий p-value — беремо лог
        if adf_log_info["p_value"] < adf_raw["p_value"]:
            y_work = y_log
            tinfo.log_applied = True
            plot_series(y_work, "Логарифмований ряд", "02_log_series.png", in_plots_dir=True)

    # 3) Різницювання d (0..2) — вибираємо краще за ADF p-value
    best_d = 0
    best_p = adf_test(y_work)["p_value"]
    y_d = y_work.copy()
    for d in [1, 2]:
        y_tmp = y_work.diff(d).dropna()
        pval = adf_test(y_tmp)["p_value"]
        if pval < best_p:
            best_p = pval
            best_d = d
            y_d = y_tmp
    tinfo.diff_d = best_d
    if best_d > 0:
        plot_series(y_d, f"Після різницювання d={best_d}", "03_diff_d.png", in_plots_dir=True)

    # 4) Сезонне різницювання D (за потреби)
    best_D = 0
    best_p2 = adf_test(y_d)["p_value"]
    y_ds = y_d.copy()
    for D in [1]:  # класичний підхід для місячних даних
        y_tmp = y_d.diff(SEASONAL_PERIODS * D).dropna()
        pval = adf_test(y_tmp)["p_value"]
        if pval < best_p2:
            best_p2 = pval
            best_D = D
            y_ds = y_tmp
    tinfo.seasonal_D = best_D

    plot_series(y_ds, f"Стаціонаризований ряд (d={tinfo.diff_d}, D={tinfo.seasonal_D})", "04_stationary_series.png", in_plots_dir=True)

    # ACF/PACF стаціонарного ряду (для ідентифікації)
    acf_vals = acf(y_ds.dropna(), nlags=min(36, max(10, len(y_ds.dropna())-1)))
    pacf_vals = pacf(y_ds.dropna(), nlags=min(36, max(10, len(y_ds.dropna())-2)), method='ywm')
    plot_stem(acf_vals, "ACF стаціонаризованого ряду", "05_stationary_acf.png")
    plot_stem(pacf_vals, "PACF стаціонаризованого ряду", "05_stationary_pacf.png")

    # збережемо кроки стаціонаризації
    save_kv_csv(os.path.join(OUTPUT_DIR, "stationarity_steps.csv"), {
        "log_applied": tinfo.log_applied,
        "d": tinfo.diff_d,
        "D": tinfo.seasonal_D,
        "seasonal_periods": SEASONAL_PERIODS,
        "adf_p_raw": adf_raw["p_value"],
        "adf_p_after": adf_test(y_ds)["p_value"]
    })

    # 5) Ґрід-пошук SARIMA за AIC
    best_model = None
    best_cfg = None
    best_aic = np.inf

    y_input = np.log(y) if tinfo.log_applied and (y > 0).all() else y

    for p in p_RANGE:
        for d in d_RANGE:
            for q in q_RANGE:
                for P in P_RANGE:
                    for D in D_CANDIDATES:
                        for Q in Q_RANGE:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, SEASONAL_PERIODS)
                            try:
                                model = SARIMAX(
                                    y_input,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                aic = model.aic
                                if aic < best_aic:
                                    best_aic = aic
                                    best_model = model
                                    best_cfg = (order, seasonal_order)
                            except Exception:
                                continue

    if best_model is None:
        raise RuntimeError("Не вдалося підібрати SARIMA у заданій решітці. Розширте діапазони параметрів.")

    order, seasonal_order = best_cfg

    # 6) Параметри моделі
    params = best_model.params.to_dict()
    pd.DataFrame([(k, v) for k, v in params.items()], columns=["param", "estimate"]) \
      .to_csv(os.path.join(OUTPUT_DIR, "model_params.csv"), index=False, sep=CSV_SEP)

    # 7) Діагностика залишків
    resid = best_model.resid.dropna()

    # 06_residuals.png
    plot_series(resid, "Залишки моделі", "06_residuals.png")

    # 07_resid_acf.png та 07_resid_pacf.png
    resid_acf = acf(resid, nlags=min(36, max(10, len(resid)-1)))
    resid_pacf = pacf(resid, nlags=min(36, max(10, len(resid)-2)), method='ywm')
    plot_stem(resid_acf, "ACF залишків", "07_resid_acf.png")
    plot_stem(resid_pacf, "PACF залишків", "07_resid_pacf.png")

    lb = acorr_ljungbox(resid, lags=[min(24, max(10, len(resid)//5))], return_df=True)
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)

    diagnostics = {
        "best_order": str(order),
        "best_seasonal_order": str(seasonal_order),
        "best_aic": best_aic,
        "sample_size": int(best_model.nobs),
        "ljung_box_stat": float(lb['lb_stat'].iloc[0]),
        "ljung_box_pvalue": float(lb['lb_pvalue'].iloc[0]),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "resid_skew": float(skew),
        "resid_kurtosis": float(kurt),
    }
    save_kv_csv(os.path.join(OUTPUT_DIR, "diagnostics.csv"), diagnostics)

    # 8) Прогноз на 10 кроків
    fc_res = best_model.get_forecast(steps=FORECAST_STEPS)
    fc_mean = fc_res.predicted_mean
    fc_ci = fc_res.conf_int(alpha=CONF_ALPHA)

    fc = pd.DataFrame({
        DATE_COL: pd.date_range(y.index[-1] + pd.tseries.frequencies.to_offset(FREQ),
                                periods=FORECAST_STEPS, freq=FREQ),
        "forecast": fc_mean.values,
        "lower": fc_ci.iloc[:, 0].values,
        "upper": fc_ci.iloc[:, 1].values,
    })

    # якщо була лог-трансформація — повертаємося до початкової шкали
    if tinfo.log_applied:
        for col in ["forecast", "lower", "upper"]:
            fc[col] = np.exp(fc[col])

    fc.to_csv(os.path.join(OUTPUT_DIR, "forecast_10.csv"), index=False, sep=CSV_SEP)

    # 9) Графік прогнозу
    plt.figure(figsize=(10, 4))
    plt.plot(y.index, y.values, label="Observed")
    plt.plot(fc[DATE_COL], fc["forecast"], label="Forecast")
    plt.fill_between(fc[DATE_COL], fc["lower"], fc["upper"], alpha=0.2, label="95% CI")
    plt.title("Прогноз на 10 періодів")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_forecast.png"), dpi=150)
    plt.close()

    # 10) Декомпозиція (для звіту; може не спрацювати при коротких рядах)
    try:
        dec = seasonal_decompose(y, model='additive', period=SEASONAL_PERIODS, extrapolate_trend='freq')
        fig = dec.plot()
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "09_decompose.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # 11) README із коротким резюме
    with open(os.path.join(OUTPUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "РЕЗЮМЕ РОБОТИ\n"
            "--------------\n"
            f"1) Спостережень: {len(y)}, частота: {FREQ}, s={SEASONAL_PERIODS}.\n"
            f"2) Стаціонаризація: log={tinfo.log_applied}, d={tinfo.diff_d}, D={tinfo.seasonal_D}.\n"
            f"3) Найкраща модель: SARIMA{order}x{seasonal_order} (AIC={best_aic:.2f}).\n"
            f"4) Ljung–Box p-value={float(lb['lb_pvalue'].iloc[0]):.4f}; "
            f"Jarque–Bera p-value={float(jb_p):.4f}.\n"
            f"5) Прогноз (10 кроків) у outputs/forecast_10.csv; усі CSV з розділювачем '~'.\n"
        )

    print("Готово. Перевірте директорію ./outputs для таблиць і графіків.")

if __name__ == "__main__":
    main()
