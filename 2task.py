# -*- coding: utf-8 -*-
# Демонстраційний ноутбук: Аналіз економічної ефективності сонячних панелей (без CSV)
# Покриває пункти: 1) структура, 2) модель, 3) прогноз, 4) причинність, 5) згладжування/фільтрація

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

np.random.seed(42)

# -----------------------------
# 0) Генерація синтетичних даних
# -----------------------------
# Місячна шкала: 2019-01 .. 2024-12 (тренування), прогноз на 12 міс (2025-01 .. 2025-12)
idx = pd.date_range('2019-01-01', '2024-12-01', freq='MS')
horizon = 12
future_idx = pd.date_range(idx[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')

# Параметри "станції"
capacity_kW = 30.0                      # встановлена потужність
base_kwh_per_month = capacity_kW * 30 * 4  # умовно 4 год еквівалентного сонця на добу
degradation_yearly = 0.005              # 0.5% на рік
degradation_monthly = (1 - degradation_yearly) ** (1/12)

# 1) Сезонність інсоляції (індекс 0.6..1.4 з шумом)
m = np.arange(len(idx))
season = 1.0 + 0.35 * np.sin(2*np.pi*(m)/12)  # максимум влітку
season += 0.05 * np.random.normal(size=len(idx))  # погодний шум
season = np.clip(season, 0.6, 1.5)

# 2) Деградація панелей
deg = degradation_monthly ** m  # повільне зниження продуктивності

# 3) Тариф (UAH/kWh): тренд + сходинка (наприклад, індексація у 2022-07)
tariff = 2.5 + 0.02 * (m/12)  # слабкий тренд угору
tariff += (m >= 42) * 0.3     # сходинка з липня 2022 (~42-й індекс)
tariff += 0.05 * np.random.normal(size=len(idx))
tariff = np.clip(tariff, 2.0, None)

# 4) Виробіток (kWh): базовий рівень * сезон * деградація * шум
prod_kwh = base_kwh_per_month * season * deg * (1 + 0.03*np.random.normal(size=len(idx)))
prod_kwh = np.clip(prod_kwh, 0, None)

# 5) Дохід і витрати
revenue = prod_kwh * tariff
fixed_monthly_cost = 800.0
maint_ratio = 0.08  # 8% від доходу в середньому
maint_cost = maint_ratio * revenue * (1 + 0.1*np.random.normal(size=len(idx)))
maint_cost = np.clip(maint_cost, 0, None)

# Річний великий платіж (наприклад, сервіс/страхування) у березні
annual_spike = np.zeros(len(idx))
annual_spike[(pd.DatetimeIndex(idx).month == 3)] = 4000.0

costs = fixed_monthly_cost + maint_cost + annual_spike
net_cashflow = revenue - costs

# Кладемо все в DataFrame
df = pd.DataFrame({
    'revenue': revenue,
    'costs': costs,
    'net_cashflow': net_cashflow,
    'prod_kwh': prod_kwh,
    'tariff': tariff,
    'insolation_idx': season,
    'degradation_idx': deg
}, index=idx)

# -----------------------------
# 1) Дослідження структури ряду
# -----------------------------
series = df['revenue']

# STL-декомпозиція
stl = STL(series, period=12, robust=True).fit()
trend = stl.trend
seasonal = stl.seasonal
resid = stl.resid

# Перевірка на стаціонарність (ADF)
adf_stat, adf_p, _, _, _, _ = adfuller(series)

print("ADF p-value (revenue):", round(adf_p, 4))

# -----------------------------
# 2) Побудова моделі: SARIMAX(endog=revenue, exog=[tariff, insolation_idx, degradation_idx])
# -----------------------------
exog = df[['tariff', 'insolation_idx', 'degradation_idx']]

# Спліт: тренування до кінця 2024-12
train_end = df.index.max()
endog_train = series
exog_train = exog

# Готуємо майбутні екзогенні фічі (2025-01..2025-12)
m_fut = np.arange(len(future_idx)) + len(idx)
season_fut = 1.0 + 0.35 * np.sin(2*np.pi*(m_fut)/12)
season_fut += 0.05 * np.random.normal(size=len(future_idx))
season_fut = np.clip(season_fut, 0.6, 1.5)

deg_fut = degradation_monthly ** m_fut

# Тариф у майбутньому: плавний ріст + шум
tariff_fut = 2.5 + 0.02 * (m_fut/12)
tariff_fut += (m_fut >= 42) * 0.3
tariff_fut += 0.05 * np.random.normal(size=len(future_idx))
tariff_fut = np.clip(tariff_fut, 2.0, None)

exog_future = pd.DataFrame({
    'tariff': tariff_fut,
    'insolation_idx': season_fut,
    'degradation_idx': deg_fut
}, index=future_idx)

# Навчання SARIMAX (базові порядки; можна тюнити)
model = SARIMAX(endog_train,
                exog=exog_train,
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False)
res = model.fit(disp=False)

print(res.summary())

# -----------------------------
# 3) Прогноз на 12 місяців
# -----------------------------
fc = res.get_forecast(steps=horizon, exog=exog_future)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.2)  # 80% ДІ

# -----------------------------
# 4) Granger causality: (tariff, insolation) -> revenue
#    Виконуємо на диференційованих рядах (приблизна стаціонаризація)
# -----------------------------
df_gc = df[['revenue', 'tariff', 'insolation_idx']].copy().dropna()
df_gc_diff = df_gc.diff().dropna()

print("\nGranger test: tariff causes revenue (maxlag=6)")
_ = grangercausalitytests(df_gc_diff[['revenue', 'tariff']], maxlag=6, verbose=True)

print("\nGranger test: insolation causes revenue (maxlag=6)")
_ = grangercausalitytests(df_gc_diff[['revenue', 'insolation_idx']], maxlag=6, verbose=True)

# -----------------------------
# 5) Згладжування/фільтрація
# -----------------------------
# Ковзна середня (12 міс)
ma12 = series.rolling(window=12, min_periods=1, center=True).mean()

# HP-фільтр (виділення тренду)
cycle_hp, trend_hp = hpfilter(series, lamb=129600)  # рекомендовано для місячних рядів

# Holt-Winters як альтернативне згладжування/модель сезонності
hw_model = ExponentialSmoothing(series, trend='add', seasonal='mul', seasonal_periods=12).fit()
hw_fc = hw_model.forecast(horizon)

# -----------------------------
# 6) (Опціонально) Простий економічний показник: LCOE та NPV
# -----------------------------
# Припустимо CAPEX = 550,000 UAH на старті; ставка дисконту r = 15% річних
capex = 550_000.0
r_annual = 0.15
r_monthly = (1 + r_annual) ** (1/12) - 1

# Фактичний LCOE за період: (CAPEX + сума витрат) / (сума виробітку)
lcoe_realized = (capex + df['costs'].sum()) / df['prod_kwh'].sum()

# NPV за історичний період:
cashflows_hist = -capex * np.ones(len(df))
cashflows_hist[0] -= 0.0  # CAPEX у першому місяці (вже врахували)
cashflows_hist += df['net_cashflow'].values
discount_factors = 1 / (1 + r_monthly) ** np.arange(len(df))
npv_hist = np.sum(cashflows_hist * discount_factors)

print(f"\nRealized LCOE (UAH/kWh) за 2019-01..2024-12: {lcoe_realized:.2f}")
print(f"NPV історичного періоду (UAH): {npv_hist:,.0f}")

# -----------------------------
# 7) Візуалізації (за бажанням)
# -----------------------------
plt.figure(figsize=(11,4))
plt.plot(series, label='Revenue')
plt.plot(trend, label='STL trend', linewidth=2)
plt.title('Дохід та тренд (STL)')
plt.legend(); plt.show()

plt.figure(figsize=(11,4))
plt.plot(series, label='Revenue')
plt.plot(ma12, label='MA(12)', linewidth=2)
plt.title('Згладжування ковзною середньою')
plt.legend(); plt.show()

plt.figure(figsize=(11,4))
plt.plot(series, label='Revenue')
plt.plot(trend_hp, label='HP-тренд', linewidth=2)
plt.title('HP-фільтр')
plt.legend(); plt.show()

# Прогноз SARIMAX
plt.figure(figsize=(11,4))
plt.plot(series, label='Revenue (train)')
plt.plot(fc_mean, label='Forecast (SARIMAX)')
plt.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, label='80% CI')
plt.title('Прогноз доходу на 12 міс (SARIMAX + exog)')
plt.legend(); plt.show()

# Порівняння Holt-Winters forecast
plt.figure(figsize=(11,4))
plt.plot(series, label='Revenue (train)')
plt.plot(pd.Series(hw_fc, index=future_idx), label='Forecast (Holt-Winters)')
plt.title('Прогноз доходу: Holt-Winters')
plt.legend(); plt.show()
