import numpy as np
import matplotlib.pyplot as plt

# 1) Згенеруємо синтетичні дані (24 години)
hours = np.arange(0, 24)

# Простий профіль ірадіації: 0 вночі, пік близько полудня
irr_Wm2 = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * 1000  # Вт/м²

# 2) Параметри "умовної" установки
area_m2 = 20       # площа масиву панелей, м²
eff_panel = 0.18   # к.к.д. панелей (18%)
eff_inverter = 0.96

# 3) Розрахунок потужності
dc_power_kw = (irr_Wm2 / 1000) * area_m2 * eff_panel          # кВт (DC)
ac_power_kw = dc_power_kw * eff_inverter                       # кВт (AC)

# 4) Побудова графіку (лише matplotlib, без seaborn/стилів/кольорів)
plt.figure(figsize=(7, 4))
plt.plot(hours, ac_power_kw, marker="o")  # колір не задаємо
plt.title("Добовий профіль AC-потужності сонячної установки (синтетика)")
plt.xlabel("Година доби")
plt.ylabel("Потужність, кВт")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# За бажанням можна зберегти файл:
# plt.savefig("solar_daily_profile.png", dpi=150)

plt.show()
