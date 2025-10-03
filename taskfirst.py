#1
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier




num_cols = ['capex','opex_year','tariff','insolation','annual_gen_kwh','degradation']
X = df[num_cols]
y = df['payback_class']


pipe = Pipeline([
    ('model', RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced'))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro')
print('CV F1-macro:', scores.mean().round(3))


pipe.fit(X, y)
y_pred = pipe.predict(X)
print(classification_report(y, y_pred))

#2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Фейкові дані для демонстрації
rng = np.random.RandomState(42)
n = 400
df = pd.DataFrame({
    'insolation': np.r_[rng.normal(4.5, 0.3, n//2), rng.normal(5.5, 0.4, n//2)],
    'lcoe':       np.r_[rng.normal(60, 8, n//2),     rng.normal(75, 7, n//2)],
    'degradation':np.r_[rng.normal(0.5, 0.05, n//2), rng.normal(0.7, 0.06, n//2)],
    'capex':      np.r_[rng.normal(750, 80, n//2),   rng.normal(900, 90, n//2)],
    'opex_year':  np.r_[rng.normal(15, 3, n//2),     rng.normal(20, 4, n//2)]
})


# Масштабування
X = StandardScaler().fit_transform(df)


# Обчислюємо silhouette score для k=2..7
scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42, algorithm="elkan", max_iter=200)
    labels = km.fit_predict(X)
    s = silhouette_score(X, labels)
    scores.append((k, s))


scores_df = pd.DataFrame(scores, columns=["k", "silhouette"])
print(scores_df)


# Побудова графіка
plt.figure(figsize=(7,5))
plt.plot(scores_df["k"], scores_df["silhouette"], marker="o")
plt.title("Аналіз економічної ефективності сонячних панелей\nSilhouette score by k")
plt.xlabel("Кількість кластерів (k)")
plt.ylabel("Silhouette score")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()


#3


import numpy as np


# Річна економія ≈ вироблена_енергія * тариф – opex_year
df['annual_saving'] = df['annual_gen_kwh'] * df['tariff'] - df['opex_year']
df['payback_years'] = df['capex'] / df['annual_saving'].clip(lower=1e-6)

bins = [0, 5, 8, np.inf]
labels = ['fast','mid','slow']
df['payback_class'] = pd.cut(df['payback_years'], bins=bins, labels=labels, right=True)
