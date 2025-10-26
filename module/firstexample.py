import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


np.random.seed(42)
os.makedirs("figs", exist_ok=True)


def savefig(fig, name):
    path = os.path.join("figs", name); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return path


# 1) Дані
X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                           n_redundant=4, n_repeated=0, flip_y=0.05,
                           class_sep=1.2, random_state=42)
feat = np.array([f"f{i}" for i in range(X.shape[1])])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# 2) Базова модель для wrapper/embedded
lr_l2 = LogisticRegression(penalty="l2", solver="liblinear", random_state=42, max_iter=2000)
lr_l1 = LogisticRegression(penalty="l1", solver="liblinear", random_state=42, max_iter=2000)
rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)


results = []


# VarianceThreshold
vt = VarianceThreshold(threshold=0.0).fit(Xtr)
mask_vt = vt.get_support()
lr_l2.fit(Xtr[:,mask_vt], ytr)
acc_vt = accuracy_score(yte, lr_l2.predict(Xte[:,mask_vt]))
results.append(("VarianceThreshold", mask_vt.sum(), acc_vt))


# SelectKBest: chi2 (масштабування в [0,1])
K = 10
sc01 = MinMaxScaler().fit(Xtr)
Xtr01, Xte01 = sc01.transform(Xtr), sc01.transform(Xte)
skb_chi2 = SelectKBest(chi2, k=K).fit(Xtr01, ytr)
mask_chi2 = skb_chi2.get_support()
lr_l2.fit(Xtr01[:,mask_chi2], ytr)
acc_chi2 = accuracy_score(yte, lr_l2.predict(Xte01[:,mask_chi2]))
results.append(("SelectKBest(chi2)", mask_chi2.sum(), acc_chi2))


# SelectKBest: ANOVA F
skb_f = SelectKBest(f_classif, k=K).fit(Xtr, ytr)
mask_f = skb_f.get_support()
lr_l2.fit(Xtr[:,mask_f], ytr)
acc_f = accuracy_score(yte, lr_l2.predict(Xte[:,mask_f]))
results.append(("SelectKBest(F)", mask_f.sum(), acc_f))


# SelectKBest: mutual_info
skb_mi = SelectKBest(mutual_info_classif, k=K).fit(Xtr, ytr)
mask_mi = skb_mi.get_support()
lr_l2.fit(Xtr[:,mask_mi], ytr)
acc_mi = accuracy_score(yte, lr_l2.predict(Xte[:,mask_mi]))
results.append(("SelectKBest(mutual_info)", mask_mi.sum(), acc_mi))


# RFE з лог. регресією
rfe = RFE(lr_l2, n_features_to_select=K, step=1).fit(Xtr, ytr)
mask_rfe = rfe.get_support()
lr_l2.fit(Xtr[:,mask_rfe], ytr)
acc_rfe = accuracy_score(yte, lr_l2.predict(Xte[:,mask_rfe]))
results.append(("RFE(LogReg)", mask_rfe.sum(), acc_rfe))


# SFS forward/backward
sfs_fwd = SequentialFeatureSelector(lr_l2, n_features_to_select=K, direction="forward", n_jobs=-1).fit(Xtr, ytr)
mask_sfs = sfs_fwd.get_support()
lr_l2.fit(Xtr[:,mask_sfs], ytr)
acc_sfs = accuracy_score(yte, lr_l2.predict(Xte[:,mask_sfs]))
results.append(("SFS-forward(LogReg)", mask_sfs.sum(), acc_sfs))


sfs_bwd = SequentialFeatureSelector(lr_l2, n_features_to_select=K, direction="backward", n_jobs=-1).fit(Xtr, ytr)
mask_sbs = sfs_bwd.get_support()
lr_l2.fit(Xtr[:,mask_sbs], ytr)
acc_sbs = accuracy_score(yte, lr_l2.predict(Xte[:,mask_sbs]))
results.append(("SBS-backward(LogReg)", mask_sbs.sum(), acc_sbs))


# Embedded L1 / ElasticNet (через LogisticRegression)
lr_l1.fit(Xtr, ytr)
mask_l1 = lr_l1.coef_.ravel()!=0
lr_l2.fit(Xtr[:,mask_l1], ytr)
acc_l1 = accuracy_score(yte, lr_l2.predict(Xte[:,mask_l1]))
results.append(("L1-LogReg (embedded)", mask_l1.sum(), acc_l1))


lr_en = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, random_state=42, max_iter=3000)
lr_en.fit(Xtr, ytr)
mask_en = lr_en.coef_.ravel()!=0
lr_l2.fit(Xtr[:,mask_en], ytr)
acc_en = accuracy_score(yte, lr_l2.predict(Xte[:,mask_en]))
results.append(("ElasticNet-LogReg", mask_en.sum(), acc_en))


# RF importance (top-K)
rf.fit(Xtr, ytr)
imp = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
topk = imp.index[:K]
mask_rf = np.isin(feat, topk)
lr_l2.fit(Xtr[:,mask_rf], ytr)
acc_rf = accuracy_score(yte, lr_l2.predict(Xte[:,mask_rf]))
results.append(("RF top-K importance", mask_rf.sum(), acc_rf))


# Звіт
df = pd.DataFrame(results, columns=["Method","Selected","TestAcc"]).sort_values("TestAcc", ascending=False)
print(df.to_string(index=False))


# Графік важливостей RF (Top-10)
fig = plt.figure(figsize=(6,3))
imp.head(10).plot(kind="bar")
plt.title("Top-10 feature importances (RandomForest)")
plt.ylabel("importance"); plt.tight_layout()
fname = savefig(fig, "feature_importances_rf.png")
print("[FIG: feature_importances_rf.png]")
