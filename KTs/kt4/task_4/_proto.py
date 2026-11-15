# -*- coding: utf-8 -*-
"""Прогон протокола КТ4 задание 4 на синтетике, чтобы получить реалистичные числа."""
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
rng = np.random.RandomState(SEED)

N, D = 80000, 12
latent_scale = np.array([3.0, 2.2, 1.6, 1.1, .8, .6, .5, .4, .3, .2, .15, .1])
Z = rng.randn(N, D) * latent_scale
w = np.stack([rng.randn(D) for _ in range(D)])
w, _ = np.linalg.qr(w)
X = Z @ w.T
sig = 1.15 * (Z[:, 0] + 0.6 * Z[:, 1] + 0.35 * Z[:, 2]) - 2.0
p = 1 / (1 + np.exp(-sig))
y = (rng.rand(N) < p).astype(int)
print("pos rate", round(y.mean(), 4))

idx_all = np.arange(N)
Xtr, Xte = train_test_split(idx_all, test_size=.25, stratify=y, random_state=SEED)

Xs = StandardScaler().fit_transform(X)
pca_full = PCA(random_state=SEED).fit(Xs[Xtr])
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
print("evr", np.round(evr, 4))
print("cum", np.round(cum[:12], 4))

ks = list(range(1, 13))
rows = []
for k in ks:
    Xt = Xs[Xtr, :k]; Xe = Xs[Xte, :k]
    t0 = time.time(); lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xt, y[Xtr]); fit_lr = time.time()-t0
    t0 = time.time(); pe = lr.predict_proba(Xe)[:, 1]; sc_lr = time.time()-t0
    t0 = time.time(); rf = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Xt, y[Xtr]); fit_rf = time.time()-t0
    t0 = time.time(); pf = rf.predict_proba(Xe)[:, 1]; sc_rf = time.time()-t0
    rows.append(dict(k=k,
                     auc_lr=roc_auc_score(y[Xte], pe), pr_lr=average_precision_score(y[Xte], pe),
                     auc_rf=roc_auc_score(y[Xte], pf), pr_rf=average_precision_score(y[Xte], pf),
                     sec_lr=sc_lr*len(Xe)/1000, sec_rf=sc_rf*len(Xe)/1000,
                     fit_lr=fit_lr, fit_rf=fit_rf))
    print({kk: round(v, 4) for kk, v in rows[-1].items()})
df = pd.DataFrame(rows)
print(df.round(4).to_string())
