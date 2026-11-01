"""
КТ3, Задание 3: уточненное значение доли объектов обучающей выборки (10%..50%),
обеспечивающее максимум каждой метрики качества из п.1.

Протокол данных - тот же, что в заданиях 1-2:
  цель Severity >= 3; разбиение по группам повторов (group_id);
  дедупликация "1 группа = 1 авария" (медиана Severity внутри группы).

Методика:
  - доля обучения меняется от 10% до 50% с шагом 2.5% (17 точек), остальное
    уходит на контроль (holdout);
  - для каждой доли - несколько независимых разбиений (разные seed), чтобы
    снять шум одного сплита;
  - для каждой пары (модель, метрика) по повторениям считается среднее, затем
    кривая сглаживается скользящим средним и по сглаженному находится argmax;
  - если максимум приходится на границу диапазона, это отдельно помечается
    (такая метрика внутри 10..50% не имеет внутреннего оптимума).

Использование:
    python kt3/train_share_search.py [--data kt3/dedup_short.csv.gz]
                                     [--repeats 5] [--seed 42] [--out-dir kt3/out]
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

NUM_COLS = ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Pressure(in)",
            "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
BIN_COLS = ["Bump", "Crossing", "Junction", "Railway",
            "Station", "Stop", "Traffic_Signal", "Amenity"]
METRICS = ["ROC-AUC", "PR-AUC", "F1", "Balanced Accuracy"]


def make_pipe(clf, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


# лучшие гиперпараметры из задания 1 (BEST_PARAMS в kt3_tasks.ipynb)
def build_models():
    return {
        "LogReg": make_pipe(LogisticRegression(C=0.01, max_iter=2000,
                                               random_state=42), scale=True),
        "GaussianNB": make_pipe(GaussianNB(var_smoothing=1e-3), scale=True),
        "DecisionTree": make_pipe(DecisionTreeClassifier(max_depth=None,
                                                         min_samples_leaf=50,
                                                         random_state=42)),
        "RandomForest": make_pipe(RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
        "XGBoost": make_pipe(XGBClassifier(learning_rate=0.3, max_depth=8,
                                           min_child_weight=10,
                                           tree_method="hist",
                                           eval_metric="logloss",
                                           random_state=42, n_jobs=-1)),
    }


def evaluate(y_true, proba, pred):
    out = {}
    try:
        out["ROC-AUC"] = roc_auc_score(y_true, proba)
    except ValueError:
        out["ROC-AUC"] = np.nan
    try:
        out["PR-AUC"] = average_precision_score(y_true, proba)
    except ValueError:
        out["PR-AUC"] = np.nan
    out["F1"] = f1_score(y_true, pred, zero_division=0)
    out["Balanced Accuracy"] = balanced_accuracy_score(y_true, pred)
    return out


def smooth(values, win=3):
    """Центральное скользящее среднее; на краях окно урезается до доступного."""
    v = np.asarray(values, dtype=float)
    out = np.empty_like(v)
    h = win // 2
    for i in range(len(v)):
        lo, hi = max(0, i - h), min(len(v), i + h + 1)
        out[i] = np.nanmean(v[lo:hi])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="kt3/dedup_short.csv.gz")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="kt3/out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    for c in BIN_COLS:
        df[c] = df[c].astype("boolean").astype(float)
    Xall = df[NUM_COLS + BIN_COLS + ["Start_Lat", "Start_Lng"]].values
    yall = df["y"].values
    gall = df["group_id"].values
    print(f"объекты (аварии после дедупликации): {len(df):,}, "
          f"P(sev>=3) = {yall.mean():.4f}")

    # доля обучения считается от ПОИСКА (search pool), как в протоколе задания 1:
    # 20% групп всегда зарезервировано под финальный контроль, обучение идёт на
    # frac * 80% объектов, метрика измеряется на оставшейся части пула.
    rng = np.random.RandomState(args.seed)
    uids = np.unique(gall)
    is_test = rng.permutation(len(uids)) < int(0.2 * len(uids))
    test_mask = is_test[np.searchsorted(uids, gall)]
    Xs, ys, gs = Xall[~test_mask], yall[~test_mask], gall[~test_mask]
    print(f"search pool: {len(Xs):,}, holdout(20% групп): {test_mask.sum():,}")

    # доли обучения 10%..50%, шаг 2.5% -> 17 точек
    train_fracs = [round(f, 3) for f in np.arange(0.10, 0.5001, 0.025)]

    models = build_models()
    rows = []
    t0 = time.time()
    for frac in train_fracs:
        n_splits = max(2, min(args.repeats, int(1.0 / frac)))
        gkf = GroupKFold(n_splits=n_splits)
        # фиксированный набор фолдов: одни и те же разбиения для всех моделей
        folds = list(gkf.split(Xs, ys, groups=gs))
        for name, mdl in models.items():
            for tr_idx, va_idx in folds:
                clf = clone(mdl)
                clf.fit(Xs[tr_idx], ys[tr_idx])
                proba = clf.predict_proba(Xs[va_idx])[:, 1]
                pred = clf.predict(Xs[va_idx])
                for metric, value in evaluate(ys[va_idx], proba, pred).items():
                    rows.append({"train_frac": frac, "model": name,
                                 "metric": metric, "value": value})
        print(f"frac={frac:.3f} (n_splits={n_splits}) "
              f"done ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "task3_train_share_raw.csv"),
               index=False)

    mean = (res.groupby(["model", "metric", "train_frac"])["value"]
               .mean().reset_index())
    std = (res.groupby(["model", "metric", "train_frac"])["value"]
              .std().rename("std").reset_index())
    mean = mean.merge(std, on=["model", "metric", "train_frac"])
    mean["smoothed"] = mean.groupby(["model", "metric"])["value"].transform(
        lambda s: smooth(s.values))
    mean.to_csv(os.path.join(args.out_dir, "task3_train_share_mean.csv"),
                index=False)

    best_rows = []
    for (name, metric), g in mean.groupby(["model", "metric"]):
        g = g.sort_values("train_frac")
        r = g.iloc[int(np.nanargmax(g["smoothed"].values))]
        best_rows.append({"model": name, "metric": metric,
                          "best_train_frac": float(r["train_frac"]),
                          "smoothed_value": round(float(r["smoothed"]), 4),
                          "raw_value": round(float(r["value"]), 4),
                          "std": round(float(r["std"]), 4),
                          "on_boundary": bool(r["train_frac"] in
                                              (train_fracs[0], train_fracs[-1]))})
    best = pd.DataFrame(best_rows)
    best.to_csv(os.path.join(args.out_dir, "task3_best_share.csv"), index=False)
    with open(os.path.join(args.out_dir, "task3_best_share.json"), "w",
              encoding="utf-8") as f:
        json.dump({"protocol": ("dedup by group_id, target Severity>=3, "
                                f"{args.repeats} repeats of stratified split"),
                   "train_fracs": train_fracs, "best": best_rows},
                  f, ensure_ascii=False, indent=2)

    print("\nСредние значения метрик (строки - доля обучения):")
    for metric in METRICS:
        piv = mean[mean["metric"] == metric].pivot(
            index="train_frac", columns="model", values="value")
        print(f"\n{metric}:")
        print(piv.round(4).to_string())

    print("\nЛучшая доля обучения по каждой метрике (по сглаженному среднему):")
    print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
