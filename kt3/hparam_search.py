"""
КТ3, Задание 1: оценка и анализ гиперпараметров алгоритмов классификации.

Модели: LogisticRegression (baseline), GaussianNB (байесовский),
        HistGradientBoostingClassifier.

Протокол (согласно выводам анализа источников):
  - цель: SeverityBin = (Severity >= 3);
  - сплит: GroupKFold(5) по group_id (без утечки через повторы);
  - метрики: PR-AUC (основная; базовая ставка ~0.195 на полном файле),
             ROC-AUC, Brier, F1 при пороге 0.5;
  - поиск: RandomizedSearchCV c фиксированным seed -> воспроизводимо;
  - лучшие параметры сохраняются в kt3/out/best_params.json для
    последующих этапов эксперимента.

Использование:
    python kt3/hparam_search.py [--data kt3/df_with_groups.csv.gz]
                                [--rows N] [--subsample 500000]
                                [--n-iter 20] [--cv 5] [--out-dir kt3/out]
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, make_scorer, roc_auc_score)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

NUM_COLS = ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Pressure(in)",
            "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
BIN_COLS = ["Bump", "Crossing", "Junction", "Railway", "Station", "Stop",
            "Traffic_Signal", "Amenity"]


# ------------------------------------------------------------------ features
def build_features(df: pd.DataFrame):
    X = df[NUM_COLS + BIN_COLS].copy()
    st = pd.to_datetime(df["Start_Time"], errors="coerce")
    X["hour"] = st.dt.hour
    X["dow"] = st.dt.dayofweek
    X["month"] = st.dt.month
    X["sin_h"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["cos_h"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["lat"] = df["Start_Lat"]
    X["lng"] = df["Start_Lng"]
    y = (pd.to_numeric(df["Severity"], errors="coerce") >= 3).astype(int).values
    return X.astype(float), y


# ---------------------------------------------------------------- search spaces
SEARCH_SPACES = {
    "LogReg": {
        "pipe": lambda: Pipeline([("sc", StandardScaler()),
                                  ("m", LogisticRegression(max_iter=3000,
                                                            random_state=0))]),
        "grid": {
            "m__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            "m__penalty": ["l1", "l2"],
            "m__solver": ["liblinear", "saga"],
            "m__class_weight": [None, "balanced"],
        },
        "n_iter": 24,
    },
    "GaussianNB": {
        "pipe": lambda: Pipeline([("sc", StandardScaler()), ("m", GaussianNB())]),
        "grid": {"m__var_smoothing": np.logspace(-9, 0, 10)},
        "n_iter": 10,
    },
    "HistGB": {
        "pipe": lambda: HistGradientBoostingClassifier(random_state=0),
        "grid": {
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [20, 50, 100],
            "max_depth": [0, 8, 12, None],          # 0 => без ограничения
            "l2_regularization": [0.0, 0.1, 1.0],
            "max_iter": [200],
            "early_stopping": [True],
            "validation_fraction": [0.1],
        },
        "n_iter": 40,
    },
}


def scorers():
    return {
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        # sklearn >= 1.4: needs_proba и response_method mutually exclusive
        "brier": make_scorer(brier_score_loss, response_method="predict_proba"),
        "f1": make_scorer(f1_score),
    }


def param_col(res: pd.DataFrame, p: str) -> pd.Series:
    """Достать значения гиперпараметра p из cv_results_ (совместимость версий)."""
    if p in res.columns:
        return res[p]
    return res["params"].apply(lambda d: d.get(p))


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="kt3/df_with_groups.csv.gz")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=500_000)
    ap.add_argument("--n-iter-scale", type=float, default=1.0,
                    help="множитель числа итераций RandomizedSearch")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="kt3/out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False, nrows=args.rows)
    # Равномерная по времени подвыборка: сортируем по Start_Time и берём
    # каждый k-й ряд -> сохраняются все эпохи/источники (в отличие от
    # случайной выборки из начала файла, где нет 2022-2023 годов).
    df["_ts"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.sort_values("_ts", kind="stable").drop(columns="_ts").reset_index(drop=True)
    if len(df) > args.subsample:
        step = int(np.ceil(len(df) / args.subsample))
        df = df.iloc[::step].reset_index(drop=True)
    print(f"Using {len(df):,} rows (even time-stride subsample)")

    X, y = build_features(df)
    keep = X.notna().all(axis=1)
    X, y = X[keep].reset_index(drop=True), y[keep.values]
    df = df[keep].reset_index(drop=True)
    groups = df["group_id"].values
    base_rate = float(y.mean())
    print(f"Pos rate P(Severity>=3) = {base_rate:.4f}  (PR-AUC baseline)")

    cv = GroupKFold(n_splits=args.cv)
    results, best_params = [], {}

    for name, spec in SEARCH_SPACES.items():
        n_iter = max(4, int(spec["n_iter"] * args.n_iter_scale))
        est = spec["pipe"]()
        gs = RandomizedSearchCV(
            estimator=est, param_distributions=spec["grid"], n_iter=n_iter,
            scoring=scorers(), refit="pr_auc", cv=cv,
            random_state=args.seed, n_jobs=args.jobs,
            return_train_score=True, verbose=0,
        )
        t0 = time.time()
        gs.fit(X, y, groups=groups)
        dt = time.time() - t0
        res = pd.DataFrame(gs.cv_results_)

        # ---- top-5 конфигураций по PR-AUC (mean/std по фолдам)
        cols = [c for c in res.columns
                if c.startswith("params") or "mean_test" in c or "std_test_pr_auc" in c]
        top = res.sort_values("rank_test_pr_auc").head(5)
        print("\n" + "=" * 70)
        print(f"{name}: RandomizedSearchCV ({n_iter} configs x {args.cv} folds, {dt:.1f}s)")
        show = []
        for _, r in top.iterrows():
            show.append({
                "params": dict(r["params"]),
                "test_PR_AUC": round(r["mean_test_pr_auc"], 4),
                "std_PR_AUC": round(r["std_test_pr_auc"], 4),
                "test_ROC_AUC": round(r["mean_test_roc_auc"], 4),
                "test_Brier": round(r["mean_test_brier"], 4),
                "train_PR_AUC": round(r["mean_train_pr_auc"], 4),
            })
        for s in show:
            print(json.dumps(s, ensure_ascii=False, default=str))

        # ---- вклад гиперпараметров: PR-AUC сгруппирована по каждому параметру
        print("\n   Вклад гиперпараметров (mean test PR-AUC по значениям):")
        for p in spec["grid"]:
            key = p.replace("m__", "") if p.startswith("m__") else p
            vals = param_col(res, p).apply(lambda v: str(v))
            g = res.groupby(vals)["mean_test_pr_auc"].agg(["mean", "max", "count"])
            if len(g) > 1:
                order = g["mean"].sort_values(ascending=False)
                line = ", ".join(f"{k}:{v:.4f}" for k, v in order.items())
                print(f"     {key:<22} {line}")

        # ---- diagnostics: train vs test gap (overfitting check)
        tr, te = res["mean_train_pr_auc"], res["mean_test_pr_auc"]
        print(f"\n   overfit gap (train-test PR-AUC): mean={np.mean(tr - te):.4f}, "
              f"best config gap={tr[gs.best_index_] - te[gs.best_index_]:.4f}")

        best_params[name] = gs.best_params_
        results.append({"model": name,
                        "best_PR_AUC": round(gs.best_score_, 4),
                        "best_ROC_AUC": round(gs.best_estimator_ and
                                              res.loc[gs.best_index_, "mean_test_roc_auc"], 4),
                        "best_Brier": round(res.loc[gs.best_index_, "mean_test_brier"], 4),
                        "best_F1": round(res.loc[gs.best_index_, "mean_test_f1"], 4),
                        "n_configs": n_iter,
                        "seconds": round(dt, 1)})

    summary = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Summary (CV GroupKFold, refit=PR-AUC):")
    print(summary.to_string(index=False))

    out_json = os.path.join(args.out_dir, "best_params.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"protocol": "GroupKFold(5) by group_id, target=Severity>=3",
                   "subsample_rows": int(len(df)),
                   "base_rate": round(base_rate, 4),
                   "best_params": best_params,
                   "summary": results}, f, ensure_ascii=False, indent=2, default=str)
    summary.to_csv(os.path.join(args.out_dir, "hparam_summary.csv"), index=False)
    print(f"\nSaved: {out_json}")
    print("Используйте эти параметры на последующих этапах эксперимента "
          "(import json; json.load(open('kt3/out/best_params.json'))['best_params']).")
    return 0


if __name__ == "__main__":
    sys.exit(main())