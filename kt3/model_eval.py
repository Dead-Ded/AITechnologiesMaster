"""
Шаги 2-4: GroupKFold с временным разбиением, Leave-One-Source-Out и калибровка.

Что проверяем (продолжение анализа источников):
  1. Утечка через повторы: сравниваем случайный Split vs GroupSplit (по
     group_id из source_analysis.py) vs TemporalSplit (train <= T0 < test).
     Ожидание: случайный split завышает метрики за счёт дублей в тесте.
  2. Leave-One-Source-Out (LOSO): обучаемся на всех источниках кроме held-out,
     предсказываем held-out -> количественная деградация PR-AUC = цена
     доменного сдвига конкретного источника.
  3. Калибровка вероятностей: reliability curves + Brier/ECE до и после
     CalibratedClassifierCV. Особенно важно для байесовских классификаторов,
     которые систематически выдают самоуверенные вероятности.

Модели: LogisticRegression (baseline), GaussianNB (байесовский), HistGB.

Использование:
    python model_eval.py [--data kt3/df_with_groups.csv.gz] [--rows N]
                         [--subsample 300000] [--out-dir kt3/out]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

NUM_COLS = ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Pressure(in)",
            "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
BIN_COLS = ["Bump", "Crossing", "Junction", "Railway", "Station", "Stop",
            "Traffic_Signal", "Amenity"]


# ------------------------------------------------------------------ features
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
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


def make_models(seed: int = 0) -> dict:
    return {
        "LogReg": Pipeline([("sc", StandardScaler()),
                            ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
        "GaussianNB": Pipeline([("sc", StandardScaler()),
                                ("m", GaussianNB())]),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
    }


# --------------------------------------------------------------- CV protocols
def cv_oof(X, y, groups, sources, year, kind: str, seed: int = 0):
    """Return dict name -> oof probability array for the given split protocol.

    random   : StratifiedKFold (обычный split; дубли одной аварии попадают в
               train и test одновременно -> утечка).
    group    : GroupKFold по group_id (утечки через повторы нет).
    temporal : train <= t0 < test по времени + запрет на группы из train
               (чистый out-of-time сценарий без дубликатной утечки).
    """
    n = len(y)
    models = make_models(seed)
    splits = []
    if kind == "random":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    elif kind == "group":
        splits = list(GroupKFold(5).split(X, y, groups=groups))
    elif kind == "temporal":
        t0 = np.quantile(year[~np.isnan(year)], 0.75)
        tr_idx = np.where((year <= t0) & ~np.isnan(year))[0]
        te_idx = np.where((year > t0) | np.isnan(year))[0]
        seen = set(groups[tr_idx].tolist())
        te_idx = np.array([i for i in te_idx if groups[i] not in seen])
        splits = [(tr_idx, te_idx)]
    else:
        raise ValueError(kind)

    out = {name: np.full(n, np.nan) for name in models}
    for tr, te in splits:
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        for name, mdl in models.items():
            mdl.fit(X.iloc[tr], y[tr])
            out[name][te] = mdl.predict_proba(X.iloc[te])[:, 1]
    return out


def metrics(y, p):
    return {"PR_AUC": average_precision_score(y, p),
            "ROC_AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p)}


# ----------------------------------------------------------------------- LOSO
def loso(X, y, sources, seed: int = 0):
    rows = []
    models = make_models(seed)
    for held in sorted(pd.unique(sources)):
        tr = sources != held
        te = ~tr
        if te.sum() < 50 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        for name, mdl in models.items():
            mdl.fit(X.loc[tr], y[tr])
            p = mdl.predict_proba(X.loc[te])[:, 1]
            m = metrics(y[te], p)
            base = float(y[te].mean())
            rows.append({"held_out_source": held, "model": name, "n_test": int(te.sum()),
                         "pos_rate": round(base, 4), **{k: round(v, 4) for k, v in m.items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration
def expected_calibration_error(y, p, bins=10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="uniform")
    bin_edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(mean_pred, bin_edges) - 1, 0, bins - 1)
    counts = np.histogram(p, bin_edges)[0]
    err = np.abs(frac_pos - mean_pred)
    return float(np.sum(counts[idx] * err) / max(counts.sum(), 1))


def calibration_report(X, y, order, seed: int = 0):
    """ECE/Brier raw vs calibrated; holdout = last 30% rows in time order."""
    rows = []
    curves = {}
    cut = int(len(y) * 0.7)
    tr_idx, te_idx = order[:cut], order[cut:]
    for name in ["LogReg", "GaussianNB", "HistGB"]:
        raw = make_models(seed)[name]
        raw.fit(X.iloc[tr_idx], y[tr_idx])
        p_raw = raw.predict_proba(X.iloc[te_idx])[:, 1]
        cal = CalibratedClassifierCV(make_models(seed)[name], method="sigmoid", cv=3)
        cal.fit(X.iloc[tr_idx], y[tr_idx])
        p_cal = cal.predict_proba(X.iloc[te_idx])[:, 1]
        yte = y[te_idx]
        rows.append({"model": name,
                     "brier_raw": round(brier_score_loss(yte, p_raw), 4),
                     "brier_cal": round(brier_score_loss(yte, p_cal), 4),
                     "ece_raw": round(expected_calibration_error(yte, p_raw), 4),
                     "ece_cal": round(expected_calibration_error(yte, p_cal), 4)})
        curves[name] = {"raw": p_raw, "cal": p_cal}
    return pd.DataFrame(rows), curves, yte


def leakage_audit(df: pd.DataFrame, t0) -> None:
    """Сколько групп повтором пересекает границу временного сплита.

    Если > 0 — temporal split сам по себе не спасает от утечки, дубли одной
    аварии попадают и в train, и в test (например, источник опубликовал
    сообщение позже). Для GroupKFold утечки гарантированно нет по построению.
    """
    ts = pd.to_datetime(df["Start_Time"], errors="coerce")
    lab = (ts <= t0).values
    tmp = pd.DataFrame({"gid": df["group_id"].values, "tr": lab})
    b = tmp.groupby("gid")["tr"].agg(["sum", "size"])
    cross = (b["sum"] > 0) & (b["sum"] < b["size"])
    dup_groups = int((b["size"] > 1).sum())
    n_cross = int(cross.sum())
    ids = b.index[cross]
    sub = tmp[tmp.gid.isin(ids)]
    print(f"   dup groups                 : {dup_groups:,}")
    print(f"   groups crossing time bound : {n_cross:,} "
          f"({(n_cross / max(dup_groups, 1)) * 100:.2f}% of dup-groups)")
    print(f"   test rows inside them      : {int((~sub.tr).sum()):,} "
          f"({(~sub.tr).sum() / max(len(sub), 1) * 100:.1f}% of their rows)")


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="kt3/df_with_groups.csv.gz")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--out-dir", default="kt3/out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False, nrows=args.rows)
    rng = np.random.default_rng(0)
    if len(df) > args.subsample:
        df = df.iloc[rng.choice(len(df), args.subsample, replace=False)].reset_index(drop=True)
    print(f"Using {len(df):,} rows; sources: {sorted(df.Source.unique())}")

    X, y = build_features(df)
    keep = X.notna().all(axis=1)
    X, y = X[keep].reset_index(drop=True), y[keep.values]
    df = df[keep].reset_index(drop=True)
    groups = df["group_id"].values
    sources = df["Source"].values
    ts = pd.to_datetime(df["Start_Time"], errors="coerce")
    year = ts.dt.year.fillna(2019).values.astype(float)
    order = np.argsort(ts.fillna(pd.Timestamp("2019-06-01")).values, kind="stable")

    print("\n" + "=" * 70)
    print("1. Split protocol comparison (leakage check)")
    res_rows = []
    for kind in ["random", "group", "temporal"]:
        oof = cv_oof(X, y, groups, sources, year, kind)
        for name, p in oof.items():
            ok = ~np.isnan(p)
            if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
                continue
            m = metrics(y[ok], p[ok])
            res_rows.append({"split": kind, "model": name, "n_eval": int(ok.sum()),
                             **{k: round(v, 4) for k, v in m.items()}})
    cmp_df = pd.DataFrame(res_rows)
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(f"{args.out_dir}/split_comparison.csv", index=False)

    print("\n   Interpretation: PR_AUC(random) >> PR_AUC(group/temporal)"
          " => утечка через повторы подтверждена.")

    print("\n" + "=" * 70)
    print("1b. Leakage audit: groups crossing the temporal boundary")
    ts_all = pd.to_datetime(df["Start_Time"], errors="coerce")
    leakage_audit(df, ts_all.quantile(0.75))

    print("\n" + "=" * 70)
    print("2. Leave-One-Source-Out (domain shift cost per source)")
    lo = loso(X, y, sources)
    if len(lo):
        print(lo.to_string(index=False))
        lo.to_csv(f"{args.out_dir}/loso.csv", index=False)
        piv = lo.pivot(index="model", columns="held_out_source", values="PR_AUC")
        print("\n   PR_AUC drop matrix (rows=model):")
        print(piv.round(4).to_string())
    else:
        print("   (источников < 2 или групп слишком мало — пропуск)")

    print("\n" + "=" * 70)
    print("3. Calibration (raw vs sigmoid-calibrated, last-30% holdout)")
    cal_df, curves, yte = calibration_report(X, y, order)
    print(cal_df.to_string(index=False))
    cal_df.to_csv(f"{args.out_dir}/calibration.csv", index=False)
    # save reliability table for notebook plotting
    rel_rows = []
    for name, d in curves.items():
        for variant, p in d.items():
            f, mp = calibration_curve(yte, p, n_bins=10, strategy="uniform")
            for a, b in zip(f, mp):
                rel_rows.append({"model": name, "variant": variant,
                                 "frac_positive": a, "mean_predicted": b})
    pd.DataFrame(rel_rows).to_csv(f"{args.out_dir}/reliability_curves.csv", index=False)
    print(f"\nSaved artifacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
