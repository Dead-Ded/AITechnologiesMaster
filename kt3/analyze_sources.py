"""
Анализ источников (Source1..Source4) датасета US Accidents.

Задачи:
  1. Пространственно-временные повторы (duplicates) с учётом погрешности замеров:
     толерантный поиск соседей через cKDTree (расстояние Haversine <= R_m,
     |dt| <= T_h часов), объединение в связные компоненты -> group_id аварии.
  2. Смещение (bias) распределений признаков между источниками:
     PSI и KS-статистика для числовых признаков, доля категории для бинарных.
  3. Полнота данных (missing rate) по источникам.
  4. Стабильность Severity внутри межсточниковых групп повтором.
  5. Временное смещение источников (сдвиг медианы даты) + сводка по году.

Использование:
    python source_analysis.py [--data PATH] [--rows N] [--radius-m 150] [--time-h 6]

По умолчанию берётся kt2/short_file.csv; --rows ограничивает число строк
(полезно для полного файла ~7.3M строк).
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

EARTH_R_M = 6_371_000.0
NUM_COLS = ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Pressure(in)",
            "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
BIN_COLS = ["Bump", "Crossing", "Junction", "Railway", "Station", "Stop",
            "Traffic_Signal", "Amenity"]


# ---------------------------------------------------------------- data loading
def load_data(path: str, nrows: int | None) -> pd.DataFrame:
    usecols = ["ID", "Source", "Severity", "Start_Time", "End_Time",
               "Start_Lat", "Start_Lng"] + NUM_COLS + BIN_COLS
    df = pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
    for c in ("Start_Time", "End_Time"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in BIN_COLS:
        if c in df.columns:
            df[c] = df[c].astype("boolean").astype(float)  # NaN сохраняется
    df = df.dropna(subset=["Start_Lat", "Start_Lng", "Source", "Severity"])
    df["SeverityBin"] = (df["Severity"] >= 3).astype(int)
    return df.reset_index(drop=True)


# ------------------------------------------------ spatial-temporal duplicates
def haversine_m(lat1, lng1, lat2, lng2):
    """Пары массивов -> расстояние в метрах (векторно)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lng1 - lng2)
    x = np.cos(p2) * np.sin(dl)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return EARTH_R_M * np.arctan2(np.hypot(x, y), np.sqrt(1 - x ** 2 - y ** 2))


def _to_xyz(lat_deg, lng_deg):
    """Перевод координат в 3D-точки на сфере (для cKDTree)."""
    lat, lng = np.radians(lat_deg), np.radians(lng_deg)
    return np.column_stack([np.cos(lat) * np.cos(lng),
                            np.cos(lat) * np.sin(lng),
                            np.sin(lat)])


def link_duplicates(df: pd.DataFrame, radius_m: float, time_h: float) -> np.ndarray:
    """Возвращает group_id: id связных компонент графа 'повторов'.

    Ребро между записями i, j, если Haversine(i,j) <= radius_m И
    |Start_Time_i - Start_Time_j| <= time_h (или пересекаются интервалы
    Start_Time..End_Time — ловит сообщения об одной аварии с разных источников
    с разным временем публикации).
    """
    n = len(df)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    xyz = _to_xyz(df["Start_Lat"].values, df["Start_Lng"].values)
    # chord distance на единичной сфере, соответствующий radius_m
    chord = 2.0 * np.sin(np.radians(radius_m / EARTH_R_M) / 2.0)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=chord, output_type="ndarray")

    st = df["Start_Time"].values.astype("datetime64[s]").astype(np.int64)
    et = df["End_Time"].fillna(df["Start_Time"]).values.astype("datetime64[s]").astype(np.int64)
    dt_lim = int(time_h * 3600)

    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        di = np.abs(st[i] - st[j])
        # пересечение интервалов [start, end]
        overlap = (st[i] <= et[j]) & (st[j] <= et[i])
        mask = (di <= dt_lim) | overlap
        for a, b in zip(i[mask], j[mask]):
            union(a, b)

    roots = np.array([find(k) for k in range(n)])
    _, group_id = np.unique(roots, return_inverse=True)
    return group_id


def duplicate_stats(df: pd.DataFrame, group_id: np.ndarray) -> dict:
    gsize = pd.Series(group_id).value_counts()
    dup_rows = int((gsize[gsize > 1]).sum())
    dup_groups = int((gsize > 1).sum())
    out = {
        "n_rows": len(df),
        "n_groups": len(gsize),
        "dup_row_share": dup_rows / len(df),
        "dup_group_share": dup_groups / len(gsize),
    }
    tmp = df[["Source", "SeverityBin"]].copy()
    tmp["gid"] = group_id
    multi_src = (tmp.groupby("gid")["Source"].nunique() > 1)
    out["groups_multi_source_share"] = float(multi_src.mean())

    sev = tmp.groupby("gid")["SeverityBin"].agg(["nunique", "size"])
    same = sev[sev["nunique"] == 1]
    out["sev_agreement_within_dup"] = float(len(same) / len(sev))
    return out


# ------------------------------------------------------------ bias metrics
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index по квантильным бинам expected."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e, _ = np.histogram(expected, qs)
    a, _ = np.histogram(actual, qs)
    e = np.clip(e / e.sum(), 1e-4, None)
    a = np.clip(a / a.sum(), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    return float(ks_2samp(a, b).statistic)


def pairwise_bias(df: pd.DataFrame, ref: str | None = None) -> pd.DataFrame:
    """PSI/KS каждого источника относительно референсного (по умолчанию самый крупный)."""
    if ref is None:
        ref = df["Source"].value_counts().index[0]
    rows = []
    srcs = sorted(df["Source"].unique())
    for col in NUM_COLS:
        base = df.loc[df.Source == ref, col].dropna().values
        for s in srcs:
            cur = df.loc[df.Source == s, col].dropna().values
            if len(cur) < 50 or len(base) < 50:
                continue
            rows.append({"feature": col, "source": s, "psi_vs_ref": psi(base, cur),
                         "ks_vs_ref": ks_stat(base, cur),
                         "mean_diff": float(cur.mean() - base.mean()),
                         "median_diff": float(np.median(cur) - np.median(base))})
    return pd.DataFrame(rows)


def binary_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Транспонированная таблица: строки — бинарные признаки, столбцы — источники.

    max_range — разброс доли True между крупными источниками (n>=100),
    чтобы мелкие источники не шумили в статистике.
    """
    piv = df.groupby("Source")[BIN_COLS].mean().T  # признак x источник
    counts = df.groupby("Source").size()
    big = counts[counts >= 100].index
    cols = big if len(big) else piv.columns
    piv["max_range"] = piv[cols].max(axis=1) - piv[cols].min(axis=1)
    return piv.sort_values("max_range", ascending=False)


# ------------------------------------------------------------- completeness
def missing_by_source(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby("Source").apply(lambda g: g.isna().mean(), include_groups=False)
    return m[NUM_COLS + BIN_COLS].round(4)


# ---------------------------------------------------------- temporal drift
def temporal_profile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(year=df["Start_Time"].dt.year)
    piv = df.pivot_table(index="year", columns="Source", values="ID", aggfunc="count")
    share = piv.div(piv.sum(axis=1), axis=0).round(3)
    med = df.groupby("Source")["Start_Time"].median().dt.date
    return share, med


# --------------------------------------------------- cross-source severity
def cross_source_severity(df: pd.DataFrame, group_id: np.ndarray) -> pd.DataFrame:
    """Согласованность SeverityBin в группах, где участвуют >=2 источника."""
    t = df[["Source", "SeverityBin"]].copy()
    t["gid"] = group_id
    if len(group_id) != len(df):
        group_id = df["group_id"].values
    multi = t.groupby("gid").filter(lambda g: g["Source"].nunique() > 1)
    if multi.empty:
        return pd.DataFrame()
    res = multi.groupby(["gid", "Source"])["SeverityBin"].mean().unstack()
    out = {}
    srcs = [s for s in sorted(df["Source"].unique()) if s in res.columns]
    for a_i, a in enumerate(srcs):
        for b in srcs[a_i + 1:]:
            pair = res[[a, b]].dropna()
            if len(pair) >= 10:
                agree = float((pair[a] == pair[b]).mean())
                bias = float(pair[b].mean() - pair[a].mean())
                out[f"{a} vs {b}"] = {"n_pairs": len(pair), "agree": agree,
                                      f"bias({b}-{a})": bias}
    return pd.DataFrame(out).T


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=r"C:\Users\musai\.cache\kagglehub\datasets\sobhanmoosavi\us-accidents\versions"
                                      r"\13\US_Accidents_March23.csv")
    ap.add_argument("--rows", type=int, default=None, help="limit rows (for the full 7M file)")
    ap.add_argument("--radius-m", type=float, default=150.0)
    ap.add_argument("--time-h", type=float, default=6.0)
    args = ap.parse_args()

    print(f"Loading {args.data} (rows={args.rows}) ...")
    df = load_data(args.data, args.rows)
    print(f"Loaded {len(df):,} rows; sources: {sorted(df.Source.unique())}\n")

    print("=" * 70)
    print("1. Composition & target rate by source")
    comp = df.groupby("Source").agg(
        n=("ID", "count"),
        sev_rate=("SeverityBin", "mean"),
        sev_mean=("Severity", "mean"),
    )
    comp["share"] = comp["n"] / comp["n"].sum()
    print(comp.round(4).to_string())

    print("\n" + "=" * 70)
    print(f"2. Spatial-temporal duplicates (R<={args.radius_m:.0f} m, "
          f"|dt|<={args.time_h:.0f} h или пересечение интервалов)")
    group_id = link_duplicates(df, args.radius_m, args.time_h)
    stats = duplicate_stats(df, group_id)
    for k, v in stats.items():
        print(f"   {k:>28}: {v:,.4f}" if isinstance(v, float) else f"   {k:>28}: {v:,}")
    df["group_id"] = group_id

    print("\n   Cross-source severity agreement:")
    cs = cross_source_severity(df, group_id)
    if len(cs):
        print(cs.round(4).to_string())
    else:
        print("   (недостаточно межсточниковых пар)")

    print("\n" + "=" * 70)
    ref = df["Source"].value_counts().index[0]
    print(f"3. Feature bias between sources (PSI/KS vs {ref})")
    bias = pairwise_bias(df, ref)
    top = bias.sort_values("psi_vs_ref", ascending=False).head(15)
    print(top.round(4).to_string(index=False))
    print("   (PSI > 0.2 — существенный сдвиг, 0.1–0.2 — умеренный)")

    print("\n   Binary feature prevalence spread:")
    print(binary_bias(df).round(4).to_string())

    print("\n" + "=" * 70)
    print("4. Missingness by source")
    print(missing_by_source(df).to_string())

    print("\n" + "=" * 70)
    print("5. Temporal profile")
    share, med = temporal_profile(df)
    print(share.to_string())
    print("\nMedian Start_Time per source:")
    print(med.to_string())

    out_path = "kt3/df_with_groups.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")
    print(f"\nSaved df with group_id -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
