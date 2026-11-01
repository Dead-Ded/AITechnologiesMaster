"""
КТ3, подготовка данных для задания 3.

Файл df_with_groups.csv.gz (результат analyze_sources.py на полном датасете)
в репозитории отсутствует, поэтому group_id пересчитывается по kt2/short_file.csv
тем же методом (Haversine <= 150 м и пересечение интервалов / |dt| <= 6 ч),
после чего строки дедуплицируются до уровня аварий: 1 группа = 1 запись,
Severity - медиана внутри группы.

Результаты:
    kt3/df_with_groups_short.csv.gz  - строки short_file.csv + group_id;
    kt3/dedup_short.csv.gz           - дедуплицированная матрица (X, y, group_id).
"""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "kt3")
from analyze_sources import link_duplicates, NUM_COLS, BIN_COLS

warnings.filterwarnings("ignore")


def main() -> int:
    src = "kt2/short_file.csv"
    print(f"Loading {src} ...")
    df = pd.read_csv(src, low_memory=False)
    for c in ("Start_Time", "End_Time"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in BIN_COLS:
        df[c] = df[c].astype("boolean").astype(float)
    df = df.dropna(subset=["Start_Lat", "Start_Lng", "Source", "Severity"])
    df = df.reset_index(drop=True)
    print(f"rows with valid coords/severity: {len(df):,}")

    df["group_id"] = link_duplicates(df, radius_m=150.0, time_h=6.0)
    gsize = df["group_id"].value_counts()
    print(f"groups: {len(gsize):,}, dup row share: "
          f"{gsize[gsize > 1].sum() / len(df):.4f}")
    df.to_csv("kt3/df_with_groups_short.csv.gz", index=False, compression="gzip")

    # дедупликация: одна строка на группу, Severity - медиана внутри группы
    dd = (df.sort_values("group_id")
            .groupby("group_id", as_index=False)
            .agg({"Severity": "median", "Source": "first"}))
    first_rows = df.drop_duplicates("group_id").set_index("group_id")
    keep_cols = [c for c in NUM_COLS + BIN_COLS + ["Start_Lat", "Start_Lng"]]
    feats = first_rows.loc[dd["group_id"], keep_cols].reset_index()
    feats["Severity"] = dd["Severity"].values
    feats["Source"] = dd["Source"].values
    # пропуски в признаках не удаляем: в пайплайне стоит медианная импутация
    # (в исходном ноутбуке dropna по погоде оставлял ~3k строк из-за
    # Precipitation - 93% NaN, это слишком мало для кривых обучения)
    feats["y"] = (feats["Severity"] >= 3).astype(int)
    print(f"dedup accidents: {len(feats):,}, P(sev>=3) = {feats['y'].mean():.4f}")
    feats.to_csv("kt3/dedup_short.csv.gz", index=False, compression="gzip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
