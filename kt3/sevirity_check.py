"""
Шаг 1. Проверка маппинга / бинаризации Severity (Severity >= 3).

Контекст: на полном файле (7.7M строк) доля класса "Severity >= 3" получилась
~8%, что противоречит ожиданиям (~60/40 или ~67/33, как на подвыборках).
Скрипт отвечает на три вопроса:

  A. Цел ли сам маппинг: совпадает ли SeverityBin с (Severity >= 3) построчно,
     нет ли битых/потерянных значений Severity, не «схлопнулись» ли классы.
  B. Как распределяется Severity по источникам и годам — т.е. объясняется ли
     низкая доляPositive-класса составом выборки (доля Source1, у которого
     тяжелые аварии регистрируются реже, выросла после 2020 г.).
  C. Согласованна ли метка внутри пространственно-временных групп повтором
     (group_id из source_analysis.py): если одна и та же авария в двух
     источниках получает разный Severity, часть расхождения — метка источника,
     а не реальность.

Использование:
    python severity_check.py [--data kt3/df_with_groups.csv.gz]
                             [--raw PATH] [--rows N]

--data — файл сохранный source_analysis.py (уже содержит group_id).
--raw  — исходный CSV (опционально), чтобы проверить полные распределения.
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def check_mapping(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("A. Построчная проверка бинаризации SeverityBin = (Severity >= 3)")
    raw_sev = pd.to_numeric(df["Severity"], errors="coerce")
    n_bad = int(raw_sev.isna().sum())
    recomputed = (raw_sev >= 3).astype(float)
    stored = pd.to_numeric(df["SeverityBin"], errors="coerce")
    mismatch = int((recomputed != stored).sum())
    print(f"   rows                : {len(df):,}")
    print(f"   non-numeric Severity: {n_bad:,}")
    print(f"   mismatches vs >=3   : {mismatch:,}"
          + ("  -> БИНАРИЗАЦИЯ КОРРЕКТНА" if mismatch == 0 else "  -> !!! РАЗБЕРИТЕСЬ"))
    print("\n   Distribution of Severity:")
    dist = raw_sev.value_counts(dropna=False).sort_index()
    for k, v in dist.items():
        print(f"     {k}: {int(v):>10,}  ({v / len(df) * 100:5.2f}%)")
    pos = float((raw_sev >= 3).mean())
    print(f"\n   P(Severity>=3) overall = {pos:.4f}")


def by_source_year(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("B. Доля P(Severity>=3) по источникам и годам (объяснение состава)")
    d = df.copy()
    d["year"] = pd.to_datetime(d["Start_Time"], errors="coerce").dt.year
    piv = d.pivot_table(index="year", columns="Source",
                        values="Severity", aggfunc=lambda s: (pd.to_numeric(s, errors="coerce") >= 3).mean())
    share = d.pivot_table(index="year", columns="Source", values="Severity", aggfunc="size")
    share = share.div(share.sum(axis=1), axis=0)
    print("\n   P(sev>=3 | source, year):")
    print(piv.round(3).to_string())
    print("\n   row share of sources per year:")
    print(share.round(3).to_string())
    # decomposition: what would the rate be with a fixed (pooled) mix?
    rates = d.groupby("year").apply(lambda g: (pd.to_numeric(g.Severity, errors="coerce") >= 3).mean(), include_groups=False)
    print("\n   P(sev>=3) per year (all sources):")
    print(rates.round(4).to_string())


def label_consistency(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("C. Согласованность метки SeverityBin внутри групп повтором (group_id)")
    if "group_id" not in df.columns:
        print("   (нет колонки group_id — пропуск)")
        return
    g = df.groupby("group_id")
    size = g.size()
    dup = size[size > 1].index
    sub = df[df.group_id.isin(dup)]
    nun = sub.groupby("group_id")["SeverityBin"].nunique()
    print(f"   groups with >1 row      : {len(dup):,}")
    print(f"   inconsistent labels     : {int((nun > 1).sum()):,} "
          f"({(nun > 1).mean() * 100:.2f}% of dup-groups)")
    # cross-source only
    multi_src = sub.groupby("group_id")["Source"].nunique() > 1
    ms_ids = multi_src[multi_src].index
    nun_ms = sub[sub.group_id.isin(ms_ids)].groupby("group_id")["SeverityBin"].nunique()
    if len(nun_ms):
        print(f"   dup-groups w/ >=2 sources : {len(ms_ids):,}")
        print(f"   inconsistent there        : {int((nun_ms > 1).sum()):,} "
              f"({(nun_ms > 1).mean() * 100:.2f}%)")
        print("   -> это верхняя оценка меточного шума между источниками;")
        print("      для обучения берите моду метки группы (median/majority vote).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="kt3/df_with_groups.csv.gz")
    ap.add_argument("--raw", default=None, help="optional original CSV to recheck full distributions")
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False, nrows=args.rows)
    check_mapping(df)
    by_source_year(df)
    label_consistency(df)

    if args.raw:
        print("\n" + "=" * 70)
        print(f"D. Полный файл {args.raw}: распределение Severity без фильтрации")
        s = pd.read_csv(args.raw, usecols=["Severity"], nrows=args.rows, low_memory=False)["Severity"]
        print(s.value_counts(dropna=False).sort_index().to_string())
        print(f"P(Severity>=3) = {(s >= 3).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
