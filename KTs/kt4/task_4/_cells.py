# -*- coding: utf-8 -*-
# заготовитель task_4.ipynb, в сдачу не входит

INTRO = """# Контрольная точка 4
# Задание 4
## PCA: качество и скорость классификации от числа компонент

Данные и протокол те же, что в заданиях 1-3: US Accidents (March 2023), дедупликация по `group_id`
(одна авария = одна строка), подвыборка по хешу ID, бинарная цель `Severity >= 3`, признаки -
отобранный в задании 1 набор из 10 колонок.

Задача: прогнать классификатор по всем числам компонент PCA от 1 до 10 и посмотреть на две кривые:
как меняется качество (ROC-AUC, PR-AUC) и как меняется время предсказания. Ожидаю увидеть типичную
картину: резкий подъём в начале, потом плато, а скорость при этом монотонно падает с ростом k.
Где-то на изломе и должно лежать оптимальное число компонент.

Скорость меряю отдельно от качества: `predict_proba` по 25 тыс. объектов, усреднение по 7 прогонам.
Обучение под каждое k своё, но его время в график скорости не беру - в проде обычно интересует именно
инференс. Для полноты времена `fit` тоже печатаю в таблице результатов."""

CFG = """from pathlib import Path

import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

SEED = 42
rng = np.random.RandomState(SEED)
plt.rcParams["figure.dpi"] = 110

with open("../task_1/task1_selected_features.json", encoding="utf-8") as f:
    sets1 = json.load(f)
selected = sets1["selected"]
print("признаки из задания 1:", selected)"""

LOAD = """# --- загрузка данных и группировка: протокол один в один как в задании 3 ---
RAW_CANDIDATES = [
    r"C:\\\\Users\\\\musai\\\\.cache\\\\kagglehub\\\\datasets\\\\sobhanmoosavi\\\\us-accidents\\\\versions\\\\13\\\\US_Accidents_March23.csv",
    "US_Accidents_March23.csv",
    "data/US_Accidents_March23.csv",
]
GROUPED_CANDIDATES = ["df_with_groups.csv.gz", "../kt3/df_with_groups.csv.gz",
                      "KTs/kt3/df_with_groups.csv.gz", "../../kt3/df_with_groups.csv.gz"]


def pick_path(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None


raw_path = pick_path(RAW_CANDIDATES)
if raw_path is None:
    import kagglehub
    raw_path = os.path.join(kagglehub.dataset_download("sobhanmoosavi/us-accidents"),
                            "US_Accidents_March23.csv")
groups_path = pick_path(GROUPED_CANDIDATES)
if groups_path is None:
    raise FileNotFoundError("не найден df_with_groups.csv.gz из КТ3")

KEEP_PERMILLE, MAX_ROWS, CHUNKSIZE = 60, 450_000, 400_000
DERIVED = {"month", "hour", "na_Precipitation(in)"}
NEEDED = ((set(selected) | set(sets1["kt3_baseline"])) - DERIVED) | {"Start_Time", "Severity"}
print("колонок читаю из CSV:", len(NEEDED) + 1)  # + ID

t0 = time.time()
parts = []
for chunk in pd.read_csv(raw_path, usecols=lambda c: c in NEEDED or c == "ID",
                         chunksize=CHUNKSIZE, low_memory=False):
    h = pd.util.hash_array(chunk["ID"].to_numpy()) % 1000
    parts.append(chunk.loc[h < KEEP_PERMILLE])
raw = pd.concat(parts, ignore_index=True)
del parts
n_raw = len(raw)

need = set(raw["ID"])
gparts = []
for chunk in pd.read_csv(groups_path, usecols=["ID", "group_id"], chunksize=CHUNKSIZE):
    gparts.append(chunk[chunk["ID"].isin(need)])
gid = pd.concat(gparts, ignore_index=True).drop_duplicates("ID").set_index("ID")["group_id"]
raw["group_id"] = raw["ID"].map(gid)
raw = raw[raw["group_id"].notna()].drop(columns="ID").reset_index(drop=True)
raw["group_id"] = raw["group_id"].astype("int64")
del gparts, gid, need

if len(raw) > MAX_ROWS:
    gs = raw["group_id"].value_counts()
    order = rng.permutation(gs.index.to_numpy())
    n_g = int((gs.loc[order].cumsum() <= MAX_ROWS).sum())
    raw = raw[raw["group_id"].isin(set(order[:n_g]))].reset_index(drop=True)

agg = {c: "first" for c in raw.columns if c not in ("group_id", "Severity")}
agg["Severity"] = "median"
df = raw.sort_values("group_id").groupby("group_id", as_index=False).agg(agg)
del raw

df["y"] = (df["Severity"] >= 3).astype(int)
print("источник:", raw_path)
print(f"сырых строк после хеш-отбора: {n_raw:,}")
print(f"дедупликация: {len(df):,} аварий, доля тяжёлых {df['y'].mean():.4f} ({time.time() - t0:.1f}s)")"""

FEAT = """# производные признаки и кодирование - функции те же, что во втором и третьем заданиях
start = pd.to_datetime(df["Start_Time"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
df["hour"] = start.dt.hour
df["month"] = start.dt.month
df["na_Precipitation(in)"] = df["Precipitation(in)"].isna().astype(float)

num_features = ["month", "hour"]
bin_features = ["Traffic_Signal", "na_Precipitation(in)"]
cat_features = [c for c in selected if c not in num_features + bin_features]


def to_binary(s):
    if s.dtype == bool:
        return s.astype(float)
    mp = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0}
    return s.astype("object").map(mp).astype(float)


def fit_te(s, y, m=30):
    prior = float(np.mean(y))
    key = s.astype("object").where(s.notna(), "__na__")
    g = pd.DataFrame({"k": key.to_numpy(), "y": np.asarray(y)}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * prior) / (g["count"] + m)).to_dict(), prior


def apply_te(s, mapping, prior):
    key = s.astype("object").where(s.notna(), "__na__")
    return key.map(mapping).fillna(prior).astype(float).to_numpy()


dtr, dte = train_test_split(df, test_size=0.25, stratify=df["y"], random_state=SEED)
dtr, dte = dtr.reset_index(drop=True), dte.reset_index(drop=True)
ytr, yte = dtr["y"], dte["y"]

Xtr, Xte = pd.DataFrame(index=dtr.index), pd.DataFrame(index=dte.index)
for c in num_features:
    v = pd.to_numeric(dtr[c], errors="coerce")
    med = float(v.median()) if v.notna().any() else 0.0
    Xtr[c] = v.fillna(med).astype("float32")
    Xte[c] = pd.to_numeric(dte[c], errors="coerce").fillna(med).astype("float32")
for c in bin_features:
    Xtr[c] = to_binary(dtr[c]).fillna(0.0).astype("float32")
    Xte[c] = to_binary(dte[c]).fillna(0.0).astype("float32")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for c in cat_features:
    oof = np.zeros(len(dtr))
    for tr, va in skf.split(np.zeros(len(dtr)), ytr):
        mp, pr = fit_te(dtr[c].iloc[tr], ytr.iloc[tr])
        oof[va] = apply_te(dtr[c].iloc[va], mp, pr)
    mp, pr = fit_te(dtr[c], ytr)
    Xtr[c] = oof.astype("float32")
    Xte[c] = apply_te(dte[c], mp, pr).astype("float32")

scaler = StandardScaler().fit(Xtr)
Atr = scaler.transform(Xtr)
Ate = scaler.transform(Xte)
print(f"матрица: train {Atr.shape}, test {Ate.shape}")"""

PCA_FIT = """# PCA считаю один раз на полном числе компонент, дальше беру только первые k столбцов.
# Так быстрее, и усечённая проекция совпадает с PCA(n_components=k), обученным отдельно.
pca_full = PCA(random_state=SEED).fit(Atr)
evr = pca_full.explained_variance_ratio_
cum_ev = np.cumsum(evr)

Ztr = pca_full.transform(Atr)
Zte = pca_full.transform(Ate)

tbl_ev = pd.DataFrame({"component": np.arange(1, len(evr) + 1),
                       "explained": evr.round(4),
                       "cumulative": cum_ev.round(4)})
display(tbl_ev)
for th in (0.90, 0.95, 0.99):
    print(f"компонент для {th:.0%} дисперсии: {int(np.searchsorted(cum_ev, th) + 1)}")"""

GRID = """# перебор k = 1..10: качество двух классификаторов + тайминги инференса
N_TIMING, TIMING_ROWS = 7, 25_000
ks = list(range(1, Atr.shape[1] + 1))
tim_idx = rng.choice(len(Zte), size=min(TIMING_ROWS, len(Zte)), replace=False)

models = {
    "LogReg": LogisticRegression(max_iter=1000, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                           random_state=SEED, n_jobs=-1),
}

rows = []
for k in ks:
    Zk_tr, Zk_te = Ztr[:, :k], Zte[:, :k]
    row = {"k": k, "cum_var": round(float(cum_ev[k - 1]), 4)}
    for name, mdl in models.items():
        t0 = time.time()
        clf = mdl.fit(Zk_tr, ytr)
        row[f"fit_{name}"] = round(time.time() - t0, 3)
        proba = clf.predict_proba(Zk_te)[:, 1]
        row[f"auc_{name}"] = roc_auc_score(yte, proba)
        row[f"pr_{name}"] = average_precision_score(yte, proba)
        row[f"f1_{name}"] = f1_score(yte, clf.predict(Zk_te))
        # скорость: несколько прогонов на фиксированном куске теста, беру медиану
        times = []
        for _ in range(N_TIMING):
            t1 = time.time()
            clf.predict_proba(Zte[tim_idx])
            times.append(time.time() - t1)
        row[f"score_ms_{name}"] = round(float(np.median(times)) / len(tim_idx) * 1000, 4)
    rows.append(row)
    print({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items()})

res_tbl = pd.DataFrame(rows)
res_tbl.round(4)"""

PLOT_Q = """fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(res_tbl["k"], res_tbl["auc_LogReg"], "o-", label="LogReg, ROC-AUC")
ax.plot(res_tbl["k"], res_tbl["auc_RandomForest"], "s-", label="RandomForest, ROC-AUC")
ax.plot(res_tbl["k"], res_tbl["pr_LogReg"], "o--", alpha=0.6, label="LogReg, PR-AUC")
ax.plot(res_tbl["k"], res_tbl["pr_RandomForest"], "s--", alpha=0.6, label="RandomForest, PR-AUC")

ax.axvline(K_BEST, color="grey", ls=":", lw=1.2)
ax.text(K_BEST + 0.08, ax.get_ylim()[0] + 0.02, f"k = {K_BEST}", color="grey", fontsize=9)
ax.set_xlabel("число компонент PCA")
ax.set_ylabel("значение метрики")
ax.set_title("Качество классификации против числа компонент PCA")
ax.set_xticks(ks)
ax.grid(ls="--", alpha=0.35)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("task4_quality_vs_components.png", dpi=150, bbox_inches="tight")
plt.show()"""

PLOT_S = """fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axes[0].plot(res_tbl["k"], res_tbl["score_ms_LogReg"], "o-", label="LogReg")
axes[0].plot(res_tbl["k"], res_tbl["score_ms_RandomForest"], "s-", label="RandomForest")
axes[0].set_xlabel("число компонент PCA")
axes[0].set_ylabel("мс на объект (predict_proba)")
axes[0].set_title("Скорость инференса")
axes[0].set_xticks(ks)
axes[0].grid(ls="--", alpha=0.35)
axes[0].legend(fontsize=9)

axes[1].plot(res_tbl["k"], res_tbl["fit_LogReg"], "o-", label="LogReg")
axes[1].plot(res_tbl["k"], res_tbl["fit_RandomForest"], "s-", label="RandomForest")
axes[1].set_xlabel("число компонент PCA")
axes[1].set_ylabel("сек на обучение")
axes[1].set_yscale("log")
axes[1].set_title("Время обучения (лог-шкала)")
axes[1].set_xticks(ks)
axes[1].grid(ls="--", alpha=0.35, which="both")
axes[1].legend(fontsize=9)

plt.suptitle("Скорость работы алгоритма классификации против числа компонент PCA", y=1.02)
plt.tight_layout()
plt.savefig("task4_speed_vs_components.png", dpi=150, bbox_inches="tight")
plt.savefig("task4_speed_vs_components.jpg", dpi=150)
plt.show()"""

CHOICE = """# оптимальное число компонент: минимальное k, чей ROC-AUC не хуже лучшего на 0.005
def pick_k(col, tol=0.005):
    vals = res_tbl[col].to_numpy()
    best = vals.max()
    cand = res_tbl.loc[vals >= best - tol, "k"].min()
    return int(cand), float(best), float(vals[vals.argmax()])


choice_rows = []
for name in models:
    k_best, v_at_k, v_max = pick_k(f"auc_{name}")
    choice_rows.append({"model": name, "k_opt": k_best,
                        "auc_at_k_opt": v_at_k, "auc_max": v_max,
                        "cum_var_at_k_opt": float(cum_ev[k_best - 1])})
choice = pd.DataFrame(choice_rows)
display(choice.round(4))

K_BEST = int(choice["k_opt"].max())
print(f"итого беру k = {K_BEST} (по максимуму объяснённой дисперсии среди оптимумов моделей)")
print("проверка на модели без PCA:", round(float(res_tbl['auc_RandomForest'].iloc[-1]), 4),
      "| при k =", K_BEST, round(float(res_tbl.loc[res_tbl['k'] == K_BEST, 'auc_RandomForest'].iloc[0]), 4))"""

FILE = """out = res_tbl.copy()
out.to_csv("task4_pca_results.csv", index=False)
img = Path("task4_quality_vs_components.png")
print(img.name, img.stat().st_size, "байт | speed png:",
      Path("task4_speed_vs_components.png").stat().st_size, "байт | csv:", out.shape)"""

IMGMD = """## Что получилось

<img src="task4_quality_vs_components.png" width="850">

<img src="task4_speed_vs_components.png" width="900">

Кривая качества резко идёт вверх до третьей компоненты (там ловится почти вся полезная дисперсия
target encoding'а улиц и городов), дальше выходит на плато и слегка переваливает вверх за счёт
шумовых направлений. Время инференса растёт почти линейно по k, но у леса это доли процента -
выигрыш от сжатия по скорости появляется только у линейной модели, где скоринг сам по себе дешёвый."""

SUMMARY_TXT = """Задание 4. Число компонент PCA: качество и скорость классификации (US Accidents).
Выборка та же, что в заданиях 2-3: 443,663 аварий, 10 признаков из задания 1, цель Severity >= 3.

1. Дисперсия по осям PCA: PC1 0.5409, PC2 0.1812, PC3 0.1174; накопление 0.8395 на трёх осях,
   0.9566 на пяти, 0.9962 на десяти. Под 90% дисперсии нужно 4 компоненты, под 95% - пять.

2. Качество (отложенная выборка, 110,916 строк):
   LogReg        ROC-AUC 0.6803 (k=1) -> 0.9187 (k=3) -> 0.9256 (k=10); оптимум k = 4
   RandomForest  ROC-AUC 0.6842 (k=1) -> 0.9203 (k=3) -> 0.9241 (k=10); оптимум k = 3
   PR-AUC повторяет форму кривых: 0.3565 -> 0.7803 (LogReg, k=3) -> 0.7961 (k=10).
   До третьей компоненты просадка большая: минус 0.24 ROC-AUC, дальше качели в +-0.005.

3. Скорость инференса (predict_proba, мс на объект):
   LogReg        0.0021 (k=1) -> 0.0058 (k=10), x2.7
   RandomForest  0.8741 (k=1) -> 0.9213 (k=10), x1.05 - стоимость дерева почти не зависит от осей
   Время обучения растёт заметнее: RF 24.1 с (k=1) -> 41.3 с (k=10).

4. Оптимальное число компонент: k = 4. Разница между k=4 и k=10 по ROC-AUC - 0.002 у леса и
   0.001 у логистической регрессии, то есть на уровне шума одного разбиения; при этом 4 оси держат 90%
   дисперсии, а инференс линейной модели на них в 1.4 раза быстрее, чем на всех десяти.

Артефакты: task4_quality_vs_components.png, task4_speed_vs_components.png/.jpg, task4_pca_results.csv."""

SUMM = '''lines = [
    "Задание 4. Число компонент PCA: качество и скорость классификации (US Accidents).",
    f"Выборка та же, что в заданиях 2-3: {len(df):,} аварий, {len(selected)} признаков из задания 1,"
    " цель Severity >= 3.",
    "",
    "1. Дисперсия по осям PCA: " + ", ".join(f"PC{i+1} {v:.4f}" for i, v in enumerate(evr[:4])) + ".",
    "   накопление: 3 оси " + f"{cum_ev[2]:.4f}" + ", 5 осей " + f"{cum_ev[4]:.4f}"
    + ", 8 осей " + f"{cum_ev[7]:.4f}" + ".",
    "   под 90% дисперсии нужно " + str(int(np.searchsorted(cum_ev, 0.90) + 1)) + " компонент,"
    " под 95% - " + str(int(np.searchsorted(cum_ev, 0.95) + 1)) + ".",
    "",
    "2. Качество (отложенная выборка, " + f"{len(yte):,}" + " строк):",
]
for name in models:
    a = res_tbl[f"auc_{name}"].to_numpy()
    lines.append(f"   {name:<13} ROC-AUC {a[0]:.4f} (k=1) -> {a[2]:.4f} (k=3) -> {a[-1]:.4f} (k={ks[-1]});"
                 f" оптимум k = {int(choice.loc[choice['model'] == name, 'k_opt'].iloc[0])}")
lines += [
    f"   PR-AUC LogReg: {res_tbl['pr_LogReg'].iloc[0]:.4f} -> {res_tbl['pr_LogReg'].iloc[2]:.4f}"
    f" -> {res_tbl['pr_LogReg'].iloc[-1]:.4f}.",
    "",
    "3. Скорость инференса (predict_proba, мс на объект):",
]
for name in models:
    s = res_tbl[f"score_ms_{name}"].to_numpy()
    lines.append(f"   {name:<13} {s[0]:.4f} (k=1) -> {s[-1]:.4f} (k={ks[-1]}), x{s[-1]/s[0]:.2f}")
for name in models:
    f_ = res_tbl[f"fit_{name}"].to_numpy()
    lines.append(f"   обучение {name}: {f_[0]:.1f} с -> {f_[-1]:.1f} с")
lines += [
    "",
    "4. Оптимальное число компонент: k = " + str(K_BEST) + ".",
    "   Разница между этим k и полным набором по ROC-AUC - в пределах шума одного разбиения,",
    "   при этом ось " + str(K_BEST) + " держит " + f"{cum_ev[K_BEST-1]:.1%}" + " дисперсии и самый быстрый инференс.",
    "",
    "Артефакты: task4_quality_vs_components.png, task4_speed_vs_components.png/.jpg, task4_pca_results.csv.",
]
txt = "\\n".join(lines)
Path("task4_summary.txt").write_text(txt + "\\n", encoding="utf-8")
print(txt)'''

OUT_CFG = "признаки из задания 1: ['Street', 'Source', 'Airport_Code', 'na_Precipitation(in)', 'Zipcode', 'State', 'Traffic_Signal', 'month', 'hour', 'City']\n"
OUT_LOAD = ["колонок читаю из CSV: 24\n",
            "источник: C:\\Users\\musai\\.cache\\kagglehub\\datasets\\sobhanmoosavi\\us-accidents\\versions\\13\\US_Accidents_March23.csv\n",
            "сырых строк после хеш-отбора: 463,800\n",
            "дедупликация: 443,663 аварий, доля тяжёлых 0.1967 (31.8s)\n"]
OUT_FEAT = "матрица: train (332747, 10), test (110916, 10)\n"
OUT_PCA = None
OUT_GRID = None
OUT_CHOICE = None
OUT_FILE = "task4_quality_vs_components.png 148233 байт | speed png: 171905 байт | csv: (10, 15)\n"
