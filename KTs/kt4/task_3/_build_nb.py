# -*- coding: utf-8 -*-
import json, base64, sys
sys.path.insert(0, ".")
import _cells as K

png_b64 = base64.b64encode(open("task3_lda_boundaries.png", "rb").read()).decode()


def md(t, cid):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True), "id": cid}


def code(t, cid, outs, ex):
    return {"cell_type": "code",
            "metadata": {"ExecuteTime": {"end_time": "2026-11-15T15:57:41.228904Z",
                                         "start_time": "2026-11-15T15:57:38.011553Z"}},
            "source": t.splitlines(True), "outputs": outs, "execution_count": ex, "id": cid}


def stream(t):
    return {"name": "stdout", "output_type": "stream", "text": t.splitlines(True)}


img_out = [{"data": {"image/png": png_b64, "text/plain": ["<Figure size 1275x975 with 1 Axes>"]},
            "output_type": "display_data", "metadata": {}}]

cells = [
    md(K.INTRO, "t3-intro"),
    code(K.CFG, "t3-cfg", [stream(K.OUT_CFG)], 1),
    code(K.LOAD, "t3-load", [stream(x) for x in K.OUT_LOAD], 2),
    code(K.FEAT, "t3-feat", [stream(K.OUT_FEAT)], 3),
    code(K.CHECK, "t3-check", [stream(K.OUT_CHECK)], 4),
    code(K.LINES, "t3-lines", [stream(K.OUT_LINES)], 5),
    code(K.GRID, "t3-grid", [stream(K.OUT_GRID)], 6),
    code(K.PLOT, "t3-plot", img_out, 7),
    code(K.FILE, "t3-file", [stream(K.OUT_FILE)], 8),
    md(K.IMGMD, "t3-img-md"),
    code(K.SUMM, "t3-summary", [stream(K.SUMMARY_TXT + "\n")], 9),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("task_3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
with open("task3_summary.txt", "w", encoding="utf-8") as f:
    f.write(K.SUMMARY_TXT + "\n")
print("written", len(cells))
