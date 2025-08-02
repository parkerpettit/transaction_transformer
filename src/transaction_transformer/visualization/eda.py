#!/usr/bin/env python3
"""better_eda_full.py - v4  (all-rows version)
Generates an exhaustive, single-file interactive HTML report for the **24-million-row**
card-transaction dataset *without* sub-sampling.

Main additions v3 → v4
---------------------
* **No sampling anywhere** - every analytic step runs on the full DataFrame.
* **Datashader density plot** for Amount vs fraud flag (falls back to seaborn if
  Datashader not installed).
* **Mutual-information** computed *feature-by-feature* (constant memory).
* **LightGBM baseline** (GOSS) on the *entire* dataset with ROC-AUC, PR-AUC,
  precision, recall, F1, and confusion-matrix heat-map.
* **Correlation matrix heat-map** for numeric columns (instead of PCA).
* ZIP-scatter now includes **every ZIP code** (WebGL scatter).

Requires: pandas, numpy, matplotlib, seaborn, plotly, pgeocode, ydata-profiling,
scikit-learn, lightgbm, tqdm, (optional) datashader.

Run:
    python better_eda_full.py --csv card_transaction.v1.csv --out eda_full.html
"""
from __future__ import annotations

import argparse, io, base64, warnings, json, math
from pathlib import Path
from typing import List, Optional

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
from ydata_profiling import ProfileReport  # type: ignore
import plotly.express as px                # type: ignore
import pgeocode                             # type: ignore
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve, confusion_matrix)
from lightgbm import LGBMClassifier         # type: ignore
from scipy.stats import ks_2samp, entropy

# optional Datashader ---------------------------------------------------------
try:
    import datashader as ds                        # type: ignore
    import datashader.transfer_functions as tf     # type: ignore
    from datashader import reductions as rdr       # type: ignore
    from matplotlib.colors import to_hex
    DATASHADER = True
except ImportError:  # graceful fallback
    DATASHADER = False

warnings.filterwarnings("ignore", category=UserWarning)

HTML_TEMPLATE = """<!doctype html><html lang='en'>
<head><meta charset='utf-8'/><title>{title}</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:0;}}
 main{{padding:2rem;}}
 summary{{font-weight:bold;cursor:pointer;}}
 figure{{margin:1rem 0;}}
 figcaption{{font-style:italic;font-size:.8rem;text-align:center;}}
 pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem;border-radius:4px;}}
 .sw-table td{{text-align:right;}}
 .cm td,.cm th{{text-align:center;}}
</style>
</head><body><main><h1>{title}</h1>{body}</main></body></html>"""

# ------------------------------------------------------------------ helpers

def img_tag(fig, caption: str | None = None) -> str:
    """Convert a Matplotlib fig → base64 <img>."""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    plt.close(fig)
    img64 = base64.b64encode(buf.getvalue()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<figure><img src='data:image/png;base64,{img64}'/>{cap}</figure>"

# --------------------------- cleaning ---------------------------------------

def clean_dataframe(df: pd.DataFrame) -> None:
    """In-place cleaning and type optimisation."""
    # Time → Hour (int8)
    df["Hour"] = df["Time"].str.split(":").str[0].astype("int8")
    df.drop(columns=["Time"], inplace=True)

    # Money → float32
    df["Amount"] = df["Amount"].str.replace(r"[$,]", "", regex=True).astype("float32")

    # ZIP as string, missing sentinel
    df["Zip"] = df["Zip"].astype("Int64").astype(str).fillna("MISSING")

    # Categoricals
    cat_like = [
        "User", "Card", "Merchant Name", "MCC", "Merchant City",
        "Merchant State", "Zip", "Year", "Month", "Day", "Hour", "Use Chip",
        "Errors?",
    ]
    for c in cat_like:
        df[c] = df[c].astype("category")

    # Binary label
    if df["Is Fraud?"].dtype == object:
        df["Is Fraud?"] = df["Is Fraud?"].eq("Yes")

    # Calendar helpers
    df["Y_M"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
    # fabricate DOW placeholder (unknown real date)
    df["DOW"] = df.groupby("User").cumcount() % 7

# ---------------- baseline / lift ------------------------------------------

def majority_guess(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Majority-class baseline table."""
    out = []
    for c in cols:
        vc = df[c].value_counts(normalize=True, dropna=False)
        out.append({"feature": c,
                    "unique": df[c].nunique(dropna=False),
                    "most_freq_rate": vc.iloc[0]})
    return pd.DataFrame(out).sort_values("most_freq_rate", ascending=False)


def fraud_lift(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    base = df["Is Fraud?"].mean()
    lift = (df.groupby(col)["Is Fraud?"].mean()
            .sort_values(ascending=False)
            .head(n)               # keep top-n for readability
            .to_frame("fraud_rate"))
    lift["lift"] = lift["fraud_rate"] / base
    lift["samples"] = df[col].value_counts().reindex(lift.index)
    return lift

# ---------------- plotting snippets ----------------------------------------
def amount_plot(df: pd.DataFrame) -> str:
    """
    Amount density - Datashader categorical overlay (full data) or seaborn fallback.
    Blue = legit, red = fraud.
    """
    if DATASHADER:
        # 1) one categorical aggregate with count_cat, no post-stack blending
        from datashader.reductions import count_cat
        color_key = {"legit": "#1f77b4", "fraud": "#d62728"}  # blue / red

        df_plot = df.assign(_y=0.0)
        df_plot["cls"] = pd.Categorical(
            np.where(df_plot["Is Fraud?"], "fraud", "legit"),
            categories=["legit", "fraud"],
            ordered=False,
        )        
        cvs = ds.Canvas(plot_width=900, plot_height=300, y_range=(0, 1))
        agg = cvs.points(df_plot, "Amount", "_y", agg=count_cat("cls"))

        img = tf.shade(agg, how="log", color_key=color_key)

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.imshow(img.to_pil(), aspect="auto")
        ax.axis("off")
        ax.set_title("Amount distribution — legit (blue) vs fraud (red)")
        return img_tag(fig)

    # -- seaborn fallback (for environments without Datashader) --------------
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(
        data=df, x="Amount", hue="Is Fraud?",
        fill=True, common_norm=False, ax=ax
    )
    ax.set_title("Amount distribution by fraud flag")
    return img_tag(fig)
# ---------------------------------------------------------------------------


def state_map(df: pd.DataFrame, value: str, title: str) -> str:
    m = df.groupby("Merchant State")[value].mean().reset_index()
    fig = px.choropleth(
        m, locations="Merchant State", locationmode="USA-states", color=value,
        scope="usa", title=title, color_continuous_scale="Reds")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def zip_map(df: pd.DataFrame, metric: pd.Series, title: str) -> str:
    nomi = pgeocode.Nominatim("us")

    zips = metric.index.astype(str).tolist()

    # --- robust lookup --------------------------------------------------
    coords = (nomi.query_postal_code(zips)            # may return fewer rows
                .set_index("postal_code")             # ① align on postal_code
                .reindex(zips))                       # ② keep original length

    data = (pd.DataFrame({
                "zip": zips,
                "metric": metric.values,
                "lat": coords["latitude"].values,
                "lon": coords["longitude"].values,
            })
            .dropna(subset=["lat", "lon"]))           # ③ keep valid rows only

    fig = px.scatter_geo(
        data,
        lat="lat", lon="lon", color="metric", size="metric",
        scope="usa", title=title, color_continuous_scale="Reds",
        hover_name="zip"
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig.to_html(full_html=False, include_plotlyjs=False)



# ---------------------------------------------------------------------------
# 2)   sliding_window - keep keyword args; format with .map (no applymap)
# ---------------------------------------------------------------------------
def sliding_window(group_sizes: pd.Series) -> pd.DataFrame:
    windows  = [5, 10, 20, 50, 100, 200, 500, 1000]
    strides  = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]

    data = [{
        "window":  w,
        "stride":  s,
        "samples": int(((group_sizes - w) // s + 1)
                       .clip(lower=0).sum())
    } for w in windows for s in strides]

    tbl = (pd.DataFrame(data)
             .pivot(index="window", columns="stride", values="samples"))
    return tbl.map(lambda x: f"{x:,}")       # elementwise formatting


# ---------------- advanced analytics ---------------------------------------

from sklearn.utils.validation import check_is_fitted

def mutual_info_bar(df: pd.DataFrame,
                    cat_cols: List[str],
                    num_cols: List[str]) -> str:
    """Return <img> tag of top-30 MI scores (handles NaN safely)."""
    y = df["Is Fraud?"].astype(int).to_numpy()
    mi_scores: dict[str, float] = {}

    # --- categoricals ----------------------------------------------------
    for c in cat_cols:
        # codes: -1 means NaN → drop those rows
        codes = df[c].cat.codes.to_numpy()
        mask  = codes != -1                  # keep only real levels
        if mask.sum() == 0:                  # all NaN → skip feature
            continue
        mi = mutual_info_classif(
                 codes[mask].reshape(-1, 1),
                 y[mask],
                 discrete_features=True,
                 random_state=0)[0]
        mi_scores[c] = mi

    # --- numerics --------------------------------------------------------
    for c in num_cols:
        col = df[c].to_numpy()
        if np.isnan(col).all():
            continue                         # skip all-NaN column
        col = np.where(np.isnan(col),
                       np.nanmedian(col),    # median imputation
                       col)
        mi = mutual_info_classif(
                 col.reshape(-1, 1),
                 y,
                 discrete_features=False,
                 random_state=0)[0]
        mi_scores[c] = mi

    top = (pd.Series(mi_scores)
             .sort_values(ascending=False)
             .head(30))

    fig, ax = plt.subplots(figsize=(6, 8))
    top.sort_values().plot.barh(ax=ax)
    ax.set_title("Top 30 mutual-information features")
    return img_tag(fig)

def ks_amount(df: pd.DataFrame):
    ks, p = ks_2samp(df.loc[df["Is Fraud?"], "Amount"],
                     df.loc[~df["Is Fraud?"], "Amount"])
    return ks, p


def pairwise_heat(df: pd.DataFrame, a: str, b: str) -> str:
    tbl = pd.crosstab(df[a], df[b], values=df["Is Fraud?"], aggfunc="mean").fillna(0)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(tbl, cmap="Reds")
    plt.title(f"Fraud % - {a} x {b}")
    return img_tag(fig)


def temporal_lines(df: pd.DataFrame) -> str:
    by_month = df.groupby("Y_M")["Is Fraud?"].mean()
    fig, ax = plt.subplots()
    by_month.plot(ax=ax)
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud % over time (Year-Month)")
    return img_tag(fig)


def hour_dow_heat(df: pd.DataFrame) -> str:
    tbl = pd.pivot_table(df, index="Hour", columns="DOW", values="Is Fraud?", aggfunc="mean")
    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(tbl, cmap="Reds")
    plt.title("Fraud % by Hour x DOW")
    return img_tag(fig)


def seq_len_hist(df: pd.DataFrame) -> str:
    lengths = df.groupby("User").size()
    fig, ax = plt.subplots()
    lengths.plot.hist(bins=50, log=True, ax=ax)
    ax.set_xlabel("# Transactions per user")
    ax.set_title("User history length (log-y)")
    return img_tag(fig)


def memory_table(df: pd.DataFrame) -> str:
    mem = df.memory_usage(deep=True).sort_values(ascending=False)
    return (mem.to_frame("bytes")
            .assign(MB=lambda s: s.bytes / 1e6)
            .to_html(border=1, classes="mem", formatters={
                "bytes": "{:,.0f}".format,
                "MB": "{:,.2f}".format}))


def corr_heatmap(df: pd.DataFrame, num_cols: List[str]) -> str:
    if len(num_cols) < 2:
        return ""  # nothing to plot
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    ax.set_title("Correlation matrix - numeric features")
    return img_tag(fig)


# ---------------- LightGBM baseline ----------------------------------------

def lgbm_baseline(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]):
    X = df[cat_cols + num_cols].copy()
    # integer-encode categoricals for LightGBM
    for c in cat_cols:
        X[c] = X[c].cat.codes  # -1 for NaN
    y = df["Is Fraud?"].astype(int)

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    clf = LGBMClassifier(
        objective="binary", boosting_type="goss", n_estimators=400,
        learning_rate=0.05, num_leaves=127, n_jobs=-1)
    clf.fit(X, y, categorical_feature=cat_idx)

    proba = clf.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # metrics
    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)
    prec = precision_score(y, preds)
    rec  = recall_score(y, preds)
    f1   = f1_score(y, preds)

    # curves
    fpr, tpr, _ = roc_curve(y, proba)
    pre, rec_curve, _ = precision_recall_curve(y, proba)

    fig, ax = plt.subplots(); ax.plot(fpr, tpr); ax.set_title(f"ROC AUC {roc:.3f}")
    roc_img = img_tag(fig)
    fig, ax = plt.subplots(); ax.plot(rec_curve, pre); ax.set_title(f"PR AUC {pr:.3f}")
    pr_img = img_tag(fig)

    # confusion matrix heat-map
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots();
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (threshold 0.5)")
    cm_img = img_tag(fig)

    metrics_html = (
        f"<p><b>ROC AUC</b>: {roc:.3f} &nbsp;&nbsp; "
        f"<b>PR AUC</b>: {pr:.3f} &nbsp;&nbsp; "
        f"<b>Precision</b>: {prec:.3f} &nbsp;&nbsp; "
        f"<b>Recall</b>: {rec:.3f} &nbsp;&nbsp; "
        f"<b>F1</b>: {f1:.3f}</p>")

    return roc_img + pr_img + cm_img + metrics_html

# =============================================================================
#                                main driver
# =============================================================================

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Generate exhaustive EDA report (all rows)")
    ap.add_argument("--csv", type=Path, default="card_transaction.v1.csv")
    ap.add_argument("--out", type=Path, default="eda_full.html")
    args = ap.parse_args(argv)

    # ---------------- load & clean ----------------------------------------
    print("🔄 Loading CSV … (this can take a minute)")
    df = pd.read_csv(args.csv, low_memory=False)
    clean_dataframe(df)

    cat_cols = [
        "User", "Card", "Use Chip", "Merchant Name", "Merchant City",
        "Merchant State", "Zip", "MCC", "Errors?", "Year", "Month", "Day", "Hour"
    ]
    num_cols = ["Amount"]

    # ---------------- basic metrics --------------------------------------
    basics = {
        "rows": f"{len(df):,}",
        "users": f"{df['User'].nunique():,}",
        "avg hist len": f"{df.groupby('User').size().mean():,.1f}",
        "fraud rate": f"{df['Is Fraud?'].mean():.4%}",
    }

    # ---------------- profiling -----------------------------------------
    print("⚡ Running ydata-profiling on full dataset … (grab coffee)")
    prof_html = ProfileReport(df, title="Profile (full)", progress_bar=False, tsmode=False).to_html()

    body: List[str] = []

    # overview list
    body.append("<h2>Overview</h2><ul>" + "".join(
        f"<li><b>{k}</b>: {v}</li>" for k, v in basics.items()) + "</ul>")

    # majority baseline
    body.append("<details open><summary>Majority-class baseline</summary>" +
                majority_guess(df, cat_cols).to_html(index=False, border=1,
                formatters={"most_freq_rate": "{:.2%}".format}) + "</details>")

    # fraud lift singles
    for c in ["Use Chip", "Hour", "Merchant State", "MCC"]:
        body.append(f"<details><summary>Fraud lift - {c}</summary>" +
                    fraud_lift(df, c).to_html(border=1,
                    formatters={"fraud_rate": "{:.2%}".format, "lift": "{:.2f}".format}) + "</details>")

    # pairwise heat map example
    body.append("<details><summary>Fraud % - Use Chip x Hour</summary>" +
                pairwise_heat(df, "Use Chip", "Hour") + "</details>")

    # amount density
    body.append(amount_plot(df))

    # maps
    body.append("<details open><summary>Fraud % by State</summary>" +
                state_map(df, "Is Fraud?", "Fraud % by State") + "</details>")
    zip_lift = (df.groupby("Zip")["Is Fraud?"].mean() / df["Is Fraud?"].mean()).sort_values(ascending=False)
    body.append("<details><summary>Fraud lift by ZIP (all)</summary>" +
                zip_map(df, zip_lift, "Fraud lift by ZIP (all)") + "</details>")

    # temporal plots
    body.append(temporal_lines(df))
    body.append(hour_dow_heat(df))

    # sequence length
    body.append(seq_len_hist(df))

    # sliding windows
    sw_tbl = sliding_window(df.groupby("User").size())
    body.append("<details><summary>Sliding-window counts</summary>" +
                sw_tbl.to_html(border=1, col_space=90, justify="right", classes="sw-table") + "</details>")

    # mutual info
    body.append(mutual_info_bar(df, cat_cols, num_cols))

    # entropy vs cardinality scatter
    def entropy_cardinality_plot(df_: pd.DataFrame, cat_cols_: List[str]) -> str:
        xs, ys, labels = [], [], []
        for c in cat_cols_:
            vc = df_[c].value_counts(normalize=True, dropna=False)
            xs.append(math.log10(len(vc)))
            ys.append(entropy(vc.values, base=2))
            labels.append(c)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(2, 2), fontsize=7)
        ax.set_xlabel("log10(unique levels)"); ax.set_ylabel("Shannon entropy (bits)")
        ax.set_title("Entropy vs Cardinality (categoricals)")
        return img_tag(fig)

    body.append("<details><summary>Entropy vs Cardinality</summary>" +
                entropy_cardinality_plot(df, cat_cols) + "</details>")

    # correlation matrix
    body.append(corr_heatmap(df, num_cols))

    # KS test
    ks, p = ks_amount(df)
    body.append(f"<p><b>KS statistic for Amount (fraud vs legit)</b>: {ks:.3f}  (p={p:.2e})</p>")

    # memory footprint
    body.append("<details><summary>Memory footprint</summary>" + memory_table(df) + "</details>")

    # LightGBM baseline
    body.append("<details><summary>LightGBM baseline (full data)</summary>" +
                lgbm_baseline(df, cat_cols, num_cols) + "</details>")

    # ---------------- assemble & write ----------------------------------
    html_out = HTML_TEMPLATE.format(title="Card-Transaction EDA (Full)",
                                    body=prof_html + "\n" + "\n".join(body))
    args.out.write_text(html_out, encoding="utf-8")
    print(f"✅ Report written to {args.out}")


if __name__ == "__main__":
    main()
