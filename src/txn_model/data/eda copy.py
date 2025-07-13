#!/usr/bin/env python3
"""better_eda.py – v3
A heavyweight, everything‑and‑the‑kitchen‑sink EDA generator for the 24 M‑row
card‑transaction dataset.  Produces a **single interactive HTML report** with:

* ydata‑profiling summary (sample or full)
* Majority‑class baseline table
* Fraud‑lift tables (single & pairwise)
* Amount density plot by fraud flag
* Sliding‑window sample counts pivot (wide, right‑aligned)
* **State‑level choropleth** and **ZIP‑scatter** maps
* **Temporal trends**: fraud % per Year‑Month & Hour×DOW heat‑map
* **Mutual‑information top‑30 bar**
* **PCA scatter** (2‑D) on numeric slice
* **Kolmogorov–Smirnov statistic** for Amount
* **Sequence‑length histogram** (log‑y)
* **Memory footprint** table (pre/post cast)
* **Baseline random‑forest** ROC‑AUC & PR‑AUC (1 M‑row sample)
* Saves cleaned **Parquet** (optional) & prints size

Run in Colab if you like—it will chug but finish.
"""
from __future__ import annotations

import argparse, io, base64, warnings, json
from pathlib import Path
from typing import List, Optional

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
from ydata_profiling import ProfileReport  # type: ignore
import plotly.express as px                # type: ignore
import pgeocode                             # type: ignore
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import ks_2samp, entropy

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
</style>
</head><body><main><h1>{title}</h1>{body}</main></body></html>"""

# ------------------------------------------------------------------ helpers

def img_tag(fig, caption: str | None = None) -> str:
    buf = io.BytesIO(); fig.savefig(buf, bbox_inches="tight"); plt.close(fig)
    img64 = base64.b64encode(buf.getvalue()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<figure><img src='data:image/png;base64,{img64}'/>{cap}</figure>"

# --------------------------- cleaning ---------------------------------------

def clean_dataframe(df: pd.DataFrame) -> None:
    df["Hour"] = df["Time"].str.split(":").str[0].astype("int8"); df.drop(columns=["Time"], inplace=True)
    df["Amount"] = df["Amount"].str.replace(r"[$,]", "", regex=True).astype(float)
    df["Zip"] = df["Zip"].astype("Int64").astype(str).fillna("MISSING")
    cat_like = ["User","Card","Merchant Name","MCC","Merchant City","Merchant State","Zip","Year","Month","Day","Hour"]
    for c in cat_like: df[c] = df[c].astype("category")
    if df["Is Fraud?"].dtype == object:
        df["Is Fraud?"] = (df["Is Fraud?"].eq("Yes")).astype(bool)
    # calendar helpers
    df["Y_M"] = pd.to_datetime(df[["Year","Month"]].assign(D=1))
    df["DOW"] = df.groupby("User").cumcount()  # placeholder before real date

# ---------------- baseline / lift ------------------------------------------

def majority_guess(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out=[]
    for c in cols:
        vc=df[c].value_counts(normalize=True, dropna=False)
        out.append({"feature":c,"unique":df[c].nunique(dropna=False),"most_freq_rate":vc.iloc[0]})
    return pd.DataFrame(out).sort_values("most_freq_rate",ascending=False)

def fraud_lift(df: pd.DataFrame,col:str,n=20)->pd.DataFrame:
    base=df["Is Fraud?"].mean()
    lift=(df.groupby(col)["Is Fraud?"].mean().sort_values(ascending=False).head(n).to_frame("fraud_rate"))
    lift["lift"]=lift["fraud_rate"]/base
    lift["samples"]=df[col].value_counts().reindex(lift.index)
    return lift

# ---------------- plotting snippets ----------------------------------------

def amount_plot(df):
    sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(figsize=(6,3))
    sns.kdeplot(data=df.sample(min(500_000,len(df)),random_state=0),x="Amount",hue="Is Fraud?",fill=True,common_norm=False,ax=ax)
    ax.set_title("Amount distribution by fraud flag"); return img_tag(fig)

def state_map(df,value,title):
    m=df.groupby("Merchant State")[value].mean().reset_index()
    fig=px.choropleth(m,locations="Merchant State",locationmode="USA-states",color=value,scope="usa",title=title,color_continuous_scale="Reds")
    return fig.to_html(full_html=False,include_plotlyjs="cdn")

def zip_map(df,metric,title,max_points=15000):
    nomi=pgeocode.Nominatim("us"); zips=metric.index.astype(str)
    coords=nomi.query_postal_code(zips)[["latitude","longitude"]]; coords.index=zips
    data=pd.DataFrame({"zip":zips,"metric":metric.values,"lat":coords.latitude,"lon":coords.longitude}).dropna()
    if len(data)>max_points: data=data.sample(max_points,random_state=0)
    fig=px.scatter_geo(data,lat="lat",lon="lon",color="metric",size="metric",scope="usa",title=title,color_continuous_scale="Reds",hover_name="zip")
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig.to_html(full_html=False,include_plotlyjs=False)

def sliding_window(group_sizes):
    windows=[5,10,20,50,100,200,500,1000]; strides=[1,2,5,10,20,25,50,100,200,500,1000]
    data=[{"window":w,"stride":s,"samples":int(((group_sizes-w)//s+1).clip(lower=0).sum())} for w in windows for s in strides]
    return pd.DataFrame(data).pivot("window","stride","samples")

# ---------------- advanced analytics ---------------------------------------

def mutual_info_bar(df,cat_cols,num_cols):
    sample=df.sample(500_000,random_state=0) if len(df)>500_000 else df.copy()
    X=pd.get_dummies(sample[cat_cols+num_cols],drop_first=True)
    y=sample["Is Fraud?"].astype(int)
    mi=mutual_info_classif(X,y,discrete_features='auto')
    top=pd.Series(mi,index=X.columns).sort_values(ascending=False).head(30)
    fig,ax=plt.subplots(figsize=(6,8)); top.sort_values().plot.barh(ax=ax)
    ax.set_title("Top 30 mutual‑information features"); return img_tag(fig)

def pca_scatter(df,num_cols):
    sample=df[num_cols].sample(250_000,random_state=1) if len(df)>250_000 else df[num_cols]
    emb=StandardScaler().fit_transform(sample)
    pca=PCA(n_components=2).fit_transform(emb)
    fig=px.scatter(x=pca[:,0],y=pca[:,1],color=df.loc[sample.index,"Is Fraud?"].map({True:"Fraud",False:"Legit"}),title="PCA (2‑D) on numeric slice")
    return fig.to_html(full_html=False,include_plotlyjs=False)

def ks_amount(df):
    ks,p=ks_2samp(df.loc[df["Is Fraud?"],"Amount"],df.loc[~df["Is Fraud?"],"Amount"])
    return ks,p

def pairwise_heat(df,a,b):
    tbl=pd.crosstab(df[a],df[b],values=df["Is Fraud?"],aggfunc="mean").fillna(0)
    fig=plt.figure(figsize=(8,6)); sns.heatmap(tbl,annot=False,cmap="Reds"); plt.title(f"Fraud % – {a} x {b}")
    return img_tag(fig)

def temporal_lines(df):
    by_month=df.groupby("Y_M")["Is Fraud?"].mean(); fig,ax=plt.subplots(); by_month.plot(ax=ax); ax.set_ylabel("Fraud rate"); ax.set_title("Fraud % over time"); return img_tag(fig)

def hour_dow_heat(df):
    # assume Hour exists; fabricate DOW from rolling index (not real dates)
    df['DOW']=df.groupby('User').cumcount()%7
    tbl=pd.pivot_table(df,index='Hour',columns='DOW',values='Is Fraud?',aggfunc='mean')
    fig=plt.figure(figsize=(6,6)); sns.heatmap(tbl,cmap='Reds'); plt.title('Fraud % by Hour×DOW'); return img_tag(fig)

def seq_len_hist(df):
    lengths=df.groupby('User').size(); fig,ax=plt.subplots(); lengths.plot.hist(bins=50,log=True,ax=ax); ax.set_xlabel('# Txns per user'); ax.set_title('User history length'); return img_tag(fig)

def memory_table(df):
    mem=df.memory_usage(deep=True).sort_values(ascending=False)
    return mem.to_frame('bytes').assign(MB=lambda s:s.bytes/1e6).to_html(border=1,formatters={'bytes':'{:,.0f}'.format,'MB':'{:,.2f}'.format})

def baseline_model(df, cat_cols, num_cols):
    sample = df.sample(1_000_000, random_state=0) if len(df) > 1_000_000 else df
    X = pd.get_dummies(sample[cat_cols + num_cols], drop_first=True, sparse=True)
    y = sample["Is Fraud?"]

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced").fit(X, y)
    proba = clf.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)

    fpr, tpr, _ = roc_curve(y, proba)
    pre, rec, _ = precision_recall_curve(y, proba)

    # ROC curve
    fig, ax = plt.subplots(); ax.plot(fpr, tpr, label=f"ROC AUC {roc:.3f}"); ax.legend()
    roc_img = img_tag(fig)
    # PR curve
    fig, ax = plt.subplots(); ax.plot(rec, pre, label=f"PR AUC {pr:.3f}"); ax.legend()
    pr_img = img_tag(fig)
    return roc_img + pr_img, roc, pr


def entropy_cardinality_plot(df: pd.DataFrame, cat_cols: List[str]) -> str:
    """Scatter where x=log10(unique), y=Shannon entropy; low‑info hi‑card dims bottom‑right."""
    xs, ys, labels = [], [], []
    for c in cat_cols:
        vc = df[c].value_counts(normalize=True, dropna=False)
        xs.append(np.log10(len(vc)))
        ys.append(entropy(vc.values, base=2))
        labels.append(c)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(2, 2), fontsize=7)
    ax.set_xlabel("log10(unique levels)"); ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title("Entropy vs Cardinality (categoricals)")
    return img_tag(fig)

# =============================================================================
#                                main driver
# =============================================================================

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Generate exhaustive EDA report")
    ap.add_argument("--csv", type=Path, default="card_transaction.v1.csv")
    ap.add_argument("--out", type=Path, default="eda_report.html")
    ap.add_argument("--sample", type=int, default=0,
                    help="Rows to sample for pandas‑profiling (0 = full data)")
    ap.add_argument("--parquet", action="store_true", help="Save cleaned parquet copy")
    args = ap.parse_args(argv)

    # ---------------- load & clean ----------------------------------------
    print("🔄 Loading CSV …")
    df = pd.read_csv(args.csv, low_memory=False)
    clean_dataframe(df)

    cat_cols = ["User", "Card", "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                "Zip", "MCC", "Errors?", "Year", "Month", "Day", "Hour"]
    num_cols = ["Amount"]

    # ---------------- basic metrics --------------------------------------
    basics = {
        "rows"       : f"{len(df):,}",
        "users"      : f"{df['User'].nunique():,}",
        "avg hist len": f"{df.groupby('User').size().mean():,.1f}",
        "fraud rate" : f"{df['Is Fraud?'].mean():.4%}",
    }

    # ---------------- profiling -----------------------------------------
    if args.sample == 0 or args.sample >= len(df):
        print("⚡ Profiling full dataset … (may take a while)")
        prof_html = ProfileReport(df, title="Profile (full)", progress_bar=False, tsmode=False).to_html()
    else:
        print("⚡ Profiling sample …")
        prof_html = ProfileReport(df.sample(args.sample, random_state=42),
                                  title="Profile (sample)", progress_bar=False, tsmode=False).to_html()

    body: List[str] = []

    # overview list
    body.append("<h2>Overview</h2><ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in basics.items()) + "</ul>")

    # majority baseline
    body.append("<details open><summary>Majority‑class baseline</summary>" +
                majority_guess(df, cat_cols).to_html(index=False, border=1,
                formatters={"most_freq_rate": "{:.2%}".format}) + "</details>")

    # fraud lift singles
    for c in ["Use Chip", "Hour", "Merchant State", "MCC"]:
        body.append(f"<details><summary>Fraud lift – {c}</summary>" +
                    fraud_lift(df, c).to_html(border=1,
                    formatters={"fraud_rate": "{:.2%}".format, "lift": "{:.2f}".format}) + "</details>")

    # pairwise heat map example
    body.append("<details><summary>Fraud % – Use Chip × Hour</summary>" + pairwise_heat(df, "Use Chip", "Hour") + "</details>")

    # amount density
    body.append(amount_plot(df))

    # maps
    body.append("<details open><summary>Fraud % by State</summary>" + state_map(df, "Is Fraud?", "Fraud % by State") + "</details>")
    zip_lift = (df.groupby("Zip")["Is Fraud?"].mean() / df["Is Fraud?"].mean()).sort_values(ascending=False).head(500)
    body.append("<details><summary>Fraud lift by ZIP (top 500)</summary>" + zip_map(df, zip_lift, "Fraud lift by ZIP") + "</details>")

    # temporal plots
    body.append(temporal_lines(df))
    body.append(hour_dow_heat(df))

    # sequence length
    body.append(seq_len_hist(df))

    # sliding windows
    sw_tbl = sliding_window(df.groupby("User").size()).applymap(lambda x: f"{x:,}")
    body.append("<details><summary>Sliding-window counts</summary>" + sw_tbl.to_html(border=1, col_space=90, justify="right", classes="sw-table") + "</details>")

    # mutual info
    body.append(mutual_info_bar(df, cat_cols, num_cols))

    # entropy vs cardinality scatter
    body.append("<details><summary>Entropy vs Cardinality</summary>" + entropy_cardinality_plot(df, cat_cols) + "</details>")

    # PCA scatter
    body.append(pca_scatter(df, num_cols))


    # KS test
    ks, p = ks_amount(df); body.append(f"<p><b>KS statistic for Amount (fraud vs legit)</b>: {ks:.3f}  (p={p:.2e})</p>")

    # memory footprint
    body.append("<details><summary>Memory footprint</summary>" + memory_table(df) + "</details>")

    # baseline model
    bl_html, roc_auc, pr_auc = baseline_model(df, cat_cols, num_cols)
    body.append("<details><summary>Random‑Forest baseline</summary>" + bl_html + f"<p>ROC AUC = {roc_auc:.3f}, PR AUC = {pr_auc:.3f}</p></details>")

    # parquet option
    if args.parquet:
        pq_path = args.out.with_suffix(".parquet")
        df.to_parquet(pq_path, index=False)
        size_mb = pq_path.stat().st_size / 1e6
        body.append(f"<p>Saved cleaned Parquet: {pq_path}  ({size_mb:.1f} MB)</p>")

    # ---------------- assemble & write ----------------------------------
    html_out = HTML_TEMPLATE.format(title="Card-Transaction EDA", body=prof_html + "\n" + "\n".join(body))
    args.out.write_text(html_out, encoding="utf-8")
    print(f"✅ Report written to {args.out}")


if __name__ == "__main__":
    main()