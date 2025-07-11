#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64, os
from pathlib import Path


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def main():
    # 1) locate files
    base = Path(__file__).parent
    csv_path = base / "card_transaction.v1.csv"
    out_html = base / "eda_report.html"

    # 2) load data
    df = pd.read_csv(csv_path)
    
    # 3) define categorical vs numeric
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # treat integer columns with few unique values as categorical
    for col in df.select_dtypes(include=["int"]).columns:
        if df[col].nunique() < 50:
            cat_cols.append(col)
    # drop duplicates
    cat_cols = list(dict.fromkeys(cat_cols))
    num_cols = df.select_dtypes(include=["float"]).columns.tolist()

    # 4) basic metrics
    total_rows      = len(df)
    num_users       = df["User"].nunique() if "User" in df.columns else None
    avg_hist_len    = df.groupby("User").size().mean() if "User" in df.columns else None

    # 5) start writing HTML
    html = ["<html><head><title>EDA Report</title></head><body>"]
    html.append(f"<h1>Exploratory Data Analysis Report</h1>")
    html.append(f"<h2>Dataset: {csv_path.name}</h2>")
    html.append("<h3>Basic Info</h3><ul>")
    html.append(f"<li>Total rows: {total_rows:,}</li>")
    if num_users is not None:
        html.append(f"<li>Unique users: {num_users:,}</li>")
        html.append(f"<li>Avg. history length: {avg_hist_len:,.2f}</li>")
    html.append("</ul>")

    # 6) data types & nulls
    html.append("<h3>Column Data Types & Missing Values</h3>")
    dt = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum()
    })
    html.append(dt.to_html(border=1))

    # 7) numeric descriptive stats
    if num_cols:
        html.append("<h3>Numeric Descriptive Statistics</h3>")
        html.append(df[num_cols].describe().to_html(border=1))

        html.append("<h3>Numeric Distributions</h3>")
        for col in num_cols:
            fig = plt.figure()
            df[col].hist(bins=30)
            plt.title(col)
            img = fig_to_base64(fig)
            html.append(f"<h4>{col}</h4>")
            html.append(f'<img src="data:image/png;base64,{img}"/>')

    # 8) categorical summaries
    if cat_cols:
        html.append("<h3>Categorical Summaries (Top 10 Counts)</h3>")
        for col in cat_cols:
            vc = df[col].value_counts().head(10)
            html.append(f"<h4>{col}</h4>")
            html.append(vc.to_frame("count").to_html(border=1,
                                   formatters={"count": lambda x: f"{int(x):,}"}))

    # 9) correlation
    html.append("<h3>Correlation Matrix (numeric)</h3>")
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        html.append(corr.to_html(border=1))
        fig = plt.figure(figsize=(6,6))
        plt.matshow(corr, fignum=fig.number)
        plt.xticks(range(len(num_cols)), num_cols, rotation=90)
        plt.yticks(range(len(num_cols)), num_cols)
        plt.title("Correlation Heatmap", y=1.15)
        img = fig_to_base64(fig)
        html.append(f'<img src="data:image/png;base64,{img}"/>')

    # 10) sliding window & stride sample counts (pivot table)
    if "User" in df.columns:
        html.append("<h3>Sliding Window Sample Counts</h3>")
        group_sizes = df.groupby("User").size()

        # Anchor window sizes
        window_sizes = [5, 10, 20, 50, 100, 200, 500, 1000]

        # A variety of stride strategies
        strides = [
            1,    # maximal overlap
            2,    # small step
            5,    # quarter-overlap for window=20
            10,   # half-overlap for window=20 or 100
            20,   # half-overlap for window=40
            25,   # quarter-overlap for window=100
            50,   # half-overlap for window=100 or full-overlap for window=100
            100,  # half-overlap for window=200 or no overlap for window=100
            200,  # no overlap for window=200
            500,  # no overlap for window=500
            1000  # no overlap for window=1000
        ]

        data = []
        for w in window_sizes:
            for s in strides:
                counts = ((group_sizes - w) // s + 1).clip(lower=0).sum()
                data.append({"window_size": w, "stride": s, "num_samples": int(counts)})
        sw_df = pd.DataFrame(data)
        pivot = sw_df.pivot(index="window_size", columns="stride", values="num_samples")
        # format with commas
        pivot_str = pivot.applymap(lambda x: f"{int(x):,}")
        html.append(pivot_str.to_html(border=1))

    # 11) random guessing accuracy for categorical features
    if cat_cols:
        html.append("<h3>Random Guessing Accuracy by Categorical Feature</h3>")
        rg_df = pd.DataFrame({
            "feature": cat_cols,
            "unique_count": [df[c].nunique() for c in cat_cols],
            "random_accuracy": [1/df[c].nunique() for c in cat_cols]
        })
        html.append(rg_df.to_html(index=False, border=1,
            formatters={
                "unique_count": lambda x: f"{int(x):,}",
                "random_accuracy": lambda x: f"{x:.4%}"
            }
        ))

    # 12) finish
    html.append("</body></html>")

    # write out
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"✅ EDA report written to {out_html}")

if __name__=="__main__":
    main()
