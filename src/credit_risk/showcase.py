"""Create aggregate-only EDA and model assets for the public project page."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import DATA_PATH, METADATA_PATH, PROJECT_ROOT, REPORT_DIR, TARGET_COLUMN

SHOWCASE_DIR = PROJECT_ROOT / "static" / "project-assets"
PAYMENT_STATUS_LABELS = {
    -2: "No use",
    -1: "Paid early",
    0: "On time",
    1: "1 mo late",
    2: "2 mo late",
    3: "3 mo late",
    4: "4 mo late",
    5: "5 mo late",
    6: "6 mo late",
    7: "7 mo late",
    8: "8+ mo late",
}


def build_summary(data: pd.DataFrame) -> dict:
    """Return only aggregate EDA findings that are safe to publish."""
    correlations = (
        data.drop(columns=["ID"], errors="ignore")
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(10)
    )
    pay_zero = data.groupby("PAY_0")[TARGET_COLUMN].agg(["count", "mean"]).reset_index()
    pay_zero["label"] = pay_zero["PAY_0"].map(PAYMENT_STATUS_LABELS).fillna(pay_zero["PAY_0"].astype(str))
    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_ids": int(data["ID"].duplicated().sum()),
        "default_count": int(data[TARGET_COLUMN].sum()),
        "non_default_count": int((1 - data[TARGET_COLUMN]).sum()),
        "default_rate": float(data[TARGET_COLUMN].mean()),
        "limit_mean": float(data["LIMIT_BAL"].mean()),
        "limit_median": float(data["LIMIT_BAL"].median()),
        "age_mean": float(data["AGE"].mean()),
        "age_median": float(data["AGE"].median()),
        "target_correlations": [
            {"feature": feature, "correlation": float(value)} for feature, value in correlations.items()
        ],
        "pay_0_default_rates": [
            {
                "status": int(row.PAY_0),
                "label": row.label,
                "count": int(row.count),
                "default_rate": float(row.mean),
            }
            for row in pay_zero.itertuples(index=False)
        ],
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def create_eda_charts(data: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    """Create a small, readable EDA set without exposing individual records."""
    colors = ["#315ecb", "#82a6ff"]

    plt.figure(figsize=(6.5, 4))
    plt.bar(["No default", "Default"], [summary["non_default_count"], summary["default_count"]], color=colors)
    plt.title("Target balance: next-month default")
    plt.ylabel("Client records")
    for index, value in enumerate([summary["non_default_count"], summary["default_count"]]):
        plt.text(index, value + 350, f"{value:,}\n({value / summary['rows']:.1%})", ha="center")
    _save_figure(output_dir / "target_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(data["LIMIT_BAL"], bins=35, color="#315ecb", edgecolor="white")
    axes[0].axvline(summary["limit_median"], color="#b45309", linestyle="--", label="Median")
    axes[0].set(title="Credit-limit distribution", xlabel="Credit limit (NT$)", ylabel="Records")
    axes[0].legend()
    axes[1].hist(data["AGE"], bins=30, color="#0f766e", edgecolor="white")
    axes[1].axvline(summary["age_median"], color="#b45309", linestyle="--", label="Median")
    axes[1].set(title="Age distribution", xlabel="Age", ylabel="Records")
    axes[1].legend()
    _save_figure(output_dir / "profile_distributions.png")

    rates = pd.DataFrame(summary["pay_0_default_rates"])
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(rates["label"], rates["default_rate"] * 100, color="#0f766e")
    plt.title("Observed default rate by most recent repayment status (PAY_0)")
    plt.ylabel("Default rate (%)")
    plt.xticks(rotation=35, ha="right")
    for bar, value in zip(bars, rates["default_rate"] * 100):
        if value >= 30:
            plt.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.0f}%", ha="center", fontsize=8)
    _save_figure(output_dir / "recent_payment_status.png")

    correlations = pd.DataFrame(summary["target_correlations"]).sort_values("correlation")
    plt.figure(figsize=(8, 5))
    bar_colors = ["#315ecb" if value >= 0 else "#b45309" for value in correlations["correlation"]]
    plt.barh(correlations["feature"], correlations["correlation"], color=bar_colors)
    plt.title("Strongest raw linear associations with recorded default")
    plt.xlabel("Pearson correlation with target")
    _save_figure(output_dir / "target_correlations.png")


def create_model_comparison_chart(metadata: dict, output_dir: Path) -> None:
    comparison = metadata["model_comparison"]
    labels = [name.replace("_", " ").title() for name in comparison]
    scores = [comparison[name]["cv_roc_auc"] for name in comparison]
    plt.figure(figsize=(7.5, 4))
    bars = plt.bar(labels, scores, color=["#94a3b8", "#64748b", "#315ecb"])
    plt.ylim(0.5, max(scores) + 0.04)
    plt.ylabel("Mean cross-validation ROC-AUC")
    plt.title("Model selection on training folds")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.003, f"{score:.4f}", ha="center", fontsize=9)
    _save_figure(output_dir / "model_comparison.png")


def generate_showcase_assets(
    data_path: Path = DATA_PATH,
    metadata_path: Path = METADATA_PATH,
    output_dir: Path = SHOWCASE_DIR,
) -> dict:
    """Write public aggregate assets and copy verified evaluation plots for Flask."""
    data = pd.read_csv(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(data)
    summary["model"] = {
        "selected_model": metadata["selected_model"],
        "threshold": metadata["threshold"],
        "test_metrics": metadata["test_metrics"],
        "comparison": metadata["model_comparison"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_eda_charts(data, summary, output_dir)
    create_model_comparison_chart(metadata, output_dir)
    for source_name, target_name in {
        "evaluation_curves.png": "evaluation_curves.png",
        "threshold_analysis.png": "threshold_analysis.png",
        "shap_summary.png": "shap_summary.png",
    }.items():
        source = REPORT_DIR / source_name
        if source.exists():
            shutil.copy2(source, output_dir / target_name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate public aggregate EDA and model showcase assets.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=SHOWCASE_DIR)
    args = parser.parse_args()
    result = generate_showcase_assets(args.data, METADATA_PATH, args.output)
    print(json.dumps({"rows": result["rows"], "default_rate": result["default_rate"]}, indent=2))
