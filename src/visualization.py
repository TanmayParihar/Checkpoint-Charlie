"""
Visualization Module
====================
Generates all figures for the BGP security simulation results.

Figures produced
----------------
1. detection_rates.png         – Detection rate by validator
2. false_positive_rates.png    – False positive rate by validator
3. processing_overhead.png     – Average processing time per announcement
4. security_cost_tradeoff.png  – Detection rate vs processing cost scatter
5. attack_type_breakdown.png   – Detection rate per attack type per validator
6. selective_hop_analysis.png  – Detection rate as a function of #hops verified
7. combined_summary.png        – 2×3 panel with all key metrics
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import config

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
VALIDATOR_ORDER = [
    "RPKI",
    "Origin + 1st Hop",
    "Origin + Last Hop",
    "Origin + 1st + Last",
    "Origin + 2 Hops + Last",
    "BGPsec (Full Path)",
    "Anomaly Detector",
    "Selective + Anomaly",
]
COLOR_MAP = {
    "RPKI":                     "#4878CF",
    "Origin + 1st Hop":         "#6ACC65",
    "Origin + Last Hop":        "#D65F5F",
    "Origin + 1st + Last":      "#B47CC7",
    "Origin + 2 Hops + Last":   "#C4AD66",
    "BGPsec (Full Path)":       "#77BEDB",
    "Anomaly Detector":         "#F28E2B",
    "Selective + Anomaly":      "#E15759",
}


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _ordered_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame rows by VALIDATOR_ORDER."""
    order = [v for v in VALIDATOR_ORDER if v in df.index]
    rest  = [v for v in df.index if v not in order]
    return df.reindex(order + rest)


# ── Individual figure functions ───────────────────────────────────────────────

def plot_detection_rates(summary: pd.DataFrame) -> None:
    df = _ordered_index(summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLOR_MAP.get(v, "#888888") for v in df.index]
    bars = ax.bar(range(len(df)), df["detection_rate"] * 100, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, val in zip(bars, df["detection_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("BGP Attack Detection Rate by Validation Strategy", fontweight="bold")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80% target")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "detection_rates.png")


def plot_false_positive_rates(summary: pd.DataFrame) -> None:
    df = _ordered_index(summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLOR_MAP.get(v, "#888888") for v in df.index]
    bars = ax.bar(range(len(df)), df["false_positive_rate"] * 100, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, df["false_positive_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate by Validation Strategy", fontweight="bold")
    fig.tight_layout()
    _save(fig, "false_positive_rates.png")


def plot_processing_overhead(summary: pd.DataFrame) -> None:
    df = _ordered_index(summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLOR_MAP.get(v, "#888888") for v in df.index]
    bars = ax.bar(range(len(df)), df["avg_time_ms"] * 1000, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, df["avg_time_ms"] * 1000):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}µs", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg. Processing Time (µs/announcement)")
    ax.set_title("Computational Overhead per BGP Announcement", fontweight="bold")
    fig.tight_layout()
    _save(fig, "processing_overhead.png")


def plot_security_cost_tradeoff(summary: pd.DataFrame) -> None:
    df = _ordered_index(summary)
    fig, ax = plt.subplots(figsize=(9, 6))

    for validator in df.index:
        row = df.loc[validator]
        color = COLOR_MAP.get(validator, "#888888")
        ax.scatter(
            row["avg_time_ms"] * 1000,
            row["detection_rate"] * 100,
            s=120, color=color, zorder=3, label=validator,
        )
        ax.annotate(
            validator,
            (row["avg_time_ms"] * 1000, row["detection_rate"] * 100),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color=color,
        )

    ax.set_xlabel("Avg. Processing Time (µs/announcement)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Security vs. Computational Cost Trade-off", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.02, 81, "80 % target", fontsize=8, color="gray")
    fig.tight_layout()
    _save(fig, "security_cost_tradeoff.png")


def plot_attack_type_breakdown(breakdown: pd.DataFrame) -> None:
    df = _ordered_index(breakdown)
    attack_types = list(df.columns)
    n_types = len(attack_types)
    n_validators = len(df)

    x = np.arange(n_validators)
    width = 0.8 / n_types

    fig, ax = plt.subplots(figsize=(12, 6))
    type_colors = sns.color_palette("Set2", n_types)

    for i, at in enumerate(attack_types):
        vals = df[at].fillna(0).values * 100
        offset = (i - n_types / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.95, label=at.replace("_", " ").title(),
               color=type_colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Detection Rate by Attack Type and Validator", fontweight="bold")
    ax.legend(title="Attack Type", fontsize=9)
    fig.tight_layout()
    _save(fig, "attack_type_breakdown.png")


def plot_selective_hop_analysis(summary: pd.DataFrame) -> None:
    """
    Line chart: detection rate vs. average processing time (cost proxy)
    for the selective-hop variants only (excludes anomaly detector).
    """
    hop_labels = [v for v in VALIDATOR_ORDER if v in summary.index and v != "Anomaly Detector"]
    df = _ordered_index(summary).loc[[v for v in hop_labels if v in summary.index]]

    fig, ax = plt.subplots(figsize=(9, 5))

    times = df["avg_time_ms"] * 1000
    rates = df["detection_rate"] * 100
    ax.plot(times, rates, marker="o", linewidth=2, markersize=9, color="#4878CF", zorder=3)

    for i, label in enumerate(df.index):
        ax.annotate(
            label, (times.iloc[i], rates.iloc[i]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("Avg. Processing Time (µs/announcement)", fontsize=11)
    ax.set_ylabel("Detection Rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[0], 81, " 80 % target", fontsize=8, color="gray")
    ax.set_title("Selective Hop Verification: Security vs. Cost Curve", fontweight="bold")
    fig.tight_layout()
    _save(fig, "selective_hop_analysis.png")


def plot_combined_summary(summary: pd.DataFrame, breakdown: pd.DataFrame) -> None:
    """6-panel combined overview figure."""
    df = _ordered_index(summary)
    colors = [COLOR_MAP.get(v, "#888888") for v in df.index]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BGP Routing Security – Simulation Results", fontsize=15, fontweight="bold", y=1.01)

    # Panel 1: Detection rate
    ax = axes[0, 0]
    ax.bar(range(len(df)), df["detection_rate"] * 100, color=colors, edgecolor="white")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Panel 2: FPR
    ax = axes[0, 1]
    ax.bar(range(len(df)), df["false_positive_rate"] * 100, color=colors, edgecolor="white")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("FPR (%)")
    ax.set_title("False Positive Rate")

    # Panel 3: Processing time
    ax = axes[0, 2]
    ax.bar(range(len(df)), df["avg_time_ms"] * 1000, color=colors, edgecolor="white")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Avg. Time (µs)")
    ax.set_title("Processing Overhead")

    # Panel 4: Security-cost scatter
    ax = axes[1, 0]
    for i, v in enumerate(df.index):
        ax.scatter(df.loc[v, "avg_time_ms"] * 1000, df.loc[v, "detection_rate"] * 100,
                   s=80, color=colors[i], zorder=3)
        ax.annotate(v, (df.loc[v, "avg_time_ms"] * 1000, df.loc[v, "detection_rate"] * 100),
                    xytext=(4, 2), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Avg. Time (µs)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Security vs. Cost")

    # Panel 5: Attack type heatmap
    ax = axes[1, 1]
    bd = _ordered_index(breakdown).fillna(0) * 100
    if not bd.empty:
        im = ax.imshow(bd.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(bd.columns)))
        ax.set_xticklabels([c.replace("_", "\n") for c in bd.columns], fontsize=8)
        ax.set_yticks(range(len(bd.index)))
        ax.set_yticklabels(bd.index, fontsize=7.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Detection %")
        for r in range(len(bd.index)):
            for c in range(len(bd.columns)):
                ax.text(c, r, f"{bd.values[r, c]:.0f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_title("Attack-Type Heatmap")

    # Panel 6: Security-cost ratio
    ax = axes[1, 2]
    ax.barh(range(len(df)), df["security_cost_ratio"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=7.5)
    ax.set_xlabel("Detection Rate / Avg. Time (ms⁻¹)")
    ax.set_title("Security-Cost Ratio (higher = better)")

    fig.tight_layout()
    _save(fig, "combined_summary.png")


def generate_all_figures(summary: pd.DataFrame, breakdown: pd.DataFrame) -> None:
    """Convenience function: generate every figure in one call."""
    print("\nGenerating figures...")
    plot_detection_rates(summary)
    plot_false_positive_rates(summary)
    plot_processing_overhead(summary)
    plot_security_cost_tradeoff(summary)
    plot_attack_type_breakdown(breakdown)
    plot_selective_hop_analysis(summary)
    plot_combined_summary(summary, breakdown)
    print(f"All figures saved to ./{config.RESULTS_DIR}/")
