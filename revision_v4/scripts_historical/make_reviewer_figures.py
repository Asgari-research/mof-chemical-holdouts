from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision" / "figures"
DATA = ROOT / "results" / "figures" / "main" / "figure_data_csv"
SPLITS = ["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped"]
SPLIT_LABELS = ["Random", "Geometry", "Metal", "Functional", "Linker", "Topology"]
TARGETS = ["co2_0015bar", "co2_015bar", "ch4_58bar", "ch4_65bar"]
TARGET_LABELS = ["CO$_2$\n0.015 bar", "CO$_2$\n0.15 bar", "CH$_4$\n5.8 bar", "CH$_4$\n65 bar"]
COLORS = {"compact_geometry": "#8492a2", "enriched_interpretable": "#3a6393"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def figure3() -> None:
    df = pd.read_csv(DATA / "figure3_v8_target_split_sensitivity_polished_plot_data.csv")
    df = df[df["source"] == "mean_r2"]
    matrix = df.pivot(index="split_family", columns="target_slug", values="mean_r2").loc[SPLITS, TARGETS].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TARGETS)), TARGET_LABELS)
    ax.set_yticks(range(len(SPLITS)), SPLIT_LABELS)
    ax.set_title("Mean $R^2$ across validation strategies", fontweight="bold", pad=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontweight="bold",
                    color="white" if matrix[i,j] >= 0.58 else "#1f2937")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cb.set_label("Mean $R^2$")
    fig.savefig(OUT / "figure3_revised_single_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def figure4() -> None:
    df = pd.read_csv(DATA / "figure4_v8_screening_consequences_polished_plot_data.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3))
    x = np.arange(len(SPLITS))
    width = 0.32
    for offset, descriptor in [(-width / 2, "compact_geometry"), (width / 2, "enriched_interpretable")]:
        d = df[df["descriptor_family"] == descriptor].set_index("split_family").loc[SPLITS]
        label = "Compact geometry" if descriptor == "compact_geometry" else "Enriched interpretable"
        axes[0].bar(x + offset, 100 * d["top_5pct_overlap_mean"], width,
                    yerr=100 * d["top_5pct_overlap_sem"], capsize=3, color=COLORS[descriptor], label=label)
        axes[1].bar(x + offset, d["top_5pct_enrichment_mean"], width,
                    yerr=d["top_5pct_enrichment_sem"], capsize=3, color=COLORS[descriptor], label=label)
    axes[0].set_ylabel("Top-5% recovery (%)")
    axes[0].set_ylim(0, 70)
    axes[0].set_yticks(np.arange(0, 71, 10))
    axes[0].set_title("Candidate recovery", fontweight="bold")
    axes[1].set_ylabel("Top-5% enrichment over random selection")
    axes[1].set_ylim(0, 14)
    axes[1].axhline(1, color="#64748b", ls="--", lw=1)
    axes[1].text(-0.45, 1.2, "Random baseline", color="#52647e", fontsize=9, fontweight="bold")
    axes[1].set_title("Enrichment", fontweight="bold")
    for ax in axes:
        ax.set_xticks(x, SPLIT_LABELS, rotation=32, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.axvspan(4.55, 5.45, color="#f4d6a6", alpha=0.28, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.76, bottom=0.19, left=0.08, right=0.98, wspace=0.20)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.88))
    fig.suptitle("Regression performance translates into candidate-shortlist instability", fontweight="bold", y=0.98)
    fig.savefig(OUT / "figure4_revised_percent.pdf", bbox_inches="tight")
    plt.close(fig)


def pair_label(pair: str) -> str:
    names = {
        "random": "Random", "geo_grouped": "Geometry", "metal_grouped": "Metal",
        "func_grouped": "Functional", "linker_grouped": "Linker", "topology_grouped": "Topology",
    }
    a, b = pair.split(" vs ")
    return f"{names[a]}\nvs {names[b]}"


def figure7() -> None:
    df = pd.read_csv(DATA / "figure7_v8_exact_elite_instability_polished_plot_data.csv")
    d = df[df["source"] == "least_stable_pairs"].sort_values("elite_jaccard", ascending=True).reset_index(drop=True)
    colors = ["#d7892b" if "topology_grouped" in p else "#74869f" for p in d["pair"]]
    labels = [pair_label(p) for p in d["pair"]]
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 6.1), sharey=True)
    axes[0].barh(y, d["elite_jaccard"], color=colors)
    axes[0].set_xlim(0, 0.85)
    axes[0].set_xlabel("Top-5% Jaccard overlap")
    axes[0].set_title("Candidate shortlist overlap", fontweight="bold")
    axes[0].set_yticks(y, labels)
    axes[1].barh(y, d["prediction_spearman"], color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks(np.linspace(0, 1, 6))
    axes[1].set_xlabel("Global prediction-rank Spearman $\\rho$")
    axes[1].set_title("Whole-list rank agreement", fontweight="bold")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.15, right=0.98, wspace=0.05)
    fig.legend([Patch(color="#d7892b"), Patch(color="#74869f")], ["Topology involved", "No topology split"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.89), frameon=True)
    fig.suptitle("High global rank agreement can hide severe elite-list instability", fontweight="bold", y=0.98)
    fig.savefig(OUT / "figure7_revised_spearman_0_1.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    figure3()
    figure4()
    figure7()
    shutil.copy2(
        ROOT / "figures" / "main" / "figure5_v9_hard_group_lollipop_polished.pdf",
        OUT / "figure5_revised_uniform_markers.pdf",
    )
    print("Created four reviewer-facing PDF assets in", OUT)


if __name__ == "__main__":
    main()
