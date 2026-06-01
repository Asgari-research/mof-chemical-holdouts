
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import os
from pathlib import Path

print("Current folder:")
print(os.getcwd())

script = Path("code/make_publication_figures.py")
print("Script exists:", script.exists())
print("Script path:", script.resolve())

txt = script.read_text(encoding="utf-8")
print("Has fixed marker:", "RUNNING FIXED FIGURE 2 FUNCTION" in txt)
# -----------------------------------------------------------------------------
# Visual grammar
# -----------------------------------------------------------------------------

SPLIT_ORDER = [
    "random",
    "geo_grouped",
    "metal_grouped",
    "func_grouped",
    "linker_grouped",
    "topology_grouped",
]

SPLIT_LABELS = {
    "random": "Random",
    "geo_grouped": "Geometry",
    "metal_grouped": "Metal",
    "func_grouped": "Functional",
    "linker_grouped": "Linker",
    "topology_grouped": "Topology",
}

TARGET_ORDER = ["co2_0015bar", "co2_015bar", "ch4_58bar", "ch4_65bar"]
TARGET_LABELS = {
    "co2_0015bar": "CO$_2$\n0.015 bar",
    "co2_015bar": "CO$_2$\n0.15 bar",
    "ch4_58bar": "CH$_4$\n5.8 bar",
    "ch4_65bar": "CH$_4$\n65 bar",
}
TARGET_LABELS_FLAT = {
    "co2_0015bar": "CO$_2$ at 0.015 bar",
    "co2_015bar": "CO$_2$ at 0.15 bar",
    "ch4_58bar": "CH$_4$ at 5.8 bar",
    "ch4_65bar": "CH$_4$ at 65 bar",
}

MODEL_ORDER = ["ridge", "rf", "hgb", "mlp"]
MODEL_LABELS = {"ridge": "Ridge", "rf": "RF", "hgb": "HGB", "mlp": "MLP"}

DESC_ORDER = ["compact_geometry", "enriched_interpretable", "geometry_plus_topology"]
DESC_LABELS = {
    "compact_geometry": "Compact geometry",
    "enriched_interpretable": "Enriched interpretable",
    "geometry_plus_topology": "Geometry + topology",
}

# Calm, manuscript-friendly palette. Hex values are deliberately consistent across figures.
SPLIT_COLORS = {
    "random": "#6B7280",
    "geo_grouped": "#2A6FBB",
    "metal_grouped": "#B84A4A",
    "func_grouped": "#3C8D5A",
    "linker_grouped": "#7A5EA8",
    "topology_grouped": "#D0832F",
}

DESC_COLORS = {
    "compact_geometry": "#7E8A97",
    "enriched_interpretable": "#315C8D",
    "geometry_plus_topology": "#5D7A51",
}

TARGET_COLORS = {
    "co2_0015bar": "#315C8D",
    "co2_015bar": "#2A6FBB",
    "ch4_58bar": "#B56B45",
    "ch4_65bar": "#7A5EA8",
}

CMAP_BLUE = LinearSegmentedColormap.from_list("paper_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#08519C"])
CMAP_ORANGE = LinearSegmentedColormap.from_list("paper_orange", ["#FFF7EC", "#FDD49E", "#FC8D59", "#B30000"])
CMAP_DIVERGE = LinearSegmentedColormap.from_list("paper_diverge", ["#8C2D04", "#FEE8C8", "#F7F7F7", "#C6DBEF", "#08519C"])


@dataclass
class FigureManifestRow:
    filename: str
    source_csv: str
    note: str


def configure_matplotlib() -> None:
    """Set a consistent, journal-friendly plotting style."""
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.8,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "figure.titlesize": 12.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.55,
        "grid.color": "#94A3B8",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
        "agg.path.chunksize": 20000,
        "axes.titlepad": 8.0,
        "axes.labelpad": 5.0,
        "xtick.major.pad": 4.0,
        "ytick.major.pad": 4.0,
    })


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=11.2, fontweight="bold", va="top", ha="left",
        color="#111827", zorder=20,
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 600) -> list[Path]:
    ensure_dir(out_dir)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    return [png, pdf]


def add_clean_legend(ax: plt.Axes, **kwargs) -> None:
    leg = ax.legend(frameon=True, framealpha=0.97, edgecolor="#CBD5E1", fancybox=True, **kwargs)
    if leg:
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")


def add_fig_legend(fig: plt.Figure, handles, labels, **kwargs):
    leg = fig.legend(handles, labels, frameon=True, framealpha=0.98, edgecolor="#CBD5E1", fancybox=True, **kwargs)
    leg.get_frame().set_linewidth(0.7)
    leg.get_frame().set_facecolor("white")
    return leg


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.18)
    if grid_axis != "both":
        ax.grid(axis="x" if grid_axis == "y" else "y", alpha=0.08)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def add_hardest_regime_band(
    ax: plt.Axes,
    x0: float = 4.45,
    x1: float = 5.55,
    text: str = "Hardest\nregime",
    label_y: float = 1.02,
) -> None:
    """
    Highlight the topology / hardest-regime region.

    The shaded band stays inside the plot.
    The text label is placed just above the axes, so it does not collide
    with bars, error bars, or grid lines.
    """
    ax.axvspan(
        x0,
        x1,
        color="#F7E7D0",
        zorder=-4,
        alpha=0.65,
    )

    ax.text(
        (x0 + x1) / 2,
        label_y,
        text,
        transform=ax.get_xaxis_transform(),  # x=data coordinates, y=axes coordinates
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#9A3412",
        fontweight="bold",
        linespacing=0.88,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.12",
            fc="white",
            ec="none",
            alpha=0.72,
        ),
        zorder=30,
    )

def split_labels(values: Sequence[str]) -> list[str]:
    return [SPLIT_LABELS.get(v, str(v).replace("_", " ").title()) for v in values]


def metric_heatmap(ax: plt.Axes, data: pd.DataFrame, value_col: str, row_order: Sequence[str], col_order: Sequence[str],
                   cmap=CMAP_BLUE, vmin=None, vmax=None, norm=None, fmt: str = ".2f") -> None:
    pivot = data.pivot(index="split_family", columns="model", values=value_col).reindex(index=row_order, columns=col_order)
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([MODEL_LABELS.get(c, c.upper()) for c in col_order])
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(split_labels(row_order))
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if (vmax is not None and val > (vmin + 0.65 * (vmax - vmin))) else "#1F2937"
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7.1, color=color)
    return im


def readable_pair(pair: str) -> str:
    out = pair
    for key, label in SPLIT_LABELS.items():
        out = out.replace(key, label)
    return out.replace(" vs ", "\nvs ")


# -----------------------------------------------------------------------------
# Main figures
# -----------------------------------------------------------------------------

def plot_figure1_workflow(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.35, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Split strategy as a controlled validation variable", ha="center", va="top",
            fontsize=12.4, fontweight="bold", color="#102A43")
    ax.text(0.5, 0.915, "Same data, descriptors and model grid; only the train/test logic changes.",
            ha="center", va="top", fontsize=8.9, color="#52616B")

    # Subtle lanes.
    for y, label in [(0.61, "Benchmark design"), (0.26, "Model-to-audit translation")]:
        ax.add_patch(patches.FancyBboxPatch((0.035, y), 0.93, 0.29,
                                            boxstyle="round,pad=0.012,rounding_size=0.025",
                                            ec="#E5E7EB", fc="#F9FAFB", lw=0.8, zorder=0))
        ax.text(0.05, y + 0.255, label, fontsize=7.5, color="#6B7280", fontweight="bold")

    for _, row in df.iterrows():
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.022",
            ec="#CBD5E1", fc=row.get("color", "#F8FAFC"), lw=0.9, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + 0.018, y + h - 0.045, str(row["heading"]), fontsize=9.5,
                fontweight="bold", color="#102A43", va="top")
        ax.text(x + 0.018, y + h - 0.082, str(row["body"]), fontsize=7.8,
                color="#334155", va="top", linespacing=1.25)

    # Connect top lane left-to-right and then to bottom lane.
    arrows = [
        ((0.300, 0.755), (0.382, 0.755)),
        ((0.627, 0.755), (0.710, 0.755)),
        ((0.833, 0.665), (0.833, 0.515)),
        ((0.710, 0.405), (0.627, 0.405)),
        ((0.382, 0.405), (0.300, 0.405)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#64748B", shrinkA=3, shrinkB=3))

    ax.text(0.5, 0.12,
            "Deliverable: auditable accuracy, screening and group-level diagnostics for MOF extrapolation.",
            ha="center", va="center", fontsize=8.8, color="#334155",
            bbox=dict(boxstyle="round,pad=0.34", fc="#FFFFFF", ec="#E2E8F0", lw=0.8))
    return save_figure(fig, out_dir, "figure1", dpi)


def plot_figure2_split_severity(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    print(">>> RUNNING FIXED FIGURE 2 FUNCTION: HARDEST REGIME LABEL OUTSIDE AXES <<<", flush=True)

    df = pd.read_csv(csv_path)
    df["split_family"] = pd.Categorical(df["split_family"], SPLIT_ORDER, ordered=True)
    df = df.sort_values(["split_family", "descriptor_family"])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.10), sharex=True)

    x = np.arange(len(SPLIT_ORDER))
    width = 0.34
    descs = ["compact_geometry", "enriched_interpretable"]

    offsets = {
        "compact_geometry": -width / 2,
        "enriched_interpretable": width / 2,
    }

    panels = [
        (
            axes[0],
            "r2_mean",
            "r2_sem",
            "Mean $R^2$",
            (-0.08, 0.86),
            "a",
        ),
        (
            axes[1],
            "mae_mean",
            "mae_sem",
            "Mean absolute error\n(mmol g$^{-1}$)",
            (0.0, 0.64),
            "b",
        ),
    ]

    for ax, metric, sem, ylabel, ylim, label in panels:
        for desc in descs:
            sub = (
                df[df["descriptor_family"] == desc]
                .set_index("split_family")
                .reindex(SPLIT_ORDER)
            )

            ax.bar(
                x + offsets[desc],
                sub[metric],
                width=width,
                yerr=sub[sem],
                capsize=2.7,
                color=DESC_COLORS[desc],
                edgecolor="white",
                linewidth=0.8,
                label=DESC_LABELS[desc],
                alpha=0.97,
                error_kw=dict(
                    elinewidth=1.0,
                    ecolor="#111827",
                ),
                zorder=3,
            )

        style_axes(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels(SPLIT_ORDER), rotation=32, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

        # Panel labels
        panel_label(ax, label, x=-0.08, y=1.08)

        # Orange topology / hardest-regime band
        ax.axvspan(
            4.45,
            5.55,
            color="#F7E7D0",
            zorder=-4,
            alpha=0.65,
        )

        # IMPORTANT:
        # x is in data coordinates, y is in axes coordinates.
        # y > 1 places the text above the subplot, not inside it.
        ax.text(
            5.0,
            1.025,
            "Hardest\nregime",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.1,
            color="#9A3412",
            fontweight="bold",
            linespacing=0.86,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.10",
                fc="white",
                ec="none",
                alpha=0.75,
            ),
            zorder=30,
        )

    handles = [
        patches.Patch(color=DESC_COLORS[d], label=DESC_LABELS[d])
        for d in descs
    ]

    add_fig_legend(
        fig,
        handles,
        [DESC_LABELS[d] for d in descs],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.suptitle(
        "Grouped hold-outs expose extrapolation penalties in the anchor CO$_2$ task",
        y=1.10,
        fontweight="bold",
    )

    fig.subplots_adjust(
        top=0.70,
        bottom=0.18,
        left=0.085,
        right=0.985,
        wspace=0.32,
    )

    # TEMPORARY TEST NAME so you cannot accidentally open the old PDF
    return save_figure(fig, out_dir, "figure2_TEST_FIXED_HARDEST_LABEL", dpi)





def plot_figure3_target_sensitivity(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)

    # Wider figure + explicit colorbar axes + spacer column between panels
    fig = plt.figure(figsize=(9.35, 4.25))

    gs = GridSpec(
        1,
        5,
        figure=fig,
        width_ratios=[1.0, 0.055, 0.41, 1.0, 0.055],
        left=0.08,
        right=0.96,
        bottom=0.17,
        top=0.78,
        wspace=0.08,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])

    spacer_ax = fig.add_subplot(gs[0, 2])
    spacer_ax.axis("off")

    ax1 = fig.add_subplot(gs[0, 3])
    cax1 = fig.add_subplot(gs[0, 4])

    axes = [ax0, ax1]

    # ------------------------------------------------------------------
    # Panel a: mean R2
    # ------------------------------------------------------------------
    mean = df[df["source"] == "mean_r2"].copy()

    mat_mean = (
        mean
        .pivot(index="split_family", columns="target_slug", values="mean_r2")
        .reindex(index=SPLIT_ORDER, columns=TARGET_ORDER)
    )

    im0 = ax0.imshow(
        mat_mean,
        cmap=CMAP_BLUE,
        vmin=-0.05,
        vmax=1.0,
        aspect="auto",
    )

    ax0.set_title("Mean $R^2$", pad=12)

    # ------------------------------------------------------------------
    # Panel b: delta R2 versus random
    # ------------------------------------------------------------------
    delta = df[df["source"] == "delta_r2_vs_random"].copy()

    mat_delta = (
        delta
        .pivot(index="split_family", columns="target_slug", values="delta_r2_vs_random")
        .reindex(index=SPLIT_ORDER, columns=TARGET_ORDER)
    )

    norm = TwoSlopeNorm(vcenter=0, vmin=-0.60, vmax=0.04)

    im1 = ax1.imshow(
        mat_delta,
        cmap=CMAP_DIVERGE,
        norm=norm,
        aspect="auto",
    )

    ax1.set_title(r"$\Delta R^2$ versus random split", pad=12)

    # ------------------------------------------------------------------
    # Shared formatting
    # ------------------------------------------------------------------
    for ax, mat, fmt, label in [
        (ax0, mat_mean, ".2f", "a"),
        (ax1, mat_delta, ".2f", "b"),
    ]:
        ax.set_xticks(range(len(TARGET_ORDER)))
        ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER])

        ax.set_yticks(range(len(SPLIT_ORDER)))
        ax.set_yticklabels(split_labels(SPLIT_ORDER))

        ax.tick_params(length=0)

        # Panel labels moved slightly upward/outward
        panel_label(ax, label, x=-0.15, y=1.06)

        arr = mat.to_numpy(dtype=float)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    color = (
                        "white"
                        if (ax is ax0 and val > 0.62) or (ax is ax1 and val < -0.35)
                        else "#1F2937"
                    )
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=7.9,
                        color=color,
                        fontweight="bold",
                    )

        ax.set_xticks(np.arange(-0.5, len(TARGET_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SPLIT_ORDER), 1), minor=True)

        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    # ------------------------------------------------------------------
    # Dedicated colorbars
    # ------------------------------------------------------------------
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label("Mean $R^2$", labelpad=7)

    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("Change in $R^2$", labelpad=7)

    fig.suptitle(
        "Target identity controls how strongly grouped validation changes the conclusion",
        y=1.02,
        fontweight="bold",
    )

    return save_figure(fig, out_dir, "figure3_v8_target_split_sensitivity_polished", dpi)


def plot_figure4_screening(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    df["split_family"] = pd.Categorical(df["split_family"], SPLIT_ORDER, ordered=True)
    df = df.sort_values(["split_family", "descriptor_family"])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.95), sharex=True)
    x = np.arange(len(SPLIT_ORDER))
    width = 0.34
    descs = ["compact_geometry", "enriched_interpretable"]
    offsets = {"compact_geometry": -width/2, "enriched_interpretable": width/2}

    panels = [
        ("top_5pct_overlap_mean", "top_5pct_overlap_sem", "Top-5% overlap", (0.0, 0.72), "a"),
        ("top_5pct_enrichment_mean", "top_5pct_enrichment_sem", "Top-5% enrichment\nover random selection", (0.0, 14.0), "b"),
    ]
    for ax, (metric, sem, ylabel, ylim, label) in zip(axes, panels):
        for desc in descs:
            sub = df[df["descriptor_family"] == desc].set_index("split_family").reindex(SPLIT_ORDER)
            ax.bar(x + offsets[desc], sub[metric], width=width, yerr=sub[sem], capsize=2.7,
                   color=DESC_COLORS[desc], edgecolor="white", linewidth=0.8, label=DESC_LABELS[desc],
                   error_kw=dict(elinewidth=1.0, ecolor="#111827"), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels(SPLIT_ORDER), rotation=32, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        if label == "b":
            panel_label(ax, label, x=-0.08, y=1.09)
        else:
            panel_label(ax, label, x=-0.08, y=1.09)        
            
            style_axes(ax, grid_axis="y")
        if "enrichment" in metric:
            ax.axhline(1, color="#475569", lw=1.1, ls="--", alpha=0.85)
            ax.text(0.02, 0.07, "Random baseline", transform=ax.transAxes, fontsize=8.2, color="#475569", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
        add_hardest_regime_band(ax)
    handles = [patches.Patch(color=DESC_COLORS[d], label=DESC_LABELS[d]) for d in descs]
    add_fig_legend(fig, handles, [DESC_LABELS[d] for d in descs], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=1.8)
    fig.suptitle("Regression performance translates into candidate-shortlist instability", y=1.14, fontweight="bold")
    fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.28)
    return save_figure(fig, out_dir, "figure4_v8_screening_consequences_polished", dpi)
def plot_figure5_hard_groups(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)

    geo = (
        df[df["source"].eq("hardest_geometry_groups_exact")]
        .copy()
        .sort_values("mean_abs_error")
    )

    chem = (
        df[df["source"].eq("hardest_chemistry_topology_groups_exact")]
        .copy()
        .sort_values("mean_abs_error")
    )

    # Wider figure + more space between panels
    fig = plt.figure(figsize=(8.8, 5.05))

    gs = GridSpec(
        1,
        2,
        width_ratios=[1.0, 1.14],
        wspace=0.41,
        figure=fig,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]

    for ax, sub, title, label in [
        (axes[0], geo, "Geometry-grouped high-error hold-outs", "a"),
        (axes[1], chem, "Chemistry/topology high-error hold-outs", "b"),
    ]:
        labels = sub["group_label_v9"].fillna(sub["group_label"]).astype(str).tolist()
        y = np.arange(len(sub))

        vals = sub["mean_abs_error"].to_numpy(float)
        signed = sub["mean_signed_residual"].to_numpy(float)
        colors = [
            SPLIT_COLORS.get(s, "#64748B")
            for s in sub["split_family"].astype(str)
        ]

        # Soft alternating background bands
        for yi in y[::2]:
            ax.axhspan(
                yi - 0.45,
                yi + 0.45,
                color="#F8FAFC",
                zorder=0,
            )

        ax.hlines(
            y,
            0,
            vals,
            color="#CBD5E1",
            lw=1.7,
            zorder=1,
        )

        ax.scatter(
            vals,
            y,
            s=np.clip(sub["n"].to_numpy(float) * 3.2, 55, 225),
            c=colors,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

        for yi, val, sr in zip(y, vals, signed):
            marker = "under-predict" if sr > 0 else "over-predict"

            ax.text(
                val + max(vals) * 0.03,
                yi,
                marker,
                va="center",
                fontsize=6.6,
                color="#64748B",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.4)
        ax.set_xlabel("Mean absolute error (mmol g$^{-1}$)")

        # More title padding; slightly smaller title to avoid crowding
        ax.set_title(
            title,
            pad=16,
            fontsize=9.4,
            fontweight="bold",
        )

        # Move panel labels clearly above the subplot titles
        panel_label(ax, label, x=-0.15, y=1.20)

        ax.set_xlim(0, max(vals) * 1.26)
        style_axes(ax, grid_axis="x")

    handles = [
        patches.Patch(color=SPLIT_COLORS[s], label=SPLIT_LABELS[s])
        for s in ["geo_grouped", "metal_grouped", "func_grouped", "linker_grouped"]
    ]

    add_fig_legend(
        fig,
        handles,
        [SPLIT_LABELS[s] for s in ["geo_grouped", "metal_grouped", "func_grouped", "linker_grouped"]],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
    )

    fig.suptitle(
        "Hard held-out groups make extrapolation error chemically auditable",
        y=1.08,
        fontweight="bold",
    )

    fig.subplots_adjust(
        top=0.74,
        bottom=0.15,
        left=0.10,
        right=0.985,
    )

    return save_figure(fig, out_dir, "figure5_v9_hard_group_lollipop_polished", dpi)






def plot_figure6_shift_descriptor_space(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path, low_memory=False)

    shift = df[df["source"].eq("shift_points")].copy()
    bar = df[df["source"].eq("shift_bar")].copy()
    pca = df[df["source"].eq("pca_example")].copy()

    # Clean layout:
    # - split-family legend above top row
    # - panel-c legend above panel c
    # - no dedicated middle legend row
    fig = plt.figure(figsize=(9.25, 6.55))

    gs = GridSpec(
        2,
        2,
        height_ratios=[1.0, 1.20],
        width_ratios=[1.24, 1.0],
        hspace=0.75,
        wspace=0.34,
        figure=fig,
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.82,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    # ------------------------------------------------------------------
    # Panel a: descriptor-space displacement vs performance
    # ------------------------------------------------------------------
    split_handles = []
    split_legend_labels = []

    for split in SPLIT_ORDER:
        sub = shift[shift["split_family"].eq(split)]
        if sub.empty:
            continue

        sc = ax0.scatter(
            sub["shift_centroid_distance"],
            sub["r2"],
            s=32,
            alpha=0.72,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
            edgecolor="white",
            linewidth=0.35,
            rasterized=True,
        )

        split_handles.append(sc)
        split_legend_labels.append(SPLIT_LABELS[split])

    ax0.set_xlabel("Centroid shift")
    ax0.set_ylabel("Fold-level $R^2$")
    ax0.set_title("Descriptor-space displacement vs performance", pad=12)
    ax0.axhline(0, lw=1.0, ls="--", color="#64748B", alpha=0.8)

    panel_label(ax0, "a", x=-0.14, y=1.08)
    style_axes(ax0, grid_axis="both")

    # ------------------------------------------------------------------
    # Panel b: average feature-distribution shift
    # ------------------------------------------------------------------
    bar["split_family"] = pd.Categorical(bar["split_family"], SPLIT_ORDER, ordered=True)
    bar = bar.sort_values("split_family")

    xpos = np.arange(len(bar))

    ax1.bar(
        xpos,
        bar["mean_shift_avg_wasserstein"],
        color=[SPLIT_COLORS[str(s)] for s in bar["split_family"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax1.set_xticks(xpos)
    ax1.set_xticklabels(
        split_labels(bar["split_family"].astype(str).tolist()),
        rotation=32,
        ha="right",
    )

    ax1.set_ylabel("Mean standardized\nWasserstein shift")
    ax1.set_title("Average feature-distribution shift", pad=12)

    panel_label(ax1, "b", x=-0.20, y=1.08)
    style_axes(ax1, grid_axis="y")

    # ------------------------------------------------------------------
    # Panel c: PCA example
    # ------------------------------------------------------------------
    highlight = pca["highlighted_geometry_group"].astype(str).str.lower().eq("true")

    bg_handle = ax2.scatter(
        pca.loc[~highlight, "pca_pc1"],
        pca.loc[~highlight, "pca_pc2"],
        s=8,
        color="#B8C3D1",
        alpha=0.30,
        linewidth=0,
        label="background frameworks",
        rasterized=True,
    )

    hi_handle = ax2.scatter(
        pca.loc[highlight, "pca_pc1"],
        pca.loc[highlight, "pca_pc2"],
        s=58,
        color="#C55A54",
        alpha=0.95,
        edgecolor="white",
        linewidth=0.75,
        label="highlighted held-out geometry group",
        rasterized=True,
    )

    ax2.set_xlabel("PCA component 1")
    ax2.set_ylabel("PCA component 2")
    ax2.set_title("Descriptor-space example: top geometry groups highlighted)", pad=25)


    panel_label(ax2, "c", x=-0.045, y=1.08)
    style_axes(ax2, grid_axis="both")

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------

    # Split-family legend above the top panels
    leg1 = fig.legend(
        split_handles,
        split_legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=True,
        framealpha=0.97,
        edgecolor="#CBD5E1",
        fancybox=True,
        columnspacing=1.0,
        handletextpad=0.35,
        borderpad=0.25,
    )

    # Panel-c legend just above panel c, safely below the top-row x-labels
    leg2 = ax2.legend(
        [bg_handle, hi_handle],
        ["background frameworks", "highlighted held-out geometry group"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
        framealpha=0.97,
        edgecolor="#CBD5E1",
        fancybox=True,
        columnspacing=1.2,
        handletextpad=0.55,
        borderpad=0.25,
    )

    for leg in [leg1, leg2]:
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")

    fig.suptitle(
        "Descriptor-space shift explains why split families are not interchangeable",
        y=0.985,
        fontweight="bold",
    )

    return save_figure(fig, out_dir, "figure6_v8_shift_and_descriptor_space_polished", dpi)

def plot_figure7_elite_instability(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    exact = df[df["source"].eq("elite_stability_exact_top5")].copy()
    exact = exact.sort_values("elite_jaccard")
    labels = [readable_pair(p) for p in exact["pair"]]
    y = np.arange(len(exact))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.15), sharey=True)
    colors = ["#D0832F" if "topology_grouped" in p else "#6C7F99" for p in exact["pair"]]

    axes[0].barh(y, exact["elite_jaccard"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    axes[0].set_xlabel("Top-5% Jaccard overlap")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.2)
    axes[0].set_xlim(0, 0.85)
    axes[0].set_title("Candidate shortlist overlap")
    panel_label(axes[0], "a", x=-0.13)

    axes[1].barh(y, exact["prediction_spearman"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    axes[1].set_xlabel(r"Global prediction-rank Spearman $\rho$")
    axes[1].set_xlim(0.88, 1.0)
    axes[1].set_title("Whole-list rank agreement")
    panel_label(axes[1], "b", x=-0.12)
    for ax in axes:
        style_axes(ax, grid_axis="x")
    handles = [patches.Patch(color="#D0832F", label="Topology involved"), patches.Patch(color="#6C7F99", label="No topology split")]
    add_fig_legend(fig, handles, ["Topology involved", "No topology split"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.1)
    fig.suptitle("High global rank agreement can hide severe elite-list instability", y=1.11, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.20, right=0.98, wspace=0.15)
    return save_figure(fig, out_dir, "figure7_v8_exact_elite_instability_polished", dpi)
# -----------------------------------------------------------------------------
# SI figures
# -----------------------------------------------------------------------------

def plot_si_heatmap(csv_path: Path, out_dir: Path, dpi: int, metric_col: str, stem: str, title: str, cmap=CMAP_BLUE) -> list[Path]:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.0, 3.75))
    val_min = float(np.nanmin(df[metric_col]))
    val_max = float(np.nanmax(df[metric_col]))
    pad = 0.02 * (val_max - val_min if val_max > val_min else 1)
    im = metric_heatmap(ax, df, metric_col, SPLIT_ORDER, MODEL_ORDER, cmap=cmap, vmin=val_min - pad, vmax=val_max + pad, fmt=".2f")
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(SPLIT_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(title.replace("Mean ", ""))
    fig.tight_layout()
    return save_figure(fig, out_dir, stem, dpi)

def plot_si_variance_decomposition(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    factor_order = ["split_family", "descriptor_family", "model", "target_slug"]
    response_order = ["r2", "mae", "spearman_rho", "top_5pct_overlap", "top_5pct_enrichment"]
    df = df[df["factor"].isin(factor_order) & df["response"].isin(response_order)].copy()
    pivot = df.pivot(index="response", columns="factor", values="eta_squared").reindex(index=response_order, columns=factor_order)

    fig, ax = plt.subplots(figsize=(7.3, 4.1))
    x = np.arange(len(response_order))
    width = 0.18
    colors = ["#2A6FBB", "#3C8D5A", "#7A5EA8", "#D0832F"]
    for k, factor in enumerate(factor_order):
        ax.bar(x + (k - 1.5) * width, pivot[factor], width=width, color=colors[k], edgecolor="white", label=factor.replace("_", " "), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").replace("r2", "$R^2$") for r in response_order], rotation=22, ha="right")
    ax.set_ylabel(r"$\eta^2$ effect size")
    ax.set_title("Variance decomposition of benchmark outcomes", fontweight="bold")
    style_axes(ax, grid_axis="y")
    add_clean_legend(ax, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return save_figure(fig, out_dir, "si_variance_decomposition", dpi)

def plot_si_group_size(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    df["split_family"] = pd.Categorical(df["split_family"], SPLIT_ORDER, ordered=True)
    df = df.sort_values("split_family")

    labels = split_labels(df["split_family"].astype(str).tolist())
    x = np.arange(len(df))

    # Robust layout:
    # top-right row is reserved ONLY for the legend,
    # bottom row contains the two panels.
    fig = plt.figure(figsize=(8.4, 4.55))

    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.16, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.10,
        right=0.98,
        bottom=0.20,
        top=0.82,
        wspace=0.38,
        hspace=0.10,
    )

    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    axes = [ax0, ax1]

    # ------------------------------------------------------------------
    # Panel a: number of groups
    # ------------------------------------------------------------------
    ax0.bar(
        x,
        df["n_groups"],
        color=[SPLIT_COLORS[str(s)] for s in df["split_family"]],
        edgecolor="white",
        zorder=3,
    )

    ax0.set_yscale("log")
    ax0.set_ylabel("Number of groups (log scale)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=32, ha="right")
    ax0.set_title("Grouping granularity", pad=12)

    panel_label(ax0, "a", x=-0.20, y=1.11)
    style_axes(ax0, grid_axis="y")

    # ------------------------------------------------------------------
    # Panel b: group-size distribution
    # ------------------------------------------------------------------
    line_median, = ax1.plot(
        x,
        df["median_group_size"],
        marker="o",
        lw=1.8,
        label="Median",
        color="#2A6FBB",
    )

    line_mean, = ax1.plot(
        x,
        df["mean_group_size"],
        marker="s",
        lw=1.8,
        label="Mean",
        color="#3C8D5A",
    )

    line_max, = ax1.plot(
        x,
        df["max_group_size"],
        marker="^",
        lw=1.8,
        label="Maximum",
        color="#B84A4A",
    )

    ax1.set_yscale("log")
    ax1.set_ylabel("Group size (log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=32, ha="right")
    ax1.set_title("Group-size distribution", pad=12)

    panel_label(ax1, "b", x=-0.20, y=1.11)
    style_axes(ax1, grid_axis="y")

    # ------------------------------------------------------------------
    # Dedicated legend row above the right panel
        # ------------------------------------------------------------------
    leg = legend_ax.legend(
        handles=[line_median, line_mean, line_max],
        labels=["Median", "Mean", "Maximum"],
        loc="center",
        bbox_to_anchor=(0.5, 0.8),
        ncol=3,
        frameon=True,
        framealpha=0.97,
        edgecolor="#CBD5E1",
        fancybox=True,
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.45,
        borderpad=0.25,
    )
    leg.get_frame().set_linewidth(0.7)
    leg.get_frame().set_facecolor("white")

    fig.suptitle(
        "Grouped splits differ strongly in the number and size of held-out families",
        y=0.96,
        fontsize=10.8,
        fontweight="bold",
    )

    return save_figure(fig, out_dir, "si_v7_group_size_summary_logscale", dpi)
def plot_si_error_vs_rank(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.9, 4.35))
    for split in SPLIT_ORDER:
        sub = df[df["split_family"].eq(split)]
        if sub.empty:
            continue
        for desc, marker in [("compact_geometry", "o"), ("enriched_interpretable", "s")]:
            s2 = sub[sub["descriptor_family"].eq(desc)]
            if s2.empty:
                continue
            sizes = np.clip((s2["r2_mean"].to_numpy(float) + 0.1) * 72, 20, 95)
            ax.scatter(s2["mae_mean"], s2["overlap_mean"], s=sizes, marker=marker,
                       color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.55, alpha=0.84,
                       label=SPLIT_LABELS[split] if desc == "compact_geometry" else None, rasterized=True)
    ax.set_xlabel("Mean absolute error")
    ax.set_ylabel("Top-5% overlap")
    ax.set_title("Prediction error versus screening recovery", fontweight="bold")
    style_axes(ax, grid_axis="both")
    add_clean_legend(ax, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
    ax.text(0.99, 0.02, "circle: compact geometry\nsquare: enriched interpretable\nmarker size: mean $R^2$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, color="#475569",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#E2E8F0", alpha=0.96))
    fig.tight_layout()
    return save_figure(fig, out_dir, "si_v7_error_vs_screening_rank_clean", dpi)

def plot_si_pld_decile_errors(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)
    df["decile_index"] = df.groupby("target_slug").cumcount() + 1
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for target in TARGET_ORDER:
        sub = df[df["target_slug"].eq(target)]
        if sub.empty:
            continue
        ax.plot(sub["decile_index"], sub["absolute_error"], marker="o", ms=4.2, lw=1.9,
                color=TARGET_COLORS[target], label=TARGET_LABELS_FLAT[target])
    ax.set_xlabel("PLD decile")
    ax.set_ylabel("Mean absolute error")
    ax.set_xticks(range(1, 11))
    ax.set_title("Error varies systematically across pore-limiting-diameter regimes", fontweight="bold")
    style_axes(ax, grid_axis="y")
    add_clean_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.tight_layout()
    return save_figure(fig, out_dir, "si_v8_pld_decile_errors_clear_labels", dpi)

def plot_si_prediction_scatter(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path, low_memory=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6))
    axes = axes.ravel()
    for ax, target, lab in zip(axes, TARGET_ORDER, ["a", "b", "c", "d"]):
        sub = df[df["target_slug"].eq(target)]
        if sub.empty:
            ax.axis("off")
            continue
        ax.scatter(sub["target"], sub["prediction"], s=7, alpha=0.26, color=TARGET_COLORS[target], linewidth=0, rasterized=True)
        minv = float(np.nanmin([sub["target"].min(), sub["prediction"].min()]))
        maxv = float(np.nanmax([sub["target"].max(), sub["prediction"].max()]))
        pad = (maxv - minv) * 0.04 if maxv > minv else 0.1
        ax.plot([minv - pad, maxv + pad], [minv - pad, maxv + pad], ls="--", lw=1.0, color="#111827", alpha=0.65)
        ax.set_xlim(minv - pad, maxv + pad)
        ax.set_ylim(minv - pad, maxv + pad)
        ax.set_title(TARGET_LABELS_FLAT[target], fontsize=9.2)
        ax.set_xlabel("True uptake")
        ax.set_ylabel("Predicted uptake")
        panel_label(ax, lab)
        style_axes(ax, grid_axis="both")
    fig.suptitle("Predicted versus true uptake across the four adsorption targets", y=1.01, fontsize=10.9, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, out_dir, "si_v8_prediction_scatter_panels_clear_labels", dpi)

def plot_si_target_distributions(csv_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    df = pd.read_csv(csv_path)

    # Slightly larger and more vertical spacing
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.35))
    axes = axes.ravel()

    for ax, target, lab in zip(axes, TARGET_ORDER, ["a", "b", "c", "d"]):
        sub = df[df["target_slug"].eq(target)]
        if sub.empty:
            ax.axis("off")
            continue

        vals = sub["target"].dropna().to_numpy(float)

        ax.hist(
            vals,
            bins=70,
            color=TARGET_COLORS[target],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.18,
        )

        ax.set_title(TARGET_LABELS_FLAT[target], fontsize=9.2, pad=13)
        ax.set_xlabel("Uptake (mmol g$^{-1}$)")
        ax.set_ylabel("Count")
        ax.set_yscale("log")

        # Move panel labels outside the plotting area so they do not hit log ticks
        panel_label(ax, lab, x=-0.13, y=1.15)

        style_axes(ax, grid_axis="y")

    fig.suptitle(
        "Target distributions span distinct uptake ranges and tail structures",
        y=1.03,
        fontsize=10.9,
        fontweight="bold",
    )

    # Avoid tight_layout because it can pull labels/title into each other
    fig.subplots_adjust(
        top=0.86,
        bottom=0.10,
        left=0.09,
        right=0.98,
        hspace=0.55,
        wspace=0.30,
    )

    return save_figure(fig, out_dir, "si_v8_target_distributions_clear_labels", dpi)



# -----------------------------------------------------------------------------
# Structural audit handling and QC
# -----------------------------------------------------------------------------

def copy_structural_audit_if_available(data_dir: Path, out_main: Path) -> list[Path]:
    src = data_dir / "structural_audit" / "Figure_main_structural_audit_grouped_extrapolation_v7_final.pdf"
    copied = []
    if src.exists():
        dst = out_main / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def write_qc_report(manifest: list[FigureManifestRow], out_dir: Path, data_dir: Path) -> Path:
    lines = []
    lines.append("Split Strategy figure regeneration QC report")
    lines.append("=" * 48)
    lines.append("")
    lines.append("Generated analytical figures from exported CSV plot data.")
    lines.append("PNG and PDF versions are provided for each regenerated figure.")
    lines.append("")
    lines.append("Figure manifest:")
    for row in manifest:
        lines.append(f"- {row.filename} | source: {row.source_csv} | {row.note}")
    lines.append("")
    structural = data_dir / "structural_audit" / "Figure_main_structural_audit_grouped_extrapolation_v7_final.pdf"
    if structural.exists():
        lines.append("Structural-audit figure note: the manuscript PDF was copied into figures/main because the archive contains the finalized structural-audit artwork but not the underlying CIF/rendering assets needed for a faithful programmatic redraw.")
    else:
        lines.append("Structural-audit figure note: no structural-audit PDF was found in data/structural_audit.")
    lines.append("")
    lines.append("Recommended manuscript use: replace the existing files in figures/main and figures/si with the matching filenames from this package.")
    path = out_dir / "QC_REPORT.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_manifest_csv(manifest: list[FigureManifestRow], out_dir: Path) -> Path:
    df = pd.DataFrame([row.__dict__ for row in manifest])
    path = out_dir / "figure_manifest.csv"
    df.to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def generate_all(data_dir: Path, out_dir: Path, dpi: int) -> None:
    configure_matplotlib()
    main_data = data_dir / "main" / "figure_data_csv"
    si_data = data_dir / "si" / "figure_data_csv"
    out_main = ensure_dir(out_dir / "main")
    out_si = ensure_dir(out_dir / "si")

    manifest: list[FigureManifestRow] = []

    def record(paths: list[Path], source: Path, note: str) -> None:
        for p in paths:
            manifest.append(FigureManifestRow(str(p.relative_to(out_dir.parent)), str(source.relative_to(data_dir)), note))

    figure_jobs = [
        (plot_figure1_workflow, main_data / "figure1_v8_polished_workflow_plot_data.csv", out_main, "redrawn workflow schematic"),
        (plot_figure2_split_severity, main_data / "figure2_v8_split_family_severity_polished_plot_data.csv", out_main, "redrawn main Figure 2"),
        (plot_figure3_target_sensitivity, main_data / "figure3_v8_target_split_sensitivity_polished_plot_data.csv", out_main, "redrawn main Figure 3"),
        (plot_figure4_screening, main_data / "figure4_v8_screening_consequences_polished_plot_data.csv", out_main, "redrawn main Figure 4"),
        (plot_figure5_hard_groups, main_data / "figure5_v9_hard_group_lollipop_polished_plot_data.csv", out_main, "redrawn main Figure 5"),
        (plot_figure6_shift_descriptor_space, main_data / "figure6_v8_shift_and_descriptor_space_polished_plot_data.csv", out_main, "redrawn main Figure 6"),
        (plot_figure7_elite_instability, main_data / "figure7_v8_exact_elite_instability_polished_plot_data.csv", out_main, "redrawn main Figure 7"),
    ]
    for func, csv_path, target_dir, note in figure_jobs:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required source data: {csv_path}")
        print(f"Generating {csv_path.name} ...", flush=True)
        paths = func(csv_path, target_dir, dpi)
        record(paths, csv_path, note)

    copied = copy_structural_audit_if_available(data_dir, out_main)
    for p in copied:
        manifest.append(FigureManifestRow(str(p.relative_to(out_dir.parent)), "structural_audit/Figure_main_structural_audit_grouped_extrapolation_v7_final.pdf", "copied original structural-audit artwork; underlying rendering assets were not present"))

    # SI metric heatmaps.
    heatmap_specs = [
        ("si_heatmap_kendall_tau_plot_data.csv", "mean_kendall_tau", "si_heatmap_kendall_tau", "Mean Kendall $\\tau$", CMAP_BLUE),
        ("si_heatmap_mae_plot_data.csv", "mean_mae", "si_heatmap_mae", "Mean MAE", CMAP_ORANGE),
        ("si_heatmap_r2_plot_data.csv", "mean_r2", "si_heatmap_r2", "Mean $R^2$", CMAP_BLUE),
        ("si_heatmap_spearman_rho_plot_data.csv", "mean_spearman_rho", "si_heatmap_spearman_rho", "Mean Spearman $\\rho$", CMAP_BLUE),
        ("si_heatmap_top_5pct_enrichment_plot_data.csv", "mean_top_5pct_enrichment", "si_heatmap_top_5pct_enrichment", "Mean top-5% enrichment", CMAP_BLUE),
        ("si_heatmap_top_5pct_overlap_plot_data.csv", "mean_top_5pct_overlap", "si_heatmap_top_5pct_overlap", "Mean top-5% overlap", CMAP_BLUE),
    ]
    for fname, col, stem, title, cmap in heatmap_specs:
        csv_path = si_data / fname
        if csv_path.exists():
            print(f"Generating {csv_path.name} ...", flush=True)
            paths = plot_si_heatmap(csv_path, out_si, dpi, col, stem, title, cmap)
            record(paths, csv_path, f"redrawn SI heatmap: {title}")

    si_jobs = [
        (plot_si_error_vs_rank, si_data / "si_v7_error_vs_screening_rank_clean_plot_data.csv", "redrawn SI error-vs-rank figure"),
        (plot_si_group_size, si_data / "si_v7_group_size_summary_logscale_plot_data.csv", "redrawn SI group-size figure"),
        (plot_si_pld_decile_errors, si_data / "si_v8_pld_decile_errors_clear_labels_plot_data.csv", "redrawn SI PLD-decile error figure"),
        (plot_si_prediction_scatter, si_data / "si_v8_prediction_scatter_panels_clear_labels_plot_data.csv", "redrawn SI prediction scatter panels"),
        (plot_si_target_distributions, si_data / "si_v8_target_distributions_clear_labels_plot_data.csv", "redrawn SI target distributions"),
        (plot_si_variance_decomposition, si_data / "si_variance_decomposition_plot_data.csv", "redrawn SI variance decomposition"),
    ]
    for func, csv_path, note in si_jobs:
        if csv_path.exists():
            print(f"Generating {csv_path.name} ...", flush=True)
            paths = func(csv_path, out_si, dpi)
            record(paths, csv_path, note)

    write_manifest_csv(manifest, out_dir)
    write_qc_report(manifest, out_dir, data_dir)
    print(f"Generated {len(manifest)} files. Output folder: {out_dir.resolve()}")



# -----------------------------------------------------------------------------
# Isolated batch runner
# -----------------------------------------------------------------------------

FIGURE_KEYS = [
    "figure1", "figure2", "figure3", "figure4", "figure5", "figure6", "figure7",
    "si_heatmap_kendall_tau", "si_heatmap_mae", "si_heatmap_r2", "si_heatmap_spearman_rho",
    "si_heatmap_top_5pct_enrichment", "si_heatmap_top_5pct_overlap",
    "si_error_vs_rank", "si_group_size", "si_pld_decile_errors",
    "si_prediction_scatter", "si_target_distributions", "si_variance_decomposition",
]


def generate_one(key: str, data_dir: Path, out_dir: Path, dpi: int) -> None:
    """Generate one figure in a fresh Python process.

    This keeps the full batch robust for large scatter/distribution PDFs because each
    figure is rendered and closed independently.
    """
    configure_matplotlib()
    main_data = data_dir / "main" / "figure_data_csv"
    si_data = data_dir / "si" / "figure_data_csv"
    out_main = ensure_dir(out_dir / "main")
    out_si = ensure_dir(out_dir / "si")

    main_jobs = {
        "figure1": (plot_figure1_workflow, main_data / "figure1_v8_polished_workflow_plot_data.csv", out_main),
        "figure2": (plot_figure2_split_severity, main_data / "figure2_v8_split_family_severity_polished_plot_data.csv", out_main),
        "figure3": (plot_figure3_target_sensitivity, main_data / "figure3_v8_target_split_sensitivity_polished_plot_data.csv", out_main),
        "figure4": (plot_figure4_screening, main_data / "figure4_v8_screening_consequences_polished_plot_data.csv", out_main),
        "figure5": (plot_figure5_hard_groups, main_data / "figure5_v9_hard_group_lollipop_polished_plot_data.csv", out_main),
        "figure6": (plot_figure6_shift_descriptor_space, main_data / "figure6_v8_shift_and_descriptor_space_polished_plot_data.csv", out_main),
        "figure7": (plot_figure7_elite_instability, main_data / "figure7_v8_exact_elite_instability_polished_plot_data.csv", out_main),
    }
    if key in main_jobs:
        func, csv_path, target_dir = main_jobs[key]
        print(f"Generating {key} from {csv_path.name} ...", flush=True)
        func(csv_path, target_dir, dpi)
        return

    heatmap_jobs = {
        "si_heatmap_kendall_tau": ("si_heatmap_kendall_tau_plot_data.csv", "mean_kendall_tau", "si_heatmap_kendall_tau", "Mean Kendall $\\tau$", CMAP_BLUE),
        "si_heatmap_mae": ("si_heatmap_mae_plot_data.csv", "mean_mae", "si_heatmap_mae", "Mean MAE", CMAP_ORANGE),
        "si_heatmap_r2": ("si_heatmap_r2_plot_data.csv", "mean_r2", "si_heatmap_r2", "Mean $R^2$", CMAP_BLUE),
        "si_heatmap_spearman_rho": ("si_heatmap_spearman_rho_plot_data.csv", "mean_spearman_rho", "si_heatmap_spearman_rho", "Mean Spearman $\\rho$", CMAP_BLUE),
        "si_heatmap_top_5pct_enrichment": ("si_heatmap_top_5pct_enrichment_plot_data.csv", "mean_top_5pct_enrichment", "si_heatmap_top_5pct_enrichment", "Mean top-5% enrichment", CMAP_BLUE),
        "si_heatmap_top_5pct_overlap": ("si_heatmap_top_5pct_overlap_plot_data.csv", "mean_top_5pct_overlap", "si_heatmap_top_5pct_overlap", "Mean top-5% overlap", CMAP_BLUE),
    }
    if key in heatmap_jobs:
        fname, col, stem, title, cmap = heatmap_jobs[key]
        csv_path = si_data / fname
        print(f"Generating {key} from {csv_path.name} ...", flush=True)
        plot_si_heatmap(csv_path, out_si, dpi, col, stem, title, cmap)
        return

    si_jobs = {
        "si_error_vs_rank": (plot_si_error_vs_rank, si_data / "si_v7_error_vs_screening_rank_clean_plot_data.csv"),
        "si_group_size": (plot_si_group_size, si_data / "si_v7_group_size_summary_logscale_plot_data.csv"),
        "si_pld_decile_errors": (plot_si_pld_decile_errors, si_data / "si_v8_pld_decile_errors_clear_labels_plot_data.csv"),
        "si_prediction_scatter": (plot_si_prediction_scatter, si_data / "si_v8_prediction_scatter_panels_clear_labels_plot_data.csv"),
        "si_target_distributions": (plot_si_target_distributions, si_data / "si_v8_target_distributions_clear_labels_plot_data.csv"),
        "si_variance_decomposition": (plot_si_variance_decomposition, si_data / "si_variance_decomposition_plot_data.csv"),
    }
    if key in si_jobs:
        func, csv_path = si_jobs[key]
        print(f"Generating {key} from {csv_path.name} ...", flush=True)
        func(csv_path, out_si, dpi)
        return

    raise ValueError(f"Unknown figure key: {key}")


def build_manifest_from_outputs(out_dir: Path, data_dir: Path) -> list[FigureManifestRow]:
    """Scan generated files and build a simple manifest."""
    rows: list[FigureManifestRow] = []
    for p in sorted((out_dir / "main").glob("*")) + sorted((out_dir / "si").glob("*")):
        if not p.is_file() or p.suffix.lower() not in {".png", ".pdf"}:
            continue
        if p.name.startswith("Figure_main_structural"):
            source = "structural_audit/Figure_main_structural_audit_grouped_extrapolation_v7_final.pdf"
            note = "copied original structural-audit artwork; underlying rendering assets were not present"
        elif p.name.startswith("figure1"):
            source = "main/figure_data_csv/figure1_v8_polished_workflow_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        elif p.name.startswith("figure2"):
            source = "main/figure_data_csv/figure2_v8_split_family_severity_polished_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        elif p.name.startswith("figure3"):
            source = "main/figure_data_csv/figure3_v8_target_split_sensitivity_polished_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        elif p.name.startswith("figure4"):
            source = "main/figure_data_csv/figure4_v8_screening_consequences_polished_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        elif p.name.startswith("figure5"):
            source = "main/figure_data_csv/figure5_v9_hard_group_lollipop_polished_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        elif p.name.startswith("figure6"):
            source = "main/figure_data_csv/figure6_v8_shift_and_descriptor_space_polished_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        elif p.name.startswith("figure7"):
            source = "main/figure_data_csv/figure7_v8_exact_elite_instability_polished_plot_data.csv"
            note = "programmatically regenerated from exported main plot-data CSV"
        else:
            source = "si/figure_data_csv/"
            note = "programmatically regenerated from exported SI plot-data CSV"
        rows.append(FigureManifestRow(str(p.relative_to(out_dir.parent)), source, note))
    return rows


def generate_all_isolated(data_dir: Path, out_dir: Path, dpi: int) -> None:
    """Generate all figures by launching one clean process per figure."""
    ensure_dir(out_dir / "main")
    ensure_dir(out_dir / "si")
    script = Path(__file__).resolve()
    for key in FIGURE_KEYS:
        cmd = [sys.executable, str(script), "--only", key, "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--dpi", str(dpi)]
        subprocess.run(cmd, check=True)
    copy_structural_audit_if_available(data_dir, out_dir / "main")
    manifest = build_manifest_from_outputs(out_dir, data_dir)
    write_manifest_csv(manifest, out_dir)
    write_qc_report(manifest, out_dir, data_dir)
    print(f"Generated {len(manifest)} files. Output folder: {out_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate publication-style Split Strategy manuscript figures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input data directory containing main/ and si/ plot-data CSVs.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output directory for regenerated figures.")
    parser.add_argument("--dpi", type=int, default=350, help="PNG export resolution. PDF files are vector/raster-hybrid where appropriate.")
    parser.add_argument("--only", choices=FIGURE_KEYS, help="Generate only one named figure. Used internally by the isolated batch runner.")
    parser.add_argument("--in-process", action="store_true", help="Generate all figures in one Python process instead of the more robust isolated batch mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.only:
        generate_one(args.only, args.data_dir, args.out_dir, args.dpi)
    elif args.in_process:
        generate_all(args.data_dir, args.out_dir, args.dpi)
    else:
        generate_all_isolated(args.data_dir, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
