from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uqct.vis.plot_scaling import process_metrics
from uqct.vis.style import (
    ICML_TEXT_WIDTH,
    MODEL_ORDER,
    get_style,
)


@click.command()
@click.option(
    "--runs-file",
    type=click.Path(path_type=Path),
    default=Path("results/consolidated.parquet"),
    help="Path to consolidated runs parquet file.",
)
@click.option(
    "--rotation-file",
    type=click.Path(path_type=Path),
    default=Path("plots/rotation/rotation_exclusion_summary.parquet"),
    help="Path to rotation results summary parquet file.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("plots/combined"),
    help="Directory to save plots.",
)
def main(runs_file: Path, rotation_file: Path, output_dir: Path):
    """Generates combined Lamino plot: Crossover Rate (Left) and Exclusion Rate (Right)."""

    if not runs_file.exists():
        print(f"Runs file not found: {runs_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and Process Crossover Rate Data ---
    print("Loading runs data...")
    df_runs = pd.read_parquet(runs_file)

    # Filter for Lamino, Sparse
    df_lamino_runs = df_runs[(df_runs["dataset"] == "lamino") & (df_runs["sparse"])]

    if df_lamino_runs.empty:
        print("No Lamino sparse runs found.")
        return

    # Process metrics to get 'rate'
    metric_df = process_metrics(df_lamino_runs)

    # Aggregate Rate
    # Group by model, intensity
    rate_stats = (
        metric_df.groupby(["model", "intensity"])["rate"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    rate_stats["sem"] = rate_stats["std"] / np.sqrt(rate_stats["count"])

    # --- 2. Load and Process Exclusion Rate Data ---
    # Try different paths if default fails? Default is summary file.
    exclusion_stats = pd.DataFrame()
    if rotation_file.exists():
        print(f"Loading rotation data from {rotation_file}...")
        df_rot = pd.read_parquet(rotation_file)

        # Check if this is the summary file or raw results
        # Summary has 'exclusion_rate', raw has 'excluded'

        if "exclusion_rate" in df_rot.columns:
            # It is summary. Filter for Lamino, Sparse, Intensity 1e9
            exclusion_stats = df_rot[
                (df_rot["dataset"] == "lamino")
                & (df_rot["sparse"])
                & (df_rot["intensity"] == 1e9)
            ]
        elif "excluded" in df_rot.columns:
            # Raw results
            df_lamino_rot = df_rot[
                (df_rot["dataset"] == "lamino")
                & (df_rot["sparse"])
                & (df_rot["intensity"] == 1e9)
            ]
            if not df_lamino_rot.empty:
                exclusion_stats = (
                    df_lamino_rot.groupby(["model", "angle"])["excluded"]
                    .agg(mean="mean", std="std", count="count")
                    .reset_index()
                )
                exclusion_stats["sem"] = exclusion_stats["std"] / np.sqrt(
                    exclusion_stats["count"]
                )
                exclusion_stats.rename(columns={"mean": "exclusion_rate"}, inplace=True)
    else:
        print(f"Rotation file not found: {rotation_file}. Skipping right plot data.")

    # --- 3. Plotting ---
    # Use tight_layout with rect
    # Thinner plot: reduce height from 2.5 to 2.0
    fig, axes = plt.subplots(1, 2, figsize=(ICML_TEXT_WIDTH, 1.65))

    # Left: Crossover Rate
    ax0 = axes[0]

    available_models_runs = set(rate_stats["model"].unique())
    models_runs = [m for m in MODEL_ORDER if m in available_models_runs]
    for m in sorted(available_models_runs):
        if m not in models_runs:
            models_runs.append(m)

    handles, labels = [], []
    seen_labels = set()

    def add_handle_label(h, lbl):
        if lbl not in seen_labels and lbl is not None:
            handles.append(h)
            labels.append(lbl)
            seen_labels.add(lbl)

    for model in models_runs:
        sub = rate_stats[rate_stats["model"] == model].sort_values("intensity")
        if sub.empty:
            continue

        style = get_style(model)

        line = ax0.plot(
            sub["intensity"],
            sub["mean"],
            label=style["label"],
            color=style["color"],
            marker="x",
            linewidth=1.5,
            markersize=4,
        )

        ax0.fill_between(
            sub["intensity"],
            (sub["mean"] - sub["sem"]).clip(lower=0),
            (sub["mean"] + sub["sem"]).clip(upper=100),
            color=style["color"],
            alpha=0.2,
        )

        add_handle_label(line[0], style["label"])

    ax0.set_xscale("log")
    ax0.set_xlabel("Total Intensity")
    ax0.set_ylabel(r"Crossover Rate (\%)")
    ax0.grid(True, which="major", linestyle="--", alpha=0.3)

    # Right: Exclusion Rate
    ax1 = axes[1]

    if not exclusion_stats.empty:
        available_models_rot = set(exclusion_stats["model"].unique())
        models_rot = [m for m in MODEL_ORDER if m in available_models_rot]
        for m in sorted(available_models_rot):
            if m not in models_rot:
                models_rot.append(m)

        for model in models_rot:
            sub = exclusion_stats[exclusion_stats["model"] == model].sort_values(
                "angle"
            )
            if sub.empty:
                continue

            style = get_style(model)

            # Capture handle here too in case exclusive to this plot
            line = ax1.plot(
                sub["angle"],
                100 * sub["exclusion_rate"],
                label=style["label"],
                color=style["color"],
                marker="x",
                linewidth=1.5,
                markersize=4,
            )

            sem = sub["sem"].fillna(0)
            ax1.fill_between(
                sub["angle"],
                (100 * (sub["exclusion_rate"] - sem)).clip(lower=0),
                (100 * (sub["exclusion_rate"] + sem)).clip(upper=100),
                color=style["color"],
                alpha=0.2,
            )

            add_handle_label(line[0], style["label"])

    ax1.set_xscale("log")
    ax1.set_xlabel("Rotation Angle (Deg)")
    ax1.set_ylabel(r"Exclusion Rate (\%)")
    # ax1.set_title(r"Exclusion Rate ($10^9$)")
    ax1.grid(True, which="major", linestyle="--", alpha=0.3)

    # Legend
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=5,
            frameon=False,
        )

    out_path = output_dir / "lamino_combined_rates.pdf"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
