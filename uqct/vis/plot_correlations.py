import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, List, Dict
from scipy.stats import spearmanr
import os
import time
from concurrent.futures import ProcessPoolExecutor
from uqct.logging import get_logger

logger = get_logger(__name__)

# Metrics configuration
metrics_config = {
    "psnr": {"label": "PSNR", "direction": "Higher is Better"},
    "ssim": {"label": "SSIM", "direction": "Higher is Better"},
    "rmse": {"label": "RMSE", "direction": "Lower is Better"},
    "l1": {"label": "L1", "direction": "Lower is Better"},
    # "zeroone": {"label": "ZeroOne", "direction": "Lower is Better"}
}

# Model style
from uqct.vis.style import MODEL_ORDER, MODEL_NAMES, MODEL_COLORS


# Refactored Plotting Function (Top Level for Pickleability)
def plot_metric_vs_nll(
    df: pd.DataFrame,
    metric_key: str,
    name_suffix: str,
    folder_name: Path,
    output_dir: Path,
    fast_mode: bool = False,
):
    if metric_key not in df.columns:
        return

    # Re-import locally for safety in processes (optional but safe)
    import matplotlib.pyplot as plt

    try:
        import matplotlib.cm as cm
    except:
        pass

    config = metrics_config.get(metric_key, {"label": metric_key, "direction": ""})
    label_text = f"{config['label']} ({config['direction']})"

    # Prepare directory
    # Prepare directory: output_dir / folder_name / metric_key
    target_dir = output_dir / folder_name / metric_key
    target_dir.mkdir(parents=True, exist_ok=True)

    scales = ["log", "linear"]

    # helper for labels (LaTeX)
    def format_intensity_label(val):
        exponent = int(np.log10(val))
        mantissa = val / (10**exponent)
        if abs(mantissa - 1.0) < 0.1:
            return f"$10^{{{exponent}}}$"
        else:
            return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"

    # helper for filenames (1e4)
    def format_intensity_file(val):
        return f"{val:.0e}".replace("+0", "").replace("+", "")

    # --- Detail Plot (Scatter) & Binned Plot (Mean +/- SEM) ---
    unique_intensities = sorted(df["intensity"].unique())
    available_models = set(df["model"].unique())
    models = [m for m in MODEL_ORDER if m in available_models]  # Enforce order

    # Add any extra models not in standard order at the end
    for m in sorted(available_models):
        if m not in models:
            models.append(m)

    # Use shared colors
    model_color_map = MODEL_COLORS

    # --- Summary Plot (Binned Mean vs NLL) ---
    if not df.empty:
        intensities = sorted(df["intensity"].unique())
        try:
            try:
                cmap = matplotlib.colormaps["viridis_r"].resampled(len(intensities))
            except:
                cmap = plt.get_cmap("viridis_r", len(intensities))
        except:
            cmap = cm.get_cmap("viridis_r", len(intensities))

        plt.figure(figsize=(8, 6))

        # Plot Lines (Intensity) and Points (Model)
        for i, inten in enumerate(intensities):
            # Define raw_sub first
            raw_sub = df[df["intensity"] == inten]
            if raw_sub.empty:
                continue

            # Define styles
            linestyles = ["-", "--", "-.", ":"]
            markers = ["o", "s", "^", "D", "v", "<"]

            # Line color = Black with variation
            # line_color = cmap(i) # Removed
            style = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]

            label_str = format_intensity_label(inten)

            # Binned Mean Line (10 bins)
            try:
                # Work on copy to avoid settingWithCopy
                raw_sub_copy = raw_sub.copy()
                raw_sub_copy["bin"] = pd.cut(raw_sub_copy[metric_key], bins=10)
                binned = (
                    raw_sub_copy.groupby("bin", observed=True)
                    .agg(
                        mean_x=(metric_key, "mean"),
                        mean_y=("nll", "mean"),
                        count=("nll", "count"),
                    )
                    .reset_index()
                )
                binned = binned[binned["count"] > 0].sort_values("mean_x")
                if not binned.empty:
                    plt.plot(
                        binned["mean_x"],
                        binned["mean_y"],
                        color="black",
                        alpha=0.8,
                        linestyle=style,
                        marker=marker,
                        markersize=6,
                        linewidth=2,
                        label=label_str,
                    )
            except Exception as e:
                pass

            # Raw Points - Colored by Model
            point_colors = [model_color_map[m] for m in raw_sub["model"]]
            plt.scatter(
                raw_sub[metric_key],
                raw_sub["nll"],
                c=point_colors,
                marker="o",
                s=15,
                alpha=0.15,
                edgecolors="none",
                zorder=1,
                rasterized=True,
            )

        # Summary Plot Legend
        # Remove title
        # Legends inside
        leg1 = plt.legend(
            title="Total Intensity", loc="upper right", bbox_to_anchor=(1.0, 1.0)
        )
        plt.gca().add_artist(leg1)

        # Model Legend
        model_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=MODEL_NAMES.get(m, m.title()),
                markerfacecolor=model_color_map.get(m, "gray"),
                markersize=10,
                markeredgecolor="w",
            )
            for m in models
        ]
        # Place Model legend to the left of Intensity legend
        plt.legend(
            handles=model_handles,
            title="Model",
            loc="upper right",
            bbox_to_anchor=(0.75, 1.0),
        )

        plt.xlabel(label_text)
        plt.ylabel(f"NLL")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.7, which="both")
        plt.tight_layout()

        out_path = target_dir / f"summary.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    for inten in unique_intensities:
        # Define raw_sub first
        raw_sub = df[df["intensity"] == inten]
        if raw_sub.empty:
            continue

        sub = raw_sub  # Alias for convenience if needed below

        label_str = format_intensity_label(inten)
        int_file_str = format_intensity_file(inten)

        for scale in scales:
            # 3. Combined Plot (All Models) with Regression
            plt.figure(figsize=(8, 6))

            # Scatter all (Colored by Model)
            # Map colors efficiently
            point_colors = sub["model"].map(model_color_map).tolist()

            plt.scatter(
                sub[metric_key],
                sub["nll"],
                c=point_colors,
                alpha=0.1,
                edgecolors="none",
                s=15,
                rasterized=True,
            )

            # Regression (Linear Scale Only)
            if scale == "linear":
                try:
                    x = sub[metric_key].values
                    y = sub["nll"].values
                    # Remove NaNs
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    x_clean = x[mask]
                    y_clean = y[mask]

                    if len(x_clean) > 1:
                        slope, intercept = np.polyfit(x_clean, y_clean, 1)
                        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                        y_line = slope * x_line + intercept

                        corr = np.corrcoef(x_clean, y_clean)[0, 1]
                        plt.plot(
                            x_line,
                            y_line,
                            "r--",
                            linewidth=2,
                            label=f"Regression (r={corr:.2f})",
                        )
                except Exception as e:
                    pass

            # Combined Binned Mean
            try:
                sub_binned = sub.copy()
                sub_binned["bin"] = pd.cut(sub_binned[metric_key], bins=15)
                binned_all = (
                    sub_binned.groupby("bin", observed=True)
                    .agg(
                        mean_x=(metric_key, "mean"),
                        mean_y=("nll", "mean"),
                        sem_y=("nll", "sem"),
                        count=("nll", "count"),
                    )
                    .reset_index()
                )
                binned_all = binned_all[binned_all["count"] > 0]

                plt.errorbar(
                    binned_all["mean_x"],
                    binned_all["mean_y"],
                    yerr=binned_all["sem_y"],
                    fmt="o-",
                    color="black",
                    capsize=3,
                    linewidth=1.5,
                    label="Binned Mean (All)",
                )
            except:
                pass

            plt.yscale(scale)
            plt.xlabel(config["label"])
            plt.ylabel("NLL")

            # Legend 1: Analysis (Regression / Binned)
            # Only add if there are labeled artists
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                # Place Analysis legend at top right
                leg1 = plt.legend(handles, labels, title="Analysis", loc="upper right")
                plt.gca().add_artist(leg1)

            # Legend 2: Models
            from matplotlib.lines import Line2D

            model_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=MODEL_NAMES.get(m, m.title()),
                    markerfacecolor=model_color_map.get(m, "gray"),
                    markersize=10,
                    markeredgecolor="w",
                )
                for m in models
            ]
            # Place Model legend below Analysis
            # Analysis is at upper right (approx y=1.0 down to 0.8ish)
            # Place Model legend starting around y=0.78 to reduce gap
            plt.legend(
                handles=model_handles,
                title="Model",
                loc="upper right",
                bbox_to_anchor=(1.0, 0.78),
            )
            plt.grid(True, linestyle="--", alpha=0.5, which="both")
            plt.tight_layout()
            plt.savefig(
                target_dir / f"combined_{int_file_str}_{scale}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


@click.command()
@click.option(
    "--consolidated-file",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to consolidated parquet file.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots"),
    help="Directory to save plots.",
)
@click.option("--fast", is_flag=True, help="Fast mode: Low resolution, no PDFs")
@click.option(
    "--parallel/--no-parallel", default=True, help="Enable parallel processing"
)
def main(consolidated_file: Path, output_dir: Path, fast: bool, parallel: bool):
    """Plot correlation between Predictive Performance (NLL) and Image Quality (PSNR)."""

    logger.info(f"Loading {consolidated_file}...")
    df = pd.read_parquet(consolidated_file)

    if df.empty:
        logger.warning("Empty dataframe.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Needs to aggregate: Group by (Dataset, Sparse, Intensity, Model)
    # Metrics: mean(psnr), mean(nll_pred)
    # Note: PSNR and NLL are often lists in the dataframe (per image in run).
    # We need to flatten or mean them appropriately.
    # The dataframe rows are "runs" (chunks of images).

    logger.info("Processing data...")
    records = []

    # Iterate rows and expand lists if necessary, or just compute means per row first
    for _, row in df.iterrows():
        try:
            # Basic Identity Info
            dataset = row["dataset"]
            sparse = row["sparse"]
            intensity = row["total_intensity"]
            model = row["model"]

            # Metrics - convert ndarrays to lists, handle mixed types
            def to_list(val):
                if isinstance(val, np.ndarray):
                    return val.tolist()
                return val

            psnr_raw = to_list(row["psnr"])
            nll_raw = to_list(row["nll_pred"])

            # Optional metrics
            ssim_raw = to_list(row.get("ssim", np.nan))
            l1_raw = to_list(row.get("l1", np.nan))
            rmse_raw = to_list(row.get("rmse", np.nan))

            # Check length - assuming all lists are same length if they are lists
            # We will iterate through them
            # Determine length of explosion
            n_items = 1
            if isinstance(psnr_raw, list):
                n_items = len(psnr_raw)
            elif isinstance(nll_raw, list):
                n_items = len(nll_raw)

            # If scalar, treat as list of 1 for uniform handling
            def ensure_list(val, length):
                if isinstance(val, list):
                    return val
                if pd.isna(val) or val is None:
                    return [np.nan] * length
                return [val] * length

            psnr_list = ensure_list(psnr_raw, n_items)
            nll_list = ensure_list(nll_raw, n_items)
            ssim_list = ensure_list(ssim_raw, n_items)
            l1_list = ensure_list(l1_raw, n_items)
            rmse_list = ensure_list(rmse_raw, n_items)

            for i in range(n_items):
                records.append(
                    {
                        "dataset": dataset,
                        "sparse": sparse,
                        "intensity": intensity,
                        "step_index": i,  # Capture the time step / angle index
                        "model": model,
                        "psnr": float(psnr_list[i]),
                        "nll": float(nll_list[i]),
                        "ssim": float(ssim_list[i]),
                        "l1": float(l1_list[i]),
                        "rmse": float(rmse_list[i]),
                        # "zeroone": ... skipped for now
                    }
                )

        except Exception as e:
            # logger.warning(f"Skipping row: {e}")
            continue

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(records)

    # Now group by (Dataset, Sparse, Side)
    # We want one plot per (Dataset, Sparse)

    groups = analysis_df.groupby(["dataset", "sparse"])

    # Helper for scientific notation
    def format_intensity(val):
        exponent = int(np.log10(val))
        mantissa = val / (10**exponent)
        if abs(mantissa - 1.0) < 0.1:
            return f"$10^{{{exponent}}}$"
        else:
            return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"

    # --- 3. Correlation Analysis (Metric Sweep) ---
    logger.info("--- Metric Correlation with NLL (Spearman) ---")

    available_metrics = [m for m in metrics_config.keys() if m in analysis_df.columns]

    # 1. Compute Correlations (Global)
    global_corrs = {}
    for m in available_metrics:
        c = analysis_df[m].corr(analysis_df["nll"], method="spearman")
        global_corrs[m] = c
        logger.info(f"{m}: {c:.4f}")

    # Visualize Correlations (Global Bar Chart)

    # Visualize Correlations (Global Bar Chart)
    if global_corrs:
        global_dir = output_dir / "global" / "correlations"
        global_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sorted_items = sorted(
            global_corrs.items(), key=lambda x: abs(x[1]), reverse=True
        )
        keys = [k for k, v in sorted_items]
        values = [v for k, v in sorted_items]
        colors = ["g" if v > 0 else "r" for v in values]

        plt.bar(keys, values, color=colors, alpha=0.7)
        plt.axhline(0, color="k", linewidth=0.8)

        plt.ylabel("Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = global_dir / "correlation_bar_global.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {out_path}")
        plt.close()

    # --- Parallel Plotting ---
    tasks = []

    # 1. Per Dataset
    for (ds, sp), group_df in groups:
        suffix = "sparse" if sp else "dense"
        name_suffix = f"{ds}_{suffix}"
        for m in available_metrics:
            # Construct relative path string for folder_name
            rel_folder = Path(ds) / f"correlations_{'sparse' if sp else 'dense'}"
            tasks.append((group_df, m, name_suffix, rel_folder, output_dir, fast))

    # 2. Global
    for m in available_metrics:
        # Global plots
        # folder_name = "global/correlations"
        # Inside function -> output_dir / global/correlations / metric_key
        rel_folder = Path("global") / "correlations"
        tasks.append((analysis_df, m, "global", rel_folder, output_dir, fast))

    logger.info(f"Generating plots for {len(tasks)} tasks (Parallel={parallel})...")

    if parallel:
        # Use max_workers=None for default (os.cpu_count()), or os.cpu_count() - 1 to leave one core free
        num_workers = max(1, os.cpu_count() - 1)
        logger.info(f"Using ProcessPoolExecutor with {num_workers} workers.")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map the plot_metric_vs_nll function to the tasks
            # The function is already top-level and pickleable
            futures = [executor.submit(plot_metric_vs_nll, *t) for t in tasks]
            for i, f in enumerate(futures):
                try:
                    f.result()  # Wait for each task to complete and check for exceptions
                    # logger.info(f"Completed task {i+1}/{len(tasks)}")
                except Exception as e:
                    logger.error(f"Error processing task {i+1}: {e}")
    else:
        logger.info("Running plot generation sequentially.")
        for i, t in enumerate(tasks):
            try:
                plot_metric_vs_nll(*t)
                # logger.info(f"Completed task {i+1}/{len(tasks)}")
            except Exception as e:
                logger.error(f"Error processing task {i+1}: {e}")

    # --- 4. Summary Table ---
    logger.info("--- Model Performance Summary (Averaged across intensities) ---")
    summary_metrics = ["nll"] + [m for m in available_metrics if m != "nll"]

    summary_table = analysis_df.groupby("model")[summary_metrics].mean().reset_index()

    # Save to CSV
    table_path = output_dir / "model_metrics_summary.csv"
    summary_table.to_csv(table_path, index=False)
    logger.info(f"Saved summary table to {table_path}")

    # Print formatted table
    # Print formatted table
    try:
        logger.info("\n" + summary_table.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        logger.info("\n" + summary_table.to_string(index=False, float_format="%.4f"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
