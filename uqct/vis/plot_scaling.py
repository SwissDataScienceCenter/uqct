import math
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from uqct.utils import get_results_dir, load_runs
from uqct.logging import get_logger
from uqct.vis.style import MODEL_ORDER, get_style, MODEL_NAMES
from uqct.eval.run import get_default_angle_schedule

plt.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

logger = get_logger(__name__)

# Constants
DELTA = 0.05
LOG_INV_DELTA = math.log(1 / DELTA)


def process_metrics(
    df: pd.DataFrame, metric_aggregation: str = "average"
) -> pd.DataFrame:
    """expands list metrics to scalars (final step) and computes crossover rate."""
    processed_records = []

    # Columns to extract (final step)
    scalar_metrics = ["psnr", "ssim", "l1", "rmse"]

    logger.info(f"Processing {len(df)} rows for metrics scaling...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Crossover Rate
        nll_pred = row.get("nll_pred")
        nll_gt = row.get("nll_gt")

        rate = np.nan
        if isinstance(nll_pred, (list, np.ndarray)) and isinstance(
            nll_gt, (list, np.ndarray)
        ):
            if len(nll_pred) > 0:
                nll_p = np.array(nll_pred)
                nll_g = np.array(nll_gt)
                p_cum = np.cumsum(nll_p)
                g_cum = np.cumsum(nll_g)

                has_crossover = np.any(g_cum > p_cum + LOG_INV_DELTA)
                rate = 1.0 if has_crossover else 0.0

        # 2. Scalar Metrics (Average over trajectory)
        # Scalar Metrics Logic

        # Helper to compute scalar from trajectory
        def compute_scalar(val, mode, row_data):
            if not (isinstance(val, (list, np.ndarray)) and len(val) > 0):
                return float(val) if isinstance(val, (int, float)) else np.nan

            if mode == "last":
                return float(val[-1])

            if mode == "schedule":
                # Simple mean over predictions as is
                return float(np.mean(val))

            if mode == "average":
                # Expanded weighted mean using schedule
                # Check for angle_schedule column
                angle_schedule = None
                val_sch = row_data.get("angle_schedule")
                if isinstance(val_sch, (list, np.ndarray)) and len(val_sch) > 0:
                    angle_schedule = list(val_sch)
                elif "angle_schedule" not in row_data:
                    angle_schedule = get_default_angle_schedule()

                if angle_schedule:
                    n_angles = 200  # Constant
                    intervals = []
                    for i in range(len(angle_schedule)):
                        start = angle_schedule[i]
                        if i < len(angle_schedule) - 1:
                            end = angle_schedule[i + 1]
                        else:
                            end = n_angles
                        intervals.append(max(0, int(end - start)))

                    if len(val) == len(intervals):
                        expanded = []
                        for v_idx, v in enumerate(val):
                            expanded.extend([v] * intervals[v_idx])
                        if expanded:
                            return float(np.mean(expanded))

                # Fallback if schedule missing or mismatch
                return float(np.mean(val))

            return np.nan

        metrics_vals = {}
        for m in scalar_metrics:
            val = row.get(m)
            metrics_vals[m] = compute_scalar(val, metric_aggregation, row)

        def compute_nll_sum(seq_nlls):
            p_arr = np.array(seq_nlls)
            valid_mask = np.isfinite(p_arr)
            return float(p_arr[valid_mask].mean() * valid_mask.sum())

        nll_sum = compute_nll_sum(row.get("nll_pred"))
        nll_sum_mix = compute_nll_sum(row.get("nll_pred_mix"))

        record = {
            "dataset": row["dataset"],
            "model": row["model"],
            "intensity": float(row["total_intensity"]),
            "sparse": bool(row["sparse"]),
            "rate": rate,
            "nll_sum": nll_sum,
            "nll_sum_mix": nll_sum_mix,
            "mix_mean_ratio": nll_sum_mix / nll_sum,
            **metrics_vals,
        }

        record["nll_pred_last"] = row.get("nll_pred_last")
        record["nll_pred_last_mix"] = row.get("nll_pred_last_mix")
        record["nll_gt_sum"] = row.get("nll_gt").sum()

        # 4. gt_nll_confcoef_diff: sum(nll_gt) - sum(nll_pred) + log(1/delta)
        record["gt_nll_confcoef_diff"] = (
            record["nll_gt_sum"] - record["nll_sum"] + LOG_INV_DELTA
        )
        record["gt_nll_confcoef_diff_mix"] = (
            record["nll_gt_sum"] - record["nll_sum_mix"] + LOG_INV_DELTA
        )
        processed_records.append(record)

    df = pd.DataFrame(processed_records)
    return df


def plot_scaling_metric(stats_df: pd.DataFrame, metric: str, output_path: Path):
    """Generic scaling plotter."""
    if stats_df.empty:
        return

    available_models = set(stats_df["model"].unique())
    models = [m for m in MODEL_ORDER if m in available_models]  # Enforce order

    # Add any extra models
    for m in sorted(available_models):
        if m not in models:
            models.append(m)

    # linestyles = ["-", "--", "-.", ":"]

    for i, model in enumerate(models):
        sub = stats_df[stats_df["model"] == model].sort_values("intensity")
        if sub.empty:
            continue

        # ls = linestyles[i % len(linestyles)]

        style = get_style(model)
        color = style["color"]
        label = style["label"]

        # Plot Mean
        plt.plot(
            sub["intensity"],
            sub["mean"],
            label=label,
            marker="x",
            # linestyle=ls,
            color=color,
            alpha=0.9,
        )

        # Plot SEM Band
        sem = sub["sem"].fillna(0)
        plt.fill_between(
            sub["intensity"],
            sub["mean"] - sem,
            sub["mean"] + sem,
            color=color,
            alpha=0.2,
        )

    plt.xscale("log")
    plt.xlabel("Total Intensity")

    # Label formatting
    pretty_name = metric.upper()
    if metric == "psnr":
        pretty_name = "PSNR (dB)"
    if metric == "rate":
        pretty_name = "Crossover Rate"
    if metric == "nll":
        pretty_name = "NLL"
    if metric == "nll_sum":
        pretty_name = "Seq. NLL"
    if metric == "nll_sum_mix":
        pretty_name = "Seq. NLL (Mix.)"
    if metric == "mix_mean_ratio":
        pretty_name = "Seq. NLL Mix-Mean Ratio"
    if metric == "nll_pred_last":
        pretty_name = "NLL Last Prediction"
    if metric == "nll_pred_last_mix":
        pretty_name = "NLL Last Prediction (Mix.)"
    if metric == "nll_gt_sum":
        pretty_name = "NLL"
    if metric in ["gt_nll_confcoef_diff", "gt_nll_confcoef_diff_mix"]:
        pretty_name = r"$L_{t_\mathrm{final}}(\boldsymbol{\theta}^\ast) - \beta_{t_\mathrm{final}}(\delta)$"

    if metric == "nll_sum":
        plt.yscale("log")

    plt.ylabel(f"{pretty_name}")

    # Legend Inside
    plt.legend(loc="best")
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_tables(
    df: pd.DataFrame, output_dir: Path, dataset_name: str, aggregation: str
):
    """Generates and saves summary tables for a dataset."""

    # 1. Aggregate: Mean over seeds/runs (if multiple)
    # Group By: model, intensity, sparse
    # Metrics to average: psnr, ssim, rmse, nll_sum

    metrics = ["psnr", "ssim", "rmse", "nll_sum"]

    # Filter only rows that have the metrics
    cols = ["model", "intensity", "sparse"] + [m for m in metrics if m in df.columns]
    agg_df = df[cols].copy()

    # Groupby mean
    table_df = agg_df.groupby(["model", "intensity", "sparse"]).mean().reset_index()

    # Filter Models
    table_df = table_df[table_df["model"].isin(MODEL_ORDER)]

    # Sort models by custom order
    table_df["model_cat"] = pd.Categorical(
        table_df["model"], categories=MODEL_ORDER, ordered=True
    )
    table_df = table_df.sort_values(["intensity", "sparse", "model_cat"])

    # Map model names to display names
    # Note: Use a lambda to avoid type errors if map expects specific types
    table_df["Model"] = table_df["model"].apply(lambda x: MODEL_NAMES.get(x, x))

    # Map Sparse to "Sparse Setting" / "Dense Setting"
    table_df["Setting"] = table_df["sparse"].apply(
        lambda x: "Sparse Setting" if x else "Dense Setting"
    )

    # --- Table 1: PSNR Comparison ---
    if "psnr" in table_df.columns:
        psnr_df = table_df.pivot_table(
            index="intensity", columns=["Setting", "Model"], values="psnr"
        )

        # Sort columns manually
        sorted_models = [MODEL_NAMES[m] for m in MODEL_ORDER]
        desired_cols = []
        for setting in ["Sparse Setting", "Dense Setting"]:
            for model in sorted_models:
                desired_cols.append((setting, model))

        existing_cols = [c for c in desired_cols if c in psnr_df.columns]
        psnr_df = psnr_df.reindex(columns=existing_cols)

        out_path = output_dir / f"{dataset_name}_psnr_comparison_{aggregation}.csv"
        psnr_df.to_csv(out_path)
        logger.info(f"Saved {out_path}")

    # --- Table 2: Detailed Metrics ---
    values = [m for m in metrics if m in table_df.columns]

    # Pivot to get (Metric, Setting) in columns initially from values
    # Actually pivot_table with multiple values creates top level = Metric
    detail_df = table_df.pivot_table(
        index=["intensity", "Model"], columns=["Setting"], values=values
    )

    # detail_df columns is MultiIndex: (Metric, Setting)
    # We want top level to be Setting.
    if not detail_df.empty:
        detail_df.columns = detail_df.columns.swaplevel(0, 1)
        detail_df.sort_index(axis=1, level=0, inplace=True)

        metric_order = ["psnr", "ssim", "rmse", "nll_sum"]
        desired_detail_cols = []

        for setting in ["Sparse Setting", "Dense Setting"]:
            for m in metric_order:
                if m in values:
                    desired_detail_cols.append((setting, m))

        existing_detail = [c for c in desired_detail_cols if c in detail_df.columns]
        detail_df = detail_df.reindex(columns=existing_detail)

        out_path_detail = (
            output_dir / f"{dataset_name}_detailed_metrics_{aggregation}.csv"
        )
        detail_df.to_csv(out_path_detail)
        logger.info(f"Saved {out_path_detail}")


@click.command()
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing run parquet files.",
)
@click.option(
    "--consolidated-file",
    type=click.Path(path_type=Path),
    help="Path to consolidated parquet file (faster).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots"),
    help="Directory to save plots.",
)
@click.option(
    "--dataset", required=False, type=str, default=None, help="Dataset name filter."
)
@click.option("--sparse/--no-sparse", default=None, help="Sparse setting filter.")
@click.option(
    "--aggregation",
    type=click.Choice(["last", "average", "schedule"]),
    default="last",
    help="Aggregation method for scalar metrics (avg=weighted expanded, schedule=simple mean, last=final value).",
)
@click.option("--job-ids", default=tuple(), multiple=True, help="Job ids to filter by.")
@click.option(
    "--filter-intensities",
    is_flag=True,
    default=False,
    help="Filter total intensities to range [1e6, 1e9].",
)
def main(
    runs_dir: Optional[Path],
    consolidated_file: Optional[Path],
    output_dir: Path,
    dataset: Optional[str],
    sparse: Optional[bool],
    aggregation: str,
    job_ids: tuple[str],
    filter_intensities: bool,
):
    """Plot scaling laws for all metrics."""

    df = pd.DataFrame()

    if consolidated_file and consolidated_file.exists():
        logger.info(f"Loading consolidated file: {consolidated_file}")
        df = pd.read_parquet(consolidated_file)
        # Filter if needed
        if dataset:
            df = df[df["dataset"] == dataset]
        if sparse is not None:
            df = df[df["sparse"] == sparse]
    else:
        if runs_dir is None:
            runs_dir = get_results_dir() / "runs"
        # load_runs returns dict of dataframes. We need to concat.
        # But load_runs does aggregation.
        logger.info("Loading runs...")
        # Note: load_runs returns 'clean' dataframes (1 row per image)
        runs_dict = load_runs(
            runs_dir, dataset, None, sparse, tuple(int(j) for j in job_ids)
        )
        if runs_dict:
            df = pd.concat(runs_dict.values(), ignore_index=True)

    if df.empty:
        logger.warning("No data found.")
        return

    if filter_intensities:
        logger.info("Filtering intensities to range [1e6, 1e9]...")
        # Ensure total_intensity is numeric
        if "total_intensity" in df.columns:
            df = df[
                (df["total_intensity"] >= 1e6 - 1e-1)
                & (df["total_intensity"] <= 1e9 + 1e-1)
            ]
        else:
            logger.warning("Column 'total_intensity' not found, skipping filter.")

    # Process metrics
    # Extract final step for scaling plots
    metric_df = process_metrics(df, aggregation)

    if metric_df.empty:
        logger.warning("No valid metrics extracted.")
        return

    if metric_df.empty:
        logger.warning("No valid metrics extracted.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics to plot
    # Replaced 'nll' with 'nll_sum'
    metrics_to_plot = [
        "rate",
        "psnr",
        "ssim",
        "l1",
        "rmse",
        "nll_sum",
        "mix_mean_ratio",
        "nll_sum_mix",
        "nll_pred_last",
        "nll_pred_last_mix",
        "nll_gt_sum",
        "gt_nll_confcoef_diff",
        "gt_nll_confcoef_diff_mix",
    ]

    range_suffix = "1e6_1e9" if filter_intensities else "1e4_1e9"

    # --- 1. Per-Dataset Plots ---
    groups = metric_df.groupby(["dataset", "sparse"])

    for (ds, sp), group_df in groups:
        suffix = "sparse" if sp else "dense"
        logger.info(f"Plotting for {ds} ({suffix})...")

        # Output directory: plots/{ds}/scaling_{suffix}
        target_dir = output_dir / ds / f"scaling_{suffix}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Loop metrics
        for m in metrics_to_plot:
            if m not in group_df.columns:
                continue

            # Stats
            stats = (
                group_df.groupby(["model", "intensity"])[m]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
            )
            stats["sem"] = stats["std"] / np.sqrt(stats["count"])

            # Filename: sparse_{m}_{aggregation}_{range_suffix}.pdf
            # If metric is already specific (like nll_pred_last), drop aggregation suffix if it matches or is irrelevant
            metric_agg_suffix = f"_{aggregation}"
            if m in [
                "nll_sum",
                "nll_pred_last",
                "nll_pred_last_mix",
                "nll_gt_sum",
                "gt_nll_confcoef_diff",
                "gt_nll_confcoef_diff_mix",
            ]:
                metric_agg_suffix = ""

            plot_path = target_dir / f"sparse_{m}{metric_agg_suffix}_{range_suffix}.pdf"
            plot_scaling_metric(stats, m, plot_path)

    # --- 2. Global Plots (3 Columns: Lamino, Composite, Lung) ---
    logger.info("Generating Global Plots (Shared Layout)...")

    global_dir = output_dir / "global" / "scaling"
    global_dir.mkdir(parents=True, exist_ok=True)

    datasets_order = ["lamino", "composite", "lung"]

    # We want one plot file per metric
    for m in metrics_to_plot:
        # Check if metric exists in any dataset
        if m not in metric_df.columns:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), constrained_layout=True)
        # axes is (3,)

        # Consistent Models for Legend
        available_models = set(metric_df["model"].unique())
        models = [mod for mod in MODEL_ORDER if mod in available_models]
        for mod in sorted(list(available_models)):
            if mod not in models:
                models.append(mod)

        # linestyles = ["-", "--", "-.", ":"]

        # Loop Columns (Datasets)
        for col_idx, ds_name in enumerate(datasets_order):
            ax = axes[col_idx]

            ds_df = metric_df[(metric_df["dataset"] == ds_name) & (metric_df["sparse"])]
            if ds_df.empty:
                ds_df = metric_df[(metric_df["dataset"] == ds_name)]

            if ds_df.empty:
                ax.set_visible(False)
                continue

            # Aggregate stats
            stats = (
                ds_df.groupby(["model", "intensity"])[m]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
            )
            stats["sem"] = stats["std"] / np.sqrt(stats["count"])

            for i, model in enumerate(models):
                sub = stats[stats["model"] == model].sort_values("intensity")
                if sub.empty:
                    continue

                # ls = linestyles[i % len(linestyles)]
                style = get_style(model)
                color = style["color"]
                label = style["label"]

                # Plot Mean
                ax.plot(
                    sub["intensity"],
                    sub["mean"],
                    label=label,
                    marker="x",
                    color=color,
                    alpha=0.9,
                    markersize=4,
                )

                # Plot SEM Band
                sem = sub["sem"].fillna(0)
                ax.fill_between(
                    sub["intensity"],
                    sub["mean"] - sem,
                    sub["mean"] + sem,
                    color=color,
                    alpha=0.2,
                )

            ax.set_xscale("log")
            ax.set_xlabel("Total Intensity")
            ax.set_title(f"{ds_name.title()} Dataset")
            ax.grid(True, which="major", linestyle="--", alpha=0.3)

            # Pretty Y-Label
            pretty_name = m.upper()
            if m == "psnr":
                pretty_name = "PSNR (dB)"
            if m == "rate":
                pretty_name = "Crossover Rate"
            if m == "nll":
                pretty_name = "NLL"
            if m == "nll_sum":
                pretty_name = "Seq. NLL"
            if m == "mix_mean_ratio":
                pretty_name = "Seq. NLL Mix-Mean Ratio"
            if m == "nll_sum_mix":
                pretty_name = "Seq. NLL (Mix.)"
            if m == "nll_pred_last":
                pretty_name = "NLL Last Prediction"
            if m == "nll_pred_last_mix":
                pretty_name = "NLL Last Prediction (Mix.)"
            if m == "nll_gt_sum":
                pretty_name = "NLL"
            if m in ["gt_nll_confcoef_diff", "gt_nll_confcoef_diff_mix"]:
                pretty_name = r"$L_{t_\mathrm{final}}(\boldsymbol{\theta}^\ast) - \beta_{t_\mathrm{final}}(\delta)$"

            if m == "nll_sum":
                ax.set_yscale("log")

            if col_idx == 0:
                ax.set_ylabel(pretty_name)
                ax.legend(fontsize=8)

        # Save
        metric_agg_suffix = f"_{aggregation}"
        if m in [
            "nll_sum",
            "nll_pred_last",
            "nll_pred_last_mix",
            "nll_gt_sum",
            "gt_nll_confcoef_diff",
            "gt_nll_confcoef_diff_mix",
            "mix_mean_ratio",
        ]:
            metric_agg_suffix = ""

        out_path = (
            global_dir / f"sparse_shared_{m}{metric_agg_suffix}_{range_suffix}.pdf"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        # logger.info(f"Saved {out_path}")
        plt.close(fig)

    # --- 3. CSV Tables ---
    logger.info("Generating CSV Tables...")

    # Per Dataset Tables
    unique_datasets = metric_df["dataset"].unique()
    for ds in unique_datasets:
        ds_dir = output_dir / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_df = metric_df[metric_df["dataset"] == ds]
        save_tables(ds_df, ds_dir, ds, aggregation)

    # Global Table (Aggregate across all datasets?)
    # Or just save the full dump
    global_tables_dir = output_dir / "global"
    global_tables_dir.mkdir(parents=True, exist_ok=True)
    save_tables(metric_df, global_tables_dir, "global", aggregation)

    logger.info("Done.")


if __name__ == "__main__":
    main()
