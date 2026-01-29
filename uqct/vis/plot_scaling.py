import math
from pathlib import Path

import click
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from uqct.eval.run import get_default_angle_schedule
from uqct.logging import get_logger
from uqct.utils import get_results_dir, load_runs
from uqct.vis.style import (
    ICML_COLUMN_WIDTH,
    ICML_TEXT_WIDTH,
    MODEL_NAMES,
    MODEL_ORDER,
    get_style,
)

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
        nll_pred = row.get("nll_pred_mix")
        nll_gt = row.get("nll_gt")

        rate = np.nan
        if isinstance(nll_pred, list | np.ndarray) and isinstance(
            nll_gt, list | np.ndarray
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
            if not (isinstance(val, list | np.ndarray) and len(val) > 0):
                return float(val) if isinstance(val, int | float) else np.nan

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
                if isinstance(val_sch, list | np.ndarray) and len(val_sch) > 0:
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
            "rate": 100 * rate,
            "nll_sum": nll_sum,
            "nll_sum_mix": nll_sum_mix,
            "mean_mix_diff": nll_sum - nll_sum_mix,
            "mean_mix_perc_diff": 100 * (nll_sum - nll_sum_mix) / nll_sum_mix,
            **metrics_vals,
        }

        record["nll_pred_last"] = row.get("nll_pred_last")
        record["nll_pred_last_mix"] = row.get("nll_pred_last_mix")
        record["nll_gt_sum"] = row.get("nll_gt").sum()

        record["snll_gt_nll_diff"] = record["nll_sum"] - record["nll_gt_sum"]
        record["snll_gt_nll_diff_mix"] = record["nll_sum_mix"] - record["nll_gt_sum"]
        record["gt_nll_confcoef_diff"] = (
            record["nll_gt_sum"] - record["nll_sum"] - LOG_INV_DELTA
        )
        record["gt_nll_confcoef_diff_mix"] = (
            record["nll_gt_sum"] - record["nll_sum_mix"] - LOG_INV_DELTA
        )
        record["gt_nll_confcoef_perc_diff"] = (
            100
            * (record["nll_gt_sum"] - (record["nll_sum"] + LOG_INV_DELTA))
            / record["nll_gt_sum"]
        )
        record["gt_nll_confcoef_perc_diff_mix"] = (
            100
            * (record["nll_gt_sum"] - (record["nll_sum_mix"] + LOG_INV_DELTA))
            / record["nll_gt_sum"]
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
        if metric in ("mean_mix_perc_diff", "mean_mix_diff") and model not in (
            "unet_ensemble",
            "diffusion",
        ):
            continue

        # ls = linestyles[i % len(linestyles)]

        style = get_style(model)
        color = style["color"]
        label = style["label"]

        if (label == "Diffusion" or label == "U-Net Ens.") and metric in (
            "snll_gt_diff_mix",
            "rate" "gt_nll_confcoef_diff_mix",
            "gt_nll_confcoef_perc_diff_mix",
        ):
            label = f"{label} (Mix.)"

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
        pretty_name = r"Cross. Rate (\%)"
    if metric == "nll":
        pretty_name = "NLL"
    if metric == "nll_sum":
        pretty_name = "Seq. NLL"
    if metric == "nll_sum_mix":
        pretty_name = "Seq. NLL (Mix.)"
    if metric == "mean_mix_perc_diff":
        pretty_name = r"Seq. NLL Mean-Mix Diff. (\%)"
    if metric == "mean_mix_diff":
        pretty_name = r"$\beta_{t_{\text{final}}}^{\text{mean}} - \beta_{t_{\text{final}}}^{\text{mix}}$"
    if metric == "nll_pred_last":
        pretty_name = "NLL Last Prediction"
    if metric == "nll_pred_last_mix":
        pretty_name = "NLL Last Prediction (Mix.)"
    if metric == "nll_gt_sum":
        pretty_name = "NLL"
    if metric in ["gt_nll_confcoef_diff", "gt_nll_confcoef_diff_mix"]:
        pretty_name = r"$L_{t_{\text{final}}}(\mathbf{x}^\ast) - \beta_{t_{\text{final}}, \delta}$"
    if metric in ["snll_gt_nll_diff", "snll_gt_nll_diff_mix"]:
        pretty_name = (
            r"$\beta_{t_{\text{final}}} - L_{t_{\text{final}}}(\mathbf{x}^\ast)$"
        )
    if metric in ["gt_nll_confcoef_perc_diff", "gt_nll_confcoef_perc_diff_mix"]:
        pretty_name = r"Diff. GT Image NLL and Conf. Coeff. (\%)"
        plt.ylim(bottom=-200, top=25)

    if metric in ("nll_sum", "mean_mix_diff"):
        plt.yscale("log")

    plt.ylabel(f"{pretty_name}")

    # Legend Inside
    plt.legend(loc="best")
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_violation_rate_vs_delta(df: pd.DataFrame, output_dir: Path, range_suffix: str):
    """
    Plots the empirical violation rate vs delta for each total intensity.

    Condition for violation:
        any_t( sum_0^t(nll_gt) > sum_0^t(nll_pred) + log(1/delta) )

    Let Diff_t = sum_0^t(nll_gt) - sum_0^t(nll_pred).
    Violation condition: max_t(Diff_t) > log(1/delta)

    We vary delta and compute the fraction of samples that violate this.
    """
    logger.info("Generating Violation Rate vs Delta plots...")

    # Pre-compute max_diff for valid rows with list inputs
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Pre-computing max diffs"):
        nll_pred = row.get("nll_pred")
        nll_gt = row.get("nll_gt")

        if isinstance(nll_pred, list | np.ndarray) and isinstance(
            nll_gt, list | np.ndarray
        ):
            if len(nll_pred) > 0:
                nll_p = np.array(nll_pred)
                nll_g = np.array(nll_gt)

                # Check shapes match
                if nll_p.shape != nll_g.shape:
                    continue

                diff = np.cumsum(nll_g) - np.cumsum(nll_p)
                max_diff = np.nanmax(
                    diff
                )  # Handle potential NaNs if any, though unlikely with standard data

                records.append(
                    {
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "sparse": row["sparse"],
                        "intensity": float(row["total_intensity"]),
                        "max_diff": max_diff,
                    }
                )

    if not records:
        logger.warning("No valid records found for violation rate plotting.")
        return

    stats_df = pd.DataFrame(records)

    # Deltas to verify
    # Log space from 1e-4 to 1.0 (since delta is probability)
    deltas = np.linspace(0, 1.0, 20)
    log_inv_deltas = np.log(1.0 / deltas)

    datasets_order = ["lamino", "composite", "lung"]

    # Iterate over sparse/dense
    for sparse in [True, False]:
        suffix = "sparse" if sparse else "dense"
        subset_df = stats_df[stats_df["sparse"] == sparse]

        if subset_df.empty:
            continue

        unique_models = subset_df["model"].unique()

        for model in unique_models:
            model_df = subset_df[subset_df["model"] == model]

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(ICML_COLUMN_WIDTH, 3.5),  # Reduced height (was 4.2)
                sharey=True,
                sharex=True,
                constrained_layout=False,  # Use tight_layout manually given rect arg
            )

            # fig.supylabel(r"Empirical Rate", x=0.03)

            # Determine global intensity range for consistent colormap across subplots
            all_intensities = sorted(model_df["intensity"].unique())
            if not all_intensities:
                plt.close(fig)
                continue

            cmap = plt.get_cmap("viridis")
            if len(all_intensities) > 1:
                norm = mcolors.Normalize(
                    vmin=np.log10(min(all_intensities)),
                    vmax=np.log10(max(all_intensities)),
                )
            else:
                norm = mcolors.Normalize(vmin=0, vmax=1)

            has_data = False

            for col_idx, ds_name in enumerate(datasets_order):
                ax = axes[col_idx]
                ds_group = model_df[model_df["dataset"] == ds_name]

                if ds_group.empty:
                    ax.set_visible(False)
                    continue

                has_data = True

                ds_intensities = sorted(ds_group["intensity"].unique())

                for intensity in ds_intensities:
                    sub = ds_group[ds_group["intensity"] == intensity]
                    vals = sub["max_diff"].values

                    # Efficient computation
                    sorted_vals = np.sort(vals)
                    n_samples = len(vals)
                    indices = np.searchsorted(sorted_vals, log_inv_deltas, side="right")
                    counts = n_samples - indices
                    rates = counts / n_samples

                    # Compute SEM for proportion
                    sem = np.sqrt(rates * (1 - rates) / n_samples)

                    color = (
                        cmap(norm(np.log10(intensity)))
                        if len(all_intensities) > 1
                        else "tab:blue"
                    )

                    # Add label for legend
                    # Formatting: 1e6 -> 10^6 if possible, or just scientific
                    label_str = r"$10^{" + f"{int(np.log10(intensity))}" + r"}$"
                    ax.plot(
                        deltas,
                        rates,
                        color=color,
                        marker="x",
                        alpha=0.8,
                        linewidth=1.5,
                        label=label_str,
                        markersize=4,
                    )

                    # Plot Error Band
                    ax.fill_between(
                        deltas,
                        (rates - sem).clip(0, 1),
                        (rates + sem).clip(0, 1),
                        color=color,
                        alpha=0.2,
                    )

                # Plot diagonal
                ax.plot(deltas, deltas, "k--", alpha=0.5)  # label="Target"

                if col_idx == 2:
                    ax.set_xlabel(r"Error Level $\delta$")

                # ax.set_title(f"{ds_name.title()} Dataset")
                ax.text(
                    1.02,
                    0.5,
                    f"{ds_name.title()}",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    #         ha="left",
                    fontsize=9,
                )
                ax.set_ylabel("Cross. Rate")
                ax.grid(True, which="major", alpha=0.3)

            if not has_data:
                plt.close(fig)
                continue

            # Legend Logic: Gather handles from subplots
            handles, labels = [], []
            seen_labels = set()

            # Helper to deduplicate
            # Helper to deduplicate
            def add_handle_label(h, lbl):
                if lbl not in seen_labels and lbl is not None:
                    handles.append(h)
                    labels.append(lbl)
                    seen_labels.add(lbl)

            for ax in axes:
                if ax.lines:
                    h_list, l_list = ax.get_legend_handles_labels()
                    for h, lbl in zip(h_list, l_list):
                        add_handle_label(h, lbl)

            # Sort handles/labels by intensity if possible?
            # They should be inserted in order of plotting (sorted intensity)

            # Layout
            fig.tight_layout(rect=(0, 0.14, 1, 1), h_pad=0.5)

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=3 if len(all_intensities) > 4 else len(all_intensities),
                    frameon=False,
                    title="Total Intensity",
                )

            # Output path
            # Save in global/scaling

            global_dir = output_dir / "global" / "scaling"
            global_dir.mkdir(parents=True, exist_ok=True)

            out_path = (
                global_dir / f"violation_rate_{model}_{suffix}_{range_suffix}.pdf"
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def save_tables(
    df: pd.DataFrame, output_dir: Path, dataset_name: str, aggregation: str
):
    """Generates and saves summary tables for a dataset."""

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
    runs_dir: Path | None,
    consolidated_file: Path | None,
    output_dir: Path,
    dataset: str | None,
    sparse: bool | None,
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
        "mean_mix_diff",
        "mean_mix_perc_diff",
        "nll_sum_mix",
        "nll_pred_last",
        "nll_pred_last_mix",
        "nll_gt_sum",
        "gt_nll_confcoef_diff",
        "gt_nll_confcoef_diff_mix",
        "snll_gt_nll_diff",
        "snll_gt_nll_diff_mix",
        "gt_nll_confcoef_perc_diff",
        "gt_nll_confcoef_perc_diff_mix",
    ]

    range_suffix = "1e6_1e9" if filter_intensities else "1e4_1e9"

    # --- 0. Violation Rate vs Delta Plots ---
    plot_violation_rate_vs_delta(df, output_dir, range_suffix)

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

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(
                ICML_COLUMN_WIDTH,
                3.5,
            ),  # Reduced height (was ICML_COLUMN_HEIGHT ~4.2)
            sharey=True,
            sharex=True,  # <--- Critical: Hides inner x-labels to save space
        )
        # axes is (3,)
        # Pretty Y-Label
        pretty_name = m.upper()
        if m == "psnr":
            pretty_name = "PSNR (dB)"
        if m == "rate":
            pretty_name = r"Cross. Rate (\%)"
        if m == "nll":
            pretty_name = "NLL"
        if m == "nll_sum":
            pretty_name = "Seq. NLL"
        if m == "mean_mix_diff":
            pretty_name = r"$\beta_{t_{\text{final}}}^{\text{mean}} - \beta_{t_{\text{final}}}^{\text{mix}}$"
        if m == "mean_mix_perc_diff":
            pretty_name = r"Seq. NLL. Diff. (\%)"
        if m == "nll_sum_mix":
            pretty_name = "Seq. NLL (Mix.)"
        if m == "nll_pred_last":
            pretty_name = "NLL Last Prediction"
        if m == "nll_pred_last_mix":
            pretty_name = "Seq. NLL Last Prediction (Mix.)"
        if m == "nll_gt_sum":
            pretty_name = "NLL"
        if m in ["snll_gt_nll_diff", "snll_gt_nll_diff_mix"]:
            pretty_name = (
                r"$\beta_{t_{\text{final}}} - L_{t_{\text{final}}}(\mathbf{x}^\ast)$"
            )
        if m in ["gt_nll_confcoef_diff", "gt_nll_confcoef_diff_mix"]:
            pretty_name = r"Difference between GT NLL and Conf. Coeff."
        if m in ["gt_nll_confcoef_perc_diff", "gt_nll_confcoef_perc_diff_mix"]:
            pretty_name = r"Difference between GT NLL and Conf. Coeff. (\%)"

        # fig.supylabel(pretty_name, x=0.03)

        # Consistent Models for Legend
        available_models = set(metric_df["model"].unique())
        models = [mod for mod in MODEL_ORDER if mod in available_models]
        for mod in sorted(list(available_models)):
            if mod not in models:
                models.append(mod)

        # linestyles = ["-", "--", "-.", ":"]

        # Loop Rows (Datasets)
        for row_idx, ds_name in enumerate(datasets_order):
            ax = axes[row_idx]

            if m in ["gt_nll_confcoef_perc_diff", "gt_nll_confcoef_perc_diff_mix"]:
                # Clamp top to 25 to avoid showing empty space due to potential outliers or large SEM
                ax.set_ylim(bottom=-200, top=25)

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

            current_max_y = -float("inf")
            for i, model in enumerate(models):
                sub = stats[stats["model"] == model].sort_values("intensity")
                if sub.empty:
                    continue
                if m in ("mean_mix_diff", "mean_mix_perc_diff") and model not in (
                    "diffusion",
                    "unet_ensemble",
                ):
                    continue

                # ls = linestyles[i % len(linestyles)]
                style = get_style(model)
                color = style["color"]
                label = style["label"]

                # mix_metrics = ["rate", "nll_sum_mix", "nll_pred_last_mix", "gt_nll_confcoef_diff_mix", "snll_gt_nll_diff_mix", "gt_nll_confcoef_perc_diff_mix"]
                # mean_metrics = ["nll_pred_last", "gt_nll_confcoef_diff", "snll_gt_nll_diff", "gt_nll_confcoef_perc_diff" ]
                # if m in mix_metrics:
                #     if model == "diffusion":
                #         label = "Diffusion (Mix.)"
                #     if model == "unet_ensemble":
                #         label = "U-Net Ens. (Mix.)"
                # elif m in mean_metrics:
                #     if model == "diffusion":
                #         label = "Diffusion (Mean)"
                #     if model == "unet_ensemble":
                #         label = "U-Net Ens. (Mean)"

                # if m == "gt_nll_confcoef_perc_diff_mix":
                if m == "mean_mix_diff":
                    print(f"{ds_name=}, {model=}, {m=}:\n\t{list(sub['mean'])}")

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

                # Track max y for this subplot
                upper_bound = (sub["mean"] + sem).max()
                if not np.isnan(upper_bound):
                    current_max_y = max(current_max_y, upper_bound)

            ax.set_xscale("log")
            if row_idx == 2:
                ax.set_xlabel("Total Intensity")

            # ax.set_title(f"{ds_name.title()} Dataset")
            ax.text(
                1.02,
                0.5,
                f"{ds_name.title()}",
                transform=ax.transAxes,
                rotation=-90,
                va="center",
                #         ha="left",
                fontsize=9,
            )
            ax.grid(True, which="major", linestyle="--", alpha=0.3)

            if m in (
                "nll_sum",
                "snll_gt_nll_diff",
                "snll_gt_nll_diff_mix",
                "mean_mix_diff",
            ):
                ax.set_yscale("log")

            ax.set_ylabel(pretty_name)

        handles, labels = axes[-1].get_legend_handles_labels()
        rect = (0, 0.085, 1, 1)
        if m == "mean_mix_diff":
            rect = (0, 0.035, 1, 1)
        fig.tight_layout(rect=rect, h_pad=0.5)

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            frameon=False,
        )

        # Save
        metric_agg_suffix = f"_{aggregation}"
        if m in [
            "nll_sum",
            "nll_pred_last",
            "nll_pred_last_mix",
            "nll_gt_sum",
            "snll_gt_nll_diff",
            "snll_gt_nll_diff_mix",
            "gt_nll_confcoef_diff",
            "gt_nll_confcoef_diff_mix",
            "gt_nll_confcoef_perc_diff",
            "gt_nll_confcoef_perc_diff_mix",
            "mean_mix_perc_diff",
            "mean_mix_diff",
        ]:
            metric_agg_suffix = ""

        out_path = (
            global_dir / f"sparse_shared_{m}{metric_agg_suffix}_{range_suffix}.pdf"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- 3. Combined Mix Metrics Plot ---
    metrics_rows = ["mean_mix_diff", "snll_gt_nll_diff_mix"]

    # Check if metrics exist
    has_metrics = True
    for m in metrics_rows:
        if m not in metric_df.columns:
            has_metrics = False
            break

    if has_metrics:
        datasets_order = ["lamino", "composite", "lung"]

        # Iterate over Sparse/Dense
        for sparse in [True, False]:
            suffix = "sparse" if sparse else "dense"
            subset_df = metric_df[metric_df["sparse"] == sparse]

            if subset_df.empty:
                continue

            fig, axes = plt.subplots(
                2, 3, figsize=(ICML_TEXT_WIDTH, 5), sharex=True, sharey="row"
            )

            # Row 1 Title handling: Dataset Titles
            for c_idx, ds in enumerate(datasets_order):
                axes[0, c_idx].set_title(ds.title())

            # Row Labels
            row_ylabels = [
                r"$\beta_{t_{\text{final}}}^{\text{mean}} - \beta_{t_{\text{final}}}^{\text{mix}}$",
                r"$\beta_{t_{\text{final}}} - L_{t_{\text{final}}}(\mathbf{x}^\ast)$",
            ]

            handles, labels = [], []
            seen_labels = set()

            for r_idx, metric in enumerate(metrics_rows):
                axes[r_idx, 0].set_ylabel(row_ylabels[r_idx])

                for c_idx, ds in enumerate(datasets_order):
                    ax = axes[r_idx, c_idx]

                    ds_group = subset_df[subset_df["dataset"] == ds]
                    if ds_group.empty:
                        continue

                    # Aggregate stats for plotting (Mean/SEM over seeds/runs)
                    # Group by model, intensity
                    stats = (
                        ds_group.groupby(["model", "intensity"])[metric]
                        .agg(mean="mean", std="std", count="count")
                        .reset_index()
                    )
                    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

                    available_models = set(stats["model"].unique())
                    models = [m for m in MODEL_ORDER if m in available_models]
                    for m in sorted(available_models):
                        if m not in models:
                            models.append(m)

                    for model in models:
                        # Filter for relevant models only for these metrics?
                        # mean_mix usually only for ensembles/diffusion?
                        # plot_scaling_metric had a filter:
                        # if metric in ("mean_mix_perc_diff", "mean_mix_diff") and model not in ("unet_ensemble", "diffusion") -> continue
                        if metric == "mean_mix_diff" and model not in (
                            "unet_ensemble",
                            "diffusion",
                        ):
                            continue

                        sub = stats[stats["model"] == model].sort_values("intensity")
                        if sub.empty:
                            continue

                        style = get_style(model)
                        color = style["color"]
                        label = style["label"]

                        if label == "Diffusion" or label == "U-Net Ens.":
                            label = f"{label} (Mix.)"

                        line = ax.plot(
                            sub["intensity"],
                            sub["mean"],
                            label=label,
                            marker="x",
                            color=color,
                            alpha=0.9,
                        )

                        ax.fill_between(
                            sub["intensity"],
                            sub["mean"] - sub["sem"].fillna(0),
                            sub["mean"] + sub["sem"].fillna(0),
                            color=color,
                            alpha=0.2,
                        )

                        # Collect legend handles
                        if label not in seen_labels:
                            handles.extend(line)
                            labels.append(label)
                            seen_labels.add(label)

                # Scales
                for ax in axes[r_idx, :]:
                    ax.set_xscale("log")
                    ax.grid(True, which="major", ls="-", alpha=0.4)

                if r_idx == 0:
                    for ax in axes[r_idx, :]:
                        ax.set_yscale("log")  # mean_mix_diff log scale

            # Shared Footer
            for ax in axes[-1, :]:
                ax.set_xlabel("Total Intensity")

            # Legend
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.05),  # Slightly below
                    ncol=len(labels),
                    frameon=False,
                )

            target_dir = output_dir / "global" / "scaling"
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"combined_mix_metrics_{suffix}_{range_suffix}.pdf"

            plt.tight_layout(rect=(0, 0.035, 1, 1))  # Space for legend
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {out_path}")

    # --- 4. CSV Tables ---
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
