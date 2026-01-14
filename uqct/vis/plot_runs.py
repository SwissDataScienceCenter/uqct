import math
from pathlib import Path
from typing import Optional, Tuple
import concurrent.futures

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from uqct.utils import get_results_dir, load_runs
from uqct.logging import get_logger

logger = get_logger(__name__)

# Set plotting style
# sns.set_theme(style="whitegrid")
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "ggplot"
)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

DELTA = 0.05
LOG_INV_DELTA = math.log(1 / DELTA)


def process_and_plot(
    latest_runs: Dict[Tuple[str, str, float, bool], pd.DataFrame],
    output_dir: Path,
    log_scale: bool = False,
    show_progress: bool = True,
):
    ranking_data = []

    min_len = min(len(df) for df in latest_runs.values()) if latest_runs else 0
    if min_len == 0:
        logger.warning("No images found in runs.")
        return pd.DataFrame()

    logger.info(f"Intersecting results to {min_len} common images.")

    for key, df in latest_runs.items():
        # Unpack key
        if isinstance(key, tuple):
            dataset, model, intensity, sparse = key
        else:
            # Fallback if somehow just model string (legacy?)
            model = str(key)

        # Crop to min_len
        df = df.iloc[:min_len].reset_index(drop=True)
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate metrics for ranking
        # Accumulators (Lists for robust NaN handling)
        psnr_finals = []
        nll_finals = []
        psnr_trajs = []
        nll_trajs = []
        crossover_hits = []
        n_images = 0

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {model}",
            disable=not show_progress,
        ):
            # Parse metrics. Parquet might load lists as numpy arrays or lists.
            # nll_pred and nll_gt are list of lists.
            # In the DF, 'nll_pred' column contains the list for that image.

            nll_pred = row["nll_pred"]
            nll_gt = row["nll_gt"]
            psnr_traj = row["psnr"]

            if isinstance(nll_pred, np.ndarray):
                nll_pred = nll_pred.tolist()
            if isinstance(nll_gt, np.ndarray):
                nll_gt = nll_gt.tolist()
            if isinstance(psnr_traj, np.ndarray):
                psnr_traj = psnr_traj.tolist()

            # Ensure they are lists
            if not isinstance(nll_pred, list) or not isinstance(nll_gt, list):
                continue

            # Compute cumulative sum for NLL
            nll_pred_cum = np.cumsum(nll_pred)
            nll_gt_cum = np.cumsum(nll_gt)

            steps = range(len(nll_pred))

            # Crossover check
            # Exists some t for which NLL_t(gt) > NLL_t(pred) + log(1/delta)
            # Using cumulative values as per request
            has_crossover = any(
                gt > pred + LOG_INV_DELTA for gt, pred in zip(nll_gt_cum, nll_pred_cum)
            )
            crossover_hits.append(1.0 if has_crossover else 0.0)

            # Accumulate metrics
            if psnr_traj:
                psnr_finals.append(psnr_traj[-1])
                psnr_trajs.append(np.mean(psnr_traj))
            else:
                psnr_finals.append(np.nan)
                psnr_trajs.append(np.nan)

            if len(nll_pred_cum) > 0:
                nll_finals.append(nll_pred_cum[-1])
                nll_trajs.append(np.mean(nll_pred_cum))
            else:
                nll_finals.append(np.nan)
                nll_trajs.append(np.nan)

            n_images += 1

            # Plotting
            plt.figure()
            plt.plot(
                steps,
                nll_gt_cum,
                label=r"$L_t(\theta^\ast)$",
                linestyle="--",
                color="black",
            )
            if log_scale:
                plt.yscale("log")

            plt.plot(
                steps,
                nll_pred_cum,
                label=r"$L_t(\theta^\text{pred}_t)$",
                color="tab:blue",
            )

            # Add crossover threshold line relative to Pred
            thresh_line = [p + LOG_INV_DELTA for p in nll_pred_cum]
            plt.plot(
                steps,
                thresh_line,
                label=r"$\beta_t(\delta)$",
                linestyle=":",
                color="tab:red",
            )

            plt.xlabel("Time step $t$")
            plt.ylabel("NLL and confidence coefficient")
            plt.legend()

            # Save plot
            plot_path = model_dir / f"img_{idx:03d}_nll.pdf"
            plt.savefig(plot_path)
            plt.close()

            # Pointwise Difference Plot (Pred - GT)
            nll_diff = [p - g for p, g in zip(nll_pred, nll_gt)]

            plt.figure()
            plt.plot(
                steps,
                nll_diff,
                label=r"$\log p_t(y_t | \theta^\text{pred}_t) - \log p_t(y_t | \theta^\ast)$",
                color="tab:purple",
            )
            plt.axhline(0, color="black", linestyle="--", linewidth=1)

            plt.xlabel("Step")
            plt.ylabel("Log density difference")
            plt.legend()

            diff_plot_path = model_dir / f"img_{idx:03d}_logp_diff.pdf"
            plt.savefig(diff_plot_path)
            plt.close()

            # Gap to Threshold Plot
            # Gap = (Pred + log(1/delta)) - GT
            # If Gap < 0, then GT > Pred + log(1/delta) => Crossover (Violation).
            # If Gap > 0, Safe.
            gap_traj = [
                (p + LOG_INV_DELTA) - g for p, g in zip(nll_pred_cum, nll_gt_cum)
            ]

            plt.figure()
            plt.plot(steps, gap_traj, label="Gap", color="tab:blue")
            plt.axhline(
                0, color="black", linestyle="--", linewidth=1.5, label="Threshold (0)"
            )

            # Shade regions where gap < 0 (violation)
            gap_arr = np.array(gap_traj)
            plt.fill_between(
                steps,
                0,
                gap_arr,
                where=(gap_arr < 0),
                color="tab:red",
                alpha=0.3,
                interpolate=True,
                label="GT Image NOT in CS",
            )
            plt.fill_between(
                steps,
                0,
                gap_arr,
                where=(gap_arr >= 0),
                color="tab:green",
                alpha=0.1,
                interpolate=True,
                label="GT Image in CS",
            )

            plt.xlabel("Time step $t$")
            plt.ylabel(r"$\beta_t(\delta) - L_t(\theta^\ast)$")
            plt.legend()

            gap_plot_path = model_dir / f"img_{idx:03d}_gap.pdf"
            plt.savefig(gap_plot_path)
            plt.close()

        # Avg metrics
        if psnr_finals and np.isnan(psnr_finals).any():
            raise ValueError(
                f"NaNs detected in PSNR for {key}. NaNs: {np.isnan(psnr_finals).sum()}, Values: {psnr_finals}"
            )

        avg_psnr_final = np.nanmean(psnr_finals) if psnr_finals else 0.0
        avg_nll_final = np.nanmean(nll_finals) if nll_finals else 0.0

        avg_psnr_traj = np.nanmean(psnr_trajs) if psnr_trajs else 0.0
        avg_nll_traj = np.nanmean(nll_trajs) if nll_trajs else 0.0

        crossover_rate = np.nanmean(crossover_hits) if crossover_hits else 0.0

        ranking_data.append(
            {
                "model": model,
                "avg_psnr_final": avg_psnr_final,
                "avg_psnr_traj": avg_psnr_traj,
                "avg_nll_final": avg_nll_final,
                "avg_nll_traj": avg_nll_traj,
                "crossover_rate": crossover_rate,
                "n_images": n_images,
            }
        )

    return pd.DataFrame(ranking_data)


def process_single_group(args):
    """
    Worker function for parallel processing.
    args: (dataset, intensity, sparse, df_group, output_dir, log_scale, show_progress)
    """
    d, i, s, setting_df, output_dir, log_scale, show_progress = args

    # Re-import check for safety if spawned
    import matplotlib.pyplot as plt
    from uqct.logging import get_logger

    logger = get_logger(__name__)
    # Ensure logging is setup in worker if needed, though basic config might propagate or need re-init
    # For now, just print is replaced by logger, hoping stdout capture works.

    try:
        # Construct latest_runs for this group
        latest_runs = {}
        # Group by model
        for model_name, model_df in setting_df.groupby("model"):
            key = (d, model_name, i, s)
            latest_runs[key] = model_df

        fmt_i = f"{i:.0e}".replace("+0", "").replace("+", "")
        # Prepare output dir
        # Hierarchical
        setting_dir = output_dir / d / f"{fmt_i}_{'sparse' if s else 'dense'}"
        setting_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing group: {d} {fmt_i} {'sparse' if s else 'dense'}")

        # Call processing
        ranking_df = process_and_plot(
            latest_runs, setting_dir, log_scale, show_progress=show_progress
        )

        if not ranking_df.empty:
            ranking_df = ranking_df.sort_values(by="avg_nll_final", ascending=True)

            # Save CSV
            csv_path = setting_dir / "ranking.csv"
            ranking_df.to_csv(csv_path, index=False)
            return f"Completed {d}/{fmt_i}: Saved ranking."
        else:
            return f"Completed {d}/{fmt_i}: Empty results."

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Error processing {d}/{i}: {str(e)}"


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
    default=None,
    help="Path to consolidated parquet file (faster).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots"),
    help="Directory to save plots.",
)
@click.option(
    "--dataset",
    required=False,
    type=str,
    default=None,
    help="Dataset name (lung, composite, lamino). Default: All.",
)
@click.option(
    "--intensity",
    required=False,
    type=float,
    default=None,
    help="Total intensity. Default: All.",
)
@click.option(
    "--sparse/--no-sparse", default=None, help="Sparse setting flag. Default: All."
)
@click.option(
    "--log-scale/--no-log-scale", default=False, help="Use log scale for NLL plots."
)
@click.option(
    "--parallel/--no-parallel", default=True, help="Enable parallel processing."
)
def main(
    runs_dir: Optional[Path],
    consolidated_file: Optional[Path],
    output_dir: Path,
    dataset: Optional[str],
    intensity: Optional[float],
    sparse: Optional[bool],
    log_scale: bool,
    parallel: bool,
):
    if runs_dir is None:
        runs_dir = get_results_dir() / "runs"

    # Handle explicit "ALL" string if passed (though CLI types might catch it, mainly for dataset)
    if dataset == "ALL":
        dataset = None

    logger.info(f"Configuration:")
    logger.info(f"  Runs Dir:  {runs_dir}")
    logger.info(f"  Output Dir:{output_dir}")
    logger.info(f"  Dataset:   {dataset if dataset else 'ALL'}")
    logger.info(f"  Intensity: {intensity if intensity else 'ALL'}")
    logger.info(f"  Sparse:    {sparse if sparse is not None else 'ALL'}")

    if consolidated_file and consolidated_file.exists():
        logger.info(f"Loading from consolidated file: {consolidated_file}")
        df = pd.read_parquet(consolidated_file)

        # Filter (only if arg provided)
        if dataset is not None:
            df = df[df["dataset"] == dataset]
        if intensity is not None:
            df = df[np.abs(df["total_intensity"] - intensity) < 1e-9]
        if sparse is not None:
            df = df[df["sparse"] == sparse]

        if df.empty:
            logger.warning("No matching runs found in consolidated file.")
            return

        # Ensure grouping columns exist
        req_cols = ["dataset", "total_intensity", "sparse"]
        if not all(c in df.columns for c in req_cols):
            logger.error(
                f"Consolidated file missing columns: {req_cols}. Available: {df.columns.tolist()}"
            )
            return

        # Iterate over unique settings
        groups = list(df.groupby(["dataset", "total_intensity", "sparse"]))
        logger.info(f"Found {len(groups)} groups to process.")

        tasks = []
        use_tqdm = not parallel
        for (d, i, s), setting_df in groups:
            tasks.append((d, i, s, setting_df, output_dir, log_scale, use_tqdm))

        if parallel:
            logger.info("Processing in parallel...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(process_single_group, tasks))
        else:
            logger.info("Processing sequentially...")
            for t in tqdm(tasks):
                process_single_group(t)

    else:
        # Legacy path - requires specific arguments
        if dataset is None or intensity is None or sparse is None:
            logger.error(
                "Error: When not using --consolidated-file, you must specify --dataset, --intensity, and --sparse."
            )
            return

        latest_runs = load_runs(runs_dir, dataset, intensity, sparse)
        if not latest_runs:
            logger.warning("No matching runs found.")
            return

        fmt_i = f"{intensity:.0e}".replace("+0", "").replace("+", "")
        # Prepare output subdirectory
        setting_dir = (
            output_dir / dataset / f"{fmt_i}_{'sparse' if sparse else 'dense'}"
        )
        setting_dir.mkdir(parents=True, exist_ok=True)

        ranking_df = process_and_plot(latest_runs, setting_dir, log_scale)

        if not ranking_df.empty:
            ranking_df = ranking_df.sort_values(by="avg_nll_final", ascending=True)
            logger.info("\nRanking Summary:")
            logger.info(ranking_df.to_string(index=False))
            csv_path = setting_dir / "ranking.csv"
            ranking_df.to_csv(csv_path, index=False)
            logger.info(f"\nRanking saved to {csv_path}")


if __name__ == "__main__":
    main()
