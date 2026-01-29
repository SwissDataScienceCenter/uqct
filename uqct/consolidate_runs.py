from pathlib import Path

import click
import numpy as np
import pandas as pd

from uqct.logging import get_logger
from uqct.utils import get_results_dir, load_runs

logger = get_logger(__name__)


@click.command()
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing run parquet files.",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save the consolidated parquet file.",
)
@click.option("--dataset", required=False, type=str, default=None, help="Dataset name.")
@click.option(
    "--intensity", required=False, type=float, default=None, help="Total intensity."
)
@click.option(
    "--jobid",
    # default=[54717753, 54757527],
    default=None,
    multiple=True,
    help="Run IDs to include. If empty, all runs are included.",
)
@click.option("--sparse/--no-sparse", default=None, help="Sparse setting flag.")
def main(
    runs_dir: Path | None,
    output_file: Path,
    jobid: tuple[int, ...],
    dataset: str | None,
    intensity: float | None,
    sparse: bool | None,
):
    """Consolidate run results into a single parquet file."""

    if runs_dir is None:
        runs_dir = get_results_dir() / "runs"

    logger.info("Configuration:")
    logger.info(f"  Runs Dir:    {runs_dir}")
    logger.info(f"  Output File: {output_file}")
    logger.info(f"  Dataset:     {dataset if dataset else 'ALL'}")
    logger.info(f"  Intensity:   {intensity if intensity is not None else 'ALL'}")
    logger.info(f"  Sparse:      {sparse if sparse is not None else 'ALL'}")

    aggregated_runs = load_runs(runs_dir, dataset, intensity, sparse, jobid)

    if not aggregated_runs:
        logger.warning("No matching runs found.")
        return

    # Consolidate all models into one dataframe
    all_runs: list[pd.DataFrame] = []

    # Logic: Intersect models WITHIN the same (dataset, intensity, sparse) application
    # to ensure fair comparison, then concat everything.

    # 1. Group keys by (dataset, intensity, sparse)
    groups: dict[tuple[str, float, bool], list[pd.DataFrame]] = {}
    for key, df in aggregated_runs.items():
        if not isinstance(key, tuple):
            continue
        ds, mod, inten, sp, seed = key
        group_key = (ds, inten, sp)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(df)

    all_runs = []

    for group_key, dfs in groups.items():
        if not dfs:
            continue

        # Intersect within group
        min_len = min(len(df) for df in dfs)
        if min_len == 0:
            continue

        for df in dfs:
            df_cropped = df.iloc[:min_len].copy()
            all_runs.append(df_cropped)

    logger.info(
        f"Consolidating {len(all_runs)} model-runs across {len(groups)} configurations."
    )

    # Filter out empty dataframes to avoid FutureWarning
    all_runs = [df for df in all_runs if not df.empty]

    # Also filter out all-NA columns from individual frames before concat to avoid FutureWarning
    # "The behavior of DataFrame concatenation with empty or all-NA entries is deprecated."
    all_runs = [df.dropna(axis=1, how="all") for df in all_runs]

    if not all_runs:
        logger.warning("No data to save.")
        return

    # Sort for tidiness
    final_df = pd.concat(all_runs, ignore_index=True)

    # Integrity check on specific columns
    required_cols = [
        "dataset",
        "model",
        "total_intensity",
        "sparse",
        "psnr",
        "ssim",
        "nll_pred",
        "nll_gt",
        "timestamp",
        "rmse",
        "l1",
    ]

    # Only check columns that exist (though they should all exist)
    present_cols = [c for c in required_cols if c in final_df.columns]

    # 1. Check for standard NaNs (top-level)
    if final_df[present_cols].isnull().values.any():
        null_counts = final_df[present_cols].isnull().sum()
        logger.error(
            f"Top-level NaN counts per column:\n{null_counts[null_counts > 0]}"
        )
        raise ValueError(
            "NaN values detected in required columns of the consolidated dataframe!"
        )

    list_cols = ["psnr", "ssim", "nll_pred", "nll_gt", "rmse", "l1"]
    present_list_cols = [c for c in list_cols if c in final_df.columns]

    for col in present_list_cols:
        has_nan = (
            final_df[col]
            .apply(
                lambda x: (
                    np.isnan(x).any() if isinstance(x, list | np.ndarray) else False
                )
            )
            .any()
        )

        if has_nan:
            logger.error(f"NaN values detected INSIDE list column '{col}'!")
            raise ValueError(f"NaN values detected INSIDE list column '{col}'!")

    # Let's save it
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_file)

    logger.info(f"Saved {len(final_df)} rows to {output_file}")


if __name__ == "__main__":
    main()
