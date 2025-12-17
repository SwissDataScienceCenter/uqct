import math
from pathlib import Path
from typing import Dict, Optional

import click
import pandas as pd
from uqct.logging import get_logger

logger = get_logger(__name__)


def load_runs(
    runs_dir: Path, dataset: str, intensity: float, sparse: bool
) -> Dict[str, pd.DataFrame]:
    """
    Scans the runs directory and returns the most recent run for each model
    matching the given criteria, aggregating across image chunks.
    """
    if not runs_dir.exists():
        logger.warning(f"Runs directory not found: {runs_dir}")
        return {}

    # Support loading from a single consolidated file
    if runs_dir.is_file():
        logger.info(f"Loading consolidated file: {runs_dir}")
        try:
            df = pd.read_parquet(runs_dir)
            if "model" not in df.columns:
                # Should not happen if created by consolidate_runs
                logger.error("Error: Consolidated file missing 'model' column.")
                return {}

            # Split back into dict
            # GroupBy model
            aggregated_runs = {}
            for model, group in df.groupby("model"):
                aggregated_runs[str(model)] = group.copy()

            return aggregated_runs
        except Exception as e:
            logger.error(f"Error loading consolidated file: {e}")
            return {}

    logger.info(f"Scanning {runs_dir} for parquet files...")

    # Store all candidate files
    candidates = []

    files = list(runs_dir.glob("*.parquet"))
    logger.info(f"Found {len(files)} files in {runs_dir}")

    # Iterate over all parquet files
    for file_path in files:
        try:
            # Filename format: model:dataset:intensity:sparse:range:timestamp.parquet
            # Example: fbp:lung:10000.0:True:0-10:2025-12-08...
            parts = file_path.stem.split(":")
            if len(parts) >= 6:
                # Basic filename filtering
                if dataset and parts[1] != dataset:
                    continue
                if intensity is not None and not math.isclose(
                    float(parts[2]), intensity
                ):
                    continue
                if sparse is not None and (parts[3] == "True") != sparse:
                    continue
            else:
                # Peek valid files?
                pass

            # Load metadata (lightweight if possible, but parquet reads whole file usually?
            # actually pd.read_parquet is okayish for small files)
            # We need to ensure it's the right run configuration
            df = pd.read_parquet(file_path)
            if df.empty:
                logger.debug(f"Skipping {file_path.name}: empty dataframe")
                continue

            row = df.iloc[0]
            row_ds = row["dataset"]
            row_int = row["total_intensity"]
            row_sp = row["sparse"]
            row_mod = row["model"]

            # Strict filtering
            if dataset and row_ds != dataset:
                logger.debug(
                    f"Skipping {file_path.name}: dataset mismatch {row_ds} != {dataset}"
                )
                continue
            if intensity is not None and not math.isclose(row_int, intensity):
                logger.debug(
                    f"Skipping {file_path.name}: intensity mismatch {row_int} != {intensity}"
                )
                continue
            if sparse is not None and row_sp != sparse:
                logger.debug(
                    f"Skipping {file_path.name}: sparse mismatch {row_sp} != {sparse}"
                )
                continue

            timestamp = pd.to_datetime(row["timestamp"])

            # Use metadata for range
            start = -1
            if "image_start_index" in row:
                start = int(row["image_start_index"])

            if start == -1 and len(parts) >= 6:
                try:
                    start, _ = map(int, parts[4].split("-"))
                except ValueError:
                    pass

            if start == -1:
                logger.warning(f"Could not determine image range for {file_path}")
                continue

            candidates.append(
                {
                    "file": file_path,
                    "timestamp": timestamp,
                    "data": df,
                    "config": (row_ds, row_mod, row_int, row_sp),
                    "start": start,
                }
            )
            # logger.debug("Appended candidate")

        except Exception as e:
            # logger.error(f"Error processing {file_path}: {e}")
            continue

    # logger.info(f"Total candidates accepted: {len(candidates)}")

    # Sort candidates by timestamp descending (newest first)
    candidates.sort(key=lambda x: x["timestamp"], reverse=True)

    aggregated_runs = {}

    # Organize by configuration
    config_candidates = {}
    for c in candidates:
        cfg = c["config"]
        if cfg not in config_candidates:
            config_candidates[cfg] = []
        config_candidates[cfg].append(c)

    # For each config, accumulate unique images
    for cfg, file_list in config_candidates.items():
        unique_images = {}

        for item in file_list:
            df = item["data"]
            start_index = item["start"]

            for i, (idx, row) in enumerate(df.iterrows()):
                if "image_index" in row:
                    global_idx = int(row["image_index"])
                else:
                    global_idx = start_index + i

                if global_idx not in unique_images:
                    unique_images[global_idx] = row

        if not unique_images:
            continue

        sorted_indices = sorted(unique_images.keys())
        rows = [unique_images[idx] for idx in sorted_indices]
        full_df = pd.DataFrame(rows)

        aggregated_runs[cfg] = full_df
        # logger.info(f"Aggregated {cfg}: {len(full_df)} images")

    return aggregated_runs
