import math
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from uqct.logging import get_logger

logger = get_logger(__name__)


def _git_root(start: Path) -> Optional[Path]:
    current = start
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _candidate_root_dirs() -> List[Path]:
    candidates: List[Path] = []
    env_root = os.environ.get("UQCT_ROOT_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates += [Path(__file__).parent.parent]
    return candidates


def get_root_dir() -> Path:
    """Return the first existing root directory from the configured candidates."""
    for path in _candidate_root_dirs():
        if path.is_dir():
            return path
    raise FileNotFoundError("None of the configured root directories exist.")


def get_checkpoint_dir() -> Path:
    """Resolve the checkpoint directory with environment variable override."""
    env_ckpt = os.environ.get("UQCT_CKPT_DIR")
    if env_ckpt:
        ckpt_path = Path(env_ckpt).expanduser()
        if ckpt_path.is_dir():
            return ckpt_path

    ckpt_dir = Path("/mydata/chip/shared/checkpoints/uqct/")
    if ckpt_dir.exists():
        return ckpt_dir
    return get_root_dir() / "checkpoints"


def get_results_dir() -> Path:
    """Resolve the results directory with environment variable override."""
    env_results = os.environ.get("UQCT_RESULTS_DIR")
    if env_results:
        results_dir = Path(env_results).expanduser()
    elif Path("/mydata/").exists():
        results_dir = Path("/mydata/chip/shared/results/uqct/")
    else:
        results_dir = get_root_dir() / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def get_cache_dir() -> Path:
    """Resolve a writable cache directory, creating it when the parent exists."""
    candidates: List[Path] = []
    env_cache = os.environ.get("UQCT_CACHE_DIR")
    if env_cache:
        candidates.append(Path(env_cache).expanduser())
    candidates.extend(
        [
            Path("/mydata/chip/shared/data/caches/"),
        ]
    )
    git_root = _git_root(Path(__file__).resolve().parent)
    if git_root:
        candidates.append(git_root / "caches")

    for cache_path in candidates:
        if cache_path.is_dir():
            return cache_path
        parent = cache_path.parent
        parent_exists = parent.exists() if parent != cache_path else cache_path.exists()
        if not parent_exists:
            continue
        try:
            cache_path.mkdir(exist_ok=True)
        except OSError:
            continue
        if cache_path.is_dir():
            return cache_path
    raise FileNotFoundError("Unable to locate or create a cache directory.")


def get_hardware_specific_engine_path(dataset: str) -> Path:
    """
    Returns a unique path inside get_cache_dir() based on the current GPU and TRT version.
    """
    import tensorrt
    import torch

    # 1. Get your project's base cache location
    # Ensure it's a Path object
    base_cache = Path(get_cache_dir())

    if not torch.cuda.is_available():
        # Fallback for CPU/Testing (unlikely to use TRT here, but safe to handle)
        return base_cache / "tensorrt_engines" / "cpu_fallback"

    # 2. Generate Hardware Identity String
    # GPU Name (e.g. "NVIDIA_A100-SXM4-40GB") -> "NVIDIA_A100_SXM4_40GB"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", gpu_name)

    # Compute Capability (e.g. "80" for A100)
    cap = torch.cuda.get_device_capability(0)
    arch_str = f"sm{cap[0]}{cap[1]}"

    # TensorRT Version (e.g. "8.6.1")
    trt_ver = tensorrt.__version__

    # 3. Construct Final Path
    # Structure: /your_cache/tensorrt_engines/NVIDIA_A100_sm80_trt8.6.1
    engine_dir = (
        base_cache
        / "tensorrt_engines"
        / f"{gpu_safe}_{arch_str}_trt{trt_ver}_{dataset}"
    )

    # Create directory if it doesn't exist
    engine_dir.mkdir(parents=True, exist_ok=True)

    return engine_dir


def load_runs(
    runs_dir: Path,
    dataset: str,
    intensity: float,
    sparse: bool,
    job_ids: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
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
        if file_path.name.startswith("bootstrap"):
            continue
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
            job_id = int(df["slurm_job_id"][0])  # type: ignore
            if df.empty or job_id not in job_ids:
                logger.debug(f"Skipping {file_path.name}")
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
