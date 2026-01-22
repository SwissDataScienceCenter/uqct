import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import click
import pandas as pd

# Filename pattern:
# method:dataset:intensity:sparse:range(?):seed:timestamp.parquet
# Example: diffusion:composite:10000.0:True:10-20:0:2026-01-21 14:39:44.465770.parquet
PATTERN = re.compile(
    r"(?P<method>[^:]+):"
    r"(?P<dataset>[^:]+):"
    r"(?P<intensity>[^:]+):"
    r"(?P<sparse>[^:]+):"
    r"(?P<range>[^:]+):"
    r"(?P<seed>[^:]+):"
    r"(?P<timestamp>.+)"
    r"\.parquet$"
)


class RunInfo(NamedTuple):
    path: Path
    method: str
    dataset: str
    intensity: float
    sparse: bool
    range_str: str
    seed: int
    timestamp: datetime


def parse_filename(path: Path) -> RunInfo | None:
    match = PATTERN.match(path.name)
    if not match:
        return None

    try:
        data = match.groupdict()
        return RunInfo(
            path=path,
            method=data["method"],
            dataset=data["dataset"],
            intensity=float(data["intensity"]),
            sparse=data["sparse"] == "True",
            range_str=data["range"],
            seed=int(data["seed"]),
            timestamp=datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S.%f"),
        )
    except Exception:
        return None


def get_recent_runs(results_dir: Path) -> list[RunInfo]:
    runs = []
    # Recursively find all parquet files
    for path in results_dir.glob("**/*.parquet"):
        info = parse_filename(path)
        if info:
            runs.append(info)
    return runs


def check_coverage(runs: list[RunInfo]) -> None:
    """
    Checks if the diffusion runs cover the expected image ranges (10-110) with stride 10.
    """
    # Group by (dataset, intensity, seed)
    groups = defaultdict(list)
    for r in runs:
        groups[(r.dataset, r.intensity, r.seed)].append(r)

    full_range_start = 10
    full_range_end = 110
    stride = 10
    expected_starts = set(range(full_range_start, full_range_end, stride))

    for (dataset, intensity, seed), group_runs in groups.items():
        found_starts = set()
        for r in group_runs:
            try:
                start, end = map(int, r.range_str.split("-"))
                if end - start == stride:
                    found_starts.add(start)
            except ValueError:
                continue

        missing = expected_starts - found_starts
        if missing:
            print(
                f"WARNING: Missing ranges for {dataset} I={intensity} seed={seed}: "
                f"Starts {sorted(list(missing))}"
            )
        # else:
        # print(f"Coverage OK for {dataset} I={intensity} seed={seed}")


def filter_by_slurm_id(runs: list[RunInfo], allowed_ids: list[str]) -> list[RunInfo]:
    if not allowed_ids:
        return runs

    kept = []
    print(f"Filtering {len(runs)} runs by SLURM Job IDs: {allowed_ids}...")
    for r in runs:
        try:
            # Read only metadata if possible, but reading parquet is safer
            # 'slurm_job_id' should be in the columns
            # We use fastparquet or pyarrow engine.
            # Just read the file.
            df = pd.read_parquet(r.path)
            # Check for column presence
            found_id = None
            if "slurm_job_id" in df.columns:
                found_id = str(df["slurm_job_id"].iloc[0])

            # The user might have it in a different column or format?
            # Based on previous tasks, it should be 'slurm_job_id'.

            if found_id and found_id in allowed_ids:
                kept.append(r)
        except Exception as e:
            # print(f"Error checking SLURM ID for {r.path}: {e}")
            pass
    return kept


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("results/runs"),
    help="Directory containing run results.",
)
@click.option("--local", is_flag=True, help="Run commands locally instead of printing.")
@click.option("--submit", is_flag=True, help="Submit jobs to SLURM.")
@click.option("--limit", type=int, default=None, help="Limit number of runs.")
@click.option("--dry-run", is_flag=True, help="Print selection logic.")
@click.option(
    "--slurm-run-ids",
    type=str,
    default=None,
    help="Comma-separated list of SLURM job IDs to filter by.",
)
def main(
    results_dir: Path,
    local: bool,
    submit: bool,
    limit: int | None,
    dry_run: bool,
    slurm_run_ids: str | None,
) -> None:
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist.")
        sys.exit(1)

    all_runs = get_recent_runs(results_dir)
    if dry_run:
        print(f"Found {len(all_runs)} total parquet files.")

    # Filter
    # method=diffusion, seed=0, 1e4 <= intensity <= 1e9
    filtered_runs = [
        r
        for r in all_runs
        if r.method == "diffusion" and r.seed == 0 and 1e4 <= r.intensity <= 1e9
    ]

    if dry_run:
        print(f"Filtered to {len(filtered_runs)} runs matching method/seed/intensity.")

    # Deduplicate: (dataset, intensity, range_str) -> latest
    latest_runs: dict[tuple[str, float, str], RunInfo] = {}

    for r in filtered_runs:
        key = (r.dataset, r.intensity, r.range_str)
        if key not in latest_runs:
            latest_runs[key] = r
        else:
            if r.timestamp > latest_runs[key].timestamp:
                latest_runs[key] = r

    final_candidates = list(latest_runs.values())

    # Check Coverage
    if dry_run or True:
        check_coverage(final_candidates)

    # Filter by SLURM ID if provided
    if slurm_run_ids:
        allowed_ids = [s.strip() for s in slurm_run_ids.split(",")]
        final_runs = filter_by_slurm_id(final_candidates, allowed_ids)
    else:
        final_runs = final_candidates

    # Sort for deterministic output
    final_runs.sort(key=lambda r: (r.dataset, r.intensity, r.range_str))

    if limit:
        final_runs = final_runs[:limit]

    print(f"Identified {len(final_runs)} target parquet files to process.")

    commands = []
    for r in final_runs:
        # Command to run diffusion_boundary
        # Output prefix: "boundary_" + input_stem
        input_stem = r.path.stem
        output_prefix = f"boundary_{input_stem}"

        # Use uv run -m uqct.eval.diffusion_boundary
        # Quote the path to handle spaces in filenames
        cmd = (
            f"uv run -m uqct.eval.diffusion_boundary '{r.path.absolute()}' "
            f"--output-prefix '{output_prefix}'"
        )
        commands.append(cmd)

    if local:
        for cmd in commands:
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
    elif submit:
        for cmd in commands:
            # Basic sbatch wrap
            slurm_cmd = (
                f"sbatch --time=04:00:00 --cpus-per-task=4 --mem-per-cpu=8G --gres=gpumem:16g "
                f"--wrap='{cmd}'"
            )
            print(f"Submitting: {slurm_cmd}")
            # dry-run prints? logic above says submit flag runs it.
            subprocess.run(slurm_cmd, shell=True, check=True)

    else:
        # Just list them
        for cmd in commands:
            print(cmd)


if __name__ == "__main__":
    main()
