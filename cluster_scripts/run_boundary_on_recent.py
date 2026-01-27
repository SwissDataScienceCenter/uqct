import re
import shlex
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
        except Exception:
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
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=Path("logs/"),
    help="Directory containing logs.",
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
    # method=diffusion, seed=0, 1e6 <= intensity <= 1e9
    filtered_runs = [
        r
        for r in all_runs
        if r.method == "diffusion" and r.seed == 0 and 1e6 <= r.intensity <= 1e9
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
            f"uv run -m uqct.eval.diffusion_boundary {shlex.quote(str(r.path.absolute()))} "
            f"--output-prefix {shlex.quote(output_prefix)}"
        )
        commands.append(cmd)

    # Create a unique ID for this batch
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    commands_file = results_dir / f"commands_boundary_{timestamp}.txt"

    if local:
        for cmd in commands:
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        return

    # Write commands to file
    if submit or dry_run:
        with open(commands_file, "w") as f:
            for cmd in commands:
                f.write(cmd + "\n")
        print(f"Written {len(commands)} commands to {commands_file}")

    if submit:
        # Create SLURM script
        log_dir = Path(os.getenv("UQCT_LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        job_name = "boundary_eval"
        num_jobs = len(commands)
        # SLURM array is 1-indexed usually, but we can use 1-N and sed -n 'Np'
        array_range = f"1-{num_jobs}"

        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%A_%a.out
#SBATCH --error={log_dir}/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --exclude=eu-g4-015,eu-g6-039,eu-g6-063
#SBATCH --array={array_range}

set -euo pipefail

PROJECT_ROOT="${{HOME}}/uq-xray-ct"
cd "${{PROJECT_ROOT}}"
export PYTHONPATH="${{PROJECT_ROOT}}"
source "${{PROJECT_ROOT}}/.venv/bin/activate"

# Extract command from the file
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{commands_file}")
echo "Running: $CMD"
eval "$CMD"
"""

        # Write temporary submission script or pipe to sbatch
        # Using pipe is cleaner to avoid extra files, but we already have the commands file.
        # Let's write the sbath script too for inspectability.
        sbatch_file = results_dir / f"submit_boundary_{timestamp}.sh"
        with open(sbatch_file, "w") as f:
            f.write(slurm_script)

        print(f"Submitting job array {array_range} via {sbatch_file}")
        subprocess.run(f"sbatch {sbatch_file}", shell=True, check=True)

    elif dry_run:
        print("Dry run: Commands file generated.")
        print(f"Would submit array 1-{len(commands)} reading from {commands_file}")
    else:
        # Just list commands if neither local, submit, nor dry-run (default behavior from before?)
        # The original code just printed them.
        for cmd in commands:
            print(cmd)


if __name__ == "__main__":
    main()

