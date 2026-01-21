#!/usr/bin/env python3
import argparse
import glob
import os
import re
import subprocess
import time
from collections import defaultdict
from typing import IO, Any

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def is_updated_within(file_obj: IO[Any], minutes: float) -> bool:
    """
    Checks if an open file was modified within the last 'minutes'.
    
    Args:
        file_obj: An open file object (e.g., from open(...)).
        minutes: The time window in minutes to check against.
        
    Returns:
        True if modified within the window, False otherwise.
    """
    # Get the file descriptor and statistics
    fd = file_obj.fileno()
    stat_info = os.fstat(fd)
    
    # Get modification time and current time
    last_modified = stat_info.st_mtime
    now = time.time()
    
    # Calculate age in seconds
    age_seconds = now - last_modified
    
    # Compare against the threshold (converted to seconds)
    return age_seconds <= (minutes * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor SLURM job logs.")
    parser.add_argument("job_id", type=str, help="SLURM job ID (or array job ID)")
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Expected range of array indices (inclusive)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/cluster/scratch/mgaetzner/logs",
        help="Directory containing log files",
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run in dashboard mode (continuous update)",
    )
    parser.add_argument(
        "--save-failed",
        type=str,
        default=None,
        help="Path to save list of failed job indices",
    )
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Automatically resubmit failed jobs using sparse_eval.sh",
    )
    parser.add_argument(
        "--resubmit-script",
        type=str,
        default="cluster_scripts/sparse_eval.sh",
        help="Path to the submission script (default: cluster_scripts/sparse_eval.sh)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resubmission command without executing it",
    )
    return parser.parse_args()


class JobMonitor:
    def __init__(self, job_id: str, log_dir: str):
        self.job_id = job_id
        self.log_dir = log_dir
        self.cache = {}  # index -> (status, msg, timestamp)
        # index -> {'out': int_offset, 'err': int_offset}
        self.file_offsets = defaultdict(lambda: {"out": 0, "err": 0})

        # Compile regex once for performance
        # Matches: ..._jobID_INDEX.out or .err
        self._filename_regex = re.compile(
            rf"_{re.escape(self.job_id)}_(\d+)\.(out|err)$"
        )

    def get_indices(self, args_range=None):
        if args_range:
            return list(range(args_range[0], args_range[1] + 1))

        # Initial scan to find indices
        pattern = os.path.join(self.log_dir, f"*_{self.job_id}_*")
        files = glob.glob(pattern)
        found_indices = set()
        for fp in files:
            match = self._filename_regex.search(fp)
            if match:
                found_indices.add(int(match.group(1)))
        return sorted(list(found_indices))

    def _scan_active_files(self):
        """
        Performs a SINGLE directory scan to find all current files for this job.
        Returns: dict[index] -> {'out': path, 'err': path}
        """
        active_map = defaultdict(dict)
        pattern = os.path.join(self.log_dir, f"*_{self.job_id}_*")

        # This is the heavy I/O operation (done once per update)
        all_files = glob.glob(pattern)

        for fp in all_files:
            match = self._filename_regex.search(fp)
            if match:
                idx = int(match.group(1))
                ftype = match.group(2)  # 'out' or 'err'
                # If multiple files exist (rare), this keeps the last one found
                active_map[idx][ftype] = fp
        return active_map

    def check_status(self, index, file_paths=None):
        # Return cached terminal state
        if index in self.cache:
            return index, *self.cache[index]

        # Check file logic
        # If no file paths provided, we treat it as Pending immediately
        if not file_paths:
            return index, "Pending", None, None

        status, msg, ts = self._check_files(index, file_paths)

        # Cache terminal states
        if status in ["Success", "Failed"]:
            self.cache[index] = (status, msg, ts)
            if index in self.file_offsets:
                del self.file_offsets[index]

        return index, status, msg, ts

    def _check_files(self, index, file_paths):
        out_file = file_paths.get("out")
        err_file = file_paths.get("err")

        if not out_file and not err_file:
            return "Pending", None, None

        timestamp = None
        if out_file:
            try:
                timestamp = os.path.getmtime(out_file)
            except OSError:
                pass

        # --- 1. Process Output File (Read Chunk) ---
        out_chunk = ""
        if out_file:
            try:
                with open(out_file, "r") as f:
                    out_chunk = f.read()
                    is_fresh = is_updated_within(f, 3)
                if "Saved run data at" in out_chunk and not is_fresh:
                    return "Success", None, timestamp
            except Exception as _:
                pass

        # --- 2. Process Error File (Read Lines) ---
        error_msg = None
        if err_file:
            try:
                with open(err_file, "r") as f:
                    for line in f:
                        if (
                            "Using an engine plan file across different models of devices"
                            in line
                        ):
                            continue
                        if re.search(
                            r"(Aborted|Error|Traceback|Exception)", line, re.IGNORECASE
                        ):
                            error_msg = line.strip()
                            return "Failed", error_msg, timestamp
                    if is_updated_within(f, 3):
                        return "Running", None, None
            except Exception:
                pass

        # --- 3. Check Output Chunk for Errors ---
        if out_chunk and not error_msg:
            try:
                for line in out_chunk.splitlines():
                    if (
                        "Using an engine plan file across different models of devices"
                        in line
                    ):
                        continue
                    if re.search(r"(Aborted|Error|Traceback|Exception)", line, re.IGNORECASE):
                        error_msg = line.strip()
                        return "Failed", error_msg, timestamp
            except Exception:
                pass

        return "Running", None, None

    def update(self, indices):
        stats = defaultdict(list)
        finished_timestamps = []

        # Fill stats from cache first
        for i, (status, msg, ts) in self.cache.items():
            stats[status].append((i, msg))
            if ts is not None:
                finished_timestamps.append(ts)

        todo_indices = [i for i in indices if i not in self.cache]

        # --- OPTIMIZATION START ---
        # 1. Scan the directory ONCE to find what's actually there
        active_files_map = self._scan_active_files()
        # --- OPTIMIZATION END ---

        total_todo = len(todo_indices)

        for j, i in enumerate(todo_indices):
            if j % 50 == 0 or j == total_todo - 1:
                print(f"Checking status: {j + 1}/{total_todo}...", end="\r", flush=True)

            # 2. Only pass the file paths if they exist in our scan
            file_paths = active_files_map.get(i)
            # If the index isn't in our active map, check_status will instantly return "Pending"
            idx, status, msg, ts = self.check_status(i, file_paths)

            stats[status].append((idx, msg))
            if status in ["Success", "Failed"] and ts is not None:
                finished_timestamps.append(ts)

        if total_todo > 0:
            print(" " * 40, end="\r", flush=True)

        return stats, finished_timestamps


def format_time(seconds):
    if seconds is None:
        return "N/A"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m"
    return f"{int(m)}m {int(s)}s"


def print_summary(stats, finished_timestamps, total_indices, job_id, clear=False):
    n_success = len(stats["Success"])
    n_failed = len(stats["Failed"])
    n_running = len(stats["Running"])
    n_pending = len(stats["Pending"])

    output = []
    if clear:
        output.append("\033[2J\033[H")

    output.append(f"Monitoring Job ID: {job_id}")
    output.append(f"Last updated: {time.strftime('%H:%M:%S')}")
    output.append("-" * 40)

    output.append("Summary:")
    output.append(f"  Total:   {total_indices}")
    output.append(
        f"  {GREEN}Success: {n_success}{RESET} ({n_success/total_indices*100:.1f}%)"
    )
    output.append(
        f"  {RED}Failed:  {n_failed}{RESET} ({n_failed/total_indices*100:.1f}%)"
    )
    output.append(
        f"  {CYAN}Running: {n_running}{RESET} ({n_running/total_indices*100:.1f}%)"
    )
    output.append(
        f"  {YELLOW}Pending: {n_pending}{RESET} ({n_pending/total_indices*100:.1f}%)"
    )

    output.append("-" * 40)

    if len(finished_timestamps) >= 2:
        finished_timestamps.sort()
        start_time = finished_timestamps[0]
        end_time = finished_timestamps[-1]
        elapsed_processing = end_time - start_time

        if elapsed_processing > 10:
            throughput = (len(finished_timestamps) - 1) / elapsed_processing
            if throughput > 0:
                n_remaining = n_running + n_pending
                eta_seconds = n_remaining / throughput
                output.append(f"  Estimated Rate: {throughput*60:.2f} jobs/min")
                output.append(f"  Estimated Remaining Time: {format_time(eta_seconds)}")
            else:
                output.append("  Estimated Remaining Time: Unknown (Throughput 0)")
        else:
            output.append(
                "  Estimated Remaining Time: Calculating... (Need more time sample)"
            )
    elif (n_running + n_pending) == 0:
        output.append("  Estimated Remaining Time: 0s (Done)")
    else:
        output.append(
            "  Estimated Remaining Time: Unknown (Need at least 2 finished jobs)"
        )

    failed_tasks = sorted(stats["Failed"], key=lambda x: x[0])
    running_tasks = sorted(stats["Running"], key=lambda x: x[0])

    if n_failed > 0:
        output.append(f"\n{RED}Failed Tasks Details:{RESET}")
        for i, msg in failed_tasks:
            output.append(f"  Index {i}: {msg}")

    if n_running > 0:
        if n_running < 20:
            output.append(
                f"\n{CYAN}Running Indices:{RESET} {[i for i, _ in running_tasks]}"
            )
        else:
            output.append(
                f"\n{CYAN}Running Indices:{RESET} (first 20) {[i for i, _ in running_tasks][:20]} ..."
            )

    print("\n".join(output))


def main():
    args = parse_args()

    monitor = JobMonitor(args.job_id, args.log_dir)
    indices = monitor.get_indices(args.range)

    if not indices:
        print(f"No logs found for job ID {args.job_id} and no range specified.")
        return

    # Load settings and build grid for mapping
    try:
        import tomllib
        from collections import defaultdict
        from uqct.eval.cli import build_grid
        from uqct.utils import get_root_dir

        root = get_root_dir()
        settings_path = root / "uqct" / "settings.toml"

        # Determine sparse/dense based on checking first log or args?
        # Actually we don't know if it's sparse or dense from CLI args here easily without parsing logs.
        # But usually we run sparse. Let's try to infer or default to sparse.
        # Or just load both and check which one matches the grid size?
        # A simpler way: just assume sparse for now as per current context.
        # Better: check the first log file content?
        # fallback: try sparse first.
        sparse = True
        section = "eval-sparse" if sparse else "eval-dense"

        idx_to_model = {}
        if settings_path.exists():
            with open(settings_path, "rb") as f:
                full_config = tomllib.load(f)
                if section in full_config:
                    settings = full_config[section]
                    grid = build_grid(settings)
                    # Map indices
                    for i, task in enumerate(grid):
                        idx_to_model[i] = task["model"]
    except ImportError:
        print(
            f"{YELLOW}Warning: Could not import uqct.eval.cli. Detailed breakdown unavailable.{RESET}"
        )
        idx_to_model = {}
    except Exception as e:
        print(
            f"{YELLOW}Warning: Could not load grid ({e}). Detailed breakdown unavailable.{RESET}"
        )
        idx_to_model = {}

    def print_breakdown(stats, finished_timestamps, total_indices):
        # Overall Summary
        print_summary(
            stats, finished_timestamps, total_indices, args.job_id, clear=False
        )

        if not idx_to_model:
            return

        # Per-Predictor Breakdown
        print("-" * 40)
        print("Predictor Breakdown:")
        print(
            f"{'Predictor':<15} | {'Total':<6} | {'Failed':<6} | {'Finished':<8} | {'Pending':<7} | {'Running':<7} | {'ETA':<10}"
        )
        print("-" * 80)

        # Helper to get model for index
        # Group indices by model
        model_indices = defaultdict(list)
        all_monitored_indices = set()
        for s in stats.values():
            for idx, _ in s:
                all_monitored_indices.add(idx)

        # Only consider indices we are actually monitoring
        for idx in all_monitored_indices:
            model = idx_to_model.get(idx, "Unknown")
            model_indices[model].append(idx)

        # Helper to get status for an index from stats
        idx_status = {}
        for s, items in stats.items():
            for idx, _ in items:
                idx_status[idx] = s

        for model in sorted(model_indices.keys()):
            m_idxs = model_indices[model]
            total = len(m_idxs)
            n_failed = sum(1 for i in m_idxs if idx_status[i] == "Failed")
            n_success = sum(1 for i in m_idxs if idx_status[i] == "Success")
            n_pending = sum(1 for i in m_idxs if idx_status[i] == "Pending")
            n_running = sum(1 for i in m_idxs if idx_status[i] == "Running")

            # Simple ETA calculation for this model
            # Retrieve timestamps for successful jobs of this model
            m_finished_ts = []
            for i in m_idxs:
                if idx_status[i] == "Success":
                    if i in monitor.cache:
                        _, _, ts = monitor.cache[i]
                        if ts:
                            m_finished_ts.append(ts)

            eta_str = "N/A"
            if len(m_finished_ts) >= 2 and (n_running + n_pending) > 0:
                m_finished_ts.sort()
                duration = m_finished_ts[-1] - m_finished_ts[0]
                if duration > 10:
                    thru = (len(m_finished_ts) - 1) / duration
                    if thru > 0:
                        rem = n_running + n_pending
                        eta_sec = rem / thru
                        eta_str = format_time(eta_sec)

            # Colors
            c_fail = RED if n_failed > 0 else ""
            c_succ = GREEN if n_success > 0 else ""
            c_run = CYAN if n_running > 0 else ""
            c_pend = YELLOW if n_pending > 0 else ""

            # Format row
            # predictor | Total | Failed | Finished | Pending | Running | ETA
            # Use colors for numbers only

            row = (
                f"{model:<15} | "
                f"{total:<6} | "
                f"{c_fail}{n_failed:<6}{RESET} | "
                f"{c_succ}{n_success:<8}{RESET} | "
                f"{c_pend}{n_pending:<7}{RESET} | "
                f"{c_run}{n_running:<7}{RESET} | "
                f"{eta_str:<10}"
            )
            print(row)
        print("-" * 80)

    if args.dashboard:
        try:
            while True:
                stats, timestamps = monitor.update(indices)
                if args.dashboard:
                    print("\033[2J\033[H", end="")  # Clear
                print_breakdown(stats, timestamps, len(indices))

                if len(stats["Running"]) == 0 and len(stats["Pending"]) == 0:
                    print("\nAll jobs finished. Exiting dashboard.")
                    break

                time.sleep(5)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")

    else:
        print(f"Monitoring Job ID: {args.job_id}")
        print(f"Checking {len(indices)} tasks...")
        print("-" * 40)
        stats, timestamps = monitor.update(indices)
        print_breakdown(stats, timestamps, len(indices))

    if args.save_failed or args.resubmit:
        failed_indices = []
        for i, (status, _, _) in monitor.cache.items():
            if status == "Failed":
                failed_indices.append(i)

        failed_indices.sort()

        if not failed_indices:
            print(f"\n{GREEN}No failed jobs found. Nothing to save or resubmit.{RESET}")
        else:
            if args.save_failed:
                failed_file = args.save_failed
            else:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                failed_file = f"failed_jobs_{args.job_id}_{timestamp_str}.txt"

            try:
                with open(failed_file, "w") as f:
                    for idx in failed_indices:
                        f.write(f"{idx}\n")
                print(
                    f"\n{RED}Saved {len(failed_indices)} failed job indices to {failed_file}{RESET}"
                )
            except IOError as e:
                print(f"\n{RED}Error writing failed jobs file: {e}{RESET}")
                return

            if args.resubmit:
                abs_failed_file = os.path.abspath(failed_file)
                array_range = f"0-{len(failed_indices) - 1}"

                if not os.path.exists(args.resubmit_script):
                    print(
                        f"\n{RED}Error: Submission script not found at {args.resubmit_script}{RESET}"
                    )
                    return

                cmd = [
                    "sbatch",
                    f"--array={array_range}",
                    args.resubmit_script,
                    abs_failed_file,
                ]

                cmd_str = " ".join(cmd)
                print(f"\nResubmitting failed jobs...")
                print(f"Command: {cmd_str}")

                if args.dry_run:
                    print(f"{YELLOW}[Dry Run] Command not executed.{RESET}")
                else:
                    try:
                        result = subprocess.run(
                            cmd, check=True, capture_output=True, text=True
                        )
                        print(f"{GREEN}Resubmission successful!{RESET}")
                        print(result.stdout.strip())
                    except subprocess.CalledProcessError as e:
                        print(
                            f"\n{RED}Resubmission failed with error code {e.returncode}:{RESET}"
                        )
                        print(e.stderr)


if __name__ == "__main__":
    main()
