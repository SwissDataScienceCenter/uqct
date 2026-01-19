#!/usr/bin/env python3
import argparse
import glob
import os
import re
import subprocess
import time
from collections import defaultdict

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


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

    def get_indices(self, args_range=None):
        if args_range:
            return list(range(args_range[0], args_range[1] + 1))

        pattern = os.path.join(self.log_dir, f"*_{self.job_id}_*")
        files = glob.glob(pattern)
        found_indices = set()
        for fp in files:
            match = re.search(rf"_{self.job_id}_(\d+)\.(out|err)$", fp)
            if match:
                found_indices.add(int(match.group(1)))
        return sorted(list(found_indices))

    def check_status(self, index):
        # Return cached terminal state
        if index in self.cache:
            return index, *self.cache[index]

        # Check file logic
        status, msg, ts = self._check_files(index)

        # Cache terminal states
        if status in ["Success", "Failed"]:
            self.cache[index] = (status, msg, ts)

        return index, status, msg, ts

    def _check_files(self, index):
        out_pattern = os.path.join(self.log_dir, f"*_{self.job_id}_{index}.out")
        err_pattern = os.path.join(self.log_dir, f"*_{self.job_id}_{index}.err")

        out_files = glob.glob(out_pattern)
        err_files = glob.glob(err_pattern)

        if not out_files and not err_files:
            return "Pending", None, None

        # Prefer latest file if multiple matches
        out_file = out_files[0] if out_files else None
        err_file = err_files[0] if err_files else None

        timestamp = None
        if out_file:
            try:
                timestamp = os.path.getmtime(out_file)
            except OSError:
                pass

        # Check for success
        if out_file:
            try:
                with open(out_file, "r") as f:
                    content = f.read()
                    if "Saved run data at" in content:
                        return "Success", None, timestamp
            except Exception:
                pass

        # Check for failure in err file
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
                            r"(Error|Traceback|Exception)", line, re.IGNORECASE
                        ):
                            error_msg = line.strip()
                            return "Failed", error_msg, timestamp
            except Exception:
                pass

        # Check for failure in out file
        if out_file and not error_msg:
            try:
                with open(out_file, "r") as f:
                    for line in f:
                        if (
                            "Using an engine plan file across different models of devices"
                            in line
                        ):
                            continue
                        if re.search(
                            r"(Error|Traceback|Exception)", line, re.IGNORECASE
                        ):
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
        total_todo = len(todo_indices)

        for j, i in enumerate(todo_indices):
            if j % 10 == 0 or j == total_todo - 1:
                print(f"Checking status: {j + 1}/{total_todo}...", end="\r", flush=True)

            idx, status, msg, ts = self.check_status(i)
            stats[status].append((idx, msg))
            if status in ["Success", "Failed"] and ts is not None:
                finished_timestamps.append(ts)

        # Clear the progress line
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
    # If using cache, 'Running' implies we checked and it's still running (or didn't check but it wasn't cached)
    # Actually, check_status returns 'Running' if files exist but no done/error.
    # Note: 'Pending' means no files.

    output = []
    if clear:
        output.append("\033[2J\033[H")  # Clear screen and move cursor home

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

    # Runtime Estimation
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

    # Details
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

    if args.dashboard:
        try:
            while True:
                stats, timestamps = monitor.update(indices)
                print_summary(stats, timestamps, len(indices), args.job_id, clear=True)
                
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
        print_summary(stats, timestamps, len(indices), args.job_id, clear=False)


    # Handle failed jobs (save or resubmit)
    if args.save_failed or args.resubmit:
        failed_indices = []
        for i, (status, _, _) in monitor.cache.items():
            if status == "Failed":
                failed_indices.append(i)

        failed_indices.sort()

        if not failed_indices:
            print(f"\n{GREEN}No failed jobs found. Nothing to save or resubmit.{RESET}")
        else:
            # Determine filename
            if args.save_failed:
                failed_file = args.save_failed
            else:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                failed_file = f"failed_jobs_{args.job_id}_{timestamp_str}.txt"
            
            # Write to file
            try:
                with open(failed_file, "w") as f:
                    for idx in failed_indices:
                        f.write(f"{idx}\n")
                print(f"\n{RED}Saved {len(failed_indices)} failed job indices to {failed_file}{RESET}")
            except IOError as e:
                print(f"\n{RED}Error writing failed jobs file: {e}{RESET}")
                return

            # Resubmit if requested
            if args.resubmit:
                abs_failed_file = os.path.abspath(failed_file)
                array_range = f"0-{len(failed_indices) - 1}"
                
                # Check if script exists
                if not os.path.exists(args.resubmit_script):
                    print(f"\n{RED}Error: Submission script not found at {args.resubmit_script}{RESET}")
                    return

                cmd = [
                    "sbatch",
                    f"--array={array_range}",
                    args.resubmit_script,
                    abs_failed_file
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
                        print(f"\n{RED}Resubmission failed with error code {e.returncode}:{RESET}")
                        print(e.stderr)


if __name__ == "__main__":
    main()

