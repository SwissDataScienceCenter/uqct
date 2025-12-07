#!/usr/bin/env python3
import argparse
import glob
import os
import re
import time
from pathlib import Path
from collections import defaultdict
import concurrent.futures

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor SLURM job logs.")
    parser.add_argument("job_id", type=str, help="SLURM job ID (or array job ID)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="Expected range of array indices (inclusive)")
    parser.add_argument("--log-dir", type=str, default="/cluster/scratch/mgaetzner/logs", help="Directory containing log files")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for parallel processing")
    return parser.parse_args()

def check_status(job_id, index, log_dir):
    out_pattern = os.path.join(log_dir, f"*_{job_id}_{index}.out")
    err_pattern = os.path.join(log_dir, f"*_{job_id}_{index}.err")
    
    out_files = glob.glob(out_pattern)
    err_files = glob.glob(err_pattern)

    if not out_files and not err_files:
        return index, "Pending", None, None

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
                    return index, "Success", None, timestamp
        except Exception:
            pass

    # Check for failure in err file
    error_msg = None
    if err_file:
        try:
            with open(err_file, "r") as f:
                for line in f:
                    if re.search(r"(Error|Traceback|Exception)", line, re.IGNORECASE):
                        error_msg = line.strip()
                        return index, "Failed", error_msg, timestamp
        except Exception:
            pass
            
    # Check for failure in out file 
    if out_file and not error_msg:
         try:
            with open(out_file, "r") as f:
                for line in f:
                     if re.search(r"(Error|Traceback|Exception)", line, re.IGNORECASE):
                        error_msg = line.strip()
                        return index, "Failed", error_msg, timestamp
         except Exception:
            pass

    return index, "Running", None, None

def format_time(seconds):
    if seconds is None:
        return "N/A"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m"
    return f"{int(m)}m {int(s)}s"

def main():
    args = parse_args()
    
    indices = []
    if args.range:
        indices = list(range(args.range[0], args.range[1] + 1))
    else:
        pattern = os.path.join(args.log_dir, f"*_{args.job_id}_*")
        files = glob.glob(pattern)
        found_indices = set()
        for fp in files:
            match = re.search(rf"_{args.job_id}_(\d+)\.(out|err)$", fp)
            if match:
                found_indices.add(int(match.group(1)))
        indices = sorted(list(found_indices))

    if not indices:
        print(f"No logs found for job ID {args.job_id} and no range specified.")
        return

    stats = defaultdict(list)
    finished_timestamps = []

    print(f"Monitoring Job ID: {args.job_id}")
    print(f"Checking {len(indices)} tasks with {args.threads} threads...")
    print("-" * 40)

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(check_status, args.job_id, i, args.log_dir) for i in indices]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, status, msg, ts = future.result()
                stats[status].append((idx, msg))
                if status in ["Success", "Failed"] and ts is not None:
                    finished_timestamps.append(ts)
            except Exception as e:
                print(f"Error checking status: {e}")

    # Summary
    n_success = len(stats["Success"])
    n_failed = len(stats["Failed"])
    n_running = len(stats["Running"])
    n_pending = len(stats["Pending"])
    n_finished = n_success + n_failed
    n_remaining = n_running + n_pending
    total = len(indices)

    print("\nSummary:")
    print(f"  Total:   {total}")
    print(f"  {GREEN}Success: {n_success}{RESET} ({n_success/total*100:.1f}%)")
    print(f"  {RED}Failed:  {n_failed}{RESET} ({n_failed/total*100:.1f}%)")
    print(f"  {CYAN}Running: {n_running}{RESET} ({n_running/total*100:.1f}%)")
    print(f"  {YELLOW}Pending: {n_pending}{RESET} ({n_pending/total*100:.1f}%)")

    # Runtime Estimation
    print("-" * 40)
    current_time = time.time()
    
    # Filter reasonable timestamps (e.g. not in the future or too far past)
    # Just sort them
    if len(finished_timestamps) >= 2:
        finished_timestamps.sort()
        start_time = finished_timestamps[0]
        end_time = finished_timestamps[-1]
        
        # Duration over which jobs have been finishing
        elapsed_processing = end_time - start_time
        
        # If elapsed time is very small, rate calculation is unstable
        if elapsed_processing > 10: 
            # Throughput: jobs per second
            # We use n_finished - 1 because the first job marks the start of the interval
            throughput = (len(finished_timestamps) - 1) / elapsed_processing
            
            if throughput > 0:
                eta_seconds = n_remaining / throughput
                print(f"  Estimated Rate: {throughput*60:.2f} jobs/min")
                print(f"  Estimated Remaining Time: {format_time(eta_seconds)}")
            else:
                 print("  Estimated Remaining Time: Unknown (Throughput 0)")
        else:
             print("  Estimated Remaining Time: Calculating... (Need more time sample)")
    elif n_remaining == 0:
        print("  Estimated Remaining Time: 0s (Done)")
    else:
        print("  Estimated Remaining Time: Unknown (Need at least 2 finished jobs)")

    # Details
    failed_tasks = sorted(stats["Failed"], key=lambda x: x[0])
    running_tasks = sorted(stats["Running"], key=lambda x: x[0])

    if n_failed > 0:
        print(f"\n{RED}Failed Tasks Details:{RESET}")
        for i, msg in failed_tasks:
            print(f"  Index {i}: {msg}")

    if n_running > 0:
        if n_running < 20:
             print(f"\n{CYAN}Running Indices:{RESET} {[i for i, _ in running_tasks]}")
        else:
             print(f"\n{CYAN}Running Indices:{RESET} (first 20) {[i for i, _ in running_tasks][:20]} ...")

if __name__ == "__main__":
    main()
