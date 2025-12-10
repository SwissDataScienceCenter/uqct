import math
from pathlib import Path
from typing import List, Dict, Optional, Union

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from uqct.utils import get_results_dir
from uqct.loading import load_runs

# Constants
DELTA = 0.05
LOG_INV_DELTA = math.log(1 / DELTA)


def process_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """expands list metrics to scalars (final step) and computes crossover rate."""
    processed_records = []
    
    # Columns to extract (final step)
    scalar_metrics = ["psnr", "ssim", "l1", "rmse", "nll_pred"]
    
    click.echo(f"Processing {len(df)} rows for metrics scaling...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Crossover Rate
            nll_pred = row.get("nll_pred")
            nll_gt = row.get("nll_gt")
            
            rate = np.nan
            if isinstance(nll_pred, (list, np.ndarray)) and isinstance(nll_gt, (list, np.ndarray)):
               if len(nll_pred) > 0:
                    nll_p = np.array(nll_pred)
                    nll_g = np.array(nll_gt)
                    p_cum = np.cumsum(nll_p)
                    g_cum = np.cumsum(nll_g)
                    
                    has_crossover = np.any(g_cum > p_cum + LOG_INV_DELTA)
                    rate = 1.0 if has_crossover else 0.0
            
            # 2. Scalar Metrics (Average over trajectory)
            metrics_vals = {}
            for m in scalar_metrics:
                val = row.get(m)
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    metrics_vals[m] = float(np.mean(val))
                elif isinstance(val, (int, float)):
                    metrics_vals[m] = float(val)
                else:
                    metrics_vals[m] = np.nan

            # Compute Relative NLL: Mean of pointwise (NLL - NLL_GT) / NLL_GT * 100
            val_pred = row.get("nll_pred")
            val_gt = row.get("nll_gt")
            
            nll_rel = np.nan
            if (isinstance(val_pred, (list, np.ndarray)) and isinstance(val_gt, (list, np.ndarray)) 
                and len(val_pred) == len(val_gt) and len(val_pred) > 0):
                
                p_arr = np.array(val_pred)
                g_arr = np.array(val_gt)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_traj = (p_arr - g_arr) / g_arr * 100.0
                    
                # Filter nan/inf if any (though NLL usually finite)
                valid_mask = np.isfinite(rel_traj)
                if np.any(valid_mask):
                     nll_rel = float(np.mean(rel_traj[valid_mask]))
            
            elif (isinstance(val_pred, (int, float)) and isinstance(val_gt, (int, float)) 
                  and abs(val_gt) > 1e-9):
                nll_rel = (val_pred - val_gt) / val_gt * 100.0

            record = {
                "dataset": row["dataset"],
                "model": row["model"],
                "intensity": float(row["total_intensity"]),
                "sparse": bool(row["sparse"]),
                "rate": rate,
                "nll_rel": nll_rel,
                **metrics_vals
            }
            # Rename nll_pred to nll (raw)
            record["nll"] = record.pop("nll_pred", np.nan)
            
            processed_records.append(record)

        except Exception:
            continue
            
    return pd.DataFrame(processed_records)


def plot_scaling_metric(stats_df: pd.DataFrame, metric: str, output_path: Path, title_suffix: str = ""):
    """Generic scaling plotter."""
    if stats_df.empty: return

    plt.figure(figsize=(8, 6))
    models = sorted(stats_df["model"].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']
    
    # Colormap
    import matplotlib
    try:
         model_cmap = matplotlib.colormaps["tab10"].resampled(len(models))
    except:
         try:
             model_cmap = plt.get_cmap("tab10", len(models))
         except:
             model_cmap = cm.get_cmap("tab10", len(models))

    for i, model in enumerate(models):
        sub = stats_df[stats_df["model"] == model].sort_values("intensity")
        if sub.empty: continue
        
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        color = model_cmap(i)
        
        # Plot Mean
        plt.plot(
            sub["intensity"], 
            sub["mean"], 
            label=model,
            marker=marker,
            linestyle=ls,
            color=color,
            alpha=0.9
        )
        
        # Plot SEM Band
        sem = sub["sem"].fillna(0)
        plt.fill_between(
            sub["intensity"],
            sub["mean"] - sem,
            sub["mean"] + sem,
            color=color,
            alpha=0.2
        )

    plt.xscale("log")
    plt.xlabel("Total Intensity (Log Scale)")
    
    # Label formatting
    pretty_name = metric.upper()
    if metric == "rate": pretty_name = "Crossover Rate"
    if metric == "nll": pretty_name = "NLL"
    if metric == "nll_rel": pretty_name = "Relative NLL (%)"

    if metric == "nll_rel":
        plt.yscale("log")
    
    plt.ylabel(f"{pretty_name} (Mean ± SEM)")
    
    # Legend Inside
    plt.legend(loc="best")
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    # click.echo(f"Saved {metric} plot to {output_path}")
    plt.close()


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
def main(runs_dir: Optional[Path], consolidated_file: Optional[Path], output_dir: Path, dataset: Optional[str], sparse: Optional[bool]):
    """Plot scaling laws for all metrics."""
    
    df = pd.DataFrame()

    if consolidated_file and consolidated_file.exists():
        click.echo(f"Loading consolidated file: {consolidated_file}")
        df = pd.read_parquet(consolidated_file)
        # Filter if needed
        if dataset: df = df[df["dataset"] == dataset]
        if sparse is not None: df = df[df["sparse"] == sparse]
    else:
        if runs_dir is None:
            runs_dir = get_results_dir() / "runs"
        # load_runs returns dict of dataframes. We need to concat.
        # But load_runs does aggregation.
        click.echo("Loading runs...")
        # Note: load_runs returns 'clean' dataframes (1 row per image)
        runs_dict = load_runs(runs_dir, dataset, None, sparse)
        if runs_dict:
            df = pd.concat(runs_dict.values(), ignore_index=True)

    if df.empty:
        click.echo("No data found.")
        return

    # Process metrics
    # Extract final step for scaling plots
    metric_df = process_metrics(df)
    
    if metric_df.empty:
        click.echo("No valid metrics extracted.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot
    # Replaced 'nll' with 'nll_rel'
    metrics_to_plot = ["rate", "psnr", "ssim", "l1", "rmse", "nll_rel"]
    
    # --- 1. Per-Dataset Plots ---
    groups = metric_df.groupby(["dataset", "sparse"])
    
    for (ds, sp), group_df in groups:
        suffix = 'sparse' if sp else 'dense'
        click.echo(f"Plotting for {ds} ({suffix})...")
        
        # Output directory: plots/{ds}/scaling_{suffix}
        target_dir = output_dir / ds / f"scaling_{suffix}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Loop metrics
        for m in metrics_to_plot:
            if m not in group_df.columns: continue
            
            # Stats
            stats = group_df.groupby(["model", "intensity"])[m].agg(
                mean="mean", std="std", count="count"
            ).reset_index()
            stats["sem"] = stats["std"] / np.sqrt(stats["count"])
            
            # Filename: scaling_{m}.png (Folder name has dataset info)
            # Or keep full name? "respect directory structure".
            # plot_correlations uses `correlation_...` filenames.
            # I will use `scaling_{m}.png`.
            plot_path = target_dir / f"scaling_{m}.png"
            plot_scaling_metric(stats, m, plot_path, title_suffix=f"({ds}, {suffix})")
            

    # --- 2. Global Plots ---
    click.echo("Generating Global Plots...")
    
    global_dir = output_dir / "global" / "scaling"
    global_dir.mkdir(parents=True, exist_ok=True)
    
    for m in metrics_to_plot:
         if m not in metric_df.columns: continue
         
         stats = metric_df.groupby(["model", "intensity"])[m].agg(
            mean="mean", std="std", count="count"
         ).reset_index()
         stats["sem"] = stats["std"] / np.sqrt(stats["count"])
         
         plot_path = global_dir / f"global_scaling_{m}.png"
         plot_scaling_metric(stats, m, plot_path, title_suffix="(Global)")
         
    click.echo("Done.")

if __name__ == "__main__":
    main()
