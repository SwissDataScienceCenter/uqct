
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, List, Dict
from scipy.stats import spearmanr
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Metrics configuration
metrics_config = {
    "psnr": {"label": "PSNR", "direction": "Higher is Better"},
    "ssim": {"label": "SSIM", "direction": "Higher is Better"},
    "rmse": {"label": "RMSE", "direction": "Lower is Better"},
    "l1": {"label": "L1", "direction": "Lower is Better"},
    # "zeroone": {"label": "ZeroOne", "direction": "Lower is Better"} 
}

# Refactored Plotting Function (Top Level for Pickleability)
def plot_metric_vs_nll(df, metric_key, name_suffix, folder_name, output_dir, fast_mode=False):
    if metric_key not in df.columns: return

    # Re-import locally for safety in processes (optional but safe)
    import matplotlib.pyplot as plt
    try:
        import matplotlib.cm as cm
    except:
        pass

    config = metrics_config.get(metric_key, {"label": metric_key, "direction": ""})
    label_text = f"Median {config['label']} ({config['direction']})"
    
    # Prepare directory
    target_dir = output_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    scales = ["log", "linear"]
    
    # helper
    def format_intensity(val):
        exponent = int(np.log10(val))
        mantissa = val / (10**exponent)
        if abs(mantissa - 1.0) < 0.1:
            return f"$10^{{{exponent}}}$"
        else:
            return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"
    
    # --- Detail Plot (Scatter) & Binned Plot (Mean +/- SEM) ---
    unique_intensities = sorted(df["intensity"].unique())
    models = sorted(df["model"].unique())
    
    # safe cmap retrieval
    import matplotlib
    from matplotlib.lines import Line2D
    try:
         model_cmap = matplotlib.colormaps["tab10"].resampled(len(models))
    except (AttributeError, KeyError):
         try:
             model_cmap = plt.get_cmap("tab10", len(models))
         except:
             model_cmap = cm.get_cmap("tab10", len(models))
             
    model_color_map = {m: model_cmap(i) for i, m in enumerate(models)}

    # --- Summary Plot (Median Trade-off) ---
    summary = df.groupby(["intensity", "model"]).agg({
        metric_key: "median", 
        "nll": "median"
    }).reset_index()
    
    if not summary.empty:
        intensities = sorted(summary["intensity"].unique())
        try:
            try:
                cmap = matplotlib.colormaps["viridis_r"].resampled(len(intensities))
            except:
                cmap = plt.get_cmap("viridis_r", len(intensities))
        except:
             cmap = cm.get_cmap("viridis_r", len(intensities))
        
        for scale in scales:
            plt.figure(figsize=(8, 6))
            
            # Plot Lines (Intensity) and Points (Model)
            for i, inten in enumerate(intensities):
                # Median Data for Line
                med_sub = summary[summary["intensity"] == inten].sort_values(metric_key)
                
                # Raw Data for Scatter
                raw_sub = df[df["intensity"] == inten]
                
                if med_sub.empty: continue
                
                # Line color = Intensity
                line_color = cmap(i)
                label_str = format_intensity(inten)
                
                # Median Line (Frontier)
                plt.plot(med_sub[metric_key], med_sub["nll"], color=line_color, alpha=0.8, linestyle='--', linewidth=2, label=label_str)
                
                # Raw Points - Colored by Model
                point_colors = [model_color_map[m] for m in raw_sub["model"]]
                plt.scatter(
                    raw_sub[metric_key], 
                    raw_sub["nll"], 
                    c=point_colors, 
                    marker='o', 
                    s=15, 
                    alpha=0.15,
                    edgecolors='none', 
                    zorder=1, 
                    rasterized=True
                )

            # Summary Plot Legend
            # Remove title
            # Legends inside
            leg1 = plt.legend(title="Intensity", loc="upper right")
            plt.gca().add_artist(leg1)
            
            # Model Legend
            model_handles = [
                Line2D([0], [0], marker='o', color='w', label=m,
                       markerfacecolor=model_color_map[m], markersize=10, markeredgecolor='w')
                for m in models
            ]
            plt.legend(handles=model_handles, title="Model", loc="lower left")

            plt.xlabel(label_text)
            plt.ylabel(f"Median NLL ({scale.capitalize()} Scale)")
            plt.yscale(scale)
            # plt.title(...) REMOVED
            plt.grid(True, linestyle='--', alpha=0.7, which="both")
            plt.tight_layout()
            
            out_path = target_dir / f"correlation_{metric_key}_nll_{name_suffix}_{scale}.png"
            plt.savefig(out_path, dpi=150 if fast_mode else 300, bbox_inches='tight')
            if not fast_mode:
                plt.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
    
    for inten in unique_intensities:
        sub = df[df["intensity"] == inten]
        if sub.empty: continue
        label_str = format_intensity(inten)
        int_str = f"{int(inten)}"
        
        for scale in scales:
            # 1. Detail Scatter
            plt.figure(figsize=(8, 6))
            for m in models:
                m_sub = sub[sub["model"] == m]
                if m_sub.empty: continue
                plt.scatter(
                    m_sub[metric_key], 
                    m_sub["nll"], 
                    label=m,
                    color=model_color_map[m],
                    alpha=0.3, # More transparent for exploded data
                    edgecolors='none',
                    s=20,
                    rasterized=True
                )
            plt.yscale(scale)
            plt.xlabel(config['label'])
            plt.ylabel(f"NLL ({scale.capitalize()} Scale)")

            plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5, which="both")
            plt.tight_layout()
            plt.savefig(target_dir / f"correlation_detail_{metric_key}_{name_suffix}_int_{int_str}_{scale}.png", dpi=150 if fast_mode else 300, bbox_inches='tight')
            if not fast_mode:
                plt.savefig(target_dir / f"correlation_detail_{metric_key}_{name_suffix}_int_{int_str}_{scale}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Binned Plot (Mean +/- SEM)
            plt.figure(figsize=(8, 6))
            for m in models:
                m_sub = sub[sub["model"] == m].copy()
                if m_sub.empty or len(m_sub) < 5: continue
                
                # Binning
                try:
                    m_sub["bin"] = pd.cut(m_sub[metric_key], bins=15)
                    binned = m_sub.groupby("bin", observed=True).agg(
                        mean_x=(metric_key, "mean"),
                        mean_y=("nll", "mean"),
                        sem_y=("nll", "sem"),
                        count=("nll", "count")
                    ).reset_index()
                    binned = binned[binned["count"] > 0]
                    
                    plt.plot(binned["mean_x"], binned["mean_y"], color=model_color_map[m], label=m, marker='o')
                    plt.fill_between(
                        binned["mean_x"],
                        binned["mean_y"] - binned["sem_y"],
                        binned["mean_y"] + binned["sem_y"],
                        color=model_color_map[m],
                        alpha=0.2
                    )
                except Exception as e:
                    pass

            plt.yscale(scale)
            plt.xlabel(f"Binned {config['label']}")
            plt.ylabel(f"Mean NLL ± SEM ({scale.capitalize()} Scale)")

            plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5, which="both")
            plt.tight_layout()
            plt.savefig(target_dir / f"correlation_binned_{metric_key}_{name_suffix}_int_{int_str}_{scale}.png", dpi=150 if fast_mode else 300, bbox_inches='tight')
            if not fast_mode:
                plt.savefig(target_dir / f"correlation_binned_{metric_key}_{name_suffix}_int_{int_str}_{scale}.pdf", dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Combined Plot (All Models) with Regression
            # 3. Combined Plot (All Models) with Regression
            plt.figure(figsize=(8, 6))
            
            # Scatter all (Colored by Model)
            # Map colors efficiently
            point_colors = sub["model"].map(model_color_map).tolist()
            
            plt.scatter(
                sub[metric_key], 
                sub["nll"], 
                c=point_colors,
                alpha=0.1, 
                edgecolors='none',
                s=15,
                rasterized=True
            )
            
            # Regression (Linear Scale Only)
            if scale == 'linear':
                try:
                    x = sub[metric_key].values
                    y = sub["nll"].values
                    # Remove NaNs
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    x_clean = x[mask]
                    y_clean = y[mask]

                    if len(x_clean) > 1:
                        slope, intercept = np.polyfit(x_clean, y_clean, 1)
                        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                        y_line = slope * x_line + intercept
                        
                        corr = np.corrcoef(x_clean, y_clean)[0, 1]
                        plt.plot(x_line, y_line, 'r--', linewidth=2, label=f'Regression (r={corr:.2f})')
                except Exception as e:
                    pass

            # Combined Binned Mean
            try:
                sub_binned = sub.copy()
                sub_binned["bin"] = pd.cut(sub_binned[metric_key], bins=15)
                binned_all = sub_binned.groupby("bin", observed=True).agg(
                    mean_x=(metric_key, "mean"),
                    mean_y=("nll", "mean"),
                    sem_y=("nll", "sem"),
                    count=("nll", "count")
                ).reset_index()
                binned_all = binned_all[binned_all["count"] > 0]
                
                plt.errorbar(
                    binned_all["mean_x"], 
                    binned_all["mean_y"], 
                    yerr=binned_all["sem_y"],
                    fmt='o-', 
                    color='black',
                    capsize=3,
                    linewidth=1.5,
                    label='Binned Mean (All)'
                )
            except:
                pass

            plt.yscale(scale)
            plt.xlabel(config['label'])
            plt.ylabel(f"NLL ({scale.capitalize()} Scale)")

            
            # Legend 1: Analysis (Regression / Binned)
            # Only add if there are labeled artists
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                leg1 = plt.legend(handles, labels, title="Analysis", bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.gca().add_artist(leg1)
            
            # Legend 2: Models
            from matplotlib.lines import Line2D
            model_handles = [
                Line2D([0], [0], marker='o', color='w', label=m,
                       markerfacecolor=model_color_map[m], markersize=10, markeredgecolor='w')
                for m in models
            ]
            plt.legend(handles=model_handles, title="Model", bbox_to_anchor=(1.02, 0.6), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5, which="both")
            plt.tight_layout()
            plt.savefig(target_dir / f"correlation_combined_{metric_key}_{name_suffix}_int_{int_str}_{scale}.png", dpi=150 if fast_mode else 300, bbox_inches='tight')
            if not fast_mode:
                plt.savefig(target_dir / f"correlation_combined_{metric_key}_{name_suffix}_int_{int_str}_{scale}.pdf", dpi=300, bbox_inches='tight')
            plt.close()


@click.command()
@click.option(
    "--consolidated-file",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to consolidated parquet file.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots"),
    help="Directory to save plots.",
)
@click.option("--fast", is_flag=True, help="Fast mode: Low resolution, no PDFs")
@click.option("--parallel/--no-parallel", default=True, help="Enable parallel processing")
def main(consolidated_file: Path, output_dir: Path, fast: bool, parallel: bool):
    """Plot correlation between Predictive Performance (NLL) and Image Quality (PSNR)."""
    
    click.echo(f"Loading {consolidated_file}...")
    df = pd.read_parquet(consolidated_file)
    
    if df.empty:
        click.echo("Empty dataframe.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Needs to aggregate: Group by (Dataset, Sparse, Intensity, Model)
    # Metrics: mean(psnr), mean(nll_pred)
    # Note: PSNR and NLL are often lists in the dataframe (per image in run).
    # We need to flatten or mean them appropriately.
    # The dataframe rows are "runs" (chunks of images).
    
    click.echo("Processing data...")
    records = []
    
    # Iterate rows and expand lists if necessary, or just compute means per row first
    for _, row in df.iterrows():
        try:
            # Basic Identity Info
            dataset = row["dataset"]
            sparse = row["sparse"]
            intensity = row["total_intensity"]
            model = row["model"]
            
            # Metrics - convert ndarrays to lists, handle mixed types
            def to_list(val):
                if isinstance(val, np.ndarray): return val.tolist()
                return val

            psnr_raw = to_list(row["psnr"])
            nll_raw = to_list(row["nll_pred"])
            
            # Optional metrics
            ssim_raw = to_list(row.get("ssim", np.nan))
            l1_raw = to_list(row.get("l1", np.nan))
            rmse_raw = to_list(row.get("rmse", np.nan))
            
            # Check length - assuming all lists are same length if they are lists
            # We will iterate through them
            # Determine length of explosion
            n_items = 1
            if isinstance(psnr_raw, list): n_items = len(psnr_raw)
            elif isinstance(nll_raw, list): n_items = len(nll_raw)
            
            # If scalar, treat as list of 1 for uniform handling
            def ensure_list(val, length):
                if isinstance(val, list): return val
                if pd.isna(val) or val is None: return [np.nan] * length
                return [val] * length

            psnr_list = ensure_list(psnr_raw, n_items)
            nll_list = ensure_list(nll_raw, n_items)
            ssim_list = ensure_list(ssim_raw, n_items)
            l1_list = ensure_list(l1_raw, n_items)
            rmse_list = ensure_list(rmse_raw, n_items)
            
            for i in range(n_items):
                records.append({
                    "dataset": dataset,
                    "sparse": sparse,
                    "intensity": intensity,
                    "step_index": i,  # Capture the time step / angle index
                    "model": model,
                    "psnr": float(psnr_list[i]),
                    "nll": float(nll_list[i]),
                    "ssim": float(ssim_list[i]),
                    "l1": float(l1_list[i]),
                    "rmse": float(rmse_list[i]),
                    # "zeroone": ... skipped for now
                })
            
        except Exception as e:
            # click.echo(f"Skipping row: {e}")
            continue
            
    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(records)
    
    # Now group by (Dataset, Sparse, Side)
    # We want one plot per (Dataset, Sparse)
    
    groups = analysis_df.groupby(["dataset", "sparse"])

    # Helper for scientific notation
    def format_intensity(val):
        exponent = int(np.log10(val))
        mantissa = val / (10**exponent)
        if abs(mantissa - 1.0) < 0.1:
            return f"$10^{{{exponent}}}$"
        else:
            return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"



    # --- 3. Correlation Analysis (Metric Sweep) ---
    click.echo("\n--- Metric Correlation with NLL (Spearman) ---")
    
    available_metrics = [m for m in metrics_config.keys() if m in analysis_df.columns]
    
    # 1. Compute Correlations (Global)
    global_corrs = {}
    for m in available_metrics:
        c = analysis_df[m].corr(analysis_df["nll"], method="spearman")
        global_corrs[m] = c
    
    # Visualize Correlations (Global Bar Chart)
    if global_corrs:
        global_dir = output_dir / "global" / "correlations"
        global_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sorted_items = sorted(global_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        keys = [k for k, v in sorted_items]
        values = [v for k, v in sorted_items]
        colors = ['g' if v > 0 else 'r' for v in values]
        
        plt.bar(keys, values, color=colors, alpha=0.7)
        plt.axhline(0, color='k', linewidth=0.8)

        plt.ylabel("Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        out_path = global_dir / "correlation_bar_global.png"
        plt.savefig(out_path, dpi=150 if fast else 300, bbox_inches='tight')
        if not fast:
            plt.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        click.echo(f"Saved {out_path} and .pdf")
        plt.close()

    # --- Parallel Plotting ---
    tasks = []
    
    # 1. Per Dataset
    for (ds, sp), group_df in groups:
        suffix = 'sparse' if sp else 'dense'
        name_suffix = f"{ds}_{suffix}"
        # Update target_dir definition here
        target_dir = output_dir / ds / f"correlations_{'sparse' if sp else 'dense'}"
        target_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        for m in available_metrics:
            tasks.append((group_df, m, name_suffix, ".", target_dir, fast)) # Pass "." as folder_name to avoid nesting

    # 2. Global
    for m in available_metrics:
        # For global plots, use the global_dir defined earlier
        global_plot_target_dir = output_dir / "global" / "correlations"
        global_plot_target_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        tasks.append((analysis_df, m, "global", ".", global_plot_target_dir, fast)) # Pass "." here too
        
    click.echo(f"Generating plots for {len(tasks)} tasks (Parallel={parallel})...")
    
    if parallel:
        # Use max_workers=None for default (os.cpu_count()), or os.cpu_count() - 1 to leave one core free
        num_workers = max(1, os.cpu_count() - 1) 
        click.echo(f"Using ProcessPoolExecutor with {num_workers} workers.")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map the plot_metric_vs_nll function to the tasks
            # The function is already top-level and pickleable
            futures = [executor.submit(plot_metric_vs_nll, *t) for t in tasks]
            for i, f in enumerate(futures):
                try:
                    f.result() # Wait for each task to complete and check for exceptions
                    # click.echo(f"Completed task {i+1}/{len(tasks)}")
                except Exception as e:
                    click.echo(f"Error processing task {i+1}: {e}")
    else:
        click.echo("Running plot generation sequentially.")
        for i, t in enumerate(tasks):
            try:
                plot_metric_vs_nll(*t)
                # click.echo(f"Completed task {i+1}/{len(tasks)}")
            except Exception as e:
                click.echo(f"Error processing task {i+1}: {e}")
        
    # --- 4. Summary Table ---
    click.echo("\n--- Model Performance Summary (Averaged across intensities) ---")
    summary_metrics = ["nll"] + [m for m in available_metrics if m != "nll"]
    
    summary_table = analysis_df.groupby("model")[summary_metrics].mean().reset_index()
    
    # Save to CSV
    table_path = output_dir / "model_metrics_summary.csv"
    summary_table.to_csv(table_path, index=False)
    click.echo(f"Saved summary table to {table_path}")
    
    # Print formatted table
    try:
        click.echo(summary_table.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        click.echo(summary_table.to_string(index=False, float_format="%.4f"))
        
    click.echo("Done.")

if __name__ == "__main__":
    main()
