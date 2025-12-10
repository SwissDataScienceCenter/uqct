import click
import pandas as pd
from pathlib import Path
from typing import Optional
from uqct.utils import get_results_dir
from uqct.loading import load_runs

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
@click.option(
    "--dataset", required=False, type=str, default=None, help="Dataset name."
)
@click.option("--intensity", required=False, type=float, default=None, help="Total intensity.")
@click.option("--sparse/--no-sparse", default=None, help="Sparse setting flag.")
def main(
    runs_dir: Path,
    output_file: Path,
    dataset: Optional[str],
    intensity: Optional[float],
    sparse: Optional[bool],
):
    """Consolidate run results into a single parquet file."""

    if runs_dir is None:
        runs_dir = get_results_dir() / "runs"

    click.echo("Configuration:")
    click.echo(f"  Runs Dir:    {runs_dir}")
    click.echo(f"  Output File: {output_file}")
    click.echo(f"  Dataset:     {dataset if dataset else 'ALL'}")
    click.echo(f"  Intensity:   {intensity if intensity is not None else 'ALL'}")
    click.echo(f"  Sparse:      {sparse if sparse is not None else 'ALL'}")

    aggregated_runs = load_runs(runs_dir, dataset, intensity, sparse)

    if not aggregated_runs:
        click.echo("No matching runs found.")
        return

    # Consolidate all models into one dataframe
    all_runs = []
    
    # We want to ensure consistent image set (intersection) like in plot_runs?
    # Or just dump everything we have?
    # The user said "deduplicates and saves the resulting dataset".
    # Usually "dataset" implies the clean, usable set.
    # Let's perform the same intersection logic as plot_runs to be safe, 
    # ensuring the output file is ready for analysis without further cleaning.
    
    # Logic: Intersect models WITHIN the same (dataset, intensity, sparse) application
    # to ensure fair comparison, then concat everything.
    
    # 1. Group keys by (dataset, intensity, sparse)
    groups = {}
    for key, df in aggregated_runs.items():
        ds, mod, inten, sp = key
        group_key = (ds, inten, sp)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(df)
        
    all_runs = []
    
    for group_key, dfs in groups.items():
        if not dfs: continue
        
        # Intersect within group
        min_len = min(len(df) for df in dfs)
        if min_len == 0: continue
        
        for df in dfs:
             df_cropped = df.iloc[:min_len].copy()
             all_runs.append(df_cropped)

    click.echo(f"Consolidating {len(all_runs)} model-runs across {len(groups)} configurations.")


    # Filter out empty dataframes to avoid FutureWarning
    all_runs = [df for df in all_runs if not df.empty]
    
    if not all_runs:
         click.echo("No data to save.")
         return

    # Sort for tidiness
    final_df = pd.concat(all_runs, ignore_index=True)
    
    # Sort for tidiness
    # Assuming we want to group by image, then model, or model then image?
    # Usually model then image is good for reading chunks.
    # But if we want to compare images: image then model.
    # Let's do Model, Image Index (implicit)
    # But wait, df has implicit index 0..N per model.
    
    # Let's save it
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_file)
    click.echo(f"Saved {len(final_df)} rows to {output_file}")

if __name__ == "__main__":
    main()
