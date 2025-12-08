import os
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from uqct.utils import get_results_dir

# Set plotting style
# sns.set_theme(style="whitegrid")
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "ggplot"
)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

DELTA = 0.05
LOG_INV_DELTA = math.log(1 / DELTA)


def load_runs(
    runs_dir: Path, dataset: str, intensity: float, sparse: bool
) -> Dict[str, pd.DataFrame]:
    """
    Scans the runs directory and returns the most recent run for each model
    matching the given criteria, aggregating across image range chunks.
    """
    if not runs_dir.exists():
        click.echo(f"Runs directory not found: {runs_dir}")
        return {}

    click.echo(f"Scanning {runs_dir} for parquet files...")

    # Store all candidate files: candidates[(model, start, end)] = [file_info, ...]
    candidates = {}

    # Iterate over all parquet files
    for file_path in runs_dir.glob("*.parquet"):
        try:
            # Filename format: model:dataset:intensity:sparse:range:timestamp.parquet
            # Example: fbp:lung:10000.0:True:0-10:2025-12-08...
            parts = file_path.stem.split(":")
            start, end = -1, -1  # Initialize to default
            if len(parts) >= 6:
                # Basic filename filtering
                if parts[1] != dataset:
                    continue
                if not math.isclose(float(parts[2]), intensity):
                    continue
                if (parts[3] == "True") != sparse:
                    continue

                # Parse range to key
                range_part = parts[4]
                try:
                    start, end = map(int, range_part.split("-"))
                except ValueError:
                    # Fallback if filename doesn't match expected pattern perfectly
                    start, end = -1, -1
            else:
                # If filename format is different, we might have to peek inside
                # But let's assume filtering by filename is efficient first
                # If we rely on metadata for everything, it's safer but slower.
                # Let's peek metadata if filename check passes or ambiguous.
                pass

            # Load metadata (lightweight if possible, but parquet reads whole file usually?
            # actually pd.read_parquet is okayish for small files)
            # We need to ensure it's the right run configuration
            df = pd.read_parquet(file_path)
            if df.empty:
                continue

            row = df.iloc[0]
            if row["dataset"] != dataset:
                continue
            if not math.isclose(row["total_intensity"], intensity):
                continue
            if row["sparse"] != sparse:
                continue

            # Identify chunk logic
            model = row["model"]
            run_id = row["run_id"]
            timestamp = pd.to_datetime(row["timestamp"])

            # Use metadata for range if available, else filename
            # The run.parquet dump includes ct_settings keys
            if "image_start_index" in row and "image_end_index" in row:
                start = int(row["image_start_index"])
                end = int(row["image_end_index"])
            else:
                # If not in metadata, rely on filename parse from above
                # If that failed, we have a problem distinguishing chunks
                if start == -1:
                    click.echo(
                        f"Warning: Could not determine image range for {file_path}"
                    )
                    continue

            key = (model, start, end)
            if key not in candidates:
                candidates[key] = []

            candidates[key].append(
                {"file": file_path, "timestamp": timestamp, "data": df}
            )

        except Exception as e:
            # click.echo(f"Skipping {file_path}: {e}")
            continue

    # For each chunk (model, start, end), pick the latest
    latest_chunks = {}
    for key, file_list in candidates.items():
        # Sort by timestamp desc
        file_list.sort(key=lambda x: x["timestamp"], reverse=True)
        selected = file_list[0]
        model = key[0]

        if model not in latest_chunks:
            latest_chunks[model] = []
        latest_chunks[model].append(selected["data"])

        # click.echo(f"Selected chunk {key[1]}-{key[2]} for {model} ({selected['timestamp']})")

    # Aggregate chunks per model
    aggregated_runs = {}
    for model, chunks in latest_chunks.items():
        if chunks:
            full_df = pd.concat(chunks, ignore_index=True)
            # Sort by image index if available? Parquet rows usually don't have explicit image index column
            # unless we infer it or it was saved.
            # Run.py logic: metric2lists has N entries.
            # But the DF doesn't explicitly store "image_index" column?
            # Run.dump_parquet: for k, v in asdict(ct_settings).items(): df[k] = v
            # It saves image_start_index.
            # But inside the DF, rows correspond to images relative to start.
            # We might want to sort to be deterministic.
            # However, plotting uses row iteration.
            aggregated_runs[model] = full_df
            click.echo(
                f"Aggregated {len(chunks)} chunks for {model}. Total images: {len(full_df)}"
            )

    return aggregated_runs


def process_and_plot(latest_runs: Dict[str, pd.DataFrame], output_dir: Path):
    ranking_data = []

    for model, df in latest_runs.items():
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate metrics for ranking
        # Metrics are lists of lists (time steps)
        # We take the FINAL value for PSNR/SSIM/etc? Or mean over time?
        # Typically PSNR is evaluated at the end or over steps.
        # User asked for "ranking in terms of PSNR and NLL".
        # Assume final step is what matters for reconstruction quality unless specified.
        # But for NLL, they want trajectories.
        # Let's use the final step for scalar ranking.

        total_psnr = 0
        total_nll = 0
        crossover_count = 0
        n_images = len(df)

        for idx, row in tqdm(df.iterrows(), total=n_images, desc=f"Processing {model}"):
            # Parse metrics. Parquet might load lists as numpy arrays or lists.
            # nll_pred and nll_gt are list of lists.
            # In the DF, 'nll_pred' column contains the list for that image.

            nll_pred = row["nll_pred"]
            nll_gt = row["nll_gt"]
            psnr_traj = row["psnr"]

            if isinstance(nll_pred, np.ndarray):
                nll_pred = nll_pred.tolist()
            if isinstance(nll_gt, np.ndarray):
                nll_gt = nll_gt.tolist()
            if isinstance(psnr_traj, np.ndarray):
                psnr_traj = psnr_traj.tolist()

            # Ensure they are lists
            if not isinstance(nll_pred, list) or not isinstance(nll_gt, list):
                continue

            # Compute cumulative sum for NLL
            nll_pred_cum = np.cumsum(nll_pred)
            nll_gt_cum = np.cumsum(nll_gt)

            steps = range(len(nll_pred))

            # Crossover check
            # Exists some t for which NLL_t(gt) > NLL_t(pred) + log(1/delta)
            # Using cumulative values as per request
            has_crossover = any(
                gt > pred + LOG_INV_DELTA for gt, pred in zip(nll_gt_cum, nll_pred_cum)
            )
            if has_crossover:
                crossover_count += 1

            # Accumulate final metrics
            if psnr_traj:
                total_psnr += psnr_traj[-1]
            if len(nll_pred_cum) > 0:
                total_nll += nll_pred_cum[-1]

            # Plotting
            plt.figure()
            plt.plot(
                steps,
                nll_gt_cum,
                label="NLL (GT, Cumulative)",
                linestyle="--",
                color="black",
            )
            plt.plot(
                steps,
                nll_pred_cum,
                label=f"NLL (Pred: {model}, Cumulative)",
                color="tab:blue",
            )

            # Add crossover threshold line relative to Pred
            thresh_line = [p + LOG_INV_DELTA for p in nll_pred_cum]
            plt.plot(
                steps,
                thresh_line,
                label=f"NLL(Pred) + log(1/{DELTA})",
                linestyle=":",
                color="tab:red",
            )

            plt.title(
                f"Cumulative NLL Trajectory - {model} - Img {idx}\nCrossover: {has_crossover}"
            )
            plt.xlabel("Step")
            plt.ylabel("Cumulative NLL")
            plt.yscale("log")
            plt.legend()

            # Save plot
            run_id = row.get("run_id", "unknown")
            plot_path = model_dir / f"run_{run_id}_img_{idx}_nll.png"
            plt.savefig(plot_path)
            plt.close()

        # Avg metrics
        avg_psnr = total_psnr / n_images if n_images > 0 else 0
        avg_nll = total_nll / n_images if n_images > 0 else 0
        crossover_rate = crossover_count / n_images if n_images > 0 else 0

        ranking_data.append(
            {
                "model": model,
                "avg_psnr_final": avg_psnr,
                "avg_nll_cumulative_final": avg_nll,
                "crossover_rate": crossover_rate,
                "n_images": n_images,
            }
        )

    return pd.DataFrame(ranking_data)


@click.command()
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing run parquet files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots"),
    help="Directory to save plots.",
)
@click.option(
    "--dataset", required=True, type=str, help="Dataset name (lung, composite, lamino)."
)
@click.option("--intensity", required=True, type=float, help="Total intensity.")
@click.option("--sparse/--no-sparse", default=True, help="Sparse setting flag.")
def main(
    runs_dir: Optional[Path],
    output_dir: Path,
    dataset: str,
    intensity: float,
    sparse: bool,
):
    """Visualize evaluation runs."""

    if runs_dir is None:
        runs_dir = get_results_dir() / "runs"

    click.echo(f"Configuration:")
    click.echo(f"  Runs Dir:  {runs_dir}")
    click.echo(f"  Output Dir:{output_dir}")
    click.echo(f"  Dataset:   {dataset}")
    click.echo(f"  Intensity: {intensity}")
    click.echo(f"  Sparse:    {sparse}")

    latest_runs = load_runs(runs_dir, dataset, intensity, sparse)

    if not latest_runs:
        click.echo("No matching runs found.")
        return

    # Prepare output subdirectory
    setting_dir = (
        output_dir / f"{dataset}_{intensity}_{'sparse' if sparse else 'dense'}"
    )
    setting_dir.mkdir(parents=True, exist_ok=True)

    ranking_df = process_and_plot(latest_runs, setting_dir)

    if not ranking_df.empty:
        # Sort by PSNR descending
        ranking_df = ranking_df.sort_values(by="avg_psnr_final", ascending=False)

        print("\nRanking Summary:")
        print(ranking_df.to_string(index=False))

        csv_path = setting_dir / "ranking.csv"
        ranking_df.to_csv(csv_path, index=False)
        click.echo(f"\nRanking saved to {csv_path}")


if __name__ == "__main__":
    main()
