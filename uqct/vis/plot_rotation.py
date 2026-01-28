import math
import click
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tomllib
import einops
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict
import scipy.ndimage as ndimage
import torch.nn.functional as F
from functools import lru_cache

import uqct
from uqct.eval.run import setup_experiment
from uqct.ct import nll_mixture_angle_schedule
from uqct.datasets.utils import get_dataset
from uqct.vis.style import ICML_COLUMN_HEIGHT, ICML_COLUMN_WIDTH, MODEL_ORDER, get_style

# Patch get_dataset to speed up repeated calls in setup_experiment
uqct.eval.run.get_dataset = lru_cache(maxsize=None)(get_dataset)

# # Plotting style
# plt.style.use(
#     "seaborn-v0_8-whitegrid"
#     if "seaborn-v0_8-whitegrid" in plt.style.available
#     else "ggplot"
# )
# plt.rcParams["figure.figsize"] = (10, 6)
# plt.rcParams["font.size"] = 12


DELTA = 0.05
LOG_INV_DELTA = math.log(1 / DELTA)


def compute_rotated_nll(
    gt: torch.Tensor, angles_deg: List[float], experiment, schedule
) -> Dict[float, List[float]]:
    """
    Computes cumulative NLL for rotated versions of GT.
    Returns: Dict[angle, nll_trajectory]

    Optimized: Stacks all rotated images across batch dim and computes NLL in chunks.
    """
    device = gt.device
    results = {}

    gt_np = gt.detach().cpu().numpy()
    T = len(schedule) if schedule is not None else 1
    n_images = gt.shape[0]
    n_angles_rot = len(angles_deg)

    # 1. Generate all rotated images
    # Output shape: (N * A, H, W)

    rotated_images_list = []

    # Pre-calculate rotations (fast on CPU)
    for angle in angles_deg:
        try:
            rot_np = ndimage.rotate(
                gt_np,
                angle,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        except TypeError:
            rot_list = []
            for i in range(n_images):
                rot_list.append(
                    ndimage.rotate(
                        gt_np[i],
                        angle,
                        reshape=False,
                        order=1,
                        mode="constant",
                        cval=0.0,
                        prefilter=False,
                    )
                )
            rot_np = np.stack(rot_list, axis=0)

        # rot_np is (N, H, W)
        rotated_images_list.append(torch.from_numpy(rot_np))

    cat_images = torch.cat(rotated_images_list, dim=0).to(device, dtype=torch.float32)

    # Create indices mapping each item in cat_images back to original image index (0..N-1)
    img_indices = torch.arange(n_images, device=device).repeat(n_angles_rot)

    # Downsample all at once
    if cat_images.shape[-1] > 128:
        cat_images = cat_images.unsqueeze(1)
        cat_images = F.interpolate(cat_images, size=(128, 128), mode="area")
        cat_images = cat_images.squeeze(1)

    # 2. Compute NLL in Batches
    total_items = cat_images.shape[0]

    # Batch Size
    batch_size = 64

    all_nlls = []

    for i in range(0, total_items, batch_size):
        batch = cat_images[i : i + batch_size]
        batch_idx = img_indices[i : i + batch_size]

        # Gather matching experiment data for this batch
        batch_counts = experiment.counts[batch_idx]
        batch_intensities = experiment.intensities[batch_idx]

        # Expand: (B, H, W) -> (B, T, 1, H, W)
        batch_expanded = einops.repeat(batch, "b w h -> b t 1 w h", t=T)
        batch_expanded = batch_expanded.contiguous()

        with torch.no_grad():
            nlls = nll_mixture_angle_schedule(
                batch_expanded,
                batch_counts,
                batch_intensities,
                experiment.angles,
                schedule,
                reduce=False,
            )

        all_nlls.append(nlls.cpu().numpy())  # (B, T_subset)

    # Concatenate all results: (A*N, T)
    full_nlls = np.concatenate(all_nlls, axis=0)

    # 3. Unpack Results into Dict
    # Layout: Angle 0 (N images), Angle 1 (N images)...

    for i, angle in enumerate(angles_deg):
        start = i * n_images
        end = start + n_images
        # Slice for this angle
        nlls_angle = full_nlls[start:end]  # (N, T)

        results[angle] = nlls_angle.tolist()

    return results


DATASETS = ["composite", "lamino", "lung"]


def generate_data(
    consolidated_file: Path,
    output_file: Path,
    device: str,
    limit: Optional[int],
    schedule_length: int = 32,
):
    """Generates rotation exclusion data and saves to Parquet."""
    click.echo(f"Loading data from {consolidated_file}...")
    df = pd.read_parquet(consolidated_file)

    # Filter for valid entries (need nll_pred_mix)
    df = df[
        df["nll_pred_mix"].notna()
        & df["total_intensity"].isin([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    ]

    # Define Angles
    # Log-linearly spaced between 0.1 and 90, plus 0.0 for GT check
    log_angles = np.logspace(np.log(0.1), np.log(90), 10, base=np.e)
    angles_deg = np.concatenate(([0.0], log_angles))
    click.echo(f"Angles: {angles_deg}")

    results_records = []

    # Group by Experiment Definition
    group_cols = ["dataset", "total_intensity", "sparse", "seed"]
    if "image_start_index" in df.columns:
        group_cols.append("image_start_index")
    if "image_end_index" in df.columns:
        group_cols.append("image_end_index")

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        click.echo(
            f"Missing columns for grouping: {missing}. Cannot proceed with analysis."
        )
        return

    groups = list(df.groupby(group_cols))
    count_groups = 0
    for group_keys, group_df in tqdm(groups, total=len(groups)):
        if limit is not None and count_groups >= limit:
            break
        count_groups += 1

        keys = dict(zip(group_cols, group_keys))

        dataset = keys["dataset"]
        total_intensity = keys["total_intensity"]
        sparse = keys["sparse"]
        seed = keys["seed"]

        start_idx = keys.get("image_start_index")
        if start_idx is not None:
            start_idx = int(start_idx)
        end_idx = keys.get("image_end_index")
        if end_idx is not None:
            end_idx = int(end_idx)

        if start_idx is None or end_idx is None:
            continue

        gt, experiment, schedule = setup_experiment(
            dataset=dataset,
            image_range=(start_idx, end_idx),
            total_intensity=total_intensity,
            sparse=sparse,
            seed=seed,
            schedule_length=schedule_length,
        )

        # Compute Rotated NLLs
        device_obj = torch.device(device)
        gt = gt.to(device_obj)
        experiment.to(device_obj)
        if schedule is not None:
            schedule = schedule.to(device_obj)

        # Map of angle -> NLL trajectory
        rot_nlls_map = compute_rotated_nll(gt, angles_deg, experiment, schedule)

        # Now Check Exclusion for each Model
        for model, model_df in group_df.groupby("model"):
            n_images = min(len(model_df), len(gt))

            for i in range(n_images):
                img_row = model_df.iloc[i]

                nll_pred_list = img_row["nll_pred_mix"]
                if not isinstance(nll_pred_list, (list, np.ndarray)):
                    continue

                pred_traj = np.array(nll_pred_list)
                pred_cum = np.cumsum(pred_traj)
                thresh_cum = pred_cum + LOG_INV_DELTA

                L_pred = len(pred_cum)

                for angle in angles_deg:
                    rot_traj = np.array(rot_nlls_map[angle][i])

                    # Align lengths
                    L_rot = len(rot_traj)
                    if L_rot > L_pred:
                        rot_traj = rot_traj[-L_pred:]
                    elif L_rot < L_pred:
                        pass

                    rot_cum = np.cumsum(rot_traj)

                    if len(rot_cum) != len(thresh_cum):
                        # Warning handled?
                        continue

                    is_excluded = np.any(rot_cum > thresh_cum)

                    results_records.append(
                        {
                            "dataset": dataset,
                            "intensity": total_intensity,
                            "sparse": sparse,
                            "model": model,
                            "angle": angle,
                            "seed": seed,
                            "image_idx": i + (start_idx if start_idx else 0),
                            "excluded": is_excluded,
                        }
                    )

        del gt, experiment, schedule
        torch.cuda.empty_cache()

    res_df = pd.DataFrame(results_records)
    if res_df.empty:
        click.echo("No results generated.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_parquet(output_file, index=False)
    click.echo(f"Saved results to {output_file}")


def plot_data(input_file: Path):
    """Plots exclusion rate from Parquet data."""
    if not input_file.exists():
        click.echo(f"Input file {input_file} does not exist.")
        return

    df = pd.read_parquet(input_file)

    # Aggregate Exclusion Rate
    # Aggregate Exclusion Rate
    agg = (
        df.groupby(["dataset", "intensity", "sparse", "model", "angle"])["excluded"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg.rename(columns={"mean": "exclusion_rate"}, inplace=True)

    # Save Summary
    output_dir = Path("./plots/rotation")
    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(output_dir / "rotation_exclusion_summary.parquet", index=False)

    # Group by Intensity and Sparse for Figures
    grouped_settings = agg.groupby(["intensity", "sparse"])
    agg.to_csv(output_dir / "rotation_exclusion_summary.csv", index=False)
    agg[agg["intensity"] == 1e9].to_csv(
        output_dir / "rotation_exclusion_summary_1e9.csv", index=False
    )

    for (intensity, sparse), group_df in grouped_settings:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(ICML_COLUMN_WIDTH, ICML_COLUMN_HEIGHT),  # <--- WIDER and SHORTER
            sharey=True,
            sharex=True,
        )
        # fig.supylabel(r"Exclusion Rate (\%)", x=0.03)

        for row_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx]

            d_df = group_df[group_df["dataset"] == dataset]

            if d_df.empty:
                ax.set_visible(False)
                continue

            # Models
            available_models = set(d_df["model"].unique())
            models = [m for m in MODEL_ORDER if m in available_models]
            for m in sorted(available_models):
                if m not in models:
                    models.append(m)

            for m in models:
                sub = d_df[d_df["model"] == m].sort_values("angle")
                style = get_style(m)

                # Fill NaNs in SEM if any (single sample case)
                sem = sub["sem"].fillna(0)

                ax.plot(
                    sub["angle"],
                    100 * sub["exclusion_rate"],
                    label=style["label"],
                    color=style["color"],
                    marker="x",
                    alpha=0.8,
                )
                ax.fill_between(
                    sub["angle"],
                    (100 * (sub["exclusion_rate"] - sem)).clip(lower=0),
                    (100 * (sub["exclusion_rate"] + sem)).clip(upper=100),
                    alpha=0.2,
                    color=style["color"],
                )

            ax.set_xscale("log")
            ax.set_title(f"{dataset.title()} Dataset")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            if row_idx == 2:
                ax.set_xlabel("Rotation Angle (Deg)")
                # ax.legend(fontsize=8, loc="best")
            ax.set_ylabel(r"Exclusion Rate (\%)")
        handles, labels = axes[-1].get_legend_handles_labels()
        # fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.tight_layout(rect=[0, 0.08, 1, 1])

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            frameon=False,
        )

        # Directory: plots/rotation/{intensity}_{sparse}/
        inten_str = f"{intensity:.0e}".replace("+0", "").replace("+", "")
        mode_str = "sparse" if sparse else "dense"

        out_path = output_dir / f"{mode_str}_rotation_exclusion_{inten_str}.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(out_path)
        plt.close(fig)
        click.echo(f"Saved plot to {out_path}")

    click.echo("Plotting done.")


@click.command()
@click.option(
    "--consolidated-file",
    type=click.Path(path_type=Path),
    help="Path to consolidated parquet file (for input data).",
)
@click.option(
    "--data-file",
    type=click.Path(path_type=Path),
    default=Path("./rotation_results.parquet"),
    help="Path to intermediate rotation results parquet file.",
)
@click.option(
    "--generate/--no-generate",
    default=True,
    help="Whether to run data generation.",
)
@click.option(
    "--plot/--no-plot",
    default=True,
    help="Whether to run plotting.",
)
@click.option(
    "--limit", default=None, type=int, help="Limit number of groups processing (debug)."
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use.",
)
def main(
    consolidated_file: Optional[Path],
    data_file: Path,
    generate: bool,
    plot: bool,
    limit: Optional[int],
    device: str,
):
    """Analyze exclusion rate of rotated GT images."""

    # Load settings for schedule_length if needed
    schedule_length = 32
    try:
        with open("uqct/settings.toml", "rb") as f:
            settings_data = tomllib.load(f)
        sparse_settings = settings_data.get("eval-sparse", {})
        schedule_length = sparse_settings.get("schedule_length", 32)
    except Exception:
        pass

    if generate:
        if not consolidated_file:
            click.echo("Error: --consolidated-file required for generation.")
            return
        if not consolidated_file.exists():
            click.echo(f"Error: {consolidated_file} does not exist.")
            return

        generate_data(consolidated_file, data_file, device, limit, schedule_length)

    if plot:
        plot_data(data_file)


if __name__ == "__main__":
    main()
