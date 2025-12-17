import os

# Set allocator config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import time
from functools import lru_cache
import gc

import uqct
from uqct.eval.run import setup_experiment
from uqct.ct import nll_mixture_angle_schedule, Experiment, sample_observations, nll
from uqct.utils import get_results_dir
from uqct.datasets.utils import get_dataset
from uqct.training.unet import N_ANGLES

# Patch get_dataset to speed up repeated calls in setup_experiment
uqct.eval.run.get_dataset = lru_cache(maxsize=None)(get_dataset)

# Plotting style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "ggplot"
)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# Model Style
from uqct.vis.style import MODEL_ORDER, MODEL_NAMES, MODEL_COLORS, get_style


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

    # Stack: (A, N, H, W) -> Permute to (N, A, H, W) -> Reshape (N*A, H, W)
    # We want to keep N grouped together? Or A?
    # Actually results are keyed by Angle.
    # But for calculation, order doesn't matter as long as we track it.
    # Let's stack as (A * N, H, W) -> [Angle0_Img0, Angle0_Img1... Angle1_Img0...]
    # This matches the loop order above.

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
    batch_size = 256

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
@click.option(
    "--limit", default=None, type=int, help="Limit number of groups processing (debug)."
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use.",
)
def main(consolidated_file: Path, output_dir: Path, device: str, limit: Optional[int]):
    """Analyze exclusion rate of rotated GT images."""

    # Load settings
    try:
        with open("uqct/settings.toml", "rb") as f:
            settings_data = tomllib.load(f)
        sparse_settings = settings_data.get("eval-sparse", {})
        default_schedule_len = sparse_settings.get("schedule_length", 32)  # Default 32
        click.echo(f"Loaded schedule_length={default_schedule_len} from settings.toml")
    except Exception as e:
        default_schedule_len = 32
        click.echo(
            f"Could not load settings.toml ({e}), defaulting to {default_schedule_len}"
        )

    click.echo(f"Loading data from {consolidated_file}...")
    df = pd.read_parquet(consolidated_file)

    # Filter for valid entries (need nll_pred)
    df = df[df["nll_pred"].notna()]

    # Define Angles
    # Log-linearly spaced between 0.1 and 90, plus 0.0 for GT check
    log_angles = np.logspace(np.log(0.1), np.log(90), 10, base=np.e)
    angles_deg = np.concatenate(([0.0], log_angles))
    click.echo(f"Angles: {angles_deg}")

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)

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

    click.echo(f"Processing {len(groups)} experiment groups...")

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

        start_idx = keys.get("image_start_index", 0)
        end_idx = keys.get("image_end_index", 5)

        schedule_length = default_schedule_len

        gt = None
        experiment = None
        schedule = None

        t0 = time.time()
        gt, experiment, schedule = setup_experiment(
            dataset=dataset,
            image_range=(start_idx, end_idx),
            total_intensity=total_intensity,
            sparse=sparse,
            seed=seed,
            schedule_length=schedule_length,
        )
        t1 = time.time()

        # Compute Rotated NLLs
        device_obj = torch.device(device)
        gt = gt.to(device_obj)
        experiment.to(device_obj)
        if schedule is not None:
            schedule = schedule.to(device_obj)

        # Map of angle -> NLL trajectory
        rot_nlls_map = compute_rotated_nll(gt, angles_deg, experiment, schedule)
        t2 = time.time()

        # Now Check Exclusion for each Model
        # Iterate over unique models in this group
        for model, model_df in group_df.groupby("model"):

            n_images = min(len(model_df), len(gt))

            # We iterate images by index to match GT
            for i in range(n_images):
                img_row = model_df.iloc[i]

                nll_pred_list = img_row["nll_pred"]
                if not isinstance(nll_pred_list, (list, np.ndarray)):
                    continue

                pred_traj = np.array(nll_pred_list)
                pred_cum = np.cumsum(pred_traj)
                thresh_cum = pred_cum + LOG_INV_DELTA

                L_pred = len(pred_cum)

                for angle in angles_deg:
                    # Get Rotated NLL for Image i
                    rot_traj = np.array(rot_nlls_map[angle][i])

                    # Align lengths (Slice from END to match prediction horizon)
                    L_rot = len(rot_traj)
                    if L_rot > L_pred:
                        rot_traj = rot_traj[-L_pred:]
                    elif L_rot < L_pred:
                        pass

                    rot_cum = np.cumsum(rot_traj)

                    # Direct comparison
                    if len(rot_cum) != len(thresh_cum):
                        click.echo(
                            f"Warning: Length Mismatch Model {model}: Rot {len(rot_cum)} vs Thresh {len(thresh_cum)}"
                        )
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
                            "image_idx": i + start_idx,
                            "excluded": is_excluded,
                        }
                    )

        # Memory Cleanup
        del gt, experiment, schedule
        torch.cuda.empty_cache()

    # Analysis & Plotting
    res_df = pd.DataFrame(results_records)
    if res_df.empty:
        click.echo("No results generated.")
        return

    res_df.to_csv(output_dir / "rotation_exclusion_raw.csv", index=False)

    # Aggregate Exclusion Rate (Include sparse)
    agg = (
        res_df.groupby(["dataset", "intensity", "sparse", "model", "angle"])["excluded"]
        .mean()
        .reset_index()
    )
    agg.rename(columns={"excluded": "exclusion_rate"}, inplace=True)
    agg.to_csv(output_dir / "rotation_exclusion_summary.csv", index=False)

    click.echo(f"Models found in summary: {agg['model'].unique()}")

    # Plotting
    plot_groups = agg.groupby(["dataset", "intensity", "sparse"])

    for (d, inten, sp), p_df in plot_groups:
        plt.figure(figsize=(8, 6))

        available_models = set(p_df["model"].unique())
        models = [m for m in MODEL_ORDER if m in available_models]  # Enforce order
        # Add any extra models
        for m in sorted(available_models):
            if m not in models:
                models.append(m)

        for m in models:
            sub = p_df[p_df["model"] == m].sort_values("angle")
            style = get_style(m)
            plt.plot(
                sub["angle"],
                sub["exclusion_rate"],
                label=style["label"],
                color=style["color"],
                marker="o",
                alpha=0.8,
            )

        plt.xscale("linear")
        plt.xlabel("Rotation Angle (Degrees)")
        plt.ylabel("Exclusion Rate")

        # Pretty Intensity
        inten_str = f"{inten:.0e}".replace("+0", "").replace("+", "")

        plt.legend()
        plt.ylim(-0.05, 1.05)

        # Directory structure: root/{dataset}/{intensity}_{sparse}/exclusion_rate.png
        # Hierarchical per user request
        target_dir = output_dir / d / f"{inten_str}_{'sparse' if sp else 'dense'}"
        target_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(target_dir / "exclusion_rate.pdf")
        plt.close()

    # --- GLOBAL OVERVIEW PLOTS ---

    # Average across datasets per intensity (keeping sparse/dense aggregated or separate?)
    # Current behavior was aggregating everything. Let's keep it but separate purely by intensity for now.

    global_agg = (
        agg.groupby(["intensity", "model", "angle"])["exclusion_rate"]
        .mean()
        .reset_index()
    )

    intensities = sorted(global_agg["intensity"].unique())
    if len(intensities) > 0:

        # Global Directory
        global_dir = output_dir / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        # 6 Plots (One per Intensity) + Stacked
        fig_stack, axes_stack = plt.subplots(
            len(intensities), 1, figsize=(10, 4 * len(intensities)), sharex=True
        )
        if len(intensities) == 1:
            # Ensure iterable if 1
            axes_stack = np.array([axes_stack])

        for idx, inten in enumerate(intensities):
            p_df = global_agg[global_agg["intensity"] == inten]

            # Individual Plot
            plt.figure(figsize=(8, 6))
            available_models = set(p_df["model"].unique())
            models = [m for m in MODEL_ORDER if m in available_models]
            # Add any extra models
            for m in sorted(available_models):
                if m not in models:
                    models.append(m)

            for m in models:
                sub = p_df[p_df["model"] == m].sort_values("angle")
                style = get_style(m)

                # Individual Plot
                plt.plot(
                    sub["angle"],
                    sub["exclusion_rate"],
                    label=style["label"],
                    color=style["color"],
                    marker="o",
                    alpha=0.8,
                )

                # Add to Stacked
                axes_stack[idx].plot(
                    sub["angle"],
                    sub["exclusion_rate"],
                    label=style["label"],
                    color=style["color"],
                    marker="o",
                    alpha=0.8,
                )

            plt.xscale("linear")
            plt.xlabel("Rotation Angle (Degrees)")
            plt.ylabel("Exclusion Rate")

            # Pretty Intensity
            inten_str = f"{inten:.0e}".replace("+0", "").replace("+", "")

            plt.legend()
            plt.ylim(-0.05, 1.05)

            # Global Directory: global/rotation/{intensity}/exclusion_rate.pdf
            global_target_dir = output_dir / "global-rotation" / inten_str
            global_target_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(global_target_dir / "exclusion_rate.pdf")
            plt.close()

            # Stacked styling

            axes_stack[idx].set_ylim(-0.05, 1.05)
            axes_stack[idx].set_ylabel("Exclusion Rate")
            axes_stack[idx].grid(True)
            axes_stack[idx].legend(
                loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small"
            )

            # Save Stacked (do it outside loop or accumulate?)
            # The figure is created outside.

        # Save Stacked Figure
        global_root = output_dir / "global-rotation"
        global_root.mkdir(parents=True, exist_ok=True)
        fig_stack.tight_layout()
        fig_stack.savefig(global_root / "exclusion_stacked.pdf")
        plt.close(fig_stack)

        # Finalize Stacked
        axes_stack[-1].set_xlabel("Rotation Angle (Degrees)")
        axes_stack[-1].set_xscale("log")

        plt.tight_layout()
        fig_stack.savefig(global_dir / "exclusion_global_stacked.pdf")
        plt.close(fig_stack)

    click.echo("Done.")


if __name__ == "__main__":
    main()
