import math
from pathlib import Path
from typing import Any, Callable

import click
import einops
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch

from uqct.ct import Experiment, nll
from uqct.eval.run import setup_experiment
from uqct.metrics import get_metrics
from uqct.models.diffusion import Diffusion
from uqct.logging import get_logger
from uqct.uq import (
    error_correlation,
    gaussian_ci,
    coverage,
    error_r2,
    percentile_ci,
    basic_ci,
    studentized_ci,
    simultaneous_ci,
    sparsification_error,
)
from uqct.utils import get_results_dir

logger = get_logger(__name__)


def load_experiment_from_parquet(
    parquet_path_str: str,
) -> tuple[pd.DataFrame, Any, float, Experiment, torch.Tensor, pd.Series, torch.Tensor]:
    """
    Loads experiment setup and nll predictions from a parquet file.

    Args:
        parquet_path_str: Path to the parquet file.
        schedule_length: Length of the schedule.

    Returns:
        tuple containing:
            - df: pd.DataFrame
            - dataset: str
            - total_intensity: float
            - experiment: Experiment
            - schedule: torch.Tensor | None
            - row: pd.Series
            - gt: torch.Tensor (Ground Truth image)
    """
    parquet_path = Path(parquet_path_str)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file {parquet_path} not found")

    df = pd.read_parquet(parquet_path)

    # Extract settings from the first row (assuming homogeneous run)
    # The parquet contains results from a run, so it has the config.
    # We need to parse the filename or look at columns to get setup args.
    # The parquet columns usually contain 'dataset', 'total_intensity', etc.

    row = df.iloc[0]
    dataset = row["dataset"]
    # Handle float/int types safely
    total_intensity = float(row["total_intensity"])
    sparse = bool(row["sparse"])
    seed = int(row["seed"])

    # Image range depends on the file content.
    # Usually the file covers a specific range.
    # We'll use the range present in the dataframe or derived from filename if needed.
    # But setup_experiment needs a range. The DF typically has one row per image.
    image_start_index = (
        int(row["image_start_index"]) if "image_start_index" in row else 0
    )
    # If the file is a chunk, we might need to deduce the range.
    # Let's assume the parquet corresponds to the requested experiment range we want to reproduce.
    # We will use the number of rows to determine end index if not explicit,
    # but run.py saves 'image_start_index' and 'image_end_index'.

    image_end_index = (
        int(row["image_end_index"])
        if "image_end_index" in row
        else image_start_index + len(df)
    )

    # For now, we'll setup the experiment for the chunks available in the DF.
    # If the parquet file is huge, this might be slow, but usually it's a batch.

    # Re-create experiment assuming sparse experiment
    schedule = df["angle_schedule"][0]
    gt, experiment, schedule = setup_experiment(
        dataset,
        (image_start_index, image_end_index),
        total_intensity,
        schedule_start=schedule[0].item(),
        schedule_length=len(schedule),
        sparse=sparse,
        seed=seed,
    )
    return df, dataset, total_intensity, experiment, schedule, row, gt


def get_boundary_guidance_loss_fn(
    experiment: Experiment,
    schedule: torch.Tensor,
    conf_coefs: torch.Tensor,
    time_step: int,
    dist_loss_fac: float = 0.5,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates the guidance loss function.

    Args:
        experiment: The CT Experiment.
        schedule: The schedule tensor of shape (S,).
        conf_coefs: Confidence coefficients of shape (N, TotalAngles).
        time_step: The relative angle index to check (0 to TotalAngles-1).
        dist_loss_fac: Weight factor for distance loss.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Loss function accepting images of shape (R, N, S, H, W).
    """

    def loss_fn(images: torch.Tensor) -> torch.Tensor:
        """
        Computes the boundary sampling loss.

        Args:
            images: Tensor of shape (R, N, S, H, W).

        Returns:
            torch.Tensor: Scalar loss.
        """
        R, N, S, H, W = images.shape
        images = images[:, :, 0, :, :]
        images_flat = einops.rearrange(images, "r n h w -> (n r) h w").contiguous()
        start_angle_idx = schedule[0].item()
        end_angle_idx = start_angle_idx + time_step + 1
        assert end_angle_idx <= len(experiment.angles), "Time step out of bounds"
        counts_slice = experiment.counts[:, start_angle_idx:end_angle_idx, :]
        intensities_slice = experiment.intensities[:, start_angle_idx:end_angle_idx, :]
        angles_slice = experiment.angles[start_angle_idx:end_angle_idx]
        counts_rep = einops.repeat(counts_slice, "n ... -> (n r) ...", r=R)
        intensities_rep = einops.repeat(intensities_slice, "n ... -> (n r) ...", r=R)
        nlls_raw = nll(images_flat, counts_rep, intensities_rep, angles_slice)
        current_nll = einops.reduce(nlls_raw, "b ... -> b", "sum")
        threshold = einops.repeat(conf_coefs, "n -> (n r)", r=R)
        cond = current_nll < threshold  # (NR,)

        loss_nll_term = (~cond).float() * current_nll
        images_vec = images.reshape(R, N, -1)
        dists = torch.square(images_vec[:, None, :, :] - images_vec[None, :, :, :]).sum(
            dim=-1
        )
        mean_dists = dists.mean(dim=1)  # (R, N)

        dist_loss_vals = einops.rearrange(mean_dists, "r n -> (n r)")
        loss_dist_term = dist_loss_fac * cond.float() * (-dist_loss_vals)

        total_loss = loss_nll_term + loss_dist_term

        # Logging
        # with torch.no_grad():
        #     rel_dist = (current_nll - threshold) / (threshold.abs() + 1e-8)
        #     logger.info(
        #         f"Mean Rel Dist to Boundary: {rel_dist.mean().item():.4f} (Pos=Outside, Neg=Inside)"
        #     )
        #     logger.info(f"Distance between samples: {dists.mean().item():.4f}")
        #     logger.info(f"Dist loss term: {loss_dist_term.mean().item():.4f}")
        #     logger.info(f"NLL loss term: {loss_nll_term.mean().item():.4f}")

        return total_loss.sum()

    return loss_fn


def evaluate_and_log_results(
    gt_lr: torch.Tensor,
    sampled_images: torch.Tensor,
    std_dev: torch.Tensor,
    output_dir: Path,
    dataset: str,
    total_intensity: float,
    run_id: str,
    original_preds: torch.Tensor | None = None,
    output_prefix: str | None = None,
) -> None:
    """
    Computes metrics, logs them, and generates plots for the run.
    """
    logger.info("Evaluating results...")

    # sampled_images shape: (N, S, R, 1, H, W).
    final_samples = sampled_images[:, -1, :, :, :, :]  # (N, R, 1, H, W)
    final_std = std_dev[:, -1, :, :, :]  # (N, 1, H, W)

    # Compute Mean Prediction
    pred_mean = final_samples.mean(dim=1)  # (N, 1, H, W)

    # Ensure GT matches shape
    if gt_lr.ndim == 3:
        gt_lr = gt_lr.unsqueeze(1)

    logger.info(f"GT shape: {gt_lr.shape}, Pred Mean shape: {pred_mean.shape}")

    if gt_lr.shape[-1] != pred_mean.shape[-1]:
        logger.warning(
            f"Resolution mismatch ({gt_lr.shape[-1]} vs {pred_mean.shape[-1]}). Interpolating GT to match Prediction."
        )

    if gt_lr.shape[0] != pred_mean.shape[0]:
        logger.warning(
            f"GT shape {gt_lr.shape} matches pred shape {pred_mean.shape} on N? Mismatch."
        )

    # Compute Reconstruction Metrics
    metrics = get_metrics(pred_mean, gt_lr, data_range=1.0)

    # UQ Metrics
    # Compute Gaussian CI
    ci_lo_g, ci_hi_g = gaussian_ci(final_samples, delta=0.05, bdim=1)
    ci_lo_percentile, ci_hi_percentile = percentile_ci(
        final_samples, alpha=0.05, bdim=1
    )
    ci_lo_basic, ci_hi_basic = basic_ci(final_samples, alpha=0.05, bdim=1)
    ci_lo_studentized, ci_hi_studentized = studentized_ci(
        final_samples, delta=0.05, bdim=1
    )
    ci_lo_simultaneous, ci_hi_simultaneous = simultaneous_ci(
        final_samples, delta=0.05, bdim=1
    )
    ci_width_g = ci_hi_g - ci_lo_g
    ci_width_percentile = ci_hi_percentile - ci_lo_percentile
    ci_width_basic = ci_hi_basic - ci_lo_basic
    ci_width_studentized = ci_hi_studentized - ci_lo_studentized
    ci_width_simultaneous = ci_hi_simultaneous - ci_lo_simultaneous

    # Coverage
    cov_g = coverage(ci_lo_g, ci_hi_g, gt_lr)
    cov_percentile = coverage(ci_lo_percentile, ci_hi_percentile, gt_lr)
    cov_basic = coverage(ci_lo_basic, ci_hi_basic, gt_lr)
    cov_studentized = coverage(ci_lo_studentized, ci_hi_studentized, gt_lr)
    cov_simultaneous = coverage(ci_lo_simultaneous, ci_hi_simultaneous, gt_lr)

    # R2
    abs_error = (pred_mean - gt_lr).abs()
    r2_g = error_r2(ci_width_g, abs_error, linear_fit=True)
    r2_percentile = error_r2(ci_width_percentile, abs_error, linear_fit=True)
    r2_basic = error_r2(ci_width_basic, abs_error, linear_fit=True)
    r2_studentized = error_r2(ci_width_studentized, abs_error, linear_fit=True)
    r2_simultaneous = error_r2(ci_width_simultaneous, abs_error, linear_fit=True)

    # Correlation
    corr = error_correlation(final_std, abs_error)

    # AUSE (Area Under Sparsification Error)
    # Measures how well uncertainty rankings match error rankings.
    ause_g = sparsification_error(ci_width_g, abs_error)
    ause_percentile = sparsification_error(ci_width_percentile, abs_error)
    ause_basic = sparsification_error(ci_width_basic, abs_error)
    ause_studentized = sparsification_error(ci_width_studentized, abs_error)
    ause_simultaneous = sparsification_error(ci_width_simultaneous, abs_error)

    # Excess Width (Gaussian)
    excess_width = ci_width_g - abs_error

    # --- Formatting Metrics Table ---
    # Collect all metrics into a flat dict.
    # We take the mean over the batch (N) for scalar representation.

    metrics_data = {
        "PSNR": metrics["PSNR"].mean().item(),
        "RMSE": metrics["RMSE"].mean().item(),
        "L1": metrics["L1"].mean().item(),
        "SSIM": metrics["SS"].mean().item(),
        "Corr(Err,Std)": corr.mean().item(),
        "Cov_Gauss": cov_g.mean().item(),
        "Width_Gauss": ci_width_g.mean().item(),
        "R2_Gauss": r2_g.mean().item(),
        "AUSE_Gauss": ause_g.mean().item(),
        "Cov_Percentile": cov_percentile.mean().item(),
        "Width_Percentile": ci_width_percentile.mean().item(),
        "R2_Percentile": r2_percentile.mean().item(),
        "AUSE_Percentile": ause_percentile.mean().item(),
        "Cov_Basic": cov_basic.mean().item(),
        "Width_Basic": ci_width_basic.mean().item(),
        "R2_Basic": r2_basic.mean().item(),
        "AUSE_Basic": ause_basic.mean().item(),
        "Cov_Studentized": cov_studentized.mean().item(),
        "Width_Studentized": ci_width_studentized.mean().item(),
        "R2_Studentized": r2_studentized.mean().item(),
        "AUSE_Studentized": ause_studentized.mean().item(),
        "Cov_Simul": cov_simultaneous.mean().item(),
        "Width_Simul": ci_width_simultaneous.mean().item(),
        "R2_Simul": r2_simultaneous.mean().item(),
        "AUSE_Simul": ause_simultaneous.mean().item(),
    }

    metrics_df = pd.DataFrame([metrics_data])

    logger.info("=== Metrics Table ===")
    # Print transposed for better readability in logs
    logger.info("\n" + metrics_df.T.to_string(header=False))

    # Save to CSV
    if output_prefix:
        csv_name = f"{output_prefix}_metrics.csv"
    else:
        csv_name = f"{dataset}_{total_intensity}_{run_id}_metrics.csv"

    csv_path = output_dir / csv_name
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    # Plotting
    num_plots = min(5, gt_lr.shape[0])
    num_replicates_to_plot = min(5, final_samples.shape[1])

    # Check if we have original predictions
    orig_mean = None
    orig_abs_err = None
    if original_preds is not None:
        # original_preds shape: (N, S, R, H, W) or (N, S, R, 1, H, W)
        # We need the final time step.
        # Check dimensions
        if original_preds.ndim == 5:
            # (N, S, R, H, W) -> unsqueeze channel
            orig_final = original_preds[:, -1, :, :, :].unsqueeze(2)  # (N, R, 1, H, W)
        elif original_preds.ndim == 6:
            orig_final = original_preds[:, -1, :, :, :, :]  # (N, R, 1, H, W)
        else:
            logger.warning(
                f"Unexpected original_preds shape: {original_preds.shape}. Skipping orig plot."
            )
            orig_final = None

        if orig_final is not None:
            orig_mean = orig_final.mean(dim=1)  # (N, 1, H, W)
            # Ensure on same device
            orig_mean = orig_mean.to(gt_lr.device)
            # Interpolate if needed
            if orig_mean.shape[-1] != gt_lr.shape[-1]:
                orig_mean = torch.nn.functional.interpolate(
                    orig_mean, size=gt_lr.shape[-2:], mode="bilinear"
                )

            orig_abs_err = (orig_mean - gt_lr).abs()

    for i in range(num_plots):
        # 3 Rows.
        # Row 1: GT, Pred Mean, Std Dev, [Orig Mean]
        # Row 2: CI Width, Abs Error, Excess Width, [Orig Error]
        # Row 3: Replicates (5 cols)

        # Determine number of columns for the grid
        cols = 4 if orig_mean is not None else 3

        fig = plt.figure(figsize=(20, 15))  # Wider

        # Helper to get subplot
        def get_ax(row, col):
            return plt.subplot2grid((3, cols), (row, col))

        # Row 1
        ax_gt = get_ax(0, 0)
        ax_pred = get_ax(0, 1)
        ax_std = get_ax(0, 2)

        # Ground Truth
        im_gt = ax_gt.imshow(gt_lr[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # Prediction Mean
        im_pred = ax_pred.imshow(
            pred_mean[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        ax_pred.set_title("Ours: Mean")
        ax_pred.axis("off")
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        # Std Dev
        std_np = final_std[i, 0].cpu().numpy()
        im_std = ax_std.imshow(std_np, cmap="inferno")
        ax_std.set_title("Ours: Std Dev")
        ax_std.axis("off")
        plt.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)

        if orig_mean is not None:
            ax_orig = get_ax(0, 3)
            im_orig = ax_orig.imshow(
                orig_mean[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1
            )
            ax_orig.set_title("Orig: Mean")
            ax_orig.axis("off")
            plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)

        # Row 2
        ax_width = get_ax(1, 0)
        ax_err = get_ax(1, 1)
        ax_diff = get_ax(1, 2)

        # CI Width (Gaussian)
        w_np = ci_width_g[i, 0].cpu().numpy()
        im_w = ax_width.imshow(w_np, cmap="inferno")
        ax_width.set_title("Ours: Gaussian CI Width")
        ax_width.axis("off")
        plt.colorbar(im_w, ax=ax_width, fraction=0.046, pad=0.04)

        # Absolute Error
        err_np = abs_error[i, 0].cpu().numpy()
        im_err = ax_err.imshow(err_np, cmap="inferno")
        ax_err.set_title("Ours: Abs Error")
        ax_err.axis("off")
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        # Excess Width (Width - Error)
        diff_np = excess_width[i, 0].cpu().numpy()
        vmax = max(abs(diff_np.min()), abs(diff_np.max()))
        im_diff = ax_diff.imshow(diff_np, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax_diff.set_title("Ours: Width - AbsError")
        ax_diff.axis("off")
        plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

        if orig_abs_err is not None:
            ax_orig_err = get_ax(1, 3)
            o_err_np = orig_abs_err[i, 0].cpu().numpy()
            im_o_err = ax_orig_err.imshow(o_err_np, cmap="inferno")
            ax_orig_err.set_title("Orig: Abs Error")
            ax_orig_err.axis("off")
            plt.colorbar(im_o_err, ax=ax_orig_err, fraction=0.046, pad=0.04)

        # Row 3: Replicates
        # We need to span 'cols' columns.
        # If cols=4, we fit 4 reps, or maybe 5 with colspan?
        # Let's just plot 'cols' replicates.
        for r in range(min(cols, num_replicates_to_plot)):
            ax_rep = get_ax(2, r)
            rep_img = final_samples[i, r, 0].cpu().numpy()
            ax_rep.imshow(rep_img, cmap="gray", vmin=0, vmax=1)
            ax_rep.set_title(f"Ours: Rep {r}")
            ax_rep.axis("off")

        if output_prefix:
            plot_name = f"{output_prefix}_sample_{i}.png"
        else:
            plot_name = f"{dataset}_{total_intensity}_{run_id}_sample_{i}.png"

        save_path = output_dir / plot_name
        plt.suptitle(
            f"Sample {i} | PSNR: {metrics['PSNR'][i].item():.2f} | Cov: {cov_g[i].item():.2f}"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved plot to {save_path}")


@click.command()
@click.argument("parquet_path_str", type=click.Path(exists=True))
@click.option(
    "--time-step",
    type=int,
    default=None,
    help="Relative angle index to enforce boundary at.",
)
@click.option(
    "--dist-loss-fac", type=float, default=1000, help="Weight factor for NLL term."
)
@click.option("--replicates", type=int, default=10, help="Number of replicates.")
@click.option(
    "--idx-range",
    type=int,
    nargs=2,
    default=None,
    help="Range of image indices to process (start, end).",
)
@click.option("--anneal-lr", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
@click.option(
    "--output-prefix",
    type=str,
    default=None,
    help="Prefix for output filenames (e.g. 'boundary_diffusion:...').",
)
def main(
    parquet_path_str: str,
    time_step: int | None,
    dist_loss_fac: float,
    replicates: int,
    idx_range: tuple[int, int] | None,
    verbose: bool,
    anneal_lr: bool,
    output_prefix: str | None,
) -> None:
    """
    Run diffusion with boundary sampling using difference maximization.

    Args:
        parquet_path_str: Path to existing parquet file from a previous run.
        time_step: Relative angle index for boundary constraint.
        dist_loss_fac: Weight factor for distance loss.
        replicates: Number of diffusion replicates per image.
        idx_range: Optional range (start, end) of indices to process.
        verbose: Enable verbose logging.
        anneal_lr: Whether to anneal the learning rate.
    """
    # 1. Load Experiment
    df, dataset, total_intensity, experiment, schedule, config_row, gt = (
        load_experiment_from_parquet(parquet_path_str)
    )
    gt_lr = torch.nn.functional.interpolate(
        gt.unsqueeze(0), size=gt.shape[-1] // 2, mode="area"
    ).squeeze(0)
    assert schedule is not None  # Hint for linter

    # 1.5 Load Original Predictions
    original_preds = None
    h5_path = Path(parquet_path_str).with_suffix(".h5")
    if h5_path.exists():
        logger.info(f"Found original H5 predictions: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if "preds" in f:
                # preds shape: (N, S, R, H, W)
                raw_preds = f["preds"][:]
                original_preds = torch.tensor(raw_preds)
                logger.info(f"Loaded original predictions: {original_preds.shape}")
            else:
                logger.warning(
                    f"Key 'preds' not found in {h5_path}. Keys: {list(f.keys())}"
                )
    else:
        logger.warning(f"Original H5 file not found: {h5_path}")

    # 2. Prepare Conf Coefs
    # Parse 'nll_pred' from dataframe.
    nll_preds_raw = df["nll_pred"].tolist()
    nll_pred_tensor = torch.tensor(
        nll_preds_raw, device=experiment.counts.device, dtype=torch.float32
    )

    # Cumulative NLL for all relative angles
    nll_pred_cum_full = torch.cumsum(nll_pred_tensor, dim=-1)

    # Adjust with delta
    log_inv_delta = math.log(1.0 / 0.05)
    nll_pred_cum_full = nll_pred_cum_full + log_inv_delta

    # Determine target time_step (relative angle index)
    max_relative_idx = nll_pred_cum_full.shape[-1] - 1
    if time_step is None:
        time_step = max_relative_idx
        logger.info(
            f"Not specified time-step. Defaulting to last relative index: {time_step}"
        )
    else:
        if time_step > max_relative_idx:
            logger.warning(f"Time step {time_step} > max {max_relative_idx}. Clipping.")
            time_step = max_relative_idx
        elif time_step < 0:
            logger.warning(f"Time step {time_step} < 0. Clipping to 0.")
            time_step = 0
    conf_coefs = nll_pred_cum_full[:, time_step]

    diffusion = Diffusion(
        dataset=dataset,
        num_steps=100,
        gradient_steps=50,
        lr=1e-2,
        cond=True,
        verbose=verbose,
        anneal_lr=anneal_lr,
    )

    # 4. Run Sampling
    total_samples = experiment.counts.shape[0]
    start_idx = 0
    end_idx = total_samples

    if idx_range is not None:
        r_start, r_end = idx_range
        # Clamp to bounds
        start_idx = max(0, r_start)
        end_idx = min(total_samples, r_end)

        logger.info(
            f"Constraining to range {start_idx}-{end_idx} (from requested {r_start}-{r_end})"
        )

    if start_idx >= end_idx:
        logger.warning(f"Empty range {start_idx}-{end_idx}. Nothing to do.")
        return

    actual_samples = end_idx - start_idx

    experiment = Experiment(
        counts=experiment.counts[start_idx:end_idx],
        intensities=experiment.intensities[start_idx:end_idx],
        angles=experiment.angles,
        sparse=experiment.sparse,
    )
    nll_pred_cum_full = nll_pred_cum_full[start_idx:end_idx]

    # Slice GT as well
    gt_lr = gt_lr[start_idx:end_idx]

    # Slice Original Preds
    if original_preds is not None:
        original_preds = original_preds[start_idx:end_idx]

    logger.info(
        f"Running boundary sampling for {actual_samples} images (indices {start_idx}-{end_idx}), {replicates} replicates."
    )

    # Define Guidance Function
    guidance_fn = get_boundary_guidance_loss_fn(
        experiment, schedule, conf_coefs, time_step, dist_loss_fac
    )

    # Sample
    sampled_images = diffusion.sample(
        experiment,
        replicates=replicates,
        schedule=torch.tensor([time_step], device=experiment.counts.device),
        guidance_loss_fn=guidance_fn,
    )

    # 5. Save Results
    output_dir = get_results_dir() / "boundary_sampling"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = config_row.get("run_id", "unknown")
    # Include range in filename if subset
    range_str = f"_{start_idx}-{end_idx}" if idx_range is not None else ""

    if output_prefix:
        # If output prefix is provided, use it directly (append .h5)
        # We assume the prefix handles unique identification
        fname = f"{output_prefix}.h5"
    else:
        fname = f"{dataset}_{total_intensity}_{run_id}{range_str}_boundary.h5"

    out_path = output_dir / fname

    std_dev = sampled_images.std(dim=2)  # (N, S, 1, H, W) assuming dim 2 is replicates

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "sampled_images", data=sampled_images.cpu().numpy(), compression="gzip"
        )
        f.create_dataset("std_dev", data=std_dev.cpu().numpy(), compression="gzip")
        f.attrs["dataset"] = dataset
        f.attrs["total_intensity"] = total_intensity
        f.attrs["time_step"] = time_step
        f.attrs["dist_loss_fac"] = dist_loss_fac
        f.attrs["start_idx"] = start_idx
        f.attrs["end_idx"] = end_idx

    logger.info(f"Saved results to {out_path}")

    # 6. Evaluate and Plot
    evaluate_and_log_results(
        gt_lr=gt_lr,
        sampled_images=sampled_images,
        std_dev=std_dev,
        output_dir=output_dir,
        dataset=dataset,
        total_intensity=total_intensity,
        run_id=f"{run_id}{range_str}",
        original_preds=original_preds,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    main()
