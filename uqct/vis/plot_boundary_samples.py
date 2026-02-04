from pathlib import Path
from typing import Any, Literal, cast

import click
import h5py
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from uqct.uq import student_t_ci
from uqct.utils import get_results_dir
from uqct.vis.style import (
    ICML_TEXT_WIDTH,
)

# Order requested: Lamino, Composite, Lung
DATASETS_ORDER = ["lamino", "composite", "lung"]
INTENSITIES = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

# Color Scheme: Inferno
CMAP = "inferno"


def find_results(
    base_dir: Path,
    dataset: str,
    total_intensity: float,
    sparse: bool = True,
    glob_pattern: str = "*.h5",
) -> list[Path]:
    """Finds result files matching criteria."""
    candidates = []
    # If directory doesn't exist, return empty
    if not base_dir.exists():
        return []

    for f in base_dir.glob(glob_pattern):
        name = f.name
        if dataset not in name:
            continue
        try:
            parts = name.split(":")
            # Basic check for enough parts
            # diffusion:dataset:intensity:sparse:range:seed:timestamp...
            if len(parts) < 4:
                continue

            f_dataset = parts[1]
            try:
                f_intensity = float(parts[2])
            except ValueError:
                continue

            f_sparse = parts[3] == "True"

            if (
                f_dataset == dataset
                and np.isclose(f_intensity, total_intensity)
                and f_sparse == sparse
            ):
                candidates.append(f)
        except Exception:
            continue

    # Sort by timestamp (latest first)
    # Timestamp is usually at the end.
    # Logic: sort by full string if timestamp parsing is hard, or specific part
    candidates.sort(key=lambda x: x.name, reverse=True)
    return candidates


def get_file_range(file_path: Path):
    try:
        parts = file_path.stem.split(":")
        # range is index 4 typically: diffusion:dataset:intensity:sparse:range:seed...
        range_part = parts[4]
        if "-" in range_part:
            start, end = map(int, range_part.split("-"))
            return start, end
    except Exception:
        pass
    return None, None


def get_seed(file_path: Path):
    try:
        parts = file_path.stem.split(":")
        # seed is index 5 typically
        return int(parts[5])
    except Exception:
        return 0


def load_gt_images(dataset: str, start: int, end: int) -> torch.Tensor:
    """Loads GT images using setup_experiment logic (simplified)."""
    from uqct.eval.run import setup_experiment

    # Suppress output if possible?
    try:
        ds_lit = cast(Literal["lung", "lamino", "composite"], dataset)
        gt_hr, _, _ = setup_experiment(ds_lit, (start, end), 1e6, True, 0, 32)
        if gt_hr.shape[-1] > 128:
            gt_lr = F.interpolate(
                gt_hr.unsqueeze(1), size=(128, 128), mode="area"
            ).squeeze(1)
        else:
            gt_lr = gt_hr
        return gt_lr
    except Exception as e:
        print(f"Failed to load GT: {e}")
        return torch.zeros((end - start, 128, 128))


def get_vlim_log(images: list[np.ndarray | None]) -> tuple[float, float]:
    """Calculates vmin and vmax for LogNorm dynamic scaling."""
    valid_imgs = [img for img in images if img is not None]
    if not valid_imgs:
        return 1e-5, 1.0

    # Max value
    vmax = np.max([np.max(img) for img in valid_imgs])
    if vmax <= 0:
        return 1e-5, 1.0

    # Min positive value for LogNorm
    vmin = 1.0  # Start high
    found_pos = False
    for img in valid_imgs:
        pos_vals = img[img > 0]
        if pos_vals.size > 0:
            min_p = np.min(pos_vals)
            if min_p < vmin:
                vmin = min_p
                found_pos = True

    if not found_pos:
        # If no positive values found (all 0) but vmax > 0?
        # Should not happen if vmax > 0 check passed, unless max is 0.
        vmin = 1e-5

    # Safety for very small values
    if vmin > vmax:
        vmin = vmax * 1e-2

    # If still essentially 0 or invalid range
    if vmax < 1e-9:
        vmax = 1.0
        vmin = 1e-5

    return vmin, vmax


def load_diffusion_mean_pred(
    dataset: str, total_intensity: float, start_idx: int, end_idx: int, seed: int
) -> torch.Tensor | None:
    """
    Loads diffusion predictions matching parameters and returns mean prediction image.
    Looks for file with matching dataset, intensity, range, and seed.
    """
    base_dir = get_results_dir() / "runs"
    # Pattern to match strictly: diffusion:dataset:intensity:sparse:start-end:seed:*.h5
    # intensity formatting might vary, so we use find_results logic but strictly filtering.

    candidates = find_results(
        base_dir,
        dataset,
        total_intensity,
        sparse=True,
        glob_pattern=f"diffusion:{dataset}:*:True:{start_idx}-{end_idx}:{seed}:*.h5",
    )

    if not candidates:
        return None

    # Take the first match (should be latest due to sort)
    f_path = candidates[0]

    try:
        with h5py.File(f_path, "r") as f:
            if "preds" not in f:
                return None

            # Shape (N, T, R, H, W) or similar
            preds = f["preds"][:]
            preds = torch.from_numpy(preds)

            # We want mean over replicates (dim 2) at last timestep (dim 1 index -1)
            # Assuming shape (N, T, R, H, W) from code analysis
            if preds.ndim == 5:
                # Select last timestep -> (N, R, H, W)
                preds_last = preds[:, -1]
                # Mean over replicates -> (N, H, W)
                preds_mean = preds_last.mean(dim=1)
                return preds_mean
            elif preds.ndim == 4:
                # Maybe (N, T, H, W)? If no replicates?
                preds_last = preds[:, -1]
                return preds_last
    except Exception as e:
        print(f"Error loading diffusion mean: {e}")
        return None
    return None


def get_data_for_dataset(
    dataset: str,
    total_intensity: float,
    boundary_dir: Path,
    distance_dir: Path,
    limit: int,
) -> list[dict]:
    """
    Returns list of data items.
    """
    boundary_files = find_results(boundary_dir, dataset, total_intensity)
    distance_files = find_results(distance_dir, dataset, total_intensity)

    collected_data: list[dict[str, Any]] = []
    seen_indices = set()

    if not boundary_files or not distance_files:
        return collected_data

    # Cache for GT chunks and Diffusion chunks
    gt_cache = {}
    diff_cache = {}

    for b_file in boundary_files:
        if len(collected_data) >= limit:
            break

        try:
            with h5py.File(b_file, "r") as fb:
                b_start_attr = fb.attrs.get("start_idx", 0)
                b_samples_all = fb["sampled_images"][:]  # (N, S, R, 1, H, W)

            # Try to get seed from boundary file if encoded, usually boundary sampling is deterministic or seed 0
            # Plotting script assumes matching seed. Most runs use seed 0.
            current_seed = 0

            b_start_fname, _ = get_file_range(b_file)
            if b_start_fname is not None:
                b_start = b_start_fname
            else:
                b_start = b_start_attr

            b_range_len = b_samples_all.shape[0]

            for d_file in distance_files:
                if len(collected_data) >= limit:
                    break

                d_start_attr, d_end_attr = get_file_range(d_file)
                try:
                    with h5py.File(d_file, "r") as fd:
                        if "uncertainty" in fd:
                            d_uncertainty_all = fd["uncertainty"][:]  # (N, H, W)
                        else:
                            continue

                    d_len = d_uncertainty_all.shape[0]
                    if d_start_attr is None:
                        d_start = 0
                    else:
                        d_start = d_start_attr

                    # Intersection
                    b_indices = range(b_start, b_start + b_range_len)
                    d_indices = range(d_start, d_start + d_len)

                    b_set = set(b_indices)
                    d_set = set(d_indices)
                    common = sorted(list(b_set.intersection(d_set)))

                    for idx in common:
                        if idx in seen_indices:
                            continue
                        if len(collected_data) >= limit:
                            break

                        # Chunk logic
                        chunk_start = (idx // 10) * 10
                        chunk_end = chunk_start + 10
                        chunk_key = (dataset, chunk_start, chunk_end)

                        # GT Loading
                        if chunk_key not in gt_cache:
                            gt_chunk = load_gt_images(dataset, chunk_start, chunk_end)
                            gt_cache[chunk_key] = gt_chunk

                        # Diffusion Prediction Loading
                        # Assume we want to match the same seed and range

                        if chunk_key not in diff_cache:
                            diff_mean = load_diffusion_mean_pred(
                                dataset,
                                total_intensity,
                                chunk_start,
                                chunk_end,
                                current_seed,
                            )
                            diff_cache[chunk_key] = diff_mean

                        gt_img = gt_cache[chunk_key][idx - chunk_start]
                        diff_img = None
                        if diff_cache[chunk_key] is not None:
                            diff_img = diff_cache[chunk_key][idx - chunk_start]

                        b_local = idx - b_start
                        d_local = idx - d_start

                        b_samples = torch.from_numpy(b_samples_all[b_local, -1, :, 0])
                        d_uncertainty = d_uncertainty_all[d_local]

                        collected_data.append(
                            {
                                "index": idx,
                                "gt": gt_img,
                                "diffusion_mean": diff_img,
                                "boundary_samples": b_samples,
                                "distance_uncertainty": d_uncertainty,
                            }
                        )
                        seen_indices.add(idx)
                except Exception:
                    continue
        except Exception:
            continue

    return collected_data


def plot_intensity(
    total_intensity: float,
    boundary_dir: Path,
    distance_dir: Path,
    output_dir: Path,
    limit: int,
    log_scale: bool,
):
    print(f"Generating plot for intensity {total_intensity:.0e}...")

    n_datasets = len(DATASETS_ORDER)
    rows_per_ds = 4  # GT, AbsError, Worst, Boundary

    # Check max rows. 3 blocks * 4 rows = 12 rows.
    # Height calculation:
    # Width ~6.75 in. limit=5 cols.
    # Col Width ~ 1.1 in.
    # Row Height ~ 1.1 in.
    # 12 rows * 1.1 = 13.2 in. + gaps.
    # 12.6 was previous for 9 rows (tight).
    # Need to increase height for 12 rows.
    # 12.6 / 9 * 12 = 16.8 ish.
    fig_height = 10.5

    # Gap between datasets.
    gap_ratio = 0.1
    height_ratios: list[float] = []
    for _ in range(n_datasets):
        height_ratios.extend([1, 1, 1, 1])
        height_ratios.append(gap_ratio)
    height_ratios.pop()

    total_grid_rows = len(height_ratios)

    width_ratios = [1] * limit + [0.15]
    total_cols = limit + 1

    fig = plt.figure(figsize=(ICML_TEXT_WIDTH, fig_height))

    gs = gridspec.GridSpec(
        total_grid_rows,
        total_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.1,
        wspace=0.1,
    )

    for i, dataset in enumerate(DATASETS_ORDER):
        start_row = i * (
            rows_per_ds + 1
        )  # +1 for gap row account in grid indexing logic? NO.
        # Height ratios list defines the grid rows directly.
        # i=0: rows 0,1,2,3. gap is row 4.
        # i=1: start at 5.
        # formula: i * (rows_per_ds + 1) -> 0, 5, 10...

        # Correction:
        # height_ratios has length 14 (4+1 + 4+1 + 4) = 14
        # i=0 -> start 0
        # i=1 -> start 5
        # i=2 -> start 10
        start_row = i * (rows_per_ds + 1)

        data = get_data_for_dataset(
            dataset, total_intensity, boundary_dir, distance_dir, limit
        )

        labels = ["Ground Truth", "Abs Error", "Worst-Case", "Boundary"]

        # Pre-process data
        processed_row_data: dict[int, list[Any]] = {0: [], 1: [], 2: [], 3: []}

        if not data:
            continue

        for item in data:
            # 0: GT
            if isinstance(item["gt"], torch.Tensor):
                gt = item["gt"].cpu().numpy()
            else:
                gt = item["gt"]
            processed_row_data[0].append(gt)

            # 1: Abs Error
            if item["diffusion_mean"] is not None:
                if isinstance(item["diffusion_mean"], torch.Tensor):
                    dm = item["diffusion_mean"].cpu().numpy()
                else:
                    dm = item["diffusion_mean"]
                # Error
                from uqct.ct import circular_mask

                mask = circular_mask(128, dtype=torch.float32).cpu().numpy()
                abs_err = np.abs(dm - gt) * mask
                processed_row_data[1].append(abs_err)
            else:
                processed_row_data[1].append(np.zeros_like(gt))

            # 2: Worst Case
            processed_row_data[2].append(item["distance_uncertainty"])

            # 3: Boundary
            samples = item["boundary_samples"].float()
            lower, upper = student_t_ci(samples, delta=0.05)
            width = (upper - lower).numpy() / 2
            processed_row_data[3].append(width)

        # Fixed log limits as requested: 0 to 1 (using epsilon for 0)
        vmin_log_fixed, vmax_log_fixed = 1e-5, 1.0

        for r_idx in range(rows_per_ds):
            grid_row = start_row + r_idx

            # Determine Vmax/Vmin for this row
            if r_idx == 1:
                vmin_row, vmax_row = vmin_log_fixed, vmax_log_fixed
                is_log = True
            elif r_idx == 2:
                # Worst Case
                if log_scale:
                    vmin_row, vmax_row = vmin_log_fixed, vmax_log_fixed
                    is_log = True
                else:
                    vmin_row, vmax_row = 0.0, 1.0
                    is_log = False
            elif r_idx == 3:
                # Boundary
                if log_scale:
                    vmin_row, vmax_row = vmin_log_fixed, vmax_log_fixed
                    is_log = True
                else:
                    vmin_row, vmax_row = 0.0, 1.0
                    is_log = False
            else:
                # GT
                vmin_row, vmax_row = 0.0, 1.0
                is_log = False

            for col in range(limit):
                ax = fig.add_subplot(gs[grid_row, col])

                if col < len(processed_row_data[r_idx]):
                    img = processed_row_data[r_idx][col]

                    if r_idx == 0:
                        # GT: Gray
                        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    else:
                        # Rows 1, 2, 3: Inferno (Log or Linear)
                        if is_log:
                            ax.set_facecolor("black")
                            norm = mcolors.LogNorm(vmin=vmin_row, vmax=vmax_row)
                            ax.imshow(img, cmap=CMAP, norm=norm)
                        else:
                            # Linear
                            ax.imshow(img, cmap=CMAP, vmin=vmin_row, vmax=vmax_row)

                else:
                    ax.imshow(np.zeros((10, 10)), cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")

                ax.set_xticks([])
                ax.set_yticks([])

                # Row Layout Labels
                if col == 0:
                    ax.set_ylabel(labels[r_idx], rotation=90, va="center", labelpad=5)

                    if r_idx == 1:
                        ax.text(
                            -0.5,
                            0.0,
                            f"{dataset.title()} Dataset",
                            transform=ax.transAxes,
                            fontweight="bold",
                            rotation=90,
                            va="center",
                            ha="right",
                        )

            # Draw Colorbar
            cbar_ax = fig.add_subplot(gs[grid_row, limit])

            if r_idx == 0:
                norm = mcolors.Normalize(vmin=0, vmax=1)
                mappable = plt.cm.ScalarMappable(norm=norm, cmap="gray")
                plt.colorbar(mappable, cax=cbar_ax)
                cbar_ax.tick_params(labelsize=6)
            else:
                if is_log:
                    norm = mcolors.LogNorm(vmin=vmin_row, vmax=vmax_row)
                else:
                    norm = mcolors.Normalize(vmin=vmin_row, vmax=vmax_row)
                mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
                plt.colorbar(mappable, cax=cbar_ax)
                cbar_ax.tick_params(labelsize=6)

    # PDF Output
    out_path = output_dir / f"boundary_sampling_vs_worstcase_{total_intensity:.0e}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


@click.command()
@click.option(
    "--boundary-dir",
    type=click.Path(path_type=Path, exists=True),
    default=get_results_dir() / "boundary_sampling",
    help="Directory containing boundary sampling results.",
)
@click.option(
    "--distance-dir",
    type=click.Path(path_type=Path, exists=True),
    default=get_results_dir() / "uncertainty_distance",
    help="Directory containing distance maximization results.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots/boundary_samples"),
    help="Directory to save plots.",
)
@click.option(
    "--total-intensity",
    type=float,
    default=None,
    help="Total intensity to filter by. If invalid or not provided, runs all.",
)
@click.option("--limit", default=5, type=int, help="Number of columns (test images).")
@click.option(
    "--all-intensities", is_flag=True, help="Run for all standard intensities."
)
@click.option(
    "--log-scale",
    is_flag=True,
    help="Apply log scale to Worst-Case and Boundary Uncertainty plots.",
)
def main(
    boundary_dir: Path,
    distance_dir: Path,
    output_dir: Path,
    total_intensity: float | None,
    limit: int,
    all_intensities: bool,
    log_scale: bool,
):
    """
    Generates summary plots for Worst Case vs Boundary Sampling Uncertainty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_intensities:
        run_list = INTENSITIES
    elif total_intensity is not None:
        run_list = [total_intensity]
    else:
        run_list = INTENSITIES

    for inten in run_list:
        plot_intensity(
            inten, boundary_dir, distance_dir, output_dir, limit, log_scale=log_scale
        )

    print("Done.")


if __name__ == "__main__":
    main()
