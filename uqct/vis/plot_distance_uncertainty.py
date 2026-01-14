import h5py
import click
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm

from uqct.eval.distance import find_prediction_files
from uqct.utils import get_results_dir
from uqct.ct import circular_mask


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path, exists=True),
    default=get_results_dir() / "uncertainty_distance",
    help="Directory containing distance uncertainty results (parquet/h5 pairs).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./plots/distance_uncertainty"),
    help="Directory to save plots.",
)
@click.option(
    "--limit", default=None, type=int, help="Limit number of runs to process."
)
def main(results_dir: Path, output_dir: Path, limit: int | None):
    """
    Visualize distance-based uncertainty results.
    Generates comparison grids (Pred, Maximizer, Uncertainty) and statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    parquet_files = list(results_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No results found in {results_dir}")
        return

    print(f"Found {len(parquet_files)} result files.")

    for i, pq_path in enumerate(tqdm(parquet_files)):
        print(f"{pq_path=}")
        if limit is not None and i >= limit:
            break

        # Load Metadata
        try:
            df = pd.read_parquet(pq_path)
            meta = df.iloc[0]
        except Exception as e:
            print(f"Error reading {pq_path}: {e}")
            continue

        # Construct H5 path (same name but .h5)
        h5_path = pq_path.with_suffix(".h5")

        uncertainty = None
        maximizers = None
        preds = None
        mean_u_val = None  # (N, S) or None if H5 missing
        mean_traj = None  # (S,)

        run_name = pq_path.stem
        run_out_dir = output_dir / run_name
        run_out_dir.mkdir(exist_ok=True)

        if h5_path.exists():
            # Load Results from H5
            with h5py.File(h5_path, "r") as f:
                uncertainty = f["uncertainty"][:]  # (N, S, H, W)
                maximizers = f["maximizers"][:]  # (N, S, H, W)

            # Dequantize if uint8
            if uncertainty.dtype == np.uint8:
                uncertainty = uncertainty.astype(np.float32) / 255.0
            if maximizers.dtype == np.uint8:
                maximizers = maximizers.astype(np.float32) / 255.0

            # Load Original Predictions (Only if H5 present)
            try:
                pred_files = find_prediction_files(
                    model=meta["model"],
                    dataset=meta["dataset"],
                    total_intensity=meta["total_intensity"],
                    sparse=meta["sparse"],
                    seed=meta["seed"],
                    image_range=(meta["image_start_index"], meta["image_end_index"]),
                )

                # Load and Concatenate
                preds_list = []
                for pred_path, _ in pred_files:
                    with h5py.File(pred_path, "r") as f:
                        preds_list.append(f["preds"][:])

                preds_full = np.concatenate(preds_list, axis=0)

                # Slice
                loaded_start = pred_files[0][1][0]
                req_start, req_end = meta["image_start_index"], meta["image_end_index"]
                rel_start = req_start - loaded_start
                rel_end = rel_start + (req_end - req_start)
                preds = preds_full[rel_start:rel_end]

                if preds.ndim == 5:
                    preds = preds.mean(axis=2)
            except Exception as e:
                print(f"Could not load original predictions: {e}")
                preds = None

        else:
            print(f"H5 file missing for {pq_path}. Checking for parquet blob...")

        # 1. Uncertainty Distribution & Samples (Requires full data)
        if uncertainty is not None:
            H, W = uncertainty.shape[-2], uncertainty.shape[-1]
            n_steps = uncertainty.shape[1]

            # Hist
            plt.figure(figsize=(8, 6))
            plt.hist(uncertainty.flatten(), bins=100, log=True)
            plt.title(f"Uncertainty Distribution\n{run_name}")
            plt.xlabel("Pixel Uncertainty (L2 Max - Pred)")
            plt.ylabel("Count (Log)")
            plt.savefig(run_out_dir / "uncertainty_hist.pdf")
            plt.close()

            # Samples
            schedule = meta["pred_angles"]
            if isinstance(schedule, np.ndarray):
                schedule = schedule.tolist()

            n_images = uncertainty.shape[0]
            n_show = min(5, n_images)
            mask_t = circular_mask(H)
            mask = mask_t.numpy().astype(bool)

            for k in range(n_show):
                for s in range(n_steps):
                    if s + 1 < len(schedule):
                        step_val = schedule[s + 1] - 1
                    else:
                        step_val = 200

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # Prediction
                    if preds is not None:
                        if preds.shape[1] == n_steps:
                            p_img = preds[k, s]
                        else:
                            p_img = preds[k, 0]
                        im0 = axes[0].imshow(p_img, cmap="gray", vmin=0, vmax=1)
                        axes[0].set_title("Prediction")
                        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                    else:
                        axes[0].text(0.5, 0.5, "Pred Not Found", ha="center")

                    # Maximizer
                    m_img = maximizers[k, s]
                    im1 = axes[1].imshow(m_img, cmap="gray", vmin=0, vmax=1)
                    axes[1].set_title(f"Maximizer (Step {step_val})")
                    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                    # Uncertainty
                    u_img = uncertainty[k, s]
                    u_mean_scalar = u_img[mask].mean()
                    im2 = axes[2].imshow(u_img, cmap="gray", vmin=0, vmax=1)
                    axes[2].set_title(
                        f"Uncertainty (Step {step_val})\nMean: {u_mean_scalar:.4f}"
                    )
                    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                    for ax in axes:
                        ax.axis("off")

                    plt.suptitle(
                        f"Image {meta['image_start_index'] + k}, Step {step_val}"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        run_out_dir / f"sample_{k}_step_{step_val}.png", dpi=150
                    )
                    plt.close()

            # Stats for evolution
            valid_uncertainty = uncertainty[..., mask]
            mean_u_val = valid_uncertainty.mean(axis=-1)  # (N, S)
            mean_traj = mean_u_val.mean(axis=0)  # (S,)

        else:
            print("Skipping histograms and visual samples (H5 missing).")

        # 3. Uncertainty Evolution
        if mean_traj is None:
            if "mean_pixel_uncertainty" in df.columns:
                try:
                    blob = df.iloc[0]["mean_pixel_uncertainty"]
                    shape = df.iloc[0]["mean_pixel_uncertainty_shape"]
                    if isinstance(shape, np.ndarray):
                        shape = list(shape)

                    mean_pixel_map = np.frombuffer(blob, dtype=np.float32).reshape(
                        shape
                    )  # (S, H, W)

                    H, W = mean_pixel_map.shape[-2], mean_pixel_map.shape[-1]
                    n_steps = mean_pixel_map.shape[0]
                    mask_t = circular_mask(H)
                    mask = mask_t.numpy().astype(bool)

                    mean_traj = mean_pixel_map[..., mask].mean(axis=-1)
                    print("Loaded mean uncertainty trajectory from Parquet blob.")
                except Exception as e:
                    print(f"Failed to load mean pixel uncertainty from Parquet: {e}")

        if mean_traj is not None:
            schedule = meta["pred_angles"]
            if isinstance(schedule, np.ndarray):
                schedule = schedule.tolist()

            x_steps = []
            # Make sure n_steps matches mean_traj
            n_steps = len(mean_traj)

            for s in range(n_steps):
                if s + 1 < len(schedule):
                    x_steps.append(schedule[s + 1] - 1)
                else:
                    x_steps.append(200)

            plt.figure(figsize=(10, 6))
            if mean_u_val is not None:
                for i in range(min(mean_u_val.shape[0], 50)):
                    plt.plot(
                        x_steps, mean_u_val[i], color="gray", alpha=0.3, linewidth=1
                    )

            plt.plot(
                x_steps, mean_traj, color="red", linewidth=2, label="Mean Uncertainty"
            )
            plt.xlabel("Time Step (t)")
            plt.ylabel("Mean Pixel Uncertainty")
            plt.title(f"Uncertainty Evolution\n{run_name}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(run_out_dir / "uncertainty_evolution.png")
            plt.close()

    print(f"Comparisons saved to {output_dir}")


if __name__ == "__main__":
    main()
