import glob
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np

from uqct.ct import sinogram_from_counts
from uqct.eval.run import setup_experiment
from uqct.vis.style import ICML_TEXT_WIDTH

DATASETS = ["lamino", "composite", "lung"]
TOTAL_INTENSITY = 1e7


def find_prediction_file(
    dataset: str, model_pattern: str, intensity: float
) -> str | None:
    """Finds the H5 prediction file for the given model/dataset/intensity."""
    intensity_str = f"{intensity:.1f}"
    search_dir = Path("results/runs")

    pattern = f"{search_dir}/{model_pattern}:{dataset}:{intensity_str}*:*.h5"
    files = glob.glob(pattern)

    if not files:
        pattern_alt = f"{search_dir}/{model_pattern}:{dataset}:{int(intensity)}*:*.h5"
        files_alt = glob.glob(pattern_alt)
        if files_alt:
            files = files_alt

    if not files:
        print(f"No file found for {model_pattern} {dataset} {intensity}")
        return None

    files.sort()
    return files[0]


def load_prediction(
    file_path: str, is_ensemble: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Loads prediction. Returns (single_sample, mean_sample)."""
    with h5py.File(file_path, "r") as f:
        if "preds" in f:
            data = f["preds"][:]
        else:
            return None, None

    # Assuming first image
    img_idx = 0
    if len(data) <= img_idx:
        return None, None

    img_data = data[img_idx]

    # Select last time step (index 31 if length 32)
    time_idx = -1

    # Case 1: (T, R, H, W)
    if img_data.ndim == 4:
        final_step = img_data[time_idx]  # (R, H, W)

        if final_step.ndim == 3:
            # (R, H, W)
            if final_step.shape[0] >= 1:
                single = final_step[0]
                mean_pred = np.mean(final_step, axis=0)
                return single, mean_pred

    # Case 2: (T, H, W) -> Deterministic model usually
    elif img_data.ndim == 3:
        final_step = img_data[time_idx]  # (H, W)
        if final_step.ndim == 2:
            return final_step, final_step

    # Case 3: (H, W) -> Single image saved?
    elif img_data.ndim == 2:
        return img_data, img_data

    print(f"DEBUG: Unhandled shape {img_data.shape} for {file_path}")
    return None, None


@click.command()
@click.option("--output-dir", default="plots", help="Directory to save the plot.")
def main(output_dir):
    """Generates comparison plot of reconstructions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # plot styling
    # ICML Text Width approx 6.75 inches

    # 3 rows (datasets), 7 columns
    fig, axes = plt.subplots(
        3, 7, figsize=(ICML_TEXT_WIDTH, 3.1), constrained_layout=True
    )

    labels = [
        "Ground Truth",
        "Sinogram",
        "FBP",
        "MLE",
        "U-Net",
        "U-Net Ens.",
        "Diffusion",
    ]

    for row_idx, dataset in enumerate(DATASETS):
        print(f"Processing {dataset}...")

        # 1. Get GT and Experiment (for Sinogram)
        fbp_file = find_prediction_file(dataset, "fbp", TOTAL_INTENSITY)
        start_idx = 0
        if fbp_file:
            try:
                parts = Path(fbp_file).name.split(":")
                range_part = [
                    p for p in parts if "-" in p and p.replace("-", "").isdigit()
                ][0]
                start_idx = int(range_part.split("-")[0])
            except Exception:
                pass

        gt, experiment, _ = setup_experiment(
            dataset=dataset,
            image_range=(start_idx, start_idx + 1),
            total_intensity=TOTAL_INTENSITY,
            sparse=True,
            seed=0,
            schedule_length=32,
        )

        # GT Image
        gt_img = gt[0].cpu().numpy()

        # Sinogram
        # Use sinogram_from_counts to compute sinogram from counts and intensities
        # Rotate by 90 degrees (transpose)
        sinogram = (
            sinogram_from_counts(experiment.counts[0], experiment.intensities[0])
            .cpu()
            .numpy()
            .T
        )

        # 3. Load Predictions
        fbp_single, _ = load_prediction(fbp_file, False) if fbp_file else (None, None)

        mle_file = find_prediction_file(dataset, "mle", TOTAL_INTENSITY)
        mle_single, _ = load_prediction(mle_file, False) if mle_file else (None, None)

        unet_file = find_prediction_file(dataset, "unet", TOTAL_INTENSITY)
        unet_single, unet_mean = (
            load_prediction(unet_file, False) if unet_file else (None, None)
        )

        diff_file = find_prediction_file(dataset, "diffusion", TOTAL_INTENSITY)
        diff_single, diff_mean = (
            load_prediction(diff_file, False) if diff_file else (None, None)
        )

        row_data = [
            gt_img,
            sinogram,
            fbp_single,
            mle_single,
            unet_single,
            unet_mean,
            diff_mean,
        ]

        for col_idx, img in enumerate(row_data):
            ax = axes[row_idx, col_idx]

            if img is not None:
                # Debug output
                if row_idx == 0:
                    print(f"  Col {col_idx} shape: {img.shape}")

                if col_idx == 1:
                    ax.imshow(
                        img,
                        cmap="gray",
                        origin="upper",
                        interpolation="none",
                    )
                else:
                    ax.imshow(
                        img,
                        cmap="gray",
                        origin="upper",
                        interpolation="none",
                        vmin=0,
                        vmax=1,
                    )
                if col_idx == 1:  # Sinogram
                    ax.set_aspect("auto")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(dataset.title())
            if row_idx == 0:
                ax.set_title(labels[col_idx])

    out_path = output_dir / "reconstructions" / "reconstruction_examples_sinogram.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
