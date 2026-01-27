import click
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from uqct.datasets.utils import get_dataset
from uqct.vis.style import ICML_COLUMN_WIDTH, MODEL_NAMES

DATASETS = ["lamino", "composite", "lung"]


@click.command()
@click.option(
    "--output-dir", default="plots/examples", help="Directory to save the plot."
)
def main(output_dir):
    """Generates a 3x3 grid of example images for each dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Plot
    # 3 columns (datasets), 3 rows (examples)
    # Width ~3.4 inches for single column
    fig, axes = plt.subplots(
        3, 3, figsize=(ICML_COLUMN_WIDTH, 3.4), constrained_layout=True
    )

    # Indices to pick: we want diverse examples.
    # Since datasets are shuffled in get_dataset with seed 0, we can pick fixed indices.
    # Let's pick indices 0, 10, 20 from the test set.
    indices = [0, 20, 40]

    for col_idx, ds_name in enumerate(DATASETS):
        print(f"Loading {ds_name}...")
        # Load Data
        _, test_set = get_dataset(ds_name)

        for row_idx, img_idx in enumerate(indices):
            ax = axes[row_idx, col_idx]

            # Get image
            if img_idx < len(test_set):
                img = test_set[img_idx]  # (1, H, W) or (H, W) or (1, 1, H, W)?

                if isinstance(img, torch.Tensor):
                    img = img.squeeze().cpu().numpy()

                # Normalize for display? Usually they are normalized or in range.
                # Just show as is with gray cmap.
                ax.imshow(img, cmap="gray", origin="upper")
            else:
                ax.axis("off")
                continue

            # Remove interaction
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Spines off?
            # for spine in ax.spines.values():
            #     spine.set_visible(False)

            # Titles on top row only
            if row_idx == 0:
                ax.set_title(ds_name.title(), fontsize=9)

    # Save
    out_path = output_dir / "dataset_examples.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
