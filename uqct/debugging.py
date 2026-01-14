import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(img: torch.Tensor | np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[0]
    if isinstance(img, torch.Tensor):
        return np.array(img.detach().cpu())
    return img


def plot_img(*images, name=None, max_cols=5, share_range=False):
    n = len(images)
    if n == 0:
        return
    cols = min(max_cols, n)  # shrink cols if fewer images
    rows = (n + cols - 1) // cols  # ceil division

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()

    for i, img in enumerate(images):
        vmin, vmax = 0, 1
        if share_range:
            vmin, vmax = 0, 1
        axes[i].imshow(_to_numpy(img), cmap="gray", vmin=vmin, vmax=vmax)
        axes[i].axis("off")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if name is not None:
        plt.savefig(f"/tmp/{name}.pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)
