import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(img: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        return np.array(img.detach().cpu())
    return img


def plot_img(*images, name=None):
    n = len(images)
    cols = 5
    rows = n // cols
    _, axes = plt.subplots(rows, cols)
    if n == 1:
        axes.imshow(_to_numpy(images[0]))
    else:
        for i in range(n):
            row = i // cols
            col = i - cols * row
            axes[row, col].imshow(_to_numpy(images[i]))
    if name is not None:
        plt.savefig(f"/tmp/{name}.png")
    else:
        plt.show()
    plt.close()
