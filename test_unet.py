import h5py
import numpy as np

from uqct.debugging import plot_img

with h5py.File(
    "./results/runs/bootstrapping_unet:lung:1000000000.0:True:10-110:0:2026-01-21 16:14:49.877511.h5",
    "r",
) as f:
    preds = np.array(f["preds"])
    breakpoint()


plot_img(*preds[:3].reshape(-1, 128, 128))
