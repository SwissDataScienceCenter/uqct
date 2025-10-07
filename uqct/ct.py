from typing import Any, Callable

import astra
import numpy as np
import torch

from uqct.debugging import plot_img


def linspace(start, end, steps, endpoint=True, device=None, dtype=None):
    """
    Torch linspace with endpoint option.
    If endpoint=False, the last value is excluded.
    """
    if endpoint:
        return torch.linspace(start, end, steps, device=device, dtype=dtype)
    else:
        if steps == 1:
            return torch.tensor([start], device=device, dtype=dtype)
        step_size = (end - start) / steps
        return torch.arange(steps, device=device, dtype=dtype) * step_size + start


class Tomogram(torch.nn.Module):
    def __init__(
        self,
        prior,
        use_sigmoid: bool,
        sigmoid_alpha: float = 5.0,
        beta_factor=0,
        delta_factor=1e-5,
        circle=False,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.sigmoid_alpha = sigmoid_alpha
        self.beta_factor = beta_factor
        self.delta_factor = delta_factor
        self.prior = prior.detach().clone()
        self.image = torch.nn.Parameter(self.prior.clone())
        self.circle = circle

    def forward(self):
        if self.use_sigmoid:
            image = torch.sigmoid(self.sigmoid_alpha * (self.prior - 0.5 + self.image))
        else:
            # image = self.prior + self.image
            image = self.image

        if self.circle:
            # Create a circular mask
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, image.shape[-2], device=image.device),
                torch.linspace(-1, 1, image.shape[-1], device=image.device),
            )
            mask = x**2 + y**2 <= 1
            image = image * mask
        return image


def anscombe_transform(x):
    """
    Anscombe transform for Poisson noise.
    x: input tensor
    """
    return torch.sqrt(x + 3 / 8)


def poisson(input):
    """
    Sample from a Poisson distribution with the given input.
    If the input is too large, it will sample on CPU to avoid overflow issues.
    """
    if torch.max(input) > 1e9 and input.device.type != "cpu":
        # https://github.com/pytorch/pytorch/issues/86782
        # print(f"Warning: Sampling from poisson distribution with max value {torch.max(input):.2e}, sampling on cpu to avoid incorrect results")
        return torch.poisson(input.cpu()).to(input.device)
    return torch.poisson(input)


def forward_ct(
    images: torch.Tensor,
    angles: torch.Tensor,
    exposure: torch.Tensor,
    l: float = 5.0,
    sinogram_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
):
    """
    forward model
    """
    # batch_dims = images.shape[:-2]
    projections = (
        sinogram_fct(images, angles.flatten())
        if sinogram_fct
        else compute_sinogram(images, angles.flatten())
    )
    scale = l / images.shape[-1]  # Normalize by the image size

    return poisson(exposure * torch.exp(-scale * projections))


def nll(
    predictions: torch.Tensor,
    counts: torch.Tensor,
    I_tot: torch.Tensor,
    angles: torch.Tensor,
    L: int = 5,
) -> torch.Tensor:
    """
    Arguments:
        predictions (torch.Tensor): (..., side_length, side_length)
        counts (torch.Tensor): (..., n_angles, side_length)
        I_tot (torch.Tensor): (..., n_angles, side_length)
        angles (torch.Tensor): (n_angles)
    Returns:
        torch.Tensor: (..., n_angles, side_length)
    """

    I_0 = I_tot / counts.shape[-2]
    radon = compute_sinogram(predictions, angles)
    scale = L / predictions.shape[-1]
    log_lambda = torch.log(I_0 + 1e-9) - scale * radon

    # The expected photon count (lambda)
    lambda_ = I_0 * torch.exp(-scale * radon)

    # The Poisson negative log-likelihood is -(k * log(lambda) - lambda - Gamma(k + 1))
    nll = -(counts * log_lambda - lambda_ - torch.lgamma(counts + 1))
    return nll


def sinogram_ct(measurements, I_0, l=5.0):
    """
    Computes the sinogram from the measurements.
    """
    scale = l / measurements.shape[-1]  # Normalize by the image size
    sinogram = torch.log(measurements / I_0 + 1e-6) / -scale
    return sinogram


def fbp_ct(measurements, angles, exposure, l=5.0, weighted=False, clip=True):
    scale = l / measurements.shape[-1]  # Normalize by the image size
    sinogram = measurements / exposure
    sinogram = torch.log(sinogram + 1e-6) / -scale

    if weighted:
        weights = exposure / exposure.sum(
            dim=-2, keepdim=True
        )  # * np.pi  # Normalize exposure to sum to 1
        sinogram_weighted = sinogram * weights
        recon_weighted = fbp(sinogram_weighted, angles)
        recon_weighted *= len(angles)

        recon = (
            recon_weighted  # if nll_weighted < nll_unweighted  else recon_unweighted
        )
    else:
        recon = fbp(sinogram, angles)
    if clip:
        recon = torch.clip(recon, min=0, max=1)

    return recon


def uniform_allocation(num_angles=360, exposure=1e5, device=None):
    """
    Create a uniform allocation vector for the angles.
    """
    angles = (
        torch.from_numpy(np.linspace(0, 180, num_angles, endpoint=False))
        .float()
        .to(device)
    )
    allocation = torch.ones(num_angles, device=device) * exposure / num_angles
    return allocation.unsqueeze(-1), angles


def random_allocation(num_angles=360, exposure=1e5, device=None):
    angles = (
        torch.from_numpy(np.linspace(0, 180, num_angles, endpoint=False))
        .float()
        .to(device)
    )
    allocation = (
        torch.distributions.Dirichlet(torch.ones(num_angles, device=device))
        .sample()
        .unsqueeze(-1)
    )  # Dirichlet distribution for exposure
    return allocation * exposure, angles


class Experiment:

    def __init__(self, exposure, measurements, angles):
        self.angles = angles
        self.exposure = exposure  # 'allocation' before
        self.measurements = measurements
        self.total_exposure = exposure.sum()  # 'exposure' before

    def aggregate(self, measurements, exposure):
        self.measurements += measurements
        self.exposure += exposure
        self.total_exposure += exposure.sum()

    def clone(self):
        return Experiment(
            self.exposure.clone(), self.measurements.clone(), self.angles.clone()
        )


def mse_ct(images, measurements, angles, exposure, vst=None, l=5.0):
    """
    Mean Squared Error loss function.
    predictions: predicted intensity values
    measurements: observed measurements
    exposure: exposure time or dose
    """
    projections = compute_sinogram(images, angles.flatten())
    scale = l / images.shape[-1]  # Normalize by the image size

    # scale_projection = scale * projections

    predictions = torch.exp(-scale * projections)
    # predictions = torch.clamp(predictions, min=1e-6)  # Ensure predictions are positive
    if vst is not None:
        # predictions_exposure = predictions * exposure
        # Apply variance-stabilizing transformation if provided
        # predictions_exposure = vst(predictions_exposure)
        # measurements = vst(measurements)
        # total_exposure = torch.sum(exposure, dim=(-1, -2), keepdim=True)
        mse_loss = torch.mean((vst(measurements / exposure) - vst(predictions)) ** 2)
    else:
        # Calculate MSE
        mse_loss = torch.mean((measurements / exposure - predictions) ** 2)
    return mse_loss


def get_astra_geometry_3d(
    angles: torch.Tensor, im_size: int, n_slices: int
) -> tuple[dict[str, Any], dict[str, dict]]:
    # ASTRA 3D geometries
    angles_rad = -torch.deg2rad(angles).detach().cpu().numpy()
    det_spacing_x = 1.0
    det_spacing_y = 1.0
    n_det_cols = int(im_size)
    n_det_rows = int(n_slices)

    proj_geom3d = astra.create_proj_geom(
        "parallel3d", det_spacing_y, det_spacing_x, n_det_rows, n_det_cols, angles_rad
    )
    # ASTRA uses (nx, ny, nz) order
    vol_geom3d = astra.create_vol_geom(im_size, im_size, int(n_slices))

    return proj_geom3d, vol_geom3d


def get_astra_geometry_from_images(angles, images):
    assert images.ndim == 3, "images must be 3D (n_slices, H, W)"
    assert (
        images.shape[-1] == images.shape[-2]
    ), f"images must be square (H, W), got images.shape={images.shape}"
    n_slices, im_size = images.shape[0], images.shape[-2]
    return get_astra_geometry_3d(angles, im_size, n_slices)


def get_astra_geometry_from_sinogram(angles, sinogram):
    assert sinogram.ndim == 3, "sinogram must be 3D (n_angles, n_det_y, n_det_x)"
    n_det_rows, n_angles, n_det_cols = sinogram.shape
    assert (
        n_angles == angles.shape[0]
    ), f"angles must match sinogram shape, got angles.shape={angles.shape}, sinogram.shape={sinogram.shape}"
    return get_astra_geometry_3d(angles, n_det_cols, n_det_rows)


class AstraParallelOp3D:
    """
    Torch ⇄ ASTRA 3D parallel-beam operator (GPU preferred).
    Volume: (nz, ny, nx)
    Sinogram: (n_angles, n_det_y, n_det_x)
    """

    def __init__(self, proj_geom: dict[str, Any], vol_geom: dict[str, Any]):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.nz, self.ny, self.nx = (
            vol_geom["GridRowCount"],
            vol_geom["GridColCount"],
            vol_geom["GridSliceCount"],
        )
        self.n_angles = proj_geom["ProjectionAngles"].shape[0]
        self.n_det_y = proj_geom["DetectorRowCount"]
        self.n_det_x = proj_geom["DetectorColCount"]

    def forward(
        self, vol_t: torch.Tensor, out_sino_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        # vol_t: (nz, ny, nx)
        if out_sino_t is None:
            out_sino_t = torch.empty(
                (self.n_det_y, self.n_angles, self.n_det_x),
                device=vol_t.device,
                dtype=torch.float32,
            )

        vol_id = astra.data3d.link("-vol", self.vol_geom, vol_t.detach())
        sino_id = astra.data3d.link("-sino", self.proj_geom, out_sino_t.detach())
        cfg = astra.astra_dict("FP3D_CUDA")
        cfg["VolumeDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        return out_sino_t

    def adjoint(
        self, sino_t: torch.Tensor, out_vol_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        # sino_t: (n_angles, n_det_y, n_det_x)
        if out_vol_t is None:
            out_vol_t = torch.zeros(
                (self.nz, self.ny, self.nx), device=sino_t.device, dtype=torch.float32
            )

        vol_id = astra.data3d.link("-vol", self.vol_geom, out_vol_t.detach())
        sino_id = astra.data3d.link("-sino", self.proj_geom, sino_t.detach())
        cfg = astra.astra_dict("BP3D_CUDA")
        cfg["ReconstructionDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        return out_vol_t


def make_radon_layer(op: AstraParallelOp3D) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Autograd wrapper.
    Input:  x (B, nz, ny, nx)
    Output: y (B, n_angles, n_det_y, n_det_x)
    """

    class ParallelBeam3DFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            y = op.forward(x)
            ctx.op = op
            return y

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:  # type: ignore
            op = ctx.op

            g_vol = torch.zeros(
                (op.nz, op.ny, op.nx), device=grad_out.device, dtype=torch.float32
            )
            op.adjoint(grad_out.detach(), g_vol)
            return g_vol

    return ParallelBeam3DFn.apply  # type: ignore


def compute_sinogram(images: torch.Tensor, angles: torch.Tensor):
    batch_dims = images.size()[:-2]
    img_shape = images.size()[-2:]
    images = images.view(-1, *img_shape)

    proj_geom_3d, vol_geom_3d = get_astra_geometry_from_images(images, angles)
    op3d = AstraParallelOp3D(proj_geom_3d, vol_geom_3d)
    parallel3d_layer = make_radon_layer(op3d)

    sinogram = parallel3d_layer(images.squeeze())

    return sinogram.view(*batch_dims, sinogram.shape[1], img_shape[0])


def _fourier_filter_1d(
    size: int, filter_name: str, device=None, dtype=torch.float32
) -> torch.Tensor:
    if filter_name not in ("ramp", "shepp-logan", "cosine", "hamming", "hann", None):
        raise ValueError(f"Unknown filter: {filter_name}")
    device = device or torch.device("cpu")
    size = int(size)

    n1 = torch.arange(1, size // 2 + 1, 2, device=device)
    n2 = torch.arange(size // 2 - 1, 0, -2, device=device)
    n = torch.cat([n1, n2], dim=0).to(torch.float64)

    f = torch.zeros(size, dtype=torch.float64, device=device)
    f[0] = 0.25
    f[1::2] = -1.0 / (torch.pi * n) ** 2

    fourier_filter = 2.0 * torch.real(torch.fft.fft(f))  # ramp

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = (
            torch.pi * torch.fft.fftfreq(size, device=device, dtype=torch.float64)[1:]
        )
        fourier_filter[1:] *= torch.sin(omega) / torch.where(
            omega == 0, torch.ones_like(omega), omega
        )
    elif filter_name == "cosine":
        # freq in [0, pi)
        freq = torch.arange(size, device=device, dtype=torch.float64) * (
            torch.pi / size
        )
        fourier_filter *= torch.fft.fftshift(torch.sin(freq))
    elif filter_name == "hamming":
        win = torch.hamming_window(
            size, periodic=False, dtype=torch.float64, device=device
        )
        fourier_filter *= torch.fft.fftshift(win)
    elif filter_name == "hann":
        win = torch.hann_window(
            size, periodic=False, dtype=torch.float64, device=device
        )
        fourier_filter *= torch.fft.fftshift(win)
    elif filter_name is None:
        fourier_filter[:] = 1.0

    return fourier_filter.to(dtype).reshape(size, 1)


def _apply_filter_batch(sino: torch.Tensor, filter_name: str) -> torch.Tensor:
    """
    sino: (B, M, N) float32 or (M, N) float32 torch tensor.
    Returns same shape as input, torch tensor.
    """
    if not isinstance(sino, torch.Tensor):
        raise TypeError("sino must be a torch.Tensor")
    single = sino.ndim == 2
    if single:
        sino = sino.unsqueeze(0)  # (1, M, N)
    B, M, N = sino.shape
    device = sino.device
    dtype = sino.dtype

    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * M))))
    P = projection_size_padded

    # pad along detector axis (dim=1) at the end
    sino_padded = torch.zeros((B, P, N), device=device, dtype=dtype)
    sino_padded[:, :M, :] = sino

    filt = _fourier_filter_1d(P, filter_name, device=device, dtype=dtype)  # (P,1)
    filt = filt.view(1, P, 1)

    proj_fft = torch.fft.fft(sino_padded, dim=1)
    proj_fft = proj_fft * filt
    sino_filt = torch.real(torch.fft.ifft(proj_fft, dim=1)).to(dtype)
    sino_filt = sino_filt[:, :M, :]

    return sino_filt[0] if single else sino_filt


def _circular_mask(img_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(img_size, device=device),
        torch.arange(img_size, device=device),
        indexing="ij",
    )
    r = img_size // 2
    mask = ((yy - r) ** 2 + (xx - r) ** 2 <= r**2).to(dtype)
    return mask


def iradon_astra(
    radon_image: torch.Tensor,
    vol_geom3d: dict[str, dict],
    proj_geom3d: dict[str, Any],
    output_size: int | None = None,
    filter_name: str = "ramp",
    circle: bool = True,
) -> torch.Tensor:
    """
    Torch-only I/O fast FBP using ASTRA's 3D backprojection to handle a batch of 2D sinograms.
    Input:  (M, N) or (B, M, N) torch float tensor
    Output: (H, W) or (B, H, W) torch float32 tensor.
    """
    if not isinstance(radon_image, torch.Tensor):
        raise TypeError("radon_image must be a torch.Tensor")
    if radon_image.ndim not in (2, 3):
        raise ValueError("radon_image must be 2-D (M,N) or 3-D (B,M,N)")
    single = radon_image.ndim == 2
    if single:
        radon_image = radon_image.unsqueeze(0)  # (1, M, N)

    # dtype and device handling
    radon_image = radon_image.detach()
    if radon_image.dtype not in (torch.float32, torch.float64):
        radon_image = radon_image.float()
    # ASTRA expects CPU-linked arrays
    radon_image = radon_image.contiguous()  # .cpu()

    B, M, N = radon_image.shape

    # Filter in frequency domain
    sino_filt = _apply_filter_batch(radon_image, filter_name)  # (B, M, N)

    # Output size
    if output_size is None:
        output_size = int(M)
    output_size = int(output_size)

    # Reorder for ASTRA: (n_det_rows, n_angles, n_det_cols)
    sino_3d = sino_filt.permute(
        0, 2, 1
    ).contiguous()  # torch tensor (B, N, M) CPU float

    # Preallocate output and link both
    vol = torch.zeros(
        (B, output_size, output_size),
        dtype=torch.float32,
        device=astra_radon_image.device,  # type: ignore
    )

    sino_id = astra.data3d.link("-sino", proj_geom3d, sino_3d)
    vol_id = astra.data3d.link("-vol", vol_geom3d, vol)

    try:
        try:
            cfg = astra.astra_dict("BP3D_CUDA")
        except Exception:
            cfg = astra.astra_dict("BP3D")
        cfg["ReconstructionDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
    finally:
        astra.data3d.delete([sino_id, vol_id])

    # Scale to match skimage.iradon
    scale = np.pi / (2.0 * float(N))
    vol.mul_(float(scale))

    if circle:
        mask = _circular_mask(output_size, device=vol.device, dtype=vol.dtype)  # (H,W)
        vol *= mask.unsqueeze(0)

    return vol[0] if single else vol


def get_astra_geometry_2d(
    angles_deg: np.ndarray | torch.Tensor, im_size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build ASTRA 2D parallel-beam geometries for a given (variable-length) angle set.
    angles_deg: 1D tensor with degrees in [0, 360]
    im_size: number of detector bins and image width/height (square)
    """
    if isinstance(angles_deg, torch.Tensor):
        angles_deg = angles_deg.cpu().detach().numpy()
    angles_rad = -np.deg2rad(angles_deg)
    det_spacing = 1.0
    n_det = int(im_size)

    proj_geom = astra.create_proj_geom("parallel", det_spacing, n_det, angles_rad)
    vol_geom = astra.create_vol_geom(im_size, im_size)
    return proj_geom, vol_geom


@torch.no_grad()
def forward_angle_sets_2d(
    img_t: torch.Tensor,
    angle_sets: list[np.ndarray],
) -> list[torch.Tensor]:
    if img_t.ndim == 2:
        ny, nx = img_t.shape
        batch = False
    elif img_t.ndim == 3:
        B, ny, nx = img_t.shape
        batch = True
        assert len(angle_sets) == B, "len(angle_sets) must match batch size."
    else:
        raise ValueError("img_t must be (ny, nx) or (B, ny, nx).")

    results = []
    out_device = img_t.device

    for i, angles_deg in enumerate(angle_sets):
        assert angles_deg.ndim == 1, "Each angle set must be a 1D tensor (degrees)."
        img_i = img_t if not batch else img_t[i]
        assert img_i.shape == (ny, nx)

        proj_geom, vol_geom = get_astra_geometry_2d(angles_deg, img_t.shape[-1])

        n_angles_i = int(proj_geom["ProjectionAngles"].shape[0])
        n_det = int(proj_geom["DetectorCount"])

        # --- link CPU arrays to ASTRA ---
        img_np = img_i.detach().contiguous().cpu().to(torch.float32).numpy()
        sino_np = np.empty((n_angles_i, n_det), dtype=np.float32)

        vol_id = astra.data2d.link("-vol", vol_geom, img_np)
        sino_id = astra.data2d.link("-sino", proj_geom, sino_np)

        try:
            try:
                cfg = astra.astra_dict("FP_CUDA")
            except Exception:
                cfg = astra.astra_dict("FP")
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, 1)
            astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([sino_id, vol_id])

        # <-- now move the *filled* sinogram back to torch on the original device
        sino_t = torch.from_numpy(sino_np).to(out_device)
        results.append(sino_t)

    return results


@torch.no_grad()
def fbp_single_from_forward(
    vol_geom: dict[str, Any],
    proj_geom: dict[str, Any],
    sino_t: torch.Tensor,  # (n_angles, n_det) on any device
    filter_name: str = "ramp",
    circle: bool = True,
) -> torch.Tensor:
    """
    Filtered Backprojection for **one** angle set, matching your FP.
    - Filters along detector axis
    - ASTRA 2D BP
    - Scales by number of angles (π/(2·Nθ))
    Returns: (im_size, im_size) on the SAME device as sino_t.
    """
    if sino_t.ndim != 2:
        raise ValueError("sino_t must be (n_angles, n_det)")

    out_device = sino_t.device
    n_angles, im_size = sino_t.shape

    # 1) Filter along detector axis (torch)
    sino_filt = _apply_filter_batch(sino_t.T, filter_name).T

    # 2) ASTRA 2D backprojection (link CPU numpy)
    sino_np = sino_filt.detach().contiguous().cpu().to(torch.float32).numpy()
    vol_np = np.zeros((im_size, im_size), dtype=np.float32)

    sino_id = astra.data2d.link("-sino", proj_geom, sino_np)
    vol_id = astra.data2d.link("-vol", vol_geom, vol_np)

    try:
        try:
            cfg = astra.astra_dict("BP_CUDA")
        except Exception:
            cfg = astra.astra_dict("BP")
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
    finally:
        astra.data2d.delete([sino_id, vol_id])

    # 3) Correct scaling for parallel-beam: scale by number of angles
    vol_np *= np.pi / (2.0 * float(n_angles))

    vol_t = torch.from_numpy(vol_np).to(out_device)
    if circle:
        vol_t *= _circular_mask(im_size, device=vol_t.device, dtype=vol_t.dtype).bool()
    return vol_t


@torch.no_grad()
def forward_and_fbp_2d(
    img_t: torch.Tensor,
    angle_sets: list[np.ndarray],
    exposures: list[float],
    filter_name: str = "ramp",
    circle: bool = True,
    l: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    img_t.squeeze_(1)
    radons = forward_angle_sets_2d(img_t, angle_sets)
    fbps = []
    I_0s = []
    for i, radon in enumerate(radons):
        n_angles = len(angle_sets[i])
        I_0 = exposures[i] / n_angles
        I_0s.append(I_0)
        scale = l / img_t.shape[-1]

        counts = poisson(I_0 * torch.exp(-scale * radon))  # (n_angles, 256)
        counts_lr = counts.view(n_angles, img_t.shape[-1] // 2, 2).sum(
            -1
        )  # (n_angles, 128)
        I_0_lr = I_0 * 2
        sino = sinogram_ct(counts_lr, I_0_lr, l).clamp_min_(0)  # (n_angles, 128)

        proj_geom_lr, vol_geom_lr = get_astra_geometry_2d(
            angle_sets[i], counts_lr.shape[-1]
        )

        fbp = fbp_single_from_forward(
            vol_geom=vol_geom_lr,
            proj_geom=proj_geom_lr,
            sino_t=sino,
            filter_name=filter_name,
            circle=circle,
        ).clip(0, 1)
        fbps.append(fbp)
    return torch.stack(fbps).to(img_t.device), torch.tensor(I_0s, device=img_t.device)


def fbp(sinogram, angles):
    """
    Filtered back projection using astra's implementation.
    Accepts batched sinograms of shape (batch, n_angles, n_detectors).
    Returns reconstructed images of shape (batch, H, W).
    """
    batch_dims = sinogram.size()[:-2]
    sinogram_size = sinogram.size()[-2:]
    sinogram = sinogram.view(-1, *sinogram_size)

    proj_geom_3d, vol_geom_3d = get_astra_geometry_from_sinogram(angles, sinogram)

    recon = iradon_astra(
        sinogram.swapaxes(-1, -2),
        vol_geom3d=vol_geom_3d,
        proj_geom3d=proj_geom_3d,
        output_size=sinogram_size[-1],
        filter_name="ramp",
        circle=True,
    )

    return recon.view(*batch_dims, sinogram_size[-1], sinogram_size[-1])


if __name__ == "__main__":
    side_lengths = [64, 128, 256]
    n_angles = [1, 2, 64]
    batch_dims = [(1,), (2, 3), (5, 6, 7)]

    for i in range(3):
        predictions = torch.rand(side_lengths[i], side_lengths[i])
        counts = torch.rand(*batch_dims[i], n_angles[i], side_lengths[i])
        I_tot = torch.rand(*batch_dims[i], n_angles[i], side_lengths[i])
        angles = torch.linspace(0, 180, n_angles[i])
        nlls = nll(predictions, counts, I_tot, angles)
