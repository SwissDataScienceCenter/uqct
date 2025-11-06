import math
from typing import Any, Callable, Literal

import astra
import numpy as np
import torch


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
            mask = circular_mask(
                image.shape[-1], device=image.device, dtype=image.dtype
            )  # (H,W)
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
        return torch.poisson(input.cpu()).to(input.device)
    return torch.poisson(input)


def nll(
    images: torch.Tensor,
    counts: torch.Tensor,
    intensities: torch.Tensor,
    angles: torch.Tensor,
    l: int = 5,
) -> torch.Tensor:
    """Poisson negative log-likelihood.

    Arguments:
        images (torch.Tensor): (..., width, height)
        counts (torch.Tensor): (..., n_angles, n_detectors)
        intensities (torch.Tensor): (..., n_angles, 1)
        angles (torch.Tensor): (n_angles)
        l (int)
    Returns:
        torch.Tensor: (..., n_angles, side_length)
    """
    assert images.ndim >= 2 and counts.ndim >= 2 and angles.ndim == 1, (
        f"angles ({angles.shape}) must be 1D and predictions ({images.shape}) and counts ({counts.shape}) must be at least two dimensional."
    )
    intensities = intensities.clip(1e-9)
    sino = radon(images, angles)
    scale = l / images.shape[-1]
    nll = (
        -counts * (torch.log(intensities) - scale * sino)
        + intensities * torch.exp(-scale * sino)
        + torch.lgamma(counts + 1)
    )
    return nll


def nll_mixture(
    images: torch.Tensor,
    counts: torch.Tensor,
    intensities: torch.Tensor,
    angles: torch.Tensor,
    l: int = 5,
) -> torch.Tensor:
    """
    Arguments:
        images (torch.Tensor): (..., n_preds, H, W)
        counts (torch.Tensor): (..., n_angles, n_detectors)
        intensities (torch.Tensor): (..., n_angles, 1)
        angles (torch.Tensor): (n_angles,)
        l: (int)
    Returns:
        torch.Tensor: (...)
    """
    n_pred = images.shape[-3]
    # (..., n_pred, n_angles, side_length)
    nlls = nll(images, counts.unsqueeze(-3), intensities, angles, l)
    nlls -= math.log(n_pred)
    nlls = nlls.sum((-1, -2))  # (..., n_pred)
    mix = torch.logsumexp(nlls, dim=-1)  # (...)
    return mix


def radon(images: torch.Tensor, angles: torch.Tensor):
    """Computes sinogram.

    Arguments:
        images (torch.Tensor): (..., H, W)
        angles (torch.Tensor): (n_angles,)
    Returns:
        sinogram (torch.Tensor): (..., n_angles, n_detectors)
    """
    batch_dims = images.size()[:-2]
    img_shape = images.size()[-2:]
    images = images.view(-1, *img_shape)

    proj_geom_3d, vol_geom_3d = get_astra_geometry_from_images(angles, images)

    op3d = AstraParallelOp3D(proj_geom_3d, vol_geom_3d)
    parallel3d_layer = make_radon_layer(op3d)

    sino = parallel3d_layer(images)

    return sino.view(*batch_dims, sino.shape[1], img_shape[0])


FilterType = Literal["ramp", "shepp-logan", "cosine", "hamming", "hann"] | None


def fbp(
    sino: torch.Tensor,
    angles: torch.Tensor,
    filter_name: FilterType = "ramp",
    circle: bool = True,
) -> torch.Tensor:
    """Computes FBP from sinogram.

    Arguments:
        sino (torch.Tensor): (..., n_angles, n_detectors)
        angles (torch.Tensor): (n_angles,)
        filter_name (str | None): One of "ramp", "shepp-logan", "cosine", "hamming", "hann" or None
        circle (bool): map values outside inscribed circle to zero
    Returns:
        fbp (torch.Tensor): (..., H, W)
    """
    if sino.ndim < 2:
        raise ValueError("sinogram must be at least 2D")

    batch_dims = sino.size()[:-2]
    sino_size = sino.size()[-2:]
    sino = sino.view(-1, *sino_size)

    proj_geom_3d, vol_geom_3d = get_astra_geometry_from_sinogram(angles, sino)

    sino.swapaxes_(-2, -1)

    single = sino.ndim == 2
    if single:
        astra_radon_image = sino.unsqueeze(0)  # (1, M, N)

    # dtype and device handling
    astra_radon_image = sino.detach()
    if astra_radon_image.dtype not in (torch.float32, torch.float64):
        astra_radon_image = astra_radon_image.float()

    # ASTRA expects CPU-linked arrays
    astra_radon_image = astra_radon_image.contiguous()

    B, _, N = astra_radon_image.shape

    # Filter in frequency domain
    sino_filt = _apply_filter_batch(astra_radon_image, filter_name)  # (B, M, N)

    # Reorder for ASTRA: (n_det_rows, n_angles, n_det_cols)
    sino_3d = sino_filt.permute(0, 2, 1)  # torch tensor (B, N, M) float

    # Fix for: BP3D_CUDA crashes with n_angles=1 on CUDA crashes
    out_device = sino_3d.device
    if sino_3d.shape[-2] == 1:
        sino_3d = sino_3d.cpu()

    # Preallocate output and link both
    sino_3d = sino_3d.contiguous()
    vol = torch.zeros(
        (B, sino_size[-1], sino_size[-1]), dtype=torch.float32, device=out_device
    ).contiguous()
    sino_id = astra.data3d.link("-sino", proj_geom_3d, sino_3d)
    vol_id = astra.data3d.link("-vol", vol_geom_3d, vol)

    try:
        cfg = astra.astra_dict("BP3D_CUDA")
        cfg["ReconstructionDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
    finally:
        astra.data3d.delete([sino_id, vol_id])

    vol = vol.to(out_device)
    # Scale to match skimage.iradon
    scale = np.pi / (2.0 * float(N))
    vol.mul_(float(scale))

    if circle:
        mask = circular_mask(sino_size[-1], device=vol.device, dtype=vol.dtype)  # (H,W)
        vol *= mask.unsqueeze(0)

    out = vol[0] if single else vol

    return out.view(*batch_dims, sino_size[-1], sino_size[-1])


def sample_observations(
    images: torch.Tensor,
    intensities: torch.Tensor,
    angles: torch.Tensor,
    l: float = 5.0,
) -> torch.Tensor:
    """Samples Poisson counts based on high-res images

    Important: Images should be high-res images to avoid inverse crime.

    Arguments:
        images (torch.Tensor): `(..., H, W)`
        intensities (torch.Tensor): `(..., n_angles, 1)`
        angles (torch.Tensor): `(n_angles,)`
    Returns:
        counts (torch.Tensor): `(..., n_angles, n_detectors)`
    """
    scale = l / images.shape[-1]
    sino = radon(images, angles)
    counts = poisson(intensities * torch.exp(-scale * sino))
    counts_lr = counts.view(*counts.shape[:-1], counts.shape[-1] // 2, 2).sum(-1)
    return counts_lr


def sinogram_from_counts(
    counts: torch.Tensor, intensities: torch.Tensor | float, l=5.0
) -> torch.Tensor:
    """
    Computes the sinogram from the measurements.
    """
    scale = l / counts.shape[-1]  # Normalize by the image size
    sino = -torch.log(counts.clip(1e-9) / intensities) / scale
    return sino


class Experiment:
    """In the dense settings we have
        counts (torch.Tensor): `(..., T, n_angles, n_detectors)`
        intensities (torch.Tensor): `(..., T, n_angles, 1)`
        angles (torch.Tensor): `(n_angles)`

    In the sparse setting
        counts (torch.Tensor): `(...,  n_angles, n_detectors)`
        intensities (torch.Tensor): `(..., n_angles, 1)`
        angles (torch.Tensor): `(n_angles)`
    """

    def __init__(
        self,
        counts: torch.Tensor,
        intensities: torch.Tensor,
        angles: torch.Tensor,
        sparse: bool,
    ):
        def _broadcast_dim(a: int, b: int, label: str) -> int:
            if a == b or a == 1 or b == 1:
                return a if a != 1 else b
            raise ValueError(
                f"Incompatible {label} dimensions for counts {counts.shape} and intensities {intensities.shape}"
            )

        trailing_dims = 2 if sparse else 3
        if counts.ndim < trailing_dims or intensities.ndim < trailing_dims:
            raise ValueError(
                f"Counts {counts.shape} and intensities {intensities.shape} are not compatible "
                f"with sparse={sparse}"
            )

        try:
            torch.broadcast_shapes(
                counts.shape[:-trailing_dims], intensities.shape[:-trailing_dims]
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Incompatible batch dimensions for counts {counts.shape} and intensities {intensities.shape}"
            ) from exc

        if sparse:
            n_angles = _broadcast_dim(counts.shape[-2], intensities.shape[-2], "angle")
            _broadcast_dim(counts.shape[-1], intensities.shape[-1], "detector")
        else:
            _broadcast_dim(counts.shape[-3], intensities.shape[-3], "time")
            n_angles = _broadcast_dim(counts.shape[-2], intensities.shape[-2], "angle")
            _broadcast_dim(counts.shape[-1], intensities.shape[-1], "detector")

        if angles.ndim != 1 or angles.shape[0] != n_angles:
            raise ValueError(
                f"Angles shape {angles.shape} does not match broadcasted angle dimension {n_angles}"
            )
        self.angles = angles
        self.intensities = intensities
        self.counts = counts
        if sparse:
            self.total_exposure = intensities.sum((-2, -1)) * self.counts.shape[-1]
            self.batch_dims = counts.shape[:-2]
        else:
            self.total_exposure = intensities.sum((-3, -2, -1)) * self.counts.shape[-1]
            self.batch_dims = counts.shape[:-3]
        self.sparse = sparse

    def __str__(self) -> str:
        return f"Experiment:\n  sparse: {self.sparse}\n  intensities: {self.intensities}\n  counts: {self.counts}\n  angles: {self.angles}"

    def __repr__(self) -> str:
        return self.__str__()

    def to(self, device: torch.device) -> "Experiment":
        self.angles = self.angles.to(device)
        self.intensities = self.intensities.to(device)
        self.counts = self.counts.to(device)
        self.total_exposure = self.total_exposure.to(device)
        return self


def get_astra_geometry_3d(
    angles: torch.Tensor, im_size: int, n_slices: int
) -> tuple[dict[str, Any], dict[str, dict]]:
    # ASTRA 3D geometries
    angles_rad = torch.deg2rad(angles).detach().cpu().numpy()
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


def get_astra_geometry_from_images(
    angles: torch.Tensor, images: torch.Tensor
) -> tuple[dict[str, Any], dict[str, dict]]:
    assert images.ndim == 3, "images must be 3D (n_slices, H, W)"
    assert images.shape[-1] == images.shape[-2], (
        f"images must be square (H, W), got images.shape={images.shape}"
    )
    n_slices, im_size = images.shape[0], images.shape[-2]
    return get_astra_geometry_3d(angles, im_size, n_slices)


def get_astra_geometry_from_sinogram(
    angles: torch.Tensor, sino: torch.Tensor
) -> tuple[dict[str, Any], dict[str, dict]]:
    assert sino.ndim == 3, "sinogram must be 3D (n_angles, n_det_y, n_det_x)"
    n_det_rows, n_angles, n_det_cols = sino.shape
    assert n_angles == angles.shape[0], (
        f"angles must match sinogram shape, got angles.shape={angles.shape}, sinogram.shape={sino.shape}"
    )
    return get_astra_geometry_3d(angles, n_det_cols, n_det_rows)


class AstraParallelOp3D:
    """
    Torch ⇄ ASTRA 3D parallel-beam operator (GPU preferred).
    Volume: `(nz, ny, nx)`
    Sinogram: `(n_angles, n_det_y, n_det_x)`
    """

    def __init__(self, proj_geom: dict[str, Any], vol_geom: dict[str, Any]):
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.ny, self.nx, self.nz = (
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

        # Fix for: FP3D_CUDA crashes with n_angles=1 on CUDA crashes
        out_device = vol_t.device
        if self.n_angles == 1:
            sino_device = torch.device("cpu")
        else:
            sino_device = out_device

        if out_sino_t is None:
            out_sino_t = torch.empty(
                (self.n_det_y, self.n_angles, self.n_det_x),
                device=sino_device,
                dtype=torch.float32,
            )
        else:
            out_sino_t.to(sino_device)

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
        return out_sino_t.to(out_device)

    def adjoint(
        self, sino_t: torch.Tensor, out_vol_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        # sino_t: (n_angles, n_det_y, n_det_x)
        if out_vol_t is None:
            out_vol_t = torch.zeros(
                (self.nz, self.ny, self.nx), device=sino_t.device, dtype=torch.float32
            )

        # Fix for: BP3D_CUDA crashes with n_angles=1 on CUDA crashes
        out_device = sino_t.device
        if sino_t.shape[-2] == 1:
            sino_t = sino_t.cpu()
        sino_t.contiguous()

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
        return out_vol_t.to(out_device)


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


def _fourier_filter_1d(
    size: int,
    filter_name: FilterType,
    device: torch.device | None = None,
    dtype=torch.float32,
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


def _apply_filter_batch(
    sino: torch.Tensor,
    filter_name: FilterType,
) -> torch.Tensor:
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


def circular_mask(img_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(img_size, device=device),
        torch.arange(img_size, device=device),
        indexing="ij",
    )
    r = img_size // 2
    mask = ((yy - r) ** 2 + (xx - r) ** 2 <= r**2).to(dtype)
    return mask


##################################################
#              2D Functions (U-Net Training)
##################################################


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


def fbp_single_from_forward(
    vol_geom: dict[str, Any],
    proj_geom: dict[str, Any],
    sino_t: torch.Tensor,  # (n_angles, n_det) on any device
    filter_name: Literal[
        "ramp", "shepp-logan", "cosine", "hamming", "hann", None
    ] = "ramp",
    circle: bool = True,
) -> torch.Tensor:
    """
    Filtered Backprojection for **one** angle set, matching your FP.
    - Filters along detector axis
    - ASTRA 2D BP
    - Scales by number of angles (π/(2·Nθ))
    Returns: `(im_size, im_size)` on the SAME device as sino_t.
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
        vol_t *= circular_mask(im_size, device=vol_t.device, dtype=vol_t.dtype).bool()
    return vol_t


@torch.no_grad()
def forward_and_fbp_2d(
    image: torch.Tensor,
    angle_sets: list[np.ndarray],
    total_intensities: list[float],
    filter_name: Literal[
        "ramp", "shepp-logan", "cosine", "hamming", "hann", None
    ] = "ramp",
    circle: bool = True,
    l: int = 5,
) -> torch.Tensor:
    """
    TODO

    Arguments:
        image (torch.Tensor): (B, side_length, side_length)
    """
    image = image.squeeze(1)
    radons = forward_angle_sets_2d(image, angle_sets)
    fbps = list()
    n_bins = radons[0].shape[-1]
    for i, radon in enumerate(radons):
        n_angles = len(angle_sets[i])
        intensity = total_intensities[i] / n_angles / n_bins
        scale = l / n_bins

        counts = poisson(intensity * torch.exp(-scale * radon))  # (n_angles, 256)
        counts_lr = counts.view(n_angles, image.shape[-1] // 2, 2).sum(
            -1
        )  # (n_angles, 128)
        intensity_lr = intensity * 2
        sino = sinogram_from_counts(counts_lr, intensity_lr, l).clamp_min_(
            0
        )  # (n_angles, 128)

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
    return torch.stack(fbps).to(image.device)


def fbp_2d(
    angle_sets: list[np.ndarray],
    intensities: list[float] | list[torch.Tensor],
    counts: list[torch.Tensor],
    filter_name: Literal[
        "ramp", "shepp-logan", "cosine", "hamming", "hann", None
    ] = "ramp",
    circle: bool = True,
) -> torch.Tensor:
    """
    Arguments:
        image (torch.Tensor): `(B, side_length, side_length)`
    """
    fbps = []
    for angle_set_i, counts_i, intensity_i in zip(angle_sets, counts, intensities):
        if not isinstance(intensity_i, torch.Tensor):
            intensity_i = torch.tensor(intensity_i)
        print(f"{intensity_i=}")
        sino = sinogram_from_counts(
            counts_i, intensity_i.clone().to(counts_i.device)
        ).clamp_min(0)
        proj_geom_lr, vol_geom_lr = get_astra_geometry_2d(
            angle_set_i, counts_i.shape[-1]
        )
        print(f"{sino=}")
        fbp = fbp_single_from_forward(
            vol_geom=vol_geom_lr,
            proj_geom=proj_geom_lr,
            sino_t=sino.view(-1, sino.shape[-1]),
            filter_name=filter_name,
            circle=circle,
        ).clip(0, 1)
        fbps.append(fbp)
    return torch.stack(fbps).to(counts[0].device)


def apply_circular_mask(x: torch.Tensor) -> torch.Tensor:
    mask = circular_mask(x.shape[-1], device=x.device)
    return x * mask


if __name__ == "__main__":
    b = 3
    r = 128
    n_pred = 5
    # n_angles = 200
    n_angles = 1
    n_detectors = r
    rates = 100.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    preds = torch.rand(b, n_pred, r, r, device=device)
    counts = poisson(torch.full((b, n_angles, n_detectors), rates, device=device))
    intensities = torch.rand(n_angles, 1, device=device)
    angles = torch.rand(n_angles, device=device) * 180
    nlls = nll_mixture(preds, counts, intensities, angles)
    assert nlls.shape[0] == b and nlls.ndim == 1
    rad = radon(preds, angles)
    assert (
        rad.shape[0] == b
        and rad.shape[1] == n_pred
        and rad.shape[2] == n_angles
        and rad.shape[3] == n_detectors
    )
    rad = radon(preds[0, 0], angles)
    assert rad.shape[0] == n_angles and rad.shape[1] == n_detectors
    rad = radon(preds[0], angles)
    assert (
        rad.shape[0] == n_pred
        and rad.shape[1] == n_angles
        and rad.shape[2] == n_detectors
    )

    fbp_ = fbp(rad, angles)
