import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from skimage.transform import iradon, radon


def batched_sinogram(images, sinogram_angles=None, interpolation: str = "bilinear"):
    assert len(images.shape) == 3

    device = images.device
    sinogram_angles = sinogram_angles.to(device)

    rotation_matrix = torch.stack(
        [
            torch.stack(
                [
                    torch.cos(torch.deg2rad(sinogram_angles)),
                    -torch.sin(torch.deg2rad(sinogram_angles)),
                    torch.zeros_like(sinogram_angles),
                ],
                1,
            ).to(device),
            torch.stack(
                [
                    torch.sin(torch.deg2rad(sinogram_angles)),
                    torch.cos(torch.deg2rad(sinogram_angles)),
                    torch.zeros_like(sinogram_angles),
                ],
                1,
            ).to(device),
        ],
        1,
    )
    current_grid = F.affine_grid(
        rotation_matrix.to(images.device),
        images.repeat(len(sinogram_angles), 1, 1, 1).size(),
        align_corners=False,
    )

    rotated = F.grid_sample(
        images.repeat(len(sinogram_angles), 1, 1, 1).float(),
        current_grid.repeat(1, 1, 1, 1),
        align_corners=False,
        mode=interpolation,
    )
    rotated = rotated.transpose(0, 1)
    # Sum over one of the dimensions to compute the projection
    sinogram = rotated.sum(axis=-2).squeeze(2)
    return sinogram


def compute_sinogram(images, angles, interpolation: str = "bilinear"):
    batch_dims = images.size()[:-2]
    img_shape = images.size()[-2:]
    images = images.view(-1, *img_shape)
    sinogram = batched_sinogram(
        images, sinogram_angles=angles, interpolation=interpolation
    )
    return sinogram.view(*batch_dims, len(angles), img_shape[0])


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


def fbp(sinogram, angles, filter_name="ramp"):
    """
    Filtered back projection using sklearn's implementation.
    Accepts batched sinograms of shape (batch, n_angles, n_detectors).
    Returns reconstructed images of shape (batch, H, W).
    """
    batch_dims = sinogram.size()[:-2]
    sinogram_size = sinogram.size()[-2:]
    sinogram = sinogram.view(-1, *sinogram_size)
    sinogram_np = sinogram.cpu().numpy()
    angles_np = angles.cpu().numpy()
    batch_size = sinogram_np.shape[0]
    recon_list = []
    for i in range(batch_size):
        recon = iradon(sinogram_np[i].T, theta=-angles_np, filter_name=filter_name)
        recon_list.append(torch.tensor(recon, device=sinogram.device))
    return torch.stack(recon_list).view(
        *batch_dims, sinogram_size[-1], sinogram_size[-1]
    )


def radon_sklearn(images, angles):
    """
    Compute the Radon transform (sinogram) using sklearn's iradon.
    Accepts batched images of shape (batch, H, W).
    Returns sinograms of shape (batch, n_angles, n_detectors).
    """
    batch_size = images.shape[0]
    sinograms = []
    for i in range(batch_size):
        sinogram = radon(images[i].cpu().numpy(), theta=-angles.cpu().numpy()).T
        sinograms.append(torch.tensor(sinogram, device=images.device))
    return torch.stack(sinograms)


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


# def finetune(image, measurements, angles, exposure, num_steps=100, verbose=False):
#     # if not isinstance(image, Tomogram):
#     #     image = Tomogram(prior=image, use_sigmoid=False, sigmoid_alpha=5.0).to(device)

#     image.train()
#     optimizer = torch.optim.Adam(image.parameters(), lr=1e-3)
#     losses = []
#     it = tqdm(range(num_steps), desc="Finetuning", disable=not verbose)
#     for step in it:
#         optimizer.zero_grad()
#         recon = image()
#         proj_recon = tomography2d(recon, angles)
#         loss = mse(proj_recon, measurements, exposure=exposure, vst=anscombe_transform)
#         # loss = mse(proj_recon, measurements, exposure=exposure)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())

#         it.set_postfix(loss=loss.item())

#         # if step % 10 == 0:
#         #     print(f"Step {step}: Loss = {loss.item():.4f}")
#     return image, losses


def forward_ct(images, angles, exposure, l=5.0, sinogram_fct=None):
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

    # scale_projection = scale * projections
    # print(f"min/max of projections: {scale_projection.min().item()}/{scale_projection.max().item()}")

    return poisson(exposure * torch.exp(-scale * projections))


def nll_ct(images, measurements, angles, exposure, l=5.0):
    """
    Computes the negative log-likelihood for Poisson distributed measurements.
    """
    sinogram = compute_sinogram(images, angles.flatten())
    scale = l / images.shape[-1]

    # The log of the expected photon count (lambda) is log(exposure * exp(-scale * sinogram))
    # which expands to log(exposure) - scale * sinogram
    log_lambda = torch.log(exposure + 1e-9) - scale * sinogram

    # The expected photon count (lambda)
    lambda_ = exposure * torch.exp(-scale * sinogram)

    # The Poisson negative log-likelihood is -(k * log(lambda) - lambda)
    nll = -(measurements * log_lambda - lambda_)

    return nll.sum(dim=(-1, -2, -3))  # Mean over all measurements


def sinogram_ct(measurements, exposure, l=5.0):
    """
    Computes the sinogram from the measurements.
    """
    scale = l / measurements.shape[-1]  # Normalize by the image size
    sinogram = torch.log(measurements / exposure + 1e-6) / -scale
    return sinogram


def fbp_ct(measurements, angles, exposure, l=5.0, weighted=False, clip=True):
    scale = l / measurements.shape[-1]  # Normalize by the image size
    sinogram = measurements / exposure
    sinogram = torch.log(sinogram + 1e-6) / -scale

    if weighted:
        weights = exposure / exposure.sum(
            dim=-2, keepdim=True
        )  # * np.pi  # Normalize exposure to sum to 1
        # print(f"min/max of weights: {weights.min().item()}/{weights.max().item()}")
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


def finetune_ct(
    image, measurements, angles, exposure, num_steps=100, verbose=False, l=5.0
):
    # if not isinstance(image, Tomogram):
    #     image = Tomogram(prior=image, use_sigmoid=False, sigmoid_alpha=5.0).to(device)

    image.train()
    optimizer = torch.optim.Adam(image.parameters(), lr=1e-3)
    losses = []
    it = tqdm(range(num_steps), desc="Finetuning", disable=not verbose)
    for step in it:
        optimizer.zero_grad()
        recon = image()
        # proj_recon = tomography2d(recon, angles)/
        # loss = mse(proj_recon, measurements, exposure=exposure, vst=anscombe_transform)
        loss = nll_ct(recon, measurements, angles, exposure, l=l)
        # loss = mse(proj_recon, measurements, exposure=exposure)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        it.set_postfix(loss=loss.item())

        # if step % 10 == 0:
        #     print(f"Step {step}: Loss = {loss.item():.4f}")
    return image, losses


def uniform_allocation(num_angles=360, exposure=1e5, device=None):
    """
    Create a uniform allocation vector for the angles.
    """
    angles = linspace(0, 180, num_angles, endpoint=False, device=device)
    allocation = torch.ones(num_angles, device=device) * exposure / num_angles
    return allocation.unsqueeze(-1), angles


def random_allocation(num_angles=360, exposure=1e5, device=None):
    angles = linspace(0, 180, num_angles, endpoint=False, device=device)
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
    # print(f"min/max of projections: {scale_projection.min().item()}/{scale_projection.max().item()}")

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


# def nll_poisson(predictions, measurements, exposure, mixture=False, include_constant_term=False):
#     """
#     Negative log-likelihood for Poisson distribution.
#     predictions: predicted intensity values
#     measurements: observed measurements
#     exposure: exposure time or dose
#     """
#     # Ensure predictions are positive
#     predictions = torch.clamp(predictions, min=1e-6)

#     # Scale predictions by exposure
#     predictions = predictions * exposure

#     if mixture:
#         n_mixture = predictions.shape[-4]
#         exponents = measurements * torch.log(predictions) - predictions - torch.log(torch.tensor(n_mixture, device=predictions.device, dtype=predictions.dtype))
#         nll = - torch.logsumexp(exponents, dim=-4)
#     else:
#         nll = - measurements * torch.log(predictions) + predictions

#     if include_constant_term:
#         # Add constant term if requested
#         nll += torch.lgamma(measurements + 1)

#     return nll.sum(dim=(-1, -2, -3))

# def nnl_gaussian(predictions, measurements, exposure):
#     """
#     Negative log-likelihood for Gaussian distribution.
#     predictions: predicted intensity values
#     measurements: observed measurements
#     exposure: exposure time or dose
#     """
#     # Ensure predictions are positive

#     # Calculate the negative log-likelihood
#     nll = 0.5 * torch.sum((measurements - predictions * exposure)**2 / (predictions * exposure))
#     return nll


# def tomography2d(images, angles, exposure=None):
#     """
#     forward model
#     """
#     # batch_dims = images.shape[:-2]
#     projections = compute_sinogram(images, angles.flatten()) #.reshape((*batch_dims, *angles.shape, -1))
#     if exposure is None:
#         return projections

#     return poisson(projections * exposure)

