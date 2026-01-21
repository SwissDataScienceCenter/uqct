import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from uqct.ct import circular_mask



def rmse(
    prediction: torch.Tensor, target: torch.Tensor, circle_mask: bool = True
) -> torch.Tensor:
    """
    Computes the Root Mean Square Error (RMSE) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        circle_mask (bool): If True, applies a circular mask to both images before computing RMSE.
    Returns:
        torch.Tensor: RMSE values for each image in the batch. Output shape: (...)
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    mse = torch.sqrt(torch.mean((target - prediction) ** 2, dim=(-2, -1)))
    return mse


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    circle_mask: bool = True,
) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        data_range (float): Maximum pixel value.
        circle_mask (bool): If True, applies a circular mask to both images before computing PSNR.
    Returns:
        torch.Tensor: PSNR values for each image in the batch. Output shape: (...)
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    mse = torch.mean((target - prediction) ** 2, dim=(-2, -1))
    psnr = 10 * torch.log10(data_range**2 / mse)  # type: ignore
    return psnr  # type: ignore


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    circle_mask: bool = True,
) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        data_range (float): Data range.
        circle_mask (bool): If True, applies a circular mask to both images before computing SSIM.
    Returns:
        torch.Tensor: SSIM values for each image in the batch. Output shape: (...)
    """
    prediction = prediction.unsqueeze(dim=-3)
    target = target.unsqueeze(dim=-3)
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    batch_dims = prediction.size()[:-3]
    img_shape = prediction.shape[-3:]
    target = target.expand_as(prediction).reshape(-1, *img_shape)
    prediction = prediction.view(-1, *img_shape)
    ssim_fct = StructuralSimilarityIndexMeasure(data_range=data_range, reduction=None)
    return ssim_fct(prediction, target).view(*batch_dims)


def get_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    circle_mask: bool = True,
    data_range: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Computes a dictionary of metrics for the given prediction and target.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        max_pixel (float | None): Maximum pixel value.
        circle_mask (bool): Whether to apply a circular mask.
        data_range (float): Data range.
    Returns:
        dict[str, torch.Tensor]: Dictionary of metrics. Each tensor has shape (...,) where ... are the batch dimensions of the input tensors.
    """

    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(-3)
    if target.ndim == 2:
        target = target.unsqueeze(-3)

    return {
        "PSNR": psnr(prediction, target, data_range=data_range, circle_mask=circle_mask),
        "RMSE": rmse(prediction, target, circle_mask=circle_mask),
        "L1": torch.mean(torch.abs(target - prediction), dim=(-2, -1)),
        "SS": ssim(prediction, target, data_range=data_range, circle_mask=circle_mask),
    }


def print_metrics(original, compressed):
    for k, v in get_metrics(original, compressed).items():
        print(k, v)


if __name__ == "__main__":
    test_shapes = [
        (1, 2, 3, 256, 256),
        (1, 1, 256, 256),
        (1, 3, 256, 256),
        (4, 1, 256, 256),
        (4, 3, 256, 256),
        (1, 1, 128, 128),
        (1, 3, 128, 128),
        (4, 1, 128, 128),
        (4, 3, 128, 128),
        (1, 64, 64),
        (3, 64, 64),
    ]
    for shape in test_shapes:
        print(f"Analyzing shape: {shape}")
        pred = torch.rand(shape)
        target = torch.rand(shape)
        metrics = get_metrics(pred, target)
        for k, v in metrics.items():
            print(f"Analyzing metric: {k}")
            print(f"Shape: {v.shape}")
            print(f"Expected shape: {shape[:-2]}")
            assert (
                v.shape == shape[:-2]
            ), f"Shape mismatch for {k}: {v.shape} != {shape[:-2]}"
        print(f"\n")
