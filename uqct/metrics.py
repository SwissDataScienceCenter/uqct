from skimage.metrics import structural_similarity as SS
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from uqct.ct import circular_mask


def psnr(prediction, target, data_range=1.0, circle_mask=True):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        data_range (float): The data range of the input images (i.e., the difference between the maximum
                            possible value and the minimum possible value).
        circle_mask (bool): If True, applies a circular mask to both images before computing PSNR.
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    mse = torch.mean((target - prediction) ** 2, dim=(-2, -1))
    psnr = 10 * torch.log10(data_range ** 2 / mse)
    return psnr


def rmse(prediction, target, circle_mask=True):
    """
    Computes the Root Mean Square Error (RMSE) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        circle_mask (bool): If True, applies a circular mask to both images before computing RMSE.
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    mse = torch.sqrt(torch.mean((target - prediction) ** 2, dim=(-2, -1)))
    return mse


def l1_loss(prediction, target, circle_mask=True):
    """
    Computes the L1 loss (Mean Absolute Error) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        circle_mask (bool): If True, applies a circular mask to both images before computing L1 loss.
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    l1 = torch.mean(torch.abs(target - prediction), dim=(-2, -1))
    return l1


def ssim(prediction, target, data_range=1.0, circle_mask=True):
    """
    Computes the Structural Similarity Index Measure (SSIM) between two images.
    Args:
        prediction (torch.Tensor): (..., H, W) Predicted image tensor.
        target (torch.Tensor): (..., H, W) Target image tensor.
        data_range (float): The data range of the input images (i.e., the difference between the maximum
                            possible value and the minimum possible value).
        circle_mask (bool): If True, applies a circular mask to both images before computing SSIM.
    """
    if circle_mask:
        mask = circular_mask(target.shape[-1], device=target.device)
        target = target * mask
        prediction = prediction * mask
    batch_dims = prediction.size()[:-2]
    img_shape = prediction.shape[-2:]
    target = target.expand_as(prediction).reshape(-1, 1, *img_shape)
    prediction = prediction.view(-1, 1, *img_shape)
    ssim_fct = StructuralSimilarityIndexMeasure(data_range=data_range, reduction=None)
    return ssim_fct(prediction, target).view(*batch_dims)


def get_metrics(prediction, target, data_range=1.0, circle_mask=True):
    target = target.squeeze().detach().to(prediction.device)
    prediction = prediction.squeeze().detach()
    
    return {
        'PSNR': psnr(prediction.unsqueeze(0), target.unsqueeze(0), data_range=data_range, circle_mask=circle_mask).item(),
        'RMSE': rmse(prediction.unsqueeze(0), target.unsqueeze(0), circle_mask=circle_mask).item(),
        "L1": l1_loss(prediction, target, circle_mask=circle_mask).item(),
        'SS': ssim(prediction.unsqueeze(0), target.unsqueeze(0), data_range=data_range, circle_mask=circle_mask).item(),
    }


def print_metrics(prediction, target):
    for k, v in get_metrics(prediction, target).items():
        print(k, v)