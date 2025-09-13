from skimage.metrics import structural_similarity as SS
import torch


def apply_circle_mask(img):
    W, H = img.shape[-2:]
    cp = torch.cartesian_prod(torch.arange(W, device=img.device), torch.arange(H, device=img.device))
    circle_mask = (cp[:, 0] - W / 2) ** 2 + (cp[:, 1] - W / 2) ** 2 <= (W / 2) ** 2
    return img * circle_mask.reshape(img.shape[-2:])

def PSNR(original, compressed, max_pixel=None):
    if max_pixel is None:
        max_pixel = torch.max(original)
    mse = torch.mean((original - compressed) ** 2)
    psnr = 10 * torch.log10(max_pixel ** 2 / mse)
    return psnr

def constrained_PSNR(target, pred):
    """ Computes PSNR for images with a circular mask applied.
    Assumes target and pred are 3D tensors with shape (B, H, W).
    The circular mask is applied to the central region of the images.
    """
    target = apply_circle_mask(target[..., 20:-20, 20:-20])
    pred = apply_circle_mask(pred[..., 20:-20, 20:-20])
    return PSNR(target, pred)

def RMSE(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    return torch.sqrt(mse)

def ZeroOne(original, compressed):
    """ average zero-one loss, clips predictions to {0, 1}"""
    return torch.sum(torch.abs(original - torch.round(compressed))) / original.numel()


b_PSNR = torch.vmap(PSNR)
b_RMSE = torch.vmap(RMSE)
b_ZeroOne = torch.vmap(ZeroOne)

def get_metrics(original, compressed, normalize_range=True, constrained=False):
    
    original = original.squeeze().detach().to(compressed.device)
    compressed = compressed.squeeze().detach()
    
    if normalize_range:
        max_value = torch.max(original)
        original = original / max_value
        compressed = compressed / max_value
    
    psnr_function = PSNR if not constrained else constrained_PSNR

    return {
        "PSNR": psnr_function(original, compressed).item(),
        "RMSE": RMSE(original, compressed).item(),
        "L1": torch.mean(torch.abs(original - compressed)).item(),
        "ZeroOne": ZeroOne(original, compressed).item(),
        "SS": SS(original.cpu().numpy(), compressed.cpu().numpy(), full=True, data_range=1)[0].item()
    }

def print_metrics(original, compressed):
    for k, v in get_metrics(original, compressed).items():
        print(k, v)