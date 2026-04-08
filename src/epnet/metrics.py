from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_f
from numpy.typing import NDArray
from torch import Tensor


def _to_y_channel(array: NDArray[np.float32]) -> NDArray[np.float32]:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB image")
    y = (
        16.0
        + (
            65.738 * array[:, :, 0]
            + 129.057 * array[:, :, 1]
            + 25.064 * array[:, :, 2]
        )
        / 256.0
    )
    return y


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def _ssim_per_channel(pred: Tensor, target: Tensor, max_val: float = 255.0) -> Tensor:
    kernel = _gaussian_kernel().to(pred.device, dtype=pred.dtype)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu_x = torch_f.conv2d(pred, kernel, padding=5)
    mu_y = torch_f.conv2d(target, kernel, padding=5)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = torch_f.conv2d(pred * pred, kernel, padding=5) - mu_x2
    sigma_y = torch_f.conv2d(target * target, kernel, padding=5) - mu_y2
    sigma_xy = torch_f.conv2d(pred * target, kernel, padding=5) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean()


@dataclass(frozen=True)
class MetricResult:
    psnr: float
    ssim: float


def calculate_psnr(prediction: NDArray[np.float32], target: NDArray[np.float32]) -> float:
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(prediction: NDArray[np.float32], target: NDArray[np.float32]) -> float:
    pred_tensor = torch.from_numpy(prediction).float().unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    return float(_ssim_per_channel(pred_tensor, target_tensor).item())


def evaluate_prediction(
    prediction: Tensor,
    target: Tensor,
    shave: int = 4,
    use_y_channel: bool = True,
) -> MetricResult:
    pred = prediction.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    tgt = target.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0

    if shave > 0:
        pred = pred[shave:-shave, shave:-shave]
        tgt = tgt[shave:-shave, shave:-shave]

    if use_y_channel:
        pred_y = _to_y_channel(pred)
        tgt_y = _to_y_channel(tgt)
        return MetricResult(
            psnr=calculate_psnr(pred_y, tgt_y),
            ssim=calculate_ssim(pred_y, tgt_y),
        )

    return MetricResult(
        psnr=calculate_psnr(pred, tgt),
        ssim=calculate_ssim(pred[:, :, 0], tgt[:, :, 0]),
    )
