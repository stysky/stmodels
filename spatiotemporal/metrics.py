from __future__ import annotations

import numpy as np
import torch


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
