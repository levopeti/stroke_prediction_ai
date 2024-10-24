import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric

class ImputedLoss(Module):
    name = "imputed_loss"

    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor):
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)
        masked_pred = predictions * (1 - mask)
        masked_targets = targets * (1 - mask)
        loss = ((masked_pred - masked_targets) ** 2).sum() / (1 - mask).sum()
        return loss * self.scale_factor


class NonImputedLoss(Module):
    name = "non_imputed_loss"

    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor):
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)
        masked_pred = predictions * mask
        masked_targets = targets * mask
        loss = ((masked_pred - masked_targets) ** 2).sum() / mask.sum()
        return loss * self.scale_factor


# class ImputedLoss(Metric):
#     higher_is_better = False
#     name = "imputed_loss"
#
#     def __init__(self):
#         super().__init__()
#         self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, predictions: Tensor, targets: Tensor, mask: Tensor):
#         assert predictions.shape == targets.shape, (predictions.shape, targets.shape)
#         masked_pred = predictions * (1 - mask)
#         masked_targets = targets * (1 - mask)
#         self.loss = self.loss.float()
#         self.loss += ((masked_pred - masked_targets) ** 2).mean().float()
#
#     def compute(self):
#         return self.loss
#
# class NonImputedLoss(Metric):
#     higher_is_better = False
#     name = "non_imputed_loss"
#
#     def __init__(self):
#         super().__init__()
#         self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, predictions: Tensor, targets: Tensor, mask: Tensor):
#         assert predictions.shape == targets.shape, (predictions.shape, targets.shape)
#         masked_pred = predictions * mask
#         masked_targets = targets * mask
#         self.loss = self.loss.float()
#         self.loss += ((masked_pred - masked_targets) ** 2).mean().float()
#
#     def compute(self):
#         return self.loss