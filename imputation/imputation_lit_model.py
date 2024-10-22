import copy
import torch

from pytorch_lightning import LightningModule
from torch.nn import Module
from functools import partial
from typing import List
from torchmetrics import Metric


class ImputationLitModel(LightningModule):
    def __init__(self,
                 model: Module,
                 loss_list: List[Module],
                 metric_list: List[Metric],
                 optimizer: partial,
                 save_grad: bool = False):
        super().__init__()
        self.model = model
        self.loss_list = loss_list
        self.train_metric_list, self.val_metric_list = metric_list, copy.deepcopy(metric_list)
        self.optimizer = optimizer
        self.save_hyperparameters(ignore=["model"])
        self.save_grad = save_grad

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        predictions = self.model(x, mask)
        loss_dict = {loss.name: loss(predictions, y, mask) for loss in self.loss_list}
        loss_sum = sum(loss_dict.values())

        log_dict = dict()
        log_dict["train_loss"] = loss_sum

        for name, loss in loss_dict.items():
            log_dict["train_" + name] = loss

        for metric in self.train_metric_list:
            metric.update(predictions, y, mask)
            log_dict["train_" + metric.name] = metric.compute()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        if self.save_grad:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_scalar(name + ".grad_abs_mean", param.grad.abs().mean(), self.global_step)
                    self.logger.experiment.add_scalar(name + ".grad_abs_max", param.grad.abs().max(), self.global_step)
        return sum(loss_dict.values())

    def on_train_epoch_end(self):
        for metric in self.train_metric_list:
            metric.compute()
            metric.reset()

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            torch.cuda.empty_cache()

        x, y, mask = batch
        predictions = self.model(x, mask)

        loss_dict = {loss.name: loss(predictions, y, mask) for loss in self.loss_list}
        loss_sum = sum(loss_dict.values())

        log_dict = dict()
        log_dict["val_loss"] = loss_sum

        for name, loss in loss_dict.items():
            log_dict["val_" + name] = loss

        for metric in self.val_metric_list:
            metric.update(predictions, y, mask)

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        # return sum(loss_dict.values())

    def on_validation_epoch_end(self):
        log_dict = dict()
        for metric in self.val_metric_list:
            log_dict["val_" + metric.name] = metric.compute()
            metric.reset()
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        print("\n\n")