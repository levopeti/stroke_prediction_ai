import copy

from pytorch_lightning import LightningModule
from torch.nn import Module
from functools import partial
from typing import List
from torchmetrics import Metric

from ai_utils.training_utils.pytorch_utils.loss_and_accuracy import get_correct_predictions


class LitModel(LightningModule):
    def __init__(self,
                 model: Module,
                 loss_list: List[Module],
                 metric_list: List[Metric],
                 optimizer: partial):
        super().__init__()
        self.model = model
        self.loss_list = loss_list
        self.train_metric_list, self.val_metric_list = metric_list, copy.deepcopy(metric_list)
        self.optimizer = optimizer
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss_dict = {loss.name: loss(logits, y) for loss in self.loss_list}
        loss_sum = sum(loss_dict.values())

        log_dict = dict()
        log_dict["train_loss"] = loss_sum

        for name, loss in loss_dict.items():
            log_dict["train_" + name] = loss

        for metric in self.train_metric_list:
            if not metric.name == "confm":
                metric.update(logits, y)
                log_dict["train_" + metric.name] = metric.compute()
            else:
                predictions = get_correct_predictions(logits, round_in_regression=True)
                metric.update(predictions, y)

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_scalar(name + ".grad_abs_mean", param.grad.abs().mean(), self.global_step)
                self.logger.experiment.add_scalar(name + ".grad_abs_max", param.grad.abs().max(), self.global_step)
        return sum(loss_dict.values())

    def on_train_epoch_end(self):
        for metric in self.train_metric_list:
            if metric.name == "confm":
                val = metric.compute()
                val = (val * 100).int()
                fig_, ax_ = metric.plot(val=val)
                self.logger.experiment.add_figure("train_conf_mat", fig_, self.current_epoch)
            metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        for metric in self.val_metric_list:
            if not metric.name == "confm":
                metric.update(logits, y)
            else:
                predictions = get_correct_predictions(logits, round_in_regression=True)
                metric.update(predictions, y)
    def on_validation_epoch_end(self):
        log_dict = dict()
        for metric in self.val_metric_list:
            if not metric.name == "confm":
                log_dict["val_" + metric.name] = metric.compute()
            else:
                val = metric.compute()
                val = (val * 100).int()
                fig_, ax_ = metric.plot(val=val)
                self.logger.experiment.add_figure("val_conf_mat", fig_, self.current_epoch)
            metric.reset()
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        print("\n\n")


