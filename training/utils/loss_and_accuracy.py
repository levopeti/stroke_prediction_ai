import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric


def get_correct_predictions(predictions: Tensor, round_in_regression: bool = False):
    """ output shape: [batch_size] """
    if len(predictions.shape) == 2:
        # classification
        # softmax is not necessary before argmax (result would be the same)
        predictions = torch.argmax(predictions, dim=1)
    else:
        # regression
        assert len(predictions.shape) == 1, predictions.shape
        predictions = torch.sigmoid(predictions)
        # scale the predictions from [0, 1] to [-0.5, 5.4]
        # Values equidistant from two integers are rounded towards the
        # nearest even value (zero is treated as even) (-0.5 -> 0; 5.5 -> 6)
        predictions = torch.mul(predictions, 5.9)
        predictions = torch.sub(predictions, 0.5)
        if round_in_regression:
            predictions = torch.round(predictions)
    return predictions


# accuracy
class MSELoss(Module):
    name = "mse_loss"

    def __init__(self):
        super().__init__()

    def forward(self, predictions: Tensor, targets: Tensor):
        predictions = get_correct_predictions(predictions)
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)

        mse_loss = torch.mean((predictions - targets) ** 2)
        return mse_loss


class StrokeLoss(Module):
    name = "stroke_loss"

    def __init__(self, stroke_loss_factor: float, healthy_class: int = 5):
        super().__init__()
        self.stroke_loss_factor = stroke_loss_factor
        self.healthy_class = healthy_class

    def forward(self, predictions: Tensor, targets: Tensor):
        predictions = get_correct_predictions(predictions, round_in_regression=True)
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)

        stroke_loss = torch.mean(
            torch.logical_xor(targets == self.healthy_class, predictions == self.healthy_class).float())
        loss = self.stroke_loss_factor * stroke_loss
        return loss


# loss
class Accuracy(Metric):
    higher_is_better = True
    name = "acc"

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor):
        predictions = get_correct_predictions(predictions, round_in_regression=True)
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)

        self.correct += torch.sum(predictions == targets)
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class StrokeAccuracy(Metric):
    """ Is it stroke or not? """
    higher_is_better = True
    name = "stroke_acc"

    def __init__(self, healthy_class: int = 5):
        super().__init__()
        self.healthy_class = healthy_class
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor):
        predictions = get_correct_predictions(predictions, round_in_regression=True)
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)

        self.correct += torch.sum(
            torch.logical_not(torch.logical_xor(targets == self.healthy_class, predictions == self.healthy_class)))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class OnlyFiveAccuracy(Metric):
    """ How many prediction were correct where the target was 5? """
    higher_is_better = True
    name = "only_5_acc"

    def __init__(self, healthy_class: int = 5):
        super().__init__()
        self.healthy_class = healthy_class
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor):
        predictions = get_correct_predictions(predictions, round_in_regression=True)
        assert predictions.shape == targets.shape, (predictions.shape, targets.shape)

        masked_predictions = predictions[targets == self.healthy_class]
        self.correct += torch.sum(masked_predictions == self.healthy_class)
        self.total += masked_predictions.numel()

    def compute(self):
        return self.correct.float() / self.total
