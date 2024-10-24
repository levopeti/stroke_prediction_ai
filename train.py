import json
import os
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pprint import pprint
from glob import glob

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import AUROC, ConfusionMatrix
from torchmetrics.classification import Accuracy
from functools import partial

from training.utils.func_utils import save_params
from training.datasets.get_dataset import get_dataset
from nn_models.define_model import model_dict
from training.utils.lit_model import LitModel
from training.utils.loss_and_accuracy import MSELoss, StrokeLoss, StrokeAccuracy, Accuracy, OnlyFiveAccuracy
from utils.arg_parser_and_config import get_config_dict

plt.switch_backend("agg")
torch.multiprocessing.set_start_method("spawn", force=True)


def train(params: dict):
    pprint(params)

    train_dataset, val_dataset = get_dataset(params)
    train_loader = DataLoader(train_dataset,
                              batch_size=params["train_batch_size"],
                              shuffle=False,
                              num_workers=params["num_workers"],
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=params["val_batch_size"],
                            shuffle=False,
                            num_workers=params["num_workers"],
                            persistent_workers=True)

    params["seq_length"] = train_dataset.seq_length
    save_params(params)

    optimizer = partial(torch.optim.Adam, lr=params["learning_rate"], weight_decay=params["wd"], amsgrad=True)
    metric_list = [Accuracy().to(params["device"]), StrokeAccuracy(params["output_shape"] - 1).to(params["device"]),
                   OnlyFiveAccuracy(params["output_shape"] - 1).to(params["device"])]
    if params["output_shape"] > 1:
        # classification problem
        # num_of_samples = [5, 5, 6, 2, 20, 100]
        # normed_weights = [1 - (x / sum(num_of_samples)) for x in num_of_samples]
        # normed_weights = torch.FloatTensor(normed_weights)
        # print("weights: {}".format(normed_weights))
        xe = CrossEntropyLoss()  # weight=normed_weights
        xe.name = "xe_loss"
        loss_list = [xe, StrokeLoss(params["stroke_loss_factor"], params["output_shape"] - 1)]
        auroc = AUROC(task="multiclass", num_classes=params["output_shape"]).to(params["device"])
        auroc.name = "auc"
        metric_list.append(auroc)
        confmat = ConfusionMatrix(task="multiclass", num_classes=params["output_shape"], normalize="true").to(
            params["device"])
        confmat.name = "confm"
        metric_list.append(confmat)
    else:
        # regression problem
        assert params["output_shape"] == 1, params["output_shape"]
        loss_list = [MSELoss()]  # , StrokeLoss(params["stroke_loss_factor"])]

    early_stop_callback = EarlyStopping(monitor="train_acc", min_delta=0.00, patience=params["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(dirpath=params["model_base_path"], save_top_k=1, monitor="val_acc",
                                          mode="max")

    if params["model_checkpoint_folder_path"] is not None:
        ckpt_path = sorted(glob(os.path.join(params["model_checkpoint_folder_path"], "*.ckpt")))[-1]
        print(ckpt_path)
        lit_model = LitModel.load_from_checkpoint(ckpt_path)
    else:
        model = model_dict[params["model_type"]](**params)
        lit_model = LitModel(model=model, loss_list=loss_list, metric_list=metric_list, optimizer=optimizer)

    # inference_mode="predict"
    trainer = pl.Trainer(max_epochs=params["num_epoch"],
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=TensorBoardLogger(params["model_base_path"], default_hp_metric=False),
                         log_every_n_steps=10,
                         accelerator=params["device"],
                         devices=1)
    trainer.fit(lit_model, train_loader, val_loader)
    log_train = trainer.validate(model=lit_model, dataloaders=train_loader, ckpt_path="best", verbose=True)
    log_val = trainer.validate(model=lit_model, dataloaders=val_loader, ckpt_path="best", verbose=True)

    with open(os.path.join(params["model_base_path"], "log_acc_{:.2f}.json".format(log_val[0]["val_acc"])), "w") as f:
        log = dict(log_val[0])
        for key in log_train[0]:
            log[key.replace("val", "train")] = log_train[0][key]
        json.dump(log, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    # PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir ./models
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    param_dict = get_config_dict()
    train(param_dict)
