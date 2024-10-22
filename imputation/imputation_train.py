import json
import os
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pprint import pprint
from glob import glob

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from functools import partial

from imputation.imputation_lit_model import ImputationLitModel
from training.utils.func_utils import save_params
from training.datasets.get_dataset import get_dataset
from imputation.get_imputation_model import imputation_model_dict
from imputation.imputation_loss_and_acc import ImputedLoss, NonImputedLoss
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

    params["seq_length"] = train_dataset.seq_length_ticks
    save_params(params)

    optimizer = partial(torch.optim.Adam, lr=params["learning_rate"], weight_decay=params["wd"], amsgrad=True)
    metric_list = []

    imp_loss = ImputedLoss(params["scale_factor"])
    non_imp_loss = NonImputedLoss(params["scale_factor"])
    loss_list = [imp_loss, non_imp_loss]

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=params["patience"], mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=params["model_base_path"], save_top_k=1, monitor="val_loss",
                                          mode="min")

    if params["model_checkpoint_folder_path"] is not None:
        ckpt_path = sorted(glob(os.path.join(params["model_checkpoint_folder_path"], "*.ckpt")))[-1]
        print(ckpt_path)
        lit_model = ImputationLitModel.load_from_checkpoint(ckpt_path)
    else:
        model = imputation_model_dict[params["model_type"]](**params)
        lit_model = ImputationLitModel(model=model, loss_list=loss_list, metric_list=metric_list, optimizer=optimizer)

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

    with open(os.path.join(params["model_base_path"], "log_loss_{:.2f}.json".format(log_val[0]["val_loss_epoch"])), "w") as f:
        log = dict(log_val[0])
        log["train_loss"], = log_train[0]["val_loss_epoch"]
        json.dump(log, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    """ python -m imputation.imputation_train --name ... """
    # PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir ./models
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    param_dict = get_config_dict()
    train(param_dict)