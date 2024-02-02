import os
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from datetime import datetime
from pprint import pprint
from glob import glob

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import AUROC, ConfusionMatrix
from torchmetrics.classification import Accuracy
from functools import partial

from training.utils.clear_measurements import ClearMeasurements, Limb
from training.utils.func_utils import save_params
from training.datasets.get_dataset import get_dataset
from nn_models.define_model import model_dict
from training.utils.lit_model import LitModel
from training.utils.loss_and_accuracy import MSELoss, StrokeLoss, StrokeAccuracy, Accuracy, OnlyFiveAccuracy
from training.utils.measure_db import MeasureDB
from utils.arg_parser_and_config import get_config_dict

plt.switch_backend("agg")
torch.multiprocessing.set_start_method("spawn", force=True)


def train(params: dict):
    pprint(params)
    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, **params)
    clear_measurements.print_stat()

    params["train_id_list"] = clear_measurements.get_meas_id_list("train")
    params["val_id_list"] = clear_measurements.get_meas_id_list("validation")
    save_params(params)

    train_dataset, val_dataset = get_dataset(params, clear_measurements)
    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=False,
                              num_workers=params["num_workers"], persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=params["val_batch_size"], shuffle=False,
                            num_workers=params["num_workers"], persistent_workers=True)

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

    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=params["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(dirpath=params["model_base_path"], save_top_k=1, monitor="val_acc",
                                          mode="max")

    if params["model_checkpoint_folder_path"] is not None:
        ckpt_path = sorted(glob(os.path.join(params["model_checkpoint_folder_path"], "*.ckpt")))[-1]
        print(ckpt_path)
        lit_model = LitModel.load_from_checkpoint(ckpt_path)
    else:
        ckpt_path = None
        model = model_dict[params["model_type"]](**params)
        lit_model = LitModel(model=model, loss_list=loss_list, metric_list=metric_list, optimizer=optimizer)

    # inference_mode="predict"
    trainer = pl.Trainer(max_epochs=params["num_epoch"],
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=TensorBoardLogger(params["model_base_path"], default_hp_metric=False),
                         log_every_n_steps=10,
                         accelerator=params["device"],
                         devices=1)
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir ./models
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # param_dict = {
    #     # data info
    #     "accdb_path": "./data/WUS-v4measure20240116.accdb",
    #     "ucanaccess_path": "./ucanaccess/",
    #     "folder_path": "./data/clear_and_synchronized/",
    #     "clear_json_path": "./data/clear_train_val_ids.json",
    #     "model_base_path": "./models/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M')),
    #     "model_checkpoint_folder_path": None,  # None
    #
    #     # measurement info
    #     "frequency": 25,  # HZ
    #     "training_length_min": 90,
    #     "step_size_min": 5,
    #     "limb": Limb.ARM,
    #
    #     # model info
    #     "model_type": "inception_time",  # mlp, inception_time
    #     "input_shape": 2,  # 18 - features, 2 - acc, gyr
    #     "output_shape": 3,  # depends on the class mapping
    #     "layer_sizes": [1024, 512, 256],  # only for mlp
    #
    #     # dataset
    #     "invert_side": False,
    #     "class_mapping": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2},  # None, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    #     "train_sample_per_meas": 10,
    #     "val_sample_per_meas": 500,  # 500
    #     "indexing_multiplier": 4,
    #     "cache_size": 1,
    #     "steps_per_epoch": 100,  # 100, only if indexing mode == 0
    #
    #     # dataloader
    #     "train_batch_size": 100,  # 100
    #     "val_batch_size": 100,
    #     "num_workers": 5,
    #
    #     # training
    #     "learning_rate": 0.0001,
    #     "wd": 0.001,
    #     "num_epoch": 1000,
    #     "stroke_loss_factor": 0.5,  # for stroke loss function
    #     "patience": 20,  # early stopping callback
    #     "device": "cuda",  # cpu, cuda
    # }
    param_dict = get_config_dict()
    train(param_dict)
