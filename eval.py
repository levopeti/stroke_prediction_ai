import json
import os
import torch
import pytorch_lightning as pl
from glob import glob
from torch.utils.data import DataLoader

from training.datasets.time_series_limb_dataset import TimeSeriesLimbDataset
from training.utils.clear_measurements import ClearMeasurements, Limb
from training.utils.lit_model import LitModel
from training.utils.measure_db import MeasureDB

# torch.multiprocessing.set_start_method("spawn", force=True)


def write_eval(base_folder):
    print(base_folder)
    # base_folder = "./models/frequency_test/2024-05-21-16-13_non-inverted_90_arm_cf_25"
    checkpoint_path = glob(os.path.join(base_folder, "*.ckpt"))[0]

    with open(os.path.join(base_folder, "params.json")) as f:
        params = json.load(f)

    params["class_mapping"] = {int(k): v for k, v in params["class_mapping"].items()}
    params["limb"] = Limb.ARM

    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, **params)
    params["val_id_list"] = clear_measurements.get_meas_id_list("validation")
    val_dataset = TimeSeriesLimbDataset("validation",
                                        clear_measurements,
                                        params["limb"],
                                        params,
                                        params["val_batch_size"],
                                        params["training_length_min"],
                                        params["val_sample_per_meas"],
                                        params["steps_per_epoch"],
                                        params["base_frequency"],
                                        params["subsampling_factor"])

    val_loader = DataLoader(val_dataset,
                            batch_size=params["val_batch_size"],
                            shuffle=False,
                            num_workers=0,
                            persistent_workers=False)

    lit_model = LitModel.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    log = trainer.validate(model=lit_model, dataloaders=val_loader, verbose=True)

    """
    [{'val_acc': 0.8237500190734863,
      'val_auc': 0.9020941257476807,
      'val_only_5_acc': 0.8846666812896729,
      'val_stroke_acc': 0.8823750019073486}]
    """

    with open(os.path.join(base_folder, "eval_log_acc_{:.2f}.json".format(log[0]["val_acc"])), "w") as f:
        json.dump(log[0], f, indent=4, sort_keys=True)



if __name__ == "__main__":
    folder_list = ["./models/frequency_test/2024-05-27-07-02_non-inverted_90_arm_cf_12_5",
                   # "./models/frequency_test/2024-05-22-15-42_non-inverted_90_arm__cf_8_3",
                   # "./models/frequency_test/2024-05-23-01-30_non-inverted_90_arm_cf_6_25",
                   # "./models/frequency_test/2024-05-23-11-00_non-inverted_90_arm__cf_5",
                   # "./models/frequency_test/2024-05-23-22-51_non-inverted_90_arm__cf_2_5",
                   # "./models/frequency_test/2024-05-24-10-18_non-inverted_90_arm__cf_1"
                   ]

    for folder_path in folder_list:
        write_eval(folder_path)
