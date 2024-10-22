from typing import Tuple

from imputation.imputation_dataset import ImputationDataset
from training.utils.clear_measurements import ClearMeasurements
from training.datasets.limb_dataset import LimbDataset
from training.datasets.time_series_limb_dataset import TimeSeriesLimbDataset
from training.utils.measure_db import MeasureDB


def get_dataset(params: dict) -> Tuple[LimbDataset, LimbDataset]:
    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, **params)
    clear_measurements.print_stat()

    params["train_id_list"] = clear_measurements.get_meas_id_list("train")
    params["val_id_list"] = clear_measurements.get_meas_id_list("validation")

    if params["model_type"] == "mlp":
        raise DeprecationWarning
    elif params["model_type"] in ["inception_time", "basic_transformer"]:
        train_dataset = TimeSeriesLimbDataset("train", clear_measurements, params)
        val_dataset = TimeSeriesLimbDataset("validation", clear_measurements, params)
    elif params["model_type"] in ["unet"]:
        train_dataset = ImputationDataset("train", clear_measurements, params)
        val_dataset = ImputationDataset("validation", clear_measurements, params)
    else:
        raise NotImplementedError
    return train_dataset, val_dataset
