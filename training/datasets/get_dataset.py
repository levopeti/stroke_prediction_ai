from typing import Tuple
from torch.utils.data import Dataset

from training.utils.clear_measurements import ClearMeasurements
from training.datasets.limb_dataset import LimbDataset
from training.datasets.time_series_limb_dataset import TimeSeriesLimbDataset


def get_dataset(params: dict, clear_measurements: ClearMeasurements) -> Tuple[Dataset, Dataset]:
    if params["model_type"] == "mlp":
        raise DeprecationWarning
    elif params["model_type"] == "inception_time":
        train_dataset = TimeSeriesLimbDataset("train", clear_measurements, params)
        val_dataset = TimeSeriesLimbDataset("validation", clear_measurements, params)
    else:
        raise NotImplementedError
    return train_dataset, val_dataset
