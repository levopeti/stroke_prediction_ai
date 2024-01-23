from typing import Tuple

from torch.utils.data import Dataset

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.pytorch_utils.datasets.limb_dataset import LimbDataset
from ai_utils.training_utils.pytorch_utils.datasets.time_series_limb_dataset import TimeSeriesLimbDataset


def get_dataset(params: dict, clear_measurements: ClearMeasurements) -> Tuple[Dataset, Dataset]:
    if params["model_type"] == "mlp":
        train_dataset = LimbDataset("train",
                                    clear_measurements,
                                    params["limb"],
                                    params["train_batch_size"],
                                    params["length"],
                                    params["train_sample_per_meas"],
                                    params["steps_per_epoch"],
                                    params["indexing_multiplier"])
        val_dataset = LimbDataset("validation",
                                  clear_measurements,
                                  params["limb"],
                                  params["val_batch_size"],
                                  params["length"],
                                  params["val_sample_per_meas"],
                                  params["steps_per_epoch"])
    elif params["model_type"] == "inception_time":
      train_dataset = TimeSeriesLimbDataset("train",
                                            clear_measurements,
                                            params["limb"],
                                            params["train_batch_size"],
                                            params["length"],
                                            params["train_sample_per_meas"],
                                            params["steps_per_epoch"],
                                            params["indexing_multiplier"])
      val_dataset = TimeSeriesLimbDataset("validation",
                                          clear_measurements,
                                          params["limb"],
                                          params["val_batch_size"],
                                          params["length"],
                                          params["val_sample_per_meas"],
                                          params["steps_per_epoch"])
    else:
        raise NotImplementedError
    return train_dataset, val_dataset