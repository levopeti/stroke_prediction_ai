import torch
from torch.utils.data import Dataset

from ai_utils.training_utils.func_utils import get_input_from_df
from ai_utils.training_utils.clear_measurements import ClearMeasurements

class ClearDataset(Dataset):
    """
    train_dataset = ClearDataset("train",
                                 clear_measurements,
                                 params["train_batch_size"],
                                 params["length"],
                                 params["train_sample_per_meas"],
                                 params["steps_per_epoch"])
    val_dataset = ClearDataset("validation",
                               clear_measurements,
                               params["val_batch_size"],
                               params["length"],
                               params["val_sample_per_meas"],
                               params["steps_per_epoch"])
    """
    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 batch_size: int,
                 length: int,
                 sample_per_meas: int,
                 steps_per_epoch: int) -> None:
        self.data_type = data_type
        self.batch_size = batch_size
        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.sample_per_meas = sample_per_meas
        self.length = length
        self.steps_per_epoch = steps_per_epoch

        # self.to_tensor = ToTensor()

    def __len__(self):
        if self.data_type == "validation":
            return len(self.meas_id_list) * self.sample_per_meas
        elif self.data_type == "train":
            return self.steps_per_epoch * self.batch_size

    def __getitem__(self, idx):
        meas_idx = idx // self.sample_per_meas % len(self.meas_id_list)
        meas_id = self.meas_id_list[meas_idx]
        meas_df = self.clear_measurements.get_measurement(meas_id)

        class_value_dict = self.clear_measurements.get_class_value_dict(meas_id=meas_id)
        input_array = get_input_from_df(meas_df, self.length, class_value_dict)
        input_tensor = torch.from_numpy(input_array).float()

        label = min(class_value_dict.values())
        return input_tensor, label