import random
import torch
from torch.utils.data import Dataset

from training.utils.clear_measurements import ClearMeasurements, Limb, Side, MeasType
from training.utils.converting_utils import min_to_ticks


class LimbDataset(Dataset):
    """
    train_dataset = LimbDataset("train", clear_measurements, params)
    val_dataset = LimbDataset("validation", clear_measurements, params)
    """

    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 params: dict) -> None:
        super().__init__()
        self.data_type = data_type
        self.params = params
        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.length_ticks = min_to_ticks(params["training_length_min"], params["dst_frequency"])
        # self.sample_per_limb = params["train_sample_per_meas"]
        # self.length_min = params["training_length_min"]
        # self.steps_per_epoch = params["steps_per_epoch"]
        # self.indexing_multiplier = params["indexing_multiplier"]
        # self.indexing_mode = params["indexing_mode"]

        self.num_of_sides = 2
        if data_type == "train" and self.params["indexing_mode"] == 1:
            self.indexing_dict = self.make_indexing()
            print("len of indices: {}".format(len(self.indexing_dict)))

    def __len__(self):
        if self.data_type == "validation":
            return len(self.meas_id_list) * self.params["train_sample_per_meas"] * self.num_of_sides
        elif self.data_type == "train":
            if self.params["indexing_mode"] == 0:
                return self.params["steps_per_epoch"] * self.params["train_batch_size"]
            elif self.params["indexing_mode"] == 1:
                return len(self.indexing_dict)
            else:
                raise NotImplementedError

    def get_meas_df(self, idx):
        if self.data_type == "validation" or self.params["indexing_mode"] == 0:
            meas_idx = idx // (self.params["train_sample_per_meas"] * self.num_of_sides) % len(self.meas_id_list)
            meas_id = self.meas_id_list[meas_idx]
            side = Side.LEFT if (idx // self.params["train_sample_per_meas"]) % self.num_of_sides == 0 else Side.RIGHT
        elif self.params["indexing_mode"] == 1:
            meas_id, side = self.indexing_dict[idx]
        else:
            raise NotImplementedError
        meas_df = self.clear_measurements.get_measurement(meas_id, side, self.params["limb"])
        return meas_df, meas_id, side

    def get_input_tensor(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, idx):
        meas_df, meas_id, side = self.get_meas_df(idx)
        subsampling_factor = round(self.params["base_frequency"] / self.params["dst_frequency"])
        start_idx = random.randint(0, len(meas_df[::subsampling_factor]) - (self.length_ticks + 1))
        input_tensor = self.get_input_tensor(meas_df, start_idx)
        label = self.clear_measurements.get_limb_class_value(meas_id, side, self.params["limb"])
        return input_tensor, label

    def make_indexing(self) -> dict:
        """ self.clear_measurements.limb_values_dict[type_of_set][limb][class_value] = [(meas_id, side), ...] """
        indexing_dict = dict()
        start_idx = 0

        num_of_classes = len(
            set(self.clear_measurements.class_mapping.values())) if self.clear_measurements.class_mapping is not None else 6
        class_value_idx_dict = {class_value: 0 for class_value in range(num_of_classes)}
        max_number_per_class = max(
            [len(self.clear_measurements.limb_values_dict["train"][self.params["limb"]][class_value]) for class_value in
             range(num_of_classes)])
        for _ in range(max_number_per_class * self.params["indexing_multiplier"]):
            for class_value in range(num_of_classes):
                class_value_idx = class_value_idx_dict[class_value]
                num_of_meas = len(self.clear_measurements.limb_values_dict["train"][self.params["limb"]][class_value])
                meas_id, side = self.clear_measurements.limb_values_dict["train"][self.params["limb"]][class_value][
                    class_value_idx % num_of_meas]
                class_value_idx_dict[class_value] += 1
                for idx in range(start_idx, start_idx + self.params["train_sample_per_meas"]):
                    indexing_dict[idx] = (meas_id, side)

                start_idx = start_idx + self.params["train_sample_per_meas"]
        return indexing_dict
