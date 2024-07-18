import random
import torch
from torch.utils.data import Dataset

from training.utils.clear_measurements import ClearMeasurements, Limb, Side, MeasType
from training.utils.converting_utils import min_to_ticks
from training.utils.data_preprocessing import (get_3d_arrays_from_df, cut_array_to_length, calculate_euclidean_length,
                                               butter_high_pass_filter, clip_values, moving_average, divide_values)
from training.utils.feature_extraction import get_features


class LimbDataset(Dataset):
    """
    train_dataset = LimbDataset("train",
                                clear_measurements,
                                params["limb"],
                                params["train_batch_size"],
                                params["length"],
                                params["train_sample_per_meas"],
                                params["steps_per_epoch"])
    val_dataset = LimbDataset("validation",
                              clear_measurements,
                              params["limb"],
                              params["val_batch_size"],
                              params["length"],
                              params["val_sample_per_meas"],
                              params["steps_per_epoch"])
    """

    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 limb: Limb,
                 param_dict: dict,
                 batch_size: int,
                 length: int,
                 sample_per_limb: int,
                 steps_per_epoch: int,
                 frequency: int,
                 indexing_multiplier: int = 1,
                 indexing_mode: int = 1) -> None:
        super().__init__()
        self.data_type = data_type
        self.limb = limb
        self.param_dict = param_dict
        self.batch_size = batch_size
        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.sample_per_limb = sample_per_limb
        self.length_min = length
        self.length = min_to_ticks(self.length_min, param_dict["dst_frequency"])
        self.steps_per_epoch = steps_per_epoch
        self.indexing_multiplier = indexing_multiplier
        self.indexing_mode = indexing_mode

        self.num_of_sides = 2
        if data_type == "train" and self.indexing_mode == 1:
            self.indexing_dict = self.make_indexing()
            print("len of indices: {}".format(len(self.indexing_dict)))

    def __len__(self):
        if self.data_type == "validation":
            return len(self.meas_id_list) * self.sample_per_limb * self.num_of_sides
        elif self.data_type == "train":
            if self.indexing_mode == 0:
                return self.steps_per_epoch * self.batch_size
            elif self.indexing_mode == 1:
                return len(self.indexing_dict)
            else:
                raise NotImplementedError

    def get_meas_df(self, idx):
        if self.data_type == "validation" or self.indexing_mode == 0:
            meas_idx = idx // (self.sample_per_limb * self.num_of_sides) % len(self.meas_id_list)
            meas_id = self.meas_id_list[meas_idx]
            side = Side.LEFT if (idx // self.sample_per_limb) % self.num_of_sides == 0 else Side.RIGHT
        elif self.indexing_mode == 1:
            meas_id, side = self.indexing_dict[idx]
        else:
            raise NotImplementedError
        meas_df = self.clear_measurements.get_measurement(meas_id, side, self.limb)
        return meas_df, meas_id, side

    def get_input_tensor(self, meas_df, start_idx) -> torch.Tensor:
        array_3d_dict = get_3d_arrays_from_df(meas_df)
        filtered_array_3d_dict = butter_high_pass_filter(array_3d_dict)
        cut_array_dict = cut_array_to_length(filtered_array_3d_dict, self.length, start_idx=start_idx)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, 100)
        # clipped_euclidean_length_dict = clip_values(euclidean_length_dict, 100, MeasType.GYR)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        feature_vector = get_features(divided_euclidean_length_dict)
        input_tensor = torch.from_numpy(feature_vector).float()
        return input_tensor

    def __getitem__(self, idx):
        meas_df, meas_id, side = self.get_meas_df(idx)
        subsampling_factor = round(self.param_dict["base_frequency"] / self.param_dict["dst_frequency"])
        start_idx = random.randint(0, len(meas_df[::subsampling_factor]) - (self.length + 1))
        input_tensor = self.get_input_tensor(meas_df, start_idx)
        label = self.clear_measurements.get_limb_class_value(meas_id, side, self.limb)
        return input_tensor, label

    def make_indexing(self) -> dict:
        """ self.clear_measurements.limb_values_dict[type_of_set][limb][class_value] = [(meas_id, side), ...] """
        indexing_dict = dict()
        start_idx = 0

        num_of_classes = len(
            set(self.clear_measurements.class_mapping.values())) if self.clear_measurements.class_mapping is not None else 6
        class_value_idx_dict = {class_value: 0 for class_value in range(num_of_classes)}
        max_number_per_class = max(
            [len(self.clear_measurements.limb_values_dict["train"][self.limb][class_value]) for class_value in
             range(num_of_classes)])
        for _ in range(max_number_per_class * self.indexing_multiplier):
            for class_value in range(num_of_classes):
                class_value_idx = class_value_idx_dict[class_value]
                num_of_meas = len(self.clear_measurements.limb_values_dict["train"][self.limb][class_value])
                meas_id, side = self.clear_measurements.limb_values_dict["train"][self.limb][class_value][
                    class_value_idx % num_of_meas]
                class_value_idx_dict[class_value] += 1
                for idx in range(start_idx, start_idx + self.sample_per_limb):
                    indexing_dict[idx] = (meas_id, side)

                start_idx = start_idx + self.sample_per_limb
        return indexing_dict
