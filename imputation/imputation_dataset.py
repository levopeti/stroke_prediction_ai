from typing import Tuple, Dict

import numpy as np
import torch

from training.utils.clear_measurements import ClearMeasurements, Limb, MeasType, Key
from training.utils.converting_utils import min_to_ticks
from training.utils.data_preprocessing import get_3d_arrays_from_df, cut_array_to_length, calculate_euclidean_length, \
    butter_high_pass_filter, moving_average, \
    divide_values, down_sampling, change_frequency
from training.datasets.limb_dataset import LimbDataset
from training.utils.feature_extraction import create_multivariate_time_series


class ImputationDataset(LimbDataset):
    """
    train_dataset = ImputationDataset("train", clear_measurements, params)
    val_dataset = ImputationDataset("validation", clear_measurements, params)
    """

    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 params: dict) -> None:
        super().__init__(data_type, clear_measurements, params)

    def __getitem__(self, idx):
        meas_df, meas_id, side = self.get_meas_df(idx)
        subsampling_factor = round(self.params["base_frequency"] / self.params["dst_frequency"])
        start_idx = np.random.randint(0, len(meas_df[::subsampling_factor]) - (self.length_ticks + 1))
        masked_input_tensor, gt_input_tensor, mask = self.get_input_tensor(meas_df, start_idx)
        return masked_input_tensor, gt_input_tensor, mask

    def get_input_tensor(self, meas_df, start_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wn = 3 if self.params["dst_frequency"] > 6 else self.params["dst_frequency"] / 3
        # TODO: why 1350 ??? -> half of 2700
        window_size = int(self.length_ticks / 1350)

        array_3d_dict = get_3d_arrays_from_df(meas_df)
        array_3d_dict = change_frequency(array_3d_dict, self.params["base_frequency"], self.params["dst_frequency"])
        cut_array_dict = cut_array_to_length(array_3d_dict, self.length_ticks, start_idx=start_idx)
        mask = self.generate_mask()
        masked_array_dict = self.apply_mask_on_arrays(cut_array_dict, mask)

        gt_input_tensor = self.calculate_input_tensor(cut_array_dict, window_size, wn)
        masked_input_tensor = self.calculate_input_tensor(masked_array_dict, window_size, wn)

        mask = mask[::self.params["subsampling_factor"]].T  # [l, 1] -> [1, l]
        mask = torch.from_numpy(mask).float()
        return masked_input_tensor, gt_input_tensor, mask

    def calculate_input_tensor(self, cut_array_dict, window_size, wn):
        cut_array_dict = butter_high_pass_filter(cut_array_dict, self.params["dst_frequency"], wn)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, window_size)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        euclidean_length_dict = down_sampling(divided_euclidean_length_dict,
                                              subsampling_factor=self.params["subsampling_factor"])

        # shape: [c, l]
        multivariate_time_series = create_multivariate_time_series(euclidean_length_dict)
        input_tensor = torch.from_numpy(multivariate_time_series).float()
        return input_tensor

    def generate_mask(self):
        """ Generate one big gap and more smaller ones. """
        def add_random_gap(_mask, gap_size):
            start_idx = np.random.randint(0, self.length_ticks - (gap_size + 1))
            _mask[start_idx: start_idx + gap_size] = 0
            return _mask

        mask = np.ones(self.length_ticks, dtype=int)
        max_big_gap_size_ticks = min_to_ticks(self.params["max_big_gap_size_min"], self.params["dst_frequency"])
        max_small_gap_size_ticks = min_to_ticks(self.params["max_small_gap_size_min"], self.params["dst_frequency"])

        big_gap_size = np.random.randint(max_big_gap_size_ticks // 2, max_big_gap_size_ticks)
        mask = add_random_gap(mask, big_gap_size)

        for _ in range(self.params["num_of_small_gaps"]):
            small_gap_size = np.random.randint(10, max_small_gap_size_ticks)
            mask = add_random_gap(mask, small_gap_size)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    @staticmethod
    def apply_mask_on_arrays(meas_3d_arrays: Dict[Key, np.ndarray],
                             mask: np.ndarray) -> Dict[Key, np.ndarray]:
        masked_array_dict = dict()
        for key, array_3d in meas_3d_arrays.items():
            masked_array_dict[key] = array_3d * mask
        return masked_array_dict