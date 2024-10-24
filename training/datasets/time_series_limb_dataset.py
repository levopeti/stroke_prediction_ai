import numpy as np
import torch

from training.utils.clear_measurements import ClearMeasurements, Limb, MeasType
from training.utils.converting_utils import min_to_ticks
from training.utils.data_preprocessing import get_3d_arrays_from_df, cut_array_to_length, calculate_euclidean_length, \
    butter_high_pass_filter, moving_average, \
    divide_values, down_sampling, change_frequency
from training.datasets.limb_dataset import LimbDataset
from training.utils.feature_extraction import create_multivariate_time_series


class TimeSeriesLimbDataset(LimbDataset):
    """
    train_dataset = TimeSeriesLimbDataset("train",
                                          clear_measurements,
                                          params)
    val_dataset = TimeSeriesLimbDataset("validation",
                                        clear_measurements,
                                        params)
    """
    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 params: dict) -> None:
        super().__init__(data_type, clear_measurements, params)

    def get_input_tensor(self, meas_df, start_idx) -> torch.Tensor:
        wn = 3 if self.params["dst_frequency"] > 6 else self.params["dst_frequency"] / 3
        # TODO: why 1350 ???
        window_size = int(self.length_ticks / 1350)

        array_3d_dict = get_3d_arrays_from_df(meas_df)
        array_3d_dict = change_frequency(array_3d_dict, self.params["base_frequency"],
                                         self.params["dst_frequency"])
        # TODO: original order: butter_high_pass_filter, cut_array_to_length, calculate_euclidean_length
        cut_array_dict = cut_array_to_length(array_3d_dict, self.length_ticks, start_idx=start_idx)
        cut_array_dict = butter_high_pass_filter(cut_array_dict, self.params["dst_frequency"], wn)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, window_size)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        euclidean_length_dict = down_sampling(divided_euclidean_length_dict,
                                              subsampling_factor=self.params["subsampling_factor"])

        multivariate_time_series = create_multivariate_time_series(euclidean_length_dict)
        input_tensor = torch.from_numpy(multivariate_time_series).float()
        return input_tensor

    def get_input_tensor_2(self, meas_df, start_idx) -> torch.Tensor:
        """ 30-60-90 in one array """
        wn = 3 if self.params["dst_frequency"] > 6 else self.params["dst_frequency"] / 3
        window_size = int(self.length_ticks / 1350)

        array_3d_dict = get_3d_arrays_from_df(meas_df)
        array_3d_dict = change_frequency(array_3d_dict, self.params["base_frequency"], self.params["dst_frequency"])
        array_3d_dict_ori = butter_high_pass_filter(array_3d_dict, self.params["dst_frequency"], wn)

        multivariate_time_series_list = list()
        for length_min in [30, 60, 90]:
            length_90 = min_to_ticks(90, self.params["dst_frequency"])
            length = min_to_ticks(length_min, self.params["dst_frequency"])
            current_start_idx = start_idx + (length_90 - length)

            array_3d_dict = array_3d_dict_ori.copy()
            cut_array_dict = cut_array_to_length(array_3d_dict, length, start_idx=current_start_idx)
            euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
            euclidean_length_dict = moving_average(euclidean_length_dict, window_size)
            divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
            current_subsampling_factor = int(self.params["subsampling_factor"] * length / length_90)
            euclidean_length_dict = down_sampling(divided_euclidean_length_dict, current_subsampling_factor)
            multivariate_time_series = create_multivariate_time_series(euclidean_length_dict)
            multivariate_time_series_list.append(multivariate_time_series)

        min_length = min([array.shape[1] for array in multivariate_time_series_list])
        multivariate_time_series_list = [array[:, -min_length:] for array in multivariate_time_series_list]
        multivariate_time_series_list = np.concatenate(multivariate_time_series_list, axis=0)
        input_tensor = torch.from_numpy(multivariate_time_series_list).float()
        return input_tensor


