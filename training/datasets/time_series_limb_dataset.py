import torch

from training.utils.clear_measurements import ClearMeasurements, Limb, MeasType
from training.utils.data_preprocessing import get_3d_arrays_from_df, cut_array_to_length, calculate_euclidean_length, butter_high_pass_filter, moving_average, \
    divide_values, down_sampling
from training.datasets.limb_dataset import LimbDataset
from training.utils.feature_extraction import create_multivariate_time_series


class TimeSeriesLimbDataset(LimbDataset):
    """
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
    """

    def __init__(self,
                 data_type: str,  # train or validation
                 clear_measurements: ClearMeasurements,
                 limb: Limb,
                 batch_size: int,
                 length: int,
                 sample_per_limb: int,
                 steps_per_epoch: int,
                 frequency: int,
                 indexing_multiplier: int = 1,
                 indexing_mode: int = 1) -> None:
        super().__init__(data_type,
                         clear_measurements,
                         limb,
                         batch_size,
                         length,
                         sample_per_limb,
                         steps_per_epoch,
                         frequency,
                         indexing_multiplier,
                         indexing_mode)

    def get_input_tensor(self, meas_df, start_idx) -> torch.Tensor:
        array_3d_dict = get_3d_arrays_from_df(meas_df)
        filtered_array_3d_dict = butter_high_pass_filter(array_3d_dict)
        cut_array_dict = cut_array_to_length(filtered_array_3d_dict, self.length, start_idx=start_idx)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, 100)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        euclidean_length_dict = down_sampling(divided_euclidean_length_dict, subsampling_factor=50)

        multivariate_time_series = create_multivariate_time_series(euclidean_length_dict)
        input_tensor = torch.from_numpy(multivariate_time_series).float()
        return input_tensor

