import os
import random

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from training.datasets.limb_dataset import LimbDataset
from training.datasets.time_series_limb_dataset import TimeSeriesLimbDataset
from training.utils.clear_measurements import MeasType, ClearMeasurements, Limb, Side
from training.utils.converting_utils import min_to_ticks
from training.utils.data_preprocessing import get_3d_arrays_from_df, butter_high_pass_filter, cut_array_to_length, \
    calculate_euclidean_length, moving_average, divide_values, change_frequency, down_sampling
from training.utils.feature_extraction import create_multivariate_time_series


class PlotDataset(TimeSeriesLimbDataset):
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
                 subsampling_factor: int,
                 indexing_multiplier: int = 1,
                 indexing_mode: int = 1) -> None:
        super().__init__(data_type,
                         clear_measurements,
                         limb,
                         param_dict,
                         batch_size,
                         length,
                         sample_per_limb,
                         steps_per_epoch,
                         frequency,
                         subsampling_factor,
                         indexing_multiplier,
                         indexing_mode)
        self.image_idx = 0

    def get_input_tensor(self, meas_df, start_idx) -> None:
        plt.rcParams['figure.figsize'] = [50, 100]
        num_of_rows = 16  # if self.param_dict["dst_frequency"] > 6 else 14
        fig, axes = plt.subplots(num_of_rows, 1)
        ax_idx = 0
        vlines = [start_idx, start_idx + self.length]
        wn = 3 if self.param_dict["dst_frequency"] > 6 else self.param_dict["dst_frequency"] / 3
        window_size = int(self.length / 1350)

        array_3d_dict = get_3d_arrays_from_df(meas_df)
        ax_idx = self.plot_3d_arrays(array_3d_dict, axes, ax_idx, num_of_rows, "full ory", dim_size=3)
        array_3d_dict = change_frequency(array_3d_dict, self.param_dict["base_frequency"],
                                         self.param_dict["dst_frequency"])
        ax_idx = self.plot_3d_arrays(array_3d_dict, axes, ax_idx, num_of_rows, "new freq", vlines, dim_size=3)
        # if self.param_dict["dst_frequency"] > 6:
        array_3d_dict = butter_high_pass_filter(array_3d_dict, self.param_dict["dst_frequency"], wn)
        ax_idx = self.plot_3d_arrays(array_3d_dict, axes, ax_idx, num_of_rows, "butter pass", vlines, dim_size=3)

        cut_array_dict = cut_array_to_length(array_3d_dict, self.length, start_idx=start_idx)
        ax_idx = self.plot_3d_arrays(cut_array_dict, axes, ax_idx, num_of_rows, "after cut", dim_size=3)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        ax_idx = self.plot_3d_arrays(euclidean_length_dict, axes, ax_idx, num_of_rows, "euclidean")
        euclidean_length_dict = moving_average(euclidean_length_dict, window_size)
        ax_idx = self.plot_3d_arrays(euclidean_length_dict, axes, ax_idx, num_of_rows, "moving avg")
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        ax_idx = self.plot_3d_arrays(divided_euclidean_length_dict, axes, ax_idx, num_of_rows, "divide")
        euclidean_length_dict = down_sampling(divided_euclidean_length_dict, subsampling_factor=self.subsampling_factor)
        self.plot_3d_arrays(euclidean_length_dict, axes, ax_idx, num_of_rows, "down sampling")

        plt.tight_layout()

        output_folder = "./plots_2"
        plt.savefig(os.path.join(output_folder, "{}_{}_ws_{}_wn_{:.2f}.png".format(self.image_idx,
                                                                                   self.param_dict["dst_frequency"],
                                                                                   window_size,
                                                                                   wn)))
        plt.clf()
        plt.close("all")
        self.image_idx += 1

    def __getitem__(self, idx):
        meas_df, meas_id, side = self.get_meas_df(idx)
        subsampling_factor = round(self.param_dict["base_frequency"] / self.param_dict["dst_frequency"])
        start_idx = random.randint(0, len(meas_df[::subsampling_factor]) - (self.length + 1))
        ori_start_idx = start_idx * subsampling_factor
        input_tensor = self.get_input_tensor_2(meas_df, start_idx, ori_start_idx)
        label = self.clear_measurements.get_limb_class_value(meas_id, side, self.limb)
        return input_tensor, label

    def get_input_tensor_2(self, meas_df, start_idx, ori_start_idx) -> None:
        plt.rcParams['figure.figsize'] = [100, 50]
        num_of_rows = 2  # if self.param_dict["dst_frequency"] > 6 else 14
        fig, axes = plt.subplots(num_of_rows, 1)
        ax_idx = 0
        vlines = [start_idx, start_idx + self.length]
        wn = 3 if self.param_dict["dst_frequency"] > 6 else self.param_dict["dst_frequency"] / 3
        window_size = int(self.length / 1350)
        ori_length = min_to_ticks(self.length_min, self.param_dict["base_frequency"])

        array_3d_dict = get_3d_arrays_from_df(meas_df)
        ori_array_3d_dict = array_3d_dict.copy()
        array_3d_dict = change_frequency(array_3d_dict, self.param_dict["base_frequency"],
                                         self.param_dict["dst_frequency"])
        array_3d_dict = butter_high_pass_filter(array_3d_dict, self.param_dict["dst_frequency"], wn)

        cut_array_dict = cut_array_to_length(array_3d_dict, self.length, start_idx=start_idx)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, window_size)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        new_euclidean_length_dict = down_sampling(divided_euclidean_length_dict,
                                                  subsampling_factor=self.subsampling_factor)

        # ori
        array_3d_dict = butter_high_pass_filter(ori_array_3d_dict, self.param_dict["base_frequency"], wn=3)

        cut_array_dict = cut_array_to_length(array_3d_dict, ori_length, start_idx=ori_start_idx)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, window_size)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
        ori_euclidean_length_dict = down_sampling(divided_euclidean_length_dict, subsampling_factor=50)

        min_shape = min(new_euclidean_length_dict[list(new_euclidean_length_dict.keys())[0]].shape[0],
                        ori_euclidean_length_dict[list(ori_euclidean_length_dict.keys())[0]].shape[0])

        for i, key in enumerate(ori_euclidean_length_dict.keys()):
            axes[i].plot(new_euclidean_length_dict[key][-min_shape:])
            axes[i].plot(ori_euclidean_length_dict[key][-min_shape:])
            axes[i].set_title(key[1].value + "_" + key[2].value)
            axes[i].grid()

        plt.tight_layout()

        output_folder = "./plots_3"
        plt.savefig(os.path.join(output_folder, "{}_{}_ws_{}_wn_{:.2f}.png".format(self.image_idx,
                                                                                   self.param_dict["dst_frequency"],
                                                                                   window_size,
                                                                                   wn)))
        plt.clf()
        plt.close("all")
        self.image_idx += 1

    @staticmethod
    def plot_3d_arrays(array_dict, axes, ax_idx, num_of_rows, title, vlines=None, dim_size=1):
        offset = False
        for key in array_dict.keys():
            if offset:
                cur_ax_idx = ax_idx
            else:
                cur_ax_idx = ax_idx + int(num_of_rows / 2)
            if dim_size > 1:
                for i in range(dim_size):
                    axes[cur_ax_idx].plot(array_dict[key][:, i])
            else:
                axes[cur_ax_idx].plot(array_dict[key])

            if vlines is not None:
                for vline in vlines:
                    axes[cur_ax_idx].vlines(vline, ymin=array_dict[key].min(), ymax=array_dict[key].max(), colors="r")

            axes[cur_ax_idx].set_title(title + "_" + key[1].value + "_" + key[2].value)
            axes[cur_ax_idx].grid()
            offset = True
        ax_idx += 1
        return ax_idx
