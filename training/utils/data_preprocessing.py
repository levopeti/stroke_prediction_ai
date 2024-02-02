import numpy as np
import pandas as pd

from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Dict, Union

from training.utils.clear_measurements import all_key_combination, Key, MeasType


def get_3d_arrays_from_df(meas_df: pd.DataFrame) -> Dict[Key, np.ndarray]:
    """
    :param meas_df: pandas dataframe
    :return: dict, keys: (side, limb, meas_type), values: numpy array with shape (length, 3)
    """
    array_3d_dict = dict()
    for side, limb, meas_type in all_key_combination:
        try:
            result_array = np.concatenate(
                [np.expand_dims(meas_df[str((side.value, limb.value, meas_type.value, "x"))].values, axis=1),
                 np.expand_dims(meas_df[str((side.value, limb.value, meas_type.value, "y"))].values, axis=1),
                 np.expand_dims(meas_df[str((side.value, limb.value, meas_type.value, "z"))].values, axis=1)], axis=1)
            array_3d_dict[(side, limb, meas_type)] = result_array
        except KeyError:
            continue
    return array_3d_dict


def butter_high_pass_filter(meas_3d_arrays: Dict[Key, np.ndarray]) -> Dict[Key, np.ndarray]:
    """ fifth-order Butterworth filter with a cut-off frequency of 3 Hz """
    sos = signal.butter(5, 3, "highpass", output="sos", fs=25)
    for key, array_3d in meas_3d_arrays.items():
        filtered_list = list()
        if key[2] == MeasType.ACC:
            for i in range(3):
                sig = array_3d[:, i]
                filtered_list.append(np.expand_dims(signal.sosfilt(sos, sig), axis=1))
            meas_3d_arrays[key] = np.concatenate(filtered_list, axis=1)
    return meas_3d_arrays


def cut_array_to_length(meas_3d_arrays: Dict[Key, np.ndarray], length: int, start_idx: int) -> Dict[Key, np.ndarray]:
    cut_array_dict = dict()
    for key, array_3d in meas_3d_arrays.items():
        assert length < len(array_3d), (length, len(array_3d))
        if start_idx > len(array_3d) - (length + 1):
            raise ValueError("start_idx is too large")

        cut_array_dict[key] = array_3d[start_idx:start_idx + length]
    return cut_array_dict


def calculate_euclidean_length(meas_3d_arrays: Dict[Key, np.ndarray]) -> Dict[Key, np.ndarray]:
    euclidean_length_dict = {key: np.linalg.norm(array_3d, ord=2, axis=1) for key, array_3d in meas_3d_arrays.items()}
    return euclidean_length_dict


def clip_values(meas_1d_arrays: Dict[Key, np.ndarray], max_value: int, meas_type: MeasType) -> Dict[Key, np.ndarray]:
    for key, array_1d in meas_1d_arrays.items():
        if key[2] == meas_type:
            clipped_array_1d = np.clip(array_1d, 0, max_value)
            meas_1d_arrays[key] = clipped_array_1d
    return meas_1d_arrays


def divide_values(meas_1d_arrays: Dict[Key, np.ndarray], div_value: Union[int, float], meas_type: MeasType) -> Dict[
    Key, np.ndarray]:
    for key, array_1d in meas_1d_arrays.items():
        if key[2] == meas_type:
            new_array_1d = array_1d / div_value
            meas_1d_arrays[key] = new_array_1d
    return meas_1d_arrays


def moving_average(meas_1d_arrays: Dict[Key, np.ndarray], window_size: int) -> Dict[Key, np.ndarray]:
    for key, array_1d in meas_1d_arrays.items():
        averaged_array = uniform_filter1d(array_1d, size=window_size, mode="constant", cval=0.0)
        meas_1d_arrays[key] = averaged_array
    return meas_1d_arrays


def down_sampling(meas_1d_arrays: Dict[Key, np.ndarray], subsampling_factor: int):
    # 135000 / 50 = 2700 (90 min)
    for key, array_1d in meas_1d_arrays.items():
        meas_1d_arrays[key] = array_1d[::subsampling_factor]
    return meas_1d_arrays
