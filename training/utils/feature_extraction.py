import numpy as np

# from numba import njit
from typing import Dict
from training.utils.clear_measurements import Key


def get_features(meas_arrays: Dict[Key, np.ndarray]) -> np.ndarray:
    # @njit
    def calculate_features(np_array: np.ndarray) -> np.ndarray:
        mean = np.mean(np_array)
        median = np.median(np_array)
        std = np.std(np_array)
        max_value = np.max(np_array)

        zero_sigma = np.sum(np_array > mean) / len(np_array)
        one_sigma = np.sum(np_array > (mean + std)) / len(np_array)
        two_sigma = np.sum(np_array > (mean + 2 * std)) / len(np_array)
        three_sigma = np.sum(np_array > (mean + 3 * std)) / len(np_array)
        four_sigma = np.sum(np_array > (mean + 4 * std)) / len(np_array)
        # mean, median, std, max_value, zero_sigma, one_sigma, two_sigma, three_sigma, four_sigma
        return np.array([mean, median, std, max_value, zero_sigma, one_sigma, two_sigma, three_sigma, four_sigma])

    feature_vector_list = list()
    for key, array in meas_arrays.items():
        features = calculate_features(array)
        feature_vector_list.append(features)
    return np.concatenate(feature_vector_list)

def create_multivariate_time_series(meas_arrays: Dict[Key, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.expand_dims(array, axis=0) for array in meas_arrays.values()], axis=0)







