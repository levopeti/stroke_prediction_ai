import os
import torch
import numpy as np

from glob import glob
from tqdm import tqdm

from training.utils.clear_measurements import ClearMeasurements, Limb, Side, MeasType
from training.utils.converting_utils import min_to_ticks
from training.utils.data_preprocessing import get_3d_arrays_from_df, cut_array_to_length, calculate_euclidean_length, \
    moving_average, divide_values, down_sampling, butter_high_pass_filter
from training.utils.feature_extraction import create_multivariate_time_series
from training.utils.measure_db import MeasureDB


def cut_array(array_3d: np.ndarray, length: int, start_idx: int) -> np.ndarray:
    assert length < len(array_3d), (length, len(array_3d))
    if start_idx > len(array_3d) - (length + 1):
        raise ValueError("start_idx is too large")

    return array_3d[start_idx:start_idx + length]


def step_1(meas_df) -> dict:
    array_3d_dict = get_3d_arrays_from_df(meas_df)
    filtered_array_3d_dict = butter_high_pass_filter(array_3d_dict)
    euclidean_length_dict = calculate_euclidean_length(filtered_array_3d_dict)
    return euclidean_length_dict


def step_2(euclidean_length_dict, start_idx, length) -> np.ndarray:
    cut_array_dict = cut_array_to_length(euclidean_length_dict, length, start_idx=start_idx)
    euclidean_length_dict = moving_average(cut_array_dict, 100)
    divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, MeasType.GYR)
    euclidean_length_dict = down_sampling(divided_euclidean_length_dict, subsampling_factor=50)

    multivariate_time_series = create_multivariate_time_series(euclidean_length_dict)
    return multivariate_time_series


def get_model(model_pt_path: str, to_cuda: bool = False) -> torch.nn.Module:
    _model = torch.jit.load(model_pt_path)
    if to_cuda:
        _model.to("cuda")
    else:
        _model.to("cpu")
    _model.eval()
    return _model


if __name__ == "__main__":
    params = {
        # data info
        "accdb_path": "./data/WUS-v4measure20240116.accdb",
        "ucanaccess_path": "./ucanaccess/",
        "folder_path": "./data/clear_and_synchronized/",
        "clear_json_path": "./data/clear_train_val_ids.json",

        # measurement info
        "base_frequency": 25,  # HZ
        "training_length_min": 90,
        "step_size_sec": 20,
        "step_size_min": 5,  # only for ClearMeasurements
        "limb": Limb.ARM,

        # dataset
        "invert_side": False,
        "class_mapping": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2},  # None, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}

        "batch_size": 1000}

    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, **params)
    # model_dict = get_model_dict()

    model_folder = "./models"
    for model_path in sorted(glob(os.path.join(model_folder, "*.pt"))):
        model_name = model_path.split("/")[-1].split(".")[0]
        training_length_min = int(model_name.split("_")[1])

        folder_path = os.path.join("./predictions", model_name)
        os.makedirs(folder_path, exist_ok=True)

        model = get_model(model_path, to_cuda=True)

        batch_idx = 0
        batch = list()
        with torch.no_grad():
            for meas_id in tqdm(
                    clear_measurements.get_meas_id_list("train") + clear_measurements.get_meas_id_list("validation"),
                    desc=model_name):
                meas_info = clear_measurements.meas_info_dict[meas_id]
                for side in (Side.LEFT, Side.RIGHT):
                    df = clear_measurements.get_measurement(meas_id, side, params["limb"])
                    array_dict = step_1(df)
                    num_of_samples = meas_info.get_number_of_samples(training_length_min,
                                                                     step_size_sec=params["step_size_sec"])
                    predictions = list()
                    for idx in range(num_of_samples):  # , "{} {}".format(meas_id, side.value)):
                        _start_idx, _ = meas_info.get_sample_start_end_index(idx, training_length_min,
                                                                             step_size_sec=params["step_size_sec"])
                        sample_array = step_2(array_dict, _start_idx,
                                              min_to_ticks(training_length_min, params["base_frequency"]))
                        batch.append(np.expand_dims(sample_array, axis=0))
                        batch_idx += 1

                        if batch_idx % params["batch_size"] == 0 or idx == (num_of_samples - 1):
                            batch_array = np.concatenate(batch, axis=0)
                            batch_tensor = torch.from_numpy(batch_array).float()
                            prediction = model(batch_tensor.to("cuda")).to("cpu").numpy()
                            if len(batch_array) == 1:
                                prediction = np.expand_dims(prediction, axis=0)

                            predictions.append(prediction)
                            batch = list()

                    with open(os.path.join(folder_path, "{}_{}_{}.npy".format(meas_id,
                                                                              params["limb"].value,
                                                                              side.value)), "wb") as f:
                        try:
                            np.save(f, np.concatenate(predictions, axis=0))
                        except ValueError:
                            breakpoint()
                            exit()
