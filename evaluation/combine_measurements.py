import os

import pandas as pd
import torch
import numpy as np

from glob import glob
from tqdm import tqdm

from evaluation.six_model_evals import get_model, step_1, step_2
from training.datasets.measurement_info import MeasurementInfo
from training.utils.clear_measurements import ClearMeasurements, Limb, Side
from training.utils.converting_utils import min_to_ticks, frequency_to_timedelta_ms
from training.utils.measure_db import MeasureDB


def get_combined_measurement(_meas_ids_to_combine, _side, _limb):
    df_1 = clear_measurements.get_measurement(_meas_ids_to_combine[0], _side, _limb)
    df_1 = df_1[:len(df_1) // 2]
    df_2 = clear_measurements.get_measurement(_meas_ids_to_combine[1], _side, _limb)
    df_2 = df_2[len(df_2) // 2:]
    combined_df = pd.concat([df_1, df_2])


    class_value = clear_measurements.get_limb_class_value(_meas_ids_to_combine[0], measurement_side, params["limb"])

    expected_timedelta = frequency_to_timedelta_ms(params["base_frequency"])
    _timestamps = np.arange(0, len(combined_df), expected_timedelta)
    _meas_info = MeasurementInfo(combined_meas_id, _timestamps, params["base_frequency"])
    return combined_df, _meas_info


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

        folder_path = os.path.join("./combined_predictions", model_name)
        os.makedirs(folder_path, exist_ok=True)

        model = get_model(model_path, to_cuda=True)
        # 202205191: left: healty, right: healty
        # 202111041: left: stroke, right: healty
        meas_ids_to_combine = [202205191, 202111041]
        combined_meas_id = 202402081

        batch_idx = 0
        batch = list()
        with torch.no_grad():
            for side in (Side.LEFT, Side.RIGHT):
                df, meas_info = get_combined_measurement(meas_ids_to_combine, side, params["limb"])
                array_dict = step_1(df)
                num_of_samples = meas_info.get_number_of_samples(training_length_min,
                                                                 step_size_sec=params["step_size_sec"])
                predictions = list()
                timestamps = list()
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

                with open(os.path.join(folder_path, "{}_{}_{}.npy".format(combined_meas_id,
                                                                          params["limb"].value,
                                                                          side.value)), "wb") as f:
                    try:
                        np.save(f, np.concatenate(predictions, axis=0))
                    except ValueError:
                        breakpoint()
                        exit()

