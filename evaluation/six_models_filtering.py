import os
from glob import glob

from evaluation.single_models_filtering import get_predictions
from training.utils.clear_measurements import Limb, ClearMeasurements, Side, get_inverted_side
from training.utils.measure_db import MeasureDB


def get_model_dict(_folder_path: str, to_cuda: bool = False) -> dict:
    # inverted_90_arm_2024-01-28-12-37.pt
    _model_dict = dict()
    for model_path in sorted(glob(os.path.join(_folder_path, "*.pt"))):
        model_name = model_path.split("/")[-1].split(".")[0]
        # inverted, length_min, limb, date = model_name.split("_")
        # model_dict[(inverted == "inverted", int(length_min))] = get_model(model_path, to_cuda=to_cuda)
        # _model_dict[(inverted == "inverted", int(length_min))] = model_path
        _model_dict[model_name] = model_path
    return _model_dict


def get_inverted_or_not(_model_name):
    return _model_name.split("_")[0] == "inverted"


if __name__ == "__main__":
    params = {
        # data info
        "accdb_path": "./data/WUS-v4measure20240116.accdb",
        "ucanaccess_path": "./ucanaccess/",
        "folder_path": "./data/clear_and_synchronized/",
        "clear_json_path": "./data/clear_train_val_ids.json",

        # measurement info
        "frequency": 25,  # HZ
        "training_length_min": 90,  # only for ClearMeasurements
        "step_size_min": 5,
        "step_size_sec": 20,
        "limb": Limb.ARM,
        # "window_length": 1,

        # dataset
        "invert_side": False,
        "class_mapping": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2},  # None, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
        "num_of_classes": 3}

    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, **params)

    model_folder_path = "./models"
    model_dict = get_model_dict(model_folder_path)

    avg_prob_threshold = 0.8
    max_window_length_sec = 60 * 60
    step_size_sec = 20


    def get_prediction_dict(_model_name_list, _meas_id, _watch_side):
        prediction_dict = dict()
        for _model_name in _model_name_list:
            measurement_side = get_inverted_side(watch_side) if get_inverted_or_not(_model_name) else watch_side
            prediction_dict[_model_name] = get_predictions(_model_name, _meas_id, measurement_side)
        return prediction_dict


    for data_type in ["train", "validation"]:
        folder_path = "./filter_results/six_model_filtering/".format(data_type)
        os.makedirs(folder_path, exist_ok=True)
        metrics_list = list()
        metrics_dict = {"window (s)": list(),
                        "threshold": list(),
                        "true positives": list(),
                        "true negative": list(),
                        "accuracy": list(),
                        "sensitivity": list(),
                        "specificity": list(),
                        "precision": list(),
                        }

        max_window_length = int(max_window_length_sec / step_size_sec)
        pred_is_stroke_list = list()
        prob_is_stroke_list = list()
        avg_pred_is_stroke_list = list()
        is_stroke_list = list()
        for meas_id in clear_measurements.get_meas_id_list(data_type):
            for watch_side in (Side.LEFT, Side.RIGHT):
                prediction_dict = get_prediction_dict(list(model_dict.keys()), meas_id, watch_side)

                for model_name, predictions in prediction_dict.items():
                    print(model_name, len(predictions))
            exit()

                # measurement_side = get_inverted_side(watch_side) if inverted else watch_side
                # class_value = clear_measurements.get_limb_class_value(meas_id, measurement_side, params["limb"])




