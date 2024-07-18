import os
from glob import glob

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import confusion_matrix

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


def get_meas_length(_model_name):
    return int(_model_name.split("_")[1])


def get_prediction_dict(_model_name_list, _meas_id, _watch_side):
    _prediction_dict = dict()
    for _model_name in _model_name_list:
        # measurement_side = get_inverted_side(_watch_side) if get_inverted_or_not(_model_name) else _watch_side
        _prediction_dict[_model_name] = get_predictions(_model_name, _meas_id, _watch_side)
    return _prediction_dict


def cut_for_same_length(_prediction_dict):
    min_length = float("inf")
    for model_name, predictions in _prediction_dict.items():
        if len(predictions) < min_length:
            min_length = len(predictions)

    for model_name, predictions in _prediction_dict.items():
        _prediction_dict[model_name] = predictions[-min_length:]

    return _prediction_dict


def get_pred_is_stroke_dict(_prediction_dict, avg_prob_threshold_dict, window_length_dict, step_size_sec):
    _pred_is_stroke_dict = dict()
    for model_name, predictions in _prediction_dict.items():
        inverted = "inverted" if get_inverted_or_not(model_name) else "non-inverted"
        meas_length = get_meas_length(model_name)
        window_length = int(window_length_dict[inverted][meas_length] / step_size_sec)
        predictions = predictions.argmax(axis=1)
        pred_is_stroke = (predictions < 1.5).astype(int)

        if window_length > 1:
            avg_prob_threshold = avg_prob_threshold_dict[inverted][meas_length]
            if len(pred_is_stroke) < window_length:
                current_window_length = len(pred_is_stroke)
            else:
                current_window_length = window_length
            avg_pred_is_stroke = sliding_window_view(pred_is_stroke, current_window_length).mean(axis=1)
            pred_is_stroke = (avg_pred_is_stroke >= avg_prob_threshold).astype(int)

        _pred_is_stroke_dict[model_name] = pred_is_stroke
    return _pred_is_stroke_dict


def get_is_stroke_arrays(_pred_is_stroke_dict):
    index_for_length = {30: 0, 60: 1, 90: 2}
    _non_inverted_array = np.zeros((len(list(_pred_is_stroke_dict.values())[0]), 3))  # 3 -> (30, 60, 90)
    _inverted_array = np.zeros((len(list(_pred_is_stroke_dict.values())[0]), 3))  # 3 -> (30, 60, 90)
    for model_name, pred_is_stroke in _pred_is_stroke_dict.items():
        meas_length = get_meas_length(model_name)
        idx = index_for_length[meas_length]
        if get_inverted_or_not(model_name):
            _inverted_array[:, idx] = pred_is_stroke
        else:
            _non_inverted_array[:, idx] = pred_is_stroke

    return _non_inverted_array, _inverted_array


def get_gt_array(_measurement_side, _pred_length, _meas_id, clear_measurements, params):
    class_value = clear_measurements.get_limb_class_value(_meas_id, _measurement_side, params["limb"])
    gt_array = np.zeros(_pred_length) if class_value > 1.5 else np.ones(_pred_length)
    return gt_array


def get_metrics(data_type, avg_prob_threshold_dict, model_dict, clear_measurements, window_length_dict, params,
                write_metrics):
    pred_is_stroke_list = {"non-inverted": list(), "inverted": list()}
    is_stroke_list = {"non-inverted": list(), "inverted": list()}
    metrics_dict = {"non-inverted": list(), "inverted": list()}
    for meas_id in clear_measurements.get_meas_id_list(data_type):
        for watch_side in (Side.LEFT, Side.RIGHT):
            prediction_dict = get_prediction_dict(list(model_dict.keys()), meas_id, watch_side)
            prediction_dict = cut_for_same_length(prediction_dict)
            pred_is_stroke_dict = get_pred_is_stroke_dict(prediction_dict, avg_prob_threshold_dict, window_length_dict,
                                                          params["step_size_sec"])
            pred_is_stroke_dict = cut_for_same_length(pred_is_stroke_dict)

            non_inverted_array, inverted_array = get_is_stroke_arrays(pred_is_stroke_dict)
            non_inverted_array = (non_inverted_array.sum(axis=1) >= 1).astype(int)
            inverted_array = (inverted_array.sum(axis=1) >= 1).astype(int)

            non_inverted_gt_array = get_gt_array(watch_side, len(inverted_array), meas_id, clear_measurements, params)
            inverted_gt_array = get_gt_array(get_inverted_side(watch_side), len(inverted_array), meas_id,
                                             clear_measurements, params)

            pred_is_stroke_list["non-inverted"].append(non_inverted_array)
            pred_is_stroke_list["inverted"].append(inverted_array)

            is_stroke_list["non-inverted"].append(non_inverted_gt_array)
            is_stroke_list["inverted"].append(inverted_gt_array)

    metrics_list = list()
    metrics_list.append("average probability threshold: {}\n".format(avg_prob_threshold_dict))
    metrics_list.append("window length (s): {}\n".format(window_length_dict))
    for is_inverted_key in is_stroke_list.keys():
        is_stroke_array = np.concatenate(is_stroke_list[is_inverted_key])
        pred_is_stroke_array = np.concatenate(pred_is_stroke_list[is_inverted_key])

        number_of_predictions = len(is_stroke_array)
        tn, fp, fn, tp = confusion_matrix(is_stroke_array, pred_is_stroke_array).ravel()

        accuracy = (tp + tn) / number_of_predictions
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        metrics_dict[is_inverted_key] = (accuracy, sensitivity, specificity, precision)

        if write_metrics:
            metrics_list.append(data_type + " " + is_inverted_key + "\n")

            metrics_list.append("number of predictions: {}\n".format(number_of_predictions))
            metrics_list.append("true positives: {} ({:.2f}%)\n".format(tp, tp / number_of_predictions * 100))
            metrics_list.append("true negative: {} ({:.2f}%)\n".format(tn, tn / number_of_predictions * 100))
            metrics_list.append("false positives: {} ({:.2f}%)\n".format(fp, fp / number_of_predictions * 100))
            metrics_list.append("false negative: {} ({:.2f}%)\n".format(fn, fn / number_of_predictions * 100))
            metrics_list.append("accuracy: {:.2f}\n".format(accuracy * 100))
            metrics_list.append("sensitivity: {:.2f}\n".format(sensitivity * 100))
            metrics_list.append("specificity: {:.2f}\n".format(specificity * 100))
            metrics_list.append("precision: {:.2f}\n\n".format(precision * 100))

    if write_metrics:
        with open(os.path.join(folder_path, "metrics.txt"), "a+") as f:
            f.writelines(metrics_list)

    return metrics_dict


if __name__ == "__main__":
    _params = {
        # data info
        "accdb_path": "./data/WUS-v4measure20240116.accdb",
        "ucanaccess_path": "./ucanaccess/",
        "folder_path": "./data/clear_and_synchronized/",
        "clear_json_path": "./data/clear_train_val_ids.json",

        # measurement info
        "base_frequency": 25,  # HZ
        "training_length_min": 90,  # only for ClearMeasurements
        "step_size_min": 5,
        "step_size_sec": 20,
        "limb": Limb.ARM,
        # "window_length": 1,

        # dataset
        "invert_side": False,
        "class_mapping": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2},  # None, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
        "num_of_classes": 3}

    measDB = MeasureDB(_params["accdb_path"], _params["ucanaccess_path"])
    _clear_measurements = ClearMeasurements(measDB, **_params)

    model_folder_path = "./models"
    _model_dict = get_model_dict(model_folder_path)

    _avg_prob_threshold_dict = {"non-inverted": {30: 0.9, 60: 0.8, 90: 0.7},
                                "inverted": {30: 0.95, 60: 0.85, 90: 0.75}}
    # max_window_length_sec = 90 * 60
    # _window_length = int(max_window_length_sec / _params["step_size_sec"])
    _window_length_dict = {"non-inverted": {30: 90 * 60, 60: 90 * 60, 90: 90 * 60},
                           "inverted": {30: 90 * 60, 60: 90 * 60, 90: 90 * 60}}

    for _data_type in ["train", "validation"]:
        folder_path = "./filter_results/six_model_filtering/".format(_data_type)
        os.makedirs(folder_path, exist_ok=True)
        get_metrics(_data_type, _avg_prob_threshold_dict, _model_dict, _clear_measurements, _window_length_dict, _params,
                    write_metrics=True)
