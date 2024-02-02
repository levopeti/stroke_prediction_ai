import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from tqdm import tqdm
from scipy.special import softmax

from training.utils.clear_measurements import Limb, ClearMeasurements, Side, get_inverted_side
from training.utils.converting_utils import sec_to_ticks
from training.utils.measure_db import MeasureDB


def get_predictions(_model_name, _meas_id, _side, limb="arm", _training_length_min=None, _meas_info=None, mocked=False):
    if mocked:
        num_of_samples = _meas_info.get_number_of_samples(_training_length_min, step_size_sec=params["step_size_sec"])
        return np.random.rand(num_of_samples, 3)
    else:
        with open("./predictions/{}/{}_{}_{}.npy".format(_model_name, _meas_id, limb, _side.value),
                  "rb") as f:
            _predictions = np.load(f)
        return _predictions


def save_roc_curve(save_path, _is_stroke_list, _pred_is_stroke_list, prefix="", postfix=""):
    RocCurveDisplay.from_predictions(_is_stroke_list,
                                     _pred_is_stroke_list,
                                     name="is it stroke?",
                                     color="darkorange",
                                     plot_chance_level=True,
                                     )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Is it stroke? ROC curves:")
    plt.legend()
    # plt.show()

    roc_folder_path = os.path.join(save_path, "roc_curves")
    os.makedirs(roc_folder_path, exist_ok=True)
    plt.savefig(os.path.join(roc_folder_path, "{}_roc_{}_{}{}.png".format(prefix, window_length_sec,
                                                                    int(avg_prob_threshold * 100), postfix)))


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

    # non-inverted_90_arm_2024-01-28-12-37
    model_name = "non-inverted_90_arm_2024-01-28-12-37"
    inverted_str, training_length_min_str, limb_str, date = model_name.split("_")
    inverted = inverted_str == "inverted"
    training_length_min = int(training_length_min_str)

    for data_type in ["train", "validation"]:
        folder_path = "./filter_results/{}_{}/".format(model_name, data_type)
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

        for avg_prob_threshold in tqdm([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]):
            for window_length_sec in [20, 60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60, 90 * 60, 120 * 60]:
                # if int(window_length_sec / 60) > training_length_min:
                #     continue
                window_length = int(window_length_sec / params["step_size_sec"])
                pred_is_stroke_list = list()
                prob_is_stroke_list = list()
                avg_pred_is_stroke_list = list()
                is_stroke_list = list()
                for meas_id in clear_measurements.get_meas_id_list(data_type):
                    # meas_info = clear_measurements.meas_info_dict[meas_id]
                    for watch_side in (Side.LEFT, Side.RIGHT):
                        predictions = get_predictions(model_name, meas_id, watch_side)
                        if window_length > len(predictions):
                            continue
                        probabilities_of_healty = softmax(predictions, axis=1)[:, 2]
                        probabilities_of_stroke = 1 - probabilities_of_healty

                        predictions = predictions.argmax(axis=1)
                        pred_is_stroke = (predictions < 1.5).astype(int)

                        if window_length > 1:
                            probabilities_of_stroke = probabilities_of_stroke[window_length - 1:]
                            avg_pred_is_stroke = sliding_window_view(pred_is_stroke, window_length).mean(axis=1)
                            pred_is_stroke = (avg_pred_is_stroke > avg_prob_threshold).astype(int)
                            avg_pred_is_stroke_list.append(avg_pred_is_stroke)

                        prob_is_stroke_list.append(probabilities_of_stroke)
                        pred_is_stroke_list.append(pred_is_stroke)

                        measurement_side = get_inverted_side(watch_side) if inverted else watch_side
                        class_value = clear_measurements.get_limb_class_value(meas_id, measurement_side, params["limb"])

                        if class_value > 1.5:
                            # healthy
                            is_stroke_list.append(np.zeros_like(pred_is_stroke))
                        else:
                            # stroke
                            is_stroke_list.append(np.ones_like(pred_is_stroke))

                is_stroke_list = np.concatenate(is_stroke_list)
                pred_is_stroke_list = np.concatenate(pred_is_stroke_list)
                prob_is_stroke_list = np.concatenate(prob_is_stroke_list)
                save_roc_curve(folder_path, is_stroke_list, prob_is_stroke_list, "prob")

                if len(avg_pred_is_stroke_list) > 0:
                    avg_pred_is_stroke_list = np.concatenate(avg_pred_is_stroke_list)
                    save_roc_curve(folder_path, is_stroke_list, avg_pred_is_stroke_list, "avg_pred")

                number_of_predictions = len(is_stroke_list)
                tn, fp, fn, tp = confusion_matrix(is_stroke_list, pred_is_stroke_list).ravel()

                accuracy = (tp + tn) / number_of_predictions
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                precision = tp / (tp + fp)

                metrics_dict["threshold"].append(avg_prob_threshold)
                metrics_dict["window (s)"].append(window_length_sec)
                metrics_dict["true positives"].append(tp / number_of_predictions * 100)
                metrics_dict["true negative"].append(tn / number_of_predictions * 100)
                metrics_dict["accuracy"].append(accuracy * 100)
                metrics_dict["sensitivity"].append(sensitivity * 100)
                metrics_dict["specificity"].append(specificity * 100)
                metrics_dict["precision"].append(precision * 100)

                metrics_list.append("average probability threshold: {}\n".format(avg_prob_threshold))
                metrics_list.append("window length (s): {}\n".format(window_length_sec))
                metrics_list.append("number of predictions: {}\n".format(number_of_predictions))
                metrics_list.append("true positives: {} ({:.2f}%)\n".format(tp, tp / number_of_predictions * 100))
                metrics_list.append("true negative: {} ({:.2f}%)\n".format(tn, tn / number_of_predictions * 100))
                metrics_list.append("false positives: {} ({:.2f}%)\n".format(fp, fp / number_of_predictions * 100))
                metrics_list.append("false negative: {} ({:.2f}%)\n".format(fn, fn / number_of_predictions * 100))
                metrics_list.append("accuracy: {:.2f}\n".format(accuracy * 100))
                metrics_list.append("sensitivity: {:.2f}\n".format(sensitivity * 100))
                metrics_list.append("specificity: {:.2f}\n".format(specificity * 100))
                metrics_list.append("precision: {:.2f}\n\n".format(precision * 100))

        with open(os.path.join(folder_path, "metrics.txt"), "a+") as f:
            f.writelines(metrics_list)

        metrics_df = pd.DataFrame.from_dict(metrics_dict)
        pt_dict = dict()
        for metric_name in metrics_dict.keys():
            if metric_name not in ["threshold", "window (s)"]:
                pt_dict[metric_name] = pd.pivot_table(metrics_df, values=metric_name, index=["threshold"],
                                                      columns=["window (s)"])

        fig, axs = plt.subplots(len(pt_dict), 1, facecolor="w", figsize=(15, 30))

        for axs_id, (metric_name, metric_pt) in enumerate(pt_dict.items()):
            sns.heatmap(metric_pt, cmap="coolwarm", linewidths=0.30, annot=True, ax=axs[axs_id])
            axs[axs_id].title.set_text(metric_name)

        plt.savefig(os.path.join(folder_path, "metrics.png"))

