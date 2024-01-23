import os
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib
from sklearn.metrics import confusion_matrix, RocCurveDisplay

from ai_utils.train_keras import custom_loss, stroke_accuracy
from ai_utils.training_utils.func_utils import get_limb_diff_mean, get_limb_ratio_mean
from measurement_utils.measure_db import MeasureDB
from utils.cache_utils import cache
from ai_utils.train_pytorch import ClearMeasurements

matplotlib.rcParams['figure.figsize'] = [15, 15]

# timestamp per sec
TS_PER_SEC = 25


def make_plot(meas_name, pred_dict, minutes, step_size, save_path=None, type_of_set="train", plot=False):
    print("make plot {}".format(meas_name))
    plt.ion()

    pred_is_healthy_list = list()
    is_healty_list = list()
    print("class value: {}".format(set(pred_dict["class_values"])))
    print("len of y pred list: {}".format(len(pred_dict["y_pred_list"])))

    pred_array = np.array(pred_dict["y_pred_list"]).argmax(axis=1)
    percentage_list = [len(pred_array[pred_array == value]) / len(pred_array) * 100 for value in range(6)]
    fig, axs = plt.subplots(4, 1, facecolor="w")

    color_list = ["blue"] * 6
    for i in set(pred_dict["class_values"]):
        color_list[i] = "red"

    graph = axs[0].bar(list(range(6)),
                       percentage_list,
                       color=color_list)

    for p in graph:
        height = p.get_height()
        axs[0].text(x=p.get_x() + p.get_width() / 2, y=height + 1,
                    s="{:.3} %".format(height),
                    ha='center')

    axs[0].set_ylim(-5, 105)
    axs[0].legend(["Ratio of detections during the measurement"], loc='best')
    axs[0].grid(True)

    pred_is_healthy = np.array(pred_dict["y_pred_list"]).argmax(axis=1) > 4.5
    is_healty = np.array(pred_dict["class_values"]) > 4.5
    ratio = np.sum(pred_is_healthy == is_healty) / len(pred_is_healthy)

    pred_is_healthy_list.append(pred_is_healthy)
    is_healty_list.append(np.ones_like(pred_is_healthy) * is_healty)

    axs[1].pie([ratio, 1 - ratio], explode=(0, 0.1), labels=["True", "False"], autopct='%1.1f%%',
               shadow=True, startangle=90)
    axs[1].legend(["Prediction in terms of is it healty or not"], loc='best')

    axs[2].plot(np.array(pred_dict["y_pred_list"]).argmax(axis=1), label=meas_name)
    axs[2].plot(pred_dict["class_values"], label="class_values ({})".format(set(pred_dict["class_values"])))
    axs[2].axis([None, None, -0.5, 5.5])
    axs[2].legend(loc='best')
    axs[2].grid()

    xformatter = md.DateFormatter('%H:%M')
    # xlocator = md.MinuteLocator(interval=8)
    xlocator = md.HourLocator(interval=100)
    axs[2].xaxis.set_major_formatter(xformatter)
    axs[2].xaxis.set_major_locator(xlocator)

    axs[3].plot(np.array(pred_dict["y_pred_list"]).max(axis=1) * 100, label="confidence of prediction")
    axs[3].plot([a[i] * 100 for a, i in zip(pred_dict["y_pred_list"], pred_dict["class_values"])],
                label="confidence of prediction for true label")
    axs[3].axis([None, None, -5, 105])
    axs[3].legend(loc='best')
    axs[3].grid()

    if save_path is not None:
        os.makedirs(
            os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/".format(minutes, step_size,
                                                                           datetime.now().strftime('%Y-%m-%d-%H'),
                                                                           type_of_set)), exist_ok=True)
        plt.savefig(os.path.join(save_path,
                                 "plots/plots_{}m_{}step_{}/{}/{}.png".format(minutes, step_size,
                                                                              datetime.now().strftime(
                                                                                  '%Y-%m-%d-%H'),
                                                                              type_of_set, meas_name)))

    if plot:
        plt.show()


def make_roc_curve(result_dict, minutes, step_size, save_path=None, type_of_set="train", plot=False):
    print("sens_spec_all_data")
    prob_is_healthy_list = list()
    is_healty_list = list()
    for meas_name, pred_dict in result_dict.items():
        prob_is_healthy = np.array(pred_dict["y_pred_list"])[:, 5]
        is_healty = np.array(pred_dict["class_values"]) > 4.5
        prob_is_healthy_list.append(prob_is_healthy)
        is_healty_list.append(np.ones_like(prob_is_healthy) * is_healty)

    RocCurveDisplay.from_predictions(np.concatenate(is_healty_list),
                                     np.concatenate(prob_is_healthy_list),
                                     name="is it healthy?",
                                     color="darkorange",
                                     plot_chance_level=True,
                                     )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Is healthy? ROC curves:")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/".format(minutes, step_size,
                                                                                   datetime.now().strftime(
                                                                                       '%Y-%m-%d-%H'),
                                                                                   type_of_set)), exist_ok=True)
        plt.savefig(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/roc.png".format(minutes, step_size,
                                                                                          datetime.now().strftime(
                                                                                              '%Y-%m-%d-%H'),
                                                                                          type_of_set)))

    if plot:
        plt.show()


def sens_spec_all_data(result_dict, minutes, step_size, save_path=None, type_of_set="train", plot=False):
    print("sens_spec_all_data")
    pred_is_healthy_list = list()
    is_healty_list = list()
    for meas_name, pred_dict in result_dict.items():
        pred_is_healthy = np.array(pred_dict["y_pred_list"]).argmax(axis=1) > 4.5
        is_healty = np.array(pred_dict["class_values"]) > 4.5
        pred_is_healthy_list.append(pred_is_healthy)
        is_healty_list.append(np.ones_like(pred_is_healthy) * is_healty)

    tn_fp_fn_tp = confusion_matrix(~np.concatenate(is_healty_list), ~np.concatenate(pred_is_healthy_list)).ravel()

    if len(tn_fp_fn_tp) == 4:
        print("tn_fp_fn_tp ", tn_fp_fn_tp)
        tn, fp, fn, tp = tn_fp_fn_tp
        # Sensitivity = TP / (TP + FN)
        sensitivity = tp / (tp + fn)
        print(sensitivity)
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp)
        print(specificity)

        color_list = ['blue', 'red']

        fig = plt.figure(facecolor="w")
        ax = fig.add_axes([0, 0, 1, 1])
        langs = ['sensitivity', 'specificity']
        students = [sensitivity * 100, specificity * 100]
        graph = ax.bar(langs, students, color=color_list)

        for p in graph:
            height = p.get_height()
            ax.text(x=p.get_x() + p.get_width() / 2, y=height + 1,
                    s="{:.2f} %".format(height), ha='center')

        ax.legend(["sensitivity - specificity"], loc='best')
        ax.grid(True)

        if save_path is not None:
            os.makedirs(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/".format(minutes, step_size,
                                                                                       datetime.now().strftime(
                                                                                           '%Y-%m-%d-%H'),
                                                                                       type_of_set)), exist_ok=True)
            plt.savefig(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/sens_spec.png".format(minutes, step_size,
                                                                                                    datetime.now().strftime(
                                                                                                        '%Y-%m-%d-%H'),
                                                                                                    type_of_set)))
    if plot:
        plt.show()


def sens_spec_window(result_dict: dict, step_size: int, save_path: str, minutes: int, type_of_set: str):
    print("sens_spec_window")
    step_size_sec = int(step_size // TS_PER_SEC)
    sens_spec_dict = {"threshold": list(),
                      "window (s)": list(),
                      "sensitivity": list(),
                      "specificity": list()}

    for avg_prob_threshold in [0.7, 0.8, 0.85, 0.9, 0.95]:
        for window_length_sec in [20, 60, 120, 300, 600]:
            window_length = int(window_length_sec / step_size_sec)

            pred_is_stroke_list = list()
            is_stroke_list = list()
            for meas_name, pred_dict in result_dict.items():
                pred_is_stroke = np.array(pred_dict["y_pred_list"]).argmax(axis=1) < 4.5
                is_stroke = np.array(pred_dict["class_values"]) < 4.5

                avg_pred = np.lib.stride_tricks.sliding_window_view(pred_is_stroke, window_length).mean(axis=1)
                avg_pred_is_stroke = avg_pred > avg_prob_threshold

                pred_is_stroke_list.append(avg_pred_is_stroke)
                is_stroke_list.append(is_stroke[-len(avg_pred_is_stroke):])

            tn_fp_fn_tp = confusion_matrix(np.concatenate(is_stroke_list), np.concatenate(pred_is_stroke_list)).ravel()
            if len(tn_fp_fn_tp) == 4:
                tn, fp, fn, tp = tn_fp_fn_tp
                # Sensitivity = TP / (TP + FN)
                sensitivity = tp / (tp + fn)
                # Specificity = TN / (TN + FP)
                specificity = tn / (tn + fp)
            else:
                sensitivity = np.nan
                specificity = 1

            sens_spec_dict["threshold"].append(avg_prob_threshold)
            sens_spec_dict["window (s)"].append(window_length_sec)
            sens_spec_dict["sensitivity"].append(sensitivity)
            sens_spec_dict["specificity"].append(specificity)

    sens_spec_df = pd.DataFrame.from_dict(sens_spec_dict)
    sens_df = pd.pivot_table(sens_spec_df, values="sensitivity", index=["threshold"], columns=["window (s)"])
    spec_df = pd.pivot_table(sens_spec_df, values="specificity", index=["threshold"], columns=["window (s)"])
    print(sens_df)
    print(spec_df)

    if len(sens_df) > 0 and len(spec_df) > 0:
        fig, axs = plt.subplots(2, 1, facecolor="w")
    else:
        fig, axs = plt.subplots(1, 1, facecolor="w")
        axs = [axs]

    axs_id = 0

    if len(sens_df) > 0:
        sns.heatmap(sens_df, vmax=1, cmap="coolwarm", linewidths=0.30, annot=True, ax=axs[axs_id])
        axs[axs_id].title.set_text("Sensitivity")
        axs_id += 1

    if len(spec_df) > 0:
        sns.heatmap(spec_df, vmax=1, cmap="coolwarm", linewidths=0.30, annot=True, ax=axs[axs_id])
        axs[axs_id].title.set_text("Specificity")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/sens_spec_hm.png".format(minutes, step_size,
                                                                                                   datetime.now().strftime(
                                                                                                       '%Y-%m-%d-%H'),
                                                                                                   type_of_set)))


def load_model(_model_path):
    _model = keras.models.load_model(_model_path, compile=False)  # custom_objects={"custom_loss": custom_loss, "stroke_accuracy": stroke_accuracy})
    _model.compile(loss=custom_loss(stroke_loss_factor=0.1), optimizer=Adam(), metrics=["accuracy", stroke_accuracy])
    _model.summary()

    return _model


def make_prediction(_model, _data_dict: dict):
    result_dict = dict()

    y_pred_list = _model.predict(_data_dict["X"])
    result_dict["y_pred_list"] = y_pred_list
    result_dict["class_values"] = _data_dict["y"]

    return result_dict


@cache
def generate_infer_data(meas_id: int,
                        length: int,
                        clear_measurements: ClearMeasurements,
                        step_size: int,
                        use_cache: bool,
                        key: str) -> dict:
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    class_value_dict = clear_measurements.get_class_value_dict(meas_id=meas_id)
    class_value = clear_measurements.get_min_class_value(meas_id=meas_id)
    meas_df = clear_measurements.get_measurement(meas_id)
    result_dict = {"X": list(),
                   "y": list()}
    start_idx = 0

    tq = tqdm(total=((len(meas_df[str(("left", "arm", "acc", "x"))].values) - length) // step_size))
    tq.set_description("generate infer data ({})".format(meas_id))

    while True:
        try:
            instance = list()
            for key in keys_in_order:
                class_value_left = class_value_dict[("left", key[0])]
                class_value_right = class_value_dict[("right", key[0])]

                left_x_y_z = [meas_df[str(("left", key[0], key[1], "x"))].values,
                              meas_df[str(("left", key[0], key[1], "y"))].values,
                              meas_df[str(("left", key[0], key[1], "z"))].values]

                right_x_y_z = [meas_df[str(("right", key[0], key[1], "x"))].values,
                               meas_df[str(("right", key[0], key[1], "y"))].values,
                               meas_df[str(("right", key[0], key[1], "z"))].values]
                meas_type_acc = key[1] == "acc"
                diff_mean = get_limb_diff_mean(left_x_y_z, right_x_y_z, meas_type_acc, length, start_idx=start_idx)
                ratio_mean_first = get_limb_ratio_mean(left_x_y_z, right_x_y_z, meas_type_acc, class_value_left, class_value_right, length, mean_first=True, start_idx=start_idx)
                ratio_mean = get_limb_ratio_mean(left_x_y_z, right_x_y_z, meas_type_acc, class_value_left, class_value_right, length, mean_first=False, start_idx=start_idx)
                instance.extend([diff_mean, ratio_mean, ratio_mean_first])

            result_dict["X"].append(instance)
            result_dict["y"].append(class_value)
        except ValueError:
            break
        start_idx = start_idx + step_size
        tq.update(1)

    tq.close()
    return result_dict


def start_evaluation(_param_dict):
    _db_path = _param_dict["db_path"]
    _base_path = _param_dict["base_path"]
    _ucanaccess_path = _param_dict["ucanaccess_path"]
    _class_values_to_use = _param_dict["class_values_to_use"]

    length = int(TS_PER_SEC * 60 * _param_dict["minutes"])
    step_size = _param_dict["step_size"]
    type_of_set = _param_dict["type_of_set"]
    save_path = _param_dict["save_path"]
    model_path = _param_dict["model_path"]

    _model = load_model(model_path)

    measDB = MeasureDB(_db_path, _ucanaccess_path)
    clear_measurements = ClearMeasurements(measDB, _base_path, param_dict["clear_json_path"])

    result_dict = dict()
    for meas_id in clear_measurements.get_meas_id_list(type_of_set):  # tqdm(clear_measurements.get_meas_id_list(type_of_set), "make evaluation"):
        key = "_".join([str(item) for item in [meas_id, length, step_size, type_of_set]])
        infer_data = generate_infer_data(meas_id, length, clear_measurements, step_size, use_cache=True, key=key)
        prediction_dict = make_prediction(_model, infer_data)
        make_plot(meas_id, prediction_dict, _param_dict["minutes"], step_size, save_path=save_path, type_of_set=type_of_set)
        result_dict[meas_id] = prediction_dict

    make_roc_curve(result_dict, _param_dict["minutes"], step_size, save_path=save_path, type_of_set=type_of_set)
    sens_spec_all_data(result_dict, _param_dict["minutes"], step_size, save_path=save_path, type_of_set=type_of_set)
    sens_spec_window(result_dict, step_size, save_path, _param_dict["minutes"], type_of_set)
    # write_prediction_to_csv(prediction_dict, length, step_size, save_path)


if __name__ == "__main__":
    param_dict = {
        "minutes": 90,
        "limb": "all",
        "class_values_to_use": None,  # [0, 1, 2]
        "step_size": 500,  # x * TIME_DELTA_SEC in sec
        "type_of_set": "test",  # train, test, mixed
        "base_path": "./data/clear_data",
        "db_path": "./data/WUS-v4measure202307311.accdb",
        "ucanaccess_path": "./ucanaccess/",
        "model_path": "./models/2023-10-04-13-43",
        "clear_json_path": "./data/clear_train_test_ids.json"
    }

    param_dict["save_path"] = param_dict["model_path"]

    start_evaluation(param_dict)

    param_dict["type_of_set"] = "train"
    start_evaluation(param_dict)
