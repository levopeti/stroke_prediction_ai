import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from evaluation.six_models_filtering import cut_for_same_length, get_is_stroke_arrays, get_pred_is_stroke_dict, \
    get_prediction_dict, get_model_dict, get_gt_array
from training.utils.clear_measurements import Side, Limb, ClearMeasurements, get_inverted_side
from training.utils.measure_db import MeasureDB


def make_plot(ax, gt_value, pred_array, pred_ts):
    color_list = ["blue" if x else "red" for x in gt_value == pred_array]
    ax.scatter(pred_ts, pred_array, c=color_list, s=50, label="Prediction")
    ax.plot(pred_ts, gt_value, c="orange", label="Ground truth")
    ax.axis([None, None, -0.25, 1.25])
    ax.legend(loc="best")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["stroke" if x == 1 else "healthy" for x in [0, 1]])
    ax.set_xlabel("minutes")
    ax.grid()


def plot_table(ax, _df):
    ax.table(cellText=_df.values,
             colLabels=_df.columns,
             # rowLabels=["Row 1", "Row 2", "Row 3"],
             # rowColours=["yellow"] * 3,
             # colColours=["red"] * 3,
             loc="center")
    # set_fontsize(14)
    # table.scale(1, 2)


def get_metrics(gt_array, pred_array):
    gt_array, pred_array = gt_array.astype(bool), pred_array.astype(bool)
    _tp = (pred_array * gt_array).sum()
    _tn = (~pred_array * ~gt_array).sum()
    _fp = (pred_array * ~gt_array).sum()
    _fn = (~pred_array * gt_array).sum()
    return _tn, _fp, _fn, _tp


def get_max_length_of_miss(gt_array, pred_array, _alert_step_size_min):
    gt_array, pred_array = gt_array.astype(bool), pred_array.astype(bool)
    condition = ~pred_array * gt_array
    len_of_miss = np.diff(np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:], [True])))[0])[::2]
    return len_of_miss.max() * _alert_step_size_min if len(len_of_miss) > 0 else 0


if __name__ == "__main__":
    params = {
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

    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, **params)

    model_folder_path = "./models"
    model_dict = get_model_dict(model_folder_path)

    """Risk 1: specificitás maxra húzva a súlyozásnál
       Risk 2: amit 02.05. este csináltál, tehát az optimális szint
       Risk 3: szenzitivitás maxra súlyozva."""

    # avg_prob_threshold_dict = {"non-inverted": {30: 0.9, 60: 0.8, 90: 0.7},
    #                            "inverted": {30: 0.95, 60: 0.85, 90: 0.75}}

    avg_prob_threshold_dict = {"risk 1": {"inverted": {30: 0.531, 60: 0.994, 90: 0.999},
                                          "non-inverted": {30: 0.874, 60: 0.999, 90: 0.717}},
                               "risk 2": {"inverted": {30: 0.405, 60: 0.467, 90: 0.999},
                                          "non-inverted": {30: 0.781, 60: 0.807, 90: 0.939}},
                               "risk 3": {"inverted": {30: 0.225, 60: 0.305, 90: 0.999},
                                          "non-inverted": {30: 0.778, 60: 0.721, 90: 0.424}}
                               }

    window_length_dict = {"risk 1": {"inverted": {30: 10586, 60: 4819, 90: 10691},
                                     "non-inverted": {30: 10281, 60: 10788, 90: 9559}},
                          "risk 2": {"inverted": {30: 10252, 60: 10549, 90: 10667},
                                     "non-inverted": {30: 10774, 60: 10731, 90: 6697}},
                          "risk 3": {"inverted": {30: 8770, 60: 10709, 90: 10680},
                                     "non-inverted": {30: 9327, 60: 8659, 90: 10570}}
                          }

    # max_window_length_sec = 90 * 60
    # _window_length = int(max_window_length_sec / _params["step_size_sec"])
    # window_length_dict = {"non-inverted": {30: 90 * 60, 60: 90 * 60, 90: 90 * 60},
    #                       "inverted": {30: 90 * 60, 60: 90 * 60, 90: 90 * 60}}

    alert_step_size_min = {"risk 1": 60,
                           "risk 2": 30,
                           "risk 3": 15
                           }

    alert_step_size = {"risk 1": int(60 * 60 / params["step_size_sec"]),
                       "risk 2": int(30 * 60 / params["step_size_sec"]),
                       "risk 3": int(15 * 60 / params["step_size_sec"])
                       }
    save_plot = True

    def fill_df_dict(_df_dict, _meas_id, _watch_side, _metrics_dict):
        for is_inverted in _metrics_dict.keys():
            _df_dict["Meas. ID"].append(_meas_id)
            _df_dict["Watch side"].append(_watch_side.value)
            _df_dict["Is inverted?"].append(is_inverted == "inverted")
            for _risk_group in _metrics_dict[is_inverted].keys():
                _tn, _fp, _fn, _tp, _max_length_of_miss, num_of_preds = metrics_dict[is_inverted][_risk_group]
                _df_dict["Correct prediction (n) [{}]".format(_risk_group)].append(_tn + _tp)
                corr_ratio = "{:.2f}%".format((_tn + _tp) / num_of_preds * 100)
                _df_dict["Correct prediction (%) [{}]".format(_risk_group)].append(corr_ratio)
                _df_dict["False alarm (n) [{}]".format(_risk_group)].append(_fp)
                fp_ratio = "{:.2f}%".format(_fp / num_of_preds * 100)
                _df_dict["False alarm (%) [{}]".format(_risk_group)].append(fp_ratio)
                _df_dict["Missed alarm (n) [{}]".format(_risk_group)].append(_fn)
                fn_ratio = "{:.2f}%".format(_fn / num_of_preds * 100)
                _df_dict["Missed alarm (%) [{}]".format(_risk_group)].append(fn_ratio)
                _df_dict["Longest 'missed alarm' period (min) [{}]".format(_risk_group)].append(_max_length_of_miss)


    for data_type in ["train", "validation"]:
        folder_path = "./filter_results/six_model_filtering_onebyone/{}".format(data_type)
        os.makedirs(folder_path, exist_ok=True)
        df_dict = {"Meas. ID": list(),
                   "Watch side": list(),
                   "Is inverted?": list(),
                   "Correct prediction (n) [risk 1]": list(),
                   "Correct prediction (%) [risk 1]": list(),
                   "False alarm (n) [risk 1]": list(),
                   "False alarm (%) [risk 1]": list(),
                   "Missed alarm (n) [risk 1]": list(),
                   "Missed alarm (%) [risk 1]": list(),
                   "Longest 'missed alarm' period (min) [risk 1]": list(),
                   "Correct prediction (n) [risk 2]": list(),
                   "Correct prediction (%) [risk 2]": list(),
                   "False alarm (n) [risk 2]": list(),
                   "False alarm (%) [risk 2]": list(),
                   "Missed alarm (n) [risk 2]": list(),
                   "Missed alarm (%) [risk 2]": list(),
                   "Longest 'missed alarm' period (min) [risk 2]": list(),
                   "Correct prediction (n) [risk 3]": list(),
                   "Correct prediction (%) [risk 3]": list(),
                   "False alarm (n) [risk 3]": list(),
                   "False alarm (%) [risk 3]": list(),
                   "Missed alarm (n) [risk 3]": list(),
                   "Missed alarm (%) [risk 3]": list(),
                   "Longest 'missed alarm' period (min) [risk 3]": list()
                   }
        for meas_id in tqdm(clear_measurements.get_meas_id_list(data_type), desc=data_type):
            if save_plot:
                fig, axs = plt.subplots(12, 1, facecolor="w", figsize=(20, 60))
                fig.tight_layout(pad=5.0)
                ax_idx = 0
            for watch_side in (Side.LEFT, Side.RIGHT):
                prediction_dict = get_prediction_dict(list(model_dict.keys()), meas_id, watch_side)
                prediction_dict = cut_for_same_length(prediction_dict)
                metrics_dict = {"non-inverted": dict(), "inverted": dict()}
                for risk_group in ["risk 1", "risk 2", "risk 3"]:
                    pred_is_stroke_dict = get_pred_is_stroke_dict(prediction_dict, avg_prob_threshold_dict[risk_group],
                                                                  window_length_dict[risk_group],
                                                                  params["step_size_sec"])
                    pred_is_stroke_dict = cut_for_same_length(pred_is_stroke_dict)

                    non_inverted_array, inverted_array = get_is_stroke_arrays(pred_is_stroke_dict)
                    non_inverted_array = (non_inverted_array.sum(axis=1) >= 1).astype(int)
                    inverted_array = (inverted_array.sum(axis=1) >= 1).astype(int)

                    non_inverted_array = non_inverted_array[::alert_step_size[risk_group]]
                    inverted_array = inverted_array[::alert_step_size[risk_group]]

                    non_inverted_gt_array = get_gt_array(watch_side, len(inverted_array), meas_id, clear_measurements,
                                                         params)
                    inverted_gt_array = get_gt_array(get_inverted_side(watch_side), len(inverted_array), meas_id,
                                                     clear_measurements, params)

                    pred_timestamps = [x * alert_step_size_min[risk_group] for x in range(len(inverted_array))]

                    # non-inverted
                    if save_plot:
                        axs[ax_idx].title.set_text("Non-inverted  watch side: {}  {}".format(watch_side.value, risk_group))
                        make_plot(axs[ax_idx], non_inverted_gt_array, non_inverted_array, pred_timestamps)
                        # ax_idx += 1

                    tn, fp, fn, tp = get_metrics(non_inverted_gt_array, non_inverted_array)
                    max_length_of_miss = get_max_length_of_miss(non_inverted_gt_array, non_inverted_array, alert_step_size_min[risk_group])
                    metrics_dict["non-inverted"][risk_group] = tn, fp, fn, tp, max_length_of_miss, len(pred_timestamps)

                    # inverted
                    if save_plot:
                        axs[ax_idx + 3].title.set_text("Inverted  watch side: {}  {}".format(watch_side.value, risk_group))
                        make_plot(axs[ax_idx + 3], inverted_gt_array, inverted_array, pred_timestamps)
                        ax_idx += 1

                    tn, fp, fn, tp = get_metrics(inverted_gt_array, inverted_array)
                    max_length_of_miss = get_max_length_of_miss(inverted_gt_array, inverted_array, alert_step_size_min[risk_group])
                    metrics_dict["inverted"][risk_group] = tn, fp, fn, tp, max_length_of_miss, len(pred_timestamps)

                fill_df_dict(df_dict, meas_id, watch_side, metrics_dict)
                if save_plot:
                    ax_idx += 3

                # df = pd.DataFrame.from_dict(df_dict)
                # df = df[df["Meas. ID"] == meas_id]
                #
                # # non-inverted
                # non_inv_df = df[~df["Is inverted?"] & (df["Watch side"] == watch_side.value)]
                # plot_table(axs[ax_idx], non_inv_df.drop(["Watch side", "Is inverted?"], axis=1))
                #
                # # inverted
                # inv_df = df[df["Is inverted?"] & (df["Watch side"] == watch_side.value)]
                # plot_table(axs[ax_idx + 4], inv_df.drop(["Watch side", "Is inverted?"], axis=1))
                # ax_idx += 5

            if save_plot:
                plt.savefig(os.path.join(folder_path, "{}.png".format(meas_id)))
                # plt.show()
                # exit()
        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(os.path.join(folder_path, "{}_set.csv".format(data_type)))
