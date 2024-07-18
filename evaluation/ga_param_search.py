import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from pprint import pprint

from evaluation.six_models_filtering import get_metrics, get_model_dict
from training.utils.clear_measurements import Limb, ClearMeasurements
from training.utils.measure_db import MeasureDB

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
_model_dict = get_model_dict(model_folder_path)

# max_window_length_sec = 90 * 60
# window_length = int(max_window_length_sec / params["step_size_sec"])

metrics_weights = [1, 1, 4, 1]  # accuracy, sensitivity, specificity, precision


def f(x):
    _avg_prob_threshold_dict = {"non-inverted": {30: x[0], 60: x[1], 90: x[2]},
                                "inverted": {30: x[3], 60: x[4], 90: x[5]}}
    _window_length_dict = {"non-inverted": {30: x[6] * 60, 60: x[7] * 60, 90: x[8] * 60},
                           "inverted": {30: x[9] * 60, 60: x[10] * 60, 90: x[11] * 60}}
    metrics_dict = get_metrics("validation",  # validation
                               _avg_prob_threshold_dict,
                               _model_dict,
                               clear_measurements,
                               _window_length_dict,
                               params,
                               write_metrics=False)

    fitness_value_to_max = 0
    for accuracy, sensitivity, specificity, precision in metrics_dict.values():
        fitness_value_to_max += (metrics_weights[0] * accuracy + metrics_weights[1] * sensitivity +
                                 metrics_weights[2] * specificity + metrics_weights[3] * precision)
    return sum(metrics_weights * 2) - fitness_value_to_max


algorithm_param = {'max_num_iteration': 1000,
                   'population_size': 100,
                   'mutation_probability': 0.2,
                   'elit_ratio': 0.05,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': 80}

model = ga(function=f,
           dimension=12,
           variable_type='real',
           variable_boundaries=np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                                         [30, 180], [30, 180], [30, 180], [30, 180], [30, 180], [30, 180]]),
           algorithm_parameters=algorithm_param)

model.run()

print(model.report)
print(model.output_dict)
x = model.output_dict["variable"]

avg_prob_threshold_dict = {"non-inverted": {30: x[0], 60: x[1], 90: x[2]},
                           "inverted": {30: x[3], 60: x[4], 90: x[5]}}
window_length_dict = {"non-inverted": {30: x[6] * 60, 60: x[7] * 60, 90: x[8] * 60},
                      "inverted": {30: x[9] * 60, 60: x[10] * 60, 90: x[11] * 60}}
metrics_dict = get_metrics("validation",
                           avg_prob_threshold_dict,
                           _model_dict,
                           clear_measurements,
                           window_length_dict,
                           params,
                           write_metrics=False)
pprint(avg_prob_threshold_dict)
pprint(window_length_dict)
print("non-inverted\naccuracy: {:.2f}\nsensitivity: {:.2f}\nspecificity: {:.2f}\nprecision: {:.2f}\n".format(
    *metrics_dict["non-inverted"]))
print("inverted\naccuracy: {:.2f}\nsensitivity: {:.2f}\nspecificity: {:.2f}\nprecision: {:.2f}\n".format(
    *metrics_dict["inverted"]))
print(metrics_weights)
