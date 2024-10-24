import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = [15, 45]

from evaluation.six_model_evals import get_model
from imputation.imputation_loss_and_acc import ImputedLoss, NonImputedLoss
from training.datasets.get_dataset import get_dataset
from training.utils.clear_measurements import Limb


def get_params(base_folder):
    with open(os.path.join(base_folder, "params.json")) as f:
        params = json.load(f)

    params["class_mapping"] = {int(k): v for k, v in params["class_mapping"].items()}
    if params["limb"] == "arm":
        params["limb"] = Limb.ARM
    else:
        raise NotImplementedError
    return params

def plot_masked_meas(i, predictions, y, mask, imp_loss_f, non_imp_loss_f):
    def add_vlines(ax, min_value, max_value):
        vlines = np.argwhere(np.diff(mask) != 0)
        for vline in vlines:
            ax.vlines(vline, ymin=min_value, ymax=max_value, colors="r")
    imp_loss = imp_loss_f(predictions, y, mask)
    non_imp_loss = non_imp_loss_f(predictions, y, mask)

    fig, axs = plt.subplots(5, 1, facecolor="w")
    fig.suptitle("imp_loss: {:.05f}, non_imp_loss: {:.05f}".format(imp_loss, non_imp_loss))
    axs[0].plot(mask.T)
    axs[0].legend(['mask'])
    axs[0].grid(True)

    axs[1].plot(predictions[0])
    axs[1].plot(y[0])
    axs[1].legend(['prediction', 'y'])
    axs[1].grid(True)

    axs[2].plot(predictions[0] - y[0])
    axs[2].legend(['diff'])
    axs[2].grid(True)

    axs[3].plot(predictions[1])
    axs[3].plot(y[1])
    axs[3].legend(['prediction', 'y'])
    axs[3].grid(True)

    axs[4].plot(predictions[1] - y[1])
    axs[4].legend(['diff'])
    axs[4].grid(True)

    for ax_i in range(1, 5):
        add_vlines(axs[ax_i], -0.05, 0.05)

    plt.savefig("./{}.jpg".format(i))
    plt.close()


def imputation_visualization(base_folder):
    print(base_folder)
    params = get_params(base_folder)

    # base_folder = "./models/frequency_test/2024-05-21-16-13_non-inverted_90_arm_cf_25"
    model_path = os.path.join(_base_folder, "{}.pt".format(os.path.basename(_base_folder)))
    model = get_model(model_path, to_cuda=False)


    params["indexing_multiplier"] = 1
    params["train_sample_per_meas"] = 1
    train_dataset, val_dataset = get_dataset(params)

    imp_loss = ImputedLoss()
    non_imp_loss = NonImputedLoss()

    with torch.no_grad():
        for i, (x, y, mask) in enumerate(train_dataset):
            print(x.unsqueeze(0).shape, mask.unsqueeze(0).shape)
            predictions = model(x.unsqueeze(0), mask.unsqueeze(0))
            plot_masked_meas(i, predictions.squeeze(), y, mask, imp_loss, non_imp_loss)
            exit()

        for i in range(len(val_dataset)):
            pass

if __name__ == "__main__":
    """ python -m imputation.vis_imputation """

    _base_folder = "./models/2024-10-22-18-02_imp_version_2_resop_no_bias"
    imputation_visualization(_base_folder)