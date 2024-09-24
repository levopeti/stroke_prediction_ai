import argparse
import git
import sys
import subprocess
import yaml
import os

from datetime import datetime
from training.utils.clear_measurements import Limb


def show_versions():
    print("SYSTEM INFO")
    print("-----------")
    print("python:", sys.version)

    print("\nPYTHON DEPENDENCIES")
    print("-------------------")
    from pip._internal.operations import freeze
    for p in freeze.freeze():
        print(p)
    print("")


def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "git is not available"


def get_command() -> str:
    return " ".join(sys.argv)


def get_args() -> argparse.Namespace:
    repo = git.Repo(search_parent_directories=True)
    branch_name = "detached" if repo.head.is_detached else repo.active_branch.name

    if branch_name == "master":
        print(f"GIT HASH\n--------\n{get_git_hash()}\n")
        print(f"COMMAND\n-------\n{get_command()}\n")
        show_versions()

    parser = argparse.ArgumentParser(description='Stroke prediction AI training')
    parser.add_argument('--name', default="", type=str,
                        help='Model folder name is the date and the name parameter.')
    parser.add_argument('--invert_side', default=False, action='store_true', help='Invert the side to get the label.')
    parser.add_argument('--training_length_min', default=90, type=int,
                        help='Considered time for prediction in minutes.')
    parser.add_argument('--subsampling_factor', default=50, type=int,
                        help='Subsampling factor in down_sampling preprocess step.')
    parser.add_argument('--dst_frequency', default=25., type=float,
                        help='Modify frequency in change_frequency preprocess step.')
    parser.add_argument('--discord', default=False, action='store_true', help='Run discord webhook.')

    args = parser.parse_args()
    return args


def get_other_config() -> dict:
    other_config_dict = {
        # data info
        "accdb_path": "./data/WUS-v4measure20240116.accdb",
        "ucanaccess_path": "./ucanaccess/",
        "folder_path": "./data/clear_and_synchronized/",
        "clear_json_path": "./data/clear_train_val_ids.json",
        # "model_base_path": "./models/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M')),
        "model_checkpoint_folder_path": None,  # None

        # measurement info
        "base_frequency": 25,  # HZ
        # "training_length_min": 90,
        "step_size_min": 5,
        "limb": Limb.ARM,

        # model info
        "model_type": "inception_time",  # mlp, inception_time
        "input_shape": 6,  # 18 - features, 2 - acc, gyr
        "output_shape": 3,  # depends on the class mapping
        "layer_sizes": [1024, 512, 256],  # only for mlp

        # dataset
        # "invert_side": False,
        "class_mapping": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2},  # None, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
        "train_sample_per_meas": 10,
        "val_sample_per_meas": 500,  # 500
        "indexing_multiplier": 4,
        "indexing_mode": 1,
        "cache_size": 1,
        "steps_per_epoch": 100,  # 100, only if indexing mode == 0
        # "subsampling_factor": 50,  # 50

        # dataloader
        "train_batch_size": 100,  # 100
        "val_batch_size": 100,
        "num_workers": 6,

        # training
        "learning_rate": 0.0001,
        "wd": 0.001,
        "num_epoch": 1000,
        "stroke_loss_factor": 0.5,  # for stroke loss function
        "patience": 30,  # early stopping callback
        "device": "cuda",  # cpu, cuda
    }
    return other_config_dict


def add_model_path(config_dict: dict) -> dict:
    name = config_dict["name"]
    config_dict["model_base_path"] = "./models/{}_{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M'), name)
    return config_dict


def get_config_dict() -> dict:
    config_dict = dict()
    args = get_args()
    config_dict.update(vars(args))
    config_dict.update(get_other_config())
    config_dict = add_model_path(config_dict)
    assert config_dict["base_frequency"] >= config_dict["dst_frequency"]
    return config_dict


def save_config_dict(config_dict: dict, log_dir_path: str):
    os.makedirs(os.path.join(log_dir_path), exist_ok=True)

    with open(os.path.join(log_dir_path, "config.yaml"), "w") as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
