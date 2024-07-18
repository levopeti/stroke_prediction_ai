from tqdm import tqdm

from training.datasets.plot_dataset import PlotDataset
from training.utils.clear_measurements import ClearMeasurements
from training.utils.measure_db import MeasureDB
from utils.arg_parser_and_config import get_config_dict

params = get_config_dict()
params["train_sample_per_meas"] = 1
params["indexing_multiplier"] = 1
measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])

for dst_frequency in [25, 12.5, 8.3, 6.25, 5, 2.5, 1]:  # 25, 12.5, 8.3, 6.25, 5, 2.5, 1
    params["dst_frequency"] = dst_frequency
    params["subsampling_factor"] = int(dst_frequency * 2)
    print("subsampling_factor: {}".format(params["subsampling_factor"]))

    clear_measurements = ClearMeasurements(measDB, **params)
    train_dataset = PlotDataset("train",
                                clear_measurements,
                                params["limb"],
                                params,
                                params["train_batch_size"],
                                params["training_length_min"],
                                params["train_sample_per_meas"],
                                params["steps_per_epoch"],
                                params["base_frequency"],
                                params["subsampling_factor"],
                                params["indexing_multiplier"])

    # for i in tqdm(range(len(train_dataset))):
    for i in tqdm(range(1)):
        _, _ = train_dataset[i]

