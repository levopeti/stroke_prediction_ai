import json
import os
import random

import numpy as np
import pandas as pd

from glob import glob
from enum import Enum
from typing import Union, Tuple

from measurement_utils.measure_db import MeasureDB


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class Limb(Enum):
    ARM = "arm"
    LEG = "leg"


class MeasType(Enum):
    ACC = "acc"
    GYR = "gyr"


def get_inverted_side(side: Side):
    return Side.RIGHT if side == Side.LEFT else Side.LEFT


Key = Tuple[Side, Limb, MeasType]
all_key_combination = [(side, limb, meas_type) for side in Side for limb in Limb for meas_type in MeasType]


class ClearMeasurements(object):
    def __init__(self,
                 measDB: MeasureDB,
                 folder_path: str,
                 clear_json_path: str,
                 class_mapping: dict,
                 invert_side: bool,
                 cache_size: int = 1,
                 **kwargs) -> None:
        assert cache_size > 0, "cache_size must be positive integer"
        self.measDB = measDB
        self.class_mapping = class_mapping
        self.invert_side = invert_side
        self.cache_size = cache_size
        self.id_path_dict = dict()
        self.cache_dict = dict()
        self.clear_ids_dict = dict()
        self.all_meas_ids = list()

        self.current_meas_key = None
        self.current_df = None

        self.read_csv_path(folder_path)
        self.read_clear_json(clear_json_path)

        # limb_values_dict[type_of_set][limb][class_value] = [(meas_id, side), ...]
        self.limb_values_dict = self.collect_limb_values(print_stat=True)

    def get_meas_id_list(self, data_type: str) -> list:
        return sorted(self.clear_ids_dict[data_type])

    def get_class_value_dict(self, meas_id: int) -> dict:
        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        return class_value_dict

    def get_min_class_value(self, meas_id: int) -> int:
        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        min_class_value = min(class_value_dict.values())
        if self.class_mapping is not None:
            min_class_value = self.class_mapping[min_class_value]
        return min_class_value

    def get_limb_class_value(self, meas_id: int, side: Side, limb: Limb) -> int:
        if self.invert_side:
            side = get_inverted_side(side)
        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        limb_class_value = class_value_dict[(side.value, limb.value)]
        if self.class_mapping is not None:
            limb_class_value = self.class_mapping[limb_class_value]
        return limb_class_value

    def read_clear_json(self, clear_json_path: str) -> None:
        with open(clear_json_path, "r") as read_file:
            self.clear_ids_dict = json.load(read_file)

        self.all_meas_ids = sorted(set(self.id_path_dict.keys()))
        for meas_id in self.clear_ids_dict["train"]:
            assert meas_id in self.all_meas_ids, meas_id

        for meas_id in self.clear_ids_dict["validation"]:
            assert meas_id in self.all_meas_ids, meas_id

    def read_csv_path(self, folder_path: str) -> None:
        for csv_path in sorted(glob(os.path.join(folder_path, "*.csv"))):
            file_name = os.path.basename(csv_path)
            meas_id = file_name.split("-")[0]
            self.id_path_dict[int(meas_id)] = csv_path

    def drop_random_from_cache_dict(self) -> None:
        self.cache_dict.pop(random.choice(list(self.cache_dict.keys())))

    @staticmethod
    def read_csv(csv_path: str,
                 side: Union[Side, Tuple[Side]],
                 limb: Union[Limb, Tuple[Limb]],
                 meas_type: Union[MeasType, Tuple[MeasType]]) -> pd.DataFrame:
        if isinstance(side, tuple):
            side_list = [inner_side.value for inner_side in side]
        else:
            side_list = [side.value]

        if isinstance(limb, tuple):
            limb_list = [inner_limb.value for inner_limb in limb]
        else:
            limb_list = [limb.value]

        if isinstance(meas_type, tuple):
            meas_types = [inner_meas_type.value for inner_meas_type in meas_type]
        else:
            meas_types = [meas_type.value]

        usecols = ["epoch"]
        dtype_dict = {"epoch": np.int64}
        for side in side_list:
            for limb in limb_list:
                for meas_type in meas_types:
                    for axis in ["x", "y", "z"]:
                        # epoch, "('left', 'arm', 'acc', 'x')", ...
                        column_name = str((side, limb, meas_type, axis))
                        usecols.append(column_name)
                        dtype_dict[column_name] = np.float32
        return pd.read_csv(csv_path, usecols=usecols, dtype=dtype_dict)

    def get_measurement(self,
                        meas_id: int,
                        side: Union[Side, Tuple[Side]] = (Side.LEFT, Side.RIGHT),
                        limb: Union[Limb, Tuple[Limb]] = (Limb.ARM, Limb.LEG),
                        meas_type: Union[MeasType, Tuple[MeasType]] = (MeasType.ACC, MeasType.GYR)) -> pd.DataFrame:
        csv_path = self.id_path_dict[meas_id]
        if self.cache_size == 1:
            if (meas_id, side, limb, meas_type) == self.current_meas_key:
                df = self.current_df
            else:
                df = self.read_csv(csv_path, side, limb, meas_type)
                self.current_df = df
                self.current_meas_key = (meas_id, side, limb, meas_type)
        else:
            raise NotImplemented
            if meas_id in self.cache_dict:
                df = self.cache_dict[meas_id]
            else:
                while len(self.cache_dict) >= self.cache_size:
                    self.drop_random_from_cache_dict()
                df = pd.read_csv(csv_path)
                self.cache_dict[meas_id] = df
                # assert len(self.cache_dict) <= self.cache_size, (len(self.cache_dict), self.cache_size)
                # if len(self.cache_dict) > self.cache_size:
                #     print("Number of cached measurements ({}) is more than the cache size ({})".format(len(self.cache_dict), self.cache_size))
        return df

    def collect_healthy_ids(self, data_type) -> list:
        healthy_ids = list()
        ids = self.all_meas_ids if data_type == "all" else self.clear_ids_dict[data_type]
        for meas_id in ids:
            min_class_value = self.get_min_class_value(meas_id)

            if min_class_value == 5:
                healthy_ids.append(meas_id)
        return healthy_ids

    def collect_stroke_ids(self, data_type) -> list:
        stroke_ids = list()
        ids = self.all_meas_ids if data_type == "all" else self.clear_ids_dict[data_type]
        for meas_id in ids:
            min_class_value = self.get_min_class_value(meas_id)

            if min_class_value < 5:
                stroke_ids.append(meas_id)
        return stroke_ids

    def collect_limb_values(self, print_stat=False) -> dict:
        num_of_classes = len(set(self.class_mapping.values())) if self.class_mapping is not None else 6
        limb_values_dict = dict()
        for type_of_set, id_list in self.clear_ids_dict.items():
            limb_values_dict[type_of_set] = {Limb.ARM: {class_value: list() for class_value in range(num_of_classes)},
                                             Limb.LEG: {class_value: list() for class_value in range(num_of_classes)}}
            for meas_id in id_list:
                for limb in Limb:
                    for side in Side:
                        class_value = self.get_limb_class_value(meas_id, side, limb)
                        limb_values_dict[type_of_set][limb][class_value].append((meas_id, side))

        if print_stat:
            for type_of_set, limb_dict in limb_values_dict.items():
                print("\n", type_of_set)
                for limb, class_value_dict in limb_dict.items():
                    total = sum([len(inside_list) for inside_list in class_value_dict.values()])
                    print(limb)
                    for class_value in range(num_of_classes):
                        print("{}: {} {:.1f}%".format(class_value,
                                                      len(class_value_dict[class_value]),
                                                      100 * len(class_value_dict[class_value]) / total))
        return limb_values_dict

    def print_stat(self) -> None:
        stat_dict = dict()
        for type_of_set, id_list in self.clear_ids_dict.items():
            stat_dict[type_of_set] = {class_value: 0 for class_value in range(6)}

            for meas_id in id_list:
                min_class_value = self.get_min_class_value(meas_id)
                stat_dict[type_of_set][min_class_value] += 1

        for type_of_set, class_value_dict in stat_dict.items():
            total = sum(class_value_dict.values())
            print("\n", type_of_set)
            for class_value in range(6):
                print("{}: {} {:.1f}%".format(class_value,
                                              class_value_dict[class_value],
                                              100 * class_value_dict[class_value] / total))
