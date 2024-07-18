from typing import Union


def min_to_ticks(time_min: int, frequency: Union[int, float]) -> int:
    # min -> sec -> sec * Hz
    num_of_tick = int(time_min * 60 * frequency)
    return num_of_tick


def sec_to_ticks(time_sec: int, frequency: Union[int, float]) -> int:
    # sec -> sec * Hz
    num_of_tick = int(time_sec * frequency)
    return num_of_tick


def ticks_to_min(num_of_tick: int, frequency: Union[int, float]) -> int:
    # n_o_t / Hz -> sec -> min
    time_min = int(num_of_tick / frequency / 60)
    return time_min


def ticks_to_h(num_of_tick: int, frequency: Union[int, float]) -> float:
    # n_o_t / Hz -> sec -> min -> h
    time_h = num_of_tick / frequency / 60 / 60
    return time_h


def frequency_to_timedelta_ms(frequency: Union[int, float]) -> int:
    # Hz -> sec -> msec
    return int(1000 / frequency)


