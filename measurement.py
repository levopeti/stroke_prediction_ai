import numpy as np

from typing import Tuple
from training.utils.converting_utils import frequency_to_timedelta_ms, min_to_ticks


class MeasurementInfo(object):
    def __init__(self,
                 meas_id: int,
                 timestamps: np.ndarray,
                 frequency: int,
                 timedelta_eps_ms: float = 1):
        self.meas_id = meas_id
        self.timestamps = timestamps
        self.frequency = frequency
        avg_timedelta = np.mean(np.diff(timestamps))
        expected_timedelta = frequency_to_timedelta_ms(frequency)
        assert expected_timedelta - timedelta_eps_ms < avg_timedelta < expected_timedelta + timedelta_eps_ms, avg_timedelta

    def get_number_of_samples(self, training_length_min: int, step_size_min: int) -> int:
        training_length_ticks = min_to_ticks(training_length_min, self.frequency)
        assert training_length_ticks <= len(self.timestamps), (training_length_ticks, len(self.timestamps))
        step_size_ticks = min_to_ticks(step_size_min, self.frequency)

        num_of_samples = int((len(self.timestamps) - training_length_ticks) / step_size_ticks) + 1
        return num_of_samples

    def get_sample_start_end_index(self,
                                   sample_index: int,
                                   training_length_min: int,
                                   step_size_min: int) -> Tuple[int, int]:
        start_idx = sample_index * step_size_min
        end_idx = start_idx + training_length_min
        assert end_idx < len(self.timestamps), (end_idx, len(self.timestamps))
        return start_idx, end_idx

    def get_length(self) -> int:
        return len(self.timestamps)

