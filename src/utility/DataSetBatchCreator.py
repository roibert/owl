from tensorflow.keras.utils import Sequence

from src.utility.utility import get_image_from_frame_name, load_image

import numpy as np


def _load_water_level_data(data_set, input_frame_path):

    water_level = data_set["Percentage Full [%%]"].tolist()
    images = data_set["Frame"].apply(get_image_from_frame_name, frame_folder=input_frame_path).tolist()

    water_level = np.array(water_level)
    images = np.array(images)

    return images, water_level


def _load_velocity_data(data_set, *args):

    velocity = data_set["Velocity"].tolist()
    images = data_set["Frame"].apply(load_image).tolist()

    velocity = np.array(velocity)
    images = np.array(images)

    return images, velocity


class _DataSetBatchCreator(Sequence):

    def __init__(self, data_set, batch_size, input_frame_path, data_load_function):
        self._data_set = data_set.reset_index()
        self._batch_size = batch_size
        self._input_frame_path = input_frame_path
        self._data_load_function = data_load_function

    def __len__(self):

        number_of_data_points = len(self._data_set)
        number_of_batches = number_of_data_points / float(self._batch_size)

        return (np.ceil(number_of_batches)).astype(np.int)

    def __getitem__(self, index):

        batch_size = self._batch_size

        batch = self._data_set[index * batch_size: (index + 1) * batch_size]

        input_batch, output_batch = self._data_load_function(batch, self._input_frame_path)
        return input_batch, output_batch


class VelocityDataSetBatchCreator(Sequence):

    def __init__(self, data_set, batch_size, input_frame_path):
        self._batch_creator = _DataSetBatchCreator(data_set, batch_size, input_frame_path, _load_velocity_data)

    def __len__(self):
        return self._batch_creator.__len__()

    def __getitem__(self, index):
        return self._batch_creator.__getitem__(index)


class WaterLevelDataSetBatchCreator(Sequence):

    def __init__(self, data_set, batch_size, input_frame_path):
        self._batch_creator = _DataSetBatchCreator(data_set, batch_size, input_frame_path, _load_water_level_data)

    def __len__(self):
        return self._batch_creator.__len__()

    def __getitem__(self, index):
        return self._batch_creator.__getitem__(index)

