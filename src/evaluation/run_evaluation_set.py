import pandas as pd
import tensorflow as tf

from src.utility.DataSetBatchCreator import WaterLevelDataSetBatchCreator
from src.water_level_train.configuration import batch_size


def _load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model


def _load_evaluation_set(frame_folder):
    evaluation_set_index_file = frame_folder + r"\test_index.csv"
    evaluation_set_information = pd.read_csv(evaluation_set_index_file)

    frame_index_file = frame_folder + r"\frame_index.csv"
    frame_index = pd.read_csv(frame_index_file)

    evaluation_set = evaluation_set_information.merge(frame_index, on="Timestamp_no_millis", sort=True)

    return evaluation_set


def _evaluate_and_store_set(model, frame_folder, evaluation_set):
    test_set_generator = WaterLevelDataSetBatchCreator(evaluation_set, batch_size, frame_folder)

    predictions = model.predict(test_set_generator)

    evaluation_set["Predictions"] = predictions

    return evaluation_set


def _store_evaluation_result(output_path, result):
    result.to_csv(output_path)


def run_evaluation(output_path, model_name, frame_folder):
    model = _load_model(model_name)
    evaluation_set = _load_evaluation_set(frame_folder)

    evaluation_result = _evaluate_and_store_set(model, frame_folder, evaluation_set)
    _store_evaluation_result(output_path, evaluation_result)
