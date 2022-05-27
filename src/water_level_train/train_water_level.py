import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from src.model_configuration.ModelConfiguration import ModelConfiguration
from src.utility.DataSetBatchCreator import WaterLevelDataSetBatchCreator
from src.utility.utility import get_model_name, create_gap_in_set

np.random.seed(1291)


def _build_model(model_configuration=None):
    number_conv = model_configuration.get_conv()
    number_dense = model_configuration.get_dense()
    image_size_x = model_configuration.get_size_x()
    image_size_y = model_configuration.get_size_y()

    model = models.Sequential()

    if number_conv > 0:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size_x, image_size_y, 3)))
        model.add(layers.MaxPooling2D((2, 2)))

    for i in range(number_conv - 1):
        model.add(layers.Conv2D(2 ** (6 + i), (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    last_layer_size = 8
    first_layer_size_max = 64

    first_layer_size = last_layer_size * (2 ** number_dense)
    first_layer_size = min(first_layer_size_max, first_layer_size)

    current_layer_size = first_layer_size

    for i in range(number_dense):
        model.add(layers.Dense(current_layer_size, activation='relu'))
        current_layer_size = current_layer_size / 2

        if current_layer_size < 3:
            break

    model.add(layers.Dense(1))

    model.compile(optimizer='adam',
                  loss="mean_squared_error",
                  metrics=['accuracy']
                  )

    print("Network created.")

    model.summary()

    return model


def _train_water_level_model(model_configuration=None):
    model = _build_model(model_configuration=model_configuration)

    input_frame_path = model_configuration.get_frame_folder()

    # only use training data until now
    training_set_index_file = input_frame_path + r"\train_index.csv"
    training_set_information = pd.read_csv(training_set_index_file)

    frame_index_file = input_frame_path + r"\frame_index.csv"
    frame_index = pd.read_csv(frame_index_file)

    input_set = training_set_information.merge(frame_index, on="Timestamp_no_millis", sort=True)

    training_set, validation_set, test_set = \
        np.split(input_set.sample(frac=1, random_state=0),
                 [int(.6 * len(input_set)), int(.8 * len(input_set))])

    key = "Percentage Full [%%]"

    # Scenario 4
    training_set, test_set = create_gap_in_set(training_set, .62, .81, test_set, key)
    validation_set, test_set = create_gap_in_set(validation_set, .62, .81, test_set, key)

    batch_size = model_configuration.get_batch_size()

    training_set_generator = WaterLevelDataSetBatchCreator(training_set, batch_size, input_frame_path)
    validation_set_generator = WaterLevelDataSetBatchCreator(validation_set, batch_size, input_frame_path)

    print(model_configuration)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-5, restore_best_weights=True)

    history = model.fit(x=training_set_generator,
                        validation_data=validation_set_generator,
                        batch_size=batch_size,
                        validation_batch_size=batch_size,
                        callbacks=[early_stopping],
                        epochs=100)

    number_conv = model_configuration.get_conv()
    number_dense = model_configuration.get_dense()
    suffix = model_configuration.get_suffix()

    model_name = get_model_name(number_conv=number_conv, number_dense=number_dense, batch_size=batch_size,
                                model_suffix=suffix)

    model_output_path = model_configuration.get_model_output_folder()
    model_path = model_output_path + model_name
    model.save(model_path)

    test_set.to_csv(model_path + "_test_data.csv")

    training_input = None
    training_output = None

    validation_input = None
    validation_output = None

    return model, model_name, test_set


def _evaluate_test_data(model, model_name, model_configuration, test_information):
    input_frame_path = model_configuration.get_frame_folder()
    batch_size = model_configuration.get_batch_size()

    test_set_generator = WaterLevelDataSetBatchCreator(test_information, batch_size, input_frame_path)

    predictions = model.predict(test_set_generator)

    test_information["Predictions"] = predictions

    output_path = model_configuration.get_model_output_folder()
    test_evaluation_path = "%s%s_test_evaluation.csv" % (output_path, model_name)

    test_information.to_csv(test_evaluation_path)

def train_and_evaluate_model(configuration_path):
    model_configuration = ModelConfiguration(configuration_path)

    model, model_name, test_information = _train_water_level_model(model_configuration=model_configuration)
    _evaluate_test_data(model=model, model_name=model_name, model_configuration=model_configuration,
                        test_information=test_information)
