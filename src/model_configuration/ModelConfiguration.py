import configparser


class ModelConfiguration:

    KEY_CONVOLUTIONAL_LAYERS = "conv"
    KEY_DENSE = "dense"
    KEY_BATCH_SIZE = "batchSize"
    KEY_SUFFIX = "suffix"
    KEY_FRAME_FOLDER = "framesFolder"
    KEY_TRAINING_INDEX = "trainingIndex"
    KEY_MODEL_OUTPUT_FOLDER = "modelOutput"
    KEY_SIZE_X = "sizeX"
    KEY_SIZE_Y = "sizeY"

    def __init__(self, path):

        config = configparser.ConfigParser()
        config.read(path)

        model = config["model"]
        self._model = model

    def _get_attribute(self, attribute):
        return self._model[attribute]

    def get_conv(self):
        return int(self._get_attribute(ModelConfiguration.KEY_CONVOLUTIONAL_LAYERS))

    def get_dense(self):
        return int(self._get_attribute(ModelConfiguration.KEY_DENSE))

    def get_batch_size(self):
        return int(self._get_attribute(ModelConfiguration.KEY_BATCH_SIZE))

    def get_suffix(self):
        return str(self._get_attribute(ModelConfiguration.KEY_SUFFIX))

    def get_frame_folder(self):
        return str(self._get_attribute(ModelConfiguration.KEY_FRAME_FOLDER))

    def get_training_index(self):
        return str(self._get_attribute(ModelConfiguration.KEY_TRAINING_INDEX))

    def get_model_output_folder(self):
        return str(self._get_attribute(ModelConfiguration.KEY_MODEL_OUTPUT_FOLDER))

    def get_size_x(self):
        return int(self._get_attribute(ModelConfiguration.KEY_SIZE_X))

    def get_size_y(self):
        return int(self._get_attribute(ModelConfiguration.KEY_SIZE_Y))

    def _pretty_print_attribute(self, attribute):
        return "%s:\t%s" % (attribute, self._get_attribute(attribute))

    def __str__(self):

        result = "ModelConfiguration:\n"

        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_CONVOLUTIONAL_LAYERS) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_DENSE) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_BATCH_SIZE) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_SUFFIX) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_FRAME_FOLDER) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_TRAINING_INDEX) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_MODEL_OUTPUT_FOLDER) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_SIZE_X) + "\n"
        result += "\t" + self._pretty_print_attribute(ModelConfiguration.KEY_SIZE_Y) + "\n"

        return result
