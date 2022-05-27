import configparser

from src.model_configuration.ModelConfiguration import ModelConfiguration
from src.surface_velocity_train.configuration import batch_size, exp1_size_x, exp2_size_x, exp3_size_x, exp1_size_y, \
    exp2_size_y, exp3_size_y


def create_sv_scenario_configurations(base_path, configurations_folder, model_output_path):
    all_commands = list()

    commands = _create_scenario_configurations(r"%s\pos1" % base_path, "_v_pos1", configurations_folder,
                                               model_output_path, exp1_size_x, exp1_size_y)
    all_commands.append(commands)

    commands = _create_scenario_configurations(r"%s\pos2" % base_path, "_v_pos2", configurations_folder,
                                               model_output_path, exp2_size_x, exp2_size_y)
    all_commands.append(commands)

    commands = _create_scenario_configurations(r"%s\pos3" % base_path, "_v_pos3", configurations_folder,
                                               model_output_path, exp3_size_x, exp3_size_y)
    all_commands.append(commands)

    flattened_command_list = list()

    for command_list in all_commands:
        for command in command_list:
            flattened_command_list.append(command)

    with open(r"%s/run.bat" % configurations_folder, "w") as run:

        run.writelines(flattened_command_list)


def _create_scenario_configurations(frame_input, suffix, path, model_output_path, size_x, size_y):
    max_number_conf = 5
    max_number_dense = 5

    run_commands = list()

    for conv in range(2, max_number_conf + 1):
        for dense in range(1, max_number_dense + 1):
            configuration = configparser.ConfigParser()
            configuration.add_section("model")

            configuration.set("model", ModelConfiguration.KEY_CONVOLUTIONAL_LAYERS, str(conv))
            configuration.set("model", ModelConfiguration.KEY_DENSE, str(dense))
            configuration.set("model", ModelConfiguration.KEY_BATCH_SIZE, str(batch_size))
            configuration.set("model", ModelConfiguration.KEY_SUFFIX, str(suffix))
            configuration.set("model", ModelConfiguration.KEY_FRAME_FOLDER, str(frame_input))
            configuration.set("model", ModelConfiguration.KEY_TRAINING_INDEX, "stacked_index.csv")
            configuration.set("model", ModelConfiguration.KEY_MODEL_OUTPUT_FOLDER, str(model_output_path))
            configuration.set("model", ModelConfiguration.KEY_SIZE_X, str(size_x))
            configuration.set("model", ModelConfiguration.KEY_SIZE_Y, str(size_y))

            name = r"%s/c%d_d%d_%s.conf" % (path, conv, dense, suffix)

            path_end_index = name.index("configuration")
            run_command = "python main_velocity_train.py %s\n" % name[path_end_index:]
            run_commands.append(run_command)

            with open(name, "w") as file:
                configuration.write(file)

    return run_commands
