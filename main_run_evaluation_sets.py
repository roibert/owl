import os

from src.evaluation.run_evaluation_set import run_evaluation


def _get_frame_folder(model_name):
    position_key = None

    if "pos1" in model_name:
        position_key = "pos1"

    if "pos2" in model_name:
        position_key = "pos2"

    if "pos3" in model_name:
        position_key = "pos3"

    return "%s%s\\" % (frame_folder_base_path, position_key)


def _get_models(scenario_base_path, scenario):
    path = scenario_base_path + scenario
    directory_contents = os.listdir(path)

    models = []

    for item in directory_contents:

        item = path + "\\" + item

        if os.path.isdir(item):
            models.append(item)

    return models


scenarios = [
    "full_train",
    "scenario2_gaps",
    "scenario3_no_extremes",
    "scenario4_inverted_extremes"
]

scenario_base_path = r"D:\robert\OneDrive - NTNU\internal\owl\processing_pipeline\models\waterlevel\round2\experiment2\\"
model_folder_base_path = r"D:\robert\OneDrive - NTNU\internal\owl\processing_pipeline\models"
frame_folder_base_path = r"D:\robert\toDelete\owl\second_round\experiment2\\"

for scenario in scenarios:

    model_names = _get_models(scenario_base_path, scenario)

    for model_name in model_names:
        frame_folder = _get_frame_folder(model_name)
        output_path = model_name + "_evaluation_set_result.csv"

        run_evaluation(output_path, model_name, frame_folder)

        print("Written: %s" % output_path)
