from src.surface_velocity_train.scenario_creation import create_sv_scenario_configurations
from src.water_level_train.scenario_creation import create_scenario_wl_configurations


velocity_base_path = "TO DEFINE"             # Where is the training data stored?
velocity_configuration_path = "TO DEFINE"    # Where can the configurations be stored?
velocity_model_output_path = "TO DEFINE"     # Where should the trained model be stored?

create_sv_scenario_configurations(velocity_base_path, velocity_configuration_path, velocity_model_output_path)

water_level_base_path = "TO DEFINE"             # Where is the training data stored?
water_level_configuration_path = "TO DEFINE"    # Where can the configurations be stored?
water_level_model_output_path = "TO DEFINE"     # Where should the trained model be stored?

create_scenario_wl_configurations(water_level_base_path, water_level_configuration_path,
                                  water_level_model_output_path)
