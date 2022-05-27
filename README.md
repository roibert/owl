# OWL
**O**ptical **W**ater **L**evel: OWL

This collection of scripts allows to 

1. Measurement conversions (jupyter script)
2. Convert video files to single frames 
3. Stacking of frames for velocity prediction (jupyter) 
4. Test set creation (jupyter)
5. Create scenario configurations
6. Train water level networks
7. Train surface velocity networks 
8. Test set evaluation 
9. Evaluate network performance (jupyter) 

See sections below for more details.

**This is a reference and not meant to be usable out of the box. Numerous paths have to be adapted, while the conversion
scripts are very sensor specific and might have to be completely re-written. The data analysis fit the original experiments
and might not be sensible in other contexts.**

### Measurement Conversion
The jupyter scripts `processing_water_level_measurements.ipynb` and `processing_surface_velocity_measurements.ipynb`
 are used to pre-process the sensor measurements. This is very specific to the Vectrino sensors and most likely has to be 
adjusted for a different kind of sensor. 

These scripts are also meant to understand the data to a certain degree, which is why both contain a great number of 
additional details and analyses.

### Video Conversion 
The script `main_video_conversion.py` wraps the functionality to convert video files, where the paths to input
and output folders can be specified.

### Frame Stacking
To train the surface velocity prediction, consecutive frames need to be stacked. This is done in the script 
`velocity_image_stack.ipynb`. 

### Test Set Creation
Test sets are created using the script `train_test_validation_set_creation.ipynb`.

### Scenario Configurations
The script `main_scenario_creation.py` wraps the creation of the scenario configurations. A scenario configuration tells 
the training script (see below) how to construct the network (frame size, layer information, etc.), where to find the 
training data and where to output the results. 

A script `run.bat` (for Windows...) will be created, which allows all scenarios to be executed consecutively. 

For this to work properly, the following paths have to be defined:

* `base_path` 
  * Base path where the training data is stored
* `configuration_path`
  * Path to store the configurations
* `model_output_path`
  * Where should the trained model be stored?
  

### Train Water Level Networks
The script `main_water_level_train.py` is used to simplify the batch execution of all scenarios created in the 
earlier step. The main training is handled by script `src/water_level_train/train_water_level.py`.


### Train Surface Velocity Networks 
The script `main_velocity_train.py.py` is used to simplify the batch execution of all scenarios created in the 
earlier step. The main training is handled by script `src/surface_velocity_train/train_surface_velocity.py`.

### Test Set Evaluation 
The script `main_run_evaluation_sets.py` is a wrapper for `src/evaluation/run_evaluation.py` which evaluates the test 
set on the trained model. Several paths can be specified to enable batch evaluation:

* scenario_base_path 
  * Base path where test sets are stored
* scenarios
  * Can be used as a prefix for the scenarios
* model_folder_base_path
  * Where are the trained models stored?
* frame_folder_base_path
  * Where are the frames stored?

### Network Performance Evaluation
The main model evaluation can be done using scripts `waterlevel_evaluation_set.ipynb` and 
`velocity_evaluation_set.ipynb`. Both scripts contain a great number of additional details to understand the 
network performance. 
