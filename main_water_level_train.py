import sys
from src.water_level_train.train_water_level import train_and_evaluate_model


path = sys.argv[1]
train_and_evaluate_model(path)


