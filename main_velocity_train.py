import sys

from src.surface_velocity_train.train_surface_velocity import train_and_evaluate_velocity_model

path = sys.argv[1]

print(path)
train_and_evaluate_velocity_model(path)


