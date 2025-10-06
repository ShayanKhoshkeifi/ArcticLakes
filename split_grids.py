import config
import json
import random


def split_grids_random(experiment_id, train_percent = 60, val_percent = 20, water_coverage_threshold = 0.01):

    # train_percent: percentage of grids used for training
    # val_percent: percentage of grids used for validation (remaining go to test)
    # water_coverage_threshold = proportion of pixels needed to be water to use a grid

    with open('sorted_water_coverage.json', 'r', encoding='utf-8') as file:
        sorted_water_coverage = json.load(file)

    water_grids = [key for key, value in sorted_water_coverage.items() if value > water_coverage_threshold]

    water_grids_shuffled = random.sample(water_grids, len(water_grids))

    train_size = len(water_grids)*train_percent//100
    val_size = len(water_grids)*val_percent//100

    train_grids = water_grids_shuffled[:train_size]
    val_grids = water_grids_shuffled[train_size:(train_size+val_size)]
    test_grids = water_grids_shuffled[train_size+val_size:]

    
    
    with open(f"experiments/{experiment_id}/train_grids.txt", "w") as file:
        file.write("\n".join(train_grids)) 
    with open(f"experiments/{experiment_id}/val_grids.txt", "w") as file:
        file.write("\n".join(val_grids)) 
    with open(f"experiments/{experiment_id}/test_grids.txt", "w") as file:
        file.write("\n".join(test_grids)) 

# Can add new functions for splitting grids according to non-random rules

if __name__ == "__main__":
    split_grids_random(config.experiment_id, config.train_percent, config.val_percent)