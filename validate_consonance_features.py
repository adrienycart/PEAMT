import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import features.utils as utils

import warnings
warnings.filterwarnings("ignore")

feature_folder = "features/consonance_statistics_no_empty_chords"
systems = ["kelz", "lisu", "google", "cheng", "target"]
models = ["hutch_78_roughness", "har_18_harmonicity", "har_19_corpus"]
statistics = ["weighted_mean", "weighted_std", "max", "min"]
values = {"kelz": [], "lisu": [], "google": [], "cheng": [], "target": []}

for example in os.listdir(feature_folder)[:]:
    example_path = os.path.join(feature_folder, example)

    for system in systems:
        # get consonance feature values
        with open(example_path+"/"+system+".pkl", 'rb') as f:
            data = pickle.load(f)
        values[system].append(data)

for system in systems:
    values[system] = np.array(values[system])

def plot_hist(data, system):

    plt.figure(figsize=(16,12))
    for i in range(len(models)*len(statistics)):
        plt.subplot(4, 3, i+1)
        plt.hist(data[:,i], bins=50)
        if i < 3:
            plt.title(models[i])
        plt.ylabel(statistics[i // 3])

    plt.show()

for system in systems:
    plot_hist(values[system], system)