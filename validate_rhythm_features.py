import os
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm
import mir_eval
import peamt.features.utils as utils
from peamt.features.rhythm import rhythm_histogram, rhythm_dispersion
from peamt.features.utils import get_time, str_to_bar_beat

import warnings
warnings.filterwarnings("ignore")

result_folder = "validate_rhythm_feature_plots_update"
utils.create_folder(result_folder)
MIDI_path = "app/static/data/all_midi_cut"
cut_points_path = "app/static/data/cut_points"
all_midi_path = "app/static/data/A-MAPS_1.2_with_pedal"
systems = ["kelz", "lisu", "google", "cheng"]
fs = 100

N_features = 8
sfo, sfd, stdmean, stdmin, stdmax, drmean, drmin, drmax = range(N_features)
N_computes = 5   # calculate quantize_over_original and noisy_over_original
noise_level = [0.1, 0.2, 0.3]
strict_quantize, quantize, noisy1, noisy2, noisy3 = range(N_computes)
all_MIDI = [elt for elt in os.listdir(MIDI_path) if not elt.startswith('.')]
N_outputs = len(all_MIDI) * len(systems)

# get cut points
cut_points_dict = dict()
for filename in os.listdir(cut_points_path):
    musicname = filename[:-4]
    cut_points_dict[musicname] = np.genfromtxt(os.path.join(cut_points_path, filename), dtype='str')

def plot_hist(x1, x2, x3, x4, x5, title, limits, filename, n_bins=50):

    plt.figure(figsize=(6.4, 8.2))
    plt.subplot(511)
    plt.hist(x1, bins=n_bins, range=limits)
    plt.ylabel("strict quantize/original")
    plt.title(title)
    plt.subplot(512)
    plt.hist(x2, bins=n_bins, range=limits)
    plt.ylabel("quantize/original")
    plt.subplot(513)
    plt.hist(x3, bins=n_bins, range=limits)
    plt.ylabel("noisy({:.1f})/original".format(noise_level[0]))
    plt.subplot(514)
    plt.hist(x4, bins=n_bins, range=limits)
    plt.ylabel("noisy({:.1f})/original".format(noise_level[1]))
    plt.subplot(515)
    plt.hist(x5, bins=n_bins, range=limits)
    plt.ylabel("noisy({:.1f})/original".format(noise_level[2]))
    plt.savefig(filename)
    # plt.show()

def add_noise(intervals,noise_level):
    return intervals + np.random.uniform(-noise_level,noise_level,size = [intervals.shape[0],1])

def print_line(values, feature_name, feature_index):
    print(feature_name+"\t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f}".format(np.mean(values[feature_index, strict_quantize]), np.std(values[feature_index, strict_quantize]), np.mean(values[feature_index, quantize]), np.std(values[feature_index, quantize]), np.mean(values[feature_index, noisy1]), np.std(values[feature_index, noisy1]), np.mean(values[feature_index, noisy2]), np.std(values[feature_index, noisy2]), np.mean(values[feature_index, noisy3]), np.std(values[feature_index, noisy3])))

if os.path.exists(result_folder+"/data.npy"):
    values = np.load(result_folder+"/data.npy")
else:
    values = np.zeros((N_features, N_computes, N_outputs), dtype=float)

    idx = 0
    for example in all_MIDI:
        if example.startswith('.'):
            continue

        example_path = os.path.join(MIDI_path, example)
        print((idx / len(systems), example_path))

        # get quantized quarter times
        musicname = example[len("MAPS_MUS-"):example.index("_ENS")]
        cut_index = int(example.split("_")[-1])
        (start_str, end_str) = cut_points_dict[musicname][cut_index]

        start_bar,start_beat,start_sub_beat = str_to_bar_beat(start_str)
        end_bar,end_beat,end_sub_beat = str_to_bar_beat(end_str)

        original_midi_data = pm.PrettyMIDI(os.path.join(all_midi_path, example[:(-2-int(cut_index>=10))]+'.mid'))
        original_PPQ = original_midi_data.resolution
        end_tick = original_midi_data.time_to_tick(original_midi_data.get_end_time())
        ticks = np.arange(0, end_tick, original_PPQ/4)
        original_quarter_times = np.array([original_midi_data.tick_to_time(t) for t in ticks])

        start_t = get_time(original_midi_data,start_bar,start_beat,start_sub_beat)
        end_t = get_time(original_midi_data,end_bar,end_beat,end_sub_beat)
        quarter_times = np.array([time-start_t for time in original_quarter_times if time >= start_t and time <= end_t])

        # get strict quantized intervals
        tempo = (quarter_times[-1] - quarter_times[0]) / (len(quarter_times) - 1)
        quarter_times_strict = np.array([quarter_times[0] + i * tempo for i in range(len(quarter_times))])

        for system in systems:
            system_data = pm.PrettyMIDI(os.path.join(example_path, system+".mid"))
            notes_system, intervals_system = utils.get_notes_intervals(system_data)

            # get quantized intervals
            intervals_system_quantized = intervals_system.copy()
            for i in range(len(intervals_system)):
                intervals_system_quantized[i][0] = quarter_times[np.argmin(np.abs(quarter_times - intervals_system[i][0]))]

            # get strict quantized intervals
            intervals_system_strict_quantized = intervals_system.copy()
            for i in range(len(intervals_system)):
                intervals_system_strict_quantized[i][0] = quarter_times_strict[np.argmin(np.abs(quarter_times_strict - intervals_system[i][0]))]

            # check quantization
            # plt.figure()
            # plt.plot([x[0] for x in intervals_system_quantized])
            # plt.plot([x[0] for x in intervals_system])
            # plt.show()

            values[sfo, strict_quantize, idx], values[sfd, strict_quantize, idx] = rhythm_histogram(intervals_system_strict_quantized, intervals_system)
            values[sfo, quantize, idx], values[sfd, quantize, idx] = rhythm_histogram(intervals_system_quantized, intervals_system)
            values[sfo, noisy1, idx], values[sfd, noisy1, idx] = rhythm_histogram(add_noise(intervals_system,noise_level[0]), intervals_system)
            values[sfo, noisy2, idx], values[sfd, noisy2, idx] = rhythm_histogram(add_noise(intervals_system,noise_level[1]), intervals_system)
            values[sfo, noisy3, idx], values[sfd, noisy3, idx] = rhythm_histogram(add_noise(intervals_system,noise_level[2]), intervals_system)

            [values[stdmean, strict_quantize, idx], values[stdmin, strict_quantize, idx], values[stdmax, strict_quantize, idx]], [values[drmean, strict_quantize, idx], values[drmin, strict_quantize, idx], values[drmax, strict_quantize, idx]] = rhythm_dispersion(intervals_system_strict_quantized, intervals_system)
            [values[stdmean, quantize, idx], values[stdmin, quantize, idx], values[stdmax, quantize, idx]], [values[drmean, quantize, idx], values[drmin, quantize, idx], values[drmax, quantize, idx]] = rhythm_dispersion(intervals_system_quantized, intervals_system)
            [values[stdmean, noisy1, idx], values[stdmin, noisy1, idx], values[stdmax, noisy1, idx]], [values[drmean, noisy1, idx], values[drmin, noisy1, idx], values[drmax, noisy1, idx]] = rhythm_dispersion(add_noise(intervals_system,noise_level[0]), intervals_system)
            [values[stdmean, noisy2, idx], values[stdmin, noisy2, idx], values[stdmax, noisy2, idx]], [values[drmean, noisy2, idx], values[drmin, noisy2, idx], values[drmax, noisy2, idx]] = rhythm_dispersion(add_noise(intervals_system,noise_level[1]), intervals_system)
            [values[stdmean, noisy3, idx], values[stdmin, noisy3, idx], values[stdmax, noisy3, idx]], [values[drmean, noisy3, idx], values[drmin, noisy3, idx], values[drmax, noisy3, idx]] = rhythm_dispersion(add_noise(intervals_system,noise_level[2]), intervals_system)

            idx += 1

    np.save(result_folder+"/data.npy", values)

#############################################################
## plot and save distributions
#############################################################

print("plot and save value distributions...")

plot_hist(values[sfo, strict_quantize], values[sfo, quantize], values[sfo, noisy1], values[sfo, noisy2], values[sfo, noisy3], limits=(-12, -4), title="spectral flatness of output", filename=result_folder+"/spectral_flatness_output.pdf")
plot_hist(values[sfd, strict_quantize], values[sfd, quantize], values[sfd, noisy1], values[sfd, noisy2], values[sfd, noisy3], limits=(-5, 6), title="spectral flatness difference", filename=result_folder+"/spectral_flatness_difference.pdf")
plot_hist(values[stdmean, strict_quantize], values[stdmean, quantize], values[stdmean, noisy1], values[stdmean, noisy2], values[stdmean, noisy3], limits=(-0.05, 0.2), title="average standard deviation change k-means", filename=result_folder+"/standard_deviation_change_average.pdf")
plot_hist(values[stdmin, strict_quantize], values[stdmin, quantize], values[stdmin, noisy1], values[stdmin, noisy2], values[stdmin, noisy3], limits=(-0.2, 0.2), title="minimum standard deviation change k-means", filename=result_folder+"/standard_deviation_change_minimum.pdf")
plot_hist(values[stdmax, strict_quantize], values[stdmax, quantize], values[stdmax, noisy1], values[stdmax, noisy2], values[stdmax, noisy3], limits=(-0.05, 0.25), title="maximum standard deviation change k-means", filename=result_folder+"/standard_deviation_change_maximum.pdf")
plot_hist(values[drmean, strict_quantize], values[drmean, quantize], values[drmean, noisy1], values[drmean, noisy2], values[drmean, noisy3], limits=(0, 0.25), title="average drifts k-means", filename=result_folder+"/drifts_average.pdf")
plot_hist(values[drmin, strict_quantize], values[drmin, quantize], values[drmin, noisy1], values[drmin, noisy2], values[drmin, noisy3], limits=(0, 0.2), title="minimum drifts k-means", filename=result_folder+"/drifts_minimum.pdf")
plot_hist(values[drmax, strict_quantize], values[drmax, quantize], values[drmax, noisy1], values[drmax, noisy2], values[drmax, noisy3], limits=(0, 0.5), title="maximum drifts k-means", filename=result_folder+"/drifts_maximum.pdf")


###############################################################
##   print results
###############################################################

print("----------------------------------------------------------------------------------------------------------------------------------------------------------")
print("      over original         \t| strict quantize   \t|     quantize      \t| noisy level: {:.1f}  \t| noisy level: {:.1f}  \t| noisy level: {:.1f}".format(noise_level[0], noise_level[1], noise_level[2]))
print("                            \t|mean      \t  std  \t|mean      \t  std  \t|mean      \t std\t|mean      \t std\t|mean      \t std")
print("----------------------------------------------------------------------------------------------------------------------------------------------------------")
print_line(values, feature_name="spectral flatness output    ", feature_index=sfo)
print_line(values, feature_name="spectral flatness difference", feature_index=sfd)
print_line(values, feature_name="average std changes k-means ", feature_index=stdmean)
print_line(values, feature_name="minimum std changes k-means ", feature_index=stdmin)
print_line(values, feature_name="maximum std changes k-means ", feature_index=stdmax)
print_line(values, feature_name="average drifts k-means      ", feature_index=drmean)
print_line(values, feature_name="minimum drifts k-means      ", feature_index=drmin)
print_line(values, feature_name="maximum drifts k-means      ", feature_index=drmax)
print("----------------------------------------------------------------------------------------------------------------------------------------------------------")
