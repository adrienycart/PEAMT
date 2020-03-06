import os
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm
import mir_eval
import features.utils as utils
from features.rhythm import rhythm_histogram, rhythm_dispersion

import warnings
warnings.filterwarnings("ignore")

utils.create_folder("validate_rhythm_feature_plots")
MIDI_path = "app/static/data/all_midi_cut"
systems = ["kelz", "lisu", "google", "cheng"]
fs = 100

N_features = 8
sfo, sfd, stdmean, stdmin, stdmax, drmean, drmin, drmax = range(N_features) # indexes for all the rhythm features
N_modifications = 3
quantize, original, noisy = range(N_modifications)  # index for modification versions
N_outputs = len(os.listdir(MIDI_path)) * len(systems)   # number of all outputs


def generate_hist_plots(x1, x2, x3, limits, title, filename, n_bins=100):

    plt.figure()
    plt.subplot(311)
    plt.hist(x1, bins=n_bins, range=limits)
    plt.ylabel("quantized")
    plt.title(title)
    plt.subplot(312)
    plt.hist(x2, bins=n_bins, range=limits)
    plt.ylabel("original")
    plt.subplot(313)
    plt.hist(x3, bins=n_bins, range=limits)
    plt.ylabel("noisy")
    plt.savefig(filename)
    plt.show()

def add_noise(intervals,noise_level):
    return intervals + np.random.uniform(-noise_level,noise_level,size = [intervals.shape[0],1])

def quantise(intervals,beats):
    onsets = intervals[:,0]
    onsets = [beats[np.argmin(np.abs(beats - onset))] for onset in onsets]
    intervals[:,0] = onsets
    return intervals

def plot_differences(d1, d2, limits, title, filename, n_bins=100):

    plt.figure()
    plt.subplot(211)
    plt.hist(d1, bins=n_bins, range=limits)
    plt.ylabel("original - quantized")
    plt.title(title)
    plt.subplot(212)
    plt.hist(d2, bins=n_bins, range=limits)
    plt.ylabel("noisy - original")
    plt.savefig(filename)
    plt.show()


def print_line(values, feature_name, feature_index):

    print(feature_name+"\t|{:.3f}\t{:.3f}\t|{:.3f}\t{:.3f}\t|{:.3f}\t{:.3f}\t|{:.3f}\t{:.3f}\t|{:.3f}\t{:.3f}".format(np.mean(values[feature_index,quantize,:]), np.std(values[feature_index,quantize,:]), np.mean(values[feature_index,original,:]), np.std(values[feature_index,original,:]), np.mean(values[feature_index,noisy,:]), np.std(values[feature_index,noisy,:]), np.mean(values[feature_index,original,:]-values[feature_index,quantize,:]), np.std(values[feature_index,original,:]-values[feature_index,quantize,:]), np.mean(values[feature_index,noisy,:]-values[feature_index,original,:]), np.std(values[feature_index,noisy,:]-values[feature_index,original,:])))

##################################################################################
## get feature values
##################################################################################

if os.path.exists("validate_rhythm_feature_plots/data.npy"):
    values = np.load("validate_rhythm_feature_plots/data.npy")
else:
    values = np.zeros((N_features, N_modifications, N_outputs), dtype=float)
    print(values.shape)

    idx = 0
    for example in os.listdir(MIDI_path)[:]:
        if example.startswith('.'):
            continue

        example_path = os.path.join(MIDI_path, example)
        print((idx / len(systems), example_path))

        target_data = pm.PrettyMIDI(os.path.join(example_path, 'target.mid'))
        target_pr = (target_data.get_piano_roll(fs)>0).astype(int)
        notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data, with_vel=True)

        target_PPQ = target_data.resolution
        end_tick = target_data.time_to_tick(target_data.get_end_time())
        ticks = np.arange(0, end_tick, target_PPQ/4)
        quarter_times = np.array([target_data.tick_to_time(t) for t in ticks])

        for system in systems:
            # print(system)
            system_data = pm.PrettyMIDI(os.path.join(example_path, system + '.mid'))
            system_pr = (system_data.get_piano_roll(fs)>0).astype(int)
            notes_system, intervals_system = utils.get_notes_intervals(system_data)

            target_pr, system_pr = utils.even_up_rolls([target_pr, system_pr])

            if len(notes_system) == 0:
                match_on = []
            else:
                match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=None, pitch_tolerance=0.25)

            values[sfo, quantize, idx], values[sfd, quantize, idx] = rhythm_histogram(quantise(intervals_system,quarter_times), intervals_target)
            values[sfo, original, idx], values[sfd, original, idx] = rhythm_histogram(intervals_system, intervals_target)
            values[sfo, noisy, idx], values[sfd, noisy, idx] = rhythm_histogram(add_noise(intervals_system,0.2), intervals_target)

            [values[stdmean, quantize, idx], values[stdmin, quantize, idx], values[stdmax, quantize, idx]], [values[drmean, quantize, idx], values[drmin, quantize, idx], values[drmax, quantize, idx]] = rhythm_dispersion(quantise(intervals_system,quarter_times), intervals_target)
            [values[stdmean, original, idx], values[stdmin, original, idx], values[stdmax, original, idx]], [values[drmean, original, idx], values[drmin, original, idx], values[drmax, original, idx]] = rhythm_dispersion(intervals_system, intervals_target)
            [values[stdmean, noisy, idx], values[stdmin, noisy, idx], values[stdmax, noisy, idx]], [values[drmean, noisy, idx], values[drmin, noisy, idx], values[drmax, noisy, idx]] = rhythm_dispersion(add_noise(intervals_system,0.2), intervals_target)


            idx += 1

    np.save("validate_rhythm_feature_plots/data.npy", values)

###################################################################################
## generate features value distributions
###################################################################################

print("calculate and save feature value distributions plots...")
generate_hist_plots(values[sfo, quantize, :], values[sfo, original, :], values[sfo, noisy, :], limits=(-13, -2), title="spectral flatness of output", filename="validate_rhythm_feature_plots/spectral_flatness_output.pdf")
generate_hist_plots(values[sfd, quantize, :], values[sfd, original, :], values[sfd, noisy, :], limits=(-4, 8), title="spectral flatness difference (output - target)", filename="validate_rhythm_feature_plots/spectral_flatness_difference.pdf")
generate_hist_plots(values[stdmean, quantize, :], values[stdmean, original, :], values[stdmean, noisy, :], limits=(-0.1, 0.4), title="average std changes for k-means", filename="validate_rhythm_feature_plots/std_changes_mean.pdf")
generate_hist_plots(values[stdmin, quantize, :], values[stdmin, original, :], values[stdmin, noisy, :], limits=(-0.2, 0.2), title="minimum std changes for k-means", filename="validate_rhythm_feature_plots/std_changes_min.pdf")
generate_hist_plots(values[stdmax, quantize, :], values[stdmax, original, :], values[stdmax, noisy, :], limits=(-0.1, 0.5), title="maximum std changes for k-means", filename="validate_rhythm_feature_plots/std_changes_max.pdf")
generate_hist_plots(values[drmean, quantize, :], values[drmean, original, :], values[drmean, noisy, :], limits=(0, 0.5), title="average centre drifts for k-means", filename="validate_rhythm_feature_plots/centre_drifts_mean.pdf")
generate_hist_plots(values[drmin, quantize, :], values[drmin, original, :], values[drmin, noisy, :], limits=(0, 0.2), title="minimum centre drifts for k-means", filename="validate_rhythm_feature_plots/centre_drifts_min.pdf")
generate_hist_plots(values[drmax, quantize, :], values[drmax, original, :], values[drmax, noisy, :], limits=(0, 1.5), title="maximum centre drifts for k-means", filename="validate_rhythm_feature_plots/centre_drifts_max.pdf")


########################################################################
##  plot differences
########################################################################

print("calculate and save feature differences for qunatized/original/noisy result plots...")
plot_differences(values[sfo, original, :]-values[sfo, quantize, :], values[sfo, noisy, :]-values[sfo, original, :], limits=(-2, 5), title="difference in spectral flatness of output", filename="validate_rhythm_feature_plots/spectral_flatness_output_difference.pdf")
plot_differences(values[sfd, original, :]-values[sfd, quantize, :], values[sfd, noisy, :]-values[sfd, original, :], limits=(-3, 5), title="difference in spectral flatness difference (output - target)", filename="validate_rhythm_feature_plots/spectral_flatness_difference_difference.pdf")
plot_differences(values[stdmean, original, :]-values[stdmean, quantize, :], values[stdmean, noisy, :]-values[stdmean, original, :], limits=(-0.1, 0.2), title="difference in average std changes for k-means", filename="validate_rhythm_feature_plots/std_changes_mean_difference.pdf")
plot_differences(values[stdmin, original, :]-values[stdmin, quantize, :], values[stdmin, noisy, :]-values[stdmin, original, :], limits=(-0.15, 0.2), title="difference in minimum std changes for k-means", filename="validate_rhythm_feature_plots/std_changes_min_difference.pdf")
plot_differences(values[stdmax, original, :]-values[stdmax, quantize, :], values[stdmax, noisy, :]-values[stdmax, original, :], limits=(-0.2, 0.3), title="difference in maximum std changes for k-means", filename="validate_rhythm_feature_plots/std_changes_max_difference.pdf")
plot_differences(values[drmean, original, :]-values[drmean, quantize, :], values[drmean, noisy, :]-values[drmean, original, :], limits=(-0.2, 0.25), title="difference in average centre drifts for k-means", filename="validate_rhythm_feature_plots/centre_drifts_mean_difference.pdf")
plot_differences(values[drmin, original, :]-values[drmin, quantize, :], values[drmin, noisy, :]-values[drmin, original, :], limits=(-0.15, 0.15), title="difference in minimum centre drifts for k-means", filename="validate_rhythm_feature_plots/centre_drifts_min_difference.pdf")
plot_differences(values[drmax, original, :]-values[drmax, quantize, :], values[drmax, noisy, :]-values[drmax, original, :], limits=(-0.25, 0.4), title="difference in maximum centre drifts for k-means", filename="validate_rhythm_feature_plots/centre_drifts_max_difference.pdf")


######################################################################
##  print distribution statistics
######################################################################

print("-------------------------------------------------------------------------------------------------------------")
print("                            \t|quant \t      \t|origi \t      \t|noisy \t      \t|or-qu \t      \t|no-or \t ")
print("                            \t|mean  \tstd   \t|mean  \tstd   \t|mean  \tstd   \t|mean  \tstd   \t|maan  \tstd")
print("-------------------------------------------------------------------------------------------------------------")
print_line(values, feature_name="spectral flatness output    ", feature_index=sfo)
print_line(values, feature_name="spectral flatness difference", feature_index=sfd)
print_line(values, feature_name="average std changes k-means ", feature_index=stdmean)
print_line(values, feature_name="minimum std changes k-means ", feature_index=stdmin)
print_line(values, feature_name="maximum std changes k-means ", feature_index=stdmax)
print_line(values, feature_name="average drifts k-means      ", feature_index=drmean)
print_line(values, feature_name="minimum drifts k-means      ", feature_index=drmin)
print_line(values, feature_name="maximum drifts k-means      ", feature_index=drmax)
print("-------------------------------------------------------------------------------------------------------------")
