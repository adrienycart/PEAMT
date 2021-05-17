import numpy as np
from numpy import random

SMALL_VALUE = 0.00001

def calculate_stds(data, means):
	# calculate stds for k-means result
	data_groups = [[] for i in range(len(means))]
	for ioi in data:
		data_groups[np.argmin(abs(np.array(means) - ioi))].append(ioi)
	stds = []
	for group in data_groups:
		if len(group) <= 1:
			stds.append(0.0)
		else:
			stds.append(np.std(group, ddof=1))
	return stds


# TESTED
def rhythm_histogram(intervals_output, intervals_target):
    # return the logged spectral flatness ratio of the IOIs, where spectral flatness ratio is defined as the ratio of the geometric mean of the histogram over its arithmetic mean.
    # 1. spectral flatness for IOI of the output transcription
    # 2. spectral flatness for IOI of the ground truth music piece

    # order note intervals by onsets
    onsets_output = [interval[0] for interval in intervals_output]
    onsets_target = [interval[0] for interval in intervals_target]
    onsets_output.sort()
    onsets_target.sort()

    ioi_output = [onsets_output[idx+1] - onsets_output[idx] for idx in range(len(onsets_output)-1)]
    ioi_target = [onsets_target[idx+1] - onsets_target[idx] for idx in range(len(onsets_target)-1)]

    # generate bins
    bins = [i*0.01 for i in range(10)]
    bins += [0.1+i*0.1 for i in range(20)]

    histogram_output = np.histogram(ioi_output, bins=bins)[0]
    histogram_target = np.histogram(ioi_target, bins=bins)[0]

    histogram_output = [max(SMALL_VALUE, histogram_output[idx]) for idx in range(len(histogram_output))]
    histogram_target = [max(SMALL_VALUE, histogram_target[idx]) for idx in range(len(histogram_target))]

    # logged geometric means
    log_gmean_output = np.mean(np.log(histogram_output))
    log_gmean_target = np.mean(np.log(histogram_target))
    # arithmetic means
    mean_output = np.mean(histogram_output)
    mean_target = np.mean(histogram_target)
    # logged spectral flatness
    log_spectral_flatness_output = log_gmean_output - np.log(mean_output)
    log_spectral_flatness_target = log_gmean_target - np.log(mean_target)

    # print(log_spectral_flatness_output)
    # print(log_spectral_flatness_target)
    # print(log_spectral_flatness_output / log_spectral_flatness_target)
    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.bar(np.arange(len(histogram_target)), histogram_target)
    # plt.subplot(212)
    # plt.bar(np.arange(len(histogram_output)), histogram_output)
    # plt.show()

    return log_spectral_flatness_output, log_spectral_flatness_output-log_spectral_flatness_target


# TESTED
def rhythm_dispersion(intervals_output, intervals_target):
    # return changes in k-means clusters
    # 1. change in standard deviations
    # 2. center drift

    # order note intervals by onsets
    onsets_output = [interval[0] for interval in intervals_output]
    onsets_target = [interval[0] for interval in intervals_target]
    onsets_output.sort()
    onsets_target.sort()


    ioi_output = [onsets_output[idx+1] - onsets_output[idx] for idx in range(len(onsets_output)-1)]
    ioi_target = [onsets_target[idx+1] - onsets_target[idx] for idx in range(len(onsets_target)-1)]

    # initialise cluster
    bins = [i*0.02 for i in range(5)]
    bins += [0.1+i*0.2 for i in range(10)]
    histogram_target = np.histogram(ioi_target, bins=bins)[0]
    means = []
    for i in range(len(histogram_target)):
        if histogram_target[i] > 0 and (i == 0 or histogram_target[i] > histogram_target[i-1]) and (i == len(histogram_target)-1 or histogram_target[i] >= histogram_target[i+1]):
            means.append(np.mean([bins[i], bins[i+1]]))

    if len(means) == 0:
        return 0.0, 0.0

    # k-means on target intervals
    moving = 1
    while moving > 0.0001:
        # cluster
        ioi_target_labels = [np.argmin(abs(np.array(means) - ioi_target[idx])) for idx in range(len(ioi_target))]
        # calculate new means
        new_means = []
        for label in range(len(means)):
            indexs = np.array([i for i in range(len(ioi_target)) if ioi_target_labels[i] == label])
            if len(indexs) > 0:
                # only update cluster if there are some points within it, else discard it.
                new_means.append(np.mean(np.array(ioi_target)[indexs]))
            else:
                means.pop(label)
        # calculate centre moving
        moving = np.sum(abs(np.array(new_means) - np.array(means)))
        # update means
        means = new_means
        # print(means)

    # calculate target std
    stds_target = calculate_stds(ioi_target, means)

    # print('--')
    # k-means on output intervals
    means_output = [means[i] for i in range(len(means))] # copy target means
    moving = 1
    while moving > 0.0001:
        # cluster
        ioi_output_labels = [np.argmin(abs(np.array(means_output) - ioi_output[idx])) for idx in range(len(ioi_output))]
        # calculate new means
        new_means_output = []
        for label in range(len(means_output)):
            indexs = np.array([i for i in range(len(ioi_output)) if ioi_output_labels[i] == label])
            if len(indexs) > 0:
                new_means_output.append(np.mean(np.array(ioi_output)[indexs]))
            else:
                # if there is no points within this cluster, keep the initial cluster mean
                new_means_output.append(means_output[label])
        # calculate centre moving
        moving = np.sum(abs(np.array(new_means_output) - np.array(means_output)))
        # update means
        means_output = new_means_output
        # print(means_output)

    # calculate stds for output
    stds_output = calculate_stds(ioi_output, means_output)
    # calcuate std chage
    stds_change = list(np.array(stds_output) - np.array(stds_target))

    # print(stds_target)
    # print(stds_output)

    # cluster centre drift
    drifts = [abs(means[idx] - means_output[idx]) for idx in range(len(means))]

    # # test with graphs
    # histogram_output = np.histogram(ioi_output, bins=bins)[0]
    # histogram_output = [max(0, histogram_output[idx]) for idx in range(len(histogram_output))]
    # histogram_target = [max(0, histogram_target[idx]) for idx in range(len(histogram_target))]
    # import matplotlib.pyplot as plt
    # plt.subplot(411)
    # plt.bar(np.arange(len(histogram_target)), histogram_target)
    # plt.subplot(412)
    # plt.bar(np.arange(len(histogram_output)), histogram_output)
    # plt.subplot(413)
    # plt.bar(np.arange(len(stds_change)), stds_change)
    # plt.subplot(414)
    # plt.bar(np.arange(len(drifts)), drifts)
    # plt.show()

    return [np.mean(stds_change),np.min(stds_change),np.max(stds_change)], [np.mean(drifts),np.min(drifts),np.max(drifts)]
