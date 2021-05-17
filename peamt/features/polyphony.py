import numpy as np


########################################
### Polyphony --- discarded
########################################

def polyphony_level_diff(roll_output,roll_target):
    poly_output = np.sum(roll_output,axis=0)
    poly_target = np.sum(roll_target,axis=0)

    poly_diff = np.abs(poly_output-poly_target)

    return np.mean(poly_diff),np.std(poly_diff),np.min(poly_diff),np.max(poly_diff)


# discarded
def false_negative_polyphony_level(roll_target,intervals_target,match):
    fs = 100

    if len(match) == 0:
        unmatched_targets = list(range(intervals_target))
    else:
        matched_targets, matched_outputs = zip(*match)
        # unmatched_targets= list(set(range(len(vel_target)))-set(matched_targets))
        unmatched_targets= list(set(range(len(intervals_target)))-set(matched_targets))

    unmatched_intervals = intervals_target[unmatched_targets,:]

    all_avg_poly = []

    for [start,end] in unmatched_intervals:
        start_idx = int(round(start*fs))
        end_idx = int(round(end*fs))
        avg_poly = np.mean(np.sum(roll_target[:,start_idx:end_idx],axis=0))
        all_avg_poly += [avg_poly]

    return all_avg_poly
