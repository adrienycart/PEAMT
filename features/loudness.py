import numpy as np
from .utils import precision, recall, Fmeasure, make_note_index_matrix, even_up_rolls, get_loudness

########################################
### Loudness
########################################

# TESTED
def false_negative_loudness(match, vel_target, intervals_target, dt=1):
    # in moving average, include the notes whose onsets are in the range of [t-dt, t+dt)

    if len(match) == 0:
        return 0.0
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_targets= list(set(range(len(vel_target)))-set(matched_targets))

        if len(unmatched_targets) == 0:
            return 0.0
        else:
            # unmatched_vels = vel_target[unmatched_targets]
            # avg_unmatched = np.mean(unmatched_vels)
            unmatched_intervals = intervals_target[unmatched_targets]
            unmatched_vels_normed = []
            for unmatched_idx in unmatched_targets:
                recent_vels = []
                for idx in range(len(intervals_target)):
                    if intervals_target[idx][0] >= intervals_target[unmatched_idx][0] - dt and intervals_target[idx][0] < intervals_target[unmatched_idx][0] + dt:
                        recent_vels.append(vel_target[idx])
                unmatched_vels_normed.append(float(vel_target[unmatched_idx]) / np.mean(recent_vels))

            return np.mean(unmatched_vels_normed)


# TESTED
def loudness_ratio_false_negative(notes_target, intervals_target, vel_target, match, min_dur=0.05):
    # loudness ratio of false negative
    # ratio = FN velocity / max loudness in ground truth at onset

    if len(match) == 0:
        return 0.0

    # compute false negatives
    matched_targets, matched_outputs = zip(*match)
    fn_indexs = [idx for idx in range(len(notes_target)) if idx not in matched_targets]
    if len(fn_indexs) == 0:
        return 0.0

    ratios = []
    for i in fn_indexs:
        # loop over false negatives (missing notes)
        note_target = notes_target[i]
        onset_target = intervals_target[i][0]
        velocity = vel_target[i]

        ccn_idxes = [idx for idx in range(len(notes_target)) if intervals_target[idx][0] <= onset_target and intervals_target[idx][1] >= onset_target]
        concurrent_notes_loudnesses = [get_loudness(notes_target[ccn_idxes[idx]], vel_target[ccn_idxes[idx]], onset_target - intervals_target[ccn_idxes[idx]][0]) for idx in range(len(ccn_idxes))]
        ratio = velocity / max(concurrent_notes_loudnesses)
        ratios.append(ratio)

    return np.mean(ratios)
