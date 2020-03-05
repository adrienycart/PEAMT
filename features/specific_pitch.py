import numpy as np
from .utils import *


##############################################
#### Framewise
##############################################


# TESTED
def specific_pitch_framewise(output,target,fs,n_semitones,down_only=False,delta=0.05):

    FPs = np.logical_and(output == 1, target == 0)

    target_shift_down = np.concatenate([target[n_semitones:,:],np.zeros([n_semitones,target.shape[1]])],axis=0)
    target_shift_up = np.concatenate([np.zeros([n_semitones,target.shape[1]]),target[:-n_semitones,:]],axis=0)

    match_down = FPs*target_shift_up # correspond to when an output is matched with a target n_semitones below
    match_up = FPs*target_shift_down # correspond to when an output is matched with a target n_semitones above

    # import matplotlib.pyplot as plt
    # plt.subplot(311)
    # plt.imshow(output)
    # plt.subplot(312)
    # plt.imshow(match_down)
    # plt.subplot(313)
    # plt.imshow(match_up)
    # plt.show()

    delta_steps = int(round(delta*fs))
    delta_steps = min(delta_steps, target.shape[1])  # limit delta_steps within segment length
    continuation_mask = np.concatenate([np.concatenate([target[:,i:],np.zeros([target.shape[0],i])],axis=1)[:,:,None] for i in range(delta_steps)],axis=2)

    match_past = FPs*np.all(continuation_mask==0,axis=2).astype(int)

    if down_only:
        n_match = np.sum(match_down*match_past)
    else:
        match_pitch = np.logical_or(match_down,match_down).astype(int)
        n_match = np.sum(match_pitch*match_past)

    if np.sum(FPs) == 0:
        return 0.0,0.0
    else:
        n_match = float(n_match)
        n_FP = np.sum(FPs)
        n_tot = np.sum(output)

        return n_match/n_FP, n_match/n_tot


########################################
### Notewise
########################################

# TESTED
def specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match, n_semitones, down_only=False, ratio=0.8):
    # return two ratios:
    # 1. the proportion of specific pitch mistakes among false positives
    # 2. the proportion of specific pitch mistakes among all detected notes
    if len(match) == 0:
        return 0.0, 0.0

    # get false positives
    matched_targets, matched_outputs = zip(*match)
    fp_idxs = [idx for idx in range(notes_output.shape[0]) if idx not in matched_outputs]
    if len(fp_idxs) == 0:
        return 0.0, 0.0

    n_specific_pitch = 0.0
    for i in fp_idxs:
        # loop over all false positives
        note_output = notes_output[i]
        interval_output = intervals_output[i]
        is_specific_pitch_error = False

        note_down = note_output - n_semitones
        intervals_down = [intervals_target[idx] for idx in range(len(notes_target)) if notes_target[idx] == note_down]

        for interval in intervals_down:
            if (min(interval_output[1], interval[1]) - max(interval_output[0], interval[0])) / (interval_output[1] - interval_output[0]) > ratio:
                is_specific_pitch_error = True
                break

        if not down_only and not is_specific_pitch_error:
            note_up = note_output + n_semitones
            intervals_up = [intervals_target[idx] for idx in range(len(notes_target)) if notes_target[idx] == note_up]

            for interval in intervals_up:
                if (min(interval_output[1], interval[1]) - max(interval_output[0], interval[0])) / (interval_output[1] - interval_output[0]) > ratio:
                    is_specific_pitch_error = True
                    break

        if is_specific_pitch_error:
            n_specific_pitch += 1.0

    return n_specific_pitch / len(fp_idxs), n_specific_pitch / len(notes_output)
