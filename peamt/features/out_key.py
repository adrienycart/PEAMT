import numpy as np

########################################
### Out-of-key notes
########################################


def make_key_mask(target_roll):

    i = np.arange(target_roll.shape[0])
    output = np.zeros([12],dtype=float)
    for p in range(12):
        active_pitch_class = np.max(target_roll[i%12==p,:],axis=0)
        output[p] = np.mean(active_pitch_class)

    return output


# TESTED
def out_key_errors(notes_output,match,mask):

    if len(match) == 0:
        unmatched_outputs = list(range(len(notes_output)))
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

    if len(unmatched_outputs) == 0:
        return 0.0,0.0
    else:
        unmatched_weights = []
        for i in unmatched_outputs:
            unmatched_weights += [1- mask[notes_output[i]%12]]

        all_weights = []
        for note in notes_output:
            all_weights += [1- mask[note%12]]

        # import matplotlib.pyplot as plt
        # plt.plot(all_weights)
        # plt.show()

        if sum(all_weights)==0:
            return 0.0, 0.0
        else:
            return np.mean(unmatched_weights), sum(unmatched_weights)/sum(all_weights)


# TESTED
def out_key_errors_binary_mask(notes_output,match,mask,mask_thresh=0.1):


    in_mask = mask>mask_thresh

    # import matplotlib.pyplot as plt
    # fig, (ax0,ax1) = plt.subplots(2,1)
    # ax0.plot(mask)
    # ax1.plot(in_mask.astype(int))
    # plt.show()

    if len(match) == 0:
        unmatched_outputs = list(range(len(notes_output)))
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

    if len(unmatched_outputs) == 0:
        return 0.0,0.0
    else:
        out_key_unmatched = []
        for i in unmatched_outputs:
            if not in_mask[notes_output[i]%12]:
                out_key_unmatched += [notes_output[i]]

        tot_out_key = float(len(out_key_unmatched))
        tot_err = len(unmatched_outputs)
        tot_notes = len(notes_output)

        return tot_out_key/tot_err, tot_out_key/tot_notes
