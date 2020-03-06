import numpy as np
from .utils import precision, recall, Fmeasure, make_note_index_matrix, even_up_rolls


##############################################
#### Framewise highest and lowest voice
##############################################

def get_highest(roll):
    # when no note, returns -1
    highest = np.argmax(roll[::-1,:],axis=0)
    highest[highest!=0] = roll.shape[0]-1-highest[highest!=0]
    highest[highest==0] = -1
    return highest

def get_lowest(roll):
    # when no note, returns roll.shape[0]
    # we give value roll.shape[0] rather than roll.shape[0]-1, otherwise,
    # we cannot distinguish the case where there is no note and where the lowest note is roll.shape[0]-1
    lowest = np.argmax(roll,axis=0)
    # We must make the distinction between no note, and lowest note is 0
    lowest[np.logical_and(lowest==0,roll[0,:]==0)] = roll.shape[0]
    return lowest

# TESTED
def framewise_highest(output, target):

    highest = get_highest(target)

    # plot_piano_roll(target)
    # print(highest)

    highest_nonzero = highest[highest!=-1]
    frames_nonzero = np.arange(len(highest))[highest!=-1]

    tp = np.sum(output[highest_nonzero,frames_nonzero])

    fn = np.sum(output[highest_nonzero,frames_nonzero]==0)

    i,j = np.indices(target.shape)
    mask = [i>highest]
    ### Count all false positives above highest reference note)
    fp = np.sum(output[tuple(mask)])

    return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)


# TESTED
def framewise_lowest(output, target):

    lowest = get_lowest(target)

    # We give value target.shape[0] when there are no notes by convention
    lowest_nonzero = lowest[lowest!=target.shape[0]]
    frames_nonzero = np.arange(len(lowest))[lowest!=target.shape[0]]

    tp = np.sum(output[lowest_nonzero,frames_nonzero])

    fn = np.sum(output[lowest_nonzero,frames_nonzero]==0)



    i,j = np.indices(target.shape)
    mask = [i<lowest]
    ### Count all false positives above highest reference note)
    fp = np.sum(output[tuple(mask)])

    return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)

########################################
### Notewise highest and lowest voice
########################################


# TESTED
def notewise_highest(notes_output,intervals_output,notes_target,intervals_target,match,min_dur=0.05):
    #min_dur represents the minimum duration a note has to be the highest to be considered
    #in the skyline
    if len(match) == 0:
        return 0.0, 0.0, 0.0
    else:

        fs = 100

        # Get the list of highest notes
        target_refs = make_note_index_matrix(notes_target,intervals_target)
        output_refs = make_note_index_matrix(notes_output,intervals_output)
        target_refs,output_refs = even_up_rolls([target_refs,output_refs],pad_value=-1)

        roll_target = (target_refs!=-1).astype(int)
        roll_output = (output_refs!=-1).astype(int)

        highest = get_highest(roll_target)

        # print(highest)
        # plt.imshow(roll_target)
        # plt.show()

        highest_nonzero = highest[highest!=-1]
        frames_nonzero = np.arange(len(highest))[highest!=-1]

        highest_notes_idx, count = np.unique(target_refs[highest_nonzero,frames_nonzero],return_counts=True)
        highest_notes_idx = highest_notes_idx[count/float(fs) > min_dur]

        # Compute true positives
        # NB: matching gives indexes (idx_target,idx_output)


        matched_targets, matched_outputs = zip(*match)
        matched_targets_is_highest = [idx for idx in matched_targets if idx in highest_notes_idx]
        tp = len(matched_targets_is_highest)

        # Compute false negatives
        unmatched_targets= list(set(range(len(notes_target)))-set(matched_targets))
        unmatched_targets_is_highest = [idx for idx in unmatched_targets if idx in highest_notes_idx]
        fn = len(unmatched_targets_is_highest)

        # Compute false positives
        # Count all false positives that are above the highest note
        i,j = np.indices(target_refs.shape)
        higher_mask = [i>highest]
        higher_notes_idx, count = np.unique(output_refs[tuple(higher_mask)],return_counts=True)
        count = count[higher_notes_idx!= -1]
        higher_notes_idx = higher_notes_idx[higher_notes_idx!= -1]
        higher_notes_idx = higher_notes_idx[count/float(fs) > min_dur]

        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))
        unmatched_outputs_is_higher = [idx for idx in unmatched_outputs if idx in higher_notes_idx]
        fp = len(unmatched_outputs_is_higher)

        # fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
        # ax0.imshow(roll_target,aspect='auto',origin='lower')
        # ax1.imshow(roll_output,aspect='auto',origin='lower')
        # display1 = np.zeros_like(roll_output)
        # display2 = np.zeros_like(roll_output)
        # for i in matchighest_lowest_voicehed_targets:
        #     display1[notes_target[i],int(intervals_target[i][0]*fs):int(intervals_target[i,1]*fs)] = 1
        # for i in unmatched_outputs_is_higher:
        #     display2[notes_output[i],int(intervals_output[i][0]*fs):int(intervals_output[i,1]*fs)] = 1

        # display2[higher_mask]+=3

        # ax2.imshow(display1,aspect='auto',origin='lower')
        # ax3.imshow(display2,aspect='auto',origin='lower')
        # plt.show()

        return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)


# TESTED
def notewise_lowest(notes_output,intervals_output,notes_target,intervals_target,match,min_dur=0.05):

    if len(match) == 0:
        return 0.0,0.0,0.0
    else:
        #min_dur represents the minimum duration a note has to be the highest to be considered
        #in the skyline
        fs = 100

        # Get the list of highest notes
        target_refs = make_note_index_matrix(notes_target,intervals_target)
        output_refs = make_note_index_matrix(notes_output,intervals_output)
        target_refs,output_refs = even_up_rolls([target_refs,output_refs],pad_value=-1)

        roll_target = (target_refs!=-1).astype(int)
        roll_output = (output_refs!=-1).astype(int)

        lowest = get_lowest(roll_target)
        lowest_nonzero = lowest[lowest!=roll_target.shape[0]]
        frames_nonzero = np.arange(len(lowest))[lowest!=roll_target.shape[0]]

        lowest_notes_idx, count = np.unique(target_refs[lowest_nonzero,frames_nonzero],return_counts=True)
        lowest_notes_idx = lowest_notes_idx[count/float(fs) > min_dur]

        # Compute true positives
        # NB: matching gives indexes (idx_target,idx_output)

        matched_targets, matched_outputs = zip(*match)
        matched_targets_is_lowest = [idx for idx in matched_targets if idx in lowest_notes_idx]
        tp = len(matched_targets_is_lowest)

        # Compute false negatives
        unmatched_targets= list(set(range(len(notes_target)))-set(matched_targets))
        unmatched_targets_is_lowest = [idx for idx in unmatched_targets if idx in lowest_notes_idx]
        fn = len(unmatched_targets_is_lowest)

        # Compute false positives
        # Count all false positives that are above the lowest note
        i,j = np.indices(target_refs.shape)
        lower_mask = [i<lowest]
        lower_notes_idx, count = np.unique(output_refs[tuple(lower_mask)],return_counts=True)
        count = count[lower_notes_idx!= -1]
        lower_notes_idx = lower_notes_idx[lower_notes_idx!= -1]
        # print(lower_notes_idx, count)
        lower_notes_idx = lower_notes_idx[count/float(fs) > min_dur]

        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))
        unmatched_outputs_is_lower = [idx for idx in unmatched_outputs if idx in lower_notes_idx]
        fp = len(unmatched_outputs_is_lower)

        # import matplotlib.pyplot as plt
        # fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
        # ax0.imshow(roll_target,aspect='auto',origin='lower')
        # ax1.imshow(roll_output,aspect='auto',origin='lower')
        # display1 = np.zeros_like(roll_output)
        # display2 = np.zeros_like(roll_output)
        # for i in matched_targets:
        #     display1[notes_target[i],int(intervals_target[i][0]*fs):int(intervals_target[i,1]*fs)] = 1
        # for i in unmatched_outputs_is_lower:
        #     display2[notes_output[i],int(intervals_output[i][0]*fs):int(intervals_output[i,1]*fs)] = 1
        #
        # display2[lower_mask]+=3
        #
        # ax2.imshow(display1,aspect='auto',origin='lower')
        # ax3.imshow(display2,aspect='auto',origin='lower')
        # plt.show()

        return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)


# TESTED
def correct_highest_lowest_note_framewise(output, target):
    # return two parameters, the proportion of correct highest notes and lowest notes framewise
    highest_output = get_highest(output)
    highest_target = get_highest(target)
    correct_highest_seq = [int(highest_output[idx] == highest_target[idx]) for idx in range(len(highest_output))]
    lowest_output = get_lowest(output)
    lowest_target = get_lowest(target)
    correct_lowest_seq = [int(lowest_output[idx] == lowest_target[idx]) for idx in range(len(lowest_output))]

    correct_highest_count = correct_highest_seq.count(1)
    correct_lowest_count = correct_lowest_seq.count(1)

    return float(correct_highest_count) / len(correct_highest_seq), float(correct_lowest_count) / len(correct_lowest_seq)
