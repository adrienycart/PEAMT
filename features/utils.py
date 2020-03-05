import numpy as np
import matplotlib.pyplot as plt
import os


def get_roll_from_times(midi_data,times):
    # quant_piano_roll = midi_data.get_piano_roll(fs=500,times=times)
    # quant_piano_roll = (quant_piano_roll>=7).astype(int)
    roll = np.zeros([128,len(times)])

    for instr in midi_data.instruments:
        for note in instr.notes:
            start = np.argmin(np.abs(times-note.start))
            end = np.argmin(np.abs(times-note.end))
            if start == end:
                end = start+1
            roll[note.pitch,start:end]=1
    return roll


def even_up_rolls(rolls,pad_value=0):
    #Makes roll1 and roll2 of same size.
    lens = [roll.shape[1] for roll in rolls]
    max_len = max(lens)
    output = []
    for roll in rolls:
        if roll.shape[1] < max_len:
            roll = np.concatenate([roll,pad_value*np.ones([roll.shape[0],max_len-roll.shape[1]])],axis=1)
        output += [roll]

    return output

def get_notes_intervals(midi_data,with_vel=False):
    notes= []
    intervals = []
    if with_vel:
        vels = []

    for instr in midi_data.instruments:
        for note in instr.notes:
            notes += [note.pitch]
            intervals += [[note.start,note.end]]
            if with_vel:
                vels += [note.velocity]
    output = [np.array(notes), np.array(intervals)]
    if with_vel:
        output += [np.array(vels)]

    return output

def make_note_index_matrix(notes,intervals,fs=100):
    end_time = np.max(intervals[:,1])
    # Allocate a matrix of zeros - we will add in as we go
    matrix = -np.ones((128, int(fs*end_time)))
    # Make a piano-roll-like matrix holding the index of each note.
    # -1 indicates no note
    for i,(note,interval) in enumerate(zip(notes,intervals)):
        matrix[note,int(interval[0]*fs):int(interval[1]*fs)] = i
    return matrix

def precision(tp,fp):
    #Compute precision for  one file
    pre = tp/(tp+fp+np.finfo(float).eps)

    return pre

def recall(tp,fn):
    #Compute recall for  one file
    rec = tp/(tp+fn+np.finfo(float).eps)
    return rec


def accuracy(tp,fp,fn):
    #Compute accuracy for one file
    acc = tp/(tp+fp+fn+np.finfo(float).eps)
    return acc

def Fmeasure(tp,fp,fn):
    #Compute F-measure  one file
    prec = precision(tp,fp)
    rec = recall(tp,fn)
    return 2*prec*rec/(prec+rec+np.finfo(float).eps)

def get_loudness(midi_pitch, velocity, time):
    # compute decay_rate according to midipitch and note velocity
    time = min(time, 1)
    decay_rate = 0.050532 + 0.021292 * midi_pitch
    loudness = velocity * np.exp(-1.0 * decay_rate * time)
    return loudness


def plot_piano_roll(pr):
    fig = plt.figure()
    fig = plt.imshow(pr)
    plt.show()
    return


def create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except:
            print("create folder error")


def get_time(data,bar,beat,sub_beat):

    first_sig = data.time_signature_changes[0]
    if first_sig.numerator == 1 and first_sig.denominator == 4:
        bar += 1

    PPQ = data.resolution
    downbeats = data.get_downbeats()
    try:
        bar_t = downbeats[bar]
    except IndexError as e:
        if bar == len(downbeats):
            # Instead of the first beat of the bar after the last one (that doesn't exist),
            # We take the before-last beat of the last bar
            bar_t = downbeats[-1]
            beat = 'last'
        else:
            raise e


    time_sigs = data.time_signature_changes
    last_sig = True
    for i, sig in enumerate(time_sigs):
        if sig.time > bar_t:
            last_sig = False
            break

    if last_sig:
        current_sig = time_sigs[i]
    else:
        current_sig = time_sigs[i-1]

    if beat == 'last':
        beat = current_sig.numerator - 1

    try:
        assert beat < current_sig.numerator
    except AssertionError:
        print(downbeats)
        for sig in time_sigs:
            print(sig)
        print('-----------')
        print(bar,beat, bar_t)
        print(current_sig)
        raise AssertionError

    beat_ticks = PPQ * 4 / current_sig.denominator
    tick = data.time_to_tick(bar_t) + beat * beat_ticks + sub_beat*beat_ticks/2
    if tick != int(tick):
        print(bar,beat,sub_beat)
        print(current_sig)
        print(tick)
        raise TypeError('Tick is not int!!!')
    else:
        tick = int(tick)
    time =data.tick_to_time(tick)

    return time

    
def str_to_bar_beat(string):
    if '.' in string:
        str_split = string.split('.')
        bar = int(str_split[0])
        beat = int(str_split[1])
        output = [bar,beat,0]

        if len(str_split)>2:
            sub_beat=int(str_split[2])
            output[2] = sub_beat
    else:
        output = [int(string),0,0]
    return output