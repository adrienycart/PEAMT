import numpy as np
import os
import pretty_midi as pm
import copy


def make_roll(intervals,pitches,shape,fs=100):
    assert len(intervals) == len(pitches)

    roll = np.zeros(shape)
    for pitch, (start,end) in zip(pitches,intervals):
        # use int() instead of int(round()) to be coherent with PrettyMIDI.get_piano_roll()
        start = int(start*fs)
        end = int(end*fs)
        if start == end:
            end = start+1
        roll[int(pitch),start:end]=1
    return roll


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

def apply_sustain_control_changes(midi):
    all_CCs = []
    for instr in midi.instruments:
        all_CCs += instr.control_changes
    all_pedals = [cc for cc in all_CCs if cc.number==64]
    pedals_sorted = sorted(all_pedals,key=lambda x: x.time)

    #Add an extra pedal off at the end, just in case
    pedals_sorted += [pm.ControlChange(64,0,midi.get_end_time())]

    #Create a pedal_ON array such that pedal_ON[i]>0 iff pedal is on at tick i
    #If pedal_ON[i]>0, its value is the time at which pedal becomes off again
    pedal_ON = np.zeros(midi._PrettyMIDI__tick_to_time.shape,dtype=float)
    # -1 if pedal is currently off, otherwise tick time of first time it is on.
    ON_idx = -1
    for cc in pedals_sorted:
        if cc.value > 64:
            if ON_idx < 0:
                ON_idx = midi.time_to_tick(cc.time)
            else:
                # Pedal is already ON
                pass
        else:
            if ON_idx>0:
                pedal_ON[ON_idx:midi.time_to_tick(cc.time)]=cc.time
                ON_idx = -1
            else:
                # Pedal is already OFF
                pass

    # Copy to keep time signatures and tempo changes, but remove notes and CCs
    new_midi = copy.deepcopy(midi)
    new_midi.instruments = []


    # Store the notes per pitch, to trim them afterwards.
    all_notes = np.empty([128],dtype=object)
    for i in range(128):
        all_notes[i] = []


    for instr in midi.instruments:

        # First, extend all the notes until the pedal is off
        for note in instr.notes:
            start_tick = midi.time_to_tick(note.start)
            end_tick = midi.time_to_tick(note.end)

            if np.any(pedal_ON[start_tick:end_tick]>0):
                # Pedal is on while note is on
                end_pedal = np.max(pedal_ON[start_tick:end_tick])
                note.end = max(note.end,end_pedal)
            else:
                # Pedal is not on while note is on, no modifications needed
                pass
            all_notes[note.pitch] += [note]

    new_instr = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'),name="Piano")

    # Then, trim notes so they don't overlap
    for note_list in all_notes:
        if note_list != []:
            note_list = sorted(note_list,key=lambda x: x.start)
            for note_1,note_2 in zip(note_list[:-1],note_list[1:]):
                note_1.end = min(note_1.end,note_2.start)
                new_instr.notes += [note_1]
            new_instr.notes += [note_list[-1]]

    new_midi.instruments.append(new_instr)

    return new_midi


def import_features(results,features_to_use):

    all_feat = []
    for feat in features_to_use:
        if feat == 'valid_cons':
            value_hut78 = results["cons_hut78_output"]
            value_har18 = results["cons_har18_output"]
            value_har19 = results["cons_har19_output"]

            value = value_hut78[:-1]+value_har18[0:2]+value_har18[3:]+value_har19[:-2]

        else:
            value = results[feat]
        if type(value) is tuple:
            all_feat += list(value)
        elif type(value) is list:
            all_feat += value
        elif type(value) is float:
            all_feat += [np.float64(value)]
        else:
            all_feat += [value]

    all_feat = [float(elt) for elt in all_feat]


    return all_feat
