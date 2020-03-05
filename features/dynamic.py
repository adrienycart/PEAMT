import numpy as np
import os
from statistics import stdev
import pandas as pd
import csv
from .utils import create_folder
from dissonant import harmonic_tone, dissonance, pitch_to_freq

def get_pitch(full_note):
    return full_note[0]

def get_onset(full_note):
    return full_note[1]

def get_offset(full_note):
    return full_note[2]

def weighted_std(values, weighted_mean, weights):
    std = np.sqrt(sum([(values[idx] - weighted_mean)**2 * weights[idx] for idx in range(len(values))]) / sum(weights))
    return std

def pad_chords(chords, pad_value=-1):
    max_poly = max(len(chord) for chord in chords)
    for chord in chords:
        chord.extend([pad_value] * (max_poly - len(chord)))
    return chords

def unpad_chords(chords, pad_value=-1):
    unpad = []
    for chord in chords:
        unpad.append([p for p in chord if p != pad_value])
    return unpad

def save_to_csv(filename, data):
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            try:
                writer.writerow(row)
            except:
                writer.writerow([row])

def read_from_csv(filename, dtype=float):
    df = pd.read_csv(filename)
    return df.values.tolist()

def get_event_based_sequence(notes, intervals, example, system, dt=0.05):

    folder = "features/chords_and_times/" + example + "/"

    # if precalculated, simple load values
    if os.path.isfile(folder + system + "_chords.csv"):
        chords = read_from_csv(folder + system + "_chords.csv", dtype=int)
        chords = unpad_chords(chords)
        event_times = read_from_csv(folder + system + "_event_times.csv", dtype=float)
        event_times = [x[0] for x in event_times]
        durations = read_from_csv(folder + system + "_durations.csv", dtype=float)
        durations = [x[0] for x in durations]
        return chords, event_times, durations

    # if not pre-calculated, calculate chords and event times.
    full_notes = [(notes[idx], intervals[idx][0], intervals[idx][1]) for idx in range(len(notes))]
    full_notes.sort(key=get_onset)
    all_times = []
    for interval in intervals:
        all_times.extend(interval)
    all_times.sort()

    # get event times from the onsets and offsets
    last_time = -1.0
    event_times = []
    for time in all_times:
        if time - last_time > dt:
            event_times.append(time)
            last_time = time

    chords = []
    for idx in range(len(event_times)-1):
        chord = []
        for note in full_notes:
            if get_onset(note) < event_times[idx] + dt and get_offset(note) >= event_times[idx+1] - dt and not get_pitch(note) in set(chord):
                chord.append(get_pitch(note))
        chords.append(chord)

    # from utils import make_note_index_matrix
    # matrix = make_note_index_matrix(notes, intervals)
    # import matplotlib.pyplot as plt
    # plt.imshow(matrix)
    # plt.show()

    durations = [event_times[idx+1] - event_times[idx] for idx in range(len(chords))]

    create_folder(folder)
    save_to_csv(folder + system + "_chords.csv", pad_chords(chords))
    save_to_csv(folder + system + "_event_times.csv", event_times)
    save_to_csv(folder + system + "_durations.csv", durations)
    # print(system + 'saved.')

    return chords, event_times, durations


def chord_dissonance(notes_output, intervals_output, notes_target, intervals_target, example, system):

    chords_target, event_times_target, durations_target = get_event_based_sequence(notes_target, intervals_target, example, "target")
    chords_output, event_times_output, durations_output = get_event_based_sequence(notes_output, intervals_output, example, system)

    dissonances_target = []
    for chord in chords_target:
        if len(chord) == 0:
            dissonances_target.append(0.0)
        else:
            freqs, amps = harmonic_tone(pitch_to_freq(chord), n_partials=10)
            dissonances_target.append(dissonance(freqs, amps, model='sethares1993'))

    dissonances_output = []
    for chord in chords_output:
        if len(chord) == 0:
            dissonances_output.append(0.0)
        else:
            freqs, amps = harmonic_tone(pitch_to_freq(chord), n_partials=10)
            dissonances_output.append(dissonance(freqs, amps, model='sethares1993'))

    ave_dissonance_target = np.average(dissonances_target, weights=durations_target)
    ave_dissonance_output = np.average(dissonances_output, weights=durations_output)

    std_dissonance_target = weighted_std(dissonances_target, ave_dissonance_target, durations_target)
    std_dissonance_output = weighted_std(dissonances_output, ave_dissonance_output, durations_output)

    return (
        ave_dissonance_target,
        ave_dissonance_output,
        std_dissonance_target,
        std_dissonance_output,
        max(dissonances_target),
        max(dissonances_output),
        min(x for x in dissonances_target if x > 0.0),
        min(x for x in dissonances_output if x > 0.0)
    )


def polyphony_level(notes_output, intervals_output, notes_target, intervals_target, example, system):

    chords_target, event_times_target, durations_target = get_event_based_sequence(notes_target, intervals_target, example, "target")
    chords_output, event_times_output, durations_output = get_event_based_sequence(notes_output, intervals_output, example, system)

    polyphony_levels_target = [len(chord) for chord in chords_target]
    polyphony_levels_output = [len(chord) for chord in chords_output]

    # weighted averages and stds
    ave_polyphony_level_target = np.average(polyphony_levels_target, weights=durations_target)
    ave_polyphony_level_output = np.average(polyphony_levels_output, weights=durations_output)
    std_polyphony_level_target = weighted_std(polyphony_levels_target, ave_polyphony_level_target, durations_target)
    std_polyphony_level_output = weighted_std(polyphony_levels_output, ave_polyphony_level_output, durations_output)

    # unweighted mean and stds.
    mean_target = np.mean(polyphony_levels_target)
    mean_output = np.mean(polyphony_levels_output)
    std_target = stdev(polyphony_levels_target)
    std_output = stdev(polyphony_levels_output)

    # import matplotlib.pyplot as plt
    # plt.plot(event_times_target, polyphony_levels_target + [0], label='target')
    # plt.plot(event_times_output, polyphony_levels_output + [0], label='output')
    # plt.legend()
    # plt.show()

    return (
        # weighted
        ave_polyphony_level_target,
        ave_polyphony_level_output,
        std_polyphony_level_target,
        std_polyphony_level_output,
        # unweighted
        mean_target,
        mean_output,
        std_target,
        std_output,
        max(polyphony_levels_target),
        max(polyphony_levels_output),
        min(polyphony_levels_target),
        min(polyphony_levels_output)
    )
