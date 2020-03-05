import os
import numpy as np
import math
import pretty_midi as pm
import mir_eval
import pickle
# import matplotlib.pyplot as plt
from utils import apply_sustain_control_changes, make_roll
from classifier_utils import import_features
import features.utils as utils
from features.benchmark import framewise, notewise
from features.high_low_voice import framewise_highest, framewise_lowest, notewise_highest, notewise_lowest
from features.loudness import false_negative_loudness, loudness_ratio_false_negative
from features.out_key import make_key_mask, out_key_errors, out_key_errors_binary_mask
from features.polyphony import polyphony_level_diff
from features.repeat_merge import repeated_notes, merged_notes
from features.specific_pitch import specific_pitch_framewise, specific_pitch_notewise
from features.rhythm import rhythm_histogram, rhythm_dispersion


ALL_FEATURES = [
                 "framewise_0.01",
                 "notewise_On_50",
                 "notewise_OnOff_50_0.2",
                 "high_f",
                 "low_f",
                 "high_n",
                 "low_n",

                 "loud_fn",
                 "loud_ratio_fn",

                 "out_key",
                 "out_key_bin",

                 "repeat",
                 "merge",

                 "poly_diff",

                 "rhythm_hist",
                 "rhythm_disp_std",
                 "rhythm_disp_drift",

                 ]

class PEAMT():

    def __init__(self,parameters='default_metric_params.pkl'):

        parameters = pickle.load(open(parameters), "rb")

        self.weight = parameters['weights']
        self.bias = parameters['bias']

    def compute_from_features(self,features):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        value = np.multiply(features,self.weight) + self.bias
        return sigmoid(value)


    def evaluate(self,ref_midi, est_midi):

        if type(ref_midi) is pm.PrettyMIDI:
            pass
        elif type(ref_midi) is str:
            ref_midi = pm.PrettyMIDI(ref_midi)
        else:
            raise ValueError("Type of 'ref_midi' arg not understood! Should be str or PrettyMIDI object.")

        if type(est_midi) is pm.PrettyMIDI:
            pass
        elif type(est_midi) is str:
            est_midi = pm.PrettyMIDI(est_midi)
        else:
            raise ValueError("Type of 'est_midi' arg not understood! Should be str or PrettyMIDI object.")



        notes_target_no_pedal, intervals_target_no_pedal = utils.get_notes_intervals(ref_midi,with_vel=True)
        midi_sustain = apply_sustain_control_changes(ref_midi)
        notes_target, intervals_target = utils.get_notes_intervals(midi_sustain,with_vel=True)
        notes_output, intervals_output = utils.get_notes_intervals(est_midi)

        ### Validate values
        mir_eval.transcription.validate_intervals(ref_intervals,est_intervals)
        mir_eval.transcription.validate_intervals(ref_intervals_sustain,est_intervals)
        if np.any(est_pitches!=est_pitches.astype(int)):
            raise ValueError('est_midi pitches should all be integers!')
        if np.any(ref_pitches!=ref_pitches.astype(int)):
            raise ValueError('ref_midi pitches should all be integers!')


        ### Create piano rolls
        fs=100

        max_len = max(np.max(ref_instervals),np.max(ref_instervals_sustain),np.max(est_intervals))
        max_pitch = int(max(np.max(ref_pitches),np.max(ref_pitches_sustain),np.max(est_pitches)))+1
        roll_shape = [max_pitch+1,max_len+1]


        output = make_roll(est_intervals,est_pitches,roll_shape)
        target_no_pedal = make_roll(ref_intervals,ref_pitches,roll_shape)
        target = make_roll(ref_intervals_sustain,ref_pitches,roll_shape)


        ### Compute features
        frame = framewise(output,target)

        match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output, onset_tolerance=0.05, offset_ratio=None, pitch_tolerance=0.25)

        # Use onset-only matchings by default
        match = match_on

        note_on = notewise(match_on,notes_output,notes_target)

        match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output,onset_tolerance=0.05, offset_ratio=0.2, pitch_tolerance=0.25)
        note_onoff = notewise(match_onoff,notes_output,notes_target)

        high_f = framewise_highest(output, target_no_pedal)
        low_f = framewise_lowest(output, target_no_pedal)

        high_n = notewise_highest(notes_output, intervals_output, notes_target_no_pedal, intervals_target_no_pedal, match)
        low_n = notewise_lowest(notes_output, intervals_output, notes_target_no_pedal, intervals_target_no_pedal, match)

        loud_fn = false_negative_loudness(match, vel_target, intervals_target)
        loud_ratio_fn = loudness_ratio_false_negative(notes_target, intervals_target, vel_target, match)

        mask = make_key_mask(target_no_pedal)
        out_key = out_key_errors(notes_output, match, mask)
        out_key_bin = out_key_errors_binary_mask(notes_output, match, mask)

        repeat = repeated_notes(notes_output, intervals_output, notes_target, intervals_target, match)
        merge = merged_notes(notes_output, intervals_output, notes_target, intervals_target, match)

        ### Specific pitch features are not included, we keep this code as example
        # semitone_f = specific_pitch_framewise(output, target, fs, 1)
        # octave_f = specific_pitch_framewise(output, target, fs, 12)
        # third_harmonic_f = specific_pitch_framewise(output, target, fs, 19,down_only=True)
        # semitone_n = specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match_on, n_semitones=1)
        # octave_n = specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match_on, n_semitones=12)
        # third_harmonic_n = specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match_on, n_semitones=19,down_only=True)

        poly_diff = polyphony_level_diff(output,target)

        rhythm_hist = rhythm_histogram(intervals_output,intervals_target)
        rhythm_disp_std,rhythm_disp_drift = rhythm_dispersion(intervals_output, intervals_target)

        results_dict.update({
                "framewise_0.01" : frame,

                "notewise_On_50" : note_on,
                "notewise_OnOff_50_0.2": note_onoff,

                "high_f": high_f,
                "low_f": low_f,
                "high_n": high_n,
                "low_n":low_n,

                "loud_fn":loud_fn,
                "loud_ratio_fn":loud_ratio_fn,

                "out_key":out_key,
                "out_key_bin":out_key_bin,

                "repeat":repeat,
                "merge":merge,

                "poly_diff": poly_diff,

                "rhythm_hist": rhythm_hist,
                "rhythm_disp_std": rhythm_disp_std,
                "rhythm_disp_drift": rhythm_disp_drift,

                })

        feature_list = import_features(results_dict, ALL_FEATURES)

        return self.compute_from_features(feature_list)
