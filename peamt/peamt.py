from __future__ import absolute_import
import os
import numpy as np
import math
import pretty_midi as pm
import mir_eval
import pickle
# import matplotlib.pyplot as plt
import peamt.features.utils as utils
from peamt.features.benchmark import framewise, notewise
from peamt.features.high_low_voice import framewise_highest, framewise_lowest, notewise_highest, notewise_lowest
from peamt.features.loudness import false_negative_loudness, loudness_ratio_false_negative
from peamt.features.out_key import make_key_mask, out_key_errors, out_key_errors_binary_mask
from peamt.features.polyphony import polyphony_level_diff
from peamt.features.repeat_merge import repeated_notes, merged_notes
from peamt.features.specific_pitch import specific_pitch_framewise, specific_pitch_notewise
from peamt.features.rhythm import rhythm_histogram, rhythm_dispersion

DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_parameters/PEAMT.pkl')

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

                 # "out_key",
                 # "out_key_bin",

                 "repeat",
                 "merge",

                 "poly_diff",

                 "rhythm_hist",
                 "rhythm_disp_std",
                 "rhythm_disp_drift",

                 ]

class PEAMT():

    def __init__(self,parameters=DEFAULT_PARAMS):

        try:
            # Python 3
            parameters = pickle.load(open(parameters, "rb"), encoding='latin1')
        except TypeError:
            # Python 2
            parameters = pickle.load(open(parameters, "rb"))

        self.weight = parameters['best_weights']
        self.bias = parameters['best_bias']
        self.data_mean = parameters['data_mean']
        self.data_std = parameters['data_std']

    def normalise(self,features):
        return (features-self.data_mean)/self.data_std

    def compute_from_features(self,features):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        features = self.normalise(features)
        value = np.matmul(features,self.weight) + self.bias
        return sigmoid(value)

    def evaluate_from_midi(self,ref_midi, est_midi):
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

        notes_target_no_pedal, intervals_target_no_pedal, vel_target_no_pedal = utils.get_notes_intervals(ref_midi,with_vel=True)
        midi_sustain = utils.apply_sustain_control_changes(ref_midi)
        notes_target, intervals_target, vel_target = utils.get_notes_intervals(midi_sustain,with_vel=True)
        notes_output, intervals_output = utils.get_notes_intervals(est_midi)

        return self.evaluate(notes_output, intervals_output, notes_target, intervals_target, vel_target, notes_target_no_pedal, intervals_target_no_pedal)


    def evaluate(self,notes_output, intervals_output, notes_target, intervals_target, vel_target, notes_target_no_pedal, intervals_target_no_pedal):

        ### Validate values
        mir_eval.transcription.validate_intervals(intervals_target,intervals_output)
        mir_eval.transcription.validate_intervals(intervals_target_no_pedal,intervals_output)
        if np.any(notes_output!=notes_output.astype(int)):
            raise ValueError('notes_output should all be integers!')
        if np.any(notes_target!=notes_target.astype(int)):
            raise ValueError('notes_target should all be integers!')
        if np.any(notes_target_no_pedal!=notes_target_no_pedal.astype(int)):
            raise ValueError('notes_target_no_pedal should all be integers!')


        ### Create piano rolls
        fs=100

        max_len = int(max(np.max(intervals_target),np.max(intervals_target_no_pedal),np.max(intervals_output)))+1
        max_pitch = int(max(np.max(notes_output),np.max(notes_target),np.max(notes_target_no_pedal)))+1
        roll_shape = [max_pitch+1,max_len+1]


        output = utils.make_roll(intervals_output,notes_output,roll_shape)
        target_no_pedal = utils.make_roll(intervals_target_no_pedal,notes_target_no_pedal,roll_shape)
        target = utils.make_roll(intervals_target,notes_target,roll_shape)


        ### Compute features
        frame = framewise(output,target)

        match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output, onset_tolerance=0.05, offset_ratio=None, pitch_tolerance=0.25)
        match_on_no_pedal = mir_eval.transcription.match_notes(intervals_target_no_pedal, notes_target_no_pedal, intervals_output, notes_output, onset_tolerance=0.05, offset_ratio=None, pitch_tolerance=0.25)

        # Use onset-only matchings by default
        match = match_on

        note_on = notewise(match_on,notes_output,notes_target)

        match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output,onset_tolerance=0.05, offset_ratio=0.2, pitch_tolerance=0.25)
        note_onoff = notewise(match_onoff,notes_output,notes_target)

        high_f = framewise_highest(output, target_no_pedal)
        low_f = framewise_lowest(output, target_no_pedal)

        high_n = notewise_highest(notes_output, intervals_output, notes_target_no_pedal, intervals_target_no_pedal, match_on_no_pedal)
        low_n = notewise_lowest(notes_output, intervals_output, notes_target_no_pedal, intervals_target_no_pedal, match_on_no_pedal)

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

        results_dict = {
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

                }

        feature_list = utils.import_features(results_dict, ALL_FEATURES)

        return self.compute_from_features(feature_list)
