import os
import numpy as np
import pretty_midi as pm
import mir_eval
import features.utils as utils
from features.benchmark import framewise
from features.high_low_voice import framewise_highest, framewise_lowest,notewise_highest, notewise_lowest
from features.loudness import false_negative_loudness
from features.out_key import out_key_errors, out_key_errors_binary_mask
from features.repeat_merge import repeated_notes, merged_notes



MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz','lisu','google','cheng']


for example in os.listdir(MIDI_path)[0:10]:
    example_path = os.path.join(MIDI_path,example)

    print('________________')
    print(example)

    target_data = pm.PrettyMIDI(os.path.join(example_path,'target.mid'))
    for system in systems:
        # if example=="MAPS_MUS-scn15_11_ENSTDkAm_5" and system=='google':
        if system=='google' or system == 'kelz':
            print(system)
            system_data = pm.PrettyMIDI(os.path.join(example_path,system+'.mid'))

            target_pr = (target_data.get_piano_roll()>0).astype(int)
            system_pr = (system_data.get_piano_roll()>0).astype(int)

            target_pr,system_pr= utils.even_up_rolls([target_pr,system_pr])

            P_f,R_f,F_f = framewise(system_pr,target_pr)
            print("Frame P,R,F:", P_f,R_f,F_f)

            notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data,with_vel=True)
            notes_system, intervals_system = utils.get_notes_intervals(system_data)


            match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system,offset_ratio=None, pitch_tolerance=0.25)
            match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system,offset_ratio=0.2, pitch_tolerance=0.25)

            P_n_on = float(len(match_on))/len(notes_system)
            R_n_on = float(len(match_on))/len(notes_target)
            F_n_on = 2*P_n_on*R_n_on/(P_n_on+R_n_on+np.finfo(float).eps)
            print("Note-On P,R,F:", P_n_on,R_n_on,F_n_on)

            P_n_onoff = float(len(match_onoff))/len(notes_system)
            R_n_onoff = float(len(match_onoff))/len(notes_target)
            F_n_onoff = 2*P_n_onoff*R_n_onoff/(P_n_onoff+R_n_onoff+np.finfo(float).eps)
            print("Note-OnOff P,R,F:", P_n_onoff,R_n_onoff,F_n_onoff)

            # notewise_highest(notes_system,intervals_system,notes_targ            mask = make_key_mask(target_pr)et,intervals_target,match_onoff)

            _,_,repeated = repeated_notes(intervals_target,notes_system,intervals_system,match_on)

            if len(repeated)>0 or True:
                import matplotlib.pyplot as plt
                fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
                for i in range(len(notes_target)):
                    target_pr[notes_target[i],int(intervals_target[i][0]*100)] += 1
                ax0.imshow(target_pr,aspect='auto',origin='lower')
                for i in range(len(notes_system)):
                    if notes_system[i] == 62:
                        print intervals_system[i]
                    system_pr[notes_system[i],int(intervals_system[i][0]*100)] += 1
                ax1.imshow(system_pr,aspect='auto',origin='lower')
                display1 = np.zeros_like(system_pr)
                display2 = np.zeros_like(system_pr)
                matched_targets,matched_outputs = zip(*match_on)
                for i in matched_targets:
                    display1[notes_target[i],int(intervals_target[i][0]*100):int(intervals_target[i,1]*100)] = 1
                    display1[notes_target[i],int(intervals_target[i][0]*100)] += 1
                for i in matched_outputs:
                    display2[notes_system[i],int(intervals_system[i][0]*100):int(intervals_system[i,1]*100)] = 1
                    display2[notes_system[i],int(intervals_system[i][0]*100)] += 1
                for i in repeated:
                    display2[notes_system[i],int(intervals_system[i][0]*100):int(intervals_system[i,1]*100)] = 3
                    display2[notes_system[i],int(intervals_system[i][0]*100)] += 1

                ax2.imshow(display1,aspect='auto',origin='lower')
                ax3.imshow(display2,aspect='auto',origin='lower')
                plt.show()

            # print match




############
##### TESTS
############
