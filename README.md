# PEAMT

This repository contains code to train and run PEAMT, a Perceptual Evaluation metric for Automatic Music Transcription.
If you use any of this, please cite:

Adrien Ycart, Lele Liu, Emmanouil Benetos and Marcus Pearce, 2020. ["Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription"](https://transactions.ismir.net/articles/10.5334/tismir.57), _Transactions of the International Society for Music Information Retrieval (TISMIR)_, 3(1), pp.68–81.

```  
    @article{ycart2019PEAMT,
       Author = {Ycart, Adrien and Liu, Lele and Benetos, Emmanouil and Pearce, Marcus},    
       Booktitle = {Transactions of the International Society for Music Information Retrieval (TISMIR)},    
       Title = {Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription},       
       Year = {2020},
       Volume = {3},
       Issue = {1},
       Pages = {68--81},
       DOI = {http://doi.org/10.5334/tismir.57},
    }  
```
## Getting started

This library is compatible with Python 2.7 and 3.6.

To install the dependencies necessary to run the evaluation, run : ```$ pip install -r requirements.txt```

To install the dependencies necessary to train your own metric, run: ```$ pip install -r requirements_training.txt```

## Evaluating some outputs

Here is a code sample to evaluate some outputs:

```
from peamt import PEAMT

output_filename = 'output.mid'
target_filename = 'target.mid'

eval = PEAMT()
value = eval.evaluate_from_midi(target_filename,output_filename)
```

```eval.evaluate_from_midi``` can take as input a string or a PrettyMIDI object.

Make sure the target filename contains correct velocities and sustain pedal activations (control change &#35;64).

To run directly from lists of notes, use ```eval.evaluate```.
Some useful processing functions can be found in ```features.utils```, such as
```apply_sustain_control_changes``` and ```get_notes_intervals```.


## Training a new metric


To train a new metric, you first need to download the ratings and MIDI files [at this address](https://zenodo.org/record/3746863).
Then, run ```$ python export_features.py <MIDI_files_location> <answers_csv_path> <output_folder>``` to precompute the features and save them, for each pair of (target, output).

Then use the script ```classifier.py```.
Create a new ```@ex.named_config``` on the same model as ```export_metric```.
Edit the entries ```features_to_use``` and ```features_to_remove``` of the ```cfg``` dictionary to specify the features to include or leave out (respectively).

To run the script, run ```$ python classifier with <your_config_name>```.

To use the obtained parameters to evaluate some outputs, use: ```eval = PEAMT(parameters=<your_parameters_filepath>)```.

## Using individual features

The individual features can also be used:

```
import pretty_midi as pm

from features.rhythm import rhythm_histogram
import features.utils as utils


output_filename = 'output.mid'
target_filename = 'target.mid'

output_data = pm.PrettyMIDI(output_filename)
target_data = pm.PrettyMIDI(target_filename)

notes_output, intervals_output = utils.get_notes_intervals(output_data)
notes_target, intervals_target = utils.get_notes_intervals(target_data)

rhythm_hist_out, rhythm_hist_diff = rhythm_histogram(intervals_output,intervals_target)
```

## More info

For more info on the way perceptual data was collected and on the way the metric is trained, please refer to [the corresponding paper](TODO).

For a detailed definition of each of the features, please refer to: [Technical report - Musical Features for Automatic Music Transcription Evaluation](https://arxiv.org/abs/2004.07171)


## Weights of features in the released metric

As an indication, we give here the weights of each feature in the released metric, ranked by
magnitude.
It appears that the most important features are the notewise onset-only F-measure, followed by rhythm features and other benchmark metrics.
Framewise mistakes in the lowest voice also seem to have some importance in the end result.

|Rank | Feature ID        | Weight           |
|-----| -------------: |-------------|
| 0 | notewise_On_F | 0.6163901 |
| 1 | rhythm_disp_drift_mean | 0.58185416 |
| 2 | rhythm_disp_drift_max | -0.51437545 |
| 3 | framewise_R | 0.3728488 |
| 4 | rhythm_disp_std_mean | -0.36873978 |
| 5 | rhythm_hist_out | -0.28920627 |
| 6 | low_f_F | -0.27877852 |
| 7 | notewise_OnOff_R | -0.25945878 |
| 8 | framewise_F | 0.21865314 |
| 9 | rhythm_hist_diff | 0.21851824 |
| 10 | low_f_R | 0.1989684 |
| 11 | notewise_On_R | -0.1954612 |
| 12 | low_f_P | 0.19219641 |
| 13 | poly_diff_mean | 0.18691508 |
| 14 | rhythm_disp_drift_min | -0.18283454 |
| 15 | high_n_F | 0.16963542 |
| 16 | merge_fp | -0.16701007 |
| 17 | high_f_F | -0.15762798 |
| 18 | notewise_OnOff_F | 0.1448518 |
| 19 | poly_diff_std | -0.14053902 |
| 20 | merge_all | 0.109077565 |
| 21 | loud_ratio_fn | -0.10902009 |
| 22 | high_f_R | 0.10647323 |
| 23 | low_n_F | -0.10201532 |
| 24 | low_n_R | 0.09830448 |
| 25 | repeat_fp | -0.097474664 |
| 26 | high_n_P | 0.08767299 |
| 27 | poly_diff_min | -0.07279301 |
| 28 | loud_fn | -0.07112731 |
| 29 | notewise_On_P | -0.060689207 |
| 30 | notewise_OnOff_P | 0.05645829 |
| 31 | poly_diff_max | 0.05339441 |
| 32 | low_n_P | 0.047426596 |
| 33 | high_f_P | -0.045074914 |
| 34 | high_n_R | 0.041347794 |
| 35 | rhythm_disp_std_max | 0.034391385 |
| 36 | framewise_P | -0.024282124 |
| 37 | rhythm_disp_std_min | 0.012041504 |
| 38 | repeat_all | -0.0018782051 |
