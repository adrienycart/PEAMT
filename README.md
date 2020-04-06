# PEAMT

This repository contains code to train and run PEAMT, a Perceptual Evaluation metric for Automatic Music Transcription.
If you use any of this, please cite:

Adrien Ycart, Lele Liu, Emmanouil Benetos and Marcus Pearce. "Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription", _Transactions of the International Society for Music Information Retrieval (TISMIR)_, Under Review.

```  
    @article{ycart2019PEAMT,
       Author = {Ycart, Adrien and Liu, Lele and Benetos, Emmanouil and Pearce, Marcus},    
       Booktitle = {Transactions of the International Society for Music Information Retrieval (TISMIR)},    
       Title = {Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription},       
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


To train a new metric, you first need to download the ratings and MIDI files from: **ADD LINK WHEN DATA IS UPLOADED**
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

For more info on the way perceptual data was collected and on the way the metric is trained, please refer to: **ADD LINK TO PAPER**

For detailed definition of each of the features, please refer to: **ADD LINK TO FEATURE REPORT**


## Weights of features in the released metric

As an indication, we give here the weights of each feature in the released metric, ranked by
magnitude.
It appears that the most important features are the notewise onset-only F-measure, followed by rhythm features and other benchmark metrics.
Framewise mistakes in the lowest voice also seem to have some importance in the end result.

| Feature ID        | Weight           |
| -------------: |-------------|
| notewise_On_F | 0.6163901 |
| rhythm_disp_drift_mean | 0.58185416 |
| rhythm_disp_drift_max | -0.51437545 |
| framewise_R | 0.3728488 |
| rhythm_disp_std_mean | -0.36873978 |
| rhythm_hist_out | -0.28920627 |
| low_f_F | -0.27877852 |
| notewise_OnOff_R | -0.25945878 |
| framewise_F | 0.21865314 |
| rhythm_hist_diff | 0.21851824 |
| low_f_R | 0.1989684 |
| notewise_On_R | -0.1954612 |
| low_f_P | 0.19219641 |
| poly_diff_mean | 0.18691508 |
| rhythm_disp_drift_min | -0.18283454 |
| high_n_F | 0.16963542 |
| merge_fp | -0.16701007 |
| high_f_F | -0.15762798 |
| notewise_OnOff_F | 0.1448518 |
| poly_diff_std | -0.14053902 |
| merge_all | 0.109077565 |
| loud_ratio_fn | -0.10902009 |
| high_f_R | 0.10647323 |
| low_n_F | -0.10201532 |
| low_n_R | 0.09830448 |
| repeat_fp | -0.097474664 |
| high_n_P | 0.08767299 |
| poly_diff_min | -0.07279301 |
| loud_fn | -0.07112731 |
| notewise_On_P | -0.060689207 |
| notewise_OnOff_P | 0.05645829 |
| poly_diff_max | 0.05339441 |
| low_n_P | 0.047426596 |
| high_f_P | -0.045074914 |
| high_n_R | 0.041347794 |
| rhythm_disp_std_max | 0.034391385 |
| framewise_P | -0.024282124 |
| rhythm_disp_std_min | 0.012041504 |
| repeat_all | -0.0018782051 |
