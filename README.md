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
# Getting started

This library is compatible with Python 2.7 and 3.6 **TODO: check compatibility**

To install the dependencies necessary to run the evaluation, run : ```$ pip install -r requirements.txt```

To install the dependencies necessary to train your own metric, run: ```$ pip install -r requirements_training.txt```

# Evaluating some outputs

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


# Training a new metric


To train a new metric, you first need to download the ratings and MIDI files from: **ADD LINK WHEN DATA IS UPLOADED**
Then, run ```$ python export_features.py <MIDI_files_location> <answers_csv_path> <output_folder>``` to precompute the features and save them, for each pair of (target, output).

Then use the script ```classifier.py```.
Create a new ```@ex.named_config``` on the same model as ```export_metric```.
Edit the entries ```features_to_use``` and ```features_to_remove``` of the ```cfg``` dictionary to specify the features to include or leave out (respectively).

To run the script, run ```$ python classifier with <your_config_name>```.

To use the obtained parameters to evaluate some outputs, use: ```eval = PEAMT(parameters=<your_parameters_filepath>)```.

# Using individual features

The individual features can also be used: **TODO: Make sure this is actually working**

```
import pretty_midi as pm

from peamt.features.rhythm import rhythm_histogram
import peamt.features.utils as utils


output_filename = 'output.mid'
target_filename = 'target.mid'

output_data = pm.PrettyMIDI(output_filename)
target_data = pm.PrettyMIDI(output_filename)

notes_output, intervals_output = utils.get_notes_intervals(output_data)
notes_target, intervals_target = utils.get_notes_intervals(target_data)

rhythm_hist_out, rhythm_hist_diff = rhythm_histogram(intervals_output,intervals_target)
```
