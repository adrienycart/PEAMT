import numpy as np
import random
import tensorflow as tf



def get_feature_labels(features_to_use):

    def get_feat_names(feat):
        if feat == 'framewise_0.01':
            return ["framewise_P","framewise_R","framewise_F"]
        elif 'notewise_On_' in feat:
            return ["notewise_On_P",
            "notewise_On_R",
            "notewise_On_F"]
        elif 'notewise_OnOff_' in feat:
            return ["notewise_OnOff_P",
                "notewise_OnOff_R",
                "notewise_OnOff_F"]
        elif feat == "high_f":
            return ["high_f_P",
            "high_f_R",
            "high_f_F",]
        elif feat == "low_f":
            return ["low_f_P",
            "low_f_R",
            "low_f_F",]
        elif feat == "high_n":
            return ["high_n_P",
            "high_n_R",
            "high_n_F",]
        elif feat == "low_n":
            return ["low_n_P",
            "low_n_R",
            "low_n_F"]
        elif feat == "loud_fn":
            return ["loud_fn"]
        elif feat == "loud_ratio_fn":
            return ["loud_ratio_fn"]
        elif feat == "out_key":
            return ["out_key_fp",
            "out_key_all",]
        elif feat == "out_key_bin":
            return ["out_key_bin_fp",
            "out_key_bin_all",]
        elif feat == "repeat":
            return ["repeat_fp",
            "repeat_all",]
        elif feat == "merge":
            return ["merge_fp",
            "merge_all",]
        elif feat == "semitone_f":
            return ["semitone_f_fp",
            "semitone_f_all"]
        elif feat == "octave_f":
            return ["octave_f_fp",
            "octave_f_all"]
        elif feat == "third_harmonic_f":
            return ["third_harmonic_f_fp",
            "third_harmonic_f_all",]
        elif feat == "semitone_n":
            return ["semitone_n_fp",
            "semitone_n_all",]
        elif feat == "octave_n":
            return ["octave_n_fp",
            "octave_n_all",]
        elif feat == "third_harmonic_n":
            return ["third_harmonic_n_fp",
            "third_harmonic_n_all",]
        elif feat == "poly_diff":
            return ['poly_diff_mean','poly_diff_std','poly_diff_min','poly_diff_max']
        elif feat == "rhythm_hist":
            return ['rhythm_hist_out','rhythm_hist_diff']
        elif feat == "rhythm_disp_std":
            return ['rhythm_disp_std_mean','rhythm_disp_std_min','rhythm_disp_std_max']
        elif feat == "rhythm_disp_drift":
            return ['rhythm_disp_drift_mean','rhythm_disp_drift_min','rhythm_disp_drift_max']
        elif feat == 'cons_hut78_output':
            return ['cons_hut78_output_mean','cons_hut78_output_std','cons_hut78_output_max','cons_hut78_output_min']
        elif feat == 'cons_har18_output':
            return ['cons_har18_output_mean','cons_har18_output_std','cons_har18_output_max','cons_har18_output_min']
        elif feat == 'cons_har19_output':
            return ['cons_har19_output_mean','cons_har19_output_std','cons_har19_output_max','cons_har19_output_min']
        elif feat == 'cons_hut78_diff':
            return ['cons_hut78_diff_mean','cons_hut78_diff_std','cons_hut78_diff_max','cons_hut78_diff_min']
        elif feat == 'cons_har18_diff':
            return ['cons_har18_diff_mean','cons_har18_diff_std','cons_har18_diff_max','cons_har18_diff_min']
        elif feat == 'cons_har19_diff':
            return ['cons_har19_diff_mean','cons_har19_diff_std','cons_har19_diff_max','cons_har19_diff_min']
        elif feat == 'valid_cons':
            return ['cons_hut78_output_mean','cons_hut78_output_std','cons_hut78_output_max','cons_har18_output_mean','cons_har18_output_std','cons_har18_output_min','cons_har19_output_mean','cons_har19_output_std']
        else:
            raise ValueError('Feature not understood! '+feat)

    return sum([get_feat_names(feat) for feat in features_to_use],[])

def get_y_z_alphas(ratings):

    y = (ratings==int(round(MAX_ANSWERS/2))).astype(int)
    z = (ratings>int(round(MAX_ANSWERS/2))).astype(int)
    # alpha is between 0 and 0.5
    alphas = np.abs(ratings-int(round(MAX_ANSWERS/2)))/int(round(MAX_ANSWERS))

    return y,z,alphas

def linear_regression_model(features,weights,bias,pca_matrix=None):
    #features_o and features_t are of shape: [batch_size, n_features]

    if pca_matrix is None:
        output = tf.sigmoid(tf.matmul(features, weights)+bias)
    else:
        output = tf.sigmoid(tf.matmul(tf.matmul(features,tf.cast(pca_matrix,tf.float32),transpose_b=True), weights)+bias)

    return output


def contrastive_loss(batch1,batch2,y,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally similar

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-tf.abs(batch1-batch2),0))
    return tf.reduce_mean(loss)

def contrastive_loss_magnitude(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-tf.abs(batch1-batch2),0)) + \
           (1-y)*(z*tf.square(batch1) + (1-z)*tf.square(batch2))
    return tf.reduce_mean(loss)

def contrastive_loss_absolute(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-z*(batch2-batch1)-(1-z)*(batch1-batch2),0))
    return tf.reduce_mean(loss)

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

def shuffle(*args):
    assert all([arg.shape[0] == args[0].shape[0] for arg in args])
    n = args[0].shape[0]
    shuffle_idx = list(range(n))
    random.shuffle(shuffle_idx)
    output = []
    for arg in args:
        output += [arg[shuffle_idx]]
    return output

def sample(n_samples,*args):
    assert all([arg.shape[0] == args[0].shape[0] for arg in args])
    n = args[0].shape[0]
    sample_idx = list(range(n))
    sample_idx = random.sample(sample_idx, n_samples)
    output = []
    for arg in args:
        output += [arg[sample_idx]]
    return output

def split_data(ranges,*args):
    output = []
    for arg in args:
        out = []
        for start,end in ranges:
            out += [arg[start:end]]
        output += [np.concatenate(out,axis=0)]
    return output
