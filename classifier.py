from sacred import Experiment
import codecs
import os
import random
import cPickle as pickle

import numpy as np
import tensorflow as tf
try:
    import matplotlib.pyplot as plt
except:
    print "No matplotlib!"

from sklearn.decomposition import PCA

import classifier_utils as utils

MAX_ANSWERS=4

ex = Experiment('AMTClassifier')

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

                 "semitone_f",
                 "octave_f",
                 "third_harmonic_f",
                 "semitone_n",
                 "octave_n",
                 "third_harmonic_n",

                 "poly_diff",

                 "rhythm_hist",
                 "rhythm_disp_std",
                 "rhythm_disp_drift",

                 # "cons_hut78_output",
                 # "cons_har18_output",
                 # "cons_har19_output",
                 #
                 # "cons_hut78_diff" ,
                 # "cons_har18_diff" ,
                 # "cons_har19_diff" ,

                 "valid_cons",
                 ]

@ex.config
def cfg():
    cfg = {"base_folder" : "results_metric", # Base folder for model checkpoints
           "n_folds" : 20, #which of the 4 folds used to run the experiments
           'n_repeats': 100,
           "batch_size": 100,
           "train_iters": 3000,

           "feature_dir": 'precomputed_features_cons_nozero',
                               #  1   2   3   4   5
           'difficulty_margins':[0.5,0.4,0.3,0.2,0.1],

           'max_difficulty_training':5,

           "config_folder": "all_features",
           'features_to_use': ALL_FEATURES,
           'features_to_remove':[],
            }


@ex.named_config
def all_features_maxdiffic4():
    cfg = {'max_difficulty_training':4,
           "config_folder": "all_features_maxdiffic4",
           }

@ex.named_config
def all_features_maxdiffic2():
    cfg = {'max_difficulty_training':2,
           "config_folder": "all_features_maxdiffic2",
           }

@ex.named_config
def no_benchmark():
    cfg = {
           "config_folder": "no_benchmark",
           'features_to_remove': [
                             "framewise_0.01",
                             "notewise_On_50",
                             "notewise_OnOff_50_0.2",
                             ],

           }

@ex.named_config
def only_benchmark():
    cfg = {
           "config_folder": "only_benchmark",
            'features_to_use': [
                             "framewise_0.01",
                             "notewise_On_50",
                             "notewise_OnOff_50_0.2",
                             ],
           }

@ex.named_config
def no_high_low():
    cfg = {
           "config_folder": "no_high_low",
            'features_to_remove': [
                             "high_f",
                             "low_f",
                             "high_n",
                             "low_n",
                             ],

           }

@ex.named_config
def no_loud():
    cfg = {
           "config_folder": "no_loud",
            'features_to_remove': [
                                    "loud_fn",
                                    "loud_ratio_fn",
                                    ],
           }

@ex.named_config
def no_out_key():
    cfg = {
           "config_folder": "no_out_key",
            'features_to_remove': [
                             "out_key",
                             "out_key_bin",
                             ],

           }

@ex.named_config
def no_repeat():
    cfg = {
           "config_folder": "no_repeat",
            'features_to_remove': [
                             "repeat",
                             "merge",
                             ],
           }

@ex.named_config
def no_specific():
    cfg = {
           "config_folder": "no_specific",
            'features_to_remove': [
                             "semitone_f",
                             "octave_f",
                             "third_harmonic_f",
                             "semitone_n",
                             "octave_n",
                             "third_harmonic_n",
                             ],

           }

@ex.named_config
def no_poly():
    cfg = {
           "config_folder": "no_poly",
            'features_to_remove': [
                             "poly_diff",
                             ],
           }

@ex.named_config
def no_rhythm():
    cfg = {
           "config_folder": "no_rhythm",
            'features_to_remove': [
                             "rhythm_hist",
                             "rhythm_disp_std",
                             "rhythm_disp_drift",
                             ],

           }

@ex.named_config
def no_consonance():
    cfg = {
           "config_folder": "no_consonance",
            'features_to_remove': [
                              "valid_cons",
                             ],
           }


@ex.named_config
def no_framewise():
    cfg = {
           "config_folder": "no_framewise",
            'features_to_use': [
                            "notewise_On_50",
                            "notewise_OnOff_50_0.2",

                            "high_n",
                            "low_n",

                            "loud_fn",
                            "loud_ratio_fn",

                            "out_key",
                            "out_key_bin",

                            "repeat",
                            "merge",


                            "semitone_n",
                            "octave_n",
                            "third_harmonic_n",


                            "rhythm_hist",
                            "rhythm_disp_std",
                            "rhythm_disp_drift",

                            ],
           }

@ex.named_config
def no_specific_consonance():
    cfg = {
           "config_folder": "no_specific_consonance",
            'features_to_remove': [
                             "semitone_f",
                             "octave_f",
                             "third_harmonic_f",
                             "semitone_n",
                             "octave_n",
                             "third_harmonic_n",

                             "valid_cons",
                             ],
           }



@ex.automain
def train_classif(cfg):

    #### Prepare data:
    feature_dir = cfg['feature_dir']

    filecp = codecs.open('db_csv/answers_data.csv', encoding = 'utf-8')
    answers = np.genfromtxt(filecp,dtype=object,delimiter=";")
    answers = answers[1:,:]

    filecp = codecs.open('db_csv/user_data.csv', encoding = 'utf-8')
    users = np.genfromtxt(filecp,dtype=object,delimiter=";")
    users = users[1:,:]

    results_dict = {}

    systems = ['cheng','google',"kelz","lisu"]
    pairs = []
    for i in range(len(systems)):
        for j in range(i+1,len(systems)):
            pairs += [[systems[i],systems[j]]]

    r = np.array(list(range(len(pairs))))

    for example in np.unique(answers[:,1]):
        example_dir = os.path.join(feature_dir,example)
        results_dict[example] = {}
        for system in systems:
            results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
            results_dict[example][system]=results



    ########################################################
    ###########      USING FEATURES
    ########################################################


    save_destination = os.path.join(cfg['base_folder'],cfg['config_folder'])
    if not os.path.exists(save_destination):
        os.makedirs(save_destination)


    features_to_use = cfg['features_to_use']
    features_to_use = [feat for feat in features_to_use if not feat in cfg['features_to_remove']]
    labels = utils.get_feature_labels(features_to_use)
    N_FEATURES = len(labels)

    print 'Total features:', N_FEATURES, '(removed:',len(utils.get_feature_labels(ALL_FEATURES))-N_FEATURES,')'


    #### Graph definition

    features1_ph = tf.placeholder(tf.float32,[None,N_FEATURES])
    features2_ph = tf.placeholder(tf.float32,[None,N_FEATURES])
    y_ph = tf.placeholder(tf.float32,[None,1])
    z_ph = tf.placeholder(tf.float32,[None,1])
    alpha_ph = tf.placeholder(tf.float32,[None,1])

    # weights = tf.Variable(tf.random_normal([N_FEATURES,1], stddev=0.35),name='weights')
    weights = tf.Variable(tf.zeros([N_FEATURES,1]),name='weights')
    bias = tf.Variable(tf.zeros([]),name='bias')

    model_output1 = utils.linear_regression_model(features1_ph,weights,bias,None)
    model_output2 = utils.linear_regression_model(features2_ph,weights,bias,None)



    loss = utils.contrastive_loss_absolute(model_output1,model_output2,y_ph,z_ph,alpha_ph)
    # loss = contrastive_loss(model_output1,model_output2,y_ph,alpha_ph)
    optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


    #### GATHER DATA
    features1=[]
    features2=[]
    ratings=[]
    notewise1 = []
    notewise2 = []
    goldmsi = []


    #### USE EACH INDIVIDUAL ANSWER AS TRAINING SAMPLE

    for row in answers:

        example = row[1]
        system1 = row[2]
        system2 = row[3]

        # print example, system1, system2

        example_dir = os.path.join(feature_dir,example)
        # print len(import_features(example_dir,system1)),import_features(example_dir,system1)
        results1 = pickle.load(open(os.path.join(example_dir,system1+'.pkl'), "rb"))
        results2 = pickle.load(open(os.path.join(example_dir,system2+'.pkl'), "rb"))

        notewise1 += [results1["notewise_On_50"][-1]]
        notewise2 += [results2["notewise_On_50"][-1]]
        goldmsi += [float(users[users[:,0]==row[4],5])]

        features1 += [utils.import_features(results1,features_to_use)]
        features2 += [utils.import_features(results2,features_to_use)]
        ratings += [row[5]]

    features1 = np.array(features1,dtype=float)
    features2 = np.array(features2,dtype=float)
    ratings = np.array(ratings,dtype=int)
    notewise1 = np.array(notewise1,dtype=float)
    notewise2 = np.array(notewise2,dtype=float)
    goldmsi = np.array(goldmsi,dtype=float)
    q_ids = answers[:,0]
    difficulties = answers[:,7].astype(float)

    y=np.zeros_like(ratings)
    z=ratings
    alpha = np.zeros_like(difficulties)
    alpha[difficulties==1] = cfg['difficulty_margins'][0]
    alpha[difficulties==2] = cfg['difficulty_margins'][1]
    alpha[difficulties==3] = cfg['difficulty_margins'][2]
    alpha[difficulties==4] = cfg['difficulty_margins'][3]
    alpha[difficulties==5] = cfg['difficulty_margins'][4]


    y = y[:,None]
    z = z[:,None]
    alpha = alpha[:,None]

    #### SPLIT DATA

    all_results = {}

    for fold in range(cfg['n_folds']):


        ###### USE EACH INDIVIDUAL ANSWER
        all_examples, indices = np.unique(answers[:,1],return_index=True)
        sort_idx = np.argsort(indices)
        all_examples = all_examples[sort_idx]
        example_indices = indices[sort_idx]


        n_examples = len(all_examples)
        n_test = int(1.0/cfg['n_folds']*n_examples)
        n_valid = int(1.0/cfg['n_folds']*n_examples)
        # n_test = 0
        # n_valid = int(0.1*n_examples)
        ex_idx_test_start = fold*n_test
        ex_idx_test_end= (fold+1)*n_test
        if fold == cfg['n_folds']-1:
            ex_idx_valid_start = 0*n_valid
            ex_idx_valid_end = 1*n_valid
        else:
            ex_idx_valid_start = ex_idx_test_end
            ex_idx_valid_end = ex_idx_test_end+n_valid

        # print ex_idx_valid_end-ex_idx_valid_start,n_valid, ex_idx_test_end-ex_idx_test_start, n_test

        idx_test_start = example_indices[ex_idx_test_start]
        idx_test_end = example_indices[ex_idx_test_end]
        idx_valid_start = example_indices[ex_idx_valid_start]
        idx_valid_end = example_indices[ex_idx_valid_end]

        idx_test = np.zeros([len(answers)],dtype=bool)
        idx_valid = np.zeros([len(answers)],dtype=bool)
        idx_train = np.zeros([len(answers)],dtype=bool)

        if fold == cfg['n_folds']-1:
            idx_test[idx_test_start:] = True
            idx_valid[idx_valid_start:idx_valid_end] = True
            idx_train[idx_valid_end:idx_test_start] = True
        else:
            idx_test[idx_test_start:idx_test_end] = True
            idx_valid[idx_valid_start:idx_valid_end] = True
            idx_train[:idx_test_start] = True
            idx_train[idx_valid_end:] = True

        # print [idx_test_start,idx_test_end], [idx_valid_start,idx_valid_end]
        # print fold, np.sum(idx_test),np.sum(idx_valid),np.sum(idx_train)
        # print np.all(idx_test.astype(int)+idx_valid.astype(int)+idx_train.astype(int)==1),np.any(idx_test.astype(int)+idx_valid.astype(int)+idx_train.astype(int)==0), np.any(idx_test.astype(int)+idx_valid.astype(int)+idx_train.astype(int)==2)
        # continue

        features1_train = features1[idx_train]
        features2_train = features2[idx_train]
        y_train = y[idx_train]
        z_train = z[idx_train]
        alpha_train = alpha[idx_train]
        notewise1_train = notewise1[idx_train]
        notewise2_train = notewise2[idx_train]
        goldmsi_train = goldmsi[idx_train]
        difficulties_train = difficulties[idx_train]

        features1_valid = features1[idx_valid]
        features2_valid = features2[idx_valid]
        y_valid = y[idx_valid]
        z_valid = z[idx_valid]
        alpha_valid = alpha[idx_valid]
        notewise1_valid = notewise1[idx_valid]
        notewise2_valid = notewise2[idx_valid]

        features1_test = features1[idx_test]
        features2_test = features2[idx_test]
        y_test = y[idx_test]
        z_test = z[idx_test]
        alpha_test = alpha[idx_test]
        notewise1_test = notewise1[idx_test]
        notewise2_test = notewise2[idx_test]
        q_ids_test = q_ids[idx_test]
        difficulties_test = difficulties[idx_test]


        ###### Apply PCA
        # pca = PCA()
        # pca.fit(np.concatenate([features1_train,features2_train],axis=0))
        # total_variance = np.cumsum(pca.explained_variance_ratio_)
        # keep_dims = np.argmax(total_variance>0.99)
        #
        # print "keep_dims", keep_dims
        # # keep_dims = 16
        # pca = PCA(n_components=keep_dims)
        # pca.fit(np.concatenate([features1_train,features2_train],axis=0))
        # pca_matrix = pca.components_

        #### No PCA:
        # pca_matrix= None
        # keep_dims = N_FEATURES

        ###############################
        ### Remove from training set

        ### Any unsure response:
        to_keep = difficulties_train<=cfg['max_difficulty_training']
        ### Any non-musician response:
        # to_keep = goldmsi_train>=np.median(goldmsi_train)
        ### Any answer that agrees with F-measure (keep only those who disagree):
        # results_F1 = (notewise1_train < notewise2_train).astype(int)
        # to_keep = np.not_equal(z_train[:,0],results_F1)
        ### Any answer that CONFIDENTLY agrees with F-measure (keep only those who disagree):
        # results_F1 = (notewise1_train < notewise2_train).astype(int)
        # to_keep = np.logical_and(np.not_equal(z_train[:,0],results_F1),difficulties_train<3)
        # print np.sum(to_keep)
        #
        features1_train = features1_train[to_keep]
        features2_train = features2_train[to_keep]
        y_train = y_train[to_keep]
        z_train = z_train[to_keep]
        alpha_train = alpha_train[to_keep]
        notewise1_train = notewise1_train[to_keep]
        notewise2_train = notewise2_train[to_keep]


        ###############################
        #### Normalise features
        all_features_train = np.concatenate([features1_train,features2_train],axis=0)
        mean = np.mean(all_features_train,axis=0)
        std = np.std(all_features_train,axis=0)
        features1_train = (features1_train-mean)/std
        features2_train = (features2_train-mean)/std
        features1_valid = (features1_valid-mean)/std
        features2_valid = (features2_valid-mean)/std
        features1_test = (features1_test-mean)/std
        features2_test = (features2_test-mean)/std


        #### AGGREGATE CONFIDENT TEST ANSWERS
        #### Only keep answers for which there is a clear majority, regardless of the number of confident answers
        features1_test_agg = []
        features2_test_agg = []
        ratings_test_agg = []
        result_f1_test_agg = []
        notewise1_test_agg = []
        notewise2_test_agg = []

        for q_id in np.unique(q_ids_test):
            idx_id = np.logical_and(q_ids_test == q_id,difficulties_test<3)
            # Skip questions without confident answers
            if np.any(idx_id):
                vote = np.mean(z_test[idx_id])
                # Skip draw cases
                if vote != 0.5:
                    features1_test_agg += [features1_test[idx_id][0,:]]
                    features2_test_agg += [features2_test[idx_id][0,:]]
                    ratings_test_agg += [int(vote > 0.5)]
                    notewise1_test_agg += [notewise1_test[idx_id][0]]
                    notewise2_test_agg += [notewise2_test[idx_id][0]]
                    # result_f1_test_agg += [(notewise1_test[idx_id] < notewise2_test[idx_id]).astype(int)[0]]
                # print answers[idx_test][idx_id]
                # print vote, int(vote > 0.5)
                # print notewise1_test[idx_id][0],notewise2_test[idx_id][0],(notewise1_test[idx_id] < notewise2_test[idx_id]).astype(int)[0]

        features1_test_agg = np.array(features1_test_agg)
        features2_test_agg = np.array(features2_test_agg)
        ratings_test_agg = np.array(ratings_test_agg)
        notewise1_test_agg = np.array(notewise1_test_agg)
        notewise2_test_agg = np.array(notewise2_test_agg)
        result_f1_test_agg = (notewise1_test_agg<notewise2_test_agg).astype(int)


        # print features1_test_agg.shape,features2_test_agg.shape,ratings_test_agg.shape,result_f1_test_agg.shape
        # print np.mean(ratings_test_agg == result_f1_test_agg)

        #### Run training
        repeat_agreement = []
        repeat_agreement_agg = []
        repeat_agreement_conf = []
        repeat_best_weights = []
        repeat_best_bias = []

        feed_dict_valid = {
            features1_ph:features1_valid,
            features2_ph:features2_valid,
            y_ph:y_valid,
            z_ph:z_valid,
            alpha_ph: alpha_valid,
            }

        N_REPEATS = cfg['n_repeats']

        results_F1 = (notewise1_test < notewise2_test).astype(int)
        agreement_F1 = np.mean((z_test[:,0]==results_F1).astype(int))
        agreement_F1_conf = np.mean((z_test[difficulties_test<3,0]==results_F1[difficulties_test<3]).astype(int))
        agreement_F1_agg = np.mean(ratings_test_agg == result_f1_test_agg)



        for i in range(N_REPEATS):
            print save_destination, "fold",fold,"repeat",i, np.mean(repeat_agreement)

            valid_costs = []
            train_costs = []

            best_valid = None
            best_weights = None
            best_bias = None


            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            for i in range(cfg['train_iters']):

                ### Batching
                features1_batch ,features2_batch,y_batch ,z_batch ,alpha_batch = utils.sample(cfg['batch_size'],features1_train,features2_train,y_train,z_train,alpha_train)
                feed_dict_train = {
                    features1_ph:features1_batch,
                    features2_ph:features2_batch,
                    y_ph:y_batch,
                    z_ph:z_batch,
                    alpha_ph: alpha_batch,
                    }

                ### Just take the whole set
                # feed_dict_train = {
                #     features1_ph:features1_train,
                #     features2_ph:features2_train,
                #     y_ph:y_train,
                #     z_ph:z_train,
                #     alpha_ph: alpha_train,
                #     }

                sess.run(optimize, feed_dict=feed_dict_train)
                valid_cost = sess.run(loss, feed_dict=feed_dict_valid)
                train_cost = sess.run(loss, feed_dict={
                                                        features1_ph:features1_train,
                                                        features2_ph:features2_train,
                                                        y_ph:y_train,
                                                        z_ph:z_train,
                                                        alpha_ph: alpha_train,
                                                        })

                #Compute agreement, removing draws:
                metrics1,metrics2 = sess.run([model_output1,model_output2],feed_dict_valid)
                result_metrics = (metrics1<metrics2).astype(int)
                valid_costs += [valid_cost]
                train_costs += [train_cost]

                # plt.clf()
                # plt.scatter(features1_valid[:,5],metrics1,color='tab:blue')
                # plt.scatter(features2_valid[:,5],metrics2,color='tab:blue')
                # plt.ylim([0,1])
                # plt.xlim([0,1])
                #
                # plt.pause(0.00000001)
                #
                # print i, valid_cost, np.mean((z_valid==result_metrics).astype(int))

                if best_valid is None or valid_cost<best_valid:
                    best_weights,best_bias = sess.run([weights,bias])

            ###### RESULTS
            #
            # print 'Best parameters:'
            # for (label,value) in zip(labels,best_parameters):
            #     print label, value
            # plt.plot(valid_costs)
            # plt.plot(train_costs)
            # plt.show()

            feed_dict_test = {
                features1_ph:features1_test,
                features2_ph:features2_test,
                weights: best_weights,
                bias: best_bias
                }

            feed_dict_test_agg = {
                features1_ph:features1_test_agg,
                features2_ph:features2_test_agg,
                weights: best_weights,
                bias: best_bias
            }

            idx_confident = difficulties_test<3
            feed_dict_test_conf = {
                features1_ph:features1_test[idx_confident],
                features2_ph:features2_test[idx_confident],
                weights: best_weights,
                bias: best_bias
            }

            metrics1,metrics2 = sess.run([model_output1,model_output2],feed_dict_test)
            result_metrics = (metrics1<metrics2).astype(int)
            agreement_metric = np.mean((z_test==result_metrics).astype(int))
            repeat_agreement += [agreement_metric]

            metrics1_conf,metrics2_conf = sess.run([model_output1,model_output2],feed_dict_test_conf)
            result_metrics_conf = (metrics1_conf<metrics2_conf).astype(int)
            agreement_metric_conf = np.mean((z_test[idx_confident]==result_metrics_conf).astype(int))
            repeat_agreement_conf += [agreement_metric_conf]

            metrics1_agg,metrics2_agg = sess.run([model_output1,model_output2],feed_dict_test_agg)
            # for m1,m2, r,n1,n2, f1 in zip(metrics1_agg,metrics2_agg,ratings_test_agg,notewise1_test_agg,notewise2_test_agg,result_f1_test_agg):
            #     print "metrics",m1,m2,int(m1<m2),"notewise",n1,n2, f1, "rating", r, "OK" if int(m1<m2)==r else "BAD"
            result_metrics_agg = (metrics1_agg<metrics2_agg).astype(int)
            agreement_metric_agg = np.mean((ratings_test_agg==result_metrics_agg[:,0]))
            repeat_agreement_agg += [agreement_metric_agg]

            repeat_best_weights += [best_weights]
            repeat_best_bias += [best_bias]




            print "average agreement new metric:", np.round(agreement_metric,3), "F-measure:", np.round(agreement_F1,3)
            print "average agreement new metric conf.:", np.round(agreement_metric_conf,3), "F-measure conf.:", np.round(agreement_F1_conf,3)
            print "average agreement new metric agg.:", np.round(agreement_metric_agg,3), "F-measure agg.:", np.round(agreement_F1_agg,3)
            # print repeat_agreement



        results_dict = {'repeat_agreement':repeat_agreement,
                        'repeat_agreement_agg':repeat_agreement_agg,
                        'repeat_agreement_conf':repeat_agreement_conf,
                        'agreement_F1': agreement_F1,
                        'agreement_F1_agg': agreement_F1_agg,
                        'agreement_F1_conf': agreement_F1_conf,
                        'repeat_best_weights':repeat_best_weights,
                        'repeat_best_bias':repeat_best_bias}

        # print np.std(repeat_agreement)
        # print np.mean(repeat_agreement)
        save_path = os.path.join(save_destination,'fold'+str(fold)+'.pkl')
        pickle.dump(results_dict, open(save_path, 'wb'))

        all_results['fold'+str(fold)]=results_dict

    save_path = os.path.join(save_destination,'all_folds.pkl')
    pickle.dump(all_results, open(save_path, 'wb'))


    #
    # plt.scatter(notewise1_test,metrics1,color='tab:blue')
    # plt.scatter(notewise2_test,metrics2,color='tab:blue')
    # plt.ylim([0,1])
    # plt.xlim([0,1])
    #
    # plt.show()
