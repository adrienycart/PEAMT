import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.stats import ttest_1samp, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


plt.rcParams.update({'font.size': 13.5,'font.family' : 'serif'})
plt.rcParams["figure.figsize"] = [7,6.5]
mpl.rc('text', usetex=True)

# base_folder = "results_metric/incorrect_high_low/20_folds"
base_folder = "results_metric/20_folds_no_consonance"

if '10_folds' in base_folder:
    n_folds=10
elif '20_folds' in base_folder:
    n_folds=20
# All 81.3 & 87.4
folders_to_compare = [
                        ["all_features","All"],
                        # ["all_features_maxdiffic2","All (Difficuly<3)"],
                        # ["all_features_maxdiffic4","All (Difficuly<5)"],
                        ["no_benchmark",'NoBench'],
                        ["only_benchmark",'NoFeatures'],
                        ["no_high_low",'NoHighLow'],
                        ["no_loud",'NoLoud'],
                        ["no_out_key",'NoOutKey'],
                        ["no_specific",'NoSpecific'],
                        ["no_poly",'NoPoly'],
                        ["no_repeat",'NoRepeat'],
                        ["no_rhythm",'NoRhythm'],
                        # ["no_consonance",'NoConsonance'],
                        # ["no_consonance_diff",'All'],
                        # ["no_consonance_diff",'NoConsDiff'],
                        # ["no_specific_consonance",'NoSpecCons'],
                        # ["no_specific_consonance_6",'NoSpecCons1'],
                        # ["no_specific_consonance_out_key",'NoSpecConsOut'],
                        ["no_specific_out_key",'NoSpecOut'],
                        ["no_framewise",'NoFramewise'],
                        # ["no_consonance_invalid",'NoInvalidCons'],
                        # ["no_useless",'NoNegative'],
                        # ["no_non_significant",'NoNonSig'],

                        ]

folders_to_compare = [[os.path.join(base_folder,folder),name] for folder,name in folders_to_compare]


reference_matrix = []
reference_matrix_conf = []
reference_matrix_agg = []
file_handle = open(os.path.join(base_folder,"all_features",'all_folds.pkl'), "rb")
reference_results = pickle.load(file_handle)

results_F1=[]
results_F1_conf=[]
results_F1_agg =[]

for i in range(n_folds):
    reference_matrix += [reference_results['fold'+str(i)]['repeat_agreement']]
    reference_matrix_conf += [reference_results['fold'+str(i)]['repeat_agreement_conf']]
    reference_matrix_agg += [reference_results['fold'+str(i)]['repeat_agreement_agg']]
    results_F1 += [reference_results['fold'+str(i)]['agreement_F1']]
    results_F1_conf += [reference_results['fold'+str(i)]['agreement_F1_conf']]
    results_F1_agg += [reference_results['fold'+str(i)]['agreement_F1_agg']]



### USING A THRESHOLD OF 75ms
# results_F1_conf = [0.887323943662 ,
# 0.864864864865 ,
# 0.869047619048 ,
# 0.832214765101 ,
# 0.925 ,
# 0.848920863309 ,
# 0.865671641791 ,
# 0.912621359223 ,
# 0.914473684211 ,
# 0.867549668874 ,
# 0.872727272727 ,
# 0.892045454545 ,
# 0.932835820896 ,
# 0.909836065574 ,
# 0.905882352941 ,
# 0.860294117647 ,
# 0.865671641791 ,
# 0.86524822695 ,
# 0.877697841727 ,
# 0.870588235294 ,]


p_values_all = []

for folder,name in folders_to_compare:
    file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
    all_results = pickle.load(file_handle)
    t_values_f = []
    t_values_all_features = []
    t_values_f_conf = []
    t_values_all_features_conf = []
    t_values_f_agg = []
    t_values_all_features_agg = []
    for i in range(n_folds):
        agreement = all_results['fold'+str(i)]['repeat_agreement']
        agreement_conf = all_results['fold'+str(i)]['repeat_agreement_conf']
        agreement_agg = all_results['fold'+str(i)]['repeat_agreement_agg']

        t_values_f += [ttest_1samp(agreement,results_F1[i])[0]]
        t_values_f_conf += [ttest_1samp(agreement,results_F1_conf[i])[0]]
        t_values_f_agg += [ttest_1samp(agreement,results_F1_agg[i])[0]]

        t_values_all_features += [ttest_ind(agreement,reference_matrix[i],equal_var=False)[0]]
        t_values_all_features_conf += [ttest_ind(agreement_conf,reference_matrix_conf[i],equal_var=False)[0]]
        t_values_all_features_agg += [ttest_ind(agreement_agg,reference_matrix_agg[i],equal_var=False)[0]]

    print '------------------'
    print name

    p_value_f = ttest_1samp(t_values_f,0)[1]
    p_value_f_conf = ttest_1samp(t_values_f_conf,0)[1]
    p_value_f_agg = ttest_1samp(t_values_f_agg,0)[1]

    p_value_all_features = ttest_1samp(t_values_all_features,0)[1]
    p_value_all_features_conf = ttest_1samp(t_values_all_features_conf,0)[1]
    p_value_all_features_agg = ttest_1samp(t_values_all_features_agg,0)[1]

    p_values_all += [p_value_all_features_conf]

    print t_values_all_features_conf
    print "p_value_f", p_value_f
    print "p_value_f_conf", p_value_f_conf
    print "p_value_f_agg", p_value_f_agg
    print
    print "p_value_all_features", p_value_all_features
    print "p_value_all_features_conf", p_value_all_features_conf
    print "p_value_all_features_agg", p_value_all_features_agg




##### PLOT RESULTS
print 'baseline', np.round(np.mean(results_F1),3), np.round(np.mean(results_F1_conf),3)
results_avg = []
results_avg_conf = []
for i,(folder,name) in enumerate(folders_to_compare):
    file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
    all_results = pickle.load(file_handle)
    results_concat = []
    results_concat_conf = []
    for fold in range(n_folds):
        results_concat_conf += all_results['fold'+str(fold)]['repeat_agreement_conf']
        results_concat += all_results['fold'+str(fold)]['repeat_agreement']

    results_avg += [np.mean(results_concat)]
    results_avg_conf += [np.mean(results_concat_conf)]


sort_idx = np.argsort(results_avg_conf,)


fig = plt.figure()
gs = mpl.gridspec.GridSpec(1, 2,width_ratios=[10,1])
ax = plt.subplot(gs[0])
ax_c = plt.subplot(gs[1])

color_bins = 5

for i,idx in enumerate(sort_idx):
    if folders_to_compare[idx][1] == 'All':
        color = 'black'
    else:
        # color = 'tab:blue'
        p_val_round = np.ceil(p_values_all[idx]*color_bins)/float(color_bins)
        color = np.array([1.0,1,1])-(1-p_val_round)*np.array([0,1,1])
    ax.barh(i,results_avg_conf[idx],color=color,edgecolor='black')
    significance = ''
    if p_values_all[idx] < 0.1:
        significance = '*'
    if p_values_all[idx] < 0.05:
        # significance = r'\textbf{**}'
        significance = '**'
    if p_values_all[idx] < 0.01:
        significance = '***'


    ax.text(results_avg_conf[idx]+0.001,i,significance,verticalalignment='center',color='black',weight='bold')

    print os.path.basename(folders_to_compare[i][1]), np.round(results_avg[i]*100,1),"&", np.round(results_avg_conf[i]*100,1)
    # if 'all_features' == os.path.basename(folder):
    #     plt.plot([mean,mean],[0,len(folders_to_compare)],linestyle='--',color='grey')

ax.set_yticks(range(len(folders_to_compare)))
ax.set_yticklabels([folders_to_compare[sort_idx[i]][1] for i in range(len(folders_to_compare))])
ax.plot([np.mean(results_F1_conf),np.mean(results_F1_conf)],[-0.5,len(folders_to_compare)-0.5],linestyle='--',color='black')
ax.set_xlim([0.86,0.897])
ax.set_xlabel(r'$A_{conf}$')

ticks = np.arange(0,1.1,1/float(color_bins))
colors = np.array([1.0,1,1])-(1-ticks[1:,None])*np.array([0,1,1])

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(ticks, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax_c, cmap=cmap,
                                norm=norm,
                                boundaries=ticks,
                                # extend='both',
                                ticks=ticks,
                                spacing='proportional',
                                # orientation='horizontal'
                                )
# cbar.set_ticks([0,0.5,1])
# cbar.set_ticklabels([0,0.5,1])
plt.tight_layout()
plt.show()
