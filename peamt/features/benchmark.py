import numpy as np
from .utils import precision, recall, Fmeasure

##################################################
### FRAMEWISE METRICS
##################################################

def TP(data,target):
    return np.sum(np.logical_and(data == 1, target == 1))

def FP(data,target):
    return np.sum(np.logical_and(data == 1, target == 0))

def FN(data,target):
    return np.sum(np.logical_and(data == 0, target == 1))




def framewise(output,target):
    tp,fp,fn = TP(output, target),FP(output, target),FN(output, target),

    P_f,R_f,F_f = precision(tp,fp), recall(tp,fn), Fmeasure(tp,fp,fn)
    return P_f,R_f,F_f


##################################################
### NOTEWISE METRICS
##################################################


def notewise(match,est_pitches,ref_pitches):
    prec = float(len(match))/(len(est_pitches)+np.finfo(float).eps)
    rec = float(len(match))/(len(ref_pitches)+np.finfo(float).eps)
    F = 2*prec*rec/(prec+rec+np.finfo(float).eps)
    return prec, rec, F
