# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:07:58 2016

@author: farznaj
"""
#def svm_plots_setVars_n(mousename, figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/', trialHistAnalysis=0, iTiFlg=2):
#%% 
# Define days that you want to analyze
if mousename=='fni16':
#    days = ['150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'];
#    days = ['150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2']; # '150914_1-2' : don't analyze!
    # all days
    # 0817,0818: don't have excInh files
    # 0817,0818,0819: don't have eachFrame, stAl. Also their FOV more superficial that the rest of the days
    if 'ch_st_goAl' in locals(): # eachFrame analysis
        if ch_st_goAl[0]==1: # chAl   
            days = ['150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!
        elif ch_st_goAl[1]==1: # stAl           
            days = ['150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!

    if 'loadInhAllexcEqexc' in locals(): # excInh_eachFrame analysis
        days = ['150819_1', '150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!
    
    if trialHistAnalysis == 1 and iTiFlg == 1:
        days.remove('151001_1') # this day has only 8 long ITI trials, 1 of which is hr ... so not enough trials for the HR class to train the classifier!


elif mousename=='fni17': 
    # '150814_2, '150820_1', '150821_1', '150825_1': dont have enough trials. ('150821_1' and '150825_1' only have the eachFrame, stAl svm file)    
    # '150814_1', '150817_1', '150818_1': cant be run for eachFrame, stAl
    days = ['151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2']     # Done on 6Jan2017: reverse the order, so it is from early days to last days
    days = ['150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1'];
    # all days
    days = ['150814_1', '150817_1', '150818_1', '150819_1-2', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'];
#    days = ['150819_1-2', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'];


elif mousename=='fni18': # eachFrame, stimAl files couldnt be run for '151211_1', '151214_1-2', '151215_1-2', '151216_1'
    if allDays:
        days = ['151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'] # alldays
    elif noZmotionDays:        
        days = ['151209_1', '151210_1', '151211_1', '151214_1-2'] # following days removed bc of z motion: '151215_1-2', '151216_1', '151217_1-2'
    elif noZmotionDays_strict:        
        days = ['151209_1', '151210_1', '151211_1'] # even '151214_1-2' is suspicious about z motion, so remove it!


elif mousename=='fni19':
    if allDays:
        # days = ['150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1']
        # days = ['150901_1', '150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1'];
        if 'ch_st_goAl' in locals() and ch_st_goAl[1]==1:  # eachFrame, stimAL: '150904_1' doesnt have enough trials. taken out from below.       
            days = ['150901_1', '150903_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'];
        else:
            days = ['150901_1', '150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'];
                        
    elif noExtraStimDays:
        days = ['151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1']
        
        
        
        
#savefigs = False

fmt = ['pdf', 'svg', 'eps'] #'png', 'pdf': preserve transparency # Format of figures for saving

#figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/' # Directory for saving figures.
import platform
if platform.system()=='Linux':
    figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/'
elif platform.system()=='Darwin':
    figsDir = '/Users/Farzaneh/Dropbox/ChurchlandLab/Farzaneh_Gamal/'
    svmfold = ''

#neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
import sys
eps = sys.float_info.epsilon #10**-10 # tiny number below which weight is considered 0
palpha = .05 # p <= palpha is significant
#thR = 2 # Exclude days with only <=thR rounds with non-0 weights
#numRounds = 10; # number of times svm analysis was ran for the same dataset but sampling different sets of neurons.    

##%%
import os
#import glob
import numpy as np   
import numpy.random as rng
import scipy as sci
import scipy.io as scio
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#import sys
#sys.path.append('/home/farznaj/Documents/trial_history/imaging/')  
#from setImagingAnalysisNamesP import *

plt.rc('font', family='helvetica')    


frameLength = 1000/30.9; # sec.  # np.diff(time_aligned_stim)[0];
'''
if neuronType==0:
    ntName = 'excit'
elif neuronType==1:
    ntName = 'inhibit'
elif neuronType==2:
    ntName = 'all'     
'''
if trialHistAnalysis==1:    
    if iTiFlg==0:
        itiName = 'short'
    elif iTiFlg==1:
        itiName = 'long'
    elif iTiFlg==2:
        itiName = 'all'
else: # wont be used, only not to get error when running setSVMname
        itiName = 'all'

'''
if trialHistAnalysis==1: # more parameters are specified in popClassifier_trialHistory.m
#        iTiFlg = 1; # 0: short ITI, 1: long ITI, 2: all ITIs.
    epEnd_rel2stimon_fr = 0 # 3; # -2 # epEnd = eventI + epEnd_rel2stimon_fr
else:
    # not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
#         ep_ms = [425, 725] # optional, it will be set according to min choice time if not provided.# training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; #[700, 900]; # [500, 700]; 
    # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.
    outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
    strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
#    thStimStrength = 3; # 2; # threshold of stim strength for defining hard, medium and easy trials.
#    th_stim_dur = 800; # min stim duration to include a trial in timeStimOnset

trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
'''
# Set fig names
if trialHistAnalysis:
#     ep_ms = np.round((ep-eventI)*frameLength)
#    th_stim_dur = []
    suffn = 'prev_'
#    suffnei = 'prev_%sITIs_excInh_' %(itiName)
else:
    suffn = 'curr_'
#    suffnei = 'curr_%s_epVar_' %('excInh')   
#print '\n', suffn[:-1], ' - ', suffnei[:-1]



# Make folder named SVM to save figures inside it
svmdir = os.path.join(figsDir, 'SVM')
if not os.path.exists(svmdir):
    os.makedirs(svmdir)    


daysOrig = days
numDays = len(days)
print 'Analyzing mouse',mousename,'-', len(days), 'days'
        
    
#    return days, suffn 

#mousename = 'fni16' #'fni17'
#trialHistAnalysis = 0;
#iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.    
'''
if mousename=='fni17':  
    ep_ms = [809, 1109]
elif mousename=='fni16':
    ep_ms = [] #[809, 1109] # set to [] if each day had a different ep # will be used only for trialHistAnalysis=0
elif mousename=='fni18':
    ep_ms = [] #[809, 1109] # set to [] if each day had a different ep # will be used only for trialHistAnalysis=0
'''        
