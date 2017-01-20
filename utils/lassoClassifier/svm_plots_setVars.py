# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:07:58 2016

@author: farznaj
"""


#%% 
mousename = 'fni17' #'fni17'

trialHistAnalysis = 1;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.    
if mousename=='fni17':  
    ep_ms = [809, 1109]
elif mousename=='fni16':
    ep_ms = [] #[809, 1109] # set to [] if each day had a different ep # will be used only for trialHistAnalysis=0
        
# Define days that you want to analyze
if mousename=='fni17':  
    # Done on 6Jan2017: reverse the order, so it is from early days to last days
    days = ['151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2']
#    days = ['151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'];
elif mousename=='fni16':
    days = ['150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'];
#    days = ['150930_1-2', '151001_1', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'];
    
    if trialHistAnalysis == 1 and iTiFlg == 1:
        days.remove('151001_1') # this day has only 8 long ITI trials, 1 of which is hr ... so not enough trials for the HR class to train the classifier!
        
    
savefigs = False

fmt = ['pdf', 'svg', 'eps'] #'png', 'pdf': preserve transparency # Format of figures for saving
figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/' # Directory for saving figures.
neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
eps = 10**-10 # tiny number below which weight is considered 0
palpha = .05 # p <= palpha is significant
thR = 2 # Exclude days with only <=thR rounds with non-0 weights
numRounds = 10; # number of times svm analysis was ran for the same dataset but sampling different sets of neurons.    

##%%
import os
#import glob
import numpy as np   
import scipy as sci
import scipy.io as scio
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.append('/home/farznaj/Documents/trial_history/imaging/')  
from setImagingAnalysisNamesP import *

plt.rc('font', family='helvetica')    


frameLength = 1000/30.9; # sec.  # np.diff(time_aligned_stim)[0];

if neuronType==0:
    ntName = 'excit'
elif neuronType==1:
    ntName = 'inhibit'
elif neuronType==2:
    ntName = 'all'     

if trialHistAnalysis==1:    
    if iTiFlg==0:
        itiName = 'short'
    elif iTiFlg==1:
        itiName = 'long'
    elif iTiFlg==2:
        itiName = 'all'
else: # wont be used, only not to get error when running setSVMname
        itiName = 'all'


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

# Set fig names
if trialHistAnalysis:
#     ep_ms = np.round((ep-eventI)*frameLength)
#    th_stim_dur = []
    suffn = 'prev_%sITIs_%sN_' %(itiName, ntName)
    suffnei = 'prev_%sITIs_excInh_' %(itiName)
else:
    if len(ep_ms)==2:
        suffn = 'curr_%sN_ep%d-%dms_' %(ntName, ep_ms[0], ep_ms[-1])   
        suffnei = 'curr_%s_ep%d-%dms_' %('excInh', ep_ms[0], ep_ms[-1])   
    elif len(ep_ms)==0:
        suffn = 'curr_%sN_epVar_' %(ntName)   
        suffnei = 'curr_%s_epVar_' %('excInh')   
print '\n', suffn[:-1], ' - ', suffnei[:-1]



# Make folder named SVM to save figures inside it
svmdir = os.path.join(figsDir, 'SVM')
if not os.path.exists(svmdir):
    os.makedirs(svmdir)    


daysOrig = days
numDays = len(days);
print 'Analyzing mouse',mousename,'-', len(days), 'days'
    


#### Function definitions ####

#%% Function to only show left and bottom axes of plots, make tick directions outward, remove every other tick label if requested.
def makeNicePlots(ax, rmv2ndXtickLabel=0, rmv2ndYtickLabel=0):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # Make tick directions outward    
    ax.tick_params(direction='out')    
    # Tweak spacing between subplots to prevent labels from overlapping
    #plt.subplots_adjust(hspace=0.5)
#    ymin, ymax = ax.get_ylim()

    # Remove every other tick label
    if rmv2ndXtickLabel:
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[::2]]
        
    if rmv2ndYtickLabel:
        [label.set_visible(False) for label in ax.yaxis.get_ticklabels()[::2]]
    
    # gap between tick labeles and axis
#    ax.tick_params(axis='x', pad=30)
        
#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
# whichfile: index of svm file, after sorting by date (latest will be 0) in case there are several svm files.Set to [] if you want all svm files.       
def setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName='all', whichfile=0):
    import glob
    
    '''
    if trialHistAnalysis:
    #    ep_ms = np.round((ep-eventI)*frameLength)
    #    th_stim_dur = []
        svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
    else:
        svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
    '''    
    if ~np.isnan(roundi): # mat file name includes the round number
        rt = 'r%d_*' %(roundi)
    else:
        rt = '*-*_*_*' # the new names that include the date and time of analysis
    
    if trialHistAnalysis:
    #     ep_ms = np.round((ep-eventI)*frameLength)
    #    th_stim_dur = []
        svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_%s' %(ntName, itiName, rt)
    else:
        svmn = 'svmCurrChoice_%sN_ep*ms_%s' %(ntName, rt)   


    svmn = svmn + pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    if isinstance(whichfile, int): # if an int only take a single svm file, otherwise output all svm files.
        svmName = svmName[whichfile] # if 0, get the latest file
    
    
    return svmName
    

#%% Function to extend the built in two tailed ttest function to one-tailed
def ttest2(a, b, **tailOption):
    import scipy.stats as stats
    import numpy as np
    h, p = stats.ttest_ind(a, b)
    d = np.mean(a)-np.mean(b)
    if tailOption.get('tail'):
        tail = tailOption.get('tail').lower()
        if tail == 'right':
            p = p/2.*(d>0) + (1-p/2.)*(d<0)
        elif tail == 'left':
            p = (1-p/2.)*(d>0) + p/2.*(d<0)
    if d==0:
        p = 1;
    return p     
    
 