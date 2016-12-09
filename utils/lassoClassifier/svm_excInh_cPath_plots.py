# -*- coding: utf-8 -*-
"""
 You don't need to run any other scripts before this one. It works on its own.
 This script includes analyses to compare exc vs inh decoding [when same  number of exc and inh are concatenated to train SVM]
 fract non0 and weights are compared
 To get vars for this script, you need to run mainSVM_excInh_cPath.py... normally on the cluster!
 The saved mat files which will be loaded here are named excInhC2_svmCurrChoice_ and excInhC2_svmPrevChoice
 (The mat files named excInhC dont have cross validation)



Pool SVM results of all days and plot summary figures

Created on Sun Oct 30 14:41:01 2016
@author: farznaj
"""

#%% 
mousename = 'fni17'

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.    
ep_ms = [809, 1109] # only for trialHistAnalysis=0
        
# Define days that you want to analyze
days = ['151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'];
#days = days[1:]
numRounds = 10; # number of times svm analysis was ran for the same dataset but sampling different sets of neurons.    
savefigs = False

fmt = ['pdf', 'svg', 'eps'] #'png', 'pdf': preserve transparency # Format of figures for saving
figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/' # Directory for saving figures.
neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
eps = 10**-10 # tiny number below which weight is considered 0
palpha = .05 # p <= palpha is significant
thR = 2 # Exclude days with only <=thR rounds with non-0 weights


##%%
import os
import glob
import numpy as np   
import scipy as sci
import scipy.io as scio
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from setImagingAnalysisNamesP import *

plt.rc('font', family='helvetica')        

##%%
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
    suffn = 'prev_%sITIs_%sN_' %(ntName, itiName)
    suffnei = 'prev_%sITIs_excInh_' %(itiName)
else:
    suffn = 'curr_%sN_ep%d-%dms_' %(ntName, ep_ms[0], ep_ms[-1])   
    suffnei = 'curr_%s_ep%d-%dms_' %('excInh', ep_ms[0], ep_ms[-1])   
print '\n', suffn[:-1], ' - ', suffnei[:-1]



# Make folder named SVM to save figures inside it
svmdir = os.path.join(figsDir, 'SVM')
if not os.path.exists(svmdir):
    os.makedirs(svmdir)    


daysOrig = days
numDays = len(days);




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
def setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName='all'):
    import glob
    
    if trialHistAnalysis:
    #    ep_ms = np.round((ep-eventI)*frameLength)
    #    th_stim_dur = []
        if np.isnan(roundi): # set excInh name
            svmn = 'excInhC2_svmPrevChoice_%sN_%sITIs_ep*-*ms_*' %(ntName, itiName) # C2 has test/trial shuffles as well. C doesn't have it.
        else:
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
    else:
        if np.isnan(roundi): # set excInh name
            svmn = 'excInhC2_svmCurrChoice_%sN_ep%d-%dms_*' %(ntName, ep_ms[0], ep_ms[-1])   
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
    
    svmn = svmn + pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
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
    
    
    
    
       
#%%
"""       
''' 
##############################################################################################   
########### Find days with all-0 weights for all rounds and exclude them ##########################     
########### Also find days with few rounds with non-zero weights (these are outlier days) ##############
####################################################################################################               
'''

w_alln = [];
choiceBefEpEndFract = np.full((1, len(days)), 0).flatten()
    
for iday in range(len(days)):
    
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    #%%
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
    
    # from setImagingAnalysisNamesP import *
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))


    #%% For current choice analysis, identify days with trials in which choice happened after the training epoch 

    if trialHistAnalysis==0:
        
        Data = scio.loadmat(postName, variable_names=['timeStimOnset', 'time1stSideTry', 'outcomes'])
        timeStimOnset = np.array(Data.pop('timeStimOnset')).flatten().astype('float')
        time1stSideTry = np.array(Data.pop('time1stSideTry')).flatten().astype('float')
        outcomes = (Data.pop('outcomes').astype('float'))[0,:]
        
        # We first set to nan timeStimOnset of trials that anyway wont matter bc their outcome is  not of interest. we do this to make sure these trials dont affect our estimate of ep_ms
        if outcome2ana == 'corr':
            timeStimOnset[outcomes!=1] = np.nan; # analyze only correct trials.
        elif outcome2ana == 'incorr':
            timeStimOnset[outcomes!=0] = np.nan; # analyze only incorrect trials.  
        
        a = (time1stSideTry - timeStimOnset)
        ii = a <= ep_ms[-1]; # trials in which choice happened earlier than the end of training epoch ... we don't want them!
        
        print '%.2f%% trials have choice before the end of ep (%dms)!' %(np.mean(ii)*100, ep_ms[-1])
        if sum(ii)>0:
            print '\t, Choice times: ', a[ii]            
            choiceBefEpEndFract[iday] = np.mean(ii);
    
    
    #%% loop over the 10 rounds of analysis for each day

    for i in range(numRounds):
        roundi = i+1        
        svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName)
#        if roundi==1:
        print '\t', os.path.basename(svmName)
            
        ##%% Load vars (w, etc)
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]    
        w = w/sci.linalg.norm(w); # normalzied weights
        w_alln.append(w)

w_alln_all = np.concatenate(w_alln, axis=0)


#%% Find days with all-0 weights (in all rounds)

b = np.reshape(w_alln, (numDays,numRounds))

# for each day find number of rounds without all-zero weights
nRoundsNonZeroW = [np.sum([abs(x)>eps for x in np.nansum(map(None, b[i,:]), axis=1)]) for i in range(numDays)]  # /float(numRounds)
print '\nnumber of rounds with non-zero weights for each day = ',  nRoundsNonZeroW


sumw = [np.nansum(map(None, b[i,:]), axis=0).sum() for i in range(numDays)] # sum across rounds and neurons for each day
all0d = [not x for x in sumw]
print '\n', sum(all0d), 'days with all-0 weights: ', np.array(days)[np.array(all0d, dtype=bool)]
all0days = np.argwhere(all0d).flatten() # index of days with all-0 weights


# days with few non-zero rounds
r0w = [nRoundsNonZeroW[i]<=thR for i in range(len(nRoundsNonZeroW))]
fewRdays = np.argwhere(r0w).squeeze()
print sum(r0w), 'days with <=', thR, 'rounds of non-zero weights: ', np.array(days)[np.array(r0w, dtype=bool)]


# Find days that have more than 10% (?) of the trials with choice earlier than ep end and perhaps exclude them! Add this later.
if trialHistAnalysis==0:
    np.set_printoptions(precision=3)
    print '%d days have choice before the end of ep (%dms)\n\t%% of trials with early choice: ' %(sum(choiceBefEpEndFract>0), ep_ms[-1]), choiceBefEpEndFract[choiceBefEpEndFract!=0]*100
        
        
#%% Exclude days with all-0 weights (in all rounds) from analysis
'''
print 'Excluding %d days from analysis because all SVM weights of all rounds are 0' %(sum(all0d))
days = np.delete(daysOrig, all0days)
'''
##%% Exclude days with only few (<=thR) rounds with non-0 weights from analysis
print 'Excluding %d days from analysis: they have <%d rounds with non-zero weights' %(len(fewRdays), thR)
days = np.delete(daysOrig, fewRdays)

numDays = len(days)
print 'Using', numDays, 'days for analysis:', days


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        fewRdays0 = fewRdays
        days0 = days
        nRoundsNonZeroW0 = nRoundsNonZeroW
    elif iTiFlg==1:
        fewRdays1 = fewRdays
        days1 = days
        nRoundsNonZeroW1 = nRoundsNonZeroW

"""


    
    
    
    
    


#%%
'''
#####################################################################################################################################################    
################# Excitatory vs inhibitory neurons:  c path #########################################################################################
######## Fraction of non-zero weights vs. different values of c #####################################################################################
#####################################################################################################################################################   
'''    
    
wall_exc = []
wall_inh = []
perActiveAll_exc = []
perActiveAll_inh = []
cvect_all = []
perClassErAll = []
ball = []
perClassErTestAll = []    

for iday in range(len(days)):

    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    #%% Set .mat file names
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
    
    # from setImagingAnalysisNamesP import *
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))
    
    # Load inhibitRois
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]

    
    #%%     
    svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName)    
    print os.path.basename(svmName)
        
    ##%% Load vars
    Data = scio.loadmat(svmName, variable_names=['perActive_exc', 'perActive_inh', 'wei_all', 'perClassEr', 'cvect_', 'bei_all', 'perClassErTest'])
    perActive_exc = Data.pop('perActive_exc') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)
    perActive_inh = Data.pop('perActive_inh') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)        
    wei_all = Data.pop('wei_all') # numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers) # numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  
    perClassEr = Data.pop('perClassEr') # numSamples x numTrialShuff x length(cvect_)
    cvect_ = Data.pop('cvect_').flatten() # length(cvect_)
    bei_all = Data.pop('bei_all') # numSamples x numTrialShuff x length(cvect_)  # numTrialShuff is number of times you shuffled trials to do cross validation. (it is variable numShuffles_ei in function inh_exc_classContribution)
    perClassErTest = Data.pop('perClassErTest') # numSamples x numTrialShuff x length(cvect_) 
    
    if abs(wei_all.sum()) < eps:
        print '\tWeights of all c and all shuffles are 0' # ... not analyzing' # I dont think you should do this!!
        
#    else:            
    ##%% Load vars (w, etc)
    Data = scio.loadmat(svmName, variable_names=['NsExcluded', 'NsRand'])        
    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
    NsRand = Data.pop('NsRand')[0,:].astype('bool')

    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
    inhRois = inhibitRois[~NsExcluded]        
    inhRois = inhRois[NsRand]
        
    n = sum(inhRois==1)                
    inhRois_ei = np.zeros((2*n));
    inhRois_ei[n:2*n] = 1;
        
        
    # if cvTrialShuffle was performed we take averages across trialShuffles        
    wall_exc.append(np.mean(wei_all[:,:,:,inhRois_ei==0], axis=1)) #average across trialShfls # numDays (each day: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers))     
    wall_inh.append(np.mean(wei_all[:,:,:,inhRois_ei==1], axis=1))                
#    wall_exc.append(wei_all[:,:,inhRois_ei==0]) # numDays (each day: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers))     
#    wall_inh.append(wei_all[:,:,inhRois_ei==1])        
    ball.append(np.mean(bei_all, axis=1)) # numDays x numSamples x length(cvect_)   
    
    perActiveAll_exc.append(np.mean(perActive_exc, axis=1)) # numDays x numSamples x length(cvect_)        
    perActiveAll_inh.append(np.mean(perActive_inh, axis=1))        

    cvect_all.append(cvect_) # numDays x length(cvect_)
    perClassErAll.append(np.mean(perClassEr, axis=1)) # numDays x numSamples x length(cvect_)     
    perClassErTestAll.append(np.mean(perClassErTest, axis=1)) # numDays x numSamples x length(cvect_)     
   
   
   
   
#%%    
perActiveAll_exc = np.array(perActiveAll_exc)  # days x numSamples x length(cvect_) # percent non-zero weights
perActiveAll_inh = np.array(perActiveAll_inh)  # days x numSamples x length(cvect_) # percent non-zero weights
perClassErAll = np.array(perClassErAll)  # days x numSamples x length(cvect_) # classification error
cvect_all = np.array(cvect_all) # days x length(cvect_)  # c values
perClassErTestAll = np.array(perClassErTestAll)
ball = np.array(ball)


#%% For each day average weights across neurons
nax = 2 # dimension of neurons in wall_exc (since u're averaging across trialShuffles, it will be 2.) (if cross validation is performed, it will be 3, ie in excinhC2 mat files. Otherwise it will be 2.)
wall_exc_aveN = []
wall_inh_aveN = []
wall_abs_exc_aveN = [] # average of abs values of weights
wall_abs_inh_aveN = []
wall_non0_exc_aveN = [] # average of abs values of weights
wall_non0_inh_aveN = []
wall_non0_abs_exc_aveN = [] # average of abs values of weights
wall_non0_abs_inh_aveN = []
for iday in range(len(days)):
    wall_exc_aveN.append(np.mean(wall_exc[iday], axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_inh_aveN.append(np.mean(wall_inh[iday], axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_abs_exc_aveN.append(np.mean(abs(wall_exc[iday]), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_abs_inh_aveN.append(np.mean(abs(wall_inh[iday]), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    a = wall_exc[iday] + 0
    a[wall_exc[iday]==0] = np.nan # only take non-0 weights
    wall_non0_exc_aveN.append(np.nanmean(a, axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_non0_abs_exc_aveN.append(np.nanmean(abs(a), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    a = wall_inh[iday] + 0 
    a[wall_inh[iday]==0] = np.nan
    wall_non0_inh_aveN.append(np.nanmean(a, axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons    
    wall_non0_abs_inh_aveN.append(np.nanmean(abs(a), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    
wall_exc_aveN = np.array(wall_exc_aveN)
wall_inh_aveN = np.array(wall_inh_aveN)
wall_abs_exc_aveN = np.array(wall_abs_exc_aveN)
wall_abs_inh_aveN = np.array(wall_abs_inh_aveN)
wall_non0_exc_aveN = np.array(wall_non0_exc_aveN)
wall_non0_inh_aveN = np.array(wall_non0_inh_aveN)
wall_non0_abs_exc_aveN = np.array(wall_non0_abs_exc_aveN)
wall_non0_abs_inh_aveN = np.array(wall_non0_abs_inh_aveN)


#%% Average weights,etc across shuffles for each day : numDays x length(cvect_) 
# NOTE: you may want to exclude shuffles with all0 weights from averaging

#avax = (0,1) # (0) # average across axis; use (0,1) if shuffles of trials (for cross validation) as well as neurons are available. Otherwise use (0) if no crossvalidation was performed

wall_exc_aveNS = np.array([np.mean(wall_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_inh_aveNS = np.array([np.mean(wall_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_abs_exc_aveNS = np.array([np.mean(wall_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_abs_inh_aveNS = np.array([np.mean(wall_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_exc_aveNS = np.array([np.nanmean(wall_non0_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_inh_aveNS = np.array([np.nanmean(wall_non0_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_abs_exc_aveNS = np.array([np.nanmean(wall_non0_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_abs_inh_aveNS = np.array([np.nanmean(wall_non0_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
perClassErAll_aveS = np.array([np.mean(perClassErAll[iday,:,:], axis=0) for iday in range(len(days))])
perClassErTestAll_aveS = np.array([np.mean(perClassErTestAll[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_exc_aveS = np.array([np.mean(perActiveAll_exc[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_inh_aveS = np.array([np.mean(perActiveAll_inh[iday,:,:], axis=0) for iday in range(len(days))])

# std
wall_exc_stdNS = np.array([np.std(wall_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # std of w across shuffles for each day
wall_inh_stdNS = np.array([np.std(wall_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_abs_exc_stdNS = np.array([np.std(wall_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # std of w across shuffles for each day
wall_abs_inh_stdNS = np.array([np.std(wall_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_exc_stdNS = np.array([np.nanstd(wall_non0_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_inh_stdNS = np.array([np.nanstd(wall_non0_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_abs_exc_stdNS = np.array([np.nanstd(wall_non0_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_abs_inh_stdNS = np.array([np.nanstd(wall_non0_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
perClassErAll_stdS = np.array([np.std(perClassErAll[iday,:,:], axis=0) for iday in range(len(days))])
perClassErTestAll_stdS = np.array([np.std(perClassErTestAll[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_exc_stdS = np.array([np.std(perActiveAll_exc[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_inh_stdS = np.array([np.std(perActiveAll_inh[iday,:,:], axis=0) for iday in range(len(days))])


#%% For each day compute exc-inh for shuffle-averages. Also set %non0, weights, etc to nan when both exc and inh percent non-0 are 0 or 100.

perActiveEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_non0 = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI_non0 = np.full((perActiveAll_exc_aveS.shape), np.nan)

perActiveEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_non0_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI_non0_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)

for iday in range(len(days)):
    perActiveEmI[iday,:] = perActiveAll_exc_aveS[iday,:] - perActiveAll_inh_aveS[iday,:] # days x length(cvect_) # at each c value, divide %non-zero (averaged across shuffles) of exc and inh  
    wabsEmI[iday,:] = wall_abs_exc_aveNS[iday,:] - wall_abs_inh_aveNS[iday,:]
    wabsEmI_non0[iday,:] = wall_non0_abs_exc_aveNS[iday,:] - wall_non0_abs_inh_aveNS[iday,:]
    wEmI[iday,:] = wall_exc_aveNS[iday,:] - wall_inh_aveNS[iday,:]
    wEmI_non0[iday,:] = wall_non0_exc_aveNS[iday,:] - wall_non0_inh_aveNS[iday,:]
#    plt.plot(perActiveEmI[iday,:])

    # exclude c values for which both exc and inh %non-0 are 0 or 100.
    # you need to do +0, if not perActiveAll_exc_aveS will be also set to nan (same for w)
    a = perActiveAll_exc_aveS[iday,:]
    b = perActiveAll_inh_aveS[iday,:]    
    i0 = np.logical_and(a==0, b==0) # c values for which all weights of both exc and inh populations are 0.
    i100 = np.logical_and(a==100, b==100)  # c values for which all weights of both exc and inh populations are 100.
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    perActiveEmI_b0100nan[iday,:] = a - b
    
    a = wall_abs_exc_aveNS[iday,:]
    b = wall_abs_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wabsEmI_b0100nan[iday,:] = a - b
    
    a = wall_non0_abs_exc_aveNS[iday,:]
    b = wall_non0_abs_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wabsEmI_non0_b0100nan[iday,:] = a - b

    a = wall_exc_aveNS[iday,:]
    b = wall_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wEmI_b0100nan[iday,:] = a - b
    
    a = wall_non0_exc_aveNS[iday,:]
    b = wall_non0_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wEmI_non0_b0100nan[iday,:] = a - b


    a = perClassErAll_aveS[iday,:] 
    a[i0] = np.nan
    a[i100] = np.nan
    
    
    a = perClassErTestAll_aveS[iday,:] 
    a[i0] = np.nan
    a[i100] = np.nan
    
    
#%% Set the null distribution for exc-inh comparison: For each day subtract random shuffles of exc (same for inh) from each other.

import numpy.random as rng
l = perActiveAll_exc.shape[1] # numSamples
randAll_perActiveAll_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # days x numSamples x length(cvect_) # exc - exc
randAll_perActiveAll_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_perActiveAll_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_wabs_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_wabs_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_wabs_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_wabs_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_wabs_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_wabs_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_w_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_w_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_w_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_w_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_w_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_w_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh

randAll_perActiveAll_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # days x numSamples x length(cvect_) # exc - exc
randAll_perActiveAll_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_perActiveAll_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_wabs_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_wabs_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_wabs_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_wabs_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_wabs_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_wabs_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_w_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_w_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_w_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_w_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_w_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_w_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh

for iday in range(len(days)):
    
    ################## perc non-zero ######################
    # exc-exc
    inds = rng.permutation(l) # random indeces of shuffles
    inds1 = rng.permutation(l)
    a = perActiveAll_exc[iday,inds1,:] - perActiveAll_exc[iday,inds,:]
    randAll_perActiveAll_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = perActiveAll_exc[iday,inds1,:]# + 0 ... s
    bb = perActiveAll_exc[iday,inds,:]# + 0
    i0 = np.logical_and(aa==0, bb==0)
    i100 = np.logical_and(aa==100, bb==100)
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_perActiveAll_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    # inh-inh
    indsi = rng.permutation(l) # random indeces of shuffles
    inds1i = rng.permutation(l)
    b = perActiveAll_inh[iday,inds1i,:] - perActiveAll_inh[iday,indsi,:]
    randAll_perActiveAll_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = perActiveAll_inh[iday,inds1i,:]# + 0
    bb = perActiveAll_inh[iday,indsi,:]# + 0
    i0i = np.logical_and(aa==0, bb==0)
    i100i = np.logical_and(aa==100, bb==100)
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_perActiveAll_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    # pool exc - exc; inh - inh
    randAll_perActiveAll_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_perActiveAll_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  
    
    
    
    ###################### w ######################
    # exc-exc
    # below uncommented, so %non-0 and ws are computed from the same set of shuffles.
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_exc_aveN[iday,inds1,:] - wall_exc_aveN[iday,inds,:]
    randAll_w_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_exc_aveN[iday,inds1,:]
    bb = wall_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_w_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.

    
    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_inh_aveN[iday,inds1i,:] - wall_inh_aveN[iday,indsi,:]
    randAll_w_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_inh_aveN[iday,inds1i,:]
    bb = wall_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_w_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_w_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_w_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  

    
    ###################### non-0 w ######################
    # exc-exc
    # below uncommented, so %non-0 and ws are computed from the same set of shuffles.
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_non0_exc_aveN[iday,inds1,:] - wall_non0_exc_aveN[iday,inds,:]
    randAll_non0_w_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_non0_exc_aveN[iday,inds1,:]
    bb = wall_non0_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_non0_w_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.

    
    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_non0_inh_aveN[iday,inds1i,:] - wall_non0_inh_aveN[iday,indsi,:]
    randAll_non0_w_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_non0_inh_aveN[iday,inds1i,:]
    bb = wall_non0_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_non0_w_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_non0_w_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_non0_w_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  


    
    ###################### abs w ######################
    # exc-exc
    # below uncommented, so %non-0 and ws are computed from the same set of shuffles.
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_abs_exc_aveN[iday,inds1,:] - wall_abs_exc_aveN[iday,inds,:]
    randAll_wabs_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_abs_exc_aveN[iday,inds1,:]
    bb = wall_abs_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_wabs_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.

    
    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_abs_inh_aveN[iday,inds1i,:] - wall_abs_inh_aveN[iday,indsi,:]
    randAll_wabs_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_abs_inh_aveN[iday,inds1i,:]
    bb = wall_abs_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_wabs_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_wabs_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_wabs_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  
    
    
    
    ###################### abs non-0 w ######################
    # exc-exc
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_non0_abs_exc_aveN[iday,inds1,:] - wall_non0_abs_exc_aveN[iday,inds,:]
    randAll_non0_wabs_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    # you don't really need to do it for here, bc it is already non0 ws!
    aa = wall_non0_abs_exc_aveN[iday,inds1,:]
    bb = wall_non0_abs_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_non0_wabs_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    

    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_non0_abs_inh_aveN[iday,inds1i,:] - wall_non0_abs_inh_aveN[iday,indsi,:]
    randAll_non0_wabs_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_non0_abs_inh_aveN[iday,inds1i,:]
    bb = wall_non0_abs_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_non0_wabs_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_non0_wabs_ei_aveS[iday,:,:] = np.concatenate((a, b))  
    randAll_non0_wabs_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  
    
    


#%%
# BEST C

##########################################################################################################################################################################
################################## compare weights and fract non-0 of exc and inh at bestc of each day ##################################
##########################################################################################################################################################################

#%% Compute bestc for each day (for the exc_inh population decdoder)
numSamples = np.shape(perClassErTestAll)[1]
cbestAll = np.full((len(days)), np.nan)

for iday in range(len(days)):

    cvect = cvect_all[iday,:]

    # Pick bestc from all range of c... it may end up a value at which all weights are 0. In the analysis below though you exclude shuffles with all0 weights.
    meanPerClassErrorTest = np.mean(perClassErTestAll[iday,:,:], axis = 0) # perClassErTestAll_aveS[iday,:]
    semPerClassErrorTest = np.std(perClassErTestAll[iday,:,:], axis = 0)/np.sqrt(numSamples) # perClassErTestAll_stdS[iday,:]/np.sqrt(np.shape(perClassErTestAll)[1])
    ix = np.nanargmin(meanPerClassErrorTest)
    cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
    cbest = cbest[0]; # best regularization term based on minError+SE criteria
    cbestAll[iday] = cbest
    
    
    # Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)        
    # compute percent non-zero w at each c
    a = perActiveAll_exc[iday,:,:].squeeze()
    b = perActiveAll_inh[iday,:,:].squeeze()    
    c = np.mean((a,b), axis=0) # percent non-0 w for each shuffle at each c (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
    
#    a = (wAllC!=0) # non-zero weights
#    b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
    b = np.mean(c, axis=(0)) # Fraction of non-zero weights (averaged across shuffles)
    c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
    
    cvectnow = cvect[c1stnon0:]
    meanPerClassErrorTestnow = np.mean(perClassErTestAll[iday,:,c1stnon0:], axis = 0);
    semPerClassErrorTestnow = np.std(perClassErTestAll[iday,:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
    ix = np.argmin(meanPerClassErrorTestnow)
    cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
    cbest = cbest[0]; # best regularization term based on minError+SE criteria
    cbestAll[iday] = cbest

print cbestAll


#%% compare exc vs inh dists (across all shuffls) at bestc for fract non0 and weights
# IMPORTANT NOTE: I think you should exclude days with chance testing-data decoding performance... bc it doesn't really make sense to compare their weight or fract-non0 of exc and inh neurons.

perActiveExc = []
perActiveInh = []
wNon0AbsExc = []
wNon0AbsInh = []    
wAbsExc = []
wAbsInh = []
perClassErTest_bestc = np.full(len(days), np.nan)
perClassErTest_bestc_sd = np.full(len(days), np.nan)

for iday in range(len(days)):
    
    ibc = cvect_all[iday,:]==cbestAll[iday]

    # class error for cv data at bestc
    perClassErTest_bestc[iday] = perClassErTestAll_aveS[iday,ibc]
    perClassErTest_bestc_sd[iday] = perClassErTestAll_stdS[iday,ibc]
    
    
    # percent non-zero w
    a = perActiveAll_exc[iday,:,ibc].squeeze()
    b = perActiveAll_inh[iday,:,ibc].squeeze()
    c = np.mean((a,b), axis=0) #nShuffs x 1 # percent non-0 w for each shuffle (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
    i0 = (c==0) # shuffles at which all w (of both exc and inh) are 0, ie no decoder was identified for these shuffles.
    a[i0] = np.nan # ignore shuffles with all-0 weights
    b[i0] = np.nan # ignore shuffles with all-0 weights    
#    a[a==0.] = np.nan # ignore shuffles with all-0 weights ... this is problematic bc u will exclude a shuffle if inh w are all 0 even if exc w are not.
#    b[b==0.] = np.nan # ignore shuffles with all-0 weights
    perActiveExc.append(a)
    perActiveInh.append(b)

    # abs non-zero w
    a = wall_non0_abs_exc_aveN[iday,:,ibc].squeeze() # for some shuffles this will be nan bc all weights were 0 ... so automatically you are ignoring shuffles w all-0 weights
    b = wall_non0_abs_inh_aveN[iday,:,ibc].squeeze()
#    print np.argwhere(np.isnan(a))
    wNon0AbsExc.append(a)
    wNon0AbsInh.append(b)
    
    # abs w
    a = wall_abs_exc_aveN[iday,:,ibc].squeeze() 
    b = wall_abs_inh_aveN[iday,:,ibc].squeeze()
    a[i0] = np.nan # ignore shuffles with all-0 weights
    b[i0] = np.nan # ignore shuffles with all-0 weights    
#    a[a==0.] = np.nan # ignore shuffles with all-0 weights
#    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
    wAbsExc.append(a)
    wAbsInh.append(b)
    

    
perActiveExc = np.array(perActiveExc)
perActiveInh = np.array(perActiveInh)
wNon0AbsExc = np.array(wNon0AbsExc)
wNon0AbsInh = np.array(wNon0AbsInh)    
wAbsExc = np.array(wAbsExc)
wAbsInh = np.array(wAbsInh) 


#%%
###################################### PLOTS ###################################### 
dnow = '/excInh_afterSFN'
if trialHistAnalysis:
    dp = '/previousChoice'
else:
    dp = '/currentChoice'

lab1 = 'exc'
lab2 = 'inh'

def histerrbar(a,b,binEvery,p):
#    import matplotlib.gridspec as gridspec
    
#    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
#    binEvery = r/float(10)

#    _, p = stats.ttest_ind(a, b, nan_policy='omit')

#    plt.figure(figsize=(5,3))    
#    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1]) 
#    h1 = gs[0,0:2]
#    h2 = gs[0,2:3]

#    lab1 = 'exc'
#    lab2 = 'inh'
    
    # set bins
    bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
    bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value
    # set hist of a
    hist, bin_edges = np.histogram(a, bins=bn)
    hist = hist/float(np.sum(hist))    
    # Plot hist of a
    ax = plt.subplot(h1) #(gs[0,0:2])
    plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.4, label=lab1)
    
    # set his of b
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist));     #d = stats.mode(np.diff(bin_edges))[0]/float(2)
    # Plot hist of b
    plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.4, label=lab2)
    
    plt.legend(loc=0, frameon=False)
    plt.ylabel('Prob (all days & N shuffs at bestc)')
    plt.title('mean diff= %.3f, p=%.3f' %(np.mean(a)-np.mean(b), p))
    #plt.xlim([-.5,.5])
    plt.xlabel(lab)
    makeNicePlots(ax)
    
    
    # errorbar
    ax = plt.subplot(h2) #(gs[0,2:3])
    plt.errorbar([0,1], [a.mean(),b.mean()], [a.std()/np.sqrt(len(a)), b.std()/np.sqrt(len(b))],marker='o',color='k', fmt='.')
    plt.xlim([-1,2])
#    plt.title('%.3f, %.3f' %(a.mean(), b.mean()))
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.ylabel(lab)
    makeNicePlots(ax)
#    plt.tick_params
    
    plt.subplots_adjust(wspace=1, hspace=.5)



def errbarAllDays(a,b,p):
    eav = np.nanmean(a, axis=1)
    iav = np.nanmean(b, axis=1)
    ele = np.shape(a)[1] - np.sum(np.isnan(a),axis=1) # number of non-nan shuffles of each day
    ile = np.shape(b)[1] - np.sum(np.isnan(b),axis=1) # number of non-nan shuffles of each day
    esd = np.divide(np.nanstd(a, axis=1), np.sqrt(ele))
    isd = np.divide(np.nanstd(b, axis=1), np.sqrt(ile))
    
    pp = p
    pp[p>.05] = np.nan
    pp[p<=.05] = np.max((eav,iav))
    x = np.arange(np.shape(eav)[0])
    
    plt.errorbar(x, eav, esd, color='k')
    plt.errorbar(x, iav, isd, color='r')
    plt.plot(x, pp, marker='*',color='r', linestyle='')
    plt.xlim([-1, x[-1]+1])
    plt.xlabel('Days')
    makeNicePlots(ax)


#%% 
######################################  All days pooled ###################################### 

###############%% classification accuracy of cv data at best c for each day ###############

plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
plt.subplot(h1)
plt.errorbar(np.arange(len(days)), 100-perClassErTest_bestc, perClassErTest_bestc_sd)
plt.xlim([-1,len(days)+1])
plt.xlabel('Days')
plt.ylabel('% Class accuracy')
makeNicePlots(plt.gca())

if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/bestC'+dp, suffn[0:5]+'classAccur'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



###############%% percent non-zero w ###############    
### hist and P val for all days pooled (exc vs inh)
lab = '% non-0 w'
binEvery = 10# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(perActiveExc,(-1,)) 
b = np.reshape(perActiveInh,(-1,))  
a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)

### show individual days
a = perActiveExc
b = perActiveInh
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
#plt.figure(); plot.subplot(221)
ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
plt.ylabel(lab)


#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)
if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/bestC'+dp, suffn[0:5]+'excVSinh_allDays_percNon0W'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



###############%% abs non-0 w ###############
### hist and P val for all days pooled (exc vs inh)
lab = 'abs non-0 w'
binEvery = .01# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(wNon0AbsExc,(-1,)) 
b = np.reshape(wNon0AbsInh,(-1,))

a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)

### show individual days
a = wNon0AbsExc
b = wNon0AbsInh
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
plt.ylabel(lab)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/bestC'+dp, suffn[0:5]+'excVSinh_allDays_absNon0W'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    
    
    
###############%% abs w  ###############
### hist and P val for all days pooled (exc vs inh)
lab = 'abs w'
binEvery = .005 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(wAbsExc,(-1,)) 
b = np.reshape(wAbsInh,(-1,))

a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)

### show individual days
a = wAbsExc
b = wAbsInh
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
plt.ylabel(lab)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/bestC'+dp, suffn[0:5]+'excVSinh_allDays_absW'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')





#%%
###################################### Plot each day separately ######################################
for iday in range(len(days)):
    
    print '\n_________________'+days[iday]+'_________________'    
    
    ###################
    lab = '% non-0 w'
#    binEvery = 10# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    a = np.reshape(perActiveExc[iday,:],(-1,)) 
    b = np.reshape(perActiveInh[iday,:],(-1,))
    
    a = a[~(np.isnan(a) + np.isinf(a))]
    b = b[~(np.isnan(b) + np.isinf(b))]
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)    

    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    '''
    print 'exc vs inh (bestc):p value= %.3f' %(p)
    print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
    '''
    plt.figure(figsize=(6,8.5))    
    gs = gridspec.GridSpec(3, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:2]
    h2 = gs[0,2:3]
    histerrbar(a,b,binEvery,p)    
    plt.title('cvClassEr= %.2f' %(perClassErTest_bestc[iday]))

    ###################
    lab = 'abs non-0 w'
    a = np.reshape(wNon0AbsExc[iday,:],(-1,)) 
    b = np.reshape(wNon0AbsInh[iday,:],(-1,))
    
    a = a[~(np.isnan(a) + np.isinf(a))]
    b = b[~(np.isnan(b) + np.isinf(b))]
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)    

    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    '''
    print 'exc vs inh (bestc):p value= %.3f' %(p)
    print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
    '''
    h1 = gs[1,0:2]
    h2 = gs[1,2:3]
    if np.logical_and(b.shape[0], a.shape[0])!=0:
        histerrbar(a,b,binEvery,p)


    ###################
    lab = 'abs w'
    a = np.reshape(wAbsExc[iday,:],(-1,)) 
    b = np.reshape(wAbsInh[iday,:],(-1,))
    
    a = a[~(np.isnan(a) + np.isinf(a))]
    b = b[~(np.isnan(b) + np.isinf(b))]
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)    

    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    '''
    print 'exc vs inh (bestc):p value= %.3f' %(p)
    print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
    '''
    h1 = gs[2,0:2]
    h2 = gs[2,2:3]
    histerrbar(a,b,binEvery,p) 
    
    
    if savefigs:#% Save the figure
        fign = os.path.join(svmdir+dnow+'/bestC'+dp, suffn[0:5]+'excVSinh_'+days[iday][0:6]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')


    plt.show()
    

#%% exc minus inh at best c (average of shuffles) and null dist (subtraction of rand shuffles) at bestc... this is already done in computing b0100nan vars: you should perhaps exclude shuffles with all-0 weights bc they may mask the effects. (?)

perActiveEmI_bestc = np.full((len(days)), np.nan)
r_perActive_bestc = np.full((len(days),2*l), np.nan)
wabsEmI_bestc = np.full((len(days)), np.nan)
r_wabs_bestc = np.full((len(days),2*l), np.nan)
wabsEmI_non0_bestc = np.full((len(days)), np.nan)
r_non0_wabs_bestc = np.full((len(days),2*l), np.nan)
wEmI_bestc = np.full((len(days)), np.nan)
r_w_bestc = np.full((len(days),2*l), np.nan)
wEmI_non0_bestc = np.full((len(days)), np.nan)
r_non0_w_bestc = np.full((len(days),2*l), np.nan)

p_perActive = np.full((len(days)), np.nan)
p_wAbsNon0 = np.full((len(days)), np.nan)

for iday in range(len(days)):
    
    ibc = cvect_all[iday,:]==cbestAll[iday] # index of cbest
#    print np.argwhere(ibc)
    
    perActiveEmI_bestc[iday] = perActiveEmI_b0100nan[iday,ibc] # perActiveAll_exc_aveS[iday, ibc] - perActiveAll_inh_aveS[iday, ibc]
    r_perActive_bestc[iday,:] = randAll_perActiveAll_ei_aveS_b0100nan[iday,:,ibc]
    
    wEmI_bestc[iday] = wEmI_b0100nan[iday,ibc]
    r_w_bestc[iday,:] = randAll_w_ei_aveS_b0100nan[iday,:,ibc]
    
    wEmI_non0_bestc[iday] = wEmI_non0_b0100nan[iday,ibc]
    r_non0_w_bestc[iday,:] = randAll_non0_w_ei_aveS_b0100nan[iday,:,ibc]
    
    wabsEmI_bestc[iday] = wabsEmI_b0100nan[iday,ibc] 
    r_wabs_bestc[iday,:] = randAll_wabs_ei_aveS_b0100nan[iday,:,ibc]   
    
    wabsEmI_non0_bestc[iday] = wabsEmI_non0_b0100nan[iday,ibc] 
    r_non0_wabs_bestc[iday,:] = randAll_non0_wabs_ei_aveS_b0100nan[iday,:,ibc]    
#    _,p = stats.ttest_ind(perActiveAll_exc[iday,:,ibc].flatten(), perActiveAll_inh[iday,:,ibc].flatten()) #, nan_policy='omit')
#plt.hist(r_perActive_bestc[iday,:])
    _,p = stats.ttest_1samp(r_perActive_bestc[iday,:], perActiveEmI_bestc[iday], nan_policy='omit') 
    print p, '---', perActiveEmI_bestc[iday]
    p_perActive[iday] = p
    _,p = stats.ttest_1samp(r_non0_wabs_bestc[iday,:], wabsEmI_non0_bestc[iday], nan_policy='omit') 
    print p, '---', wabsEmI_non0_bestc[iday]
    p_wAbsNon0[iday] = p
    print '____'

'''
perActiveEmI_bestc[p_perActive<=.05]
wabsEmI_non0_bestc[p_wAbsNon0<=.05]


plt.plot(p_perActive); plt.plot(perActiveEmI_bestc)
plt.plot(p_wAbsNon0); plt.plot(wabsEmI_non0_bestc)

np.mean(p_perActive<=.05)
np.mean(p_wAbsNon0<=.05)
'''

########################################### PLOTS ###########################################
lab1 = 'exc - inh'
lab2 = 'rand'

###############%% percent non-zero w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(perActiveEmI_bestc,(-1,)) # pool across days x length(cvect_)  
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_perActive_bestc,(-1,)) # pool across days x numSamples x length(cvect_)
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


### hist and P val for all days pooled (exc vs inh)
lab = '% non-0 w'
binEvery = 3 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(10)
plt.figure(figsize=(5,15))    
gs = gridspec.GridSpec(5, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)


###############%% abs non-0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_non0_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_non0_wabs_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'abs non-0 w'
binEvery = .006# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(10)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[1,0:2]
h2 = gs[1,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)


###############%% abs w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_wabs_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'abs w'
binEvery = .002# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(10)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[2,0:2]
h2 = gs[2,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)


###############%% w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_w_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'w'
binEvery = .003 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(10)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[3,0:2]
h2 = gs[3,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)


###############%% non0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_non0_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_non0_w_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'non-0 w'
binEvery = .006# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(10)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[4,0:2]
h2 = gs[4,2:3]
histerrbar(a,b,binEvery,p)
plt.xlabel(lab)



#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)
if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/bestC'+dp, suffn[0:5]+'excMinusInh_allDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')








########################################################################################################################################################################
# ALL C

#%%
################################################################################################################################################################################################
########################## Pool all days and all c values, make histograms of exc - inh, and compare with null dists  ################################################
################################################################################################################################################################################################

###############%% percent non-zero w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(perActiveEmI_b0100nan,(-1,)) # pool across days x length(cvect_)  
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_perActiveAll_ei_aveS_b0100nan,(-1,)) # pool across days x numSamples x length(cvect_)
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)



binEvery = .5
#mv = 2
#bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
#bn = np.arange(-4,4, binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value
#bn[0] = np.min([np.min(a),np.min(b)])

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)

plt.figure(figsize=(3,20))    
plt.subplot(511)
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-20,20])
plt.xlabel('%non-zero w')
#plt.savefig('cpath_all_percnon0.svg', format='svg', dpi=300)


###############%% w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_w_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()    
plt.subplot(512)
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('w')
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)


###############%% non0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_non0_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_non0_w_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()   
plt.subplot(513) 
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('non-0 w')
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)



###############%% abs w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_wabs_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()  
plt.subplot(514)  
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('abs w')
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)


###############%% abs non-0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_non0_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_non0_wabs_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()    
plt.subplot(515)
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend(loc=0)
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('abs non-0 w')
#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)

plt.subplots_adjust(hspace=.3)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/allC'+dp, suffn[0:5]+'excMinusInh_AllDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



#%%
############################# Plot c-path of exc and inh for each day one by one #################################
####################################################################################
##%% Look at days one by one (averages across shuffles)

for iday in range(len(days)):
    
    print '____________________', days[iday], '____________________'
#    plt.figure()    
#    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:], perClassErTestAll_stdS[iday,:], color = 'g')
#    plt.xscale('log')
    
    plt.figure(figsize=(12,2))
    '''    
    plt.subplot(121)
    plt.errorbar(perClassErAll_aveS[iday,:], wall_exc_aveNS[iday,:], wall_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(perClassErAll_aveS[iday,:], wall_inh_aveNS[iday,:], wall_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.xscale('log')    
    plt.xlabel('Training error %')
    plt.ylabel('Mean+std of weights')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1,1))
    '''
    ax = plt.subplot(151)
    plt.errorbar(cvect_all[iday,:], perActiveAll_exc_aveS[iday,:], perActiveAll_exc_stdS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], perActiveAll_inh_aveS[iday,:], perActiveAll_inh_stdS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:], perClassErAll_stdS[iday,:], color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:], perClassErTestAll_stdS[iday,:], color = 'g')
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('% non-0 weights')   
    aa = perActiveEmI_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_perActiveAll_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]    
    _, p_ei = stats.ttest_ind(aa, bb)
    plt.title('%%non-0\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    
    
    ax = plt.subplot(152)
    plt.errorbar(cvect_all[iday,:], wall_exc_aveNS[iday,:], wall_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_inh_aveNS[iday,:], wall_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Weights')
    aa = wEmI_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_w_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)    
    plt.title('W\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    

    ax = plt.subplot(153)
    plt.errorbar(cvect_all[iday,:], wall_non0_exc_aveNS[iday,:], wall_non0_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_non0_inh_aveNS[iday,:], wall_non0_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Non-0 weights')
    aa = wEmI_non0_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_w_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)    
    plt.title('Non0 w\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    
    
    ax = plt.subplot(154)
    plt.errorbar(cvect_all[iday,:], wall_abs_exc_aveNS[iday,:], wall_abs_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_abs_inh_aveNS[iday,:], wall_abs_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Abs weights')
    aa = wabsEmI_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_wabs_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    plt.title('Abs w\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    

    ax = plt.subplot(155)
    plt.errorbar(cvect_all[iday,:], wall_non0_abs_exc_aveNS[iday,:], wall_non0_abs_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_non0_abs_inh_aveNS[iday,:], wall_non0_abs_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Abs non-0 weights')   
    aa = wabsEmI_non0_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_wabs_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    plt.title('Abs non0 w\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])    
    
    plt.subplots_adjust(hspace=.5, wspace=.35)      
#    raw_input()
#    plt.savefig('cpath'+days[iday]+'.svg', format='svg', dpi=300)    
      
    if savefigs:#% Save the figure
        fign = os.path.join(svmdir+dnow+'/allC'+dp, suffn[0:5]+'excInh_'+days[iday][0:6]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')

   
    plt.show()       
   
    ######################################################
    ###### p values: compare dists w each other ######
    # percent non-zero
    aa = perActiveAll_exc_aveS[iday,:]+0
    bb = perActiveAll_inh_aveS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\n%%non-zero p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)   
    '''
    # plot hists
    hist, bin_edges = np.histogram(aa, bins=30)
    hist = hist/float(np.sum(hist))
    fig = plt.figure(figsize=(4,2))
    plt.bar(bin_edges[0:-1], hist, .7, color='b')
    
    hist, bin_edges = np.histogram(bb, bins=30)
    hist = hist/float(np.sum(hist))
    plt.bar(bin_edges[0:-1], hist, .7, color='r')
    '''

    # weight
    aa = wall_exc_aveNS[iday,:]+0
    bb = wall_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nw: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)


    # non-0 weight
    aa = wall_non0_exc_aveNS[iday,:]+0
    bb = wall_non0_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nnon0 w: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)

    
    # abs weight
    aa = wall_abs_exc_aveNS[iday,:]+0
    bb = wall_abs_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nabs w: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)   


    # abs non0 weight
    aa = wall_non0_abs_exc_aveNS[iday,:]+0
    bb = wall_non0_abs_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nabs non0 w: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)   
    
    
    '''
    # plot hists
    hist, bin_edges = np.histogram(aa, bins=30)
    hist = hist/float(np.sum(hist))
    fig = plt.figure(figsize=(4,2))
    plt.bar(bin_edges[0:-1], hist, .01, color='b')
    
    hist, bin_edges = np.histogram(bb, bins=30)
    hist = hist/float(np.sum(hist))
    plt.bar(bin_edges[0:-1], hist, .01, color='r')
    '''

 

    '''
    ########################################################################
    ###### p values of difference of dists (diff (exc - inh)) relative to null dists you created above
    # percent non-0 w 
    aa = perActiveEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    bb = randAll_perActiveAll_ei_aveS[iday,:,:].flatten()
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\n%%non-0: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))
    

    # w
    aa = wEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_w_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nw: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))


    # non-0 w
    aa = wEmI_non0[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_w_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nnon0 w: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))
    

    # abs w
    aa = wabsEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nabs w: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))


    # abs non-0 w
    aa = wabsEmI_non0[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nabs non-0 w: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))
    '''


#%% For each day compare p value for exc - inh vs 0 with p value for exc - inh vs random exc-exc (inh-inh)    
"""
for iday in range(len(days)):    
    
    print '____________________', days[iday], '____________________'
    
    aa = perActiveEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
#    _, pdiff = stats.ttest_1samp(aa, 1, nan_policy='omit')
    _, pdiff = stats.ttest_1samp(aa, 0, nan_policy='omit')
    
    bb = randAll_perActiveAll_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb, nan_policy='omit')
    print '%.3f, %.3f' %(pdiff, p_ei)
    '''    
    bb = randAll_perActiveAll_exc_aveS[iday,:,:].flatten() # exc - exc for rand shuffles (not averaged across shuffles)    
    _, p_e = stats.ttest_ind(aa, bb)
    bb = randAll_perActiveAll_inh_aveS[iday,:,:].flatten()
    _, p_i = stats.ttest_ind(aa, bb)    
    print '%.3f, %.3f, %.3f, %.3f' %(pdiff, p_ei, p_e, p_i)
    '''
    

    aa = wabsEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
#    _, pdiff = stats.ttest_1samp(aa, 1, nan_policy='omit')
    _, pdiff = stats.ttest_1samp(aa, 0, nan_policy='omit')
    
    bb = randAll_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb, nan_policy='omit')
    print '%.3f, %.3f' %(pdiff, p_ei)
    
#    print '_____'
"""


#%% Average across days ... not sure if this is valid bc the relatin of weights vs c (or training error) is not the same for different days and by averaging we will cancel out any difference between exc and inh populations.

def plotav(a,b):
    ave = np.mean(a, axis=0) # length(cvect_)
    sde = np.std(a, axis=0)
    avi = np.mean(b, axis=0)
    sdi = np.std(b, axis=0)
    
    plt.errorbar(cvect_all[0,:], ave, sde, fmt='b.-')
    plt.errorbar(cvect_all[0,:], avi, sdi, fmt='r.-')
    plt.xscale('log')
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    plt.plot([np.mean(cbestAll), np.mean(cbestAll)],[ymin,ymax])    
    
    
    
##%% Average across all days (perc class error for train and test)
aa = perClassErAll_aveS
aat = perClassErTestAll_aveS

avr = np.mean(aa, axis=0) # length(cvect_)
sdr = np.std(aa, axis=0)
avt = np.mean(aat, axis=0) # length(cvect_)
sdt = np.std(aat, axis=0)

plt.figure(figsize=(3,15))
ax = plt.subplot(611)    
plt.errorbar(cvect_all[0,:], avr, sdr, fmt='k.-', label='train')
plt.errorbar(cvect_all[0,:], avt, sdt, fmt='g.-', label='test')
plt.xscale('log')
plt.xlabel('c')
plt.ylabel('class error')
plt.legend(loc=0, frameon=False)
ymin, ymax = ax.get_ylim()
plt.plot([np.mean(cbestAll), np.mean(cbestAll)],[ymin,ymax])    

    
plt.subplot(612)    
a = perActiveAll_exc_aveS
b = perActiveAll_inh_aveS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('% non0 w')
    
plt.subplot(613)        
a = wall_exc_aveNS # wall_non0_abs_exc_aveNS
b = wall_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('w')

plt.subplot(614)    
a = wall_non0_exc_aveNS
b = wall_non0_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('non0 w')

plt.subplot(615)    
a = wall_abs_exc_aveNS
b = wall_abs_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('abs w')

plt.subplot(616)    
a = wall_non0_abs_exc_aveNS
b = wall_non0_abs_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('abs non0 w')


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/allC'+dp, suffn[0:5]+'excInh_aveAllDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



#%% Plot average of fractnon0 and weights against class error (instad of c)

# is averaging right... remember u had to get diff, or perhaps do ratio...
# look at abs of weights and ave of non-zeros w


# u're going to run the code w the same cvect so u can easily plot all days against cvect... but u should be fine to plot it against classError
# work on this: plot ave and avi against classError

#np.min(perClassErAll_aveS)
step = 5
b = np.arange(0, np.ceil(np.nanmax(perClassErAll_aveS)).astype(int)+step, step)
b = np.arange(0, np.ceil(np.nanmax(perClassErTestAll_aveS)).astype(int)+step, step)

perActiveAll_exc_cerr = np.full((len(days), len(b)), np.nan)
perActiveAll_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_abs_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_abs_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_abs_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_abs_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_inh_cerr = np.full((len(days), len(b)), np.nan)

for iday in range(len(days)):
    
    inds = np.digitize(perClassErAll_aveS[iday,:], bins=b)
    inds = np.digitize(perClassErTestAll_aveS[iday,:], bins=b)

    for i in range(len(b)): #b[-1]

       perActiveAll_exc_cerr[iday,i] = np.mean(perActiveAll_exc_aveS[iday, inds==i+1])
       perActiveAll_inh_cerr[iday,i] = np.mean(perActiveAll_inh_aveS[iday, inds==i+1])
       
       wall_abs_exc_cerr[iday,i] = np.mean(wall_abs_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_abs_inh_cerr[iday,i] = np.mean(wall_abs_inh_aveNS[iday, inds==i+1])

       wall_non0_abs_exc_cerr[iday,i] = np.mean(wall_non0_abs_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_non0_abs_inh_cerr[iday,i] = np.mean(wall_non0_abs_inh_aveNS[iday, inds==i+1])

       wall_exc_cerr[iday,i] = np.mean(wall_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_inh_cerr[iday,i] = np.mean(wall_inh_aveNS[iday, inds==i+1])

       wall_non0_exc_cerr[iday,i] = np.mean(wall_non0_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_non0_inh_cerr[iday,i] = np.mean(wall_non0_inh_aveNS[iday, inds==i+1])


#%%       
#plt.plot(b, wall_exc_cerr[iday,:],'o')
plt.figure(figsize=(3,10))

plt.subplot(511)
plt.errorbar(b, np.nanmean(perActiveAll_exc_cerr, axis=0), yerr=np.nanstd(perActiveAll_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(perActiveAll_inh_cerr, axis=0), yerr=np.nanstd(perActiveAll_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('% non 0')

plt.subplot(512)
plt.errorbar(b, np.nanmean(wall_non0_abs_exc_cerr, axis=0), yerr=np.nanstd(wall_non0_abs_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_non0_abs_inh_cerr, axis=0), yerr=np.nanstd(wall_non0_abs_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('non0 abs w')

plt.subplot(513)
plt.errorbar(b, np.nanmean(wall_abs_exc_cerr, axis=0), yerr=np.nanstd(wall_abs_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_abs_inh_cerr, axis=0), yerr=np.nanstd(wall_abs_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('abs w')

plt.subplot(514)
plt.errorbar(b, np.nanmean(wall_exc_cerr, axis=0), yerr=np.nanstd(wall_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_inh_cerr, axis=0), yerr=np.nanstd(wall_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('w')

plt.subplot(515)
plt.errorbar(b, np.nanmean(wall_non0_exc_cerr, axis=0), yerr=np.nanstd(wall_non0_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_non0_inh_cerr, axis=0), yerr=np.nanstd(wall_non0_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('non0 w')


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow+'/allC'+dp, suffn[0:5]+'excInh_vsClassErr_aveAllDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')

