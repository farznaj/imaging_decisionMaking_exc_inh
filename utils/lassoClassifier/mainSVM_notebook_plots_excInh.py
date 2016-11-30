# -*- coding: utf-8 -*-
"""
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


#%%
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

#%%
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
            svmn = 'excInhC_svmPrevChoice_%sN_%sITIs_ep*-*ms_*' %(ntName, itiName)
        else:
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
    else:
        if np.isnan(roundi): # set excInh name
            svmn = 'excInhC_svmCurrChoice_%sN_ep%d-%dms_*' %(ntName, ep_ms[0], ep_ms[-1])   
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
    Data = scio.loadmat(svmName, variable_names=['perActive_exc', 'perActive_inh', 'wei_all', 'perClassEr', 'cvect_'])
    perActive_exc = Data.pop('perActive_exc') # numSamples x length(cvect_)
    perActive_inh = Data.pop('perActive_inh') # numSamples x length(cvect_)        
    wei_all = Data.pop('wei_all') # numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  
    perClassEr = Data.pop('perClassEr')
    cvect_ = Data.pop('cvect_').flatten()
    
    if abs(wei_all.sum()) < eps:
        print '\tWeights of all c and all shuffles are 0 ... not analyzing'
        
    else:            
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
            
        wall_exc.append(wei_all[:,:,inhRois_ei==0]) # numDays (each day: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall_inh.append(wei_all[:,:,inhRois_ei==1])        
        
        
        perActiveAll_exc.append(perActive_exc) # numDays x numSamples x length(cvect_)        
        perActiveAll_inh.append(perActive_inh)        
    
        perClassErAll.append(perClassEr) # numDays x numSamples x length(cvect_) 
        cvect_all.append(cvect_) # numDays x length(cvect_)
   
   
#%%    
perActiveAll_exc = np.array(perActiveAll_exc)  # days x numSamples x length(cvect_) # percent non-zero weights
perActiveAll_inh = np.array(perActiveAll_inh)  # days x numSamples x length(cvect_) # percent non-zero weights
perClassErAll = np.array(perClassErAll)  # days x numSamples x length(cvect_) # classification error
cvect_all = np.array(cvect_all) # days x length(cvect_)  # c values


#%% For each day average weights across neurons
wall_exc_aveN = []
wall_inh_aveN = []
wall_abs_exc_aveN = [] # average of abs values of weights
wall_abs_inh_aveN = []
wall_non0_abs_exc_aveN = [] # average of abs values of weights
wall_non0_abs_inh_aveN = []
for iday in range(len(days)):
    wall_exc_aveN.append(np.mean(wall_exc[iday], axis=2)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_inh_aveN.append(np.mean(wall_inh[iday], axis=2)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_abs_exc_aveN.append(np.mean(abs(wall_exc[iday]), axis=2)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_abs_inh_aveN.append(np.mean(abs(wall_inh[iday]), axis=2)) # days x numSamples x length(cvect_)  # ave w across neurons
    a = wall_exc[iday] 
    a[wall_exc[iday]==0] = np.nan    # only take non-0 weights
    wall_non0_abs_exc_aveN.append(np.nanmean(abs(a), axis=2)) # days x numSamples x length(cvect_)  # ave w across neurons
    a = wall_inh[iday] 
    a[wall_inh[iday]==0] = np.nan
    wall_non0_abs_inh_aveN.append(np.nanmean(abs(a), axis=2)) # days x numSamples x length(cvect_)  # ave w across neurons
    
wall_exc_aveN = np.array(wall_exc_aveN)
wall_inh_aveN = np.array(wall_inh_aveN)
wall_abs_exc_aveN = np.array(wall_abs_exc_aveN)
wall_abs_inh_aveN = np.array(wall_abs_inh_aveN)
wall_non0_abs_exc_aveN = np.array(wall_non0_abs_exc_aveN)
wall_non0_abs_inh_aveN = np.array(wall_non0_abs_inh_aveN)

#%% Average weights,etc across shuffles for each day : numDays x length(cvect_) 
wall_exc_aveNS = np.array([np.mean(wall_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_inh_aveNS = np.array([np.mean(wall_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_abs_exc_aveNS = np.array([np.mean(wall_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_abs_inh_aveNS = np.array([np.mean(wall_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_abs_exc_aveNS = np.array([np.nanmean(wall_non0_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_abs_inh_aveNS = np.array([np.nanmean(wall_non0_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
perClassErAll_aveS = np.array([np.mean(perClassErAll[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_exc_aveS = np.array([np.mean(perActiveAll_exc[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_inh_aveS = np.array([np.mean(perActiveAll_inh[iday,:,:], axis=0) for iday in range(len(days))])

# std
wall_exc_stdNS = np.array([np.std(wall_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # std of w across shuffles for each day
wall_inh_stdNS = np.array([np.std(wall_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_abs_exc_stdNS = np.array([np.std(wall_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # std of w across shuffles for each day
wall_abs_inh_stdNS = np.array([np.std(wall_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
perClassErAll_stdS = np.array([np.std(perClassErAll[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_exc_stdS = np.array([np.std(perActiveAll_exc[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_inh_stdS = np.array([np.std(perActiveAll_inh[iday,:,:], axis=0) for iday in range(len(days))])


#%% same analysis as before: for each day compute exc-inh for shuffle-averaged perc non-zero and weights
perActiveEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_non0 = np.full((perActiveAll_exc_aveS.shape), np.nan)
perActiveEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_non0_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)

for iday in range(len(days)):
    perActiveEmI[iday,:] = perActiveAll_exc_aveS[iday,:] - perActiveAll_inh_aveS[iday,:] # days x length(cvect_) # at each c value, divide %non-zero (averaged across shuffles) of exc and inh  
    wabsEmI[iday,:] = wall_abs_exc_aveNS[iday,:] - wall_abs_inh_aveNS[iday,:]
    wabsEmI_non0[iday,:] = wall_non0_abs_exc_aveNS[iday,:] - wall_non0_abs_inh_aveNS[iday,:]
#    plt.plot(perActiveEmI[iday,:])

    # exclude c values for which both exc and inh %non-0 are 0 or 100.
    a = perActiveAll_exc_aveS[iday,:]
    b = perActiveAll_inh_aveS[iday,:]    
    i0 = np.logical_and(a==0, b==0)
    i100 = np.logical_and(a==100, b==100)
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

randAll_perActiveAll_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # days x numSamples x length(cvect_) # exc - exc
randAll_perActiveAll_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_perActiveAll_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_wabs_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_wabs_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_wabs_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_wabs_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_wabs_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_wabs_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh

for iday in range(len(days)):
    # perc non-zero
    # exc-exc
    inds = rng.permutation(l) # random indeces of shuffles
    inds1 = rng.permutation(l)
    a = perActiveAll_exc[iday,inds1,:] - perActiveAll_exc[iday,inds,:]
    randAll_perActiveAll_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = perActiveAll_exc[iday,inds1,:]
    bb = perActiveAll_exc[iday,inds,:]
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
    aa = perActiveAll_inh[iday,inds1i,:]
    bb = perActiveAll_inh[iday,indsi,:]
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
    
    
    
    ### abs w
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
    
    
    
    ### abs non-0 w
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

#%% percent non-zero w: hist and P val for all days pooled (exc - inh)
a = np.reshape(perActiveEmI_b0100nan,(-1,)) # pool across days x length(cvect_)  
a = a[~(np.isnan(a) + np.isinf(a))]
# against 0
_, pdiff = stats.ttest_1samp(a, 1, nan_policy='omit')
print 'p value:diff c path (all c) = %.3f' %(pdiff)
print '\tmean = %.2f' %(np.mean(a))

# against random null dist
b = np.reshape(randAll_perActiveAll_ei_aveS_b0100nan,(-1,)) # pool across days x numSamples x length(cvect_)
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean = %.2f' %(np.mean(b))


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

plt.figure()    
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
plt.xlabel('%non-zero w')
plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.2f, p=%.3f' %(np.mean(a), p))
plt.xlim([-20,20])
#plt.savefig('cpath_all_percnon0.svg', format='svg', dpi=300)


#%% abs w: hist and P val for all days pooled (exc - inh)
a = np.reshape(wabsEmI_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against 0
_, pdiff = stats.ttest_1samp(a, 1, nan_policy='omit')
print 'p value:diff c path (all c) = %.3f' %(pdiff)
print '\tmean = %.2f' %(np.mean(a))

# against random null dist
b = np.reshape(randAll_wabs_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean = %.2f' %(np.mean(b))

binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.figure()    
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
plt.xlabel('abs w')
plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.2f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)


#%% abs non-0 w: hist and P val for all days pooled (exc - inh)
a = np.reshape(wabsEmI_non0,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against 0
_, pdiff = stats.ttest_1samp(a, 1, nan_policy='omit')
print 'p value:diff c path (all c) = %.3f' %(pdiff)
print '\tmean = %.2f' %(np.mean(a))

# against random null dist
b = np.reshape(randAll_non0_wabs_ei_aveS,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean = %.2f' %(np.mean(b))

binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.figure()    
plt.bar(bin_edges[0:-1], hist, binEvery, color='r', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend(loc=0)
plt.xlabel('abs w')
plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.2f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)


#%% Look at days one by one (averages across shuffles)
for iday in range(len(days)):
    
    print '____________________', days[iday], '____________________'
    
    plt.figure(figsize=(12,4))
    '''    
    plt.subplot(121)
    plt.errorbar(perClassErAll_aveS[iday,:], wall_exc_aveNS[iday,:], wall_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(perClassErAll_aveS[iday,:], wall_inh_aveNS[iday,:], wall_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.xscale('log')    
    plt.xlabel('Training error %')
    plt.ylabel('Mean+std of weights')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1,1))
    '''
    plt.subplot(131)
    plt.errorbar(cvect_all[iday,:], perActiveAll_exc_aveS[iday,:], perActiveAll_exc_stdS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], perActiveAll_inh_aveS[iday,:], perActiveAll_inh_stdS[iday,:], color = 'r', label = 'inhibit')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylabel('Fraction non-0 weights')
    
    plt.subplot(132)
    plt.errorbar(cvect_all[iday,:], wall_abs_exc_aveNS[iday,:], wall_abs_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_abs_inh_aveNS[iday,:], wall_abs_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylabel('Mean+std of abs weights')
    
    plt.subplot(133)
    plt.errorbar(cvect_all[iday,:], wall_exc_aveNS[iday,:], wall_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_inh_aveNS[iday,:], wall_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylabel('Mean+std of weights')
   
    plt.subplots_adjust(hspace=.5, wspace=.25)      
#    raw_input()
#    plt.savefig('cpath'+days[iday]+'.svg', format='svg', dpi=300)
    plt.show()       
   
    # percent non-zero
    aa = perActiveAll_exc_aveS[iday,:]
    bb = perActiveAll_inh_aveS[iday,:]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\n%%non-zero p value (pooled for all values of c):\nexc ~= inh : %.2f\nexc < inh : %.2f\nexc > inh : %.2f' %(p_two, p_tl, p_tr)
    
    # plot hists 
    '''
    hist, bin_edges = np.histogram(aa, bins=30)
    hist = hist/float(np.sum(hist))
    fig = plt.figure(figsize=(4,2))
    plt.bar(bin_edges[0:-1], hist, .7, color='b')
    
    hist, bin_edges = np.histogram(bb, bins=30)
    hist = hist/float(np.sum(hist))
    plt.bar(bin_edges[0:-1], hist, .7, color='r')
    '''
    
    # abs weight
    aa = wall_abs_exc_aveNS[iday,:]
    bb = wall_abs_inh_aveNS[iday,:]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nabs w: p value (pooled for all values of c):\nexc ~= inh : %.2f\nexc < inh : %.2f\nexc > inh : %.2f' %(p_two, p_tl, p_tr)
    
    # plot hists
    '''
    hist, bin_edges = np.histogram(aa, bins=30)
    hist = hist/float(np.sum(hist))
    fig = plt.figure(figsize=(4,2))
    plt.bar(bin_edges[0:-1], hist, .01, color='b')
    
    hist, bin_edges = np.histogram(bb, bins=30)
    hist = hist/float(np.sum(hist))
    plt.bar(bin_edges[0:-1], hist, .01, color='r')
'''

    # diff (exc - inh)
    _, pdiff = stats.ttest_1samp(perActiveEmI[iday,:], 0)
    print '\n%%non-0: p value:diff c path (all c) = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f' %(np.mean(perActiveEmI[iday,:]))
    
    aa = perActiveEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
#    _, pdiff = stats.ttest_1samp(aa, 0)
    bb = randAll_perActiveAll_ei_aveS[iday,:,:].flatten()
    _, p_ei = stats.ttest_ind(aa, bb)
    print 'rand null dist: p value:diff c path (all c) = %.3f' %(p_ei)
    print '\tmean of diff of c path= %.2f' %(np.mean(bb))
    

    aa = wabsEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
#    _, pdiff = stats.ttest_1samp(aa, 1, nan_policy='omit')
    _, pdiff = stats.ttest_1samp(aa, 0, nan_policy='omit')
    print '\nabs w: p value:diff c path (all c) = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f' %(np.mean(aa))
    
    bb = randAll_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb, nan_policy='omit')
    print 'rand null dist: p value:diff c path (all c) = %.3f' %(p_ei)
    print '\tmean of diff of c path= %.2f' %(np.mean(bb))    


    aa = wabsEmI_non0[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
#    _, pdiff = stats.ttest_1samp(aa, 1, nan_policy='omit')
    _, pdiff = stats.ttest_1samp(aa, 0, nan_policy='omit')
    print '\nabs non0 w: p value:diff c path (all c) = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f' %(np.mean(aa))
    
    bb = randAll_non0_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb, nan_policy='omit')
    print 'rand null dist: p value:diff c path (all c) = %.3f' %(p_ei)
    print '\tmean of diff of c path= %.2f' %(np.mean(bb))    
    
    
#%% Average across days ... not sure if this is valid bc the relatin of weights vs c (or training error) is not the same for different days and by averaging we will cancel out any difference between exc and inh populations.
ave = np.mean(wall_exc_aveNS, axis=0) # length(cvect_)
sde = np.std(wall_exc_aveNS, axis=0)

avi = np.mean(wall_inh_aveNS, axis=0)
sdi = np.std(wall_inh_aveNS, axis=0)

#%% u're going to run the code w the same cvect so u can easily plot all days against cvect... but u should be fine to plot it against classError
# work on this: plot ave and avi against classError

#np.min(perClassErAll_aveS)
step = 5
b = np.arange(0, np.ceil(np.max(perClassErAll_aveS)).astype(int)+step, step)

wall_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_inh_cerr = np.full((len(days), len(b)), np.nan)
perActiveAll_exc_cerr = np.full((len(days), len(b)), np.nan)
perActiveAll_inh_cerr = np.full((len(days), len(b)), np.nan)
for iday in range(len(days)):
    inds = np.digitize(perClassErAll_aveS[iday,:], bins=b)
    for i in range(len(b)): #b[-1]
       wall_exc_cerr[iday,i] = np.mean(wall_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_inh_cerr[iday,i] = np.mean(wall_inh_aveNS[iday, inds==i+1])
       perActiveAll_exc_cerr[iday,i] = np.mean(perActiveAll_exc_aveS[iday, inds==i+1])
       perActiveAll_inh_cerr[iday,i] = np.mean(perActiveAll_inh_aveS[iday, inds==i+1])

#%%       
#plt.plot(b, wall_exc_cerr[iday,:],'o')
plt.figure()
plt.errorbar(b, np.nanmean(wall_exc_cerr, axis=0), yerr=np.nanstd(wall_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_inh_cerr, axis=0), yerr=np.nanstd(wall_inh_cerr, axis=0), fmt='r.-')

plt.figure()
plt.plot(b, np.nanmean(perActiveAll_exc_cerr, axis=0), 'b.-')
plt.plot(b, np.nanmean(perActiveAll_inh_cerr, axis=0), 'r.-')

# is averaging right... remember u had to get diff, or perhaps do ratio...
# look at abs of weights and ave of non-zeros w

#%% show c path diff of each day 
'''
m = [np.mean(excInhDiffAllC_alld[i]) for i in range(numDays)]    
pdiffall = np.full((1,numDays), np.nan).squeeze()
for i in range(numDays):
    _, pdiffall[i] = stats.ttest_1samp(excInhDiffAllC_alld[i], 0)
    
plt.figure()
plt.plot([0,numDays], [0,0])
for i in range(numDays):
    plt.errorbar(i, np.mean(excInhDiffAllC_alld[i]), np.std(excInhDiffAllC_alld[i]), marker='o', fmt=' ', color='k')
    if pdiffall[i]<=.05:
        mmax = 6
        plt.plot(i,mmax, marker='*', color='r')
plt.xlabel('days')    
plt.ylabel('mean,std of excInhDiffAllC')

#fign_ = suffnei+'cPath_diff_alldays'
#fign = os.path.join(svmdir, fign_+'.'+fmt[0])            
#plt.savefig(fign)#, bbox_extra_artists=(lgd,), bbox_inches='tight')    
    
    
# average of each day    
mm=np.array(m)
msig = mm[np.array(pdiffall<.05)]
#mnow = m
mnow = msig
hist, bin_edges = np.histogram(mnow, bins=10)
hist = hist/float(np.sum(hist))

fig = plt.figure(figsize=(4,2))
#ax = plt.subplot(gs[1,:])
plt.bar(bin_edges[0:-1], hist, .7, color='k')
#    plt.title('p value: = %.3f\nmean = %.3f' %(pdiff, np.mean(mnow)), y=.8)
plt.xlabel('% Non-zero weights (exc - inh), all c values', fontsize=13)
plt.ylabel('Normalized count', fontsize=13)
plt.ylim([-.03, 1])

ax = plt.gca()
makeNicePlots(ax) 
'''



#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        excInhDiffAllC0 = excInhDiffAllC        

    elif iTiFlg==1:
        excInhDiffAllC1 = excInhDiffAllC
        
        
#%% Is exc - inh c-path significantly different from 0?

_, pdiff = stats.ttest_1samp(excInhDiffAllC, 0)
print 'p value:diff c path (all c) = %.3f' %(pdiff)
print '\tmean of diff of c path= %.2f' %(np.mean(excInhDiffAllC))


#%% Plot hist and erro bar for exc-inh c-path for all rounds of all days and all values of c

hist, bin_edges = np.histogram(excInhDiffAllC, bins=50)
hist = hist/float(np.sum(hist))

fig = plt.figure(figsize=(4,2))
#ax = plt.subplot(gs[1,:])
plt.bar(bin_edges[0:-1], hist, .7, color='k')
plt.title('p value: = %.3f\nmean = %.2f' %(pdiff, np.mean(excInhDiffAllC)), y=.8)
plt.xlabel('% Non-zero weights (exc - inh), all c values', fontsize=13)
plt.ylabel('Normalized count', fontsize=13)
plt.ylim([-.03, 1])

ax = plt.gca()
makeNicePlots(ax)

'''
excInhDiffAveC = np.mean(perActive_excInhDiff_rounds, axis=1) # mean across values of c  # (numDays x numValidRounds) x 1
_, pdiffave = stats.ttest_1samp(excInhDiffAveC, 0)
print 'p value:diff c path (mean along c) = %.3f' %(pdiffave)


gs = gridspec.GridSpec(2, 3) #, width_ratios=[3, 1]) 

hist, bin_edges = np.histogram(excInhDiffAveC, bins=50)
hist = hist/float(np.sum(hist))

fig = plt.figure()
ax = plt.subplot(gs[0,:-1]) #plt.subplot(gs[0]) #plt.subplot2grid((1,2), (0,0), colspan=2)
ax.bar(bin_edges[0:-1]+0, hist, .1, color='k')
plt.xlabel('c-path (exc - inh)')
plt.ylabel('Normalized count')
plt.title('p value:diff c path (mean along c) = %.3f' %(pdiffave), y=1.08)

makeNicePlots(ax)

ax = plt.subplot(gs[0,2]) # plt.subplot(gs[1]) #plt.subplot2grid((1,2), (0,1), colspan=1)
ax.errorbar(1, np.mean(excInhDiffAveC), np.std(excInhDiffAveC), marker='o', fmt=' ', color='k')
#plt.locator_params(axis='x',nbins=1)
plt.xticks([1]) # [mean']
makeNicePlots(ax)
'''
#plt.subplots_adjust(wspace=.6, hspace=.8) #, top = 1)


#%% Save the figure

if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffnei+'cPath_hist_d%d' %(iday)
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign)#, bbox_extra_artists=(lgd,), bbox_inches='tight')
 
 

 
#%% Not useful because of averaging across days: 

# Average and std across days

# average exc and inh c-path across days
perActive_exc_ave_allDays_ave = np.mean(perActive_exc_ave_allDays, axis=0) # perActive_exc_ave_allDays: numDays x length(cvect_)
perActive_exc_ave_allDays_std = np.std(perActive_exc_ave_allDays, axis=0)

perActive_inh_ave_allDays_ave = np.mean(perActive_inh_ave_allDays, axis=0)
perActive_inh_ave_allDays_std = np.std(perActive_inh_ave_allDays, axis=0)


# average exc-inh c-path for all rounds and all days
perActive_excInhDiff_ave = np.mean(perActive_excInhDiff_rounds, axis=0) # length(cvect_) 
perActive_excInhDiff_std = np.std(perActive_excInhDiff_rounds, axis=0)

       
#%% Not useful due to averaging across days
# Compute p value for exc vs inh c path (pooled across all c values) after averaging across days 

aa = np.reshape(perActive_exc_ave_allDays,(-1,))
bb = np.reshape(perActive_inh_ave_allDays,(-1,))
#aa = np.array(perActive_exc).flatten()
#bb = np.array(perActive_inh).flatten()
"""
aa = aa[np.logical_and(aa>0 , aa<100)]
np.shape(aa)

bb = bb[np.logical_and(bb>0 , bb<100)]
np.shape(bb)
"""

h, p_two = stats.ttest_ind(aa, bb)
p_tl = ttest2(aa, bb, tail='left')
p_tr = ttest2(aa, bb, tail='right')

print '\np value (exc vs inh; pooled for all values of c):\nexc ~= inh : %.2f\nexc < inh : %.2f\nexc > inh : %.2f' %(p_two, p_tl, p_tr)


#%% Not useful plots, because we have to average c-path across days or rounds and this masks any difference between exc and inh (which could be different from day to day)
# look at c-path plot


# at each value of c, is exc-inh significantly different from 0 using data from all rounds and all days
_, pdiffave0 = stats.ttest_1samp(perActive_excInhDiff_rounds, 0) # length(cvect_) # perActive_excInhDiff_rounds: (numRounds x numDays) x length(cvect_)
#print 'p value:diff c path (mean along c) = %.3f' %(pdiffave0)

# Load cvect_
Data = scio.loadmat(svmName, variable_names=['cvect_'])
cvect_ = Data.pop('cvect_')[0,:] 


plt.figure(figsize=(4,5))

# average exc-inh for all rounds and all days
plt.subplot(211)
plt.fill_between(cvect_, perActive_excInhDiff_ave - perActive_excInhDiff_std, perActive_excInhDiff_ave + perActive_excInhDiff_std, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(cvect_, perActive_excInhDiff_ave, 'k')
ax = plt.gca()
ymin, ymax = ax.get_ylim()
pp = pdiffave0+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(cvect_, pp, color='k')
#plt.ylim([-10,110])
plt.xscale('log')
plt.xlabel('c (inverse of regularization parameter)')
plt.ylabel('% non-zero weights (exc - inh)')
#plt.title('p value (pooled for all values of c):\nexc ~= inh : %.2f; exc < inh : %.2f; exc > inh : %.2f' %(p_two, p_tl, p_tr))

ax = plt.gca()
makeNicePlots(ax)


##%% Plot the average c path across all days for exc and inh
#    plt.subplot(212)
plt.figure(figsize=(4,3))
plt.fill_between(1/cvect_, perActive_exc_ave_allDays_ave - perActive_exc_ave_allDays_std, perActive_exc_ave_allDays_ave + perActive_exc_ave_allDays_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(1/cvect_, perActive_exc_ave_allDays_ave, 'b', label = 'Excitatory')

plt.fill_between(1/cvect_, perActive_inh_ave_allDays_ave - perActive_inh_ave_allDays_std, perActive_inh_ave_allDays_ave + perActive_inh_ave_allDays_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(1/cvect_, perActive_inh_ave_allDays_ave, 'r', label = 'Inhibitory')

plt.ylim([-10,110])
plt.xscale('log')
plt.xlabel('Regularization parameter', fontsize=18)
plt.ylabel('% Neurons in the decoder', fontsize=18)
lgd = plt.legend()
plt.legend(loc='best', frameon=False) #, bbox_to_anchor=(.4, .7))
#    plt.title('p value (pooled for all values of c):\nexc ~= inh : %.2f; exc < inh : %.2f; exc > inh : %.2f' %(p_two, p_tl, p_tr))
plt.xlim([.5, 200])
ax = plt.gca()
makeNicePlots(ax)

plt.subplots_adjust(hspace=1.2)
 

 
#%% Save the figure

if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffnei+'cPath_d%d' %(iday)
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
            
            
            
            
            

#%% Toy example 

plt.figure(figsize=(4,3))
#plt.subplot(212)
#plt.fill_between(cvect_, perActive_exc_ave_allDays_ave - perActive_exc_ave_allDays_std, perActive_exc_ave_allDays_ave + perActive_exc_ave_allDays_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(1/cvect_, perActive_exc_ave_allDays_ave, 'k', label = 'A')

#plt.fill_between(cvect_, perActive_inh_ave_allDays_ave - perActive_inh_ave_allDays_std, perActive_inh_ave_allDays_ave + perActive_inh_ave_allDays_std, alpha=0.5, edgecolor='r', facecolor='r')
#plt.plot(cvect_, perActive_inh_ave_allDays_ave, 'r', label = 'inhibit')
#a=perActive_exc_ave_allDays_ave[1:-1]
#b=(np.ones((1,len(cvect_)-len(a)))*100).squeeze()
#c = np.concatenate((a,b))

a2=perActive_exc_ave_allDays_ave[0:-5]
b2=(np.zeros((1,len(cvect_)-len(a2)))).squeeze()
c2 = np.concatenate((b2,a2))

#plt.plot(1/cvect_, c, 'k', label = 'A')
plt.plot(1/cvect_, c2, 'k', label = 'B')

plt.xlim([.1, 200])
plt.ylim([-10,110])
plt.xscale('log')
plt.xlabel('Regularization parameter', fontsize=18)
plt.ylabel('% Neurons in the decoder', fontsize=18)
plt.legend()
plt.legend(loc='best', frameon=False) #, bbox_to_anchor=(.4, .7))

ax = plt.gca()
makeNicePlots(ax)

#%%
for i in range(np.shape(fmt)[0]): # save in all format of fmt         
    fign_ = suffnei+'cPath_toyEx'
    fign = os.path.join(svmdir, fign_+'.'+fmt[i])
    
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
        
        




 
'''
#####################################################################################################################################################    
################# Excitatory vs inhibitory neurons:  class error of testing dataset ###################################################
#####################################################################################################################################################
#####################################################################################################################################################   
'''  
 
 #%% Loop over days

perClassErrorTest_data_inh_ave_allDays = (np.ones((numRounds, numDays))+np.nan)
perClassErrorTest_shfl_inh_ave_allDays = (np.ones((numRounds, numDays))+np.nan)
perClassErrorTest_data_exc_ave_allDays = (np.ones((numRounds, numDays))+np.nan)
perClassErrorTest_shfl_exc_ave_allDays = (np.ones((numRounds, numDays))+np.nan)
perClassErrorTest_data_allExc_ave_allDays = (np.ones((numRounds, numDays))+np.nan)
perClassErrorTest_shfl_allExc_ave_allDays = (np.ones((numRounds, numDays))+np.nan)

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
    
    #%% loop over the 10 rounds of analysis for each day
    
    perClassErrorTest_data_inh_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    perClassErrorTest_shfl_inh_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    perClassErrorTest_data_exc_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    perClassErrorTest_shfl_exc_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    perClassErrorTest_data_allExc_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    perClassErrorTest_shfl_allExc_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    
    for i in range(numRounds):
        roundi = i+1
        svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName)

        if roundi==1:
            print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if abs(w.sum()) < eps:
            print '\tIn round %d all weights are 0 ... not analyzing' %(roundi)
            
        else:
        
            ##%% Load the class-loss array for the testing dataset (actual and shuffled) (length of array = number of samples, usually 100) 
            ##%% Load vars
            Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc'])
            perClassErrorTest_data_inh = Data.pop('perClassErrorTest_data_inh').squeeze() # numSamples 
            perClassErrorTest_shfl_inh = Data.pop('perClassErrorTest_shfl_inh').squeeze() 
            perClassErrorTest_data_exc = Data.pop('perClassErrorTest_data_exc').squeeze()
            perClassErrorTest_shfl_exc = Data.pop('perClassErrorTest_shfl_exc').squeeze()
            perClassErrorTest_data_allExc = Data.pop('perClassErrorTest_data_allExc').squeeze()
            perClassErrorTest_shfl_allExc = Data.pop('perClassErrorTest_shfl_allExc').squeeze()
            
            # Compute average across samples for each round
            perClassErrorTest_data_inh_ave[i] = np.mean(perClassErrorTest_data_inh) # numRounds
            perClassErrorTest_shfl_inh_ave[i] = np.mean(perClassErrorTest_shfl_inh)
            perClassErrorTest_data_exc_ave[i] = np.mean(perClassErrorTest_data_exc) 
            perClassErrorTest_shfl_exc_ave[i] = np.mean(perClassErrorTest_shfl_exc)
            perClassErrorTest_data_allExc_ave[i] = np.mean(perClassErrorTest_data_allExc) 
            perClassErrorTest_shfl_allExc_ave[i] = np.mean(perClassErrorTest_shfl_allExc)
            
    # Keep results of all rounds for each day
    perClassErrorTest_data_inh_ave_allDays[:, iday] = perClassErrorTest_data_inh_ave  # rounds x days
    perClassErrorTest_shfl_inh_ave_allDays[:, iday] = perClassErrorTest_shfl_inh_ave
    perClassErrorTest_data_exc_ave_allDays[:, iday] = perClassErrorTest_data_exc_ave  # rounds x days
    perClassErrorTest_shfl_exc_ave_allDays[:, iday] = perClassErrorTest_shfl_exc_ave
    perClassErrorTest_data_allExc_ave_allDays[:, iday] = perClassErrorTest_data_allExc_ave  # rounds x days
    perClassErrorTest_shfl_allExc_ave_allDays[:, iday] = perClassErrorTest_shfl_allExc_ave
    

#%% Average and std across rounds for each day
ave_test_inh_d = np.nanmean(perClassErrorTest_data_inh_ave_allDays, axis=0) # numDays
ave_test_inh_s = np.nanmean(perClassErrorTest_shfl_inh_ave_allDays, axis=0) 
sd_test_inh_d = np.nanstd(perClassErrorTest_data_inh_ave_allDays, axis=0)
sd_test_inh_s = np.nanstd(perClassErrorTest_shfl_inh_ave_allDays, axis=0)

ave_test_exc_d = np.nanmean(perClassErrorTest_data_exc_ave_allDays, axis=0) # numDays
ave_test_exc_s = np.nanmean(perClassErrorTest_shfl_exc_ave_allDays, axis=0) 
sd_test_exc_d = np.nanstd(perClassErrorTest_data_exc_ave_allDays, axis=0)
sd_test_exc_s = np.nanstd(perClassErrorTest_shfl_exc_ave_allDays, axis=0)

ave_test_allExc_d = np.nanmean(perClassErrorTest_data_allExc_ave_allDays, axis=0) # numDays
ave_test_allExc_s = np.nanmean(perClassErrorTest_shfl_allExc_ave_allDays, axis=0) 
sd_test_allExc_d = np.nanstd(perClassErrorTest_data_allExc_ave_allDays, axis=0)
sd_test_allExc_s = np.nanstd(perClassErrorTest_shfl_allExc_ave_allDays, axis=0)
    
# same as nRoundsNonZeroW    
#[sum(~np.isnan(perClassErrorTest_data_inh_ave_allDays[:,i])) for i in range(numDays)]
#[sum(~np.isnan(perClassErrorTest_data_exc_ave_allDays[:,i])) for i in range(numDays)]


#%% keep short and long ITI results to plot them against each other later
if trialHistAnalysis:
    if iTiFlg==0:
        perClassErrorTest_data_inh_ave_allDays0 = perClassErrorTest_data_inh_ave_allDays
        perClassErrorTest_shfl_inh_ave_allDays0 = perClassErrorTest_shfl_inh_ave_allDays
        perClassErrorTest_data_exc_ave_allDays0 = perClassErrorTest_data_exc_ave_allDays
        perClassErrorTest_shfl_exc_ave_allDays0 = perClassErrorTest_shfl_exc_ave_allDays
        perClassErrorTest_data_allExc_ave_allDays0 = perClassErrorTest_data_allExc_ave_allDays
        perClassErrorTest_shfl_allExc_ave_allDays0 = perClassErrorTest_shfl_allExc_ave_allDays
        
    elif iTiFlg==1:
        perClassErrorTest_data_inh_ave_allDays1 = perClassErrorTest_data_inh_ave_allDays
        perClassErrorTest_shfl_inh_ave_allDays1 = perClassErrorTest_shfl_inh_ave_allDays
        perClassErrorTest_data_exc_ave_allDays1 = perClassErrorTest_data_exc_ave_allDays
        perClassErrorTest_shfl_exc_ave_allDays1 = perClassErrorTest_shfl_exc_ave_allDays
        perClassErrorTest_data_allExc_ave_allDays1 = perClassErrorTest_data_allExc_ave_allDays
        perClassErrorTest_shfl_allExc_ave_allDays1 = perClassErrorTest_shfl_allExc_ave_allDays
        
        
#%% ttest comparing exc vs inh
    
_, pinh = stats.ttest_ind(ave_test_inh_d, ave_test_inh_s)    
print 'pval_inh: %.3f' %(pinh)

_, pexc = stats.ttest_ind(ave_test_exc_d, ave_test_exc_s)    
print 'pval_exc: %.3f' %(pexc)

_, pexcinh = stats.ttest_ind(ave_test_exc_d, ave_test_inh_d)    
print 'pval_excinh: %.3f' %(pexcinh)

_, pallexcinh = stats.ttest_ind(ave_test_allExc_d, ave_test_inh_d)    
print 'pval_allexcinh: %.3f' %(pallexcinh)

# exc vs inh for each day
p = np.full((numDays,1), np.nan)
for i in range(numDays):
    _, p[i] = stats.ttest_ind(perClassErrorTest_data_inh_ave_allDays[:,i], perClassErrorTest_data_exc_ave_allDays[:,i], nan_policy='omit')

p, p<=palpha


#%% Plot class error for inh and exc neurons
    
xval = range(len(ave_test_inh_d))    
# Average across rounds for each day
plt.figure(figsize=(7,9.5))
gs = gridspec.GridSpec(4, 3) #, width_ratios=[3, 1])

# Inhibitory vs excitatory vs all excitatory neurons 
# each day
ax = plt.subplot(gs[0,:-1])  # ax = plt.subplot(221)
plt.errorbar(xval, ave_test_inh_d, yerr = sd_test_inh_d, color='r', label='inh')
plt.errorbar(xval, ave_test_exc_d, yerr = sd_test_exc_d, color='g', label='n exc')
plt.errorbar(xval, ave_test_allExc_d, yerr = sd_test_allExc_d, color='m', label='all exc', alpha=.5)
plt.xlabel('Days')
plt.ylabel('Class error(%)-testing data')
#plt.title('Inhibitory', y=1.05)
plt.xlim([-1, len(ave_test_inh_d)])
lgd = plt.legend(loc='upper center', bbox_to_anchor=(.9,1.7), frameon=False, fontsize = 'medium')
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax,0,1)
ymin, ymax = ax.get_ylim()
pp = p+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(xval, pp, 'k*')    
plt.ylim([ymin, ymax+5])


##%% Average across days
x = [0,1,2]
labels = ['Inh', 'Exc', 'AllExc']
ax = plt.subplot(gs[0,2]) #ax = plt.subplot(222)
plt.errorbar(x, [np.mean(ave_test_inh_d), np.mean(ave_test_exc_d), np.mean(ave_test_allExc_d)], yerr = [np.std(ave_test_inh_d), np.std(ave_test_exc_d), np.std(ave_test_allExc_d)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels)    
plt.title('p (exc vs inh): %.3f\np (allExc vs inh): %.3f' %(pexcinh,pallexcinh), y=1.08)

makeNicePlots(ax,0,1)





# Inhibitory neurons
# each day
ax = plt.subplot(gs[1,:-1])  # ax = plt.subplot(221)
plt.errorbar(xval, ave_test_inh_d, yerr = sd_test_inh_d, color='g', label='data')
plt.errorbar(xval, ave_test_inh_s, yerr = sd_test_inh_s, color='k', label='shuffled')
plt.xlabel('Days')
#plt.ylabel('Class error(%)-testing data')
plt.title('Inhibitory', y=1.05)
plt.xlim([-1, len(ave_test_inh_d)])
lgd = plt.legend(loc='upper center', bbox_to_anchor=(.9,1.7), frameon=False, fontsize = 'medium')
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax,0,1)
    

##%% Average across days
x =[0,1]
labels = ['data', 'shfl']
ax = plt.subplot(gs[1,2]) #ax = plt.subplot(222)
plt.errorbar(x, [np.mean(ave_test_inh_d), np.mean(ave_test_inh_s)], yerr = [np.std(ave_test_inh_d), np.std(ave_test_inh_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels)    
plt.title('pval_inh: %.3f' %(pinh), y=1.08)

makeNicePlots(ax,0,1)





# Excitatory neurons
# each day
ax = plt.subplot(gs[2,:-1])  # ax = plt.subplot(221)
plt.errorbar(xval, ave_test_exc_d, yerr = sd_test_exc_d, color='g', label='data')
plt.errorbar(xval, ave_test_exc_s, yerr = sd_test_exc_s, color='k', label='shuffled')
#plt.ylabel('Class error(%)-testing data')
plt.title('Excitatory', y=1.05)
plt.xlim([-1, len(ave_test_inh_d)])
#lgd = plt.legend(loc='upper center', bbox_to_anchor=(.8,1.4), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax,0,1)
    

##%% Average across days
x =[0,1]
labels = ['data', 'shfl']
ax = plt.subplot(gs[2,2]) #ax = plt.subplot(222)
plt.errorbar(x, [np.mean(ave_test_exc_d), np.mean(ave_test_exc_s)], yerr = [np.std(ave_test_exc_d), np.std(ave_test_exc_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels)    
plt.title('pval_exc: %.3f' %(pexc), y=1.08)

plt.subplots_adjust(wspace=.6, hspace=1.7)
makeNicePlots(ax,0,1)





# All excitatory neurons
# each day
ax = plt.subplot(gs[3,:-1])  # ax = plt.subplot(221)
plt.errorbar(xval, ave_test_allExc_d, yerr = sd_test_allExc_d, color='g', label='data')
plt.errorbar(xval, ave_test_allExc_s, yerr = sd_test_allExc_s, color='k', label='shuffled')
#plt.ylabel('Class error(%)-testing data')
plt.title('All excitatory', y=1.05)
plt.xlim([-1, len(ave_test_inh_d)])
#lgd = plt.legend(loc='upper center', bbox_to_anchor=(.8,1.4), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax,0,1)
    

##%% Average across days
x =[0,1]
labels = ['data', 'shfl']
ax = plt.subplot(gs[3,2]) #ax = plt.subplot(222)
plt.errorbar(x, [np.mean(ave_test_allExc_d), np.mean(ave_test_allExc_s)], yerr = [np.std(ave_test_allExc_d), np.std(ave_test_allExc_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels)    
plt.title('pval_exc: %.3f' %(pexc), y=1.08)

plt.subplots_adjust(wspace=.6, hspace=1.7)
makeNicePlots(ax,0,1)

        
#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffnei+'classError'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    






#%%
'''
########################################################################################################################################################       
#################### Excitatory vs inhibitory neurons: #################################################################################################       
#################### Classification accuracy at all times ##############################################################################################    
########################################################################################################################################################    
'''

#%% Loop over days

corrClass_exc_ave_allDays = []
corrClass_inh_ave_allDays = []
corrClass_allExc_ave_allDays = []
#eventI_allDays = np.full([numDays,1], np.nan).flatten().astype('int')

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
    
    #%% Load eventI (we need it for the final alignment of corrClass traces of all days)
    '''
    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
#        traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
#        time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')
    
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
#        traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
#        time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
        
    eventI_allDays[iday] = eventI        
    '''
        
    #%% loop over the 10 rounds of analysis for each day
    
    corrClass_exc_rounds = []
    corrClass_inh_rounds = []
    corrClass_allExc_rounds = []
    
    for i in range(numRounds):
        roundi = i+1
        svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName)
        
        if roundi==1:
            print os.path.basename(svmName)        


        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
     
        if abs(w.sum()) < eps:
            print '\tIn round %d all weights are 0 ... not analyzing' %(roundi)
            
        else:        
            ##%% Load corrClass
            Data = scio.loadmat(svmName, variable_names=['corrClass_exc', 'corrClass_inh', 'corrClass_allExc'])
            corrClass_exc = Data.pop('corrClass_exc') # frames x (nShfExc x trials)
            corrClass_inh = Data.pop('corrClass_inh') # frames x trials
            corrClass_allExc = Data.pop('corrClass_allExc')
            
            corrClass_exc_rounds.append(np.mean(corrClass_exc, axis=1)) # average across trials : numRounds x numFrames
            corrClass_inh_rounds.append(np.mean(corrClass_inh, axis=1))
            corrClass_allExc_rounds.append(np.mean(corrClass_allExc, axis=1))
            
    # Compute average across rounds for each day
    corrClass_exc_ave = np.mean(corrClass_exc_rounds, axis=0) # numFrames x 1
    corrClass_inh_ave = np.mean(corrClass_inh_rounds, axis=0)
    corrClass_allExc_ave = np.mean(corrClass_allExc_rounds, axis=0)
    
    
    #%% Pool corrClass_ave (average of corrClass across all rounds) of all days
    
    corrClass_exc_ave_allDays.append(corrClass_exc_ave)
    corrClass_inh_ave_allDays.append(corrClass_inh_ave)
    corrClass_allExc_ave_allDays.append(corrClass_allExc_ave)
    



#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(corrClass_ave_allDays[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

corrClass_exc_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
corrClass_inh_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
corrClass_allExc_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan

for iday in range(numDays):
    corrClass_exc_aligned[:, iday] = corrClass_exc_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    corrClass_inh_aligned[:, iday] = corrClass_inh_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    corrClass_allExc_aligned[:, iday] = corrClass_allExc_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]


#%% keep short and long ITI results to plot them against each other later
if trialHistAnalysis:
    if iTiFlg==0:
        corrClass_exc_aligned0 = corrClass_exc_aligned
        corrClass_inh_aligned0 = corrClass_inh_aligned
        corrClass_allExc_aligned0 = corrClass_allExc_aligned
        
    elif iTiFlg==1:
        corrClass_exc_aligned1 = corrClass_exc_aligned
        corrClass_inh_aligned1 = corrClass_inh_aligned
        corrClass_allExc_aligned1 = corrClass_allExc_aligned
        
        
#%% Average across days

corrClass_exc_aligned_ave = np.mean(corrClass_exc_aligned, axis=1) * 100
corrClass_exc_aligned_std = np.std(corrClass_exc_aligned, axis=1) * 100

corrClass_inh_aligned_ave = np.mean(corrClass_inh_aligned, axis=1) * 100
corrClass_inh_aligned_std = np.std(corrClass_inh_aligned, axis=1) * 100

corrClass_allExc_aligned_ave = np.mean(corrClass_allExc_aligned, axis=1) * 100
corrClass_allExc_aligned_std = np.std(corrClass_allExc_aligned, axis=1) * 100


_,pexc = stats.ttest_1samp(corrClass_exc_aligned.transpose(), 50) # p value of class accuracy being different from 50
_,pinh = stats.ttest_1samp(corrClass_inh_aligned.transpose(), 50) # p value of class accuracy being different from 50
_,pexcinh = stats.ttest_ind(corrClass_inh_aligned.transpose(), corrClass_exc_aligned.transpose()) # exc vs inh p value of class accuracy
_,pallexcinh = stats.ttest_ind(corrClass_inh_aligned.transpose(), corrClass_allExc_aligned.transpose()) # allExc vs inh p value of class accuracy


#%% Plot the average traces across all days
#ep_ms_allDays
plt.figure()

plt.fill_between(time_aligned, corrClass_exc_aligned_ave - corrClass_exc_aligned_std, corrClass_exc_aligned_ave + corrClass_exc_aligned_std, alpha=0.5, edgecolor='g', facecolor='g', label='N excitatory')
plt.plot(time_aligned, corrClass_exc_aligned_ave, 'g')

plt.fill_between(time_aligned, corrClass_inh_aligned_ave - corrClass_inh_aligned_std, corrClass_inh_aligned_ave + corrClass_inh_aligned_std, alpha=0.5, edgecolor='r', facecolor='r', label='Inhibitory')
plt.plot(time_aligned, corrClass_inh_aligned_ave, 'r')

plt.fill_between(time_aligned, corrClass_allExc_aligned_ave - corrClass_allExc_aligned_std, corrClass_allExc_aligned_ave + corrClass_allExc_aligned_std, alpha=0.5, edgecolor='m', facecolor='m', label='All excitatory')
plt.plot(time_aligned, corrClass_allExc_aligned_ave, 'm')

plt.xlabel('Time since stimulus onset (ms)')
plt.ylabel('Classification accuracy (%)')
lgd = plt.legend(loc='best', frameon=False)

ax = plt.gca()
makeNicePlots(ax)

# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()
pp = pexcinh+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

# Plot lines for the training epoch
win = np.mean(ep_ms_allDays, axis=0)
plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])


#%% Save the figure    
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffnei+'corrClassTrace'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')




