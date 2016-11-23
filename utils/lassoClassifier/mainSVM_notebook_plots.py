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
        svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
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
############################ Classification accuracy (testing data) ###################################################################################################     
#####################################################################################################################################################
'''

#%% Loop over days

err_test_data_ave_allDays = (np.ones((numRounds, numDays))+np.nan)
err_test_shfl_ave_allDays = (np.ones((numRounds, numDays))+np.nan)

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
    
    err_test_data_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    err_test_shfl_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    
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
            Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data', 'perClassErrorTest_shfl'])
            perClassErrorTest_data = Data.pop('perClassErrorTest_data')[0,:] # numSamples
            perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')[0,:]
            perClassErrorTest_shfl.shape
            
            # Average class error across samples for each round
            err_test_data_ave[i] = perClassErrorTest_data.mean() # numRounds
            err_test_shfl_ave[i] = perClassErrorTest_shfl.mean()   
    
    # pool results of all rounds for days
    err_test_data_ave_allDays[:, iday] = err_test_data_ave # rounds x days. #it will be nan for rounds with all-0 weights
    err_test_shfl_ave_allDays[:, iday] = err_test_shfl_ave
    
    
#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        err_test_data_ave_allDays0 = err_test_data_ave_allDays
        err_test_shfl_ave_allDays0 = err_test_shfl_ave_allDays

    elif iTiFlg==1:
        err_test_data_ave_allDays1 = err_test_data_ave_allDays
        err_test_shfl_ave_allDays1 = err_test_shfl_ave_allDays
        
    
#%% Average and std across rounds
ave_test_d = 100-np.nanmean(err_test_data_ave_allDays, axis=0) # numDays
ave_test_s = 100-np.nanmean(err_test_shfl_ave_allDays, axis=0) 
sd_test_d = np.nanstd(err_test_data_ave_allDays, axis=0) 
sd_test_s = np.nanstd(err_test_shfl_ave_allDays, axis=0) 


#%% Plot average across rounds for each day
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-2])
plt.errorbar(range(numDays), ave_test_d, yerr = sd_test_d, color='g', label='Data')
plt.errorbar(range(numDays), ave_test_s, yerr = sd_test_s, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()


##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[-2:-1])
plt.errorbar(x, [np.mean(ave_test_d), np.mean(ave_test_s)], yerr = [np.std(ave_test_d), np.std(ave_test_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)

        
#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'classError'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    
    
'''
#%%
##################################################################################################################################################    
########################### Get eventI for all days: needed for several of the analyses below ####################################################
##################################################################################################################################################
'''

#%% Loop over days

eventI_allDays = np.full([numDays,1], np.nan).flatten().astype('int')

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
    
    print(imfilename)        
    
    #%% Load eventI (we need it for the final alignment of corrClass traces of all days)

    # Load aligned traces, times, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
    
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    
    eventI_allDays[iday] = eventI  

print 'eventI of all days:', eventI_allDays

   
   
   
#%%   
'''
##################################################################################################################################################
########################### Projection Traces ####################################################################################################     
##################################################################################################################################################
'''

#%% Loop over days

#trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
#outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
#strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        
tr1_ave_allDays = []
tr0_ave_allDays = []
tr1_std_allDays = []
tr0_std_allDays = []
tr1_raw_ave_allDays = []
tr0_raw_ave_allDays = []
tr1_raw_std_allDays = []
tr0_raw_std_allDays = []

#eventI_allDays = (np.ones((numDays,1))+np.nan).flatten().astype('int')
ep_ms_allDays = np.full([numDays,2],np.nan) 
    
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
    

    #%% Load vars (traces, etc)

    # Load aligned traces, times, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
#        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
        time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')
    
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
#        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
        time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
        # time_aligned_stimAll = Data['stimAl_allTrs'].time.astype('float') # same as time_aligned_stim
    
#    print 'size of stimulus-aligned traces:', np.shape(traces_al_stimAll), '(frames x units x trials)'
    DataS = Data
    
#    eventI_allDays[iday] = eventI

    '''
    # Load 1stSideTry-aligned traces, frames, frame of event of interest
    # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
    time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    # print(np.shape(traces_al_1stSide))
    
    
    # Load goTone-aligned traces, frames, frame of event of interest
    # use goToneAl_noStimAft to make sure there was no stim after go tone.
    Data = scio.loadmat(postName, variable_names=['goToneAl'],squeeze_me=True,struct_as_record=False)
    traces_al_go = Data['goToneAl'].traces.astype('float')
    time_aligned_go = Data['goToneAl'].time.astype('float')
    # print(np.shape(traces_al_go))
    
    
    # Load reward-aligned traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['rewardAl'],squeeze_me=True,struct_as_record=False)
    traces_al_rew = Data['rewardAl'].traces.astype('float')
    time_aligned_rew = Data['rewardAl'].time.astype('float')
    # print(np.shape(traces_al_rew))
    
    
    # Load commitIncorrect-aligned traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['commitIncorrAl'],squeeze_me=True,struct_as_record=False)
    traces_al_incorrResp = Data['commitIncorrAl'].traces.astype('float')
    time_aligned_incorrResp = Data['commitIncorrAl'].time.astype('float')
    # print(np.shape(traces_al_incorrResp))
    
    
    # Load initiationTone-aligned traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['initToneAl'],squeeze_me=True,struct_as_record=False)
    traces_al_init = Data['initToneAl'].traces.astype('float')
    time_aligned_init = Data['initToneAl'].time.astype('float')
    # print(np.shape(traces_al_init))
    # DataI = Data
    '''        
        
        

    # Load outcomes and choice (allResp_HR_LR) for the current trial
    # if trialHistAnalysis==0:
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    # choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))    
    
    if trialHistAnalysis:
        # Load trialHistory structure to get choice vector of the previous trial
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
    
    
        
    # Set trials strength and identify trials with stim strength of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
        stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
        cb = np.array(Data.pop('cb')).flatten().astype('float')
    
        s = stimrate-cb; # how far is the stimulus rate from the category boundary?
        if strength2ana == 'easy':
            str2ana = (abs(s) >= (max(abs(s)) - thStimStrength));
        elif strength2ana == 'hard':
            str2ana = (abs(s) <= thStimStrength);
        elif strength2ana == 'medium':
            str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength))); 
        else:
            str2ana = np.full((1, np.shape(outcomes)[0]), True, dtype=bool).flatten();    
#        print 'Number of trials with stim strength of interest = %i' %(str2ana.sum())
#        print 'Stim rates for training = {}'.format(np.unique(stimrate[str2ana]))

        
        
        
    
    # Load inhibitRois
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
    # print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
    
        
    # Set traces for specific neuron types: inhibitory, excitatory or all neurons
    if neuronType!=2:    
        nt = (inhibitRois==neuronType) # 0: excitatory, 1: inhibitory, 2: all types.
        # good_excit = inhibitRois==0;
        # good_inhibit = inhibitRois==1;        
        
#        traces_al_stim = traces_al_stim[:, nt, :];
#        traces_al_1stSide = traces_al_1stSide[:, nt, :];
#        traces_al_go = traces_al_go[:, nt, :];
#        traces_al_rew = traces_al_rew[:, nt, :];
#        traces_al_incorrResp = traces_al_incorrResp[:, nt, :];
#        traces_al_init = traces_al_init[:, nt, :];
        traces_al_stimAll = traces_al_stimAll[:, nt, :];
    else:
        nt = np.arange(np.shape(traces_al_stimAll)[1])    

        
    #%% Set choiceVec0 (Y)
    
    if trialHistAnalysis:
        choiceVec0 = choiceVec0All[:,iTiFlg] # choice on the previous trial for short (or long or all) ITIs
    else: # set choice for the current trial
        choiceVec0 = allResp_HR_LR;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        # choiceVec0 = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        if outcome2ana == 'corr':
            choiceVec0[outcomes!=1] = np.nan; # analyze only correct trials.
        elif outcome2ana == 'incorr':
            choiceVec0[outcomes!=0] = np.nan; # analyze only incorrect trials.   
        
        choiceVec0[~str2ana] = np.nan   
        # Y = choiceVec0
        # print(choiceVec0.shape)
        
    
    #%% loop over the 10 rounds of analysis for each day
    
    tr1_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    tr0_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    tr1_raw_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    tr0_raw_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    
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
                    
            ##%% Load vars (w, etc)
            Data = scio.loadmat(svmName, variable_names=['trsExcluded', 'NsExcluded', 'NsRand', 'meanX', 'stdX', 'ep_ms'])
            trsExcluded = Data.pop('trsExcluded')[0,:].astype('bool')
            NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
            NsRand = Data.pop('NsRand')[0,:].astype('bool')
            meanX = Data.pop('meanX')[0,:].astype('float')
            stdX = Data.pop('stdX')[0,:].astype('float')
            ep_ms = Data.pop('ep_ms')[0,:].astype('float')
            
            Y = choiceVec0[~trsExcluded];
            ep_ms_allDays[iday,:] = ep_ms[[0,-1]]
            
            
            #%% Set the traces that will be used for projections    
        
            if trs4project=='all': 
                Xt_stimAl_all = traces_al_stimAll
    #            Xt = traces_al_stim
    #            Xt_choiceAl = traces_al_1stSide
    #            Xt_goAl = traces_al_go
    #            Xt_rewAl = traces_al_rew
    #            Xt_incorrRespAl = traces_al_incorrResp
    #            Xt_initAl = traces_al_init            
                choiceVecNow = choiceVecAll
            elif trs4project=='trained':
                Xt_stimAl_all = traces_al_stimAll[:, :, ~trsExcluded];
    #            Xt = traces_al_stim[:, :, ~trsExcluded];
    #            Xt_choiceAl = traces_al_1stSide[:, :, ~trsExcluded];
    #            Xt_goAl = traces_al_go[:, :, ~trsExcluded];
    #            Xt_rewAl = traces_al_rew[:, :, ~trsExcluded];
    #            Xt_incorrRespAl = traces_al_incorrResp[:, :, ~trsExcluded];
    #            Xt_initAl = traces_al_init[:, :, ~trsExcluded];
                choiceVecNow = Y    
            elif trs4project=='corr':
                Xt_stimAl_all = traces_al_stimAll[:, :, outcomes==1];
    #            Xt = traces_al_stim[:, :, outcomes==1];
    #            Xt_choiceAl = traces_al_1stSide[:, :, outcomes==1];
    #            Xt_goAl = traces_al_go[:, :, outcomes==1];
    #            Xt_rewAl = traces_al_rew[:, :, outcomes==1];
    #            Xt_incorrRespAl = traces_al_incorrResp[:, :, outcomes==1];
    #            Xt_initAl = traces_al_init[:, :, outcomes==1];            
                choiceVecNow = choiceVecAll[outcomes==1]
                
            elif trs4project=='incorr':
                Xt_stimAl_all = traces_al_stimAll[:, :, outcomes==0];
    #            Xt = traces_al_stim[:, :, outcomes==0];
    #            Xt_choiceAl = traces_al_1stSide[:, :, outcomes==0];
    #            Xt_goAl = traces_al_go[:, :, outcomes==0];
    #            Xt_rewAl = traces_al_rew[:, :, outcomes==0];
    #            Xt_incorrRespAl = traces_al_incorrResp[:, :, outcomes==0];
    #            Xt_initAl = traces_al_init[:, :, outcomes==0];
                choiceVecNow = choiceVecAll[outcomes==0]
                
            ## Xt = traces_al_stim[:, :, np.sum(np.sum(np.isnan(traces_al_stim), axis =0), axis =0)==0];
            ## Xt_choiceAl = traces_al_1stSide[:, :, np.sum(np.sum(np.isnan(traces_al_1stSide), axis =0), axis =0)==0];
            
            
            
            # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
            '''
    #        Xt = Xt[:,~NsExcluded,:]
            Xt_choiceAl = Xt_choiceAl[:,~NsExcluded,:]
            Xt_goAl = Xt_goAl[:,~NsExcluded,:]
            Xt_rewAl = Xt_rewAl[:,~NsExcluded,:]
            Xt_incorrRespAl = Xt_incorrRespAl[:,~NsExcluded,:]
            Xt_initAl = Xt_initAl[:,~NsExcluded,:]
            '''
            Xt_stimAl_all = Xt_stimAl_all[:,~NsExcluded,:]
                
            # Only include the randomly selected set of neurons
            '''
    #        Xt = Xt[:,NsRand,:]
            Xt_choiceAl = Xt_choiceAl[:,NsRand,:]
            Xt_goAl = Xt_goAl[:,NsRand,:]
            Xt_rewAl = Xt_rewAl[:,NsRand,:]
            Xt_incorrRespAl = Xt_incorrRespAl[:,NsRand,:]
            Xt_initAl = Xt_initAl[:,NsRand,:]
            '''
            Xt_stimAl_all = Xt_stimAl_all[:,NsRand,:]
            
            
            
            
            #%% Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
            hr_trs = (choiceVecNow==1)
            lr_trs = (choiceVecNow==0)
            # print 'Projection traces have %d high-rate trials, and %d low-rate trials' %(np.sum(hr_trs), np.sum(lr_trs))
                
                
            #%% Feature normalization and scaling        
                
            # normalize stimAll-aligned traces
            Tsa, Nsa, Csa = Xt_stimAl_all.shape
            Xtsa_N = np.reshape(Xt_stimAl_all.transpose(0 ,2 ,1), (Tsa*Csa, Nsa), order = 'F')
            Xtsa_N = (Xtsa_N-meanX)/stdX
            Xtsa = np.reshape(Xtsa_N, (Tsa, Csa, Nsa), order = 'F').transpose(0 ,2 ,1)
            
            '''
            # normalize stim-aligned traces
    #        T, N, C = Xt.shape
    #        Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    #        Xt_N = (Xt_N-meanX)/stdX
    #        Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)
            
            # normalize goTome-aligned traces
            Tg, Ng, Cg = Xt_goAl.shape
            Xtg_N = np.reshape(Xt_goAl.transpose(0 ,2 ,1), (Tg*Cg, Ng), order = 'F')
            Xtg_N = (Xtg_N-meanX)/stdX
            Xtg = np.reshape(Xtg_N, (Tg, Cg, Ng), order = 'F').transpose(0 ,2 ,1)
            
            # normalize choice-aligned traces
            Tc, Nc, Cc = Xt_choiceAl.shape
            Xtc_N = np.reshape(Xt_choiceAl.transpose(0 ,2 ,1), (Tc*Cc, Nc), order = 'F')
            Xtc_N = (Xtc_N-meanX)/stdX
            Xtc = np.reshape(Xtc_N, (Tc, Cc, Nc), order = 'F').transpose(0 ,2 ,1)
            
            # normalize reward-aligned traces
            Tr, Nr, Cr = Xt_rewAl.shape
            Xtr_N = np.reshape(Xt_rewAl.transpose(0 ,2 ,1), (Tr*Cr, Nr), order = 'F')
            Xtr_N = (Xtr_N-meanX)/stdX
            Xtr = np.reshape(Xtr_N, (Tr, Cr, Nr), order = 'F').transpose(0 ,2 ,1)
            
            # normalize commitIncorrect-aligned traces
            Tp, Np, Cp = Xt_incorrRespAl.shape
            Xtp_N = np.reshape(Xt_incorrRespAl.transpose(0 ,2 ,1), (Tp*Cp, Np), order = 'F')
            Xtp_N = (Xtp_N-meanX)/stdX
            Xtp = np.reshape(Xtp_N, (Tp, Cp, Np), order = 'F').transpose(0 ,2 ,1)        
            
            # normalize init-aligned traces
            Ti, Ni, Ci = Xt_initAl.shape
            Xti_N = np.reshape(Xt_initAl.transpose(0 ,2 ,1), (Ti*Ci, Ni), order = 'F')
            Xti_N = (Xti_N-meanX)/stdX
            Xti = np.reshape(Xti_N, (Ti, Ci, Ni), order = 'F').transpose(0 ,2 ,1)    
            '''
    #        np.shape(Xt)        
            # window of training (ep)
    #        win = ep_ms; # (ep-eventI)*frameLength
        
        
            #%% Project traces onto SVM weights
        
            # w = np.zeros(numNeurons)
            w_normalized = w/sci.linalg.norm(w);
        
            # stim-aligned traces
            # XtN_w = np.dot(Xt_N, w_normalized);
            # Xt_w = np.reshape(XtN_w, (T,C), order='F');
            # I think below can replace above, test it....
            XtN_w = np.dot(Xtsa_N, w_normalized);
            Xt_w = np.reshape(XtN_w, (Tsa,Csa), order='F');
        
            '''
            # goTone-aligned 
            XtgN_w = np.dot(Xtg_N, w_normalized);
            Xtg_w = np.reshape(XtgN_w, (Tg,Cg), order='F');
        
            # choice-aligned 
            XtcN_w = np.dot(Xtc_N, w_normalized);
            Xtc_w = np.reshape(XtcN_w, (Tc,Cc), order='F');
        
            # reward-aligned 
            XtrN_w = np.dot(Xtr_N, w_normalized);
            Xtr_w = np.reshape(XtrN_w, (Tr,Cr), order='F');
        
            # incommitResp-aligned 
            XtpN_w = np.dot(Xtp_N, w_normalized);
            Xtp_w = np.reshape(XtpN_w, (Tp,Cp), order='F');
        
            # initTone-aligned 
            XtiN_w = np.dot(Xti_N, w_normalized);
            Xti_w = np.reshape(XtiN_w, (Ti,Ci), order='F');
            '''
      
            #%% Average across trials For each round
      
            tr1 = np.nanmean(Xt_w[:, hr_trs],  axis = 1)        
            tr0 = np.nanmean(Xt_w[:, lr_trs],  axis = 1)
            
            tr1_raw = np.nanmean(Xtsa[:, :, hr_trs],  axis=(1,2)) # frames x 1; Raw averages, across neurons and trials.
            tr0_raw = np.nanmean(Xtsa[:, :, lr_trs],  axis=(1,2))
            
            
            #%% pool results of all rounds for days
            
            tr1_allRounds[:, i] = tr1  # frames x rounds
            tr0_allRounds[:, i] = tr0  # frames x rounds
          
            tr1_raw_allRounds[:, i] = tr1_raw  # frames x rounds
            tr0_raw_allRounds[:, i] = tr0_raw  # frames x rounds

        
    #%% Take average and std across rounds
        
    tr1_ave_allRounds = np.nanmean(tr1_allRounds, axis=1) # average across rounds # frames x 1
    tr0_ave_allRounds = np.nanmean(tr0_allRounds, axis=1) # average across rounds
    tr1_std_allRounds = np.nanstd(tr1_allRounds, axis=1) # std across roundss
    tr0_std_allRounds = np.nanstd(tr0_allRounds, axis=1) # std across rounds

    tr1_raw_ave_allRounds = np.nanmean(tr1_raw_allRounds, axis=1) # average across rounds # frames x 1
    tr0_raw_ave_allRounds = np.nanmean(tr0_raw_allRounds, axis=1) # average across rounds
    tr1_raw_std_allRounds = np.nanstd(tr1_raw_allRounds, axis=1) # std across roundss
    tr0_raw_std_allRounds = np.nanstd(tr0_raw_allRounds, axis=1) # std across rounds
    
          
    #%% Pool the averaged projection traces across all days

    tr1_ave_allDays.append(tr1_ave_allRounds)
    tr0_ave_allDays.append(tr0_ave_allRounds)
    tr1_std_allDays.append(tr1_std_allRounds)
    tr0_std_allDays.append(tr0_std_allRounds)    
        
    tr1_raw_ave_allDays.append(tr1_raw_ave_allRounds)
    tr0_raw_ave_allDays.append(tr0_raw_ave_allRounds)
    tr1_raw_std_allDays.append(tr1_raw_std_allRounds)
    tr0_raw_std_allDays.append(tr0_raw_std_allRounds)

        
#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(tr1_ave_allDays[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

tr1_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr0_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr1_raw_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr0_raw_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    tr1_aligned[:, iday] = tr1_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_aligned[:, iday] = tr0_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr1_raw_aligned[:, iday] = tr1_raw_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_raw_aligned[:, iday] = tr0_raw_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]

# Number of nan days
#a = np.sum(tr1_aligned, axis=0)
#np.isnan(a).sum()


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        time_aligned0 = time_aligned
        tr1_aligned0 = tr1_aligned
        tr0_aligned0 = tr0_aligned
        tr1_raw_aligned0 = tr1_raw_aligned
        tr0_raw_aligned0 = tr0_raw_aligned

    elif iTiFlg==1:
        time_aligned1 = time_aligned
        tr1_aligned1 = tr1_aligned
        tr0_aligned1 = tr0_aligned
        tr1_raw_aligned1 = tr1_raw_aligned
        tr0_raw_aligned1 = tr0_raw_aligned
        
        
#%% Average across days

tr1_aligned_ave = np.nanmean(tr1_aligned, axis=1)
tr0_aligned_ave = np.nanmean(tr0_aligned, axis=1)
tr1_aligned_std = np.nanstd(tr1_aligned, axis=1)
tr0_aligned_std = np.nanstd(tr0_aligned, axis=1)

tr1_raw_aligned_ave = np.nanmean(tr1_raw_aligned, axis=1)
tr0_raw_aligned_ave = np.nanmean(tr0_raw_aligned, axis=1)
tr1_raw_aligned_std = np.nanstd(tr1_raw_aligned, axis=1)
tr0_raw_aligned_std = np.nanstd(tr0_raw_aligned, axis=1)

_,pproj = stats.ttest_ind(tr1_aligned.transpose(), tr0_aligned.transpose()) # p value of projections being different for hr vs lr at each time point


#%% Plot the average projections across all days
#ep_ms_allDays
plt.figure(figsize=(4.5,4))

ax = plt.subplot(211)
plt.fill_between(time_aligned, tr1_aligned_ave - tr1_aligned_std, tr1_aligned_ave + tr1_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_aligned_ave - tr0_aligned_std, tr0_aligned_ave + tr0_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
plt.ylabel('SVM Projections')
makeNicePlots(ax)

# Plot a dot for time points with significantly different hr and lr projections
ymin, ymax = ax.get_ylim()
pp = pproj+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')


ax = plt.subplot(212)
plt.fill_between(time_aligned, tr1_raw_aligned_ave - tr1_raw_aligned_std, tr1_raw_aligned_ave + tr1_raw_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_raw_aligned_ave, 'b', label = 'High-rate choice')

plt.fill_between(time_aligned, tr0_raw_aligned_ave - tr0_raw_aligned_std, tr0_raw_aligned_ave + tr0_raw_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_raw_aligned_ave, 'r', label = 'Low-rate choice')

plt.xlabel('Time since stimulus onset')
plt.ylabel('Raw averages')
lgd = plt.legend(loc=0, frameon=False)
plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
makeNicePlots(ax)


print 'ep_ms_allDays: \n', ep_ms_allDays


#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'projTraces_svm_raw'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')




#%%
'''
########################################################################################################################################################       
#################### Classification accuracy at all times ##############################################################################################    
########################################################################################################################################################    
'''

#%% Loop over days

corrClass_ave_allDays = []
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
    
    corrClass_rounds = []
    
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
            Data = scio.loadmat(svmName, variable_names=['corrClass'])
            corrClass = Data.pop('corrClass') # frames x trials
            
            corrClass_rounds.append(np.mean(corrClass, axis=1)) # average across trials : numRounds x numFrames
    
    # Compute average across rounds
    corrClass_ave = np.mean(corrClass_rounds, axis=0) # numFrames x 1
    
    
    #%% Pool corrClass_ave (average of corrClass across all rounds) of all days
    
    corrClass_ave_allDays.append(corrClass_ave)
    


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

corrClass_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    corrClass_aligned[:, iday] = corrClass_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        corrClass_aligned0 = corrClass_aligned

    elif iTiFlg==1:
        corrClass_aligned1 = corrClass_aligned
        
        
#%% Average across days

corrClass_aligned_ave = np.mean(corrClass_aligned, axis=1) * 100
corrClass_aligned_std = np.std(corrClass_aligned, axis=1) * 100

_,pcorrtrace = stats.ttest_1samp(corrClass_aligned.transpose(), 50) # p value of class accuracy being different from 50

        
       
#%% Plot the average traces across all days
#ep_ms_allDays
plt.figure(figsize=(4.5,3))

plt.fill_between(time_aligned, corrClass_aligned_ave - corrClass_aligned_std, corrClass_aligned_ave + corrClass_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, corrClass_aligned_ave, 'k')

plt.xlabel('Time since stim onset (ms)', fontsize=13)
plt.ylabel('Classification accuracy (%)', fontsize=13)
plt.legend()
ax = plt.gca()
plt.xticks(np.arange(0,1400,400))
makeNicePlots(ax)

# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()
'''
pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')
'''
#plt.xticks(np.arange(-400,1400,200))
#win=[time_aligned[0], 0]
#plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
#plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])
    
# Plot lines for the training epoch
if 'ep_ms_allDays' in locals():
    win = np.mean(ep_ms_allDays, axis=0)
    plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
    plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])


#%% Save the figure    
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'corrClassTrace'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_inches='tight')#, bbox_extra_artists=(lgd,))



    
    
    
    
    
    
    
    
    
    
#%%
'''
#####################################################################################################################################################
#####################################################################################################################################################    
############### Excitatory vs inhibitory neurons:  weights ##########################################################################################
#####################################################################################################################################################   
#####################################################################################################################################################
'''

#w_alln = [];
w_inh = [];
w_exc = [];
w_uns = [];
w_inh_fractNonZero = []
w_exc_fractNonZero = []    

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

   
   # Load inhibitRois
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
    # print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))


    #%% loop over the 10 rounds of analysis for each day

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
            ##%% Load vars (w, etc)
            Data = scio.loadmat(svmName, variable_names=['NsExcluded', 'NsRand'])        
            NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
            NsRand = Data.pop('NsRand')[0,:].astype('bool')
    
            # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
            if neuronType==2:
                inhRois = inhibitRois[~NsExcluded]        
                inhRois = inhRois[NsRand]
    
            w = w/sci.linalg.norm(w); # normalzied weights
    
    #        w_alln.append(w)
            w_inh.append(w[inhRois==1]) # pool all rounds and all days
            w_exc.append(w[inhRois==0])
            w_uns.append(w[np.isnan(inhRois)])

            w_inh_fractNonZero.append(np.mean([w[inhRois==1]!=0])) # fraction of non-0 weights; # pool all rounds and all days
            w_exc_fractNonZero.append(np.mean([w[inhRois==0]!=0]))
            

#w_alln_all = np.concatenate(w_alln, axis=0)
w_exc_all = np.concatenate(w_exc, axis=0)
w_inh_all = np.concatenate(w_inh, axis=0)
w_uns_all = np.concatenate(w_uns, axis=0)

#w_inh_fractNonZero_all = np.concatenate(w_inh_fractNonZero, axis=0)
#w_exc_fractNonZero_all = np.concatenate(w_exc_fractNonZero, axis=0)

#len_exc = np.shape(w_exc_all)[0]
#len_inh = np.shape(w_inh_all)[0]
#len_uns = np.shape(w_uns_all)[0]
#print 'length of w for exc = %d, inh = %d, unsure = %d' %(len_exc, len_inh, len_uns)

w_exc_all_o = w_exc_all+0;
w_inh_all_o = w_inh_all+0;


#%% Only get non-zero weights
w_exc_all = w_exc_all_o[w_exc_all_o!=0]
w_inh_all = w_inh_all_o[w_inh_all_o!=0]

print '\nLength of w for exc, inh (all, non-zero):', np.shape(w_exc_all_o), np.shape(w_inh_all_o), np.shape(w_exc_all), np.shape(w_inh_all)


#%% Compute p values
h, p = stats.ttest_ind(w_inh_all, w_exc_all)
print '\nmean(w): inhibit = %.3f  excit = %.3f' %(np.nanmean(w_inh_all), np.nanmean(w_exc_all))
print 'p-val_2tailed (inhibit vs excit weights) = %.2f' %p
_, pabs = stats.ttest_ind(abs(w_inh_all), abs(w_exc_all))
print '\nmean(abs(w)): inhibit = %.3f  excit = %.3f' %(np.nanmean(abs(w_inh_all)), np.nanmean(abs(w_exc_all)))
print 'p-val_2tailed (inhibit vs excit weights) = %.2f' %pabs
_, pfrac = stats.ttest_ind(w_inh_fractNonZero, w_exc_fractNonZero)
print '\nmean fraction of non-zero weights: inhibit = %.3f  excit = %.3f' %(np.nanmean(w_inh_fractNonZero), np.nanmean(w_exc_fractNonZero))
print 'p-val_2tailed (inhibit vs excit fract non-0 weights) = %.2f' %pfrac


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        w_exc_all0 = w_exc_all
        w_inh_all0 = w_inh_all
        w_exc_fractNonZero0 = w_exc_fractNonZero
        w_inh_fractNonZero0 = w_inh_fractNonZero

    elif iTiFlg==1:
        w_exc_all1 = w_exc_all
        w_inh_all1 = w_inh_all
        w_exc_fractNonZero1 = w_exc_fractNonZero
        w_inh_fractNonZero1 = w_inh_fractNonZero
        
        
#%% Plot histograms of weights


#% hist of normalized non-zero weights for exc and inh neurons
plt.figure()
plt.subplots(2,2,figsize=(7,4))

plt.subplot(221)
hist, bin_edges = np.histogram(w_exc_all[~np.isnan(w_exc_all)], bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='exc')

hist, bin_edges = np.histogram(w_inh_all[~np.isnan(w_inh_all)], bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='inh')

plt.legend(loc=0, frameon=False)
plt.xlabel('Norm weight (non-zero)')
plt.ylabel('Normalized count')
#plt.ylim([-.1, 1.1])
#plt.title('w')
plt.title('mean: inhibit = %.2f  excit = %.2f\np-val_2tailed = %.2f ' %(np.mean((w_inh_all)), np.mean((w_exc_all)), p), y=1.08)
ax = plt.gca()
makeNicePlots(ax)



#% hist of normalized abs non-zero weights for exc and inh neurons
plt.subplot(222)
a = abs(w_exc_all)
hist, bin_edges = np.histogram(a[~np.isnan(a)], bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='exc')

a = abs(w_inh_all)
hist, bin_edges = np.histogram(a[~np.isnan(a)], bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='inh')
plt.title('mean: inhibit = %.2f  excit = %.2f\np-val_2tailed = %.2f ' %(np.mean(abs(w_inh_all)), np.mean(abs(w_exc_all)), pabs), y=1.08)

lgd = plt.legend(loc=0, frameon=False)
plt.xlabel('Norm abs weight (non-zero)')
plt.ylabel('Normalized count')
#plt.ylim([-.1, 1.1])
#plt.title('abs(w)')
plt.tight_layout()
ax = plt.gca()
makeNicePlots(ax)


##%% hist of fraction of non zero weights for exc vs inh

#plt.figure();
plt.subplot(223)
hist, bin_edges = np.histogram(w_exc_fractNonZero, bins=50)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='exc')

hist, bin_edges = np.histogram(w_inh_fractNonZero, bins=50)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='inh')

plt.legend(loc=0, frameon=False)
plt.xlabel('Fraction of non-0 weights')
plt.ylabel('Normalized count')
#plt.ylim([-.1, 1.1])
#plt.title('w')
#plt.title('mean(w): inhibit = %.3f  excit = %.3f\np-val_2tailed (inhibit vs excit weights) = %.2f ' %(np.mean((w_inh_all)), np.mean((w_exc_all)), p))
ax = plt.gca()
makeNicePlots(ax)

plt.title('mean: inhibit = %.2f  excit = %.2f\np-val_2tailed = %.2f ' %(np.nanmean(w_inh_fractNonZero), np.nanmean(w_exc_fractNonZero), pfrac), y=1.08)

plt.subplots_adjust(hspace=1, top = 1)

plt.subplot(224)
plt.axis('off')


#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffnei+'weights'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')






            
#%%
'''
#####################################################################################################################################################    
################ Excitatory vs inhibitory neurons:  projections #######################################################################################
#####################################################################################################################################################   
'''    

#%% Loop over days

trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        

tr1_i_ave_allDays = []
tr0_i_ave_allDays = []
tr1_i_std_allDays = []
tr0_i_std_allDays = []
tr1_e_ave_allDays = []
tr0_e_ave_allDays = []
tr1_e_std_allDays = []
tr0_e_std_allDays = []
tr1_raw_i_ave_allDays = []
tr0_raw_i_ave_allDays = []
tr1_raw_i_std_allDays = []
tr0_raw_i_std_allDays = []
tr1_raw_e_ave_allDays = []
tr0_raw_e_ave_allDays = []
tr1_raw_e_std_allDays = []
tr0_raw_e_std_allDays = []
    
#eventI_allDays = (np.ones((numDays,1))+np.nan).flatten().astype('int')
ep_ms_allDays = np.full([numDays,2],np.nan) 
    
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
    

    #%% Load vars (traces, etc)

    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
#        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
        time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')
    
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
#        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
        time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
        # time_aligned_stimAll = Data['stimAl_allTrs'].time.astype('float') # same as time_aligned_stim
    
#    print 'size of stimulus-aligned traces:', np.shape(traces_al_stimAll), '(frames x units x trials)'
    
#    eventI_allDays[iday] = eventI

    '''
    # Load 1stSideTry-aligned traces, frames, frame of event of interest
    # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
    time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    # print(np.shape(traces_al_1stSide))
    
    
    # Load goTone-aligned traces, frames, frame of event of interest
    # use goToneAl_noStimAft to make sure there was no stim after go tone.
    Data = scio.loadmat(postName, variable_names=['goToneAl'],squeeze_me=True,struct_as_record=False)
    traces_al_go = Data['goToneAl'].traces.astype('float')
    time_aligned_go = Data['goToneAl'].time.astype('float')
    # print(np.shape(traces_al_go))
    
    
    # Load reward-aligned traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['rewardAl'],squeeze_me=True,struct_as_record=False)
    traces_al_rew = Data['rewardAl'].traces.astype('float')
    time_aligned_rew = Data['rewardAl'].time.astype('float')
    # print(np.shape(traces_al_rew))
    
    
    # Load commitIncorrect-aligned traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['commitIncorrAl'],squeeze_me=True,struct_as_record=False)
    traces_al_incorrResp = Data['commitIncorrAl'].traces.astype('float')
    time_aligned_incorrResp = Data['commitIncorrAl'].time.astype('float')
    # print(np.shape(traces_al_incorrResp))
    
    
    # Load initiationTone-aligned traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['initToneAl'],squeeze_me=True,struct_as_record=False)
    traces_al_init = Data['initToneAl'].traces.astype('float')
    time_aligned_init = Data['initToneAl'].time.astype('float')
    # print(np.shape(traces_al_init))
    # DataI = Data
    '''
    
        

    # Load outcomes and choice (allResp_HR_LR) for the current trial
    # if trialHistAnalysis==0:
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    # choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
#    print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))    
    
    if trialHistAnalysis:
        # Load trialHistory structure to get choice vector of the previous trial
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
    
    
        
    # Set trials strength and identify trials with stim strength of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
        stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
        cb = np.array(Data.pop('cb')).flatten().astype('float')
    
        s = stimrate-cb; # how far is the stimulus rate from the category boundary?
        if strength2ana == 'easy':
            str2ana = (abs(s) >= (max(abs(s)) - thStimStrength));
        elif strength2ana == 'hard':
            str2ana = (abs(s) <= thStimStrength);
        elif strength2ana == 'medium':
            str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength))); 
        else:
            str2ana = np.full((1, np.shape(outcomes)[0]), True, dtype=bool).flatten();    
#        print 'Number of trials with stim strength of interest = %i' %(str2ana.sum())
#        print 'Stim rates for training = {}'.format(np.unique(stimrate[str2ana]))

        
        
    '''    
    # Set ep for trialHist case
    if trialHistAnalysis:
        # either of the two below (stimulus-aligned and initTone-aligned) would be fine
        # eventI = DataI['initToneAl'].eventI
        eventI = DataS['stimAl_allTrs'].eventI    
        epEnd = eventI + epEnd_rel2stimon_fr #- 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
        # epEnd = DataI['initToneAl'].eventI - 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
        ep = np.arange(epEnd+1)
#        print 'training epoch is {} ms'.format(np.round((ep-eventI)*frameLength))
    '''    
        
    
    # Load inhibitRois
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
    # print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
    
        
    # Set traces for specific neuron types: inhibitory, excitatory or all neurons
    if neuronType!=2:    
        nt = (inhibitRois==neuronType) # 0: excitatory, 1: inhibitory, 2: all types.
        # good_excit = inhibitRois==0;
        # good_inhibit = inhibitRois==1;        
        
#        traces_al_stim = traces_al_stim[:, nt, :];
#        traces_al_1stSide = traces_al_1stSide[:, nt, :];
#        traces_al_go = traces_al_go[:, nt, :];
#        traces_al_rew = traces_al_rew[:, nt, :];
#        traces_al_incorrResp = traces_al_incorrResp[:, nt, :];
#        traces_al_init = traces_al_init[:, nt, :];
        traces_al_stimAll = traces_al_stimAll[:, nt, :];
    else:
        nt = np.arange(np.shape(traces_al_stimAll)[1])    

       
    #%% Set choiceVec0 (Y)
    
    if trialHistAnalysis:
        choiceVec0 = choiceVec0All[:,iTiFlg] # choice on the previous trial for short (or long or all) ITIs
    else: # set choice for the current trial
        choiceVec0 = allResp_HR_LR;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        # choiceVec0 = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        if outcome2ana == 'corr':
            choiceVec0[outcomes!=1] = np.nan; # analyze only correct trials.
        elif outcome2ana == 'incorr':
            choiceVec0[outcomes!=0] = np.nan; # analyze only incorrect trials.   
        
        choiceVec0[~str2ana] = np.nan   
        # Y = choiceVec0
        # print(choiceVec0.shape)

    
    #%% loop over the 10 rounds of analysis for each day
    
    tr1_i_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    tr0_i_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds    
    tr1_e_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    tr0_e_allRounds = np.ones((traces_al_stimAll.shape[0], numRounds))+np.nan # frames x rounds
    tr1_raw_i_allRounds = np.full((traces_al_stimAll.shape[0], numRounds), np.nan)
    tr0_raw_i_allRounds = np.full((traces_al_stimAll.shape[0], numRounds), np.nan)
    tr1_raw_e_allRounds = np.full((traces_al_stimAll.shape[0], numRounds), np.nan)
    tr0_raw_e_allRounds = np.full((traces_al_stimAll.shape[0], numRounds), np.nan)

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
            ##%% Load vars (w, etc)
            Data = scio.loadmat(svmName, variable_names=['trsExcluded', 'NsExcluded', 'NsRand', 'meanX', 'stdX', 'ep_ms'])        
            trsExcluded = Data.pop('trsExcluded')[0,:].astype('bool')
            NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
            NsRand = Data.pop('NsRand')[0,:].astype('bool')
            meanX = Data.pop('meanX')[0,:].astype('float')
            stdX = Data.pop('stdX')[0,:].astype('float')
            ep_ms = Data.pop('ep_ms')[0,:].astype('float')
            
            Y = choiceVec0[~trsExcluded];
            ep_ms_allDays[iday,:] = ep_ms[[0,-1]]
            
            if roundi==1:
                print '%d high-rate choices, and %d low-rate choices\n' %(np.sum(Y==1), np.sum(Y==0))
        
            # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
            if neuronType==2:
                inhRois = inhibitRois[~NsExcluded]        
                inhRois = inhRois[NsRand]
                
                if roundi==1:
                    print (inhRois==1).sum(), 'inhibitory and', (inhRois==0).sum(), 'excitatory neurons'
    
    
            #%% Set the traces that will be used for projections    
        
            if trs4project=='all': 
                Xt_stimAl_all = traces_al_stimAll
    #            Xt = traces_al_stim
    #            Xt_choiceAl = traces_al_1stSide
    #            Xt_goAl = traces_al_go
    #            Xt_rewAl = traces_al_rew
    #            Xt_incorrRespAl = traces_al_incorrResp
    #            Xt_initAl = traces_al_init            
                choiceVecNow = choiceVecAll
            elif trs4project=='trained':
                Xt_stimAl_all = traces_al_stimAll[:, :, ~trsExcluded];
    #            Xt = traces_al_stim[:, :, ~trsExcluded];
    #            Xt_choiceAl = traces_al_1stSide[:, :, ~trsExcluded];
    #            Xt_goAl = traces_al_go[:, :, ~trsExcluded];
    #            Xt_rewAl = traces_al_rew[:, :, ~trsExcluded];
    #            Xt_incorrRespAl = traces_al_incorrResp[:, :, ~trsExcluded];
    #            Xt_initAl = traces_al_init[:, :, ~trsExcluded];
                choiceVecNow = Y    
            elif trs4project=='corr':
                Xt_stimAl_all = traces_al_stimAll[:, :, outcomes==1];
    #            Xt = traces_al_stim[:, :, outcomes==1];
    #            Xt_choiceAl = traces_al_1stSide[:, :, outcomes==1];
    #            Xt_goAl = traces_al_go[:, :, outcomes==1];
    #            Xt_rewAl = traces_al_rew[:, :, outcomes==1];
    #            Xt_incorrRespAl = traces_al_incorrResp[:, :, outcomes==1];
    #            Xt_initAl = traces_al_init[:, :, outcomes==1];            
                choiceVecNow = choiceVecAll[outcomes==1]
                
            elif trs4project=='incorr':
                Xt_stimAl_all = traces_al_stimAll[:, :, outcomes==0];
    #            Xt = traces_al_stim[:, :, outcomes==0];
    #            Xt_choiceAl = traces_al_1stSide[:, :, outcomes==0];
    #            Xt_goAl = traces_al_go[:, :, outcomes==0];
    #            Xt_rewAl = traces_al_rew[:, :, outcomes==0];
    #            Xt_incorrRespAl = traces_al_incorrResp[:, :, outcomes==0];
    #            Xt_initAl = traces_al_init[:, :, outcomes==0];
                choiceVecNow = choiceVecAll[outcomes==0]
                
            ## Xt = traces_al_stim[:, :, np.sum(np.sum(np.isnan(traces_al_stim), axis =0), axis =0)==0];
            ## Xt_choiceAl = traces_al_1stSide[:, :, np.sum(np.sum(np.isnan(traces_al_1stSide), axis =0), axis =0)==0];
            
            
            
            # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
            '''
    #        Xt = Xt[:,~NsExcluded,:]
            Xt_choiceAl = Xt_choiceAl[:,~NsExcluded,:]
            Xt_goAl = Xt_goAl[:,~NsExcluded,:]
            Xt_rewAl = Xt_rewAl[:,~NsExcluded,:]
            Xt_incorrRespAl = Xt_incorrRespAl[:,~NsExcluded,:]
            Xt_initAl = Xt_initAl[:,~NsExcluded,:]
            '''
            Xt_stimAl_all = Xt_stimAl_all[:,~NsExcluded,:]
                
            # Only include the randomly selected set of neurons
            '''
    #        Xt = Xt[:,NsRand,:]
            Xt_choiceAl = Xt_choiceAl[:,NsRand,:]
            Xt_goAl = Xt_goAl[:,NsRand,:]
            Xt_rewAl = Xt_rewAl[:,NsRand,:]
            Xt_incorrRespAl = Xt_incorrRespAl[:,NsRand,:]
            Xt_initAl = Xt_initAl[:,NsRand,:]
            '''
            Xt_stimAl_all = Xt_stimAl_all[:,NsRand,:]
            
            
            
            
            #%% Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
            hr_trs = (choiceVecNow==1)
            lr_trs = (choiceVecNow==0)
            # print 'Projection traces have %d high-rate trials, and %d low-rate trials' %(np.sum(hr_trs), np.sum(lr_trs))
                
                
            #%% Feature normalization and scaling        
                
            # normalize stimAll-aligned traces
            Tsa, Nsa, Csa = Xt_stimAl_all.shape
            Xtsa_N = np.reshape(Xt_stimAl_all.transpose(0 ,2 ,1), (Tsa*Csa, Nsa), order = 'F')
            Xtsa_N = (Xtsa_N-meanX)/stdX
            Xtsa = np.reshape(Xtsa_N, (Tsa, Csa, Nsa), order = 'F').transpose(0 ,2 ,1)
            
            '''
            # normalize stim-aligned traces
    #        T, N, C = Xt.shape
    #        Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    #        Xt_N = (Xt_N-meanX)/stdX
    #        Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)
            
            # normalize goTome-aligned traces
            Tg, Ng, Cg = Xt_goAl.shape
            Xtg_N = np.reshape(Xt_goAl.transpose(0 ,2 ,1), (Tg*Cg, Ng), order = 'F')
            Xtg_N = (Xtg_N-meanX)/stdX
            Xtg = np.reshape(Xtg_N, (Tg, Cg, Ng), order = 'F').transpose(0 ,2 ,1)
            
            # normalize choice-aligned traces
            Tc, Nc, Cc = Xt_choiceAl.shape
            Xtc_N = np.reshape(Xt_choiceAl.transpose(0 ,2 ,1), (Tc*Cc, Nc), order = 'F')
            Xtc_N = (Xtc_N-meanX)/stdX
            Xtc = np.reshape(Xtc_N, (Tc, Cc, Nc), order = 'F').transpose(0 ,2 ,1)
            
            # normalize reward-aligned traces
            Tr, Nr, Cr = Xt_rewAl.shape
            Xtr_N = np.reshape(Xt_rewAl.transpose(0 ,2 ,1), (Tr*Cr, Nr), order = 'F')
            Xtr_N = (Xtr_N-meanX)/stdX
            Xtr = np.reshape(Xtr_N, (Tr, Cr, Nr), order = 'F').transpose(0 ,2 ,1)
            
            # normalize commitIncorrect-aligned traces
            Tp, Np, Cp = Xt_incorrRespAl.shape
            Xtp_N = np.reshape(Xt_incorrRespAl.transpose(0 ,2 ,1), (Tp*Cp, Np), order = 'F')
            Xtp_N = (Xtp_N-meanX)/stdX
            Xtp = np.reshape(Xtp_N, (Tp, Cp, Np), order = 'F').transpose(0 ,2 ,1)        
            
            # normalize init-aligned traces
            Ti, Ni, Ci = Xt_initAl.shape
            Xti_N = np.reshape(Xt_initAl.transpose(0 ,2 ,1), (Ti*Ci, Ni), order = 'F')
            Xti_N = (Xti_N-meanX)/stdX
            Xti = np.reshape(Xti_N, (Ti, Ci, Ni), order = 'F').transpose(0 ,2 ,1)    
            '''
    #        np.shape(Xt)        
            # window of training (ep)
    #        win = ep_ms; # (ep-eventI)*frameLength
        
        
            #%% Project traces onto SVM weights
        
            # Set weights of excit neurons to 0 and project the traces onto this new decoder that cares only about inhibitory neurons.
            ww = w+0;
            ww[inhRois==0] = 0
        
            # w = np.zeros(numNeurons)
            w_normalized = ww/sci.linalg.norm(ww);
        
            # stim-aligned traces
            # XtN_w = np.dot(Xt_N, w_normalized);
            # Xt_w = np.reshape(XtN_w, (T,C), order='F');
            # I think below can replace above, test it....
            XtN_w = np.dot(Xtsa_N, w_normalized);
            Xt_w_i = np.reshape(XtN_w, (Tsa,Csa), order='F');
        
            '''
            # goTone-aligned 
            XtgN_w = np.dot(Xtg_N, w_normalized);
            Xtg_w_i = np.reshape(XtgN_w, (Tg,Cg), order='F');
        
            # choice-aligned 
            XtcN_w = np.dot(Xtc_N, w_normalized);
            Xtc_w_i = np.reshape(XtcN_w, (Tc,Cc), order='F');
        
            # reward-aligned 
            XtrN_w = np.dot(Xtr_N, w_normalized);
            Xtr_w_i = np.reshape(XtrN_w, (Tr,Cr), order='F');
        
            # incommitResp-aligned 
            XtpN_w = np.dot(Xtp_N, w_normalized);
            Xtp_w_i = np.reshape(XtpN_w, (Tp,Cp), order='F');
        
            # initTone-aligned 
            XtiN_w = np.dot(Xti_N, w_normalized);
            Xti_w_i = np.reshape(XtiN_w, (Ti,Ci), order='F');
            '''
      
      
            # Set weights of inhibit neurons to 0 and project the traces onto this new decoder that cares only about the excitatory neurons.  
            ww = w+0;
            ww[inhRois==1] = 0
        
            # w = np.zeros(numNeurons)
            w_normalized = ww/sci.linalg.norm(ww);
        
            # stim-aligned traces
            # XtN_w = np.dot(Xt_N, w_normalized);
            # Xt_w = np.reshape(XtN_w, (T,C), order='F');
            # I think below can replace above, test it....
            XtN_w = np.dot(Xtsa_N, w_normalized);
            Xt_w_e = np.reshape(XtN_w, (Tsa,Csa), order='F');
        
            '''
            # goTone-aligned 
            XtgN_w = np.dot(Xtg_N, w_normalized);
            Xtg_w_e = np.reshape(XtgN_w, (Tg,Cg), order='F');
        
            # choice-aligned 
            XtcN_w = np.dot(Xtc_N, w_normalized);
            Xtc_w_e = np.reshape(XtcN_w, (Tc,Cc), order='F');
        
            # reward-aligned 
            XtrN_w = np.dot(Xtr_N, w_normalized);
            Xtr_w_e = np.reshape(XtrN_w, (Tr,Cr), order='F');
        
            # incommitResp-aligned 
            XtpN_w = np.dot(Xtp_N, w_normalized);
            Xtp_w_e = np.reshape(XtpN_w, (Tp,Cp), order='F');
        
            # initTone-aligned 
            XtiN_w = np.dot(Xti_N, w_normalized);
            Xti_w_e = np.reshape(XtiN_w, (Ti,Ci), order='F');
            '''
            
            
            #%% Average across trials For each round
      
            tr1_i = np.nanmean(Xt_w_i[:, hr_trs],  axis = 1)        
            tr0_i = np.nanmean(Xt_w_i[:, lr_trs],  axis = 1)        
            tr1_e = np.nanmean(Xt_w_e[:, hr_trs],  axis = 1)        
            tr0_e = np.nanmean(Xt_w_e[:, lr_trs],  axis = 1)
    
            tr1_raw_i = np.nanmean(Xtsa[np.ix_(range(Xtsa.shape[0]), inhRois==1, hr_trs)],  axis=(1,2)) # frames x 1; Raw averages, across neurons and trials.
            tr0_raw_i = np.nanmean(Xtsa[np.ix_(range(Xtsa.shape[0]), inhRois==1, lr_trs)],  axis=(1,2))
            tr1_raw_e = np.nanmean(Xtsa[np.ix_(range(Xtsa.shape[0]), inhRois==0, hr_trs)],  axis=(1,2)) # frames x 1; Raw averages, across neurons and trials.
            tr0_raw_e = np.nanmean(Xtsa[np.ix_(range(Xtsa.shape[0]), inhRois==0, lr_trs)],  axis=(1,2))
    
            
            #%% pool results of all rounds for days
            
            tr1_i_allRounds[:, i] = tr1_i  # frames x rounds
            tr0_i_allRounds[:, i] = tr0_i  # frames x rounds      
            tr1_e_allRounds[:, i] = tr1_e  # frames x rounds
            tr0_e_allRounds[:, i] = tr0_e  # frames x rounds
    
            tr1_raw_i_allRounds[:, i] = tr1_raw_i  # frames x rounds
            tr0_raw_i_allRounds[:, i] = tr0_raw_i  # frames x rounds      
            tr1_raw_e_allRounds[:, i] = tr1_raw_e  # frames x rounds
            tr0_raw_e_allRounds[:, i] = tr0_raw_e  # frames x rounds
        
                
    #%% Take average and std across rounds
        
    tr1_i_ave_allRounds = np.nanmean(tr1_i_allRounds, axis=1) # average across rounds # frames x 1
    tr0_i_ave_allRounds = np.nanmean(tr0_i_allRounds, axis=1) # average across rounds
    tr1_i_std_allRounds = np.nanstd(tr1_i_allRounds, axis=1) # std across roundss
    tr0_i_std_allRounds = np.nanstd(tr0_i_allRounds, axis=1) # std across rounds
    tr1_e_ave_allRounds = np.nanmean(tr1_e_allRounds, axis=1) # average across rounds # frames x 1
    tr0_e_ave_allRounds = np.nanmean(tr0_e_allRounds, axis=1) # average across rounds
    tr1_e_std_allRounds = np.nanstd(tr1_e_allRounds, axis=1) # std across roundss
    tr0_e_std_allRounds = np.nanstd(tr0_e_allRounds, axis=1) # std across rounds
    
    tr1_raw_i_ave_allRounds = np.nanmean(tr1_raw_i_allRounds, axis=1) # average across rounds # frames x 1
    tr0_raw_i_ave_allRounds = np.nanmean(tr0_raw_i_allRounds, axis=1) # average across rounds
    tr1_raw_i_std_allRounds = np.nanstd(tr1_raw_i_allRounds, axis=1) # std across roundss
    tr0_raw_i_std_allRounds = np.nanstd(tr0_raw_i_allRounds, axis=1) # std across rounds
    tr1_raw_e_ave_allRounds = np.nanmean(tr1_raw_e_allRounds, axis=1) # average across rounds # frames x 1
    tr0_raw_e_ave_allRounds = np.nanmean(tr0_raw_e_allRounds, axis=1) # average across rounds
    tr1_raw_e_std_allRounds = np.nanstd(tr1_raw_e_allRounds, axis=1) # std across roundss
    tr0_raw_e_std_allRounds = np.nanstd(tr0_raw_e_allRounds, axis=1) # std across rounds


    #%% Pool the averaged projection traces across all days

    tr1_i_ave_allDays.append(tr1_i_ave_allRounds)
    tr0_i_ave_allDays.append(tr0_i_ave_allRounds)
    tr1_i_std_allDays.append(tr1_i_std_allRounds)
    tr0_i_std_allDays.append(tr0_i_std_allRounds)
    tr1_e_ave_allDays.append(tr1_e_ave_allRounds)
    tr0_e_ave_allDays.append(tr0_e_ave_allRounds)
    tr1_e_std_allDays.append(tr1_e_std_allRounds)
    tr0_e_std_allDays.append(tr0_e_std_allRounds) 
    
    tr1_raw_i_ave_allDays.append(tr1_raw_i_ave_allRounds)
    tr0_raw_i_ave_allDays.append(tr0_raw_i_ave_allRounds)
    tr1_raw_i_std_allDays.append(tr1_raw_i_std_allRounds)
    tr0_raw_i_std_allDays.append(tr0_raw_i_std_allRounds)
    tr1_raw_e_ave_allDays.append(tr1_raw_e_ave_allRounds)
    tr0_raw_e_ave_allDays.append(tr0_raw_e_ave_allRounds)
    tr1_raw_e_std_allDays.append(tr1_raw_e_std_allRounds)
    tr0_raw_e_std_allDays.append(tr0_raw_e_std_allRounds)    
        
        
#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(tr1_i_ave_allDays[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

tr1_i_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr0_i_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr1_e_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr0_e_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr1_raw_i_aligned = np.full((nPreMin + nPostMin + 1, numDays), np.nan)
tr0_raw_i_aligned = np.full((nPreMin + nPostMin + 1, numDays), np.nan)
tr1_raw_e_aligned = np.full((nPreMin + nPostMin + 1, numDays), np.nan)
tr0_raw_e_aligned = np.full((nPreMin + nPostMin + 1, numDays), np.nan)

for iday in range(numDays):
    tr1_i_aligned[:, iday] = tr1_i_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_i_aligned[:, iday] = tr0_i_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr1_e_aligned[:, iday] = tr1_e_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_e_aligned[:, iday] = tr0_e_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]

    tr1_raw_i_aligned[:, iday] = tr1_raw_i_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_raw_i_aligned[:, iday] = tr0_raw_i_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr1_raw_e_aligned[:, iday] = tr1_raw_e_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_raw_e_aligned[:, iday] = tr0_raw_e_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    
    
#Number of nan days
#a = np.sum(tr1_i_aligned, axis=0)
#np.isnan(a).sum()


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        tr1_i_aligned0 = tr1_i_aligned
        tr0_i_aligned0 = tr0_i_aligned
        tr1_i_raw_aligned0 = tr1_raw_i_aligned
        tr0_i_raw_aligned0 = tr0_raw_i_aligned
        tr1_e_aligned0 = tr1_e_aligned
        tr0_e_aligned0 = tr0_e_aligned
        tr1_e_raw_aligned0 = tr1_raw_e_aligned
        tr0_e_raw_aligned0 = tr0_raw_e_aligned
        
    elif iTiFlg==1:
        tr1_i_aligned1 = tr1_i_aligned
        tr0_i_aligned1 = tr0_i_aligned
        tr1_i_raw_aligned1 = tr1_raw_i_aligned
        tr0_i_raw_aligned1 = tr0_raw_i_aligned
        tr1_e_aligned1 = tr1_e_aligned
        tr0_e_aligned1 = tr0_e_aligned
        tr1_e_raw_aligned1 = tr1_raw_e_aligned
        tr0_e_raw_aligned1 = tr0_raw_e_aligned
        
    
#%% Average across days

tr1_i_aligned_ave = np.mean(tr1_i_aligned, axis=1)
tr0_i_aligned_ave = np.mean(tr0_i_aligned, axis=1)
tr1_i_aligned_std = np.std(tr1_i_aligned, axis=1)
tr0_i_aligned_std = np.std(tr0_i_aligned, axis=1)
tr1_e_aligned_ave = np.mean(tr1_e_aligned, axis=1)
tr0_e_aligned_ave = np.mean(tr0_e_aligned, axis=1)
tr1_e_aligned_std = np.std(tr1_e_aligned, axis=1)
tr0_e_aligned_std = np.std(tr0_e_aligned, axis=1)

tr1_raw_i_aligned_ave = np.mean(tr1_raw_i_aligned, axis=1)
tr0_raw_i_aligned_ave = np.mean(tr0_raw_i_aligned, axis=1)
tr1_raw_i_aligned_std = np.std(tr1_raw_i_aligned, axis=1)
tr0_raw_i_aligned_std = np.std(tr0_raw_i_aligned, axis=1)
tr1_raw_e_aligned_ave = np.mean(tr1_raw_e_aligned, axis=1)
tr0_raw_e_aligned_ave = np.mean(tr0_raw_e_aligned, axis=1)
tr1_raw_e_aligned_std = np.std(tr1_raw_e_aligned, axis=1)
tr0_raw_e_aligned_std = np.std(tr0_raw_e_aligned, axis=1)

_,pi = stats.ttest_ind(tr1_i_aligned.transpose(), tr0_i_aligned.transpose()) # p value of projections being different for hr vs lr at each time point
_,pe = stats.ttest_ind(tr1_e_aligned.transpose(), tr0_e_aligned.transpose()) 


#%% Plot the average projections across all days

# SVM projections

plt.figure(figsize=(9,6)) #

plt.subplot(221)
plt.fill_between(time_aligned, tr1_e_aligned_ave - tr1_e_aligned_std, tr1_e_aligned_ave + tr1_e_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_e_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_e_aligned_ave - tr0_e_aligned_std, tr0_e_aligned_ave + tr0_e_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_e_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stim onset', fontsize=13) #, labelpad=10)
plt.ylabel('SVM projections', fontsize=13)
plt.title('Excitatory')
ax = plt.gca()
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.xticks(np.arange(0,xmax,400))    
ax.text(xmax, ymin-.4, 'ms') #, style='italic') #,      bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
# Plot a dot for time points with significantly different hr and lr projections
pp = pe+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

makeNicePlots(ax,0,1)
#f.set_figheight(6)
#f.set_figwidth(2.5)
#f.set_size_inches(2.5)


plt.subplot(222)
plt.fill_between(time_aligned, tr1_i_aligned_ave - tr1_i_aligned_std, tr1_i_aligned_ave + tr1_i_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_i_aligned_ave, 'b', label = 'high-rate choice')

plt.fill_between(time_aligned, tr0_i_aligned_ave - tr0_i_aligned_std, tr0_i_aligned_ave + tr0_i_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_i_aligned_ave, 'r', label = 'low-rate choice')

plt.xlabel('Time since stim onset', fontsize=13)
plt.ylabel('SVM projections', fontsize=13)
plt.title('Inhibitory')
plt.ylim([ymin, ymax])
ax = plt.gca()
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.xticks(np.arange(0,xmax,400))    
ax.text(xmax, ymin-.4, 'ms') #, style='italic') #,      bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
#lgd = plt.legend(loc=0, bbox_to_anchor=(.46, 1.1), frameon=False)
lgd = plt.legend(loc=0, bbox_to_anchor=(1, 1.1), frameon=False)
# Plot a dot for time points with significantly different hr and lr projections
pp = pi+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

makeNicePlots(ax,0,1)

plt.subplots_adjust(hspace=.8, wspace=.3)

# Raw averages (normalized)
'''
plt.subplot(223)
plt.fill_between(time_aligned, tr1_raw_e_aligned_ave - tr1_raw_e_aligned_std, tr1_raw_e_aligned_ave + tr1_raw_e_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_raw_e_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_raw_e_aligned_ave - tr0_raw_e_aligned_std, tr0_raw_e_aligned_ave + tr0_raw_e_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_raw_e_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
plt.ylabel('Normed raw averages')
plt.title('Excitatory')
ax = plt.gca()
makeNicePlots(ax,1,1)



plt.subplot(224)
plt.fill_between(time_aligned, tr1_raw_i_aligned_ave - tr1_raw_i_aligned_std, tr1_raw_i_aligned_ave + tr1_raw_i_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_raw_i_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_raw_i_aligned_ave - tr0_raw_i_aligned_std, tr0_raw_i_aligned_ave + tr0_raw_i_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_raw_i_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
plt.title('Inhibitory')
plt.legend(loc='best', bbox_to_anchor=(1, .7), frameon=False)
#plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.2)

ax = plt.gca()
makeNicePlots(ax,1,1)
'''


#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffnei+'projTraces_svm_raw'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')


    
    


#%%
'''
#####################################################################################################################################################    
################# Excitatory vs inhibitory neurons:  c path #########################################################################################
######## Fraction of non-zero weights vs. different values of c #####################################################################################
#####################################################################################################################################################   
'''    
    
perActive_exc_ave_allDays = []
perActive_inh_ave_allDays = []
perActive_excInhDiff_rounds = []

excInhDiffAllC_alld = np.array(np.full((1,numDays),np.nan).squeeze(), dtype=list)
for iday in range(len(days)):
    
#    perActive_exc_ave_allDays = []
#    perActive_inh_ave_allDays = []
#    perActive_excInhDiff_rounds = []

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
    
    perActive_exc_rounds = []
    perActive_inh_rounds = []
    
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
            ##%% Load vars
            Data = scio.loadmat(svmName, variable_names=['perActive_exc', 'perActive_inh'])
            perActive_exc = Data.pop('perActive_exc') # numSamples x length(cvect_)
            perActive_inh = Data.pop('perActive_inh') # numSamples x length(cvect_)        
            
            exc_ave = np.mean(perActive_exc, axis=0)
            inh_ave = np.mean(perActive_inh, axis=0)

            # difference btwn exc and inh c path (after averaging across samples)
            perActive_excInhDiff_rounds.append(exc_ave - inh_ave) # numValidRounds x length(cvect_)         
            
            perActive_exc_rounds.append(exc_ave) # average across samples : numRounds x length(cvect_)
            perActive_inh_rounds.append(inh_ave) # average across samples : numRounds x length(cvect_)    
        
        
    # Compute average across rounds
    perActive_exc_ave = np.mean(perActive_exc_rounds, axis=0) # length(cvect_) x 1
    perActive_inh_ave = np.mean(perActive_inh_rounds, axis=0) # length(cvect_) x 1
    
    
    #%% Pool perActive_ave (average of percent non-zero neurons across all rounds) of all days
    
    perActive_exc_ave_allDays.append(perActive_exc_ave) # numDays x length(cvect_)
    perActive_inh_ave_allDays.append(perActive_inh_ave)
    
    
    #%%
    # pool exc-inh for all values of c of all rounds and all days # numDays x numValidRounds x length(cvect_)
    excInhDiffAllC = np.reshape(perActive_excInhDiff_rounds,(-1,)) 
    
    excInhDiffAllC_alld[iday] = excInhDiffAllC
        
        
        
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




