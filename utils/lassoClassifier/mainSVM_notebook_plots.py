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

# Define days that you want to analyze
#days = ['151102_1-2', '151101_1', '151029_2-3', '151026_1', '151020_1-2', '151019_1-2', '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151007_1'];
days = ['151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'];
numRounds = 10; # number of times svm analysis was ran for the same dataset but sampling different sets of neurons.    
savefigs = True

fmt = 'eps' #'png', 'pdf': preserve transparency # Format of figures for saving
figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/' # Directory for saving figures.
neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    

 
#%%
import os
import glob
import numpy as np   
import scipy as sci
import scipy.io as scio
import scipy.stats as stats
from matplotlib import pyplot as plt
from setImagingAnalysisNamesP import *

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

if trialHistAnalysis:
    suffn = 'prev_'+itiName+'ITIs_'
else:
    suffn = 'curr_'   
    

days0 = days
numDays = len(days);


#%%
############################################################################    
########### Find days with all-0 weights for all rounds and exclude them ##############
############################################################################   

w_alln = [];
    
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
    
#    print(imfilename)   


    #%% loop over the 10 rounds of analysis for each day

    for i in range(numRounds):
        roundi = i+1
        
        if trialHistAnalysis:
#            ep_ms = np.round((ep-eventI)*frameLength)
#            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]

        if roundi==1:
            print os.path.basename(svmName)
            
        ##%% Load vars (w, etc)
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]    
        w = w/sci.linalg.norm(w); # normalzied weights
        w_alln.append(w)

w_alln_all = np.concatenate(w_alln, axis=0)


#%% Find days with all-0 weights (in all rounds)

b = np.reshape(w_alln, (numDays,numRounds))
#c = np.nansum(b,axis=1) # sum across rounds
#sumw = [np.nansum(i) for i in c]
sumw = [np.nansum(map(None, b[i,:]), axis=0).sum() for i in range(numDays)] # sum across rounds and neurons for each day
all0d = [not x for x in sumw]
print sum(all0d), 'days with all-0 weights: ', np.array(days)[np.array(all0d, dtype=bool)]
all0days = np.argwhere(all0d).flatten() # index of days with all-0 weights

# for each day find number of rounds without all-zero weights
nRoundsNonZeroW = [np.sum([x!=0 for x in np.nansum(map(None, b[i,:]), axis=1)]) for i in range(numDays)]  # /float(numRounds)
print 'number of rounds with non-zero weights for each day = ',  nRoundsNonZeroW


#%% Exclude days with all-0 weights (in all rounds) from analysis

print 'Excluding %d days from analysis because all SVM weights of all rounds are 0' %(sum(all0d))
days = np.delete(days0, all0days)
print 'days for analysis:', days
numDays = len(days)




#%%
############################################################################    
############################ Classification Error ##########################     
############################################################################

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
    
#    print(imfilename)        
    
    #%% loop over the 10 rounds of analysis for each day
    
    err_test_data_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    err_test_shfl_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    
    for i in range(numRounds):
        roundi = i+1
        
        if trialHistAnalysis:
#            ep_ms = np.round((ep-eventI)*frameLength)
#            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]
        #print '\n', svmName
        #np.shape(svmNowName)[0]==0    
        if roundi==1:
            print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if w.sum()==0:
            print 'In round %d all weights are 0 ... not analyzing' %(roundi)
            
        else:
        
            ##%% Load the class-loss array for the testing dataset (actual and shuffled) (length of array = number of samples, usually 100) 
            Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data', 'perClassErrorTest_shfl'])
            perClassErrorTest_data = Data.pop('perClassErrorTest_data')[0,:]
            perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')[0,:]
            perClassErrorTest_shfl.shape
            
            err_test_data_ave[i] = perClassErrorTest_data.mean()
            err_test_shfl_ave[i] = perClassErrorTest_shfl.mean()   
    
    # pool results of all rounds for days
    err_test_data_ave_allDays[:, iday] = err_test_data_ave # it will be nan for rounds with all-0 weights
    err_test_shfl_ave_allDays[:, iday] = err_test_shfl_ave
    
    
    
#%%
ave_test_d = np.nanmean(err_test_data_ave_allDays, axis=0) # average across rounds
ave_test_s = np.nanmean(err_test_shfl_ave_allDays, axis=0) # average across rounds
sd_test_d = np.nanstd(err_test_data_ave_allDays, axis=0) # std across roundss
sd_test_s = np.nanstd(err_test_shfl_ave_allDays, axis=0) # std across rounds

   
#%% Average across rounds for each day
plt.figure()
plt.subplot(211)
plt.errorbar(range(numDays), ave_test_d, yerr = sd_test_d)
plt.errorbar(range(numDays), ave_test_s, yerr = sd_test_s)
plt.xlabel('Days')
plt.ylabel('Classification error (%) - testing data')
#plt.savefig('hi.eps')

##%% Average across days
x =[0,1]
labels = ['data', 'shuffled']
#plt.figure()
plt.subplot(212)
plt.errorbar(x, [np.mean(ave_test_d), np.mean(ave_test_s)], yerr = [np.std(ave_test_d), np.std(ave_test_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels)    
plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    

    
#%% Save the figure
if savefigs:
    d = os.path.join(figsDir, 'SVM')
    if not os.path.exists(d):
        os.makedirs(d)
     
    fign_ = suffn+'classError'
    fign = os.path.join(d, fign_+'.'+fmt)
    
    plt.savefig(fign)
    


#%%
'''
#########################################################################    
########################### Get eventI for all days ###########################     
#########################################################################
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

    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
    eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    
    eventI_allDays[iday] = eventI
    
'''    
    
#%%
#########################################################################    
########################### Projection Traces ###########################     
#########################################################################
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

eventI_allDays = (np.ones((numDays,1))+np.nan).flatten().astype('int')
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
    
#    print(imfilename)        
    
    #%% Load outcomes
    
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    choiceVecAll = (Data.pop('allResp_HR_LR').astype('float'))[0,:]

    #%% Load vars (traces, etc)

    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
    eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
    time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
    DataS = Data
    
    eventI_allDays[iday] = eventI

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
    
    if trialHistAnalysis:
        # either of the two below (stimulus-aligned and initTone-aligned) would be fine
        # eventI = DataI['initToneAl'].eventI
        eventI = DataS['stimAl_allTrs'].eventI    
        epEnd = eventI + epEnd_rel2stimon_fr #- 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
        # epEnd = DataI['initToneAl'].eventI - 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
        ep = np.arange(epEnd+1)
#        print 'training epoch is {} ms'.format(np.round((ep-eventI)*frameLength))
        
        
    
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
        
        
        
    # Load outcomes and allResp_HR_LR
    # if trialHistAnalysis==0:
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    # choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    print '%d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))



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
        # Set to nan those trials in outcomes and allRes that are nan in traces_al_stim
        I = (np.argwhere((~np.isnan(traces_al_stim).sum(axis=0)).sum(axis=1)))[0][0] # first non-nan neuron
        allTrs2rmv = np.argwhere(sum(np.isnan(traces_al_stim[:,I,:])))
        print(np.shape(allTrs2rmv))
    
        outcomes[allTrs2rmv] = np.nan
        allResp_HR_LR[allTrs2rmv] = np.nan
        '''
        
    #%% Set choiceVec0 (Y)
    
    if trialHistAnalysis:
        # Load trialHistory structure
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
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
        
        if trialHistAnalysis:
            ep_ms = np.round((ep-eventI)*frameLength)
            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]
        
        if roundi==1:
            print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if w.sum()==0:
            print 'In round %d all weights are 0 ... not analyzing' %(roundi)
            
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


#%% Number of nan days

a = np.sum(tr1_aligned, axis=0)
np.isnan(a).sum()


#%% Average across days

tr1_aligned_ave = np.nanmean(tr1_aligned, axis=1)
tr0_aligned_ave = np.nanmean(tr0_aligned, axis=1)
tr1_aligned_std = np.nanstd(tr1_aligned, axis=1)
tr0_aligned_std = np.nanstd(tr0_aligned, axis=1)

tr1_raw_aligned_ave = np.nanmean(tr1_raw_aligned, axis=1)
tr0_raw_aligned_ave = np.nanmean(tr0_raw_aligned, axis=1)
tr1_raw_aligned_std = np.nanstd(tr1_raw_aligned, axis=1)
tr0_raw_aligned_std = np.nanstd(tr0_raw_aligned, axis=1)


#%% Plot the average projections across all days
#ep_ms_allDays
plt.figure()

plt.subplot(211)
plt.fill_between(time_aligned, tr1_aligned_ave - tr1_aligned_std, tr1_aligned_ave + tr1_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_aligned_ave - tr0_aligned_std, tr0_aligned_ave + tr0_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
plt.ylabel('SVM Projections')
#plt.legend()


plt.subplot(212)
plt.fill_between(time_aligned, tr1_raw_aligned_ave - tr1_raw_aligned_std, tr1_raw_aligned_ave + tr1_raw_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_raw_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_raw_aligned_ave - tr0_raw_aligned_std, tr0_raw_aligned_ave + tr0_raw_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_raw_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
plt.ylabel('Raw averages')
plt.legend(loc=0)
plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)

print 'ep_ms_allDays: \n', ep_ms_allDays


#%% Save the figure
if savefigs:
    fmt = 'png'
    d = os.path.join(figsDir, 'SVM')
    if not os.path.exists(d):
        os.makedirs(d)
    
    fign_ = suffn+'projTraces_svm_raw'
    fign = os.path.join(d, fign_+'.'+fmt)
    
    #ax.set_rasterized(True)
    plt.savefig(fign)






#%%
############################################################################    
#################### Classification accuracy at all times ##################
############################################################################

#%% Loop over days

corrClass_ave_allDays = []
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
    
#    print(imfilename)        
    
    #%% Load eventI (we need it for the final alignment of corrClass traces of all days)

    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
    eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    
    eventI_allDays[iday] = eventI
    
    #%% loop over the 10 rounds of analysis for each day
    
    corrClass_rounds = []
    
    for i in range(numRounds):
        roundi = i+1
        
        if trialHistAnalysis:
            ep_ms = np.round((ep-eventI)*frameLength)
            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]
        
        if roundi==1:
            print os.path.basename(svmName)        


        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if w.sum()==0:
            print 'In round %d all weights are 0 ... not analyzing' %(roundi)
            
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


#%% Average across days

corrClass_aligned_ave = np.mean(corrClass_aligned, axis=1) * 100
corrClass_aligned_std = np.std(corrClass_aligned, axis=1) * 100


#%% Plot the average traces across all days
#ep_ms_allDays
ax = plt.figure()

plt.fill_between(time_aligned, corrClass_aligned_ave - corrClass_aligned_std, corrClass_aligned_ave + corrClass_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, corrClass_aligned_ave, 'b')

plt.xlabel('Time since stimulus onset (ms)')
plt.ylabel('classification accuracy (%)')
plt.legend()


#%% Save the figure
if savefigs:
    #fmt = 'png'
    d = os.path.join(figsDir, 'SVM')
    if not os.path.exists(d):
        os.makedirs(d)
    
    fign_ = suffn+'corrClassTrace'
    fign = os.path.join(d, fign_+'.'+fmt)
    
    #ax.set_rasterized(True)
    plt.savefig(fign)
    



    
#%%
############################################################################    
############### Excitatory vs inhibitory neurons:  weights #################
############################################################################   

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
    
#    print(imfilename)   

   
   # Load inhibitRois
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
    # print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))


    #%% loop over the 10 rounds of analysis for each day

    for i in range(numRounds):
        roundi = i+1
        
        if trialHistAnalysis:
#            ep_ms = np.round((ep-eventI)*frameLength)
#            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]        

        if roundi==1:
            print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if w.sum()==0:
            print 'In round %d all weights are 0 ... not analyzing' %(roundi)
            
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
            w_inh.append(w[inhRois==1])
            w_exc.append(w[inhRois==0])
            w_uns.append(w[np.isnan(inhRois)])

            w_inh_fractNonZero.append(np.mean([w[inhRois==1]!=0]))
            w_exc_fractNonZero.append(np.mean([w[inhRois==0]!=0]))
            

#w_alln_all = np.concatenate(w_alln, axis=0)
w_exc_all = np.concatenate(w_exc, axis=0)
w_inh_all = np.concatenate(w_inh, axis=0)
w_uns_all = np.concatenate(w_uns, axis=0)

#w_inh_fractNonZero_all = np.concatenate(w_inh_fractNonZero, axis=0)
#w_exc_fractNonZero_all = np.concatenate(w_exc_fractNonZero, axis=0)

len_exc = np.shape(w_exc_all)[0]
len_inh = np.shape(w_inh_all)[0]
len_uns = np.shape(w_uns_all)[0]
#print 'length of w for exc = %d, inh = %d, unsure = %d' %(len_exc, len_inh, len_uns)


w_exc_all_o = w_exc_all+0;
w_inh_all_o = w_inh_all+0;

np.shape(w_exc_all_o), np.shape(w_inh_all_o)

w_exc_all = w_exc_all_o[w_exc_all_o!=0]
w_inh_all = w_exc_all_o[w_inh_all_o!=0]

np.shape(w_exc_all), np.shape(w_inh_all)


#%% Plot histogram of normalized weights for exc and inh neurons

'''
mxw = np.max(np.concatenate((w_exc_all, w_inh_all), axis=0))
mnw = np.min(np.concatenate((w_exc_all, w_inh_all), axis=0))
r = mxw - mnw
rr = np.arange(mnw, mxw, r/1000)
#rr = np.arange(-.5, .5, r/100)
plt.hist(w_exc_all, normed=True, cumulative=True, histtype='stepfilled', alpha = .5, color = 'b', label = 'excit');
'''
print 'length of w for exc = %d, inh = %d, unsure = %d' %(len_exc, len_inh, len_uns)
h, p = stats.ttest_ind(w_inh_all, w_exc_all)
print '\nmean(w): inhibit = %.3f  excit = %.3f' %(np.nanmean(w_inh_all), np.nanmean(w_exc_all))
print '\nmean(abs(w)): inhibit = %.3f  excit = %.3f' %(np.nanmean(abs(w_inh_all)), np.nanmean(abs(w_exc_all)))
print 'p-val_2tailed (inhibit vs excit weights) = %.2f' %p



plt.figure()
plt.subplot(211)
hist, bin_edges = np.histogram(w_exc_all[~np.isnan(w_exc_all)], bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='exc')

hist, bin_edges = np.histogram(w_inh_all[~np.isnan(w_inh_all)], bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='inh')

plt.legend(loc=0)
plt.xlabel('Normalized weight')
plt.ylabel('Normalized count')
#plt.ylim([-.1, 1.1])
#plt.title('w')
plt.title('mean(w): inhibit = %.3f  excit = %.3f\np-val_2tailed (inhibit vs excit weights) = %.2f ' %(np.mean((w_inh_all)), np.mean((w_exc_all)), p))


plt.subplot(212)
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

plt.legend(loc=0)
plt.xlabel('Normalized weight')
plt.ylabel('Normalized count')
#plt.ylim([-.1, 1.1])
plt.title('abs(w)')
plt.tight_layout()


#%% Fraction of non zero
#print 'length of w for exc = %d, inh = %d, unsure = %d' %(len_exc, len_inh, len_uns)
h, p = stats.ttest_ind(w_inh_fractNonZero, w_exc_fractNonZero)
print '\nmean(w): inhibit = %.3f  excit = %.3f' %(np.nanmean(w_inh_fractNonZero), np.nanmean(w_exc_fractNonZero))
#print '\nmean(abs(w)): inhibit = %.3f  excit = %.3f' %(np.nanmean(abs(w_inh_all)), np.nanmean(abs(w_exc_all)))
print 'p-val_2tailed (inhibit vs excit weights) = %.2f' %p



plt.figure()
plt.subplot(211)
hist, bin_edges = np.histogram(w_exc_fractNonZero, bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='exc')

hist, bin_edges = np.histogram(w_inh_fractNonZero, bins=100)
hist = hist/float(np.sum(hist))
#hist = np.cumsum(hist)
plt.plot(bin_edges[0:-1], hist, label='inh')

plt.legend(loc=0)
plt.xlabel('Normalized weight')
plt.ylabel('Normalized count')
#plt.ylim([-.1, 1.1])
#plt.title('w')
#plt.title('mean(w): inhibit = %.3f  excit = %.3f\np-val_2tailed (inhibit vs excit weights) = %.2f ' %(np.mean((w_inh_all)), np.mean((w_exc_all)), p))



#%% Save the figure
if savefigs:
    fmt = 'eps'
    d = os.path.join(figsDir, 'SVM')
    if not os.path.exists(d):
        os.makedirs(d)
    
    fign_ = suffn+'weights_inh_exc'
    fign = os.path.join(d, fign_+'.'+fmt)
    
    #ax.set_rasterized(True)
    plt.savefig(fign)





            
#%%
############################################################################    
################ Excitatory vs inhibitory neurons:  projections ##############
############################################################################   
    
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
    
eventI_allDays = (np.ones((numDays,1))+np.nan).flatten().astype('int')
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
    
#    print(imfilename)        
    
    #%% Load outcomes
    
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    choiceVecAll = (Data.pop('allResp_HR_LR').astype('float'))[0,:]

    #%% Load vars (traces, etc)

    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
    eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
    time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
    
    eventI_allDays[iday] = eventI

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
    
    if trialHistAnalysis:
        # either of the two below (stimulus-aligned and initTone-aligned) would be fine
        # eventI = DataI['initToneAl'].eventI
        eventI = DataS['stimAl_allTrs'].eventI    
        epEnd = eventI + epEnd_rel2stimon_fr #- 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
        # epEnd = DataI['initToneAl'].eventI - 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
        ep = np.arange(epEnd+1)
#        print 'training epoch is {} ms'.format(np.round((ep-eventI)*frameLength))
        
        
    
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
        
        
        
    # Load outcomes and allResp_HR_LR
    # if trialHistAnalysis==0:
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    # choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    print '%d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))



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
        # Set to nan those trials in outcomes and allRes that are nan in traces_al_stim
        I = (np.argwhere((~np.isnan(traces_al_stim).sum(axis=0)).sum(axis=1)))[0][0] # first non-nan neuron
        allTrs2rmv = np.argwhere(sum(np.isnan(traces_al_stim[:,I,:])))
        print(np.shape(allTrs2rmv))
    
        outcomes[allTrs2rmv] = np.nan
        allResp_HR_LR[allTrs2rmv] = np.nan
        '''
        
    #%% Set choiceVec0 (Y)
    
    if trialHistAnalysis:
        # Load trialHistory structure
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
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
        
        if trialHistAnalysis:
            ep_ms = np.round((ep-eventI)*frameLength)
            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]
        
        if roundi==1:
            print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if w.sum()==0:
            print 'In round %d all weights are 0 ... not analyzing' %(roundi)
            
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
        
        
            # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
            if neuronType==2:
                inhRois = inhibitRois[~NsExcluded]        
                inhRois = inhRois[NsRand]
    
    
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


#%% Plot the average projections across all days
#ep_ms_allDays
ax = plt.figure()

plt.subplot(221)
plt.fill_between(time_aligned, tr1_e_aligned_ave - tr1_e_aligned_std, tr1_e_aligned_ave + tr1_e_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_e_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_e_aligned_ave - tr0_e_aligned_std, tr0_e_aligned_ave + tr0_e_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_e_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
#plt.ylabel()
#plt.legend(loc='left', bbox_to_anchor=(1, .7))
plt.title('Excitatory projections')


plt.subplot(222)
plt.fill_between(time_aligned, tr1_i_aligned_ave - tr1_i_aligned_std, tr1_i_aligned_ave + tr1_i_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_i_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_i_aligned_ave - tr0_i_aligned_std, tr0_i_aligned_ave + tr0_i_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_i_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
#plt.ylabel()
plt.legend(loc='left', bbox_to_anchor=(1, .7))
plt.title('Inhibitory projections')



#ax = plt.figure()

plt.subplot(223)
plt.fill_between(time_aligned, tr1_raw_e_aligned_ave - tr1_raw_e_aligned_std, tr1_raw_e_aligned_ave + tr1_raw_e_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_raw_e_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_raw_e_aligned_ave - tr0_raw_e_aligned_std, tr0_raw_e_aligned_ave + tr0_raw_e_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_raw_e_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
#plt.ylabel()
#plt.legend(loc='left', bbox_to_anchor=(1, .7))
plt.title('Excitatory raw averages')


plt.subplot(224)
plt.fill_between(time_aligned, tr1_raw_i_aligned_ave - tr1_raw_i_aligned_std, tr1_raw_i_aligned_ave + tr1_raw_i_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_raw_i_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_raw_i_aligned_ave - tr0_raw_i_aligned_std, tr0_raw_i_aligned_ave + tr0_raw_i_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_raw_i_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
#plt.ylabel()
plt.legend(loc='left', bbox_to_anchor=(1, .7))
plt.title('Inhibitory raw averages')
plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)


#%% Save the figure
if savefigs:
    fmt = 'png'
    d = os.path.join(figsDir, 'SVM')
    if not os.path.exists(d):
        os.makedirs(d)
    
    fign_ = suffn+'projTraces_excInh_svm_raw'
    fign = os.path.join(d, fign_+'.'+fmt)
    
    #ax.set_rasterized(True)
    plt.savefig(fign)


    
    


#%%
############################################################################    
################# Excitatory vs inhibitory neurons:  c path ################
######## Fraction of non-zero weights vs. different values of c ############
############################################################################   
    
    
perActive_exc_ave_allDays = []
perActive_inh_ave_allDays = []
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
    
#    print(imfilename)
    
    
    #%% loop over the 10 rounds of analysis for each day
    
    perActive_exc_rounds = []
    perActive_inh_rounds = []
    
    for i in range(numRounds):
        roundi = i+1
        
        if trialHistAnalysis:
            ep_ms = np.round((ep-eventI)*frameLength)
            th_stim_dur = []
            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
        else:
            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
        
        svmn = svmn + pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))[0]
     
        if roundi==1:
            print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        if w.sum()==0:
            print 'In round %d all weights are 0 ... not analyzing' %(roundi)
            
        else:        
            ##%% Load corrClass
            Data = scio.loadmat(svmName, variable_names=['perActive_exc', 'perActive_inh'])
            perActive_exc = Data.pop('perActive_exc') # numSamples x length(cvect_)
            perActive_inh = Data.pop('perActive_inh') # numSamples x length(cvect_)        
            
            perActive_exc_rounds.append(np.mean(perActive_exc, axis=0)) # average across samples : numRounds x length(cvect_)
            perActive_inh_rounds.append(np.mean(perActive_inh, axis=0)) # average across samples : numRounds x length(cvect_)    
        
    # Compute average across rounds
    perActive_exc_ave = np.mean(perActive_exc_rounds, axis=0) # length(cvect_) x 1
    perActive_inh_ave = np.mean(perActive_inh_rounds, axis=0) # length(cvect_) x 1
    
    
    #%% Pool perActive_ave (average of percent non-zero neurons across all rounds) of all days
    
    perActive_exc_ave_allDays.append(perActive_exc_ave) # numDays x length(cvect_)
    perActive_inh_ave_allDays.append(perActive_inh_ave)



#%% Average and std across days

perActive_exc_ave_allDays_ave = np.mean(perActive_exc_ave_allDays, axis=0)
perActive_exc_ave_allDays_std = np.std(perActive_exc_ave_allDays, axis=0)

perActive_inh_ave_allDays_ave = np.mean(perActive_inh_ave_allDays, axis=0)
perActive_inh_ave_allDays_std = np.std(perActive_inh_ave_allDays, axis=0)


#%% Load cvect_

Data = scio.loadmat(svmName, variable_names=['cvect_'])
cvect_ = Data.pop('cvect_')[0,:] 


#%% Extend the built in two tailed ttest function to one-tailed
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

       
#%% Compute p value for exc vs inh c path (pooled across all c values)
aa = np.reshape(perActive_exc_ave_allDays,(-1,))
bb = np.reshape(perActive_inh_ave_allDays,(-1,))
#aa = np.array(perActive_exc).flatten()
#bb = np.array(perActive_inh).flatten()
'''
aa = aa[np.logical_and(aa>0 , aa<100)]
np.shape(aa)

bb = bb[np.logical_and(bb>0 , bb<100)]
np.shape(bb)
'''

h, p_two = stats.ttest_ind(aa, bb)
p_tl = ttest2(aa, bb, tail='left')
p_tr = ttest2(aa, bb, tail='right')

print '\np value (pooled for all values of c):\nexc ~= inh : %.2f\nexc < inh : %.2f\nexc > inh : %.2f' %(p_two, p_tl, p_tr)


#%% Plot the average c path across all days
#ep_ms_allDays
ax = plt.figure()

plt.fill_between(cvect_, perActive_exc_ave_allDays_ave - perActive_exc_ave_allDays_std, perActive_exc_ave_allDays_ave + perActive_exc_ave_allDays_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(cvect_, perActive_exc_ave_allDays_ave, 'b', label = 'excit')

plt.fill_between(cvect_, perActive_inh_ave_allDays_ave - perActive_inh_ave_allDays_std, perActive_inh_ave_allDays_ave + perActive_inh_ave_allDays_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(cvect_, perActive_inh_ave_allDays_ave, 'r', label = 'inhibit')

plt.ylim([-10,110])
plt.xscale('log')
plt.xlabel('c (inverse of regularization parameter)')
plt.ylabel('% non-zero weights')
plt.legend()
plt.legend(loc='left') #, bbox_to_anchor=(.4, .7))
plt.title('p value (pooled for all values of c):\nexc ~= inh : %.2f; exc < inh : %.2f; exc > inh : %.2f' %(p_two, p_tl, p_tr))


#%% Save the figure
if savefigs:
    fmt = 'png'
    d = os.path.join(figsDir, 'SVM')
    if not os.path.exists(d):
        os.makedirs(d)
    
    fign_ = suffn+'cPath_excInh'
    fign = os.path.join(d, fign_+'.'+fmt)
    
    #ax.set_rasterized(True)
    plt.savefig(fign)
 