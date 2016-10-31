# -*- coding: utf-8 -*-
"""
Pool SVM results of all days

Created on Sun Oct 30 14:41:01 2016
@author: farznaj
"""

#%% 
mousename = 'fni17'

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.    
neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    

# Define days that you want to analyze
days = ['151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'];
numRounds = 10; # number of times svm analysis was ran for the same dataset but sampling different sets of neurons.    

fmt = 'eps' #'png', 'pdf': preserve transparency # Format of figures for saving
figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/' # Directory for saving figures.

#%%
from setImagingAnalysisNamesP import *
import os
import glob
import numpy as np   
import scipy as sci
import scipy.io as scio
from matplotlib import pyplot as plt

#%%
numDays = len(days);

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
    
    print(imfilename)        
    
    #%% loop over the 10 rounds of analysis for each day
    
    err_test_data_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    err_test_shfl_ave = (np.ones((1,numRounds))+np.nan)[0,:]
    
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
        #print '\n', svmName
        #np.shape(svmNowName)[0]==0    
        
        ##%% Load the class-loss array for the testing dataset (actual and shuffled) (length of array = number of samples, usually 100) 
        Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data', 'perClassErrorTest_shfl'])
        perClassErrorTest_data = Data.pop('perClassErrorTest_data')[0,:]
        perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')[0,:]
        perClassErrorTest_shfl.shape
        
        err_test_data_ave[i] = perClassErrorTest_data.mean()
        err_test_shfl_ave[i] = perClassErrorTest_shfl.mean()   
    
    # pool results of all rounds for days
    err_test_data_ave_allDays[:, iday] = err_test_data_ave
    err_test_shfl_ave_allDays[:, iday] = err_test_shfl_ave
    
    
    
#%%
ave_test_d = np.mean(err_test_data_ave_allDays, axis=0) # average across rounds
ave_test_s = np.mean(err_test_shfl_ave_allDays, axis=0) # average across rounds
sd_test_d = np.std(err_test_data_ave_allDays, axis=0) # std across roundss
sd_test_s = np.std(err_test_shfl_ave_allDays, axis=0) # std across rounds

   
#%% Average across rounds for each day
plt.figure()
plt.subplot(211)
plt.errorbar(range(numDays), ave_test_d, yerr = sd_test_d)
plt.errorbar(range(numDays), ave_test_s, yerr = sd_test_s)
plt.xlabel('Days')
plt.ylabel('Classification error (%) - testing data')
#plt.savefig('hi.eps')

#%% Average across days
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
d = os.path.join(figsDir, 'SVM')
if not os.path.exists(d):
    os.makedirs(d)

fign_ = 'classError'
fign = os.path.join(d, fign_+'.'+fmt)

plt.savefig(fign)
    


#%%
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
    
    
    
#%%
#########################################################################    
########################### Projection Traces ###########################     
#########################################################################
#%% Loop over days

trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        

tr1_ave_allDays = []
tr0_ave_allDays = []
tr1_std_allDays = []
tr0_std_allDays = []
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
    
    print(imfilename)        
    
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
        print 'training epoch is {} ms'.format(np.round((ep-eventI)*frameLength))
        
        
    
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
        
        ##%% Load vars (w, etc)
        Data = scio.loadmat(svmName, variable_names=['w', 'trsExcluded', 'NsExcluded', 'NsRand', 'meanX', 'stdX', 'ep_ms'])
        w = Data.pop('w')[0,:]    
        trsExcluded = Data.pop('trsExcluded')[0,:].astype('bool')
        NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
        NsRand = Data.pop('NsRand')[0,:].astype('bool')
        meanX = Data.pop('meanX')[0,:].astype('float')
        stdX = Data.pop('stdX')[0,:].astype('float')
        ep_ms = Data.pop('ep_ms')[0,:].astype('float')
        
        Y = choiceVec0[~trsExcluded];
        ep_ms_allDays[iday,:] = ep_ms
    
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
        
        
        #%% pool results of all rounds for days
        
        tr1_allRounds[:, i] = tr1  # frames x rounds
        tr0_allRounds[:, i] = tr0  # frames x rounds
      
        
    #%% Take average and std across rounds
        
    tr1_ave_allRounds = np.nanmean(tr1_allRounds, axis=1) # average across rounds # frames x 1
    tr0_ave_allRounds = np.nanmean(tr0_allRounds, axis=1) # average across rounds
    tr1_std_allRounds = np.nanstd(tr1_allRounds, axis=1) # std across roundss
    tr0_std_allRounds = np.nanstd(tr0_allRounds, axis=1) # std across rounds

          
    #%% Pool the averaged projection traces across all days

    tr1_ave_allDays.append(tr1_ave_allRounds)
    tr0_ave_allDays.append(tr0_ave_allRounds)
    tr1_std_allDays.append(tr1_std_allRounds)
    tr0_std_allDays.append(tr0_std_allRounds)    
        
        
#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(tr1_ave_allDays[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

frameLength = 1000/30.9; # sec.  # np.diff(time_aligned_stim)[0];

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

tr1_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
tr0_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    tr1_aligned[:, iday] = tr1_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    tr0_aligned[:, iday] = tr0_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]


#%% Average across days

tr1_aligned_ave = np.mean(tr1_aligned, axis=1)
tr0_aligned_ave = np.mean(tr0_aligned, axis=1)
tr1_aligned_std = np.std(tr1_aligned, axis=1)
tr0_aligned_std = np.std(tr0_aligned, axis=1)


#%% Plot the average projections across all days
#ep_ms_allDays
ax = plt.figure()

plt.fill_between(time_aligned, tr1_aligned_ave - tr1_aligned_std, tr1_aligned_ave + tr1_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_aligned_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_aligned_ave - tr0_aligned_std, tr0_aligned_ave + tr0_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_aligned_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
#plt.ylabel()
plt.legend()


#%% Save the figure

d = os.path.join(figsDir, 'SVM')
if not os.path.exists(d):
    os.makedirs(d)

fign_ = 'projTraces'
fign = os.path.join(d, fign_+'.'+fmt)

#ax.set_rasterized(True)
plt.savefig(fign)






#%%
############################################################################    
####################### Classification accuracy for all times ##############
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
    
    print(imfilename)        
    
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

frameLength = 1000/30.9; # sec.  # np.diff(time_aligned_stim)[0];

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

corrClass_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    corrClass_aligned[:, iday] = corrClass_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]


#%% Average across days

corrClass_aligned_ave = np.mean(corrClass_aligned, axis=1)
corrClass_aligned_std = np.std(corrClass_aligned, axis=1)


#%% Plot the average projections across all days
#ep_ms_allDays
ax = plt.figure()

plt.fill_between(time_aligned, corrClass_aligned_ave - corrClass_aligned_std, corrClass_aligned_ave + corrClass_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, corrClass_aligned_ave, 'b')

plt.xlabel('Time since stimulus onset (ms)')
plt.ylabel('classification accuracy (%)')
plt.legend()


#%% Save the figure
#fmt = 'png'
d = os.path.join(figsDir, 'SVM')
if not os.path.exists(d):
    os.makedirs(d)

fign_ = 'corrClassTrace'
fign = os.path.join(d, fign_+'.'+fmt)

#ax.set_rasterized(True)
plt.savefig(fign)
    



    
#%%




    