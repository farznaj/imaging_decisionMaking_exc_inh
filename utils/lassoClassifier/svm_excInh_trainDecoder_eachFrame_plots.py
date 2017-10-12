# -*- coding: utf-8 -*-
"""
Plot time course of class accuracy: svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     

#%% Change the following vars:

mousename = 'fni18' #'fni17'

savefigs = 0
doAllN = 1 # plot allN, instead of allExc
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
lastTimeBinMissed = 1 # if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
corrTrained = 1
doIncorr = 0
loadWeights = 0
    
if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
if mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   
    
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
#loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'

frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

#if loadInhAllexcEqexc==1:
dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/'
#else: # old svm files
#    dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/inhRois/'
if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'

execfile("defFuns.py")
execfile("svm_plots_setVars_n.py")  


   
#%% 
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
#####################################################################################################################################################
'''
            
#%% Loop over days    

eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
eventI_ds_allDays = np.full((len(days)), np.nan)
perClassErrorTest_data_inh_all = []
perClassErrorTest_shfl_inh_all = []
perClassErrorTest_chance_inh_all = []
perClassErrorTest_data_allExc_all = []
perClassErrorTest_shfl_allExc_all = []
perClassErrorTest_chance_allExc_all = []
perClassErrorTest_data_exc_all = []
perClassErrorTest_shfl_exc_all = []
perClassErrorTest_chance_exc_all = []
numInh = np.full((len(days)), np.nan)
numAllexc = np.full((len(days)), np.nan)
corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
  
for iday in range(len(days)):  #

    #%%            
    print '___________________'
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    ##%% Set .mat file names
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
    
    # from setImagingAnalysisNamesP import *
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))


    #%% Get number of hr, lr trials that were used for svm training
    
    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins)[0]   
    
    corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)

    corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
    
    
    #%% Load matlab vars to set eventI_ds (downsampled eventI)

    eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
    
    eventI_allDays[iday] = eventI
    eventI_ds_allDays[iday] = eventI_ds    

    
    #%% Load SVM vars

    perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN)
    
    ##%% Get number of inh and exc        
    if loadWeights==1:
        numInh[iday] = w_data_inh.shape[1]
        numAllexc[iday] = w_data_allExc.shape[1]
        
        
    #%% Once done with all frames, save vars for all days
    
    perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    
    perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
    perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
    perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    
    perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
    perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
    perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames    
    perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
    perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)

    # Delete vars before starting the next day    
    del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    if loadWeights==1:
        del w_data_inh, w_data_exc, w_data_allExc
        

eventI_allDays = eventI_allDays.astype(int)   
eventI_ds_allDays = eventI_ds_allDays.astype(int)
numExcSamples = perClassErrorTest_data_exc_all[iday].shape[0]
#perClassErrorTest_data_inh_all = np.array(perClassErrorTest_data_inh_all)
#perClassErrorTest_shfl_inh_all = np.array(perClassErrorTest_shfl_inh_all)
#perClassErrorTest_chance_inh_all = np.array(perClassErrorTest_chance_inh_all)
#perClassErrorTest_data_allExc_all = np.array(perClassErrorTest_data_allExc_all)
#perClassErrorTest_shfl_allExc_all = np.array(perClassErrorTest_shfl_allExc_all)
#perClassErrorTest_chance_allExc_all = np.array(perClassErrorTest_chance_allExc_all)
#perClassErrorTest_data_exc_all = np.array(perClassErrorTest_data_exc_all) # numShufflesExc x numSamples x numFrames
#perClassErrorTest_shfl_exc_all = np.array(perClassErrorTest_shfl_exc_all)
#perClassErrorTest_chance_exc_all = np.array(perClassErrorTest_chance_exc_all)


#%%    
######################################################################################################################################################    
######################################################################################################################################################          

#%% Average and st error of class accuracies across CV samples ... for each day

numD = len(eventI_allDays)
numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]

#### inh
av_test_data_inh = np.array([100-np.nanmean(perClassErrorTest_data_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
sd_test_data_inh = np.array([np.nanstd(perClassErrorTest_data_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

av_test_shfl_inh = np.array([100-np.nanmean(perClassErrorTest_shfl_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
sd_test_shfl_inh = np.array([np.nanstd(perClassErrorTest_shfl_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

av_test_chance_inh = np.array([100-np.nanmean(perClassErrorTest_chance_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
sd_test_chance_inh = np.array([np.nanstd(perClassErrorTest_chance_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  


#### exc (average across cv samples and exc shuffles)
av_test_data_exc = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
sd_test_data_exc = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  

av_test_shfl_exc = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
sd_test_shfl_exc = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  

av_test_chance_exc = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
sd_test_chance_exc = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  


#### allExc
av_test_data_allExc = np.array([100-np.nanmean(perClassErrorTest_data_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
sd_test_data_allExc = np.array([np.nanstd(perClassErrorTest_data_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

av_test_shfl_allExc = np.array([100-np.nanmean(perClassErrorTest_shfl_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
sd_test_shfl_allExc = np.array([np.nanstd(perClassErrorTest_shfl_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

av_test_chance_allExc = np.array([100-np.nanmean(perClassErrorTest_chance_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
sd_test_chance_allExc = np.array([np.nanstd(perClassErrorTest_chance_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

# below wont work if different days have different number of frames
'''
av_test_data_inh = 100-np.mean(perClassErrorTest_data_inh_all, axis=1) # average across cv samples
sd_test_data_inh = np.std(perClassErrorTest_data_inh_all, axis=1) / np.sqrt(numSamples)
av_test_shfl_inh = 100-np.mean(perClassErrorTest_shfl_inh_all, axis=1)
sd_test_shfl_inh = np.std(perClassErrorTest_shfl_inh_all, axis=1) / np.sqrt(numSamples)
av_test_chance_inh = 100-np.mean(perClassErrorTest_chance_inh_all, axis=1)
sd_test_chance_inh = np.std(perClassErrorTest_chance_inh_all, axis=1) / np.sqrt(numSamples)

av_test_data_allExc = 100-np.mean(perClassErrorTest_data_allExc_all, axis=1) # average across cv samples
sd_test_data_allExc = np.std(perClassErrorTest_data_allExc_all, axis=1) / np.sqrt(numSamples)
av_test_shfl_allExc = 100-np.mean(perClassErrorTest_shfl_allExc_all, axis=1)
sd_test_shfl_allExc = np.std(perClassErrorTest_shfl_allExc_all, axis=1) / np.sqrt(numSamples)
av_test_chance_allExc = 100-np.mean(perClassErrorTest_chance_allExc_all, axis=1)
sd_test_chance_allExc = np.std(perClassErrorTest_chance_allExc_all, axis=1) / np.sqrt(numSamples)

av_test_data_exc = 100-np.mean(perClassErrorTest_data_exc_all, axis=(1,2)) # average across cv samples and excShuffles
sd_test_data_exc = np.std(perClassErrorTest_data_exc_all, axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
av_test_shfl_exc = 100-np.mean(perClassErrorTest_shfl_exc_all, axis=(1,2))
sd_test_shfl_exc = np.std(perClassErrorTest_shfl_exc_all, axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
av_test_chance_exc = 100-np.mean(perClassErrorTest_chance_exc_all, axis=(1,2))
sd_test_chance_exc = np.std(perClassErrorTest_chance_exc_all, axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
'''
#_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl, nan_policy = 'omit')
#p


#%% Keep vars for chAl and stAl

if chAl==1:
    
    eventI_allDays_ch = eventI_allDays + 0
    
    av_test_data_inh_ch =  av_test_data_inh + 0
    sd_test_data_inh_ch = sd_test_data_inh + 0 
    av_test_shfl_inh_ch = av_test_shfl_inh + 0
    sd_test_shfl_inh_ch = sd_test_shfl_inh + 0
    av_test_chance_inh_ch = av_test_chance_inh + 0
    sd_test_chance_inh_ch = sd_test_chance_inh + 0
    
    av_test_data_allExc_ch = av_test_data_allExc + 0
    sd_test_data_allExc_ch = sd_test_data_allExc + 0
    av_test_shfl_allExc_ch = av_test_shfl_allExc + 0
    sd_test_shfl_allExc_ch = sd_test_shfl_allExc + 0
    av_test_chance_allExc_ch = av_test_chance_allExc + 0
    sd_test_chance_allExc_ch = sd_test_chance_allExc + 0
    
    av_test_data_exc_ch = av_test_data_exc + 0
    sd_test_data_exc_ch = sd_test_data_exc + 0
    av_test_shfl_exc_ch = av_test_shfl_exc + 0
    sd_test_shfl_exc_ch = sd_test_shfl_exc + 0
    av_test_chance_exc_ch = av_test_chance_exc + 0
    sd_test_chance_exc_ch = sd_test_chance_exc + 0
        
else:        
    eventI_allDays_st = eventI_allDays + 0    
    av_l2_train_d_st = av_l2_train_d + 0
    sd_l2_train_d_st = sd_l2_train_d + 0    
    av_l2_test_d_st = av_l2_test_d + 0
    sd_l2_test_d_st = sd_l2_test_d + 0    
    av_l2_test_s_st = av_l2_test_s + 0
    sd_l2_test_s_st = sd_l2_test_s + 0    
    av_l2_test_c_st = av_l2_test_c + 0
    sd_l2_test_c_st = sd_l2_test_c + 0
    


#%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.

# th for min number of trs of each class
'''
thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
thIncorr = 4 #5
'''
mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.

print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)

print np.array(days)[mn_corr < thTrained]




#%%
##################################################################################################
############## Align class accur traces of all days to make a final average trace ##############
################################################################################################## 
  
##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (len(av_test_data_inh_ch[iday]) - eventI_ds_allDays[iday] - 1)

nPreMin = min(eventI_ds_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

'''
if corrTrained==0: # remember below has issues...
    # old svm files were ran using the old downsampling method
    a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
    b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
    time_aligned = np.concatenate((a,b))
else:
'''
totLen = nPreMin + nPostMin +1
time_aligned = set_time_al(totLen, min(eventI_allDays), lastTimeBinMissed)

print time_aligned




#%% Align traces of all days on the common eventI (use nan for days with few trianed svm trials)
# not including days with too few svm trained trials.

def alTrace(trace, eventI_ds_allDays, nPreMin, nPostMin):    

    trace_aligned = np.ones((nPreMin + nPostMin + 1, trace.shape[0])) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
    for iday in range(trace.shape[0]):
        if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
            trace_aligned[:, iday] = trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
    return trace_aligned
    

av_test_data_inh_aligned = alTrace(av_test_data_inh, eventI_ds_allDays, nPreMin, nPostMin)
av_test_shfl_inh_aligned = alTrace(av_test_shfl_inh, eventI_ds_allDays, nPreMin, nPostMin)
av_test_chance_inh_aligned = alTrace(av_test_chance_inh, eventI_ds_allDays, nPreMin, nPostMin)

av_test_data_exc_aligned = alTrace(av_test_data_exc, eventI_ds_allDays, nPreMin, nPostMin)
av_test_shfl_exc_aligned = alTrace(av_test_shfl_exc, eventI_ds_allDays, nPreMin, nPostMin)
av_test_chance_exc_aligned = alTrace(av_test_chance_exc, eventI_ds_allDays, nPreMin, nPostMin)

av_test_data_allExc_aligned = alTrace(av_test_data_allExc, eventI_ds_allDays, nPreMin, nPostMin)
av_test_shfl_allExc_aligned = alTrace(av_test_shfl_allExc, eventI_ds_allDays, nPreMin, nPostMin)
av_test_chance_allExc_aligned = alTrace(av_test_chance_allExc, eventI_ds_allDays, nPreMin, nPostMin)


##%% Average and STD across days (each day includes the average class accuracy across samples.)

av_av_test_data_inh_aligned = np.nanmean(av_test_data_inh_aligned, axis=1)
sd_av_test_data_inh_aligned = np.nanstd(av_test_data_inh_aligned, axis=1)

av_av_test_shfl_inh_aligned = np.nanmean(av_test_shfl_inh_aligned, axis=1)
sd_av_test_shfl_inh_aligned = np.nanstd(av_test_shfl_inh_aligned, axis=1)

av_av_test_chance_inh_aligned = np.nanmean(av_test_chance_inh_aligned, axis=1)
sd_av_test_chance_inh_aligned = np.nanstd(av_test_chance_inh_aligned, axis=1)


av_av_test_data_exc_aligned = np.nanmean(av_test_data_exc_aligned, axis=1)
sd_av_test_data_exc_aligned = np.nanstd(av_test_data_exc_aligned, axis=1)

av_av_test_shfl_exc_aligned = np.nanmean(av_test_shfl_exc_aligned, axis=1)
sd_av_test_shfl_exc_aligned = np.nanstd(av_test_shfl_exc_aligned, axis=1)

av_av_test_chance_exc_aligned = np.nanmean(av_test_chance_exc_aligned, axis=1)
sd_av_test_chance_exc_aligned = np.nanstd(av_test_chance_exc_aligned, axis=1)


av_av_test_data_allExc_aligned = np.nanmean(av_test_data_allExc_aligned, axis=1)
sd_av_test_data_allExc_aligned = np.nanstd(av_test_data_allExc_aligned, axis=1)

av_av_test_shfl_allExc_aligned = np.nanmean(av_test_shfl_allExc_aligned, axis=1)
sd_av_test_shfl_allExc_aligned = np.nanstd(av_test_shfl_allExc_aligned, axis=1)

av_av_test_chance_allExc_aligned = np.nanmean(av_test_chance_allExc_aligned, axis=1)
sd_av_test_chance_allExc_aligned = np.nanstd(av_test_chance_allExc_aligned, axis=1)


#_,pcorrtrace0 = stats.ttest_1samp(av_l2_test_d_aligned.transpose(), 50) # p value of class accuracy being different from 50

_,pcorrtrace = stats.ttest_ind(av_test_data_exc_aligned.transpose(), av_test_data_inh_aligned.transpose()) # p value of class accuracy being different from 50
        
        

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
######################## PLOTS ########################

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')


#%% Compare number of inh vs exc for each day

if loadWeights:
    plt.figure(figsize=(4.5,3)) #plt.figure()
    
    plt.plot(numInh, label='inh', color='r')
    plt.plot(numAllexc, label=labAll, color='k')
    if doAllN:
        plt.title('average: inh=%d allN=%d inh/exc=%.2f' %(np.round(numInh.mean()), np.round(numAllexc.mean()), np.mean(numInh) / np.mean(numAllexc)))
    else:
        plt.title('average: inh=%d allExc=%d inh/exc=%.2f' %(np.round(numInh.mean()), np.round(numAllexc.mean()), np.mean(numInh) / np.mean(numAllexc)))    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
    plt.xlabel('Days')
    plt.ylabel('# Neurons')
    plt.xticks(range(len(days)), [days[i][0:6] for i in range(len(days))], rotation='vertical')
    
    makeNicePlots(plt.gca(),1)
    
    
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' +nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    

    
#%% Plot class accur trace for all days (only use days with svm trained trials above thTrained) on top of each other

plt.figure()

for iday in range(len(days)):    

    if mn_corr[iday] >= thTrained:    
        totLen = len(av_test_data_inh_ch[iday])
        # old downsampling method.
        '''
        if corrTrained==0: # remember below has problems...but if downsampling in svm_eachFrame was done in the old way, you have to use below.
            nPre = eventI_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
            nPost = (len(av_test_data_inh_ch[iday]) - eventI_allDays[iday] - 1)
            
            a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
            b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
            time_al = np.concatenate((a,b))
        else:
        '''    
        time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)    
#        plt.figure()
        plt.subplot(221)
        plt.errorbar(time_al, av_test_data_exc_ch[iday], yerr = sd_test_data_exc_ch[iday], label='exc', color='b')
        plt.errorbar(time_al, av_test_data_inh_ch[iday], yerr = sd_test_data_inh_ch[iday], label='inh', color='r')    
        plt.errorbar(time_al, av_test_data_allExc_ch[iday], yerr = sd_test_data_allExc_ch[iday], label=labAll, color='k')
#        plt.title(days[iday])
    
        plt.subplot(222)
        plt.plot(time_al, av_test_shfl_exc_ch[iday], label=' ', color='b')        
        plt.plot(time_al, av_test_shfl_inh_ch[iday], label=' ', color='r')    
        plt.plot(time_al, av_test_shfl_allExc_ch[iday], label=' ', color='k')        
        
        
        plt.subplot(223)
        h0,=plt.plot(time_al, av_test_chance_exc_ch[iday], label='exc', color='b')        
        h1,=plt.plot(time_al, av_test_chance_inh_ch[iday], label='inh', color='r')    
        h2,=plt.plot(time_al, av_test_chance_allExc_ch[iday], label=labAll, color='k')        
        
    #    plt.subplot(223)
    #    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
    #    plt.show()

##%%
plt.subplot(221)
plt.title('testing')
if chAl==1:
    plt.xlabel('Time since choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time since stim onset (ms)', fontsize=13)
plt.ylabel('Classification accuracy (%)', fontsize=13)
ax = plt.gca()
makeNicePlots(ax,1)

plt.subplot(222)
plt.title('shfl')
ax = plt.gca()
makeNicePlots(ax,1)

plt.subplot(223)
plt.title('chance')
ax = plt.gca()
makeNicePlots(ax,1)
plt.legend(handles=[h0,h1,h2], loc='center left', bbox_to_anchor=(1, .7), frameon=False) 

#plt.show()
plt.subplots_adjust(hspace=0.75)


##%% Save the figure    
if savefigs:
    if chAl==1:
        dd = 'chAl_allDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    else:
        dd = 'stAl_allDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
 
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)

    

   
#%% Plot a few example sessions (exc, inh, allExc superimposed)
   
'''   
# fni17, 151015: example day, excShfl 3
####### pick an example exc shfl
iexcshfl = rng.permutation(perClassErrorTest_data_exc_all[iday].shape[0])[0]
print iexcshfl

av_test_data_exc_1shfl = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday][iexcshfl], axis=0,) for iday in range(numD)]) # numDays
sd_test_data_exc_1shfl = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday][iexcshfl], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

av_test_shfl_exc_1shfl = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday][iexcshfl], axis=0) for iday in range(numD)]) # numDays
sd_test_shfl_exc_1shfl = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday][iexcshfl], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  

av_test_chance_exc_1shfl = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday][iexcshfl], axis=0) for iday in range(numD)]) # numDays
sd_test_chance_exc_1shfl = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday][iexcshfl], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  


####### plot
for iday in range(len(days)): #[30,35]:#,43]:
    plt.figure() #(figsize=(4.5,3))
    totLen = len(av_test_data_inh[iday])
    time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)    
    #        plt.errorbar(time_al, av_test_data_exc_ch[iday], yerr = sd_test_data_exc_ch[iday], label='exc', color='b')
            
    #### testing data
    plt.subplot(221)
    # exc
    plt.fill_between(time_al, av_test_data_exc_1shfl[iday] - sd_test_data_exc_1shfl[iday], av_test_data_exc_1shfl[iday] + sd_test_data_exc_1shfl[iday], alpha=0.5, edgecolor='b', facecolor='b')
    plt.plot(time_al, av_test_data_exc_1shfl[iday], 'b', label='exc')
    # inh
    plt.fill_between(time_al, av_test_data_inh[iday] - sd_test_data_inh[iday], av_test_data_inh[iday] + sd_test_data_inh[iday], alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_al, av_test_data_inh[iday], 'r', label='inh')
    # allExc
    plt.fill_between(time_al, av_test_data_allExc[iday] - sd_test_data_allExc[iday], av_test_data_allExc[iday] + sd_test_data_allExc[iday], alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_al, av_test_data_allExc[iday], 'k', label=labAll)
    
    if chAl==1:
        plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
    else:
        plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
    plt.ylabel('Classification accuracy (%)', fontsize=11)
    if superimpose==0:    
        plt.title('Testing data')
    #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
    #plt.legend()
    ax = plt.gca()
    makeNicePlots(ax,1,1)
    # Plot a dot for significant time points
    ymin, ymax = ax.get_ylim()
    #pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    #plt.plot(time_al, pp, color='k')
    
    
    #### shfl
    
    # exc
    plt.fill_between(time_al, av_test_shfl_exc_1shfl[iday] - sd_test_shfl_exc_1shfl[iday], av_test_shfl_exc_1shfl[iday] + sd_test_shfl_exc_1shfl[iday], alpha=0.3, edgecolor='b', facecolor='b')
    plt.plot(time_al, av_test_shfl_exc_1shfl[iday], 'b')
    # inh
    plt.fill_between(time_al, av_test_shfl_inh[iday] - sd_test_shfl_inh[iday], av_test_shfl_inh[iday] + sd_test_shfl_inh[iday], alpha=0.3, edgecolor='r', facecolor='r')
    plt.plot(time_al, av_test_shfl_inh[iday], 'r')
    # allExc
    plt.fill_between(time_al, av_test_shfl_allExc[iday] - sd_test_shfl_allExc[iday], av_test_shfl_allExc[iday] + sd_test_shfl_allExc[iday], alpha=0.3, edgecolor='k', facecolor='k')
    plt.plot(time_al, av_test_shfl_allExc[iday], 'k')
    
    ax = plt.gca()
    makeNicePlots(ax,1,1)
    
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 


    #xmin, xmax = ax.get_xlim()
    #plt.xlim([-1400,500])
    
    ##%% Save the figure    
    if savefigs:

        if chAl==1:
            dd = 'chAl_day' + days[iday][0:6] + '_exShfl' + str(iexcshfl) + '_' + nowStr
        else:
            dd = 'stAl_day' + days[iday][0:6] + '_exShfl' + str(iexcshfl) + '_' + nowStr
    
        if superimpose==1:        
            dd = dd+'_sup'
            
        d = os.path.join(svmdir+dnow+'CA_exampDay')
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    #    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+'svg')
    #    plt.savefig(fign, dpi=300, bbox_inches='tight') # , bbox_extra_artists=(lgd,)

'''

    
#%%       
#####################%% Plot the average of aligned traces across all days (exc, inh, allExc superimposed)
# only days with enough svm trained trials are used.
       
plt.figure() #(figsize=(4.5,3))

#### testing data
plt.subplot(221)
# exc
plt.fill_between(time_aligned, av_av_test_data_exc_aligned - sd_av_test_data_exc_aligned, av_av_test_data_exc_aligned + sd_av_test_data_exc_aligned, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, av_av_test_data_exc_aligned, 'b', label='exc')
# inh
plt.fill_between(time_aligned, av_av_test_data_inh_aligned - sd_av_test_data_inh_aligned, av_av_test_data_inh_aligned + sd_av_test_data_inh_aligned, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, av_av_test_data_inh_aligned, 'r', label='inh')
# allExc
plt.fill_between(time_aligned, av_av_test_data_allExc_aligned - sd_av_test_data_allExc_aligned, av_av_test_data_allExc_aligned + sd_av_test_data_allExc_aligned, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, av_av_test_data_allExc_aligned, 'k', label=labAll)

if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
else:
    plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
plt.ylabel('Classification accuracy (%)', fontsize=11)
if superimpose==0:    
    plt.title('Testing data')
#plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
#plt.legend()
ax = plt.gca()
makeNicePlots(ax,1,1)
# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()
pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')


#### shfl
if superimpose==0:
    plt.subplot(222)
# exc
plt.fill_between(time_aligned, av_av_test_shfl_exc_aligned - sd_av_test_shfl_exc_aligned, av_av_test_shfl_exc_aligned + sd_av_test_shfl_exc_aligned, alpha=0.3, edgecolor='b', facecolor='b')
plt.plot(time_aligned, av_av_test_shfl_exc_aligned, 'b')
# inh
plt.fill_between(time_aligned, av_av_test_shfl_inh_aligned - sd_av_test_shfl_inh_aligned, av_av_test_shfl_inh_aligned + sd_av_test_shfl_inh_aligned, alpha=0.3, edgecolor='r', facecolor='r')
plt.plot(time_aligned, av_av_test_shfl_inh_aligned, 'r')
# allExc
plt.fill_between(time_aligned, av_av_test_shfl_allExc_aligned - sd_av_test_shfl_allExc_aligned, av_av_test_shfl_allExc_aligned + sd_av_test_shfl_allExc_aligned, alpha=0.3, edgecolor='k', facecolor='k')
plt.plot(time_aligned, av_av_test_shfl_allExc_aligned, 'k')
if superimpose==0:    
    plt.title('Shuffled Y')
ax = plt.gca()
makeNicePlots(ax,1,1)


#### chance
if superimpose==0:    
    plt.subplot(223)
    # exc
    plt.fill_between(time_aligned, av_av_test_chance_exc_aligned - sd_av_test_chance_exc_aligned, av_av_test_chance_exc_aligned + sd_av_test_chance_exc_aligned, alpha=0.5, edgecolor='b', facecolor='b')
    plt.plot(time_aligned, av_av_test_chance_exc_aligned, 'b')
    # inh
    plt.fill_between(time_aligned, av_av_test_chance_inh_aligned - sd_av_test_chance_inh_aligned, av_av_test_chance_inh_aligned + sd_av_test_chance_inh_aligned, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned, av_av_test_chance_inh_aligned, 'r')
    # allExc
    plt.fill_between(time_aligned, av_av_test_chance_allExc_aligned - sd_av_test_chance_allExc_aligned, av_av_test_chance_allExc_aligned + sd_av_test_chance_allExc_aligned, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_aligned, av_av_test_chance_allExc_aligned, 'k')
    plt.title('Chance Y')
    ax = plt.gca()
    makeNicePlots(ax,1,1)

    plt.subplots_adjust(hspace=0.65)
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 


#xmin, xmax = ax.get_xlim()
#plt.xlim([-1400,500])

##%% Save the figure    
if savefigs:
    if chAl==1:
        dd = 'chAl_aveDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    else:
        dd = 'stAl_aveDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr

    if superimpose==1:        
        dd = dd+'_sup'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
 
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
#    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+'svg')
#    plt.savefig(fign, dpi=300, bbox_inches='tight') # , bbox_extra_artists=(lgd,)






##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% # For each day plot dist of event times (match it with class accur traces). Do it for both stim-aligned and choice-aligned data 
# For this section you need to run eventTimesDist.py; also run the codes above once with chAl=0 and once with chAl=1.
"""
if 'eventI_allDays_ch' in locals() and 'eventI_allDays_st' in locals():
    execfile("eventTimesDist.py")
    
    
    for iday in range(len(days)):    
    
        #%%
        # relative to stim onset
        timeStimOffset0_all_relOn = np.nanmean(timeStimOffset0_all[iday] - timeStimOnset_all[iday])
        timeStimOffset_all_relOn = np.nanmean(timeStimOffset_all[iday] - timeStimOnset_all[iday])
        timeCommitCL_CR_Gotone_all_relOn = np.nanmean(timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday])
        time1stSideTry_all_relOn = np.nanmean(time1stSideTry_all[iday] - timeStimOnset_all[iday])
    
        # relative to choice onset
        timeStimOnset_all_relCh = np.nanmean(timeStimOnset_all[iday] - time1stSideTry_all[iday])    
        timeStimOffset0_all_relCh = np.nanmean(timeStimOffset0_all[iday] - time1stSideTry_all[iday])
        timeStimOffset_all_relCh = np.nanmean(timeStimOffset_all[iday] - time1stSideTry_all[iday])
        timeCommitCL_CR_Gotone_all_relCh = np.nanmean(timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday])
        
        
        #%% Plot class accur trace for each day
        
        plt.figure()
        
        ######## stAl
        colors = 'k','r','b','m'
        nPre = eventI_allDays_st[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d_st[iday]) - eventI_allDays_st[iday] - 1)
        
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
        plt.subplot(223)
        plt.errorbar(time_al, av_l2_test_d_st[iday], yerr = sd_l2_test_d_st[iday])    
    #    plt.title(days[iday])
        #plt.errorbar(range(len(av_l2_test_d[iday])), av_l2_test_d[iday], yerr = sd_l2_test_d[iday])
    
        # mark event times relative to stim onset
        plt.plot([0, 0], [50, 100], color='g')
        plt.plot([timeStimOffset0_all_relOn, timeStimOffset0_all_relOn], [50, 100], color=colors[0])
        plt.plot([timeStimOffset_all_relOn, timeStimOffset_all_relOn], [50, 100], color=colors[1])
        plt.plot([timeCommitCL_CR_Gotone_all_relOn, timeCommitCL_CR_Gotone_all_relOn], [50, 100], color=colors[2])
        plt.plot([time1stSideTry_all_relOn, time1stSideTry_all_relOn], [50, 100], color=colors[3])
        plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('Classification accuracy (%)') #, fontsize=13)
        
        ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
        
        
        ###### Plot hist of event times
        stimOffset0_st = timeStimOffset0_all[iday] - timeStimOnset_all[iday]
        stimOffset_st = timeStimOffset_all[iday] - timeStimOnset_all[iday]
        goTone_st = timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday]
        sideTry_st = time1stSideTry_all[iday] - timeStimOnset_all[iday]
    
        stimOffset0_st = stimOffset0_st[~np.isnan(stimOffset0_st)]
        stimOffset_st = stimOffset_st[~np.isnan(stimOffset_st)]
        goTone_st = goTone_st[~np.isnan(goTone_st)]
        sideTry_st = sideTry_st[~np.isnan(sideTry_st)]
        
        labs = 'stimOffset_1rep', 'stimOffset', 'goTone', '1stSideTry'
        a = stimOffset0_st, stimOffset_st, goTone_st, sideTry_st
        binEvery = 100
        
        bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
        bn[-1] = np.max(a) # unlike digitize, histogram doesn't count the right most value
        
        plt.subplot(221)
        # set hists
        for i in range(len(a)):
            hist, bin_edges = np.histogram(a[i], bins=bn)
        #    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
            hb = plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 
        
    #    plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('Number of trials')
        plt.legend(handles=[hb], loc='center left', bbox_to_anchor=(.7, .7)) 
        
        plt.title(days[iday])
        plt.xlim(xl)    
        ax = plt.gca(); makeNicePlots(ax,1)
    
    
    
        #%% chAl
        colors = 'g','k','r','b'
        nPre = eventI_allDays_ch[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d_ch[iday]) - eventI_allDays_ch[iday] - 1)
        
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
        plt.subplot(224)
        plt.errorbar(time_al, av_l2_test_d_ch[iday], yerr = sd_l2_test_d_ch[iday])    
    #    plt.title(days[iday])
    
        # mark event times relative to choice onset
        plt.plot([timeStimOnset_all_relCh, timeStimOnset_all_relCh], [50, 100], color=colors[0])    
        plt.plot([timeStimOffset0_all_relCh, timeStimOffset0_all_relCh], [50, 100], color=colors[1])
        plt.plot([timeStimOffset_all_relCh, timeStimOffset_all_relCh], [50, 100], color=colors[2])
        plt.plot([timeCommitCL_CR_Gotone_all_relCh, timeCommitCL_CR_Gotone_all_relCh], [50, 100], color=colors[3])
        plt.plot([0, 0], [50, 100], color='m')
        plt.xlabel('Time relative to choice onset (ms)')
        ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
        
        
        ###### Plot hist of event times
        stimOnset_ch = timeStimOnset_all[iday] - time1stSideTry_all[iday]
        stimOffset0_ch = timeStimOffset0_all[iday] - time1stSideTry_all[iday]
        stimOffset_ch = timeStimOffset_all[iday] - time1stSideTry_all[iday]
        goTone_ch = timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday]
        
        stimOnset_ch = stimOnset_ch[~np.isnan(stimOnset_ch)]
        stimOffset0_ch = stimOffset0_ch[~np.isnan(stimOffset0_ch)]
        stimOffset_ch = stimOffset_ch[~np.isnan(stimOffset_ch)]
        goTone_ch = goTone_ch[~np.isnan(goTone_ch)]    
        
        labs = 'stimOnset','stimOffset_1rep', 'stimOffset', 'goTone'
        a = stimOnset_ch, stimOffset0_ch, stimOffset_ch, goTone_ch
        binEvery = 100
        
        bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
        bn[-1] = np.max(a) # unlike digitize, histogram doesn't count the right most value
        
        plt.subplot(222)
        # set hists
        for i in range(len(a)):
            hist, bin_edges = np.histogram(a[i], bins=bn)
        #    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
            plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 
        
    #    plt.xlabel('Time relative to choice onset (ms)')
    #    plt.ylabel('Number of trials')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
        
        plt.xlim(xl)    
        ax = plt.gca(); makeNicePlots(ax,1)
        
        plt.subplots_adjust(wspace=1, hspace=.5)
        
        
        '''
        plt.subplot(223)
        a = timeStimOffset0_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='b')
        
        a = timeStimOffset_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='r')
    
        a = timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='k')
    
        a = time1stSideTry_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='m')
        
        plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('# trials')           
        '''
    
        
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
