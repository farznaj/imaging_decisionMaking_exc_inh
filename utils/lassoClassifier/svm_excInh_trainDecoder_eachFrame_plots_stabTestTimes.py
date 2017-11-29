# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:15:20 2017

@author: farznaj
"""

#%%

mice = 'fni16', 'fni17', 'fni18', 'fni19'

savefigs = 0
corrTrained = 1
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
useEqualTrNums = 1
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")

regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.


#### vars needed for loading class accuracies of decoders trained and tested on the same time bin
loadWeights = 0
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
doAllN = 1 # plot allN, instead of allExc
doIncorr = 0
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'
if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'    
    
dir0 = '/stability_decoderTestedAllTimes/'
set_save_p_samps = 0 # set and save p value across samples per day (takes time to be done !!)


#%% Set days (all days and good days) for each mouse

days_allMice, numDaysAll, daysGood_allMice, dayinds_allMice, mn_corr_allMice = svm_setDays_allMice(mice, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, regressBins, useEqualTrNums, shflTrsEachNeuron, thTrained)

numDaysGood = np.array([len(daysGood_allMice[im]) for im in range(len(mice))])


#%%

classErr_allN_allDays_allMice = []
classErr_inh_allDays_allMice = []
classErr_exc_allDays_allMice = []

classErr_allN_shfl_allDays_allMice = []
classErr_inh_shfl_allDays_allMice = []
classErr_exc_shfl_allDays_allMice = []

nowStr_allMice= []
eventI_ds_allMice = []
eventI_ds_allDays_allMice = []

perClassErrorTest_data_inh_allMice = []
perClassErrorTest_shfl_inh_allMice = []
perClassErrorTest_data_allExc_allMice = []
perClassErrorTest_shfl_allExc_allMice = []
perClassErrorTest_data_exc_allMice = []
perClassErrorTest_shfl_exc_allMice = []


#%%    
for im in range(len(mice)):
        
    #%% Set svm_stab_testing mat file name

    mousename = mice[im] # mousename = 'fni16' #'fni17'    

    imagingDir = setImagingAnalysisNamesP(mousename)

    fname = os.path.join(imagingDir, 'analysis')    
    if not os.path.exists(fname):
        print 'creating folder'
        os.makedirs(fname)    
        
    finame = os.path.join(fname, 'svm_testEachDecoderOnAllTimes_[0-9]*.mat')

    stabTestDecodeName = glob.glob(finame)
    stabTestDecodeName = sorted(stabTestDecodeName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    stabTestDecodeName = stabTestDecodeName[0]
    print stabTestDecodeName
    
    nowStr = stabTestDecodeName[-17:-4]
    nowStr_allMice.append(nowStr)


    #%% load stab_testing vars

    Data = scio.loadmat(stabTestDecodeName)
    
    classErr_allN_allDays = Data.pop('classErr_allN_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_allDays = Data.pop('classErr_inh_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_allDays = Data.pop('classErr_exc_allDays') # nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    # shfl : class accuracies when trial labels were shuffled:
    classErr_allN_shfl_allDays = Data.pop('classErr_allN_shfl_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_shfl_allDays = Data.pop('classErr_inh_shfl_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_shfl_allDays = Data.pop('classErr_exc_shfl_allDays') # nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    
    eventI_ds_allDays = Data.pop('eventI_ds_allDays').flatten().astype(int)
#    eventI_allDays = Data.pop('eventI_allDays').flatten()
#    lastTimeBinMissed_allDays = Data.pop('lastTimeBinMissed_allDays').flatten()        
    
    ##%% Plot each day    
    '''
    iday = 0                               
    classErrAveSamps = np.mean(classErr_allN_allDays[iday], axis=1) # trainedFrs x testingFrs
#    classErrAveSamps = np.mean(classErr_allN_shfl_allDays[iday], axis=1) # trainedFrs x testingFrs
    
    nFrs = classErr_allN_allDays[iday].shape[-1]
    
    plt.figure()
    plt.imshow(classErrAveSamps); plt.colorbar()
    plt.axhline(eventI_ds_allDays[iday], 0, nFrs)
    plt.axvline(eventI_ds_allDays[iday], 0, nFrs)
    '''


    ############### ############### ############### ############### 
    #%% Keep vars of all mice

    classErr_allN_allDays_allMice.append(classErr_allN_allDays) # nMice # each mouse: nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_allDays_allMice.append(classErr_inh_allDays) # nMice # each mouse:  nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_allDays_allMice.append(classErr_exc_allDays) # nMice # each mouse:  nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)

    classErr_allN_shfl_allDays_allMice.append(classErr_allN_shfl_allDays) # nMice # each mouse: nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_shfl_allDays_allMice.append(classErr_inh_shfl_allDays) # nMice # each mouse:  nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_shfl_allDays_allMice.append(classErr_exc_shfl_allDays) # nMice # each mouse:  nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    
    eventI_ds_allDays_allMice.append(eventI_ds_allDays)
    
    

    #%%
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ################# Now get CA from decoder trained and tested on the same time bin #################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################

    perClassErrorTest_data_inh_all = []
    perClassErrorTest_shfl_inh_all = []
#    perClassErrorTest_chance_inh_all = []
    perClassErrorTest_data_allExc_all = []
    perClassErrorTest_shfl_allExc_all = []
#    perClassErrorTest_chance_allExc_all = []
    perClassErrorTest_data_exc_all = []
    perClassErrorTest_shfl_exc_all = []
#    perClassErrorTest_chance_exc_all = []
      
    for iday in range(numDaysAll[im]):  
    
        #%%            
        print '___________________'
        imagingFolder = days_allMice[im][iday][0:6]; #'151013'
        mdfFileNumber = map(int, (days_allMice[im][iday][7:]).split("-")); #[1,2] 
            
        ##%% Set .mat file names
        pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
        signalCh = [2] # since gcamp is channel 2, should be always 2.
        postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
        
        imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)       
#        postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
#        moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))        
        print(os.path.basename(imfilename))
    
    
        #%% Get number of hr, lr trials that were used for svm training
        '''
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
        
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)
    
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
        '''
        
        #%% Load matlab vars to set eventI_ds (downsampled eventI)
        '''
        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds    
        '''
        
        #%% Load SVM vars
    
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = \
            loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron)
        
        ##%% Get number of inh and exc        
#        if loadWeights==1:
#            numInh[iday] = w_data_inh.shape[1]
#            numAllexc[iday] = w_data_allExc.shape[1]
            
            
        #%% Keep vars for all days
        
        perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    
        perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
#        perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
        perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    
        perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
#        perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
        perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames    
        perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
#        perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)
    
        # Delete vars before starting the next day    
        del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
        if loadWeights==1:
            del w_data_inh, w_data_exc, w_data_allExc


    #%% Keep vars for all mice

    perClassErrorTest_data_inh_allMice.append(perClassErrorTest_data_inh_all) # nMice; each mouse: nDays; each day: samps x numFrs    
    perClassErrorTest_shfl_inh_allMice.append(perClassErrorTest_shfl_inh_all)
    perClassErrorTest_data_allExc_allMice.append(perClassErrorTest_data_allExc_all) # each day: samps x numFrs    
    perClassErrorTest_shfl_allExc_allMice.append(perClassErrorTest_shfl_allExc_all)
    perClassErrorTest_data_exc_allMice.append(perClassErrorTest_data_exc_all) # each day: numShufflesExc x numSamples x numFrames    
    perClassErrorTest_shfl_exc_allMice.append(perClassErrorTest_shfl_exc_all)
        

        
#%%
############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### 
############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### 
############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### 
############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### 

numExcSamps = classErr_exc_allDays[0].shape[0]
numSamps = classErr_inh_allDays[0].shape[1]


#%% For each mouse, align traces of all days, also replace the diagonal elements of matrices testTrainDiffFrs with testTrainSameFr

classAcc_allN_allDays_alig_allMice = []
classAcc_inh_allDays_alig_allMice = []
classAcc_exc_allDays_alig_allMice = []

classAcc_allN_shfl_allDays_alig_allMice = []
classAcc_inh_shfl_allDays_alig_allMice = []
classAcc_exc_shfl_allDays_alig_allMice = []

time_aligned_allMice = []
nPreMin_allMice = []   
 
for im in range(len(mice)):
    
    #%% Align class error traces of all samples and all days for each mouse    

    time_aligned, nPreMin, nPostMin = set_nprepost(classErr_allN_allDays_allMice[im], eventI_ds_allDays_allMice[im], mn_corr_allMice[im], thTrained, regressBins)
    time_aligned_allMice.append(time_aligned)
    nPreMin_allMice.append(nPreMin)
    
    # Align CA traces trained and tested on the same time point
    # output: nGoodDays x nAlignedFrs x nSamps
    perClassErrorTest_data_allN_alig = alTrace(perClassErrorTest_data_allExc_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=1)
    perClassErrorTest_shfl_allN_alig = alTrace(perClassErrorTest_shfl_allExc_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=1)
    # output: nGoodDays x nAlignedFrs x nSamps
    perClassErrorTest_data_inh_alig = alTrace(perClassErrorTest_data_inh_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=1)
    perClassErrorTest_shfl_inh_alig = alTrace(perClassErrorTest_shfl_inh_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=1)
    # output: nGoodDays x nAlignedFrs x numShufflesExc x nSamps
    perClassErrorTest_data_exc_alig = alTrace(perClassErrorTest_data_exc_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=2)
    perClassErrorTest_shfl_exc_alig = alTrace(perClassErrorTest_shfl_exc_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=2)
    
    # CA trained one time and tested all times   
    # output: nGoodDays x nTrainedFrs x nTestingFrs x nSamps
    classErr_allN_allDays_alig = alTrace_frfr(classErr_allN_allDays_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=[0,2])
    classErr_allN_shfl_allDays_alig = alTrace_frfr(classErr_allN_shfl_allDays_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=[0,2])
    # output: nGoodDays x nTrainedFrs x nTestingFrs x nSamps
    classErr_inh_allDays_alig = alTrace_frfr(classErr_inh_allDays_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=[0,2])
    classErr_inh_shfl_allDays_alig = alTrace_frfr(classErr_inh_shfl_allDays_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=[0,2])
    # output: nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs x nSamps    
    classErr_exc_allDays_alig = [alTrace_frfr(classErr_exc_allDays_allMice[im][:,isamp], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=[0,2]) for isamp in range(numExcSamps)]
    classErr_exc_shfl_allDays_alig = [alTrace_frfr(classErr_exc_shfl_allDays_allMice[im][:,isamp], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr_allMice[im], thTrained, frsDim=[0,2]) for isamp in range(numExcSamps)]
    
    
    #%% Replace the diagonal of class error matrix in testTrainDiffFrs with testTrainSameFr
    # Now you can compare each row of matrices testTrainDiffFrs (which shows how well each decoder does in all time points) with the diagonal element on that row (which shows how well the decoder does on its trained frame)
    
    for iday in range(numDaysGood[im]):
        for ifr in range(len(time_aligned)):
            
            classErr_allN_allDays_alig[iday][ifr,ifr,:] = perClassErrorTest_data_allN_alig[iday][ifr,:] # nSamps
            classErr_inh_allDays_alig[iday][ifr,ifr,:] = perClassErrorTest_data_inh_alig[iday][ifr,:]
            for iexc in range(numExcSamps):
                classErr_exc_allDays_alig[iexc][iday][ifr,ifr,:] = perClassErrorTest_data_exc_alig[iday][ifr,iexc,:]
    
            # shfl
            classErr_allN_shfl_allDays_alig[iday][ifr,ifr,:] = perClassErrorTest_shfl_allN_alig[iday][ifr,:] # nSamps
            classErr_inh_shfl_allDays_alig[iday][ifr,ifr,:] = perClassErrorTest_shfl_inh_alig[iday][ifr,:]
            for iexc in range(numExcSamps):
                classErr_exc_shfl_allDays_alig[iexc][iday][ifr,ifr,:] = perClassErrorTest_shfl_exc_alig[iday][ifr,iexc,:]
                
                
    #%% Subtract from 100, to trun class error to class accuracy   
    
    classAcc_allN_allDays_alig = 100 - np.array(classErr_allN_allDays_alig) # nGoodDays x nTrainedFrs x nTestingFrs x nSamps
    classAcc_inh_allDays_alig = 100 - np.array(classErr_inh_allDays_alig) # nGoodDays x nTrainedFrs x nTestingFrs x nSamps
    classAcc_exc_allDays_alig = 100 - np.array(classErr_exc_allDays_alig) # nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs x nSamps 
    # shfl
    classAcc_allN_shfl_allDays_alig = 100 - np.array(classErr_allN_shfl_allDays_alig) # nGoodDays x nTrainedFrs x nTestingFrs x nSamps
    classAcc_inh_shfl_allDays_alig = 100 - np.array(classErr_inh_shfl_allDays_alig) # nGoodDays x nTrainedFrs x nTestingFrs x nSamps
    classAcc_exc_shfl_allDays_alig = 100 - np.array(classErr_exc_shfl_allDays_alig) # nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs x nSamps 
    

    #%% Keep vars of all mice
    
    classAcc_allN_allDays_alig_allMice.append(classAcc_allN_allDays_alig)
    classAcc_inh_allDays_alig_allMice.append(classAcc_inh_allDays_alig)
    classAcc_exc_allDays_alig_allMice.append(classAcc_exc_allDays_alig)
    # shfl
    classAcc_allN_shfl_allDays_alig_allMice.append(classAcc_allN_shfl_allDays_alig)
    classAcc_inh_shfl_allDays_alig_allMice.append(classAcc_inh_shfl_allDays_alig)
    classAcc_exc_shfl_allDays_alig_allMice.append(classAcc_exc_shfl_allDays_alig)
    


#%%
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################

#%% Average across samples for each day (of each mouse)

classAcc_allN_allDays_alig_avSamps_allMice = []
classAcc_inh_allDays_alig_avSamps_allMice = []
classAcc_exc_allDays_alig_avSamps_allMice = []

classAcc_allN_shfl_allDays_alig_avSamps_allMice = []
classAcc_inh_shfl_allDays_alig_avSamps_allMice = []
classAcc_exc_shfl_allDays_alig_avSamps_allMice = []

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_avSamps_allMice.append(np.mean(classAcc_allN_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_avSamps_allMice.append(np.mean(classAcc_inh_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_avSamps_allMice.append(np.mean(classAcc_exc_allDays_alig_allMice[im], axis=-1)) # nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs
    # shfl
    classAcc_allN_shfl_allDays_alig_avSamps_allMice.append(np.mean(classAcc_allN_shfl_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_avSamps_allMice.append(np.mean(classAcc_inh_shfl_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_avSamps_allMice.append(np.mean(classAcc_exc_shfl_allDays_alig_allMice[im], axis=-1)) # nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs    


####%% exc: average across excSamps for each day (of each mouse)
classAcc_exc_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs
# shfl
classAcc_exc_shfl_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs


############## Standard error across days ############## 

classAcc_allN_allDays_alig_sdSamps_allMice = []
classAcc_inh_allDays_alig_sdSamps_allMice = []
classAcc_exc_allDays_alig_sdSamps_allMice = []

classAcc_allN_shfl_allDays_alig_sdSamps_allMice = []
classAcc_inh_shfl_allDays_alig_sdSamps_allMice = []
classAcc_exc_shfl_allDays_alig_sdSamps_allMice = []

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_sdSamps_allMice.append(np.std(classAcc_allN_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_sdSamps_allMice.append(np.std(classAcc_inh_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_sdSamps_allMice.append(np.std(classAcc_exc_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs
    # shfl
    classAcc_allN_shfl_allDays_alig_sdSamps_allMice.append(np.std(classAcc_allN_shfl_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_sdSamps_allMice.append(np.std(classAcc_inh_shfl_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_sdSamps_allMice.append(np.std(classAcc_exc_shfl_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nExcSamps x nGoodDays x nTrainedFrs x nTestingFrs    



#%% Average across days for each mouse

classAcc_allN_allDays_alig_avSamps_avDays_allMice = []
classAcc_inh_allDays_alig_avSamps_avDays_allMice = []
classAcc_exc_allDays_alig_avSamps_avDays_allMice = []

classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice = []
classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice = []
classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice = []

#AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    # shfl
    classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    


######%% SD across days for each mouse

classAcc_allN_allDays_alig_avSamps_sdDays_allMice = []
classAcc_inh_allDays_alig_avSamps_sdDays_allMice = []
classAcc_exc_allDays_alig_avSamps_sdDays_allMice = []

classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice = []
classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice = []
classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice = []

#AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    # shfl
    classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs




#%%
###############################################################################      
###############################################################################


#%% Plot CA tested at all time points

def plotAngInhExcAllN(classAcc_inh_allDays_alig_avSamps_avDays_allMice, classAcc_exc_allDays_alig_avSamps_avDays_allMice, classAcc_allN_allDays_alig_avSamps_avDays_allMice, plot_CA_diff_fractDiff, cblab012, do4, doCA=0):
    #plot_CA_diff_fractDiff : 0--> plot CA; 1: plot diff in CA rel2 max; 2: plot fract change in max CA 
    
    #linstyl = '-','-.',':'
    cols = 'black', 'gray', 'silver'
    colsi = 'mediumblue', 'blue', 'cyan'
    colse = 'red', 'tomato', 'lightsalmon'


    for im in range(len(mice)):    
        
        # data class accuracies
        topi = classAcc_inh_allDays_alig_avSamps_avDays_allMice[im]    
        tope = classAcc_exc_allDays_alig_avSamps_avDays_allMice[im]    
        top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im]        
        cblab = cblab012[0] #'Class accuracy (%)'   
        cmap='jet' #    extent = setExtent_imshow(time_aligned)
        
        # shfl class accuracies
    #    topi = classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice[im]    
    #    tope = classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice[im]    
    #    top = classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice[im]        
        
        # p values
    #    topi = p_inh_allMice_01[im]    
    #    tope = p_exc_allMice_01[im]    
    #    top = p_allN_allMice_01[im]        
    
        
    #    topi = [(topi[ifr,ifr]-topi[:,ifr]) / topi[ifr,ifr] for ifr in range(len(top))]
    #    tope = [(tope[ifr,ifr]-tope[:,ifr]) / tope[ifr,ifr] for ifr in range(len(top))]
    #    top = [(top[ifr,ifr]-top[:,ifr]) / top[ifr,ifr] for ifr in range(len(top))]
#        cblab = 'Fract drop in class accuracy' 
        
#        topi = [(topi[ifr,ifr]-topi[:,ifr]) for ifr in range(len(top))] # CA(trained at ifr and tested at ifr, this is max CA for the decoder tested at ifr) - CA(tested at ifr but trained at a different time, we want to see how much )
#        tope = [(tope[ifr,ifr]-tope[:,ifr]) for ifr in range(len(top))]
#        top = [(top[ifr,ifr]-top[:,ifr]) for ifr in range(len(top))]
        

        # Change in class accuracy relative to max class accuracy (as a result of testing the decoder at a different time point) 
        if plot_CA_diff_fractDiff==1:
#            topi = np.full((topi.shape), np.nan) # nTrainedFrs x nTestingFrs ... how well can trainedFr decoder predict choice at testingFrs, relative to max (optimal) decode accuracy at those testingFrs which is obtained by training decoders at frs = testingFrs
            for ifr in range(len(top)):
                topi[:,ifr] = topi[:,ifr] - topi[ifr,ifr]  # topi_diff[ifr0, ifr]: how well decoder trained at ifr0 generalize to time ifr # should be same as: topi_diff[ifr,:] = topi[ifr,ifr] - topi[ifr,:]
                tope[:,ifr] = tope[:,ifr] - tope[ifr,ifr]
                top[:,ifr] = top[:,ifr] - top[ifr,ifr]
            cblab = cblab012[1] #'Class accuracy rel2 max (%)'   
            cmap='jet' #'jet_r'
        
        # Fraction of max class accuracy that is changed after testing the decoder at a different time point     
        elif plot_CA_diff_fractDiff==2:
#            topi = np.full((topi.shape), np.nan) # nTrainedFrs x nTestingFrs ... how well can trainedFr decoder predict choice at testingFrs, relative to max (optimal) decode accuracy at those testingFrs which is obtained by training decoders at frs = testingFrs
            for ifr in range(len(top)):
                topi[:,ifr] = (topi[:,ifr] - topi[ifr,ifr]) / topi[ifr,ifr] # topi_diff[ifr0, ifr]: how well decoder trained at ifr0 generalize to time ifr # should be same as: topi_diff[ifr,:] = topi[ifr,ifr] - topi[ifr,:]
                tope[:,ifr] = (tope[:,ifr] - tope[ifr,ifr]) / tope[ifr,ifr]
                top[:,ifr] = (top[:,ifr] - top[ifr,ifr]) / top[ifr,ifr]
            cblab = cblab012[2] #'Fract change in max class accuracy'   
            cmap='jet' #'jet_r' 
        
        '''    
        thdrop = .1
        topi = [x<=thdrop for x in topi]    
        tope = [x<=thdrop for x in tope]    
        top = [x<=thdrop for x in top]    
        cblab = '% drop in class accuracy < 0.1'   
        cmap='jet'
        '''
        
        ##############################
        cmin = np.nanmin(np.concatenate((topi,tope)))
        cmax = np.nanmax(np.concatenate((topi,tope)))
    
        time_aligned = time_aligned_allMice[im]
        nPreMin = nPreMin_allMice[im]
    
       
        plt.figure(figsize=(6,6))    
        
        
        ################# inh
        plt.subplot(221)
     
        lab = 'inh' 
    #    cblab = '' # 'Class accuracy (%)'    
        yl = 'Decoder trained at t (ms)'
        xl = 'Decoder tested at t (ms)'  
        
        plotAng(topi, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
    
    
        ############### exc
        plt.subplot(223)
     
        lab = 'exc' 
    #    cblab = 'Class accuracy (%)'    
        yl = 'Decoder trained at t (ms)'
        xl = 'Decoder tested at t (ms)'  
        
        plotAng(tope, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
    
    
        ################# allN
        plt.subplot(222)
     
        lab = 'allN'    
        cmin = np.nanmin(top)
        cmax = np.nanmax(top)
    #    cblab = 'Class accuracy (%)'    
        yl = '' #'Decoder trained at t (ms)'
        xl = 'Decoder tested at t (ms)'  
        
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    

        # Mark a few time points (their CA will be plotted in subplot 224)
        if doCA:
            cnt = -1
            for fr2an in [nPreMin-4, nPreMin-1, nPreMin+2]:               
                cnt = cnt+1
                plt.axhline(time_aligned[fr2an],np.min(time_aligned),np.max(time_aligned), color=cols[cnt])
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
                
            
            
        #################### allN: show example traces: decoders trained at a few time points and tested on all other times
        if do4: 
            plt.subplot(224)
            if doCA:
                top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im] # nTrainedFrs x nTestingFrs   # plot the actual class accuracies instead of the drop in CA
                topsd = classAcc_allN_allDays_alig_avSamps_sdDays_allMice[im] / np.sqrt(numDaysGood[im])
                cnt = -1    
                for fr2an in [nPreMin-4, nPreMin+2, nPreMin-1]:       
                    cnt = cnt+1
                    # plot class accuracy tested at all times (x axis) when decoder was trained at time fr2an
                    
            #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.fill_between(time_aligned, top[fr2an] - topsd[fr2an], top[fr2an] + topsd[fr2an], color=cols[cnt])
            #        plt.plot(time_aligned, topi[fr2an], color=colsi[cnt])#, linestyle=linstyl[cnt])
            #        plt.plot(time_aligned, tope[fr2an], color=colse[cnt])#, linestyle=linstyl[cnt])        
                    plt.vlines(time_aligned[fr2an],np.min(top),np.max(top), color=cols[cnt])#, linestyle=linstyl[cnt])
                plt.xlim(xlim)
                plt.colorbar()
            
            else:
                plt.plot(time_aligned, stab_inh[im], color='r')
                plt.plot(time_aligned, stab_exc[im], color='b')
                plt.plot(time_aligned, stab_allN[im], color='k')          
                
            makeNicePlots(plt.gca(), 1, 1)        

        
        plt.subplots_adjust(hspace=.2, wspace=.5)
    

    
    
        ############%% Save figure for each mouse
    
        if savefigs:#% Save the figure
            mousename = mice[im]        
            dnow = dir0 + mousename + '/'
            days = daysGood_allMice[im]
            
            if chAl==1:
                dd = 'chAl_classAccur_testTrainDiffTimes_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr_allMice[im]
            else:
                dd = 'stAl_classAccur_testTrainDiffTimes_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr_allMice[im]
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




#%% Plot CA tested at all time points

plot_CA_diff_fractDiff = 1
cblab012 = 'Class accuracy (%)', 'Class accuracy rel2 max (%)', 'Fract change in max class accuracy' 
plotAngInhExcAllN(classAcc_inh_allDays_alig_avSamps_avDays_allMice, classAcc_exc_allDays_alig_avSamps_avDays_allMice, classAcc_allN_allDays_alig_avSamps_avDays_allMice, plot_CA_diff_fractDiff, cblab012, 1, 1)




#%%
###############################################################################
########### P values ###########
###############################################################################

#%% Load mat file: p value across samples per day (takes time to be done !!)
# the measure is OK except we dont know what to do with exc bc it has exc samps... ... there is no easy way to set a threshold for p value that works for both inh and exc... given that exc is averaged across exc samps!
# you can perhaps do some sort of bootstrap!

pnam = glob.glob(os.path.join(svmdir+dir0, '*pval_*_svm_testEachDecoderOnAllTimes*'))[0]
data = scio.loadmat(pnam)

p_inh_samps_allMice = data.pop('p_inh_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
p_exc_samps_allMice = data.pop('p_exc_samps_allMice').flatten() # nMice; each mouse: days x frs x frs x excSamps
p_allN_samps_allMice = data.pop('p_allN_samps_allMice').flatten() # nMice; each mouse: days x frs x frs


#%% Set p value (data vs shfl) for each pair of time points across days: across all days is CA of decoder trained at time t_i and tested at time t_j different from the trial-label shuffled case?

whichP = 2 # which statistical test to use: 0: wilcoxon, 1: MW, 2: ttest) 

p_inh_allMice = []
p_exc_allMice = []
p_allN_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    p_inh = np.full((nfrs,nfrs), np.nan)
    p_exc = np.full((nfrs,nfrs), np.nan)
    p_allN = np.full((nfrs,nfrs), np.nan)
    
    for ifr1 in range(nfrs):
        for ifr2 in range(nfrs):
            # inh
            a = classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]
            b = classAcc_inh_shfl_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]       
            p_inh[ifr1,ifr2] = pwm(a,b)[whichP]
    
            # exc
            a = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr2]
            b = classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr2]       
            p_exc[ifr1,ifr2] = pwm(a,b)[whichP]

            # allN
            a = classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]
            b = classAcc_allN_shfl_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]       
            p_allN[ifr1,ifr2] = pwm(a,b)[whichP]

            
    p_inh_allMice.append(p_inh)
    p_exc_allMice.append(p_exc)
    p_allN_allMice.append(p_allN)



plot_CA_diff_fractDiff = 0
cblab012 = 'P (data vs shfl)',
plotAngInhExcAllN(p_inh_allMice, p_exc_allMice, p_allN_allMice, plot_CA_diff_fractDiff, cblab012, 0)




################################################ 
# Now set the significancy matrix ... CA at which time points is siginificantly different from shuffle.
################################################

alph = .001 # p value for significancy

p_inh_allMice_01 = [x <= alph for x in p_inh_allMice]
p_exc_allMice_01 = [x <= alph for x in p_exc_allMice]
p_allN_allMice_01 = [x <= alph for x in p_allN_allMice]

plot_CA_diff_fractDiff = 0
cblab012 = 'Sig (data vs shfl)',
plotAngInhExcAllN(p_inh_allMice_01, p_exc_allMice_01, p_allN_allMice_01, plot_CA_diff_fractDiff, cblab012, 0)


#from itertools import compress
#list(compress(list_a, fil))

'''
a = np.mean(classAcc_inh_allDays_alig_allMice[im], axis=-1)
s = np.std(classAcc_inh_allDays_alig_allMice[im], axis=-1) / np.sqrt(numExcSamps)

aa = np.diagonal(a, axis1=1, axis2=2)
ss = np.diagonal(s, axis1=1, axis2=2)

ll = aa-ss
ul = aa+ss


plt.imshow(a[0]); plt.colorbar()
plt.plot(ul[0]); plt.plot(ll[0]); plt.plot(aa[0])
a[0][0,:]

# sig and above chance
'''


    
#%% Is CA(trained at t but tested at t') within 1 std of CA(trained and tested at t) 
    # do this across days (ie ave and std across days)

alph = .05 # if diagonal CA is sig (comapred to shfl), then see if it generalizes or not... othereise dont use that time point for analysis.

fractStd = 1

sameCA_inh_diag_allMice = []
sameCA_exc_diag_allMice = []
sameCA_allN_diag_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    sameCA_inh = np.full((nfrs,nfrs), np.nan) # if 1 then CA(trained at t but tested at t') < 1 std of CA(trained and tested at t) 
    sameCA_exc = np.full((nfrs,nfrs), np.nan)
    sameCA_allN = np.full((nfrs,nfrs), np.nan)
    
    for ifr1 in range(nfrs): # training time point
        
        # Set average and sd across days
        bi = np.mean(classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr1]), fractStd*np.std(classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr1]) # tested and trained on the same time point
        be = np.mean(classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr1]), fractStd*np.std(classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr1])               
        b = np.mean(classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr1]), fractStd*np.std(classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr1])       
        
        for ifr2 in range(nfrs): # testing time point
            # inh
            if p_inh_allMice[im][ifr1,ifr1] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                a = classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2].mean()           
                sameCA_inh[ifr1,ifr2] = (a>=bi[0]-bi[1]) #+ (a>=bi[0]+bi[1]) # sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(bi)[~np.isnan(bi)])[1]
#            sameCA_inh[ifr1,ifr2] = ttest2(a,bi,tail='left') # a < b
            
            # exc
            if p_exc_allMice[im][ifr1,ifr1] <= alph:
                a = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr2].mean()           
                sameCA_exc[ifr1,ifr2] = (a>=be[0]-be[1]) #+ (a>=be[0]+be[1]) # sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(be)[~np.isnan(be)])[1]
    #            sameCA_exc[ifr1,ifr2] = ttest2(a,be,tail='left') # a < b
            
            # allN
            if p_allN_allMice[im][ifr1,ifr1] <= alph:
                a = classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2].mean()                       
                sameCA_allN[ifr1,ifr2] = (a>=b[0]-b[1]) #+ (a>=b[0]+b[1]) # sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
    #            sameCA_allN[ifr1,ifr2] = ttest2(a,b,tail='left') # a < b
            
            
    sameCA_inh_diag_allMice.append(sameCA_inh)
    sameCA_exc_diag_allMice.append(sameCA_exc)
    sameCA_allN_diag_allMice.append(sameCA_allN)


#%% Plot of the above analysis

# For how long each decoder is stable (ie its CA when tested on a different time point is within 1 std)
stab_inh = regressBins*frameLength*np.array([np.sum(sameCA_inh_diag_allMice[im], axis=1) for im in range(len(mice))])
stab_exc = regressBins*frameLength*np.array([np.sum(sameCA_exc_diag_allMice[im], axis=1) for im in range(len(mice))])
stab_allN = regressBins*frameLength*np.array([np.sum(sameCA_allN_diag_allMice[im], axis=1) for im in range(len(mice))])


plot_CA_diff_fractDiff = 0
cblab012 = 'Diagonal decoder generalizes?',
plotAngInhExcAllN(sameCA_inh_diag_allMice, sameCA_exc_diag_allMice, sameCA_allN_diag_allMice, plot_CA_diff_fractDiff, cblab012,1)




#%% Same as above but done for each day, across samps... ie is CA at other times points similar (ie within mean - 1 se across samps) of the diagonal CA. This is done for each day, then an average is formed across days

#alph = .05 # if diagonal CA is sig (comapred to shfl), then see if it generalizes or not... othereise dont use that time point for analysis.
sameCA_inh_samps_allMice = []
sameCA_exc_samps_allMice = []
sameCA_allN_samps_allMice = []

for im in range(len(mice)):
#    print 'mouse ', im
    nfrs = len(time_aligned_allMice[im])    

    sameCA_inh_samps_allDays = []
    sameCA_exc_samps_allDays = []
    sameCA_allN_samps_allDays = []
        
    for iday in range(numDaysGood[im]):
#        print 'day ', iday
        sameCA_inh = np.full((nfrs,nfrs), np.nan)
        sameCA_exc = np.full((nfrs,nfrs,numExcSamps), np.nan)
        sameCA_allN = np.full((nfrs,nfrs), np.nan)
        
        for ifr1 in range(nfrs):
            
            bi = classAcc_inh_allDays_alig_avSamps_allMice[im][iday,ifr1,ifr1], fractStd*classAcc_inh_allDays_alig_sdSamps_allMice[im][iday,ifr1,ifr1]        # tested and trained on the same time point                
            b = classAcc_allN_allDays_alig_avSamps_allMice[im][iday,ifr1,ifr1], fractStd*classAcc_allN_allDays_alig_sdSamps_allMice[im][iday,ifr1,ifr1]       
            # do for each excSamp
            bea = np.full((2, numExcSamps), np.nan)
            for iexc in range(numExcSamps):
                bea[:, iexc] = classAcc_exc_allDays_alig_avSamps_allMice[im][iexc,iday,ifr1,ifr1], fractStd*classAcc_exc_allDays_alig_sdSamps_allMice[im][iexc,iday,ifr1,ifr1]
            # average (across excSamps) the ave and sd across samps
#            be = np.mean(bea, axis=1)
            
                
            for ifr2 in range(nfrs):                
                
                # inh
                if p_inh_samps_allMice[im][iday,ifr1,ifr1] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                    a = classAcc_inh_allDays_alig_avSamps_allMice[im][iday,ifr1,ifr2]
                    sameCA_inh[ifr1,ifr2] = (a>=bi[0]-bi[1]) 
                
                # allN
                if p_allN_samps_allMice[im][iday,ifr1,ifr1] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                    a = classAcc_allN_allDays_alig_avSamps_allMice[im][iday,ifr1,ifr2]
                    sameCA_allN[ifr1,ifr2] = (a>=bi[0]-bi[1]) 
                
                # exc ... compute p value across (trial) samps for each excSamp
                for iexc in range(numExcSamps):
                    if p_exc_samps_allMice[im][iday,ifr1,ifr1,iexc] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                        a = classAcc_exc_allDays_alig_avSamps_allMice[im][iexc,iday,ifr1,ifr2]                    
                        sameCA_exc[ifr1,ifr2,iexc] = (a>=bea[0,iexc]-bea[1,iexc])
    
    
        # keep vars of all days
        sameCA_inh_samps_allDays.append(sameCA_inh)
        sameCA_exc_samps_allDays.append(sameCA_exc)
        sameCA_allN_samps_allDays.append(sameCA_allN)
    
    # keep vars of all mice
    sameCA_inh_samps_allMice.append(sameCA_inh_samps_allDays)
    sameCA_exc_samps_allMice.append(sameCA_exc_samps_allDays)
    sameCA_allN_samps_allMice.append(sameCA_allN_samps_allDays)
    


#%% Average of above values (whether for each day diagonal CA generalizes to other times points or not) across days

sameCA_inh_samps_aveDays_allMice = [np.nanmean(sameCA_inh_samps_allMice[im],axis=0) for im in range(len(mice))]
sameCA_allN_samps_aveDays_allMice = [np.nanmean(sameCA_allN_samps_allMice[im],axis=0) for im in range(len(mice))]
# exc: average across days and exc samps
sameCA_exc_samps_aveDays_allMice = [np.nanmean(sameCA_exc_samps_allMice[im],axis=(0,-1)) for im in range(len(mice))]
# use one random exc samp
#r = rng.permutation(numExcSamps)[0]
#sameCA_exc_samps_aveDays_allMice = [np.nanmean(np.array(sameCA_exc_samps_allMice[im])[:,:,:,r],axis=0) for im in range(len(mice))]


# Average of fract days with stable decoders across time points (ie its CA when tested on a different time point is within 1 std)
#stab_inh = np.array([np.nanmean(sameCA_inh_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_exc = np.array([np.nanmean(sameCA_exc_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_allN = np.array([np.nanmean(sameCA_allN_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])


# Compute for each day, at each time point, the duration of stability (ie to how many other time points the decoder generalizes)

stab_inh_eachDay_allMice = []
stab_exc_eachDay_allMice = []
stab_allN_eachDay_allMice = []

for im in range(len(mice)):
    stab_inh_eachDay = []
    stab_exc_eachDay = []
    stab_allN_eachDay = []
    
    for iday in range(numDaysGood[im]):
        stab_inh_eachDay.append(np.sum(sameCA_inh_samps_allMice[im][iday], axis=1)) # days x trainedFrs        
        stab_allN_eachDay.append(np.sum(sameCA_allN_samps_allMice[im][iday], axis=1)) # days x trainedFrs        
        stab_exc_eachDay.append(np.sum(sameCA_exc_samps_allMice[im][iday], axis=1)) # days x trainedFrs x excSamps
        
    stab_inh_eachDay_allMice.append(stab_inh_eachDay)
    stab_allN_eachDay_allMice.append(stab_allN_eachDay)
    stab_exc_eachDay_allMice.append(stab_exc_eachDay)
    

'''
extent = setExtent_imshow(time_aligned_allMice[2])               
for iday in range(numDaysGood[2]): 
    a = sameCA_allN_samps_allMice[2][iday]; 
    plt.figure(); plt.imshow(a, extent=extent); plt.colorbar()

sameCA_allN_samps_aveDays_allMice[2] = np.mean(a, axis=0) #np.mean(sameCA_allN_samps_allMice[2][[0,1,2]],axis=0)
'''


#%% Plot of the above analysis

# On average across days, for how long each decoder generalizes.
stab_inh = regressBins*frameLength * np.array([np.nanmean(stab_inh_eachDay_allMice[im],axis=0) for im in range(len(mice))])
stab_allN = regressBins*frameLength * np.array([np.nanmean(stab_allN_eachDay_allMice[im],axis=0) for im in range(len(mice))])
# exc: average across days and exc samps
stab_exc = regressBins*frameLength * np.array([np.nanmean(stab_exc_eachDay_allMice[im],axis=(0,-1)) for im in range(len(mice))])
# use one random exc samp
#stab_exc = regressBins*frameLength * np.array([np.nanmean(np.array(stab_exc_eachDay_allMice[im])[:,:,r],axis=0) for im in range(len(mice))])


plot_CA_diff_fractDiff = 0
cblab012 = 'Fract days with generalized diagonal decoder',
plotAngInhExcAllN(sameCA_inh_samps_aveDays_allMice, sameCA_exc_samps_aveDays_allMice, sameCA_allN_samps_aveDays_allMice, plot_CA_diff_fractDiff, cblab012, 1)





#%% Do ttest between CA(trained and tested at t) and CA(trained at t but tested at t')
    # do this across days

p_inh_diag_allMice = []
p_exc_diag_allMice = []
p_allN_diag_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    p_inh = np.full((nfrs,nfrs), np.nan)
    p_exc = np.full((nfrs,nfrs), np.nan)
    p_allN = np.full((nfrs,nfrs), np.nan)
    
    for ifr1 in range(nfrs): # training time point

        # run p value against the CA(trained and tested at ifr1)
        bi = classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr1] # tested and trained on the same time point
        be = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr1]               
        b = classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr1]       
        
        for ifr2 in range(nfrs): # testing time point
            # inh
            if p_inh_allMice[im][ifr1,ifr1] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                
                a = classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]           
#                # we dont care if CA increases in another time compared to diagonal CA... I think this is bc diagonal CA has trial subselection as cross validation, but other times CA dont have that (since they are already cross validated by testing the decoder at other times)
#                m = a<=bi0
#                bi = bi0[m]
#                a = a[m]                

                p_inh[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(bi)[~np.isnan(bi)])[1]
    #            p_inh[ifr1,ifr2] = ttest2(a,bi,tail='left') # a < b
            
            # exc
            if p_exc_allMice[im][ifr1,ifr1] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.

                a = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr2]
#                # we dont care if CA increases in another time compared to diagonal CA... I think this is bc diagonal CA has trial subselection as cross validation, but other times CA dont have that (since they are already cross validated by testing the decoder at other times)
#                m = a<=be0
#                be = be0[m]
#                a = a[m]
                
                p_exc[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(be)[~np.isnan(be)])[1]
    #            p_exc[ifr1,ifr2] = ttest2(a,be,tail='left') # a < b
            
            # allN
            if p_allN_allMice[im][ifr1,ifr1] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                a = classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]            
#                # we dont care if CA increases in another time compared to diagonal CA... I think this is bc diagonal CA has trial subselection as cross validation, but other times CA dont have that (since they are already cross validated by testing the decoder at other times)
#                m = a<=b0
#                b = b0[m]
#                a = a[m]
                
                p_allN[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
    #            p_allN[ifr1,ifr2] = ttest2(a,b,tail='left') # a < b
            
            
    p_inh_diag_allMice.append(p_inh)
    p_exc_diag_allMice.append(p_exc)
    p_allN_diag_allMice.append(p_allN)
    


#%% Plot above computed p values

plot_CA_diff_fractDiff = 0
cblab012 = 'p (CA diagonal vs t)',
plotAngInhExcAllN(p_inh_diag_allMice, p_exc_diag_allMice, p_allN_diag_allMice, plot_CA_diff_fractDiff, cblab012, 0)


################################################ 
# Now set the significancy matrix ... 
################################################
alph = .05 # p value for significancy

p_inh_diag_allMice_01 = [x <= alph for x in p_inh_diag_allMice]
p_exc_diag_allMice_01 = [x <= alph for x in p_exc_diag_allMice]
p_allN_diag_allMice_01 = [x <= alph for x in p_allN_diag_allMice]


plot_CA_diff_fractDiff = 0
cblab012 = 'sig (CA diagonal vs t)',
plotAngInhExcAllN(p_inh_diag_allMice_01, p_exc_diag_allMice_01, p_allN_diag_allMice_01, plot_CA_diff_fractDiff, cblab012, 0)



#%% p value across samples per day (takes time to be done !!)
# it is fine except we dont know what to do with exc bc it has exc samps... ... there is no easy way to set a threshold for p value that works for both inh and exc... given that exc is averaged across exc samps!
# you can perhaps do some sort of bootstrap!

"""
if set_save_p_samps == 0 :
    pnam = glob.glob(os.path.join(svmdir+dir0, '*pval_*_svm_testEachDecoderOnAllTimes*'))[0]
    data = scio.loadmat(pnam)

    p_inh_samps_allMice = data.pop('p_inh_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
    p_exc_samps_allMice = data.pop('p_exc_samps_allMice').flatten() # nMice; each mouse: days x frs x frs x excSamps
    p_allN_samps_allMice = data.pop('p_allN_samps_allMice').flatten() # nMice; each mouse: days x frs x frs

else:
    p_inh_samps_allMice = []
    p_exc_samps_allMice = []
    p_allN_samps_allMice = []
    
    for im in range(len(mice)):
        print 'mouse ', im
        nfrs = len(time_aligned_allMice[im])    
    
        p_inh_samps_allDays = []
        p_exc_samps_allDays = []
        p_allN_samps_allDays = []
            
        for iday in range(numDaysGood[im]):
            print 'day ', iday
            p_inh = np.full((nfrs,nfrs), np.nan)
            p_exc = np.full((nfrs,nfrs,numExcSamps), np.nan)
            p_allN = np.full((nfrs,nfrs), np.nan)
            
            for ifr1 in range(nfrs):
                for ifr2 in range(nfrs):
                    
                    # inh
                    a = classAcc_inh_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    b = classAcc_inh_shfl_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    p_inh[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
            
                    # allN
                    a = classAcc_allN_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    b = classAcc_allN_shfl_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    p_allN[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
                    
                    # exc ... compute p value across (trial) samps for each excSamp
                    for iexc in range(numExcSamps):
    #                    print iexc
                        a = classAcc_exc_allDays_alig_allMice[im][iexc,iday,ifr1,ifr2,:]
                        b = classAcc_exc_shfl_allDays_alig_allMice[im][iexc,iday,ifr1,ifr2,:]
                        p_exc[ifr1,ifr2,iexc] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
        
            # keep vars of all days
            p_inh_samps_allDays.append(p_inh)
            p_exc_samps_allDays.append(p_exc)
            p_allN_samps_allDays.append(p_allN)
        
        # keep vars of all mice
        p_inh_samps_allMice.append(p_inh_samps_allDays)
        p_exc_samps_allMice.append(p_exc_samps_allDays)
        p_allN_samps_allMice.append(p_allN_samps_allDays)
                
    
    #from datetime import datetime
    #nowStr = datetime.now().strftime('%y%m%d-%H%M%S')            
    pnam = os.path.join(svmdir+dir0, 'pval_' + '_'.join(mice) + '_' + os.path.basename(stabTestDecodeName))
    scio.savemat(pnam, {'p_inh_samps_allMice':p_inh_samps_allMice, 'p_exc_samps_allMice':p_exc_samps_allMice, 'p_allN_samps_allMice':p_allN_samps_allMice})


#%%
### exc: median p value across excSamps
p_exc_samps2_allMice = [np.nanmedian(p_exc_samps_allMice[im], axis=-1) for im in range(len(mice))] # nMice; each mouse: days x frs x frs
# pick one random exc samp instead of above
iexc = rng.permutation(numExcSamps)[0]
p_exc_samps2_allMice = [p_exc_samps_allMice[im][:,:,:,iexc] for im in range(len(mice))]
    

### median of p value (computed across samps for each day) across days
p_allN_samps_avDays_allMice = [np.mean(p_allN_samps_allMice[im], axis=0) for im in range(len(mice))] # nMice; each mouse: frs x frs
p_inh_samps_avDays_allMice = [np.mean(p_inh_samps_allMice[im], axis=0) for im in range(len(mice))]
p_exc_samps_avDays_allMice = [np.mean(p_exc_samps2_allMice[im], axis=0) for im in range(len(mice))]

plt.figure(); plt.imshow(p_allN_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(p_inh_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(p_exc_samps_avDays_allMice[0]); plt.colorbar()



alph = .05

p_allN_samps_allMice_01 = [x <= alph for x in p_allN_samps_avDays_allMice]
p_inh_samps_allMice_01 = [x <= alph for x in p_inh_samps_avDays_allMice]
p_exc_samps_allMice_01 = [x <= alph for x in p_exc_samps_avDays_allMice]

plt.figure(); plt.imshow(p_allN_samps_allMice_01[0]); plt.colorbar()
plt.figure(); plt.imshow(p_inh_samps_allMice_01[0]); plt.colorbar()
plt.figure(); plt.imshow(p_exc_samps_allMice_01[0]); plt.colorbar()



### fraction of days with sig values of p for each pair of time points
# average sig value (computed across samps for each day) across days
sig_allN_samps_avDays_allMice = [np.nanmean(p_allN_samps_allMice[im]<=alph, axis=0) for im in range(len(mice))] # nMice; each mouse: frs x frs
sig_inh_samps_avDays_allMice = [np.nanmean(p_inh_samps_allMice[im]<=alph, axis=0) for im in range(len(mice))]
sig_exc_samps_avDays_allMice = [np.nanmean(p_exc_samps2_allMice[im]<=alph, axis=0) for im in range(len(mice))]

plt.figure(); plt.imshow(sig_allN_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(sig_inh_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(sig_exc_samps_avDays_allMice[0]); plt.colorbar()

# show time point in which >80% of the days are significant
for im in range(len(mice)):
    plt.figure(); plt.imshow(sig_allN_samps_avDays_allMice[im]>.9); plt.colorbar()
    plt.figure(); plt.imshow(sig_inh_samps_avDays_allMice[im]>.9); plt.colorbar()
    plt.figure(); plt.imshow(sig_exc_samps_avDays_allMice[im]>.9); plt.colorbar()




plt.imshow(p_allN_samps_allMice_01[0]); plt.colorbar()

for im in range(len(mice)):
    plt.figure(); plt.imshow(p_allN_allMice_01[im]); plt.colorbar()

"""


#%%
##compute only continues ones
#    
#stab = np.full(len(time_aligned), np.nan)    
#for fr2an in range(len(time_aligned)):
#    stab[fr2an] = sum(top[fr2an] >= (top[fr2an,fr2an] - topsd[fr2an,fr2an]))
#
#
#classAcc_allN_allDays_alig_avSamps_allMice[im]
#    
#    

#%%
"""
'''
for ifr in range(18):
    d = (top[ifr,ifr]-top[ifr,:])/top[ifr,ifr]; 
    plt.figure()
    plt.plot(d); 
    plt.vlines(ifr,d.min(),d.max())
    plt.hlines(.1,0,18)    
'''

di_allMice = []
de_allMice = []    
d_allMice = []    

for im in range(len(mice)):    
    
    # data class accuracies
    topi = classAcc_inh_allDays_alig_avSamps_avDays_allMice[im]    
    tope = classAcc_exc_allDays_alig_avSamps_avDays_allMice[im]    
    top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im]       
   
   
    di = [(topi[ifr,ifr]-topi[ifr,:]) / topi[ifr,ifr] for ifr in range(len(top))]
    de = [(tope[ifr,ifr]-tope[ifr,:]) / tope[ifr,ifr] for ifr in range(len(top))]
    d = [(top[ifr,ifr]-top[ifr,:]) / top[ifr,ifr] for ifr in range(len(top))]
    
    
    di_allMice.append(di)
    de_allMice.append(de)
    d_allMice.append(d)
    


plt.imshow(d_allMice, cmap='jet_r'); plt.colorbar(); 
"""

#%% Old stuff

'''
    #%% For each day average class accur across samps
    
    # subtract 100 to turn it into class accur
    classAcc_allN_aveSamps = 100-np.array([np.mean(classErr_allN_allDays_allMice[im][iday], axis=1) for iday in range(numDaysAll[im])]) # numDays; each day: trainedFrs x testingFrs
    classAcc_inh_aveSamps = 100-np.array([np.mean(classErr_inh_allDays_allMice[im][iday], axis=1) for iday in range(numDaysAll[im])]) # numDays; each day: trainedFrs x testingFrs
    classAcc_exc_aveSamps = 100-np.array([([np.mean(classErr_exc_allDays_allMice[im][iday][iexc], axis=1) for iexc in range(numExcSamps)]) for iday in range(numDaysAll[im])]) # numDays x nExcSamps; each day: trainedFrs x testingFrs



    ############### Align samp-averaged traces of all days ###############

    #%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(classAcc_allN_aveSamps, eventI_ds_allDays_allMice[im], mn_corr_allMice[im], thTrained, regressBins) # av_test_data_inh

    eventI_ds_allMice.append(nPreMin) # you need this var to align traces of all mice
    nFrs = len(time_aligned) #classAcc_allN_aveSamps_alig_avD.shape[0]
    
    
    #%% Align traces of all days

    classAcc_allN_aveSamps_alig = alTrace_frfr(classAcc_allN_aveSamps, eventI_ds_allDays_allMice[im], nPreMin, nPostMin) # days x trainedAlFrs x testedAlFrs
    classAcc_inh_aveSamps_alig = alTrace_frfr(classAcc_inh_aveSamps, eventI_ds_allDays_allMice[im], nPreMin, nPostMin) # days x trainedAlFrs x testedAlFrs
    classAcc_exc_aveSamps_alig = np.array([alTrace_frfr(classAcc_exc_aveSamps[:,iexc], eventI_ds_allDays_allMice[im], nPreMin, nPostMin) for iexc in range(numExcSamps)]) # excSamps x days x alFrs x alFrs
    
    
    #%% Set diagonal to nan (for the diagonal elements decoder is trained using 90% of the trials and tested on all trials... so they are not informative... accuracy is close to 90-100% for all time points)
    
    [np.fill_diagonal(classAcc_allN_aveSamps_alig[iday,:,:], np.nan) for iday in range(numDaysAll[im])];
    [np.fill_diagonal(classAcc_inh_aveSamps_alig[iday,:,:], np.nan) for iday in range(numDaysAll[im])];
    for iday in range(numDaysAll[im]):
        [np.fill_diagonal(classAcc_exc_aveSamps_alig[iexc,iday,:,:], np.nan) for iexc in range(numExcSamps)];

    
    #%% exc: average across exc samps for each day

    classAcc_exc_aveSamps_alig_avSamps = np.mean(classAcc_exc_aveSamps_alig, axis=0)   # days x trainedAlFrs x testedAlFrs
      
      
    #%% Average above aligned traces across days

    classAcc_allN_aveSamps_alig_avD = np.mean(classAcc_allN_aveSamps_alig, axis=0)   # trainedAlFrs x testedAlFrs
    classAcc_inh_aveSamps_alig_avD = np.mean(classAcc_inh_aveSamps_alig, axis=0)   # trainedAlFrs x testedAlFrs
    classAcc_exc_aveSamps_alig_avD = np.mean(classAcc_exc_aveSamps_alig_avSamps, axis=0) # already averaged across exc samps for each day  # trainedAlFrs x testedAlFrs
    
    
    #%% Plot CA tested at all time points
    
    plt.figure(figsize=(6,6))
    
    cmap='jet'
#    extent = setExtent_imshow(time_aligned)

    ##### inh
    plt.subplot(221)
 
    lab = 'inh' 
    top = classAcc_inh_aveSamps_alig_avD
    cmin = np.nanmin(top)
    cmax = np.nanmax(top)
    cblab = '' # 'Class accuracy (%)'    
    yl = 'Decoder trained at t (ms)'
    xl = 'Decoder tested at t (ms)'  
    
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    


    ##### exc
    plt.subplot(223)
 
    lab = 'exc' 
    top = classAcc_exc_aveSamps_alig_avD
    cmin = np.nanmin(top)
    cmax = np.nanmax(top)
    cblab = 'Class accuracy (%)'    
    yl = 'Decoder trained at t (ms)'
    xl = 'Decoder tested at t (ms)'  
    
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    



    ##### allN
    plt.subplot(222)
 
    lab = 'allN'
    top = classAcc_allN_aveSamps_alig_avD
    cmin = np.nanmin(top)
    cmax = np.nanmax(top)
    cblab = 'Class accuracy (%)'    
    yl = '' #'Decoder trained at t (ms)'
    xl = 'Decoder tested at t (ms)'  
    
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
    
    plt.subplots_adjust(hspace=.2, wspace=.5)
'''    





