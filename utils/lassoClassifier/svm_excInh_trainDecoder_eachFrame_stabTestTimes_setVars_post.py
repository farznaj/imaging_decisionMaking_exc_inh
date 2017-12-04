# -*- coding: utf-8 -*-
"""
# vars are generated in svm_excInh_trainDecoder_eachFrame_plots_stabTestTimes_setVars.py
# here we do the next set of postporc (aligning, replacing the diagonal with cross-validated data)

Created on Mon Dec  4 10:53:56 2017

@author: farznaj
"""

#%%

mice = 'fni16', 'fni17', 'fni18', 'fni19'

saveDir_allMice = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice'

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

perClassErrorTest_data_inh_allMice = []
perClassErrorTest_shfl_inh_allMice = []
perClassErrorTest_data_allExc_allMice = []
perClassErrorTest_shfl_allExc_allMice = []
perClassErrorTest_data_exc_allMice = []
perClassErrorTest_shfl_exc_allMice = []

nowStr_allMice= []
eventI_ds_allMice = []
eventI_ds_allDays_allMice = []

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

    Data = scio.loadmat(stabTestDecodeName)#, variable_names=['eventI_ds_allDays'])

    eventI_ds_allDays = Data.pop('eventI_ds_allDays').flatten().astype(int)
#    eventI_ds_allDays_allMice.append(eventI_ds_allDays)    
    
    classErr_allN_allDays = Data.pop('classErr_allN_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_allDays = Data.pop('classErr_inh_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_allDays = Data.pop('classErr_exc_allDays') # nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    # shfl : class accuracies when trial labels were shuffled:
    classErr_allN_shfl_allDays = Data.pop('classErr_allN_shfl_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_shfl_allDays = Data.pop('classErr_inh_shfl_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_shfl_allDays = Data.pop('classErr_exc_shfl_allDays') # nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    
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
        

        
########################################################################################################

#%%        
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')
   
     
#%% 
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
    

# change the size of _exc arrays to : # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps x nSamps (otherwise they give error when trying to save them as mat file, bc it tries to convert them to an array since the first dimentions is 50 in all of them, but it cannot... so we move the excSamps dimention to the end!)
classAcc_exc_allDays_alig_allMice = [np.transpose(classAcc_exc_allDays_alig_allMice[im], (1,2,3,0,4)) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps x nSamps 
classAcc_exc_shfl_allDays_alig_allMice = [np.transpose(classAcc_exc_shfl_allDays_alig_allMice[im], (1,2,3,0,4)) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps x nSamps


#%% Set averages across samples for each day (of each mouse)

################################################################################################
################################################################################################


classAcc_allN_allDays_alig_avSamps_allMice = []
classAcc_inh_allDays_alig_avSamps_allMice = []
classAcc_exc_allDays_alig_avSamps_allMice = []

classAcc_allN_shfl_allDays_alig_avSamps_allMice = []
classAcc_inh_shfl_allDays_alig_avSamps_allMice = []
classAcc_exc_shfl_allDays_alig_avSamps_allMice = []

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_avSamps_allMice.append(np.mean(classAcc_allN_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_avSamps_allMice.append(np.mean(classAcc_inh_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_avSamps_allMice.append(np.mean(classAcc_exc_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps
    # shfl
    classAcc_allN_shfl_allDays_alig_avSamps_allMice.append(np.mean(classAcc_allN_shfl_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_avSamps_allMice.append(np.mean(classAcc_inh_shfl_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_avSamps_allMice.append(np.mean(classAcc_exc_shfl_allDays_alig_allMice[im], axis=-1)) # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps    


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
    classAcc_exc_allDays_alig_sdSamps_allMice.append(np.std(classAcc_exc_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps
    # shfl
    classAcc_allN_shfl_allDays_alig_sdSamps_allMice.append(np.std(classAcc_allN_shfl_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_sdSamps_allMice.append(np.std(classAcc_inh_shfl_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_sdSamps_allMice.append(np.std(classAcc_exc_shfl_allDays_alig_allMice[im], axis=-1) / np.sqrt(numSamps)) # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps



#%% Save the above vars

cnam = os.path.join(saveDir_allMice, 'stability_decoderTestedAllTimes_' + '_'.join(mice) + '_' + stabTestDecodeName[-17:-4] + '_' + nowStr)

scio.savemat(cnam, {'eventI_ds_allDays_allMice':eventI_ds_allDays_allMice,
                    'classAcc_allN_allDays_alig_allMice': classAcc_allN_allDays_alig_allMice, 
                    'classAcc_inh_allDays_alig_allMice': classAcc_inh_allDays_alig_allMice, 
                    'classAcc_exc_allDays_alig_allMice': classAcc_exc_allDays_alig_allMice, 
                    'classAcc_allN_shfl_allDays_alig_allMice': classAcc_allN_shfl_allDays_alig_allMice, 
                    'classAcc_inh_shfl_allDays_alig_allMice': classAcc_inh_shfl_allDays_alig_allMice, 
                    'classAcc_exc_shfl_allDays_alig_allMice': classAcc_exc_shfl_allDays_alig_allMice,              
                    'classAcc_allN_allDays_alig_avSamps_allMice': classAcc_allN_allDays_alig_avSamps_allMice,
                    'classAcc_inh_allDays_alig_avSamps_allMice': classAcc_inh_allDays_alig_avSamps_allMice,
                    'classAcc_exc_allDays_alig_avSamps_allMice': classAcc_exc_allDays_alig_avSamps_allMice,
                    'classAcc_allN_shfl_allDays_alig_avSamps_allMice': classAcc_allN_shfl_allDays_alig_avSamps_allMice,
                    'classAcc_inh_shfl_allDays_alig_avSamps_allMice': classAcc_inh_shfl_allDays_alig_avSamps_allMice,
                    'classAcc_exc_shfl_allDays_alig_avSamps_allMice': classAcc_exc_shfl_allDays_alig_avSamps_allMice,                    
                    'classAcc_allN_allDays_alig_sdSamps_allMice': classAcc_allN_allDays_alig_sdSamps_allMice,
                    'classAcc_inh_allDays_alig_sdSamps_allMice': classAcc_inh_allDays_alig_sdSamps_allMice,
                    'classAcc_exc_allDays_alig_sdSamps_allMice': classAcc_exc_allDays_alig_sdSamps_allMice,
                    'classAcc_allN_shfl_allDays_alig_sdSamps_allMice': classAcc_allN_shfl_allDays_alig_sdSamps_allMice,
                    'classAcc_inh_shfl_allDays_alig_sdSamps_allMice': classAcc_inh_shfl_allDays_alig_sdSamps_allMice,
                    'classAcc_exc_shfl_allDays_alig_sdSamps_allMice': classAcc_exc_shfl_allDays_alig_sdSamps_allMice})
    
