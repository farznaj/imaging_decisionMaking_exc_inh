#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Load classAcc vars that were set in svm_excInh_trainDecoder_eachFrame_stabTestTimes_setVars_post.py and set their averages across days

Created on Mon Apr 30 22:26:04 2018
@author: farznaj
"""

#%% ############################ Load matlab vars ##############################
################################################################################
################################################################################

"""
if testIncorr:
    snn = 'decoderTestedIncorrTrs_'
else:
    snn = 'stability_decoderTestedAllTimes_'
"""    


##%% Load class accur vars (averages of samples)

cnam = glob.glob(os.path.join(saveDir_allMice, snn + nts + nw + '_'.join(mice) + '_*'))
cnam = sorted(cnam, key=os.path.getmtime) # sort pnevFileNames by date (descending)
cnam = cnam[::-1][0] # so the latest file is the 1st one.
print cnam

data = scio.loadmat(cnam, variable_names=['eventI_ds_allDays_allMice',
                    'classAcc_allN_allDays_alig_avSamps_allMice', 'classAcc_inh_allDays_alig_avSamps_allMice', 'classAcc_exc_allDays_alig_avSamps_allMice',
                    'classAcc_allN_shfl_allDays_alig_avSamps_allMice', 'classAcc_inh_shfl_allDays_alig_avSamps_allMice', 'classAcc_exc_shfl_allDays_alig_avSamps_allMice',
                    'classAcc_allN_allDays_alig_sdSamps_allMice', 'classAcc_inh_allDays_alig_sdSamps_allMice', 'classAcc_exc_allDays_alig_sdSamps_allMice',
                    'classAcc_allN_shfl_allDays_alig_sdSamps_allMice', 'classAcc_inh_shfl_allDays_alig_sdSamps_allMice', 'classAcc_exc_shfl_allDays_alig_sdSamps_allMice'])

                    
eventI_ds_allDays_allMice = data.pop('eventI_ds_allDays_allMice').flatten()                    
eventI_ds_allDays_allMice = [eventI_ds_allDays_allMice[im].flatten() for im in range(len(mice))]

# average of samps
classAcc_allN_allDays_alig_avSamps_allMice = data.pop('classAcc_allN_allDays_alig_avSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_avSamps_allMice = data.pop('classAcc_inh_allDays_alig_avSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_avSamps_allMice = data.pop('classAcc_exc_allDays_alig_avSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps
classAcc_allN_shfl_allDays_alig_avSamps_allMice = data.pop('classAcc_allN_shfl_allDays_alig_avSamps_allMice').flatten()
classAcc_inh_shfl_allDays_alig_avSamps_allMice = data.pop('classAcc_inh_shfl_allDays_alig_avSamps_allMice').flatten()
classAcc_exc_shfl_allDays_alig_avSamps_allMice = data.pop('classAcc_exc_shfl_allDays_alig_avSamps_allMice').flatten()
# sd of samps
classAcc_allN_allDays_alig_sdSamps_allMice = data.pop('classAcc_allN_allDays_alig_sdSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_sdSamps_allMice = data.pop('classAcc_inh_allDays_alig_sdSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_sdSamps_allMice = data.pop('classAcc_exc_allDays_alig_sdSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps
classAcc_allN_shfl_allDays_alig_sdSamps_allMice = data.pop('classAcc_allN_shfl_allDays_alig_sdSamps_allMice').flatten()
classAcc_inh_shfl_allDays_alig_sdSamps_allMice = data.pop('classAcc_inh_shfl_allDays_alig_sdSamps_allMice').flatten()
classAcc_exc_shfl_allDays_alig_sdSamps_allMice = data.pop('classAcc_exc_shfl_allDays_alig_sdSamps_allMice').flatten()
    
classAcc_allN_allDays_alig_avSamps_allMice0 = classAcc_allN_allDays_alig_avSamps_allMice
classAcc_inh_allDays_alig_avSamps_allMice0 = classAcc_inh_allDays_alig_avSamps_allMice
classAcc_exc_allDays_alig_avSamps_allMice0 = classAcc_exc_allDays_alig_avSamps_allMice


if testIncorr: # load class accur vars for testing correct trials
    data = scio.loadmat(cnam, variable_names=['classAccTest_data_allN_alig_avSamps_allMice', 'classAccTest_shfl_allN_alig_avSamps_allMice', 'classAccTest_data_inh_alig_avSamps_allMice',
        'classAccTest_shfl_inh_alig_avSamps_allMice', 'classAccTest_data_exc_alig_avSamps_allMice', 'classAccTest_shfl_exc_alig_avSamps_allMice',
        'classAccTest_data_allN_alig_sdSamps_allMice', 'classAccTest_shfl_allN_alig_sdSamps_allMice', 'classAccTest_data_inh_alig_sdSamps_allMice',
        'classAccTest_shfl_inh_alig_sdSamps_allMice', 'classAccTest_data_exc_alig_sdSamps_allMice', 'classAccTest_shfl_exc_alig_sdSamps_allMice',
        'classAccTest_data_allN_alig_allMice', 'classAccTest_shfl_allN_alig_allMice', 'classAccTest_data_inh_alig_allMice',
        'classAccTest_shfl_inh_alig_allMice', 'classAccTest_data_exc_alig_allMice', 'classAccTest_shfl_exc_alig_allMice'])
    
    classAccTest_data_allN_alig_avSamps_allMice = data.pop('classAccTest_data_allN_alig_avSamps_allMice').flatten()
    classAccTest_shfl_allN_alig_avSamps_allMice = data.pop('classAccTest_shfl_allN_alig_avSamps_allMice').flatten()
    classAccTest_data_inh_alig_avSamps_allMice = data.pop('classAccTest_data_inh_alig_avSamps_allMice').flatten()
    classAccTest_shfl_inh_alig_avSamps_allMice = data.pop('classAccTest_shfl_inh_alig_avSamps_allMice').flatten()
    classAccTest_data_exc_alig_avSamps_allMice = data.pop('classAccTest_data_exc_alig_avSamps_allMice').flatten()
    classAccTest_shfl_exc_alig_avSamps_allMice = data.pop('classAccTest_shfl_exc_alig_avSamps_allMice').flatten()
    classAccTest_data_allN_alig_sdSamps_allMice = data.pop('classAccTest_data_allN_alig_sdSamps_allMice').flatten()
    classAccTest_shfl_allN_alig_sdSamps_allMice = data.pop('classAccTest_shfl_allN_alig_sdSamps_allMice').flatten()
    classAccTest_data_inh_alig_sdSamps_allMice = data.pop('classAccTest_data_inh_alig_sdSamps_allMice').flatten()
    classAccTest_shfl_inh_alig_sdSamps_allMice = data.pop('classAccTest_shfl_inh_alig_sdSamps_allMice').flatten()
    classAccTest_data_exc_alig_sdSamps_allMice = data.pop('classAccTest_data_exc_alig_sdSamps_allMice').flatten()
    classAccTest_shfl_exc_alig_sdSamps_allMice = data.pop('classAccTest_shfl_exc_alig_sdSamps_allMice').flatten()
    classAccTest_data_allN_alig_allMice = data.pop('classAccTest_data_allN_alig_allMice').flatten()
    classAccTest_shfl_allN_alig_allMice = data.pop('classAccTest_shfl_allN_alig_allMice').flatten()
    classAccTest_data_inh_alig_allMice = data.pop('classAccTest_data_inh_alig_allMice').flatten()
    classAccTest_shfl_inh_alig_allMice = data.pop('classAccTest_shfl_inh_alig_allMice').flatten()
    classAccTest_data_exc_alig_allMice = data.pop('classAccTest_data_exc_alig_allMice').flatten()
    classAccTest_shfl_exc_alig_allMice = data.pop('classAccTest_shfl_exc_alig_allMice').flatten()
        
    data = scio.loadmat(cnam, variable_names=['classAcc_allN_allDays_alig_allMice', 'classAcc_inh_allDays_alig_allMice', 'classAcc_exc_allDays_alig_allMice', 'classAcc_allN_shfl_allDays_alig_allMice', 'classAcc_inh_shfl_allDays_alig_allMice', 'classAcc_exc_shfl_allDays_alig_allMice'])
    classAcc_allN_allDays_alig_allMice = data.pop('classAcc_allN_allDays_alig_allMice').flatten()
    classAcc_inh_allDays_alig_allMice = data.pop('classAcc_inh_allDays_alig_allMice').flatten()
    classAcc_exc_allDays_alig_allMice = data.pop('classAcc_exc_allDays_alig_allMice').flatten()
    classAcc_allN_shfl_allDays_alig_allMice = data.pop('classAcc_allN_shfl_allDays_alig_allMice').flatten()
    classAcc_inh_shfl_allDays_alig_allMice = data.pop('classAcc_inh_shfl_allDays_alig_allMice').flatten()
    classAcc_exc_shfl_allDays_alig_allMice = data.pop('classAcc_exc_shfl_allDays_alig_allMice').flatten()
                        
                        
#%% Set some vars

numExcSamps = np.shape(classAcc_exc_allDays_alig_avSamps_allMice[0])[-1]
numSamps = 50 # this is the usual value, get it from classAcc_allN_allDays_alig_allMice[im].shape[-1]

nowStr_allMice = [cnam[-17:-4]] * len(mice)


#%% Set time_aligned for all mice

time_aligned_allMice = []
nPreMin_allMice = []   
 
for im in range(len(mice)):  
    time_aligned, nPreMin, nPostMin = set_nprepost(classAcc_allN_allDays_alig_avSamps_allMice[im], eventI_ds_allDays_allMice[im], mn_corr_allMice[im], thTrained, regressBins, traceAl=1)    
    
    time_aligned_allMice.append(time_aligned)
    nPreMin_allMice.append(nPreMin)


lenTraces_AllMice = np.array([len(time_aligned_allMice[im]) for im in range(len(mice))])



#%% ########################## Set averages of CA ##############################
################################################################################
################################################################################

##%% exc: average across excSamps for each day (of each mouse)

classAcc_exc_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_allDays_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_shfl_allDays_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs


#%% Average CA across days for each mouse

#AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

classAcc_allN_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
# shfl
classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    


######%% SD across days for each mouse

# AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]
    
classAcc_allN_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
# shfl
classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs



#%% ########################## Same as above but for testing corr trials ##############################

if testIncorr:
    ##%% exc: average across excSamps for each day (of each mouse)
    
    classAccTest_data_exc_alig_avSamps2_allMice = [np.mean(classAccTest_data_exc_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs
    classAccTest_shfl_exc_alig_avSamps2_allMice = [np.mean(classAccTest_shfl_exc_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs
    
    
    #%% Average CA across days for each mouse
    
    #AVERAGE only the last 10 days! in case naive vs trained makes a difference!
    #[np.max([-10,-len(days_allMice[im])]):]
    
    classAccTest_data_allN_alig_avSamps_avDays_allMice = np.array([np.mean(classAccTest_data_allN_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_data_inh_alig_avSamps_avDays_allMice = np.array([np.mean(classAccTest_data_inh_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_data_exc_alig_avSamps_avDays_allMice = np.array([np.mean(classAccTest_data_exc_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    # shfl
    classAccTest_shfl_allN_alig_avSamps_avDays_allMice = np.array([np.mean(classAccTest_shfl_allN_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_shfl_inh_alig_avSamps_avDays_allMice = np.array([np.mean(classAccTest_shfl_inh_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_shfl_exc_alig_avSamps_avDays_allMice = np.array([np.mean(classAccTest_shfl_exc_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
        
    
    
    ######%% SD across days for each mouse
    
    # AVERAGE only the last 10 days! in case naive vs trained makes a difference!
    #[np.max([-10,-len(days_allMice[im])]):]
        
    classAccTest_data_allN_alig_avSamps_sdDays_allMice = np.array([np.std(classAccTest_data_allN_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_data_inh_alig_avSamps_sdDays_allMice = np.array([np.std(classAccTest_data_inh_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_data_exc_alig_avSamps_sdDays_allMice = np.array([np.std(classAccTest_data_exc_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    # shfl
    classAccTest_shfl_allN_alig_avSamps_sdDays_allMice = np.array([np.std(classAccTest_shfl_allN_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_shfl_inh_alig_avSamps_sdDays_allMice = np.array([np.std(classAccTest_shfl_inh_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    classAccTest_shfl_exc_alig_avSamps_sdDays_allMice = np.array([np.std(classAccTest_shfl_exc_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs



