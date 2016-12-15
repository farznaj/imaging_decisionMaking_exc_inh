# -*- coding: utf-8 -*-
"""
Get plots to compare class accuracy of decoders trained using only inh, or only exc (also finding bestc for each separately)
Vars are provided by svm_excInh_trainDecoder.py (normally ran on the cluster)

Created on Tue Dec 13 17:15:11 2016
@author: farznaj
"""

# Go to svm_plots_setVars and define vars!
execfile("svm_plots_setVars.py")    


#%%
dnow = '/excInh_trainDecoder/setTo50ErrOfSampsWith0Weights'
#dnow = '/l1_l2_subsel_comparison'
thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
setToNaN = 1 # if 1, the above two jobs will be done.


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
def setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName='all'):
    import glob
    
    if trialHistAnalysis:
    #    ep_ms = np.round((ep-eventI)*frameLength)
    #    th_stim_dur = []
        if np.isnan(roundi): # set excInh name
            svmn = 'excInh_SVMtrained_prevChoice_%sN_%sITIs_ep*-*ms_*' %(ntName, itiName) # C2 has test/trial shuffles as well. C doesn't have it.
#        else:
#            svmn = 'svmPrevChoice_%sN_%sITIs_ep*ms_r%d_*' %(ntName, itiName, roundi)
    else:
        if np.isnan(roundi): # set excInh name
            svmn = 'excInh_SVMtrained_currChoice_%sN_ep%d-%dms_*' %(ntName, ep_ms[0], ep_ms[-1])   
#        else:
#            svmn = 'svmCurrChoice_%sN_ep*ms_r%d_*' %(ntName, roundi)   
    
    svmn = svmn + pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName
    
    
    
#%%
perClassErrorTrain_data_inh_allD = []
perClassErrorTrain_shfl_inh_allD = []
perClassErrorTest_data_inh_allD = []
perClassErrorTest_shfl_inh_allD = []

perClassErrorTrain_data_exc_allD = []
perClassErrorTrain_shfl_exc_allD = []
perClassErrorTest_data_exc_allD = []
perClassErrorTest_shfl_exc_allD = []

perClassErrorTrain_data_allExc_allD = []
perClassErrorTrain_shfl_allExc_allD = []
perClassErrorTest_data_allExc_allD = []
perClassErrorTest_shfl_allExc_allD = []
    
numNon0SampShfl_inh = np.full((1, len(days)), np.nan).flatten() # For each day: number of cv samples with >=2 non-0 weights
numNon0SampData_inh = np.full((1, len(days)), np.nan).flatten()
numNon0WShfl_inh = np.full((1, len(days)), np.nan).flatten() # For each day: average number of non0 weights across all samples of shuffled data

numNon0SampShfl_exc = np.full((1, len(days)), np.nan).flatten() # For each day: number of cv samples with >=2 non-0 weights
numNon0SampData_exc = np.full((1, len(days)), np.nan).flatten()
numNon0WShfl_exc = np.full((1, len(days)), np.nan).flatten() # For each day: average number of non0 weights across all samples of shuffled data

numNon0SampShfl_allExc = np.full((1, len(days)), np.nan).flatten() # For each day: number of cv samples with >=2 non-0 weights
numNon0SampData_allExc = np.full((1, len(days)), np.nan).flatten()
numNon0WShfl_allExc = np.full((1, len(days)), np.nan).flatten() # For each day: average number of non0 weights across all samples of shuffled data

fractNon0_inh = []
fractNon0_exc = []
fractNon0_allExc = []

absNon0_inh = []
absNon0_exc = []
absNon0_allExc = []
    
for iday in range(len(days)):
    
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
    
    
    #%%
    svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).
#    svmNameAll = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName, 0) # go w latest for 500 shuffles and bestc including at least one non0 w.
    print os.path.basename(svmName)
        
    '''    
    Data = scio.loadmat(svmName, variable_names=['w'])
    w = Data.pop('w')[0,:]
    
    if abs(w.sum()) < eps:  # I think it is wrong to do this. When ws are all 0, it means there was no decoder for that day, ie no info about the choice.
        print '\tAll weights in the allTrials-trained decoder are 0 ... '            
#    else:    
    '''        
    Data = scio.loadmat(svmName, variable_names=['perClassErrorTrain_data_inh', 'perClassErrorTrain_shfl_inh', 'perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh'])    
    perClassErrorTrain_data_inh = Data.pop('perClassErrorTrain_data_inh').squeeze()
    perClassErrorTrain_shfl_inh = Data.pop('perClassErrorTrain_shfl_inh').squeeze()
    perClassErrorTest_data_inh = Data.pop('perClassErrorTest_data_inh').squeeze()
    perClassErrorTest_shfl_inh = Data.pop('perClassErrorTest_shfl_inh').squeeze()

    Data = scio.loadmat(svmName, variable_names=['perClassErrorTrain_data_exc', 'perClassErrorTrain_shfl_exc', 'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc'])    
    perClassErrorTrain_data_exc = Data.pop('perClassErrorTrain_data_exc').squeeze()
    perClassErrorTrain_shfl_exc = Data.pop('perClassErrorTrain_shfl_exc').squeeze()
    perClassErrorTest_data_exc = Data.pop('perClassErrorTest_data_exc').squeeze()
    perClassErrorTest_shfl_exc = Data.pop('perClassErrorTest_shfl_exc').squeeze()

    Data = scio.loadmat(svmName, variable_names=['perClassErrorTrain_data_allExc', 'perClassErrorTrain_shfl_allExc', 'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc'])    
    perClassErrorTrain_data_allExc = Data.pop('perClassErrorTrain_data_allExc').squeeze()
    perClassErrorTrain_shfl_allExc = Data.pop('perClassErrorTrain_shfl_allExc').squeeze()
    perClassErrorTest_data_allExc = Data.pop('perClassErrorTest_data_allExc').squeeze()
    perClassErrorTest_shfl_allExc = Data.pop('perClassErrorTest_shfl_allExc').squeeze()
    
  
    
    # Load weights to see if there are shuffles with very few non-0 weights
    ### Inh    
    Data = scio.loadmat(svmName, variable_names=['w_shfl_inh', 'w_data_inh'])
    w_shfl_inh = Data.pop('w_shfl_inh')
#        ash = np.mean(w_shfl!=0,axis=1) # fraction non-0 weights
#        ash = ash>0 # index of samples that have at least 1 non0 weight for shuffled  
    ash_inh = np.sum(w_shfl_inh!=0,axis=1)<thNon0Ws # samples w fewer than 2 non-0 weights
    ash_inh = ~ash_inh # samples w >=2 non0 weights
    w_data_inh = Data.pop('w_data_inh')
#        ada = np.mean(w_data!=0,axis=1) # fraction non-0 weights                
#        ada = ada>0 # index of samples that have at least 1 non0 weight for data
    ada_inh = np.sum(w_data_inh!=0,axis=1)<thNon0Ws # samples w fewer than 2 non-0 weights        
    ada_inh = ~ada_inh # samples w >=2 non0 weights
    numNon0SampShfl_inh[iday] = ash_inh.sum() # number of cv samples with >=2 non-0 weights
    numNon0SampData_inh[iday] = ada_inh.sum()
    numNon0WShfl_inh[iday] = np.sum(w_shfl_inh!=0,axis=1).mean() # average number of non0 weights across all samples of shuffled data
 
     
    ### Exc
    Data = scio.loadmat(svmName, variable_names=['w_shfl_exc', 'w_data_exc'])
    w_shfl_exc = Data.pop('w_shfl_exc')
#        ash = np.mean(w_shfl!=0,axis=1) # fraction non-0 weights
#        ash = ash>0 # index of samples that have at least 1 non0 weight for shuffled  
    ash_exc = np.sum(w_shfl_exc!=0,axis=2)<thNon0Ws # samples w fewer than 2 non-0 weights
    ash_exc = ~ash_exc # samples w >=2 non0 weights
    w_data_exc = Data.pop('w_data_exc')
#        ada = np.mean(w_data!=0,axis=1) # fraction non-0 weights                
#        ada = ada>0 # index of samples that have at least 1 non0 weight for data
    ada_exc = np.sum(w_data_exc!=0,axis=2)<thNon0Ws # samples w fewer than 2 non-0 weights        
    ada_exc = ~ada_exc # samples w >=2 non0 weights
    numNon0SampShfl_exc0 = np.sum(ash_exc,axis=0) # 10x1; for each neuron shuffle, it shows number of cv samples with >=2 non-0 weights
    numNon0SampData_exc0 = np.sum(ada_exc,axis=0)
    numNon0WShfl_exc0 = np.mean(np.sum(w_shfl_exc!=0,axis=2), axis=0) # for each neuron shuffle, it shows average number of non0 weights across all samples of shuffled data
    


    ### All exc
    Data = scio.loadmat(svmName, variable_names=['w_shfl_allExc', 'w_data_allExc'])
    w_shfl_allExc = Data.pop('w_shfl_allExc')
#        ash = np.mean(w_shfl!=0,axis=1) # fraction non-0 weights
#        ash = ash>0 # index of samples that have at least 1 non0 weight for shuffled  
    ash_allExc = np.sum(w_shfl_allExc!=0,axis=1)<thNon0Ws # samples w fewer than 2 non-0 weights
    ash_allExc = ~ash_allExc # samples w >=2 non0 weights
    w_data_allExc = Data.pop('w_data_allExc')
#        ada = np.mean(w_data!=0,axis=1) # fraction non-0 weights                
#        ada = ada>0 # index of samples that have at least 1 non0 weight for data
    ada_allExc = np.sum(w_data_allExc!=0,axis=1)<thNon0Ws # samples w fewer than 2 non-0 weights        
    ada_allExc = ~ada_allExc # samples w >=2 non0 weights
    numNon0SampShfl_allExc[iday] = ash_allExc.sum() # number of cv samples with >=2 non-0 weights
    numNon0SampData_allExc[iday] = ada_allExc.sum()
    numNon0WShfl_allExc[iday] = np.sum(w_shfl_allExc!=0,axis=1).mean() # average number of non0 weights across all samples of shuffled data


    # fraction of non0 w for each sample
    fractNon0_inh.append(np.mean(w_data_inh!=0,axis=-1))
    fractNon0_exc.append(np.ravel(np.mean(w_data_exc!=0,axis=-1), order='F'))
    fractNon0_allExc.append(np.mean(w_data_allExc!=0,axis=-1))

    # mean of abs of non0 ws for each samples
    a = w_data_inh+0
    a[w_data_inh==0] = np.nan # set 0 w to nan; so you only take non-0 weights
    absNon0_inh.append(np.nanmean(abs(a),axis=-1))
    a = w_data_exc+0
    a[w_data_exc==0] = np.nan # set 0 w to nan; so you only take non-0 weights    
    absNon0_exc.append(np.ravel(np.nanmean(abs(a),axis=-1), order='F'))
    a = w_data_allExc+0
    a[w_data_allExc==0] = np.nan # set 0 w to nan; so you only take non-0 weights    
    absNon0_allExc.append(np.nanmean(abs(a),axis=-1))


    if setToNaN:
        # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
        # The reson I don't simply exclude them is that I want them to count when I get average across days... lets say I am comparing two conditions, one doesn't have much decoding info (as a result mostly 0 weights)... I don't want to throw it away... not having information about the choice is informative for me.
        # Set to nan the test/train error of those samples that have all-0 weights
        perClassErrorTest_data_inh[~ada_inh] = 50#np.nan #it will be nan for samples with all-0 weights
        perClassErrorTest_shfl_inh[~ash_inh] = 50#np.nan
        perClassErrorTrain_data_inh[~ada_inh] = 50#np.nan #it will be nan for samples with all-0 weights
        perClassErrorTrain_shfl_inh[~ash_inh] = 50#np.nan

        perClassErrorTest_data_exc[~ada_exc] = 50#np.nan #it will be nan for samples with all-0 weights
        perClassErrorTest_shfl_exc[~ash_exc] = 50#np.nan
        perClassErrorTrain_data_exc[~ada_exc] = 50#np.nan #it will be nan for samples with all-0 weights
        perClassErrorTrain_shfl_exc[~ash_exc] = 50#np.nan

        perClassErrorTest_data_allExc[~ada_allExc] = 50#np.nan #it will be nan for samples with all-0 weights
        perClassErrorTest_shfl_allExc[~ash_allExc] = 50#np.nan
        perClassErrorTrain_data_allExc[~ada_allExc] = 50#np.nan #it will be nan for samples with all-0 weights
        perClassErrorTrain_shfl_allExc[~ash_allExc] = 50#np.nan

        # NOTE: I think you need to finish the following... 
#        absNon0_inh[~ada_inh] = np.nan 
 
    if setToNaN:
        # For days that have <10 samples that satisfy >=2 non0 weights, we will manually set the class error of all samples to 50 ... bc we think <10 samples will not give us an accurate measure of class error of a day.
        # inh        
        if numNon0SampShfl_inh[iday]<thSamps:
            perClassErrorTest_shfl_inh = 50*np.ones((perClassErrorTest_data_inh.shape))#np.nan
            perClassErrorTrain_shfl_inh = 50*np.ones((perClassErrorTest_data_inh.shape))#np.nan
            numNon0SampShfl_inh[iday] = perClassErrorTest_shfl_inh.shape[0]
            
        if numNon0SampData_inh[iday]<thSamps:
            perClassErrorTest_data_inh = 50*np.ones((perClassErrorTest_data_inh.shape))#np.nan
            perClassErrorTrain_data_inh = 50*np.ones((perClassErrorTest_data_inh.shape))#np.nan
            numNon0SampData_inh[iday] = perClassErrorTest_data_inh.shape[0]


        # exc
        if (numNon0SampShfl_exc0 < thSamps).any(): # find those neural shuffles that have <thSamp non0 samples ... and set all their samples to chance.
            perClassErrorTest_shfl_exc[:, numNon0SampShfl_exc0 < thSamps] = 50*np.ones((perClassErrorTest_data_exc[:, numNon0SampShfl_exc0 < thSamps].shape))#np.nan
            perClassErrorTrain_shfl_exc[:, numNon0SampShfl_exc0 < thSamps] = 50*np.ones((perClassErrorTest_data_exc[:, numNon0SampShfl_exc0 < thSamps].shape))#np.nan
            numNon0SampShfl_exc0[numNon0SampShfl_exc0 < thSamps] = perClassErrorTest_shfl_exc.shape[0]
            
        if (numNon0SampData_exc0 < thSamps).any():
            perClassErrorTest_data_exc[:, numNon0SampData_exc0 < thSamps] = 50*np.ones((perClassErrorTest_data_exc[:, numNon0SampData_exc0 < thSamps].shape))#np.nan
            perClassErrorTrain_data_exc[:, numNon0SampData_exc0 < thSamps] = 50*np.ones((perClassErrorTest_data_exc[:, numNon0SampData_exc0 < thSamps].shape))#np.nan
            numNon0SampData_exc0[numNon0SampData_exc0 < thSamps] = perClassErrorTest_data_exc.shape[0]
                
        # allExc
        if numNon0SampShfl_allExc[iday]<thSamps:
            perClassErrorTest_shfl_allExc = 50*np.ones((perClassErrorTest_data_allExc.shape))#np.nan
            perClassErrorTrain_shfl_allExc = 50*np.ones((perClassErrorTest_data_allExc.shape))#np.nan
            numNon0SampShfl_allExc[iday] = perClassErrorTest_shfl_allExc.shape[0]
            
        if numNon0SampData_allExc[iday]<thSamps:
            perClassErrorTest_data_allExc = 50*np.ones((perClassErrorTest_data_allExc.shape))#np.nan
            perClassErrorTrain_data_allExc = 50*np.ones((perClassErrorTest_data_allExc.shape))#np.nan
            numNon0SampData_allExc[iday] = perClassErrorTest_data_allExc.shape[0]
                
                
    # exc: pool all trial shuffles and neural shuffles.           
    perClassErrorTest_shfl_exc = np.ravel(perClassErrorTest_shfl_exc, order='F') # 1000x1; make a vector: first all trial shuffles, then all neural shuffles
    perClassErrorTrain_shfl_exc = np.ravel(perClassErrorTrain_shfl_exc, order='F')            
    perClassErrorTest_data_exc = np.ravel(perClassErrorTest_data_exc, order='F')
    perClassErrorTrain_data_exc = np.ravel(perClassErrorTrain_data_exc, order='F')
    numNon0SampShfl_exc[iday] = np.sum(numNon0SampShfl_exc0)
    numNon0SampData_exc[iday] = np.sum(numNon0SampData_exc0)
            

    # keep results from all days
    perClassErrorTrain_data_inh_allD.append(perClassErrorTrain_data_inh) # 100x1
    perClassErrorTrain_shfl_inh_allD.append(perClassErrorTrain_shfl_inh)    
    perClassErrorTest_data_inh_allD.append(perClassErrorTest_data_inh)
    perClassErrorTest_shfl_inh_allD.append(perClassErrorTest_shfl_inh)        
    
    perClassErrorTrain_data_exc_allD.append(perClassErrorTrain_data_exc) # 1000x1 bc there are 10 neural samples and 100 trial samples, all pooled.
    perClassErrorTrain_shfl_exc_allD.append(perClassErrorTrain_shfl_exc)    
    perClassErrorTest_data_exc_allD.append(perClassErrorTest_data_exc)
    perClassErrorTest_shfl_exc_allD.append(perClassErrorTest_shfl_exc)

    perClassErrorTrain_data_allExc_allD.append(perClassErrorTrain_data_allExc)
    perClassErrorTrain_shfl_allExc_allD.append(perClassErrorTrain_shfl_allExc)    
    perClassErrorTest_data_allExc_allD.append(perClassErrorTest_data_allExc)
    perClassErrorTest_shfl_allExc_allD.append(perClassErrorTest_shfl_allExc)    


#%% Average and std across shuffles for each day ... compute class accuracy
    
classAccTestData_inh_aveS = 100 - np.mean(perClassErrorTest_data_inh_allD, axis=1)
classAccTestData_exc_aveS = 100 - np.mean(perClassErrorTest_data_exc_allD, axis=1)
classAccTestData_allExc_aveS = 100 - np.mean(perClassErrorTest_data_allExc_allD, axis=1)
classAccTestData_inh_sdS = np.std(perClassErrorTest_data_inh_allD, axis=1)
classAccTestData_exc_sdS = np.std(perClassErrorTest_data_exc_allD, axis=1)
classAccTestData_allExc_sdS = np.std(perClassErrorTest_data_allExc_allD, axis=1)

classAccTestShfl_inh_aveS = 100 - np.mean(perClassErrorTest_shfl_inh_allD, axis=1)
classAccTestShfl_exc_aveS = 100 - np.mean(perClassErrorTest_shfl_exc_allD, axis=1)
classAccTestShfl_allExc_aveS = 100 - np.mean(perClassErrorTest_shfl_allExc_allD, axis=1)
classAccTestShfl_inh_sdS = np.std(perClassErrorTest_shfl_inh_allD, axis=1)
classAccTestShfl_exc_sdS = np.std(perClassErrorTest_shfl_exc_allD, axis=1)
classAccTestShfl_allExc_sdS = np.std(perClassErrorTest_shfl_allExc_allD, axis=1)

classAccTrainData_inh_aveS = 100 - np.mean(perClassErrorTrain_data_inh_allD, axis=1)
classAccTrainData_exc_aveS = 100 - np.mean(perClassErrorTrain_data_exc_allD, axis=1)
classAccTrainData_allExc_aveS = 100 - np.mean(perClassErrorTrain_data_allExc_allD, axis=1)
classAccTrainData_inh_sdS = np.std(perClassErrorTrain_data_inh_allD, axis=1)
classAccTrainData_exc_sdS = np.std(perClassErrorTrain_data_exc_allD, axis=1)
classAccTrainData_allExc_sdS = np.std(perClassErrorTrain_data_allExc_allD, axis=1)

classAccTrainShfl_inh_aveS = 100 - np.mean(perClassErrorTrain_shfl_inh_allD, axis=1)
classAccTrainShfl_exc_aveS = 100 - np.mean(perClassErrorTrain_shfl_exc_allD, axis=1)
classAccTrainShfl_allExc_aveS = 100 - np.mean(perClassErrorTrain_shfl_allExc_allD, axis=1)
classAccTrainShfl_inh_sdS = np.std(perClassErrorTrain_shfl_inh_allD, axis=1)
classAccTrainShfl_exc_sdS = np.std(perClassErrorTrain_shfl_exc_allD, axis=1)
classAccTrainShfl_allExc_sdS = np.std(perClassErrorTrain_shfl_allExc_allD, axis=1)

#_,p = stats.ttest_ind(classAccTestData_inh_aveS, classAccTestData_exc_aveS)
#_,p = stats.ttest_ind(classAccTestData_allExc_aveS, classAccTestData_exc_aveS)


fractNon0_inh_aveS = np.mean(fractNon0_inh, axis=1)
fractNon0_exc_aveS = np.mean(fractNon0_exc, axis=1)
fractNon0_allExc_aveS = np.mean(fractNon0_allExc, axis=1)
fractNon0_inh_sdS = np.std(fractNon0_inh, axis=1)
fractNon0_exc_sdS = np.std(fractNon0_exc, axis=1)
fractNon0_allExc_sdS = np.std(fractNon0_allExc, axis=1)

absNon0_inh_aveS = np.nanmean(absNon0_inh, axis=1)
absNon0_exc_aveS = np.nanmean(absNon0_exc, axis=1) # average across all neural and trial shuffles.
absNon0_allExc_aveS = np.nanmean(absNon0_allExc, axis=1)
absNon0_inh_sdS = np.nanstd(absNon0_inh, axis=1)
absNon0_exc_sdS = np.nanstd(absNon0_exc, axis=1)
absNon0_allExc_sdS = np.nanstd(absNon0_allExc, axis=1)

#sd_l1_test_d = np.true_divide(np.nanstd(l1_err_test_data, axis=0), np.sqrt(numNon0SampData)) #/ np.sqrt(numSamp) 




#%%
########################################## PLOTS ##########################################

#%% Fract non0 weights and abs of non0 w

plt.figure(figsize=(6,5.5))
gs = gridspec.GridSpec(2, 5)#, width_ratios=[2, 1]) 

#### Fract non0 weights
ax = plt.subplot(gs[0,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), fractNon0_inh_aveS, yerr = fractNon0_inh_sdS, color='r', label='inh')
plt.errorbar(range(numDays), fractNon0_exc_aveS, yerr = fractNon0_exc_sdS, color='b', label='exc')
plt.errorbar(range(numDays), fractNon0_allExc_aveS, yerr = fractNon0_allExc_sdS, color='k', label='allExc')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Fraction non-0 w', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.4), frameon=False)
#plt.title('Data')
makeNicePlots(ax)


##%% Average across days
# data
x = [0,1,2]
labels = ['inh', 'exc', 'allExc']
y = [np.mean(fractNon0_inh_aveS), np.mean(fractNon0_exc_aveS), np.mean(fractNon0_allExc_aveS)]
yerr = [np.std(fractNon0_inh_aveS), np.std(fractNon0_exc_aveS), np.std(fractNon0_allExc_aveS)]
   
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, y, yerr, marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#plt.title('Data')
ymin, ymax = ax.get_ylim()
makeNicePlots(ax)



#### abs non0 weights

ax = plt.subplot(gs[1,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), absNon0_inh_aveS, yerr = absNon0_inh_sdS, color='r', label='inh')
plt.errorbar(range(numDays), absNon0_exc_aveS, yerr = absNon0_exc_sdS, color='b', label='exc')
plt.errorbar(range(numDays), absNon0_allExc_aveS, yerr = absNon0_allExc_sdS, color='k', label='allExc')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('abs non-0 w', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.4), frameon=False)
#plt.title('Data')
makeNicePlots(ax)


##%% Average across days
# data
x = [0,1,2]
labels = ['inh', 'exc', 'allExc']
y = [np.mean(absNon0_inh_aveS), np.mean(absNon0_exc_aveS), np.mean(absNon0_allExc_aveS)]
yerr = [np.std(absNon0_inh_aveS), np.std(absNon0_exc_aveS), np.std(absNon0_allExc_aveS)]
   
ax = plt.subplot(gs[1,3:4])
plt.errorbar(x, y, yerr, marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#plt.title('Data')
ymin, ymax = ax.get_ylim()
makeNicePlots(ax)

plt.subplots_adjust(wspace=1, hspace=.5)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow, suffn+'ei_fractNon0_absW'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')



#%% Testing data: inh, exc, allExc superimposed:

plt.figure(figsize=(6,5.5))
gs = gridspec.GridSpec(2, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTestData_inh_aveS, yerr = classAccTestData_inh_sdS, color='r', label='inh')
plt.errorbar(range(numDays), classAccTestData_exc_aveS, yerr = classAccTestData_exc_sdS, color='b', label='exc')
plt.errorbar(range(numDays), classAccTestData_allExc_aveS, yerr = classAccTestData_allExc_sdS, color='k', label='allExc')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.4), frameon=False)
plt.title('Data')
makeNicePlots(ax)


##%% Average across days
# data
x = [0,1,2]
labels = ['inh', 'exc', 'allExc']
y = [np.mean(classAccTestData_inh_aveS), np.mean(classAccTestData_exc_aveS), np.mean(classAccTestData_allExc_aveS)]
yerr = [np.std(classAccTestData_inh_aveS), np.std(classAccTestData_exc_aveS), np.std(classAccTestData_allExc_aveS)]
# Pool all days
#y = [100-np.mean(np.ravel(perClassErrorTest_data_inh_allD, order='F')), 100-np.mean(np.ravel(perClassErrorTest_data_exc_allD, order='F')), 100-np.mean(np.ravel(perClassErrorTest_data_allExc_allD, order='F'))]
#yerr = [np.std(np.ravel(perClassErrorTest_data_inh_allD, order='F')), np.std(np.ravel(perClassErrorTest_data_exc_allD, order='F')), np.std(np.ravel(perClassErrorTest_data_allExc_allD, order='F'))]
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, y, yerr, marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.title('Data')
ymin, ymax = ax.get_ylim()
makeNicePlots(ax)



# shuffle
ax = plt.subplot(gs[1,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTestShfl_inh_aveS, yerr = classAccTestShfl_inh_sdS, color='r', label='inh')
plt.errorbar(range(numDays), classAccTestShfl_exc_aveS, yerr = classAccTestShfl_exc_sdS, color='b', label='exc')
plt.errorbar(range(numDays), classAccTestShfl_allExc_aveS, yerr = classAccTestShfl_allExc_sdS, color='k', label='allExc')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.4), frameon=False)
plt.title('Shuffle')
makeNicePlots(ax)


# average across days
x = [0,1,2]
labels = ['inh', 'exc', 'allExc']
y = [np.mean(classAccTestShfl_inh_aveS), np.mean(classAccTestShfl_exc_aveS), np.mean(classAccTestShfl_allExc_aveS)]
yerr = [np.std(classAccTestShfl_inh_aveS), np.std(classAccTestShfl_exc_aveS), np.std(classAccTestShfl_allExc_aveS)]
    
ax = plt.subplot(gs[0,4:5])
plt.errorbar(x, y, yerr, marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
ymin, _ = ax.get_ylim()
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
plt.title('Shuffle')
makeNicePlots(ax)

plt.subplot(gs[0,3:4])
plt.ylim([ymin, ymax])


plt.subplots_adjust(wspace=1, hspace=.9)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow, suffn+'ei_test_all'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')



#%% Testing data: Plot average across shuffles for each day, inh, exc, allExc each one plot
### L1;
plt.figure(figsize=(6,7.5))
gs = gridspec.GridSpec(3, 5)#, width_ratios=[2, 1]) 

### INH
ax = plt.subplot(gs[0,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTestData_inh_aveS, yerr = classAccTestData_inh_sdS, color='g', label='Data')
plt.errorbar(range(numDays), classAccTestShfl_inh_aveS, yerr = classAccTestShfl_inh_sdS, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
#plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, [np.mean(classAccTestData_inh_aveS), np.mean(classAccTestShfl_inh_aveS)], yerr = [np.std(classAccTestData_inh_aveS), np.std(classAccTestShfl_inh_aveS)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#av_l1_test_d = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#av_l1_test_s = 100-np.nanmean(l1_err_test_shfl0, axis=0) 
#_,p = stats.ttest_ind(av_l1_test_d, av_l1_test_s)
#plt.title('p= %.3f' %(p))
plt.title('Inh')

makeNicePlots(ax)



### EXC
ax = plt.subplot(gs[1,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTestData_exc_aveS, yerr = classAccTestData_exc_sdS, color='g', label='Data')
plt.errorbar(range(numDays), classAccTestShfl_exc_aveS, yerr = classAccTestShfl_exc_sdS, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[1,3:4])
plt.errorbar(x, [np.mean(classAccTestData_exc_aveS), np.mean(classAccTestShfl_exc_aveS)], yerr = [np.std(classAccTestData_exc_aveS), np.std(classAccTestShfl_exc_aveS)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#av_l1_test_d = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#av_l1_test_s = 100-np.nanmean(l1_err_test_shfl0, axis=0) 
#_,p = stats.ttest_ind(av_l1_test_d, av_l1_test_s)
#plt.title('p= %.3f' %(p))
plt.title('Exc')

makeNicePlots(ax)



### All eXC
ax = plt.subplot(gs[2,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTestData_allExc_aveS, yerr = classAccTestData_allExc_sdS, color='g', label='Data')
plt.errorbar(range(numDays), classAccTestShfl_allExc_aveS, yerr = classAccTestShfl_allExc_sdS, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
#plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[2,3:4])
plt.errorbar(x, [np.mean(classAccTestData_allExc_aveS), np.mean(classAccTestShfl_allExc_aveS)], yerr = [np.std(classAccTestData_allExc_aveS), np.std(classAccTestShfl_allExc_aveS)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#av_l1_test_d = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#av_l1_test_s = 100-np.nanmean(l1_err_test_shfl0, axis=0) 
#_,p = stats.ttest_ind(av_l1_test_d, av_l1_test_s)
#plt.title('p= %.3f' %(p))
plt.title('All exc')

makeNicePlots(ax)



plt.subplots_adjust(wspace=1, hspace=.7)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow, suffn+'ei_test_sep'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')  





#%% Training data:

#%% Training data: inh, exc, allExc superimposed:

plt.figure(figsize=(6,5.5))
gs = gridspec.GridSpec(2, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTrainData_inh_aveS, yerr = classAccTrainData_inh_sdS, color='r', label='inh')
plt.errorbar(range(numDays), classAccTrainData_exc_aveS, yerr = classAccTrainData_exc_sdS, color='b', label='exc')
plt.errorbar(range(numDays), classAccTrainData_allExc_aveS, yerr = classAccTrainData_allExc_sdS, color='k', label='allExc')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(training data)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.4), frameon=False)
plt.title('Data')
makeNicePlots(ax)


##%% Average across days
# data
x = [0,1,2]
labels = ['inh', 'exc', 'allExc']
y = [np.mean(classAccTrainData_inh_aveS), np.mean(classAccTrainData_exc_aveS), np.mean(classAccTrainData_allExc_aveS)]
yerr = [np.std(classAccTrainData_inh_aveS), np.std(classAccTrainData_exc_aveS), np.std(classAccTrainData_allExc_aveS)]
   
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, y, yerr, marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.title('Data')
ymin, ymax = ax.get_ylim()
makeNicePlots(ax)



# shuffle
ax = plt.subplot(gs[1,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTrainShfl_inh_aveS, yerr = classAccTrainShfl_inh_sdS, color='r', label='inh')
plt.errorbar(range(numDays), classAccTrainShfl_exc_aveS, yerr = classAccTrainShfl_exc_sdS, color='b', label='exc')
plt.errorbar(range(numDays), classAccTrainShfl_allExc_aveS, yerr = classAccTrainShfl_allExc_sdS, color='k', label='allExc')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(training data)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.4), frameon=False)
plt.title('Shuffle')
makeNicePlots(ax)


# average across days
x = [0,1,2]
labels = ['inh', 'exc', 'allExc']
y = [np.mean(classAccTrainShfl_inh_aveS), np.mean(classAccTrainShfl_exc_aveS), np.mean(classAccTrainShfl_allExc_aveS)]
yerr = [np.std(classAccTrainShfl_inh_aveS), np.std(classAccTrainShfl_exc_aveS), np.std(classAccTrainShfl_allExc_aveS)]
    
ax = plt.subplot(gs[0,4:5])
plt.errorbar(x, y, yerr, marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
ymin, _ = ax.get_ylim()
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
plt.title('Shuffle')
makeNicePlots(ax)

plt.subplot(gs[0,3:4])
plt.ylim([ymin, ymax])


plt.subplots_adjust(wspace=1, hspace=.9)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow, suffn+'ei_train_all'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')



#%% Training data: Plot average across shuffles for each day, inh, exc, allExc each one plot
### L1;
plt.figure(figsize=(6,7.5))
gs = gridspec.GridSpec(3, 5)#, width_ratios=[2, 1]) 

### INH
ax = plt.subplot(gs[0,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTrainData_inh_aveS, yerr = classAccTrainData_inh_sdS, color='g', label='Data')
plt.errorbar(range(numDays), classAccTrainShfl_inh_aveS, yerr = classAccTrainShfl_inh_sdS, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
#plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, [np.mean(classAccTrainData_inh_aveS), np.mean(classAccTrainShfl_inh_aveS)], yerr = [np.std(classAccTrainData_inh_aveS), np.std(classAccTrainShfl_inh_aveS)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#av_l1_test_d = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#av_l1_test_s = 100-np.nanmean(l1_err_test_shfl0, axis=0) 
#_,p = stats.ttest_ind(av_l1_test_d, av_l1_test_s)
#plt.title('p= %.3f' %(p))
plt.title('Inh')

makeNicePlots(ax)



### EXC
ax = plt.subplot(gs[1,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTrainData_exc_aveS, yerr = classAccTrainData_exc_sdS, color='g', label='Data')
plt.errorbar(range(numDays), classAccTrainShfl_exc_aveS, yerr = classAccTrainShfl_exc_sdS, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(training data)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[1,3:4])
plt.errorbar(x, [np.mean(classAccTrainData_exc_aveS), np.mean(classAccTrainShfl_exc_aveS)], yerr = [np.std(classAccTrainData_exc_aveS), np.std(classAccTrainShfl_exc_aveS)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#av_l1_test_d = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#av_l1_test_s = 100-np.nanmean(l1_err_test_shfl0, axis=0) 
#_,p = stats.ttest_ind(av_l1_test_d, av_l1_test_s)
#plt.title('p= %.3f' %(p))
plt.title('Exc')

makeNicePlots(ax)



### All Exc
ax = plt.subplot(gs[2,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), classAccTrainData_allExc_aveS, yerr = classAccTrainData_allExc_sdS, color='g', label='Data')
plt.errorbar(range(numDays), classAccTrainShfl_allExc_aveS, yerr = classAccTrainShfl_allExc_sdS, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
#plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[2,3:4])
plt.errorbar(x, [np.mean(classAccTrainData_allExc_aveS), np.mean(classAccTrainShfl_allExc_aveS)], yerr = [np.std(classAccTrainData_allExc_aveS), np.std(classAccTrainShfl_allExc_aveS)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
#av_l1_test_d = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#av_l1_test_s = 100-np.nanmean(l1_err_test_shfl0, axis=0) 
#_,p = stats.ttest_ind(av_l1_test_d, av_l1_test_s)
#plt.title('p= %.3f' %(p))
plt.title('All exc')

makeNicePlots(ax)



plt.subplots_adjust(wspace=1, hspace=.7)


if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow, suffn+'ei_train_sep'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')  






   