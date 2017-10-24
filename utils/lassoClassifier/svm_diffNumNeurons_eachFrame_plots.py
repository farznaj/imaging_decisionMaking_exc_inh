# -*- coding: utf-8 -*-
"""
Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     


#%% Change the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19' # if you want to use only one mouse, make sure you put comma at the end; eg. mice = 'fni19',

doPlotsEachMouse = 0
savefigs = 0
types2an = [0,1,2]#[0,1] # inh,allN,exc; range(3) # normally we want all, unless exc is not ready yet!

ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone.
corrTrained = 1 # you did the eachFrame diffNeuronNum analysis using corr trials.
numShufflesExc = 50 # this number you need to know from svm_diffNumNeurons_eachFrame.py
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")
#execfile("svm_plots_setVars_n.py")  
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.




#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc=[], regressBins=3, useEqualTrNums=1, outcome2ana='corr'):
    import glob

    if chAl==1:
        al = 'chAl'
    else:
        al = 'stAl'
            
    if outcome2ana == 'corr': # save incorr vars too bc SVM was trained on corr, and tested on icorr.
        o2a = '_corr'
    else:
        o2a = ''    
        
    if doInhAllexcEqexc[0] == 1:
        ntype = 'inh'
    elif doInhAllexcEqexc[1] == 1:
        ntype = 'allNeur'
    elif doInhAllexcEqexc[2] == 1:
        ntype = 'eqExc' 
    
        
    if trialHistAnalysis:
        if useEqualTrNums:
            svmn = 'diffNumNs_excInh_SVMtrained_eachFrame_%s_prevChoice_%s%s_ds%d_eqTrs_*' %(ntype,al,o2a,regressBins)
        else:
            svmn = 'diffNumNs_excInh_SVMtrained_eachFrame_%s_prevChoice_%s%s_ds%d_*' %(ntype,al,o2a,regressBins)
    else:
        if useEqualTrNums:
            svmn = 'diffNumNs_excInh_SVMtrained_eachFrame_%s_currChoice_%s%s_ds%d_eqTrs_*' %(ntype,al,o2a,regressBins)
        else:
            svmn = 'diffNumNs_excInh_SVMtrained_eachFrame_%s_currChoice_%s%s_ds%d_*' %(ntype,al,o2a,regressBins)
#    print '\n', svmn[:-1]        
        
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
#    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    # below finds that most recent analyzed file (based on the date of the analysis). Above will be problematic if an earlier is modified after the latest file
    dateAna = []
    for isv in range(len(svmName)):
        dateAna.append(svmName[isv][str.find(svmName[isv], 'eqTrs')+6:str.find(svmName[isv], 'eqTrs')+12])
        
    inds = np.argsort(dateAna)
    svmName = np.array(svmName)[inds][::-1]  # get the file analyzed the latest
        
    return svmName
    

           
#%% 
#####################################################################################################################################################   
#####################################################################################################################################################

#%% Loop through mice

CAav_test_allN_alig_sameMaxN_allMice = [] # average CA across resamps of neurons
CAav_test_inh_alig_sameMaxN_allMice = []
CAav_test_exc_alig_sameMaxN_allMice = []
sdCA_allExcSamps_maxN_ave_allMice = []
CAsd_test_allN_alig_sameMaxN_allMice = [] # sd of CA across resamps of neurons
CAsd_test_inh_alig_sameMaxN_allMice = []
CAsd_test_exc_alig_sameMaxN_allMice = []
time_aligned_allMice = []
    
for im in range(len(mice)):
        
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    if mousename == 'fni18': #set one of the following to 1:
        allDays = 1# all 7 days will be used (last 3 days have z motion!)
        noZmotionDays = 0 # 4 days that dont have z motion will be used.
        noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    if mousename == 'fni19':    
        allDays = 1
        noExtraStimDays = 0   
    
    execfile("svm_plots_setVars_n.py")      

    
    #%% 
    dnow = '/classAccur_diffNumNeurons_eachFrame/'+mousename+'/'
            
            
    #%% Loop over days    
    
    perClassErrorTrain_inh_all = []
    perClassErrorTest_data_inh_all = []
    perClassErrorTest_shfl_inh_all = []
    perClassErrorTest_chance_inh_all = []
    perClassErrorTrain_allN_all = []
    perClassErrorTest_data_allN_all = []
    perClassErrorTest_shfl_allN_all = []
    perClassErrorTest_chance_allN_all = []
    perClassErrorTrain_exc_all = []
    perClassErrorTest_data_exc_all = []
    perClassErrorTest_shfl_exc_all = []
    perClassErrorTest_chance_exc_all = []
    
    numInh = np.full((len(days)), np.nan)
    numAlln = np.full((len(days)), np.nan)
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
    eventI_allDays = np.full((len(days)), np.nan)
    eventI_ds_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
       
    for iday in range(len(days)): 
    
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
    
    
        #%% Set eventI (downsampled)
                
        if chAl==1:    #%% Use choice-aligned traces 
            # Load 1stSideTry-aligned traces, frames, frame of event of interest
            # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
            Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)        
            traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
            time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
            time_trace = time_aligned_1stSide
            eventI = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!           
            
        else:   #%% Use stimulus-aligned traces           
            # Load stim-aligned_allTrials traces, frames, frame of event of interest
            if trialHistAnalysis==0:
                Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
    #            eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
    #            traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
                time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')        
            else:
                Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
    #            eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    #            traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
                time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
                # time_aligned_stimAll = Data['stimAl_allTrs'].time.astype('float') # same as time_aligned_stim        
            
            time_trace = time_aligned_stim        
        print(np.shape(time_trace))
        eventI_allDays[iday] = eventI  
        
        ##%% Downsample traces: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)    
        if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
            print 'Downsampling traces ....'    
                
            # set frames before frame0 (not including it)
            f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)            
            T1 = f.shape[0]
            eventI_ds = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0
        
        else:
            print 'Not downsampling traces ....'        
    #        eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    #        eventI_ds = eventI_ch
        
        eventI_ds_allDays[iday] = eventI_ds
    
    
        #%% Get number of hr, lr trials that were used for svm training and for incorr testing
        
        Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
        outcomes = (Data.pop('outcomes').astype('float'))[0,:]
        allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    #    print '%d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))
        
        
        svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins)[0] #inh,exc,allN have the same trsExcluded, so just load one of them.       
        Data = scio.loadmat(svmName, variable_names=['trsExcluded'])
        trsExcluded = Data.pop('trsExcluded').flatten().astype('bool')
        
        corr_hr = sum(np.logical_and(allResp_HR_LR==1 , ~trsExcluded)).astype(int)
        corr_lr = sum(np.logical_and(allResp_HR_LR==0 , ~trsExcluded)).astype(int)    
    #    print min(corr_hr, corr_lr) # number of trials of each class used in svm training
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
        
        
        
        #%% Load SVM vars
        
        for idi in types2an: # inh,allN,exc
            doInhAllexcEqexc = np.full((3), False)
            doInhAllexcEqexc[idi] = True 
            svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins)        
            svmName = svmName[0]
            print os.path.basename(svmName)    
    
            if doInhAllexcEqexc[0] == 1:
    #            Datai = scio.loadmat(svmName, variable_names=['perClassErrorTrain_nN_inh', 'perClassErrorTest_nN_inh', 'perClassErrorTest_shfl_nN_inh', 'perClassErrorTest_chance_nN_inh'])               
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_nN_inh'])               
                   #   'perClassErrorTrain_nN_inh', 'perClassErrorTest_nN_inh', 'perClassErrorTest_shfl_nN_inh', 'perClassErrorTest_chance_nN_inh', 'wAllC_nN_inh', 'bAllC_nN_inh'])
            elif doInhAllexcEqexc[1] == 1:
    #            Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTrain_nN_allN', 'perClassErrorTest_nN_allN', 'perClassErrorTest_shfl_nN_allN', 'perClassErrorTest_chance_nN_allN'])
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_nN_allN'])
            elif doInhAllexcEqexc[2] == 1:
    #            Datae = scio.loadmat(svmName, variable_names=['perClassErrorTrain_nN_exc', 'perClassErrorTest_nN_exc', 'perClassErrorTest_shfl_nN_exc', 'perClassErrorTest_chance_nN_exc'])                                                 
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_nN_exc'])                                                 
        
            
        ###%%       
        if np.in1d(0,types2an):        
            perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_nN_inh').flatten()  # len: num neurons; each element has size: # numShufflesN x nSamples x nFrs        
    #        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_nN_inh').flatten()
    #        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_nN_inh').flatten()
    #        perClassErrorTrain_inh = Datai.pop('perClassErrorTrain_nN_inh').flatten()
        #    w_data_inh = Datai.pop('w_data_inh') 
    
        if np.in1d(1,types2an):        
            perClassErrorTest_data_allN = Dataae.pop('perClassErrorTest_nN_allN').flatten()  # len: num neurons; each element has size: # numShufflesN x nSamples x nFrs        
    #        perClassErrorTest_shfl_allN = Dataae.pop('perClassErrorTest_shfl_nN_allN').flatten()
    #        perClassErrorTest_chance_allN = Dataae.pop('perClassErrorTest_chance_nN_allN').flatten()
    #        perClassErrorTrain_allN = Dataae.pop('perClassErrorTrain_nN_allN').flatten()        
            #    w_data_allExc = Dataae.pop('w_data_allExc') 
        
        if np.in1d(2,types2an):
            # by using flatten you are pooling all numShuflExc data, so the length of the arrays below would be (numShuflExc x num neurons)
            # later you need to do reshape(perClassErrorTest_data_exc, (numShuflExc x num neurons)); remember size of perClassErrorTest_nN_exc is (numShuflExc x num neurons)        
            perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_nN_exc').flatten()  # len: numShuflExc x num neurons ; each element has size: # numShufflesN x nSamples x nFrs        
    #        perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_nN_exc'.flatten()
    #        perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_nN_exc').flatten()
    #        perClassErrorTrain_exc = Datae.pop('perClassErrorTrain_nN_exc').flatten()
            #    w_data_exc = Datae.pop('w_data_exc') 
        
                            
       
       
        ###%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.
        '''
        perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_exc = setTo50classErr(perClassErrorTest_data_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_exc = setTo50classErr(perClassErrorTest_shfl_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_exc = setTo50classErr(perClassErrorTest_chance_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        '''
    
        #%% Get number of allN and exc
    
        if np.in1d(0,types2an):        
            numInh[iday] = perClassErrorTest_data_inh.shape[0]
        if np.in1d(1,types2an):                
            numAlln[iday] = perClassErrorTest_data_allN.shape[0]
    

        #%% Once done with all frames, save vars for all days
        ######################################################################################################################################################    
        ######################################################################################################################################################          
           
        if np.in1d(0,types2an):        
            perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # len: num days; each day: num neurons; each neuron: numShufflesN x samps x numFrs    
    #        perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
    #        perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
    #        perClassErrorTrain_inh_all.append(perClassErrorTrain_inh)
    #        del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh
        if np.in1d(1,types2an):        
            perClassErrorTest_data_allN_all.append(perClassErrorTest_data_allN) # len: num days; each day: num neurons; each neuron: numShufflesN x samps x numFrs    
    #        perClassErrorTest_shfl_allN_all.append(perClassErrorTest_shfl_allN)
    #        perClassErrorTest_chance_allN_all.append(perClassErrorTest_chance_allN) 
    #        perClassErrorTrain_allN_all.append(perClassErrorTrain_allN)
    #        del perClassErrorTest_data_allN, perClassErrorTest_shfl_allN, perClassErrorTest_chance_allN
        if np.in1d(2,types2an):                
            perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # len: num days; each day: (numShuflExc x num neurons); each element: numShufflesN x samps x numFrs    
    #        perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
    #        perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)
    #        perClassErrorTrain_exc_all.append(perClassErrorTrain_exc)
    #        del perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    
        # Delete vars before starting the next day    
        
    
    #%%
    eventI_allDays = eventI_allDays.astype(int)
    eventI_ds_allDays = eventI_ds_allDays.astype(int)
    
    perClassErrorTest_data_inh_all = np.array(perClassErrorTest_data_inh_all)
    #perClassErrorTest_shfl_inh_all = np.array(perClassErrorTest_shfl_inh_all)
    #perClassErrorTest_chance_inh_all = np.array(perClassErrorTest_chance_inh_all)
    perClassErrorTest_data_allN_all = np.array(perClassErrorTest_data_allN_all)
    #perClassErrorTest_shfl_allExc_all = np.array(perClassErrorTest_shfl_allExc_all)
    #perClassErrorTest_chance_allExc_all = np.array(perClassErrorTest_chance_allExc_all)
    perClassErrorTest_data_exc_all = np.array(perClassErrorTest_data_exc_all)
    #perClassErrorTest_shfl_exc_all = np.array(perClassErrorTest_shfl_exc_all)
    #perClassErrorTest_chance_exc_all = np.array(perClassErrorTest_chance_exc_all)



    
    #%% Keep the original value of perClassErrorTest_data_exc_all before averaging across numShflExc
    # NOT SURE if you want to do this, bc this array is huge in size and consumes a lot of memory!!
    '''
    #perClassErrorTest_data_exc_all0 = perClassErrorTest_data_exc_all + 0
    import copy
    perClassErrorTest_data_exc_all0 = copy.deepcopy(perClassErrorTest_data_exc_all) 
    
    '''
    #%% For exc, average across numShuflExc (assuming different subsamples of exc neurons were representing the same underlying population)
    
#    numShufflesExc = 50 # this number you need to know from svm_diffNumNeurons_eachFrame.py    
    # reshape so we get data from each numShuflExc
    perClassErrorTest_data_exc_all_eachExcShfl = [np.reshape(perClassErrorTest_data_exc_all[iday], (numShufflesExc,numInh[iday])) for iday in range(len(days))] # each day: numShuflExc x num neurons; each pair of elements (ie a[i,j]): numShufflesN x samps x numFrs
    
    # Now average across numShuflExc
    # Now perClassErrorTest_data_exc_all has the same size as perClassErrorTest_data_inh_all
    # len: num days; each day: num neurons; each neuron: numShufflesN x samps x numFrs    
    perClassErrorTest_data_exc_all = np.array([np.mean(perClassErrorTest_data_exc_all_eachExcShfl[iday], axis=0) for iday in range(len(days))]) # each day: num neurons; each element: numShufflesN x samps x numFrs
    
    
    
    
    #%% Set vars for each mouse
    
    ######################################################################################################################################################    
    ######################################################################################################################################################          
    
    #%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
    
    # th for min number of trs of each class
    thTrained = 10 #30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training   
    mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
    print 'num days to be excluded:\n\tlow trained trs:', sum(mn_corr < thTrained)   
    print np.array(days)[mn_corr < thTrained]
    
    days2use = mn_corr>=thTrained    
    
    
    #%% Set vars for aligning class accur traces of all days on common eventI
     
    ##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
            
    nPost = (np.ones((numDays,1))+np.nan).flatten()
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained:
            nPost[iday] = (perClassErrorTest_data_inh_all[iday][0].shape[-1] - eventI_ds_allDays[iday] - 1)
    
    nPreMin = int(min(eventI_ds_allDays)) # number of frames before the common eventI, also the index of common eventI. 
    nPostMin = int(np.nanmin(nPost))
    print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)
    
    
    #%% Set the time array for the across-day aligned traces
    
    # totLen_ds = len(av_l2_test_diday)
    def set_time_al(totLen_ds, eventI):
        # totLen_ds: length of downsample trace
        # eventI : eventI on the original trace (non-downsampled)
    #    eventI = eventI_allDaysiday #np.argwhere(time_trace==0).flatten()
        time_trace = frameLength * (np.arange(0, np.ceil(regressBins*totLen_ds)) - eventI) # time_trace = time_aligned_1stSide
        
        f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
        x = time_trace[f] # time_trace including frames before frame0
        T1 = x.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0 # same as eventI_ds
        xdb = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames before frame0
        
        
        # set frames after frame0 (not including it)
        f = (np.arange(eventI+1 , eventI+1+regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
        x = time_trace[f] # X_svm including frames after frame0
        T1 = x.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
        xda = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames after frame0
        
        # set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
        time_trace_d = np.concatenate((xdb, [0], xda))    
        time_al = time_trace_d[0:int(totLen_ds)]
    
        return time_al
        
        
    #######################################    
    
    totLen = nPreMin + nPostMin +1
    time_aligned = set_time_al(totLen, min(eventI_allDays))
    
    print time_aligned
    
    
    
    #%%
    #######################################    #######################################    
    #######################################    #######################################    
    #######################################    #######################################    
    
    # take average across cv samps for each neuron samp; then compute ave and sd across neuron samps, so we get class accuracy for each frame when a particular number of neurons was used for training.
    
    def avCVnNsamps(perClassError_all):
       
        CAav_all = []
        CAsd_all = []
        CAse_all = []
        
        for iday in range(len(days)):
        
            a = 100 - perClassError_all[iday] # len: num neurons used for training;
            
            # Take ave across cv samps for each neuron samp
            c = np.array([np.nanmean(a[nN], axis=1) for nN in range(len(a))]) # len: num neurons used for training; each element's size: numShufflesN x nfrs
            # the following is not true anymore, bc above you took average across numShflExc: if exc, len c: numShufflesExc x num neurons used for training; each element's size: numShufflesN x nfrs
    
            # Now take ave and sd across numShufflesN
            # output array size: # num neurons used for training x nfrs... dav[i,j] class accur when i neurons were used for svm training on frame j
            # the following is not true anymore, bc above you took average across numShflExc: if exc: output array size: # (numShufflesExc x num neurons used for training) x nfrs... dav[i,j] class accur when i neurons were used for svm training on frame j
            dav = np.array([np.nanmean(c[nN], axis=0) for nN in range(len(a))]) # num neurons used for training x nfrs... dav[i,j] class accur when i neurons were used for svm training on frame j
            dsd = np.array([np.nanstd(c[nN], axis=0) for nN in range(len(a))])
            dse = np.array([np.nanstd(c[nN], axis=0)/np.sqrt(c[nN].shape[0]) for nN in range(len(a))])
            
            CAav_all.append(dav)
            CAsd_all.append(dsd)
            CAse_all.append(dse)
            
            if doPlots:
                plt.figure()
                plt.imshow(dav); plt.colorbar() # num neurons used for training x nfrs
                plt.figure()
                plt.imshow(dse); plt.colorbar()
                plt.title(days[iday][0:6])
                
                # compare se and sd (across neuron samps) vs numneurons used for training svm
                d1 = np.mean(dsd, axis=1)
                d2 = np.mean(dse, axis=1)
                
                plt.figure(figsize=(4,4))
                plt.plot(d1,label='sd')
                plt.plot(d2,label='se')
                plt.legend()
                plt.title(days[iday][0:6])        
        
        CAav_all = np.array(CAav_all) # numDays #each day: num neurons used for training x nfrs.
        CAsd_all = np.array(CAsd_all)
        CAse_all = np.array(CAse_all)
        
        return CAav_all, CAsd_all, CAse_all
        
       
    
    ###%% same as above but for exc (it has nShflExc, so we have to do the averaging differently!)
    # for each nShflExc, we do exactly what we did above, ie:
    # take average across cv samps for each neuron samp; then compute ave and sd across neuron samps, so we get class accuracy for each frame when a particular number of neurons was used for training.
    
    def avCVnNsamps_exc(perClassError_all):
       
        CAav_all = []
        CAsd_all = []
        CAse_all = []
    
        for iday in range(len(days)):
    
            # perClassErrorTest_data_exc_all_eachExcShfl[iday][anyExcShfl][0 (ie trainingWith1Neuron)] size: numShufflesN x samps x numFrs
            nNs = perClassError_all[iday][0][0].shape[0]
            nFrs = perClassError_all[iday][0][0].shape[2]
            CAav_alle = np.full((nNs, nFrs, numShufflesExc), np.nan)
            CAsd_alle = np.full((nNs, nFrs, numShufflesExc), np.nan)
            CAse_alle = np.full((nNs, nFrs, numShufflesExc), np.nan)
            
            for nExcSamp in range(numShufflesExc):
    
                a = perClassError_all[iday][nExcSamp] # num neurons; each neuron: numShufflesN x samps x numFrs   
                a = 100 - a # len: num neurons used for training;
                
                # Take ave across cv samps for each neuron samp
                c = np.array([np.nanmean(a[nN], axis=1) for nN in range(len(a))]) # len: num neurons used for training; each element's size: numShufflesN x nfrs
                # the following is not true anymore, bc above you took average across numShflExc: if exc, len c: numShufflesExc x num neurons used for training; each element's size: numShufflesN x nfrs
        
                # Now take ave and sd across numShufflesN
                # output array size: # num neurons used for training x nfrs... dav[i,j] class accur when i neurons were used for svm training on frame j
                # the following is not true anymore, bc above you took average across numShflExc: if exc: output array size: # (numShufflesExc x num neurons used for training) x nfrs... dav[i,j] class accur when i neurons were used for svm training on frame j
                dav = np.array([np.nanmean(c[nN], axis=0) for nN in range(len(a))]) # num neurons used for training x nfrs... dav[i,j] class accur when i neurons were used for svm training on frame j
                dsd = np.array([np.nanstd(c[nN], axis=0) for nN in range(len(a))])
                dse = np.array([np.nanstd(c[nN], axis=0)/np.sqrt(c[nN].shape[0]) for nN in range(len(a))])
                
                CAav_alle[:,:,nExcSamp] = dav # num neurons used for training x nfrs x nExcShfls
                CAsd_alle[:,:,nExcSamp] = dsd
                CAse_alle[:,:,nExcSamp] = dse
            
            CAav_all.append(CAav_alle)
            CAsd_all.append(CAsd_alle)
            CAse_all.append(CAse_alle)
        
            if doPlots:
                plt.figure()
                plt.imshow(dav); plt.colorbar() # num neurons used for training x nfrs
                plt.figure()
                plt.imshow(dse); plt.colorbar()
                plt.title(days[iday][0:6])
                
                # compare se and sd (across neuron samps) vs numneurons used for training svm
                d1 = np.mean(dsd, axis=1)
                d2 = np.mean(dse, axis=1)
                
                plt.figure(figsize=(4,4))
                plt.plot(d1,label='sd')
                plt.plot(d2,label='se')
                plt.legend()
                plt.title(days[iday][0:6])        
        
        CAav_all = np.array(CAav_all) # numDays #each day: num neurons used for training x nfrs.
        CAsd_all = np.array(CAsd_all)
        CAse_all = np.array(CAse_all)
        
        return CAav_all, CAsd_all, CAse_all
        
           
    #%% Take average across cv samps for each neuron samp; then compute ave and sd across neuron samps, so we get class accuracy for each frame when a particular number of neurons was used for training.
    # output array size for inh and allN: numDays; each day: nNeurons x nFrs
    # output array size for exc: numDays; each day: nNeurons x nFrs x nExcShfl
    # input array len: num days; each day: num neurons; each neuron: numShufflesN x samps x numFrs     
    # the following is not true anymore, bc above you took average across numShflExc: # exc input array len:  num days; each day: numShuflExc x num neurons; each neuron: numShufflesN x samps x numFrs    
        
    doPlots = 0 # plots of each day
    
    CAav_test_inh_all, CAsd_test_inh_all, CAse_test_inh_all = avCVnNsamps(perClassErrorTest_data_inh_all)
    #CAav_test_shfl_inh_all, CAsd_test_shfl_inh_all, CAse_test_shfl_inh_all = avCVnNsamps(perClassErrorTest_shfl_inh_all)
    #CAav_test_chance_inh_all, CAsd_test_chance_inh_all, CAse_test_chance_inh_all = avCVnNsamps(perClassErrorTest_chance_inh_all)
    #CAav_train_inh_all, CAsd_train_inh_all, CAse_train_inh_all = avCVnNsamps(perClassErrorTrain_inh_all)    
    
    CAav_test_allN_all, CAsd_test_allN_all, CAse_test_allN_all = avCVnNsamps(perClassErrorTest_data_allN_all)
    #CAav_test_shfl_allN_all, CAsd_test_shfl_allN_all, CAse_test_shfl_allN_all = avCVnNsamps(perClassErrorTest_shfl_allN_all)
    #CAav_test_chance_allN_all, CAsd_test_chance_allN_all, CAse_test_chance_allN_all = avCVnNsamps(perClassErrorTest_chance_allN_all)
    #CAav_train_allN_all, CAsd_train_allN_all, CAse_train_allN_all = avCVnNsamps(perClassErrorTrain_allN_all)    
    
    # output array size for exc: numDays; each day: nNeurons x nFrs x nExcShfl
    CAav_test_exc_all, CAsd_test_exc_all, CAse_test_exc_all = avCVnNsamps_exc(perClassErrorTest_data_exc_all_eachExcShfl) 
    #CAav_test_shfl_exc_all, CAsd_test_shfl_exc_all, CAse_test_shfl_exc_all = avCVnNsamps(perClassErrorTest_shfl_exc_all)
    #CAav_test_chance_exc_all, CAsd_test_chance_exc_all, CAse_test_chance_exc_all = avCVnNsamps(perClassErrorTest_chance_exc_all)
    #CAav_train_exc_all, CAsd_train_exc_all, CAse_train_exc_all = avCVnNsamps(perClassErrorTrain_exc_all)    
    
    #a = perClassErrorTest_data_inh_all[iday]
    # we subselected 1 neuron for b.shape[0] times
    #nN = 0 # 0:len(a)
    #b = a[nN] # numShufflesN x samps x numFrs
    
    #inN = 0 #:b.shape[0] # 1st subsampling
    # now for this subsample, compute average class accur across cv (trial) samples
    #c = np.nanmean(b, axis=1) # numShufflesN x numFrs (average class accur across cv (trial) samples for all neuron samples)
    
    
    #%% Align CAav_all on the common eventI
    
    def alFrs(CAav_all):
        CAav_alig = [] #np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
            
        for iday in range(len(CAav_all)):    
        #    if mn_corr[iday] >= thTrained:
            CAav_alig.append(CAav_all[iday][:, eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]) #nNeurons x nAlignedFrs
        
        CAav_alig = np.array(CAav_alig)
        
        return CAav_alig
    
    
    ### same as above but for exc (bc it has nExcShfls)
    def alFrs_exc(CAav_all):
        CAav_alig = [] #np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
            
        for iday in range(len(CAav_all)):    
        #    if mn_corr[iday] >= thTrained:
            CAav_alig.append(CAav_all[iday][:, eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1, :]) #nNeurons x nAlignedFrs
        
        CAav_alig = np.array(CAav_alig)
        
        return CAav_alig
    
        
    #%% Align CAav_all on the common eventI
    # output array size: numDays; each day: nNeurons x nAlignedFrs
    # input array size: numDays; each day: nNeurons x nFrs
    # for exc:
    # output array size: numDays; each day: nNeurons x nAlignedFrs x nExcShfl
        
    CAav_test_inh_alig = alFrs(CAav_test_inh_all)
    CAsd_test_inh_alig = alFrs(CAsd_test_inh_all)
    #CAav_test_shfl_inh_alig = alFrs(CAav_test_shfl_inh_all)
    #CAav_test_chance_inh_alig = alFrs(CAav_test_chance_inh_all)
    #CAav_train_inh_alig = alFrs(CAav_train_inh_all)
    
    CAav_test_allN_alig = alFrs(CAav_test_allN_all)
    CAsd_test_allN_alig = alFrs(CAsd_test_allN_all)
    #CAav_test_shfl_allN_alig = alFrs(CAav_test_shfl_allN_all)
    #CAav_test_chance_allN_alig = alFrs(CAav_test_chance_allN_all)
    #CAav_train_allN_alig = alFrs(CAav_train_allN_all)
    
    # exc: output array size: numDays; each day: nNeurons x nAlignedFrs x nExcShfl
    CAav_test_exc_alig = alFrs_exc(CAav_test_exc_all)
    CAsd_test_exc_alig = alFrs_exc(CAsd_test_exc_all)
    #CAav_test_shfl_exc_alig = alFrs(CAav_test_shfl_exc_all)
    #CAav_test_chance_exc_alig = alFrs(CAav_test_chance_exc_all)
    #CAav_train_exc_alig = alFrs(CAav_train_exc_all)
    
    
    
    #%% for each excSamp compute std across days
    
    def sdEachExcSamp(CAav):
        sdCA_allExcSamps = np.full(np.array(CAav.shape)[[1,2,3]], np.nan)
        for nExcSamp in range(numShufflesExc):
            sd_ca = np.nanstd(CAav[:,:,:,nExcSamp], axis=0)
            sdCA_allExcSamps[:,:,nExcSamp] = sd_ca
        
        sdCA_allExcSamps = np.array(sdCA_allExcSamps)
        
        return sdCA_allExcSamps
    
    
    #### instead of going with min (at the end of this script), go with lowest 20 percentile of neuron nums across days, so we get CA vs nN upto larger values of nN.
    
    #%% Add nan for days with fewer neurons, so we can get ave across days using as many neurons as possible
    
    def sameMaxNs(CAav_alig, mxNumNeur):
        CAav_alig_sameMaxN = []
        for iday in range(len(days)):
            a = np.full((mxNumNeur, len(time_aligned)), np.nan)
            a[0: min(mxNumNeur, CAav_alig[iday].shape[0]) , :] = CAav_alig[iday][0: min(mxNumNeur, CAav_alig[iday].shape[0])]
            CAav_alig_sameMaxN.append(a) # numDays x max_numNeurs_used_for_training x nFrs
    
        CAav_alig_sameMaxN = np.array(CAav_alig_sameMaxN)
        return CAav_alig_sameMaxN 
        
        
    ### Same as above but for exc
    def sameMaxNs_exc(CAav_alig, mxNumNeur):
        CAav_alig_sameMaxN = []
        for iday in range(len(days)):
            a = np.full((mxNumNeur, len(time_aligned), CAav_alig[iday].shape[2]), np.nan) # max_numNeurs_used_for_training x nFrs x nExcShfl
            a[0: min(mxNumNeur, CAav_alig[iday].shape[0]) , :,:] = CAav_alig[iday][0: min(mxNumNeur, CAav_alig[iday].shape[0])]
            CAav_alig_sameMaxN.append(a) # numDays x max_numNeurs_used_for_training x nFrs x nExcShfl
    
        CAav_alig_sameMaxN = np.array(CAav_alig_sameMaxN)
        return CAav_alig_sameMaxN 
        
        
    
        
    #%% Add nan for days with fewer neurons, so we can get ave across days using as many neurons as possible
    # output array size: numDays x max_nNeurs x nAlignedFrs
    # input array size: numDays; each day: nNeurons x nAlignedFrs
    # for exc:
    # output array size: numDays x max_nNeurs x nAlignedFrs x nExcShfl
    # input array size: numDays; each day: nNeurons x nAlignedFrs x nExcShfl
        
    #mxNumNeur = max(numInh).astype(int)    
    mxNumNeur = np.percentile(numInh,20).astype(int) # dont go with max, bc only very few sessions will have that many neurons, also dont go with min so you dont loose all data from sessions with more neurons, instead go with the 20th percentile of neuron numbers across days, so we can see how CA changes with nN using higher values of nN, but still making sure most days contributed to the plot.
    CAav_test_inh_alig_sameMaxN = sameMaxNs(CAav_test_inh_alig, mxNumNeur)
    CAsd_test_inh_alig_sameMaxN = sameMaxNs(CAsd_test_inh_alig, mxNumNeur)
    
    #mxNumNeur = max(numInh).astype(int)    
    mxNumNeur = np.percentile(numInh,20).astype(int)
    CAav_test_exc_alig_sameMaxN = sameMaxNs_exc(CAav_test_exc_alig, mxNumNeur)
    CAsd_test_exc_alig_sameMaxN = sameMaxNs_exc(CAsd_test_exc_alig, mxNumNeur)
    
    # allN    
    #mxNumNeur = max(numAlln).astype(int)    
    mxNumNeur = np.percentile(numAlln,20).astype(int)
    CAav_test_allN_alig_sameMaxN = sameMaxNs(CAav_test_allN_alig, mxNumNeur)
    CAsd_test_allN_alig_sameMaxN = sameMaxNs(CAsd_test_allN_alig, mxNumNeur)
    
    
    #%% Keep original vals of exc before averaging across nExcShfl
    
    CAav_test_exc_alig_sameMaxN0 = CAav_test_exc_alig_sameMaxN+0
    CAsd_test_exc_alig_sameMaxN0 = CAsd_test_exc_alig_sameMaxN+0 # this is sd for each day (across samps)
    
    
    #%% for each excSamp compute std across days
    
    sdCA_allExcSamps_maxN = sdEachExcSamp(CAav_test_exc_alig_sameMaxN0[days2use]) # max_nNeurs_used_for_training x nAlignedFrs x nExcShfl
    # now take ave of sd across excSamps
    sdCA_allExcSamps_maxN_ave = np.mean(sdCA_allExcSamps_maxN, axis=2) # max_nNeurs_used_for_training x nAlignedFrs
    
    
    #%% Now for exc take average across nExcShfls, so it will have same size as inh and allN
    # output array size: numDays x min_nNeurs_used_for_training x nAlignedFrs
    # input array size for exc: numDays x min_nNeurs_used_for_training x nAlignedFrs x nExcShfl
    
    CAav_test_exc_alig_sameMaxN = np.mean(CAav_test_exc_alig_sameMaxN0, axis=3)
    CAsd_test_exc_alig_sameMaxN = np.mean(CAsd_test_exc_alig_sameMaxN0, axis=3)
    

    #%%    
    del perClassErrorTest_data_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_data_allN_all, perClassErrorTest_data_inh, perClassErrorTest_data_exc, perClassErrorTest_data_allN
    
    
    #%%    
    ####################################################################################################
    ######################### PLOTS of each mouse #########################
    ####################################################################################################
    
    if doPlotsEachMouse:
        #%% plot ave and sd (across days) of CA vs. neurNum for each time bin
        
        ####%% For each num_neuron_usedForTraining, compute whether CA (acorss all days) is diff btwn exc and inh. Do this for each time bin separately.
        '''
        def rmvNans(a): # remove nans
            mask = ~np.isnan(a)
        #    if isinstance(a,list):
            a = np.array([d[m] for d, m in zip(a, mask)])
        #    else:
        #        a = a[mask]
            return a
        '''
        def pwmt_frs(a,b): # a and b: nDays x nNerus
            nN = a.shape[1]
            
        #    a = rmvNans(a)
        #    b = rmvNans(b)    
            
            if np.shape(a)[1]>0 and np.shape(b)[1]>0:
                # loop over nN to compare across days a and b
                p0  = np.array([sci.stats.ranksums(a[:,n], b[:,n])[1] for n in range(b.shape[1])]) # nNerus
                p1  = np.array([sci.stats.mannwhitneyu(a[:,n], b[:,n])[1] for n in range(b.shape[1])]) # nNerus
        #        _,p2  = sci.stats.ttest_ind(a, b) # nNerus
                # for ttest we cant have nans, so remove them 
                p2 = np.full((b.shape[1]),np.nan) 
                for n in range(b.shape[1]): # loop over each nN
                    aa = a[:,n]; aa = aa[~np.isnan(aa)]
                    bb = b[:,n]; bb = bb[~np.isnan(bb)]
                    _,p20  = sci.stats.ttest_ind(aa, bb) 
                    p2[n] = p20  # nNerus
                    
                p = [p0,p1,p2]
                p = np.array(p)
            else:
                p = np.full((3,nN),np.nan)
            
            return p
            
        
        ###############################
        def plotCA_nN(CA, labs, cols, excInd=-1, sdDexc=None, doP=0): # excInd is the index of exc in array CAav_alig_sameN... default is -1 so assuming we will get sd across days and not have it as an input
             
            nDays = np.shape(CA[0])[0]
            nNcommon = np.shape(CA[0])[1]
            nTimeBins = np.shape(CA[0])[2] # size CAav_alig_sameN[0]: numDays x max_nNeurs x nAlignedFrs
        
        
            if doP: # compute p value
                pwmt = np.full((3,nNcommon,nTimeBins), np.nan)  # 3 x max_nNeurs x nAlignedFrs
                
            plt.figure(figsize=(8,10))    
            c = 4.
            r = np.ceil(nTimeBins/c)    
        
            for f in range(nTimeBins): # loop over frames
                    
                plt.subplot(r,c,f+1)
                for ntyp in range(len(CA)): 
                    a = CA[ntyp][:,:,f] # nDays x nNerus
                #    plt.imshow(a); plt.colorbar()     
                    avD = np.nanmean(a,axis=0) # average across days
                    
                    if excInd!=ntyp:
                        # compute it by getting sd across days; if sdD is provided use that value instead of taking it across day. we use this for exc, bc there are nExcShfls, so sd of excSufl-averaged traces across days will be lower than sd of inh traces across days bc inh is not averaged... so for exc I use average of sd across samples instead of sd across days
                        sdDNow = np.nanstd(a,axis=0) # stdev across days
                        sdD = sdDNow/np.sqrt(nDays)
                    else:
                        sdD = sdDexc[:,f]/np.sqrt(nDays)                        
                        
                    # plot errorbar
        #            plt.errorbar(range(a.shape[1]), avD, sdD, label=labs[ntyp], color=cols[ntyp])        
                    
                    # fill bounds            
                    plt.fill_between(range(a.shape[1]), avD-sdD, avD+sdD, alpha=0.5, edgecolor=cols[ntyp], facecolor=cols[ntyp])
                    plt.plot(range(a.shape[1]), avD, color=cols[ntyp], linewidth=2, label=labs[ntyp])
        
                plt.title('%.0f ms' %(np.round(time_aligned[f])))
                makeNicePlots(plt.gca())  
                
                if f==0:        
                    plt.xlabel('#Neurons used in decoder')
                    plt.ylabel('Class accuracy (testing data)')        
                if f==nTimeBins-1: 
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False) 
                
                # plot sig p vals
                if doP: # compute p value between exc and inh
                    a = CA[0][:,:,f] # nDays x nNerus
                    b = CA[1][:,:,f]
                    p = pwmt_frs(a,b) # 3 x nNeurons
                    
                    yl = plt.gca().get_ylim()
                    pp = np.full(np.shape(p), np.nan)
                    pp[p<=.05] = yl[1] - (yl[1]-yl[0])/10
                    pp2p = pp[2] # whether to use wilcoxon, mann-whitney, or ttest
                    plt.plot(range(a.shape[1]), pp2p, marker='o', markersize=5, color='k') #, linestyle=':')            
                    
                    pwmt[:,:,f] = p # 3 x nNeurons x nfrs # 3 is the 3 tests: wilcoxon, mann-whitney, or ttest
            
            plt.subplots_adjust(wspace=.5, hspace=.5)
            
            # Save the figure
            if savefigs:#% Save the figure
                    
                if chAl==1:
                    dd = 'chAl_CAvsNnum_' + '_'.join(labs)+'_' + days[0][0:6] + '-to-' + days[-1][0:6]
                else:
                    dd = 'stAl_CAvsNnum_' + '_'.join(labs)+'_' + days[0][0:6] + '-to-' + days[-1][0:6]
                        
                d = os.path.join(svmdir+dnow)
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                        
                fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
          
          
            if doP==0:
                pwmt = []
            else:
                pwmt = np.array(pwmt)
                
            return pwmt
                    
            
        #%% plot ave and sd (across days) of CA vs. neurNum for each time bin
        
        # set what days to use
    #    days2use = mn_corr>=thTrained
        
        labs = 'inh', 'exc'
        cols = 'r', 'b'
        CA = CAav_test_inh_alig_sameMaxN[days2use,:,:], CAav_test_exc_alig_sameMaxN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN(CA, labs, cols, excInd=1, sdDexc=sdCA_allExcSamps_maxN_ave, doP=1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
        
        
        labs = 'allN',
        cols = 'k',
        CA = CAav_test_allN_alig_sameMaxN[days2use,:,:],
        plotCA_nN(CA, labs, cols)    
            
        
        labs = 'allN', 'inh', 'exc'
        cols = 'k', 'r', 'b'
        CA = CAav_test_allN_alig_sameMaxN[days2use,:,:], CAav_test_inh_alig_sameMaxN[days2use,:,:], CAav_test_exc_alig_sameMaxN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        plotCA_nN(CA, labs, cols, excInd=2, sdDexc=sdCA_allExcSamps_maxN_ave)
        
        
        #iexcshfl = 2
        #CA = CAav_test_inh_alig_sameMaxN[days2use,:,:], CAav_test_exc_alig_sameMaxN0[days2use,:,:,iexcshfl]
        #pwmt = plotCA_nN(CA, labs, cols, excInd=1)
        
        
        #%% #%% Learning-related fig: For each timebin, see each day's CA vs nN plot...  
        #each frame, all days
        
        def plotCA_nN_eachDay(CA, CS, labs, cols, doP=0, times2an=[]): # excInd is the index of exc in array CAav_alig_sameN... default is -1 so assuming we will get sd across days and not have it as an input
            
            nDays = np.shape(CA[0])[0] # size CAav_alig_sameN[0]: numDays x max_nNeurs x nAlignedFrs
            nNcommon = np.shape(CA[0])[1]
            nTimeBins = np.shape(CA[0])[2]    
            if np.shape(times2an)[0]!=0:
                nTimeBins = np.shape(times2an.flatten())[0]
                
            if doP: # compute p value
                pwmt = np.full((3,nNcommon,nTimeBins), np.nan)  # 3 x max_nNeurs x nAlignedFrs
                
        #    plt.figure(figsize=(2,30))    
            c = nTimeBins #1.
            r = nDays #np.ceil(nDays/c)    
            plt.figure(figsize=(2.2*nTimeBins , 2*nDays))    
            
            if np.shape(times2an)[0]==0:
                times2an = range(nTimeBins)
            
            cnt=0
            for f in times2an: # loop over frames        
                cnt = cnt+1
                for iday in range(nDays): # loop over frames                    
                    nsp = nTimeBins*iday + cnt  # nsp = nTimeBins*iday + f+1
                    plt.subplot(r,c,nsp)
                        
                    for ntyp in range(len(CA)): 
                        avD = CA[ntyp][iday,:,f]
                        sdD = CS[ntyp][iday,:,f]
                    #    plt.imshow(a); plt.colorbar()                  
                        # plot errorbar
            #            plt.errorbar(range(a.shape[1]), avD, sdD, label=labs[ntyp], color=cols[ntyp])        
                        
                        # fill bounds            
                        plt.fill_between(range(CA[ntyp].shape[1]), avD-sdD, avD+sdD, alpha=0.5, edgecolor=cols[ntyp], facecolor=cols[ntyp])
                        plt.plot(range(CA[ntyp].shape[1]), avD, color=cols[ntyp], linewidth=2, label=labs[ntyp])
            
                    plt.title('%s' %(days[iday][0:6]))
                    makeNicePlots(plt.gca())  
                    
                    if iday==0:            
                        plt.title('%.0f ms; %s' %(np.round(time_aligned[f]), days[iday][0:6]))                
                    
                    if iday==0 and f==0:        
                        plt.xlabel('#Neurons used in decoder')
                        plt.ylabel('Class accuracy (testing data)')        
                        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False) 
        #            if iday==nTimeBins-1:                 
                    
                    # plot sig p vals
                    '''
                    if doP: # compute p value between exc and inh
                        a = CA[0][:,:,f] # nDays x nNerus
                        b = CA[1][:,:,f]
                        p = pwmt_frs(a,b) # 3 x nNeurons
                        
                        yl = plt.gca().get_ylim()
                        pp = np.full(np.shape(p), np.nan)
                        pp[p<=.05] = yl[1] - (yl[1]-yl[0])/10
                        pp2p = pp[2] # whether to use wilcoxon, mann-whitney, or ttest
                        plt.plot(range(a.shape[1]), pp2p, marker='o', markersize=5, color='k') #, linestyle=':')            
                        
                        pwmt[:,:,f] = p # 3 x nNeurons x nfrs # 3 is the 3 tests: wilcoxon, mann-whitney, or ttest
                    '''
            plt.subplots_adjust(hspace=.5, wspace=.5)
            
            # Save the figure
            if savefigs:#% Save the figure
                   
                if chAl==1:
                    dd = 'chAl_CAvsNnum_eachDay_' + '_'.join(labs)+'_' + days[0][0:6] + '-to-' + days[-1][0:6]
                else:
                    dd = 'stAl_CAvsNnum_eachDay_' + '_'.join(labs)+'_' + days[0][0:6] + '-to-' + days[-1][0:6]
                        
                d = os.path.join(svmdir+dnow)
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                        
                fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
          
          
            if doP==0:
                pwmt = []
            else:
                pwmt = np.array(pwmt)
                
            return pwmt
            
                        
        #%% Learning-related fig: For each timebin, see each day's CA vs nN plot...   ave +/- sd 
            # average across cv samps for each neuron samp; then compute ave and sd across neuron samps, so we get class accuracy for each frame when a particular number of neurons was used for training.
        
        labs = 'inh', 'exc'
        cols = 'r', 'b'
        CA = CAav_test_inh_alig_sameMaxN[days2use,:,:], CAav_test_exc_alig_sameMaxN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        CS = CAsd_test_inh_alig_sameMaxN[days2use,:,:], CAsd_test_exc_alig_sameMaxN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN_eachDay(CA, CS, labs, cols, doP=1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
        
        
        # example day: fni17, 151007_1
        """
        labs = 'inh', 'exc'
        cols = 'r', 'b'
        iexcshfl = 12
        CA = CAav_test_inh_alig_sameMaxN[days2use,:,:], CAav_test_exc_alig_sameMaxN0[days2use,:,:,iexcshfl] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        CS = CAsd_test_inh_alig_sameMaxN[days2use,:,:], CAsd_test_exc_alig_sameMaxN0[days2use,:,:,iexcshfl] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN_eachDay(CA, CS, labs, cols, doP=1, times2an=eventI_ds_allDays[days2use]-1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
        plt.xlabel('Number of neurons in the classifier')
        plt.ylabel('Class accuracy (%)')
        
        labs = 'inh', 'exc', 'allN'
        cols = 'r', 'b', 'k'
        iexcshfl = 12
        CA = CAav_test_inh_alig_sameMaxN[days2use,:,:], CAav_test_exc_alig_sameMaxN0[days2use,:,:,iexcshfl], CAav_test_allN_alig_sameMaxN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        CS = CAsd_test_inh_alig_sameMaxN[days2use,:,:], CAsd_test_exc_alig_sameMaxN0[days2use,:,:,iexcshfl], CAsd_test_allN_alig_sameMaxN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN_eachDay(CA, CS, labs, cols, doP=1, times2an=eventI_ds_allDays[days2use]-1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
        plt.xlabel('Number of neurons in the classifier')
        plt.ylabel('Class accuracy (%)')
        
        labs = 'allN',
        cols = 'k',
        iexcshfl = 12
        CA = CAav_test_allN_alig_sameMaxN[days2use,:,:], # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        CS = CAsd_test_allN_alig_sameMaxN[days2use,:,:], # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN_eachDay(CA, CS, labs, cols, doP=1, times2an=eventI_ds_allDays[days2use]-1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
        plt.xlabel('Number of neurons in the classifier')
        plt.ylabel('Class accuracy (%)')
        """
        
        '''
        iday = 0    
        
        labs = 'inh', 'exc'
        cols = 'r', 'b'
        CA = CAav_test_inh_alig_sameMaxN[iday], CAav_test_exc_alig_sameMaxN[iday] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        CS = CAsd_test_inh_alig_sameMaxN[iday], CAsd_test_exc_alig_sameMaxN[iday] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN_eachDay(CA, CS, labs, cols, excInd=1, sdD=sdCA_allExcSamps_maxN_ave, doP=1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
            
            
        f = 0    
        CA = CAav_test_inh_alig_sameMaxN[:,:,f], CAav_test_exc_alig_sameMaxN[:,:,f] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        CS = CAsd_test_inh_alig_sameMaxN[:,:,f], CAsd_test_exc_alig_sameMaxN[:,:,f] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
        #plotCA_nN(CA, labs, cols)
        pwmt = plotCA_nN_eachDay(CA, CS, labs, cols, excInd=1, sdD=sdCA_allExcSamps_maxN_ave, doP=1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
        ''' 
    
        plt.close('all')

    
    
    #%% Keep vars for all mice
    
    ########################################################################################################################
    ########################################################################################################################    
    ########################################################################################################################    

    CAav_test_allN_alig_sameMaxN_allMice.append(CAav_test_allN_alig_sameMaxN[days2use,:,:])
    CAav_test_inh_alig_sameMaxN_allMice.append(CAav_test_inh_alig_sameMaxN[days2use,:,:]) 
    CAav_test_exc_alig_sameMaxN_allMice.append(CAav_test_exc_alig_sameMaxN[days2use,:,:]) # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
    
    CAsd_test_allN_alig_sameMaxN_allMice.append(CAsd_test_allN_alig_sameMaxN[days2use,:,:])
    CAsd_test_inh_alig_sameMaxN_allMice.append(CAsd_test_inh_alig_sameMaxN[days2use,:,:]) 
    CAsd_test_exc_alig_sameMaxN_allMice.append(CAsd_test_exc_alig_sameMaxN[days2use,:,:]) # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs   

    sdCA_allExcSamps_maxN_ave_allMice.append(sdCA_allExcSamps_maxN_ave)
    time_aligned_allMice.append(time_aligned)    
    

#%% Done with all mice    

########################################################################################################################
########################################################################################################################    

dnow0 = '/classAccur_diffNumNeurons_eachFrame/'

#%%    
CAav_test_allN_alig_sameMaxN_allMice = np.array(CAav_test_allN_alig_sameMaxN_allMice)
CAav_test_inh_alig_sameMaxN_allMice = np.array(CAav_test_inh_alig_sameMaxN_allMice)
CAav_test_exc_alig_sameMaxN_allMice = np.array(CAav_test_exc_alig_sameMaxN_allMice) # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
CAsd_test_allN_alig_sameMaxN_allMice = np.array(CAsd_test_allN_alig_sameMaxN_allMice)
CAsd_test_inh_alig_sameMaxN_allMice = np.array(CAsd_test_inh_alig_sameMaxN_allMice)
CAsd_test_exc_alig_sameMaxN_allMice = np.array(CAsd_test_exc_alig_sameMaxN_allMice)
sdCA_allExcSamps_maxN_ave_allMice = np.array(sdCA_allExcSamps_maxN_ave_allMice)
time_aligned_allMice = np.array(time_aligned_allMice)

        
#%%        
def plotCA_nN(CA, labs, cols, excInd=-1, sdDexc=None, doP=0, times2an=[], labels=[]): # excInd is the index of exc in array CAav_alig_sameN... default is -1 so assuming we will get sd across days and not have it as an input
       
    nDays = np.shape(CA[0])[0]       
    nNcommon = np.shape(CA[0])[1]
    nTimeBins = np.shape(CA[0])[2] # size CAav_alig_sameN[0]: numDays x max_nNeurs x nAlignedFrs
    if np.shape(times2an)[0]!=0: # what frames to plot is provided
        nTimeBins = np.shape(times2an.flatten())[0]
#        plt.figure(figsize=(2,2))    
#        r=1;c=1;
#    else:
#        plt.figure(figsize=(8,10))    
#        c = 4.
#        r = np.ceil(nTimeBins/c)    

    if doP: # compute p value
        pwmt = np.full((3,nNcommon,nTimeBins), np.nan)  # 3 x max_nNeurs x nAlignedFrs
        
                
    if np.shape(times2an)[0]==0:
        times2an = range(nTimeBins)
    
#    cnt=0

    for f in times2an: # loop over frames        
#        cnt = cnt+1                
        plt.subplot(r,c,cnt) # f+1
        
        for ntyp in range(len(CA)): 
            a = CA[ntyp][:,:,f] # nDays x nNerus
        #    plt.imshow(a); plt.colorbar()     
            avD = np.nanmean(a,axis=0) # average across days
            
            if excInd!=ntyp:
                # compute it by getting sd across days; if sdD is provided use that value instead of taking it across day. we use this for exc, bc there are nExcShfls, so sd of excSufl-averaged traces across days will be lower than sd of inh traces across days bc inh is not averaged... so for exc I use average of sd across samples instead of sd across days
                sdDNow = np.nanstd(a,axis=0) # stdev across days
                # do se
                sdDNow = np.nanstd(a,axis=0) /np.sqrt(nDays)
                sdD = sdDNow
            else:
                sdD = sdDexc[:,f]
                # do se
                sdD = sdDexc[:,f] /np.sqrt(nDays)
                
            # plot errorbar            
#            x = np.array([0,1])+ntyp/10. #range(a.shape[1])
#            plt.errorbar(x, avD, sdD, label=labs[ntyp], color=cols[ntyp], linestyle='', marker='.')            
#            plt.xticks(x, labels)#, rotation='vertical')
#            plt.xlim([-.5, 1.5+ntyp/10])
            
            # fill bounds            
            plt.fill_between(range(a.shape[1]), avD-sdD, avD+sdD, alpha=0.5, edgecolor=cols[ntyp], facecolor=cols[ntyp])
            plt.plot(range(a.shape[1]), avD, color=cols[ntyp], linewidth=2, label=labs[ntyp])

#        plt.title('%.0f ms' %(np.round(time_aligned[f])))
        plt.title(mice[im])
        makeNicePlots(plt.gca())  
        
        if cnt==2:        
            plt.xlabel('Number of neurons used in decoder')
        if cnt==1:
            plt.ylabel('Class accuracy (%)')        
        if cnt==nTimeBins-1: 
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False) 
        
        # plot sig p vals
        if doP: # compute p value between exc and inh
            a = CA[0][:,:,f] # nDays x nNerus
            b = CA[1][:,:,f]
            p = pwmt_frs(a,b) # 3 x nNeurons
            
            yl = plt.gca().get_ylim()
            pp = np.full(np.shape(p), np.nan)
            pp[p<=.05] = yl[1] - (yl[1]-yl[0])/10
            pp2p = pp[2] # whether to use wilcoxon, mann-whitney, or ttest
            plt.plot(range(a.shape[1]), pp2p, marker='o', markersize=5, color='k') #, linestyle=':')            
            
            pwmt[:,:,cnt-1] = p # 3 x nNeurons x nfrs # 3 is the 3 tests: wilcoxon, mann-whitney, or ttest
    
    plt.subplots_adjust(wspace=.5, hspace=.5)
    
    # Save the figure
    if savefigs:#% Save the figure
            
        if chAl==1:
            dd = 'chAl_CAvsNnum_time-1_' + '_'.join(mice)
        else:
            dd = 'stAl_CAvsNnum_time-1_' + '_'.join(mice)
                
        d = os.path.join(svmdir+dnow0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
  
  
    if doP==0:
        pwmt = []
    else:
        pwmt = np.array(pwmt)
        
    return pwmt


#%% For each mouse plot ave and se (across days) of CA vs. neurNum for the last time bin before the choice

# set what days to use
#    days2use = mn_corr>=thTrained

plt.figure(figsize=(7,1.7))    
r=1
c=len(mice)

labs = 'inh', 'exc'
cols = 'r', 'b'
for im in range(len(mice)):

    cnt = im+1                
#    plt.subplot(r,c,cnt)
        
    CA = CAav_test_inh_alig_sameMaxN_allMice[im], CAav_test_exc_alig_sameMaxN_allMice[im] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
    #plotCA_nN(CA, labs, cols)
    time_aligned = time_aligned_allMice[im]
    ev = (np.argwhere(time_aligned==0)-1).flatten()
    pwmt = plotCA_nN(CA, labs, cols, excInd=1, sdDexc=sdCA_allExcSamps_maxN_ave_allMice[im], doP=0, times2an=ev) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)



#%%    only 10 and 20 neurons

#%%        
def plotCA_nN(CA, labs, cols, excInd=-1, sdDexc=None, doP=0, times2an=[], labels=[]): # excInd is the index of exc in array CAav_alig_sameN... default is -1 so assuming we will get sd across days and not have it as an input
       
    nDays = np.shape(CA[0])[0]       
    nNcommon = np.shape(CA[0])[1]
    nTimeBins = np.shape(CA[0])[2] # size CAav_alig_sameN[0]: numDays x max_nNeurs x nAlignedFrs
    if np.shape(times2an)[0]!=0: # what frames to plot is provided
        nTimeBins = np.shape(times2an.flatten())[0]
#        plt.figure(figsize=(2,2))    
#        r=1;c=1;
#    else:
#        plt.figure(figsize=(8,10))    
#        c = 4.
#        r = np.ceil(nTimeBins/c)    

    if doP: # compute p value
        pwmt = np.full((3,nNcommon,nTimeBins), np.nan)  # 3 x max_nNeurs x nAlignedFrs
        
                
    if np.shape(times2an)[0]==0:
        times2an = range(nTimeBins)
    
#    cnt=0

    for f in times2an: # loop over frames        
#        cnt = cnt+1                
        plt.subplot(r,c,cnt) # f+1
        
        for ntyp in range(len(CA)): 
            a = CA[ntyp][:,:,f] # nDays x nNerus
        #    plt.imshow(a); plt.colorbar()     
            avD = np.nanmean(a,axis=0) # average across days
            
            if excInd!=ntyp:
                # compute it by getting sd across days; if sdD is provided use that value instead of taking it across day. we use this for exc, bc there are nExcShfls, so sd of excSufl-averaged traces across days will be lower than sd of inh traces across days bc inh is not averaged... so for exc I use average of sd across samples instead of sd across days
                sdDNow = np.nanstd(a,axis=0) # stdev across days
                # do se
                sdDNow = np.nanstd(a,axis=0) /np.sqrt(nDays)
                sdD = sdDNow
            else:
                sdD = sdDexc[:,f]
                # do se
                sdD = sdDexc[:,f] /np.sqrt(nDays)
                
            # plot errorbar            
            x = np.array([0,1])+ntyp/5. #range(a.shape[1])
            plt.errorbar(x, avD, sdD, label=labs[ntyp], color=cols[ntyp], linestyle='', marker='.')            
            plt.xticks(x, labels)#, rotation='vertical')
            plt.xlim([-.5, 1.5+ntyp/10])
            
            # fill bounds            
#            plt.fill_between(range(a.shape[1]), avD-sdD, avD+sdD, alpha=0.5, edgecolor=cols[ntyp], facecolor=cols[ntyp])
#            plt.plot(range(a.shape[1]), avD, color=cols[ntyp], linewidth=2, label=labs[ntyp])

#        plt.title('%.0f ms' %(np.round(time_aligned[f])))
        plt.title(mice[im])
        if cnt!=1:
            plt.gca().yaxis.set_visible(False)
            plt.gca().spines['left'].set_color('white')
        
        makeNicePlots(plt.gca())  
        
        if cnt==2:        
            plt.xlabel('Number of neurons used in decoder')
        if cnt==1:                    
            plt.ylabel('Class accuracy (%)')        
        if cnt==nTimeBins-1: 
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False) 
        
        # plot sig p vals
        if doP: # compute p value between exc and inh
            a = CA[0][:,:,f] # nDays x nNerus
            b = CA[1][:,:,f]
            p = pwmt_frs(a,b) # 3 x nNeurons
            
            yl = plt.gca().get_ylim()
            pp = np.full(np.shape(p), np.nan)
            pp[p<=.05] = yl[1] - (yl[1]-yl[0])/10
            pp2p = pp[2] # whether to use wilcoxon, mann-whitney, or ttest
            plt.plot(range(a.shape[1]), pp2p, marker='o', markersize=5, color='k') #, linestyle=':')            
            
            pwmt[:,:,cnt-1] = p # 3 x nNeurons x nfrs # 3 is the 3 tests: wilcoxon, mann-whitney, or ttest
    
    plt.subplots_adjust(wspace=.5, hspace=.5)
    
    # Save the figure
    if savefigs:#% Save the figure
            
        if chAl==1:
            dd = 'chAl_CAvsNnum10_20_time-1_' + '_'.join(mice)
        else:
            dd = 'stAl_CAvsNnum10_20_time-1_' + '_'.join(mice)
                
        d = os.path.join(svmdir+dnow0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
  
  
    if doP==0:
        pwmt = []
    else:
        pwmt = np.array(pwmt)
        
    return pwmt


#%%
nn = np.array([10,20])
    
labs = 'inh', 'exc'
cols = 'r', 'b'

plt.figure(figsize=(3,1.7))    
r=1
c=len(mice)

for im in range(len(mice)):

    cnt = im+1                
#    plt.subplot(r,c,cnt)
    
    CA = CAav_test_inh_alig_sameMaxN_allMice[im][:,nn-1,:], CAav_test_exc_alig_sameMaxN_allMice[im][:,nn-1,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
    #plotCA_nN(CA, labs, cols)
    time_aligned = time_aligned_allMice[im]
    ev = (np.argwhere(time_aligned==0)-1).flatten()
    pwmt = plotCA_nN(CA, labs, cols, excInd=1, sdDexc=sdCA_allExcSamps_maxN_ave_allMice[im][nn-1,:], doP=0, times2an=ev, labels=['10','20']) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)
    
    
    
#%% Across day changes (learning related changes)
'''
def plotCA_nN_eachDay(CA, CS, labs, cols, excInd=-1, sdD=None, doP=0): # excInd is the index of exc in array CAav_alig_sameN... default is -1 so assuming we will get sd across days and not have it as an input
       
    nNcommon = np.shape(CA[0])[0]
    nTimeBins = np.shape(CA[0])[1] # size CAav_alig_sameN[0]: numDays x max_nNeurs x nAlignedFrs


    if doP: # compute p value
        pwmt = np.full((3,nNcommon,nTimeBins), np.nan)  # 3 x max_nNeurs x nAlignedFrs
        
    plt.figure(figsize=(2,30))    
    c = 1.
    r = np.ceil(nTimeBins/c)    

    for f in range(nTimeBins): # loop over frames
            
        plt.subplot(r,c,f+1)
        for ntyp in range(len(CA)): 
            avD = CA[ntyp][:,f]
            sdD = CS[ntyp][:,f]
        #    plt.imshow(a); plt.colorbar()                  
            # plot errorbar
#            plt.errorbar(range(a.shape[1]), avD, sdD, label=labs[ntyp], color=cols[ntyp])        
            
            # fill bounds            
            plt.fill_between(range(nNcommon), avD-sdD, avD+sdD, alpha=0.5, edgecolor=cols[ntyp], facecolor=cols[ntyp])
            plt.plot(range(nNcommon), avD, color=cols[ntyp], linewidth=2, label=labs[ntyp])

        plt.title('%.0f ms' %(np.round(time_aligned[f])))
        makeNicePlots(plt.gca())  
        
        if f==0:        
            plt.xlabel('#Neurons used in decoder')
            plt.ylabel('Class accuracy (testing data)')        
        if f==nTimeBins-1: 
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False) 
        
        # plot sig p vals
        """
        if doP: # compute p value between exc and inh
            a = CA[0][:,:,f] # nDays x nNerus
            b = CA[1][:,:,f]
            p = pwmt_frs(a,b) # 3 x nNeurons
            
            yl = plt.gca().get_ylim()
            pp = np.full(np.shape(p), np.nan)
            pp[p<=.05] = yl[1] - (yl[1]-yl[0])/10
            pp2p = pp[2] # whether to use wilcoxon, mann-whitney, or ttest
            plt.plot(range(a.shape[1]), pp2p, marker='o', markersize=5, color='k') #, linestyle=':')            
            
            pwmt[:,:,f] = p # 3 x nNeurons x nfrs # 3 is the 3 tests: wilcoxon, mann-whitney, or ttest
        """
    plt.subplots_adjust(hspace=.5)
    
    # Save the figure
    if savefigs:#% Save the figure
            
        if chAl==1:
            dd = 'chAl_CAvsNnum_' + '_'.join(labs)+'_' + days[0][0:6] + '-to-' + days[-1][0:6]
        else:
            dd = 'stAl_CAvsNnum_' + '_'.join(labs)+'_' + days[0][0:6] + '-to-' + days[-1][0:6]
                
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
  
  
    if doP==0:
        pwmt = []
    else:
        pwmt = np.array(pwmt)
        
    return pwmt
                
'''                
#################### #################### #################### #################### 
#%%
#################### same as above but using min nNeuron across days as the max value of x axis ####################
'''
#%% use the same number of neurons for all days, so you can make an average across days

def sameNs(CAav_alig, mnNumNeur):
    CAav_alig_sameN = np.array([CAav_alig[iday][0:mnNumNeur,:] for iday in range(len(days))]) # numDays x numNeurs_used_for_training x nFrs

    return CAav_alig_sameN 
    

### same as above but for exc (bc it has nExcShfls)    
def sameNs_exc(CAav_alig, mnNumNeur):
    CAav_alig_sameN = np.array([CAav_alig[iday][0:mnNumNeur,:,:] for iday in range(len(days))]) # numDays x numNeurs_used_for_training x nFrs

    return CAav_alig_sameN 


#%% use the same number of neurons for all days, so you can make an average
# output array size: numDays x min_nNeurs_used_for_training x nAlignedFrs
# input array size: numDays; each day: nNeurons x nAlignedFrs
# for exc:
# output array size: numDays x min_nNeurs_used_for_training x nAlignedFrs x nExcShfl
# input array size: numDays; each day: nNeurons x nAlignedFrs x nExcShfl

mnNumNeur = min(numInh[mn_corr >= thTrained]).astype(int)
CAav_test_inh_alig_sameN = sameNs(CAav_test_inh_alig, mnNumNeur)
#CAav_test_shfl_inh_alig_sameN = sameNs(CAav_test_shfl_inh_alig)
#CAav_test_chance_inh_alig_sameN = sameNs(CAav_test_chance_inh_alig)
#CAav_train_inh_alig_sameN = sameNs(CAav_train_inh_alig) # numDays x numNeurs_used_for_training x nFrs

# exc output array size: numDays x min_nNeurs_used_for_training x nAlignedFrs x nExcShfl
mnNumNeur = min(numInh[mn_corr >= thTrained]).astype(int)
CAav_test_exc_alig_sameN = sameNs_exc(CAav_test_exc_alig, mnNumNeur)
#CAav_test_shfl_exc_alig_sameN = sameNs(CAav_test_shfl_exc_alig)
#CAav_test_chance_exc_alig_sameN = sameNs(CAav_test_chance_exc_alig)
#CAav_train_exc_alig_sameN = sameNs(CAav_train_exc_alig) # numDays x numNeurs_used_for_training x nFrs


mnNumNeur = min(numAlln[mn_corr >= thTrained]).astype(int)

CAav_test_allN_alig_sameN = sameNs(CAav_test_allN_alig, mnNumNeur)
#CAav_test_shfl_allN_alig_sameN = sameNs(CAav_test_shfl_allN_alig)
#CAav_test_chance_allN_alig_sameN = sameNs(CAav_test_chance_allN_alig)
#CAav_train_allN_alig_sameN = sameNs(CAav_train_allN_alig) # numDays x numNeurs_used_for_training x nFrs


#%% Keep original vals of exc before averaging across nExcShfl

CAav_test_exc_alig_sameN0 = CAav_test_exc_alig_sameN+0


#%% for each excSamp compute std across days

sdCA_allExcSamps = sdEachExcSamp(CAav_test_exc_alig_sameN0) # min_nNeurs_used_for_training x nAlignedFrs x nExcShfl
# now take ave of sd across excSamps
sdCA_allExcSamps_ave = np.mean(sdCA_allExcSamps,axis=2)


#%% Now for exc take average across nExcShfls, so it will have same size as inh and allN
# output array size: numDays x min_nNeurs_used_for_training x nAlignedFrs
# input array size for exc: numDays x min_nNeurs_used_for_training x nAlignedFrs x nExcShfl

CAav_test_exc_alig_sameN = np.mean(CAav_test_exc_alig_sameN0, axis=3)


#%% Set color order
"""
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.style.use('default')


#% Change color order to jet 
colorOrder(nlines=len(days))
"""

    
#%% plot ave and sd (across days) of CA vs. neurNum for each time bin

# set what days to use
days2use = mn_corr>=thTrained

labs = 'inh', 'exc'
cols = 'r', 'b' # inh, exc
CA = CAav_test_inh_alig_sameN[days2use,:,:], CAav_test_exc_alig_sameN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
pwmt = plotCA_nN(CA, labs, cols, excInd=1, sdD=sdCA_allExcSamps_ave, doP=1) # for exc use sdCA_allExcSamps_ave as sd instead of taking sd across days (this is more comparable to sd of inh, bc sdCA_allExcSamps_ave is sd across days for each nExcShfl, then averaged across nExcShfl, if we get sd acorr excShfl-averaged traces it will be much lower than inh sd)


labs = 'allN',
cols = 'k',
CA = CAav_test_allN_alig_sameN[days2use,:,:],
plotCA_nN(CA, labs, cols)


labs = 'allN', 'inh', 'exc'
cols = 'k', 'r', 'b'
CA = CAav_test_allN_alig_sameN[days2use,:,:], CAav_test_inh_alig_sameN[days2use,:,:], CAav_test_exc_alig_sameN[days2use,:,:] # each ones len:  numDays x min_nNeurs_used_for_training x nAlignedFrs
plotCA_nN(CA, labs, cols, excInd=2, sdD=sdCA_allExcSamps_ave)

'''


'''
#plotCA_nN(CAav_test_shfl_inh_alig_sameN)
#plotCA_nN(CAav_test_chance_inh_alig_sameN)
#plotCA_nN(CAav_train_inh_alig_sameN)

plotCA_nN(CAav_test_allN_alig_sameN)
#plotCA_nN(CAav_test_shfl_allN_alig_sameN)
#plotCA_nN(CAav_test_chance_allN_alig_sameN)
#plotCA_nN(CAav_train_allN_alig_sameN)

plotCA_nN(CAav_test_exc_alig_sameN)
#plotCA_nN(CAav_test_shfl_exc_alig_sameN)
#plotCA_nN(CAav_test_chance_exc_alig_sameN)
#plotCA_nN(CAav_train_exc_alig_sameN)
'''
