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

mousename = 'fni17' #'fni17'
if mousename == 'fni18': #set one of the following to 1:
    allDays = 0# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 1 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!

loadInhAllexcEqexc = 0 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       


trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("svm_plots_setVars.py")  

chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
savefigs = 0
superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure

#doPlots = 0 #1 # plot c path of each day 
#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.


#%% 
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

if loadInhAllexcEqexc==1:
    dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/'
else:
    dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/inhRois/'


smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc=[], regressBins=3, useEqualTrNums=1):
    import glob

    if chAl==1:
        al = 'chAl'
    else:
        al = 'stAl'
        
    if len(doInhAllexcEqexc)==0: # 1st run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc at the same time, also for all days (except a few days of fni18), inhRois was used (not the new inhRois_pix)       
        if trialHistAnalysis:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_prevChoice_%s_ds%d_eqTrs_*' %(al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_prevChoice_%s_ds%d_*' %(al,regressBins)
        else:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_currChoice_%s_ds%d_eqTrs_*' %(al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_currChoice_%s_ds%d_*' %(al,regressBins)
        
    else: # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
        if doInhAllexcEqexc[0] == 1:
            ntype = 'inh'
        elif doInhAllexcEqexc[1] == 1:
            ntype = 'allExc'
        elif doInhAllexcEqexc[2] == 1:
            ntype = 'eqExc'           
            
        if trialHistAnalysis:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_%s_prevChoice_%s_ds%d_eqTrs_*' %(ntype, al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_%s_prevChoice_%s_ds%d_*' %(ntype, al,regressBins)
        else:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_%s_currChoice_%s_ds%d_eqTrs_*' %(ntype, al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_%s_currChoice_%s_ds%d_*' %(ntype, al,regressBins)
        
        
        
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.

    return svmName
    


#%%   
def setTo50classErr(classError, w, thNon0Ws = .05, thSamps = 10, eps = 1e-10):
#            classError: perClassErrorTest_data_inh
#            wname = 'w_data_inh'
#            thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#            thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.

#    d = scio.loadmat(svmName, variable_names=[wname])
#    w = d.pop(wname)            
    a = abs(w) > eps
    # average abs(w) across neurons:
    if w.ndim==4: #exc : average across excShuffles
        a = np.mean(a, axis=0)
    aa = np.mean(a, axis=1) # samples x frames; shows average number of neurons with non0 weights for each sample and frame 
#        plt.imshow(aa), plt.colorbar()
    goodSamps = aa > thNon0Ws # samples with >.05 of neurons having non-0 weight 
#        sum(goodSamps) # for each frame, it shows number of samples with >.05 of neurons having non-0 weight        
    
#    print ~goodSamps
    if sum(sum(goodSamps)<thSamps)>0:
        print sum(goodSamps)<thSamps
    
    if w.ndim==3: 
        classError[~goodSamps] = 50 # set to 50 class error of samples which have <=.05 of non-0-weight neurons
        classError[:,sum(goodSamps)<thSamps] = 50 # if fewer than 10 samples contributed to a frame, set the perClassError of all samples for that frame to 50...       
    elif w.ndim==4: #exc : average across excShuffles
        classError[:,~goodSamps] = 50 # set to 50 class error of samples which have <=.05 of non-0-weight neurons
        classError[:,:,sum(goodSamps)<thSamps] = 50 # if fewer than 10 samples contributed to a frame, set the perClassError of all samples for that frame to 50...       
    
    modClassError = classError+0
    
    return modClassError
    
           
#%% 
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
#####################################################################################################################################################
'''
            
#%% Loop over days    

eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
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
    #    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
        time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
        time_trace = time_aligned_1stSide
        
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
      
    
    ##%% Downsample traces: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)    
    if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
        print 'Downsampling traces ....'    
            
        T1 = time_trace.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X            

        time_trace = time_trace[0:regressBins*tt]
        time_trace = np.round(np.mean(np.reshape(time_trace, (regressBins, tt), order = 'F'), axis=0), 2)
        print time_trace.shape
    
        eventI_ds = np.argwhere(np.sign(time_trace)>0)[0] # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
    
    else:
        print 'Not downsampling traces ....'        
#        eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
#        eventI_ds = eventI_ch
    
    eventI_allDays[iday] = eventI_ds
    
    
    #%% Load SVM vars
    
    if loadInhAllexcEqexc==0: # 1st run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc at the same time, also for all days (except a few days of fni18), inhRois was used (not the new inhRois_pix)       
        svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, [], regressBins)
        svmName = svmName[0]
        print os.path.basename(svmName)    

        Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh',    
                                                 'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc',    
                                                 'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc',
                                                 'w_data_inh', 'w_data_allExc', 'w_data_exc'])        
        Datai = Dataae = Datae = Data                                                 

    else:  # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
        
        for idi in range(3):
            doInhAllexcEqexc = np.full((3), False)
            doInhAllexcEqexc[idi] = True 
            svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins)        
            svmName = svmName[0]
            print os.path.basename(svmName)    

            if doInhAllexcEqexc[0] == 1:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'w_data_inh'])
            elif doInhAllexcEqexc[1] == 1:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc'])
            elif doInhAllexcEqexc[2] == 1:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'w_data_exc'])                                                 
        
        
    ###%%             
    perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh')
    perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh')
    perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh') 
    w_data_inh = Datai.pop('w_data_inh') 
    
    perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc')
    perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc')
    perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc')   
    w_data_allExc = Dataae.pop('w_data_allExc') 
    
    perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_data_exc')    
    perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_exc')
    perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_exc')
    w_data_exc = Datae.pop('w_data_exc') 
    
   

    #%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.
 
    perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
    perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
    perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
    
    perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
    perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
    perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
    
    perClassErrorTest_data_exc = setTo50classErr(perClassErrorTest_data_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
    perClassErrorTest_shfl_exc = setTo50classErr(perClassErrorTest_shfl_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
    perClassErrorTest_chance_exc = setTo50classErr(perClassErrorTest_chance_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
    

    #%% Get number of inh and exc

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


eventI_allDays = eventI_allDays.astype('int')

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

#%% Average and std of class accuracies across CV samples ... for each day

numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]

#### inh
av_test_data_inh = np.array([100-np.nanmean(perClassErrorTest_data_inh_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_test_data_inh = np.array([np.nanstd(perClassErrorTest_data_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_test_shfl_inh = np.array([100-np.nanmean(perClassErrorTest_shfl_inh_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_test_shfl_inh = np.array([np.nanstd(perClassErrorTest_shfl_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_test_chance_inh = np.array([100-np.nanmean(perClassErrorTest_chance_inh_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_test_chance_inh = np.array([np.nanstd(perClassErrorTest_chance_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

#### exc (average across cv samples and exc shuffles)
av_test_data_exc = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday], axis=(0,1)) for iday in range(len(days))]) # numDays
sd_test_data_exc = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(len(days))])  

av_test_shfl_exc = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) for iday in range(len(days))]) # numDays
sd_test_shfl_exc = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(len(days))])  

av_test_chance_exc = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) for iday in range(len(days))]) # numDays
sd_test_chance_exc = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(len(days))])  

#### allExc
av_test_data_allExc = np.array([100-np.nanmean(perClassErrorTest_data_allExc_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_test_data_allExc = np.array([np.nanstd(perClassErrorTest_data_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_test_shfl_allExc = np.array([100-np.nanmean(perClassErrorTest_shfl_allExc_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_test_shfl_allExc = np.array([np.nanstd(perClassErrorTest_shfl_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_test_chance_allExc = np.array([100-np.nanmean(perClassErrorTest_chance_allExc_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_test_chance_allExc = np.array([np.nanstd(perClassErrorTest_chance_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

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
    


#%%
##################################################################################################
############## Align class accur traces of all days to make a final average trace ##############
################################################################################################## 
  
##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (len(av_test_data_inh_ch[iday]) - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


###%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

def alTrace(trace, eventI_allDays, nPreMin, nPostMin):    

    trace_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
    for iday in range(trace.shape[0]):
        trace_aligned[:, iday] = trace[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]    
    return trace_aligned
    

av_test_data_inh_aligned = alTrace(av_test_data_inh, eventI_allDays, nPreMin, nPostMin)
av_test_shfl_inh_aligned = alTrace(av_test_shfl_inh, eventI_allDays, nPreMin, nPostMin)
av_test_chance_inh_aligned = alTrace(av_test_chance_inh, eventI_allDays, nPreMin, nPostMin)

av_test_data_exc_aligned = alTrace(av_test_data_exc, eventI_allDays, nPreMin, nPostMin)
av_test_shfl_exc_aligned = alTrace(av_test_shfl_exc, eventI_allDays, nPreMin, nPostMin)
av_test_chance_exc_aligned = alTrace(av_test_chance_exc, eventI_allDays, nPreMin, nPostMin)

av_test_data_allExc_aligned = alTrace(av_test_data_allExc, eventI_allDays, nPreMin, nPostMin)
av_test_shfl_allExc_aligned = alTrace(av_test_shfl_allExc, eventI_allDays, nPreMin, nPostMin)
av_test_chance_allExc_aligned = alTrace(av_test_chance_allExc, eventI_allDays, nPreMin, nPostMin)


##%% Average and STD across days (each day includes the average class accuracy across samples.)

av_av_test_data_inh_aligned = np.mean(av_test_data_inh_aligned, axis=1)
sd_av_test_data_inh_aligned = np.std(av_test_data_inh_aligned, axis=1)

av_av_test_shfl_inh_aligned = np.mean(av_test_shfl_inh_aligned, axis=1)
sd_av_test_shfl_inh_aligned = np.std(av_test_shfl_inh_aligned, axis=1)

av_av_test_chance_inh_aligned = np.mean(av_test_chance_inh_aligned, axis=1)
sd_av_test_chance_inh_aligned = np.std(av_test_chance_inh_aligned, axis=1)


av_av_test_data_exc_aligned = np.mean(av_test_data_exc_aligned, axis=1)
sd_av_test_data_exc_aligned = np.std(av_test_data_exc_aligned, axis=1)

av_av_test_shfl_exc_aligned = np.mean(av_test_shfl_exc_aligned, axis=1)
sd_av_test_shfl_exc_aligned = np.std(av_test_shfl_exc_aligned, axis=1)

av_av_test_chance_exc_aligned = np.mean(av_test_chance_exc_aligned, axis=1)
sd_av_test_chance_exc_aligned = np.std(av_test_chance_exc_aligned, axis=1)


av_av_test_data_allExc_aligned = np.mean(av_test_data_allExc_aligned, axis=1)
sd_av_test_data_allExc_aligned = np.std(av_test_data_allExc_aligned, axis=1)

av_av_test_shfl_allExc_aligned = np.mean(av_test_shfl_allExc_aligned, axis=1)
sd_av_test_shfl_allExc_aligned = np.std(av_test_shfl_allExc_aligned, axis=1)

av_av_test_chance_allExc_aligned = np.mean(av_test_chance_allExc_aligned, axis=1)
sd_av_test_chance_allExc_aligned = np.std(av_test_chance_allExc_aligned, axis=1)


#_,pcorrtrace0 = stats.ttest_1samp(av_l2_test_d_aligned.transpose(), 50) # p value of class accuracy being different from 50

_,pcorrtrace = stats.ttest_ind(av_test_data_exc_aligned.transpose(), av_test_data_inh_aligned.transpose()) # p value of class accuracy being different from 50
        
        

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
######################## PLOTS ########################


#%% Compare number of inh vs exc for each day

plt.figure(figsize=(4.5,3)) #plt.figure()
plt.plot(numInh, label='inh')
plt.plot(numAllexc, label='exc')
plt.title('average: inh=%d exc=%d inh/exc=%.2f' %(np.round(numInh.mean()), np.round(numAllexc.mean()), np.mean(numInh) / np.mean(numAllexc)))
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
plt.xlabel('days')
plt.ylabel('# neurons')

if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_numNeurons'
    else:
        dd = 'stAl_numNeurons'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
#%% Plot class accur trace for each day

plt.figure()
for iday in range(len(days)):    
#    plt.figure()
    nPre = eventI_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
    nPost = (len(av_test_data_inh_ch[iday]) - eventI_allDays[iday] - 1)
    
    a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
    b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
    time_al = np.concatenate((a,b))
    
    
    plt.subplot(221)
    plt.errorbar(time_al, av_test_data_inh_ch[iday], yerr = sd_test_data_inh_ch[iday], label='inh', color='r')    
#    plt.title(days[iday])
    plt.errorbar(time_al, av_test_data_exc_ch[iday], yerr = sd_test_data_exc_ch[iday], label='exc', color='b')
    plt.errorbar(time_al, av_test_data_allExc_ch[iday], yerr = sd_test_data_allExc_ch[iday], label='allExc', color='k')


    plt.subplot(222)
    plt.plot(time_al, av_test_shfl_inh_ch[iday], label=' ', color='r')    
    plt.plot(time_al, av_test_shfl_exc_ch[iday], label=' ', color='b')    
    plt.plot(time_al, av_test_shfl_allExc_ch[iday], label=' ', color='k')        
    
    
    plt.subplot(223)
    h0,=plt.plot(time_al, av_test_chance_inh_ch[iday], label='inh', color='r')    
    h1,=plt.plot(time_al, av_test_chance_exc_ch[iday], label='exc', color='b')    
    h2,=plt.plot(time_al, av_test_chance_allExc_ch[iday], label='allExc', color='k')        
    
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
plt.legend([h0,h1,h2],['inh','exc','allExc'],loc='center left', bbox_to_anchor=(1, .7)) 

#plt.show()
plt.subplots_adjust(hspace=0.65)


##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_allDays'
    else:
        dd = 'stAl_allDays'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
    

       
#%% Plot the average of aligned traces across all days

plt.figure() #(figsize=(4.5,3))

#### testing data
plt.subplot(221)
# exc
plt.fill_between(time_aligned, av_av_test_data_exc_aligned - sd_av_test_data_exc_aligned, av_av_test_data_exc_aligned + sd_av_test_data_exc_aligned, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, av_av_test_data_exc_aligned, 'b')
# inh
plt.fill_between(time_aligned, av_av_test_data_inh_aligned - sd_av_test_data_inh_aligned, av_av_test_data_inh_aligned + sd_av_test_data_inh_aligned, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, av_av_test_data_inh_aligned, 'r')
# allExc
plt.fill_between(time_aligned, av_av_test_data_allExc_aligned - sd_av_test_data_allExc_aligned, av_av_test_data_allExc_aligned + sd_av_test_data_allExc_aligned, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, av_av_test_data_allExc_aligned, 'k')

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
    
    
plt.legend(['inh','exc','allExc'],loc='center left', bbox_to_anchor=(1, .7)) 


#xmin, xmax = ax.get_xlim()
plt.xlim([-1400,500])

##%% Save the figure    
if savefigs:
    if chAl==1:
        dd = 'chAl_aveDays'
    else:
        dd = 'stAl_aveDays'
        
    if superimpose==1:        
        dd = dd+'_sup'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
#            
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