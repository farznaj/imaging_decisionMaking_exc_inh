# -*- coding: utf-8 -*-
"""
compute angles between weights found in script excInh_SVMtrained_eachFrame, ie
weights computed by training the decoder using only only inh or only exc neurons.


Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     

#%% Change the following vars:

mousename = 'fni18' #'fni17'

if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
if mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   

loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")
execfile("svm_plots_setVars_n.py")  

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

dnow = '/stability/'+mousename+'/'

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
#            classError = perClassErrorTest_data_inh
#            w = w_data_inh
#            thNon0Ws = .05; thSamps = 10; eps = 1e-10
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
    goodSamps = aa > thNon0Ws # samples x frames; # samples with >.05 of neurons having non-0 weight 
#        sum(goodSamps) # for each frame, it shows number of samples with >.05 of neurons having non-0 weight        
    
#    print ~goodSamps
    if (sum(~goodSamps)>0).any():
        print 'For each frame showing samples with fraction of non-0-weights <=.05 : ', sum(~goodSamps)
    if sum(sum(goodSamps)<thSamps)>0:
        print sum(sum(goodSamps)<thSamps), ' frames have fewer than 10 good samples!' 
    
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
angleInh_all = []
angleExc_all = []
angleAllExc_all = []
angleInhS_all = []
angleExcS_all = []
angleAllExcS_all = []
time_trace_all = []
hrnAll = []
lrnAll = []

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

   
    #%% Set number of hr and lr trials for each session

    svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins)        
    svmName = svmName[0]
    print os.path.basename(svmName)   

    Datai = scio.loadmat(svmName, variable_names=['trsExcluded'])
    trsExcluded = Datai.pop('trsExcluded').flatten().astype('bool')
   
    Data = scio.loadmat(postName, variable_names=['allResp_HR_LR'])
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    Y = allResp_HR_LR[~trsExcluded]
    hrn = (Y==1).sum()
    lrn = (Y==0).sum() 
    
    hrnAll.append(hrn)
    lrnAll.append(lrn)


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
    time_trace_all.append(time_trace)
       
    
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
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'w_data_inh'])#, 'trsExcluded'])
            elif doInhAllexcEqexc[1] == 1:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc'])#, 'trsExcluded'])
            elif doInhAllexcEqexc[2] == 1:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'w_data_exc'])#, 'trsExcluded'])                                                 
                
        
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
    w_data_exc = Datae.pop('w_data_exc')  # numShufflesExc x numSamples x numNeurons x numFrames
    
    
    
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


    #%%
    #%% Normalize weights (ie the w vector at each frame must have length 1)
    
    # inh
    w = w_data_inh + 0 # numSamples x nNeurons x nFrames
    normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
    wInh_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
    if sum(normw<=eps).sum()!=0:
        print 'take care of this; you need to reshape w_normed first'
    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
    
    # exc
    w = w_data_exc + 0 # numShufflesExc x numSamples x numNeurons x numFrames
    normw = np.linalg.norm(w, axis=2) # numShufflesExc x numSamples x frames ; 2-norm of weights 
    wExc_normed = np.transpose(np.transpose(w,(2,0,1,3))/normw, (1,2,0,3)) # numShufflesExc x numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
    if sum(normw<=eps).sum()!=0:
        print 'take care of this; you need to reshape w_normed first'
    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
     
    # allExc            
    w = w_data_allExc + 0 # numSamples x nNeurons x nFrames
    normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
    wAllExc_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
    if sum(normw<=eps).sum()!=0:
        print 'take care of this; you need to reshape w_normed first'
    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

    
    #%% Set the final decoders by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
    
    # inh                    
    w_nb = np.mean(wInh_normed, axis=(0)) # neurons x frames 
    nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
    wInh_n2b = w_nb/nw # neurons x frames

    # exc (average across trial subselects as well as exc neurons subselect)
    w_nb = np.mean(wExc_normed, axis=(0,1)) # neurons x frames 
    nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
    wExc_n2b = w_nb/nw # neurons x frames

    # allExc
    w_nb = np.mean(wAllExc_normed, axis=(0)) # neurons x frames 
    nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
    wAllExc_n2b = w_nb/nw # neurons x frames

    
    #%% Compute angle between decoders (weights) at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)
    
    # some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!    
    # we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
    
    angleInh = np.arccos(abs(np.dot(wInh_n2b.transpose(), wInh_n2b)))*180/np.pi # frames x frames; angle between ws at different times        
    angleExc = np.arccos(abs(np.dot(wExc_n2b.transpose(), wExc_n2b)))*180/np.pi # frames x frames; angle between ws at different times
    angleAllExc = np.arccos(abs(np.dot(wAllExc_n2b.transpose(), wAllExc_n2b)))*180/np.pi # frames x frames; angle between ws at different times
 
    # shuffles: angle between the real decoders and the shuffled decoders (same as real except the indexing of weights is shuffled)     
    nsh = 1000 # number of times to shuffle the decoders to get null distribution of angles
    frs = wInh_n2b.shape[1] # number of frames
    angleInhS = np.full((frs, frs, nsh), np.nan)
    angleExcS = np.full((frs, frs, nsh), np.nan)
    angleAllExcS = np.full((frs, frs, nsh), np.nan)
    for ish in range(nsh):
        angleInhS[:,:,ish] = np.arccos(abs(np.dot(wInh_n2b[rng.permutation(wInh_n2b.shape[0]),:].transpose(), wInh_n2b[rng.permutation(wInh_n2b.shape[0]),:])))*180/np.pi 
        angleExcS[:,:,ish] = np.arccos(abs(np.dot(wExc_n2b[rng.permutation(wExc_n2b.shape[0]),:].transpose(), wExc_n2b[rng.permutation(wExc_n2b.shape[0]),:])))*180/np.pi 
        angleAllExcS[:,:,ish] = np.arccos(abs(np.dot(wAllExc_n2b[rng.permutation(wAllExc_n2b.shape[0]),:].transpose(), wAllExc_n2b[rng.permutation(wAllExc_n2b.shape[0]),:])))*180/np.pi
    
    
    #%% Keep vars from all days
     
    angleInh_all.append(angleInh)
    angleExc_all.append(angleExc)
    angleAllExc_all.append(angleAllExc)    
     
    angleInhS_all.append(angleInhS)
    angleExcS_all.append(angleExcS)
    angleAllExcS_all.append(angleAllExcS)    
    
     
    #%% Delete vars before starting the next day    
    del w_data_inh, w_data_allExc, w_data_exc, perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc


#%% Done with all days, keep values of all mice

eventI_allDays = eventI_allDays.astype('int')
angleInh_all = np.array(angleInh_all)
angleExc_all = np.array(angleExc_all)
angleAllExc_all = np.array(angleAllExc_all)
angleInhS_all = np.array(angleInhS_all)
angleExcS_all = np.array(angleExcS_all)
angleAllExcS_all = np.array(angleAllExcS_all)

#perClassErrorTest_data_inh_all = np.array(perClassErrorTest_data_inh_all)
#perClassErrorTest_shfl_inh_all = np.array(perClassErrorTest_shfl_inh_all)
#perClassErrorTest_chance_inh_all = np.array(perClassErrorTest_chance_inh_all)
#perClassErrorTest_data_allExc_all = np.array(perClassErrorTest_data_allExc_all)
#perClassErrorTest_shfl_allExc_all = np.array(perClassErrorTest_shfl_allExc_all)
#perClassErrorTest_chance_allExc_all = np.array(perClassErrorTest_chance_allExc_all)
#perClassErrorTest_data_exc_all = np.array(perClassErrorTest_data_exc_all) # numShufflesExc x numSamples x numFrames
#perClassErrorTest_shfl_exc_all = np.array(perClassErrorTest_shfl_exc_all)
#perClassErrorTest_chance_exc_all = np.array(perClassErrorTest_chance_exc_all)



#%% Identify days with too few trials (to exclude them!)
# The total trial numbers are twice ntrs

hrnAll = np.array(hrnAll)
lrnAll = np.array(lrnAll)

print 'Fraction of days with more LR trials = ', (lrnAll > hrnAll).mean()
print 'LR - HR trials (average across days) = ',(lrnAll - hrnAll).mean()

thnt = 10 # if either hr or lr trials were fewer than 10, dont analyze that day!
ntrs = np.min([hrnAll,lrnAll], axis=0)
print (ntrs < thnt).sum(), ' days will be excluded bc either HR or LR is < ', thnt # days to be excluded

plt.figure
plt.plot(ntrs)
plt.xlabel('Days')
plt.ylabel('Trial number (min HR,LR)')
makeNicePlots(plt.gca())

##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    daysnew = (np.array(days))[dayinds]
    if chAl==1:
        dd = 'chAl_ntrs_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]
    else:
        dd = 'stAl_ntrs_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



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

# this is wrong. ... look at evernote for an explanation of it!
'''
a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))
'''

##%% Align traces of all days on the common eventI
time_trace_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    time_trace_aligned[:, iday] = time_trace_all[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    
time_aligned = np.mean(time_trace_aligned,1)


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
        
        




#%% Align angles of all days on the common eventI

r = nPreMin + nPostMin + 1

angleInh_aligned = np.ones((r,r, numDays)) + np.nan
angleExc_aligned = np.ones((r,r, numDays)) + np.nan
angleAllExc_aligned = np.ones((r,r, numDays)) + np.nan  # frames x frames x days, aligned on common eventI (equals nPreMin)
angleInhS_aligned = np.ones((r,r,nsh, numDays)) + np.nan
angleExcS_aligned = np.ones((r,r,nsh, numDays)) + np.nan
angleAllExcS_aligned = np.ones((r,r,nsh, numDays)) + np.nan  # frames x frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
#    angleInh_aligned[:, iday] = angleInh_all[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1  ,  eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    inds = np.arange(eventI_allDays[iday] - nPreMin,  eventI_allDays[iday] + nPostMin + 1)
    
    angleInh_aligned[:,:,iday] = angleInh_all[iday][inds][:,inds]
    angleExc_aligned[:,:,iday] = angleExc_all[iday][inds][:,inds]
    angleAllExc_aligned[:,:,iday] = angleAllExc_all[iday][inds][:,inds]
    # shuffles
    angleInhS_aligned[:,:,:, iday] = angleInhS_all[iday][inds][:,inds] # frames x frames x nShfls
    angleExcS_aligned[:,:,:, iday] = angleExcS_all[iday][inds][:,inds] # frames x frames x nShfls
    angleAllExcS_aligned[:,:,:, iday] = angleAllExcS_all[iday][inds][:,inds] # frames x frames x nShfls
        


#%% Find time points whose decoders angle is significantly different from the null distribution

sigAngInh = np.full((np.shape(angleInh_aligned)), False, dtype=bool) # frames x frames x days # matrix of 0s and 1s; 0 mean the decoders of those time points are mis-aligned (ie their angle is similar to null dist); 1 means the 2 decoders are aligned.
sigAngExc = np.full((np.shape(angleExc_aligned)), False, dtype=bool)
sigAngAllExc = np.full((np.shape(angleAllExc_aligned)), False, dtype=bool)

pvals = np.arange(0,99,1) # percentile values # I will use 90 degrees as the 100th percentile, that's why I dont do np.arange(0,100,1)
percAngInh = np.full((np.shape(angleExc_aligned)), np.nan)
percAngExc = np.full((np.shape(angleExc_aligned)), np.nan)
percAngAllExc = np.full((np.shape(angleExc_aligned)), np.nan)

for iday in range(len(days)):
    for f1 in range(r):
        for f2 in np.delete(range(r),f1): #range(r):
            # whether real angle is smaller than 5th percentile of shuffle angles.
            h = np.percentile(angleInhS_aligned[f1,f2,:, iday], 5)
            sigAngInh[f1,f2,iday] = angleInh_aligned[f1,f2, iday] < h # if the real angle is < 5th percentile of the shuffled angles, then it is significantly differnet from null dist.  
            h = np.percentile(angleExcS_aligned[f1,f2,:, iday], 5)
            sigAngExc[f1,f2,iday] = angleExc_aligned[f1,f2, iday] < h            
            h = np.percentile(angleAllExcS_aligned[f1,f2,:, iday], 5)
            sigAngAllExc[f1,f2,iday] = angleAllExc_aligned[f1,f2, iday] < h

            # compute percentile of significancy
            # percentile of shuffled angles that are bigger than the real angle (ie more mis-aligned)
            # larger percAngInh means more alignment relative to shuffled dist. (big percentage of shuffled dist has larger angles than the real angle)
            percAngInh[f1,f2,iday] = 100 - np.argwhere(angleInh_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleInhS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
            percAngExc[f1,f2,iday] = 100 - np.argwhere(angleExc_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleExcS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
            percAngAllExc[f1,f2,iday] = 100 - np.argwhere(angleAllExc_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleAllExcS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]

#np.mean(sigAngInh, axis=-1)) # each element shows fraction of days in which angle between decoders was significant.
# eg I will call an angle siginificant if in >=.5 of days it was significant.
#plt.imshow(np.mean(sigAngInh, axis=-1)); plt.colorbar() # fraction of days in which angle between decoders was significant.
#plt.imshow(np.mean(angleInh_aligned, axis=-1)); plt.colorbar()        




#%% Average across days
########################### NOTE: use daysinds to set averages so all plots below have used the same days
##################################################################################################################################

angInh_av = np.nanmean(angleInh_aligned, axis=-1) # frames x frames
angExc_av = np.nanmean(angleExc_aligned, axis=-1)
angAllExc_av = np.nanmean(angleAllExc_aligned, axis=-1)

### set diagonal elements (whose angles are 0) to nan, so heatmaps span a smaller range and can be better seen.
np.fill_diagonal(angInh_av, np.nan)
np.fill_diagonal(angExc_av, np.nan)
np.fill_diagonal(angAllExc_av, np.nan)
#np.diagonal(ang_choice_av)


######## shuffles 
# average across days and shuffles # frames x frames
angInhS_av = np.nanmean(angleInhS_aligned, axis=(2,-1))
angExcS_av = np.nanmean(angleExcS_aligned, axis=(2,-1))
angAllExcS_av = np.nanmean(angleAllExcS_aligned, axis=(2,-1))


# average of significance values
#np.mean(sigAngInh, axis=-1)) # each element shows fraction of days in which angle between decoders was significant.
# eg I will call an angle siginificant if in more than .5 of days it was significant.
angInhSig_av = np.nanmean(sigAngInh, axis=-1)
angExcSig_av = np.nanmean(sigAngExc, axis=-1)
angAllExcSig_av = np.nanmean(sigAngAllExc, axis=-1)

### set diagonal elements (whose angles are 0) to nan, so heatmaps span a smaller range and can be better seen.
np.fill_diagonal(angInhSig_av, np.nan)
np.fill_diagonal(angExcSig_av, np.nan)
np.fill_diagonal(angAllExcSig_av, np.nan)


# average of significancy percentiles (I believe diag elements are already nan, so we dont need to do the step of fill_diag)
percAngInh_av = np.nanmean(percAngInh, axis=-1)
percAngExc_av = np.nanmean(percAngExc, axis=-1)
percAngAllExc_av = np.nanmean(percAngAllExc, axis=-1)




##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
######################## PLOTS ########################


#%% Plot Angles (averaged across days (plot angle between decoders at different time points))

step = 4
x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
           np.arange(np.argwhere(time_aligned>=0)[0], r, step))))).astype(int)
#x = np.arange(0,r,step)
if chAl==1:
    xl = 'Time since choice onset (ms)'
else:
    xl = 'Time since stimulus onset (ms)'

#cmap='jet_r'
#cmap='hot'
def plotAng(top, lab, cmin, cmax, cmap='jet', cblab=''):
    plt.imshow(top, cmap) #, interpolation='nearest') #, extent=time_aligned)
    plt.colorbar(label=cblab)
    plt.xticks(x, np.round(time_aligned[x]).astype(int))
    plt.yticks(x, np.round(time_aligned[x]).astype(int))
    makeNicePlots(plt.gca())
    #ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
    plt.title(lab)
    plt.xlabel(xl)
    plt.clim(cmin, cmax)
    plt.plot([nPreMin, nPreMin], [0, len(time_aligned)], color='r')
    plt.plot([0, len(time_aligned)], [nPreMin, nPreMin], color='r')
    plt.xlim([0, len(time_aligned)])
#    plt.ylim([0, len(time_aligned)])
    plt.ylim([0, len(time_aligned)][::-1])

    



######## Plot angles, averaged across days
# real     
plt.figure(figsize=(8,8))
cmin = np.floor(np.min([np.nanmin(angInh_av), np.nanmin(angExc_av), np.nanmin(angAllExc_av)]))
cmax = 90
cmap = 'jet_r' # lower angles: more aligned: red
cblab = 'Angle between decoders'
lab = 'inh (data)'; top = angInh_av; plt.subplot(221); plotAng(top, lab, cmin, cmax, cmap, cblab)
lab = 'exc (data)'; top = angExc_av; plt.subplot(223); plotAng(top, lab, cmin, cmax, cmap, cblab)
lab = 'allExc (data)'; top = angAllExc_av; plt.subplot(222); plotAng(top, lab, cmin, cmax, cmap, cblab)
plt.subplots_adjust(hspace=.2, wspace=.3)

##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    if chAl==1:
        dd = 'chAl_anglesAveDays_inhExcAllExc_data_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_anglesAveDays_inhExcAllExc_data_' + days[0][0:6] + '-to-' + days[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




# shfl
plt.figure(figsize=(8,8))
cmin = np.floor(np.min([np.nanmin(angInhS_av), np.nanmin(angExcS_av), np.nanmin(angAllExcS_av)]))
cmax = 90
cmap = 'jet_r' # lower angles: more aligned: red
cblab = 'Angle between decoders'
lab = 'inh (shfl)'; top = angInhS_av; plt.subplot(221); plotAng(top, lab, cmin, cmax, cmap, cblab)
lab = 'exc (shfl)'; top = angExcS_av; plt.subplot(223); plotAng(top, lab, cmin, cmax, cmap, cblab)
lab = 'allExc (shfl)'; top = angAllExcS_av; plt.subplot(222); plotAng(top, lab, cmin, cmax, cmap, cblab)
plt.subplots_adjust(hspace=.2, wspace=.3)

##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    if chAl==1:
        dd = 'chAl_anglesAveDays_inhExcAllExc_shfl_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_anglesAveDays_inhExcAllExc_shfl_' + days[0][0:6] + '-to-' + days[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




##### Most useful plot:
# shfl-real 
plt.figure(figsize=(8,8))
cmin = np.floor(np.min([np.nanmin(angInhS_av-angInh_av), np.nanmin(angExcS_av-angExc_av), np.nanmin(angAllExcS_av-angAllExc_av)]))
cmax = np.floor(np.max([np.nanmax(angInhS_av-angInh_av), np.nanmax(angExcS_av-angExc_av), np.nanmax(angAllExcS_av-angAllExc_av)]))
cmap = 'jet' # larger value: lower angle for real: more aligned: red
cblab = 'Angle between decoders'
lab = 'inh (shfl-data)'; 
top = angInhS_av - angInh_av; 
top[np.triu_indices(len(top),k=0)] = np.nan # plot only the lower triangle (since values are rep)
plt.subplot(221); plotAng(top, lab, cmin, cmax, cmap, cblab)
lab = 'exc (shfl-data)'; 
top = angExcS_av - angExc_av; 
top[np.triu_indices(len(top),k=0)] = np.nan
plt.subplot(223); plotAng(top, lab, cmin, cmax, cmap, cblab)
lab = 'allExc (shfl-data)'; 
top = angAllExcS_av - angAllExc_av; 
top[np.triu_indices(len(top),k=0)] = np.nan
plt.subplot(222); plotAng(top, lab, cmin, cmax, cmap, cblab)

# last subplot: inh(shfl-real) - exc(shfl-real)
cmin = np.nanmin((angInhS_av - angInh_av) - (angExcS_av - angExc_av)); 
cmax = np.nanmax((angInhS_av - angInh_av) - (angExcS_av - angExc_av))
cmap = 'jet' # more diff: higher inh: lower angle for inh: inh more aligned: red
cblab = 'Inh-Exc: angle between decoders'
lab = 'inh-exc'; 
top = ((angInhS_av - angInh_av) - (angExcS_av - angExc_av)); 
top[np.triu_indices(len(top),k=0)] = np.nan
plt.subplot(224); plotAng(top, lab, cmin, cmax, cmap, cblab)

# Is inh and exc stability different? 
# for each population (inh,exc) subtract shuffle-averaged angles from real angles... do ttest across days for each pair of time points
_,pei = stats.ttest_ind(angleInh_aligned - np.mean(angleInhS_aligned,axis=2) , angleExc_aligned - np.mean(angleExcS_aligned,axis=2), axis=-1) 
pei[np.triu_indices(len(pei),k=0)] = np.nan
# mark sig points by a dot:
for f1 in range(r):
    for f2 in range(r): #np.delete(range(r),f1): #
        if pei[f1,f2]<.05:
            plt.plot(f2,f1, marker='.', color='b', markersize=2)

plt.subplots_adjust(hspace=.2, wspace=.3)

##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    if chAl==1:
        dd = 'chAl_anglesAveDays_inhExcAllExc_dataMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_anglesAveDays_inhExcAllExc_dataMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



    
'''
cmin = np.nanmin(pei); cmax = np.nanmax(pei)
top = pei; lab = 'p'; plt.subplot(222); plotAng(top, lab, cmin, cmax)
cmin = 0; cmax = 1
top = pei<.05; lab = 'p<.05'; plt.subplot(223); plotAng(top, lab, cmin, cmax); plt.set_cmap('jet'); 
'''

#%% Set diagonal elements (whose angles are 0) to nan, so heatmaps span a smaller range and can be better seen.
           
[np.fill_diagonal(angleInh_aligned[:,:,iday], np.nan) for iday in range(len(days))];
[np.fill_diagonal(angleExc_aligned[:,:,iday], np.nan) for iday in range(len(days))];
[np.fill_diagonal(angleAllExc_aligned[:,:,iday], np.nan) for iday in range(len(days))];

# shuffled
[np.fill_diagonal(sigAngInh[:,:,iday], np.nan) for iday in range(len(days))];
[np.fill_diagonal(sigAngExc[:,:,iday], np.nan) for iday in range(len(days))];
[np.fill_diagonal(sigAngAllExc[:,:,iday], np.nan) for iday in range(len(days))];


#%% Compute a measure for stability of each decoder

# define days to be analyzed 
dayinds = np.arange(len(days)); 
# use below if you want at least 10 of each hr,lr trials for a day to be included in the analysis
dayinds = np.delete(dayinds, np.argwhere(ntrs < thnt))

dostabplots = 0 # set to 1 if you want to compare the measures below
 
# I find meas=3 the best one. 
for meas in [2]: #range(3): # which one of the measures below to use

    if meas==0:    # number of decoders that are "aligned" with each decoder (at each timepoint) (we get this by summing sigAng across rows gives us the)
                   # sigAng: for each day shows whether angle betweeen the decoders at specific timepoints is significantly different than null dist (ie angle is smaller than 5th percentile of shuffle angles).
        stabScoreInh = np.array([np.sum(sigAngInh[:,:,iday], axis=0) for iday in dayinds]) 
        stabScoreExc = np.array([np.sum(sigAngExc[:,:,iday], axis=0) for iday in dayinds]) 
        stabScoreAllExc = np.array([np.sum(sigAngAllExc[:,:,iday], axis=0) for iday in dayinds]) 
        stabLab = 'Stability (number of aligned decoders)' # with the decoder of interest (ie the decoder at the specific timepont we are looking at)
        # aligned decoder is defined as a decoder whose angle with the decoder of interest is <5th percentile of dist of shuffled angles.
        
    elif meas==1: # how many decoders are at least 10 degrees away from the null decoder. # how stable is each decoder at other time points measure as how many decoders are at 70 degrees or less with each of the decoders (trained on a particular frame).. or ... .      
        th = 10  
        #np.mean(angle_aligned[:,:,iday],axis=0) # average stability ? (angle with all decoders at other time points)
        #stabAng = angle_aligned[:,:,iday] < th # how stable is each decoder at other time points measure as how many decoders are at 70 degrees or less with each of the decoders (trained on a particular frame).. or ... .
    #    stabScoreInh = np.array([np.sum(angleInh_aligned[:,:,iday]<th, axis=0) for iday in dayinds]) 
    #    stabScoreExc = np.array([np.sum(angleExc_aligned[:,:,iday]<th, axis=0) for iday in dayinds]) 
        stabScoreInh = np.array([np.sum(((np.mean(angleInhS_aligned,axis=2))[:,:,iday] - angleInh_aligned[:,:,iday]) >= th, axis=0) for iday in dayinds]) 
        stabScoreExc = np.array([np.sum(((np.mean(angleExcS_aligned,axis=2))[:,:,iday] - angleExc_aligned[:,:,iday]) >= th, axis=0) for iday in dayinds])     
        stabScoreAllExc = np.array([np.sum(((np.mean(angleAllExcS_aligned,axis=2))[:,:,iday] - angleAllExc_aligned[:,:,iday]) >= th, axis=0) for iday in dayinds]) 
        stabLab = 'Stability (number of decoders > 10 degrees away from full mis-alignment )'
    
    elif meas==2:  # Average angle of each decoder with other decoders (shfl-data)
    #    stabScoreInh = np.array([np.nanmean(angleInh_aligned[:,:,iday], axis=0) for iday in dayinds]) 
    #    stabScoreExc = np.array([np.nanmean(angleExc_aligned[:,:,iday], axis=0) for iday in dayinds]) 
        stabScoreInh = np.array([np.nanmean((np.mean(angleInhS_aligned,axis=2))[:,:,iday] - angleInh_aligned[:,:,iday] , axis=0) for iday in dayinds]) 
        stabScoreExc = np.array([np.nanmean((np.mean(angleExcS_aligned,axis=2))[:,:,iday] - angleExc_aligned[:,:,iday] , axis=0) for iday in dayinds]) 
        stabScoreAllExc = np.array([np.nanmean((np.mean(angleAllExcS_aligned,axis=2))[:,:,iday] - angleAllExc_aligned[:,:,iday] , axis=0) for iday in dayinds])         
        stabLab = 'Stability (degrees away from full mis-alignment)'    
        
    elif meas==3:  # percentile of shuffled angles that are bigger than the real angle (ie more mis-aligned)
        # not a good measure bc does not properly distinguish eg .99 from .94 (.99 is significant, but .94 is not)
        stabScoreInh = np.array([np.nanmean(percAngInh[:,:,iday], axis=0) for iday in dayinds]) 
        stabScoreExc = np.array([np.nanmean(percAngExc[:,:,iday], axis=0) for iday in dayinds]) 
        stabScoreAllExc = np.array([np.nanmean(percAngAllExc[:,:,iday], axis=0) for iday in dayinds]) 
        stabLab = 'Stability (number of decoders aligned with)'
        
        
    cmins = np.floor(np.min([np.nanmin(stabScoreInh), np.nanmin(stabScoreExc), np.nanmin(stabScoreAllExc)]))
    cmaxs = np.floor(np.max([np.nanmax(stabScoreInh), np.nanmax(stabScoreExc), np.nanmax(stabScoreAllExc)]))
        
    #################### stability
    if dostabplots:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
        
        lab = 'inh'
        top = stabScoreInh
        cmap ='jet'
        cblab = stabLab
        ax = axes.flat[0]
        im = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
        
        lab = 'exc'
        top = stabScoreExc
        cmap ='jet'
        cblab = stabLab
        ax = axes.flat[1]
        plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
        
        lab = 'allExc'
        top = stabScoreAllExc
        cmap ='jet'
        cblab = stabLab
        ax = axes.flat[2]
        im = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
        #plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
        #add_colorbar(im)
        cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
        cbar = fig.colorbar(im, cax=cb_ax, label=cblab)
        #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        #fig.colorbar(im, ax=axes.ravel().tolist())
        # Make an axis for the colorbar on the right side
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(im, cax=cax)
        
        plt.subplots_adjust(wspace=.4)



#%% Class accuracy

classAccurTMS_inh = np.array([(av_test_data_inh_aligned[:, iday] - av_test_shfl_inh_aligned[:, iday]) for iday in dayinds])
classAccurTMS_exc = np.array([(av_test_data_exc_aligned[:, iday] - av_test_shfl_exc_aligned[:, iday]) for iday in dayinds])
classAccurTMS_allExc = np.array([(av_test_data_allExc_aligned[:, iday] - av_test_shfl_allExc_aligned[:, iday]) for iday in dayinds])

cminc = np.floor(np.min([np.nanmin(classAccurTMS_inh), np.nanmin(classAccurTMS_exc), np.nanmin(classAccurTMS_allExc)]))
cmaxc = np.floor(np.max([np.nanmax(classAccurTMS_inh), np.nanmax(classAccurTMS_exc), np.nanmax(classAccurTMS_allExc)]))

'''
##%% Set the colormap

from matplotlib import cm
from numpy import linspace
start = 0.0
stop = 1.0
number_of_lines = len(days)
cm_subsection = linspace(start, stop, number_of_lines) 
colors = [ cm.jet(x) for x in cm_subsection ]
           
# change color order to jet 
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
'''

#%% significancy of stability and class accurr across days

_,ps = stats.ttest_ind(stabScoreInh, stabScoreExc) 
_,pc = stats.ttest_ind(classAccurTMS_inh, classAccurTMS_exc) 

_,psa = stats.ttest_ind(stabScoreInh, stabScoreAllExc) 
_,pca = stats.ttest_ind(classAccurTMS_inh, classAccurTMS_allExc) 

plt.figure()
plt.subplot(221); plt.plot(ps); plt.ylabel('p_stability'); makeNicePlots(plt.gca())
plt.title('exc vs inh'); 
plt.xticks(x, np.round(time_aligned[x]).astype(int))
plt.axhline(y=.05, c='r', lw=1)

plt.subplot(223); plt.plot(pc); plt.ylabel('p_class accur'); makeNicePlots(plt.gca())
plt.xticks(x, np.round(time_aligned[x]).astype(int))
plt.axhline(y=.05, c='r', lw=1)

plt.subplot(222); plt.plot(psa); plt.ylabel('p_stability'); makeNicePlots(plt.gca())
plt.title('all exc vs inh')
plt.xticks(x, np.round(time_aligned[x]).astype(int))
plt.axhline(y=.05, c='r', lw=1)

plt.subplot(224); plt.plot(pca); plt.ylabel('p_class accur'); makeNicePlots(plt.gca())
plt.xticks(x, np.round(time_aligned[x]).astype(int))
plt.axhline(y=.05, c='r', lw=1)

plt.subplots_adjust(hspace=.4, wspace=.5)


##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    daysnew = (np.array(days))[dayinds]
    if chAl==1:
        dd = 'chAl_stab_classAcc_sig_inhExcAllExc_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]
    else:
        dd = 'stAl_stab_classAcc_sig_inhExcAllExc_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



#%%
#plt.plot(stabScore.T);
#plt.legend(days, loc='center left', bbox_to_anchor=(1, .7))

step = 4
x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
           np.arange(np.argwhere(time_aligned>=0)[0], r, step))))).astype(int)
#cmins,cmaxs, cminc,cmaxc
asp = 'auto' #2 #

def plotStabScore(top, lab, cmins, cmaxs, cmap='jet', cblab='', ax=plt):
#    im = ax.imshow(top, cmap)
    im = ax.imshow(top, cmap, vmin=cmins, vmax=cmaxs)
#    plt.plot([nPreMin, nPreMin], [0, len(dayinds)], color='r')
    ax.set_xlim([-1, len(time_aligned)])
    ax.set_ylim([-2, len(dayinds)][::-1])
#    ax.autoscale(False)
    ax.axvline(x=nPreMin, c='w', lw=1)
#    fig.colorbar(label=cblab);     
#    plt.clim(cmins, cmaxs)
    ax.set_xticks(x)
    ax.set_xticklabels(np.round(time_aligned[x]).astype(int))
    ax.set_ylabel('Days')
    ax.set_xlabel(xl)
    ax.set_title(lab)
    makeNicePlots(ax)
    return im
  
  
#################### stability
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7,5))

lab = 'inh'
top = stabScoreInh
cmap ='jet'
cblab = stabLab
ax = axes.flat[0]
im = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
ax.set_aspect(asp)


lab = 'exc'
top = stabScoreExc
cmap ='jet'
cblab = stabLab
ax = axes.flat[1]
plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
# mark sig timepoints (exc sig diff from inh)
y=-1; pp = np.full(ps.shape, np.nan); pp[ps<=.05] = y
ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
ax.set_aspect(asp)


lab = 'allExc'
top = stabScoreAllExc
cmap ='jet'
cblab = stabLab
ax = axes.flat[2]
im = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
#plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
#add_colorbar(im)
pos1 = ax.get_position()
cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72], adjustable='box-forced')
#cb_ax = fig.add_axes([0.92, np.prod([pos1.y0,asp]), 0.02, pos1.width])
cbar = fig.colorbar(im, cax=cb_ax, label=cblab)
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
#fig.colorbar(im, ax=axes.ravel().tolist())
# Make an axis for the colorbar on the right side
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#fig.colorbar(im, cax=cax)
# mark sig timepoints (all exc sig diff from inh)
y=-1; pp = np.full(ps.shape, np.nan); pp[psa<=.05] = y
ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
ax.set_aspect(asp)
#fig.tight_layout()   

plt.subplots_adjust(wspace=.4)

##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    daysnew = (np.array(days))[dayinds]
    if chAl==1:
        dd = 'chAl_stab_eachDay_inhExcAllExc_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]
    else:
        dd = 'stAl_stab_eachDay_inhExcAllExc_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




######################## class accuracy
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7,5))

lab = 'inh'
top = classAccurTMS_inh
cmap ='jet'
cblab = 'Class accuracy (data-shfl)'
ax = axes.flat[0]
im = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax)
ax.set_aspect(asp)


lab = 'exc'
top = classAccurTMS_exc
cmap ='jet'
cblab = 'Class accuracy (data-shfl)'
ax = axes.flat[1]
plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax)
y=-1; pp = np.full(ps.shape, np.nan); pp[pc<=.05] = y
ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
ax.set_aspect(asp)


lab = 'allExc'
top = classAccurTMS_allExc
cmap ='jet'
cblab = 'Class accuracy (data-shfl)'
ax = axes.flat[2]
im = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax)
#plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
#add_colorbar(im)
cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
cbar = fig.colorbar(im, cax=cb_ax, label=cblab)
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
#fig.colorbar(im, ax=axes.ravel().tolist())
# Make an axis for the colorbar on the right side
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#fig.colorbar(im, cax=cax)
y=-1; pp = np.full(ps.shape, np.nan); pp[pca<=.05] = y
ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
ax.set_aspect(asp)

plt.subplots_adjust(wspace=.4)

##%% Save the figure    
if savefigs:
    d = os.path.join(svmdir+dnow) #,mousename)       
    daysnew = (np.array(days))[dayinds]
    if chAl==1:
        dd = 'chAl_classAcc_eachDay_inhExcAllExc_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]
    else:
        dd = 'stAl_classAcc_eachDay_inhExcAllExc_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6]       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)






#%% Compare number of inh vs exc for each day
'''
plt.figure(figsize=(4.5,3)) #plt.figure()
plt.plot(numInh, label='inh', color='r')
plt.plot(numAllexc, label='allExc', color='k')
plt.title('average: inh=%d exc=%d inh/exc=%.2f' %(np.round(numInh.mean()), np.round(numAllexc.mean()), np.mean(numInh) / np.mean(numAllexc)))
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
plt.xlabel('Days')
plt.ylabel('# Neurons')
plt.xticks(range(len(days)), [days[i][0:6] for i in range(len(days))], rotation='vertical')

makeNicePlots(plt.gca(),1)


if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
'''    
    
    


########
'''
plt.figure(figsize=(8,15))

# inh
lab = 'inh'
top = stabScoreInh
cmap ='jet'
cblab = stabLab
plt.subplot(231)
plotStabScore(top, lab, cmins, cmaxs, cmap, cblab)

top = classAccurTMS_inh
cmap ='jet'
cblab = 'Class accuracy (data-shfl)'
plt.subplot(234)
plotStabScore(top, lab, cminc, cmaxc, cmap, cblab)


# exc
lab = 'exc'
top = stabScoreExc
cmap ='jet'
cblab = stabLab
plt.subplot(232)
plotStabScore(top, lab, cmins, cmaxs, cmap, cblab)

top = classAccurTMS_exc
cmap ='jet'
cblab = 'Class accuracy (data-shfl)'
plt.subplot(235)
plotStabScore(top, lab, cminc, cmaxc, cmap, cblab)


# allExc
lab = 'allExc'
top = stabScoreAllExc
cmap ='jet'
cblab = stabLab
plt.subplot(233)
im = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab)
#plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
add_colorbar(im)

top = classAccurTMS_allExc
cmap ='jet'
cblab = 'Class accuracy (data-shfl)'
plt.subplot(236)
im = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab)
plt.colorbar(label=cblab,fraction=0.046, pad=0.04); #plt.clim(cminc, cmaxc)    
cbar_ax = plt.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(im, cax=cbar_ax)

plt.subplots_adjust(wspace=.6)
'''
