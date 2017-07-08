# -*- coding: utf-8 -*-
"""
Summary of all mice: 
Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     


#%%
mice = 'fni19', #'fni16', 'fni17', 'fni18', 'fni19' # if you want to use only one mouse, make sure you put comma at the end; eg. mice = 'fni19',


loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)

time2an = -1; # relative to eventI, look at classErr in what time stamp.
chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
savefigs = 1
superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
loadWeights = 0

# following will be needed for 'fni18': #set one of the following to 1:
allDays = 0# all 7 days will be used (last 3 days have z motion!)
noZmotionDays = 1 # 4 days that dont have z motion will be used.
noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

#doPlots = 0 #1 # plot c path of each day 
#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
    
smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
    

dnow0 = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'
        


#%% Define common funcitons

execfile("defFuns.py")


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
    goodSamps = aa > thNon0Ws # samples with >.05 of neurons having non-0 weight 
#        sum(goodSamps) # for each frame, it shows number of samples with >.05 of neurons having non-0 weight        
    
    if sum(sum(~goodSamps))>0:
        print 'All frames together have %d bad samples (samples w < %.2f non-0 weights)... setting their classErr to 50' %(sum(sum(~goodSamps)), thNon0Ws)
        
        if sum(sum(goodSamps)<thSamps)>0:
            print 'There are %d frames with < %d good samples... setting all samples classErr to 50' %(sum(goodSamps)<thSamps, thSamps)
    
        if w.ndim==3: 
            classError[~goodSamps] = 50 # set to 50 class error of samples which have <=.05 of non-0-weight neurons
            classError[:,sum(goodSamps)<thSamps] = 50 # if fewer than 10 samples contributed to a frame, set the perClassError of all samples for that frame to 50...       
        elif w.ndim==4: #exc : average across excShuffles
            classError[:,~goodSamps] = 50 # set to 50 class error of samples which have <=.05 of non-0-weight neurons
            classError[:,:,sum(goodSamps)<thSamps] = 50 # if fewer than 10 samples contributed to a frame, set the perClassError of all samples for that frame to 50...       
 
   
    modClassError = classError+0
    
    return modClassError
    

#%%
av_test_data_inh_allMice = []
sd_test_data_inh_allMice = []
av_test_data_exc_allMice = []
sd_test_data_exc_allMice = []
av_test_data_allExc_allMice = []
sd_test_data_allExc_allMice = []

av_test_shfl_inh_allMice = []
sd_test_shfl_inh_allMice = []
av_test_shfl_exc_allMice = []
sd_test_shfl_exc_allMice = []
av_test_shfl_allExc_allMice = []
sd_test_shfl_allExc_allMice = []

av_test_chance_inh_allMice = []
sd_test_chance_inh_allMice = []
av_test_chance_exc_allMice = []
sd_test_chance_exc_allMice = []
av_test_chance_allExc_allMice = []
sd_test_chance_allExc_allMice = []


#%%
for im in range(len(mice)):
        
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    execfile("svm_plots_setVars_n.py")      
#    execfile("svm_plots_setVars.py")      
    
    #%%     
    if loadInhAllexcEqexc==1:
        dnow = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'+mousename+'/'
    else:
        dnow = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'+mousename+'/inhRois/'
              
                
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
    if loadWeights==1:            
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

            if loadWeights==1:    
                Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh',    
                                                         'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc',    
                                                         'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc',
                                                         'w_data_inh', 'w_data_allExc', 'w_data_exc'])        
            else:
                Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh',    
                                                     'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc',    
                                                     'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc'])
            Datai = Dataae = Datae = Data                                                 
    
        else:  # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
            
            for idi in range(3):
                doInhAllexcEqexc = np.full((3), False)
                doInhAllexcEqexc[idi] = True 
                svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins)        
                svmName = svmName[0]
                print os.path.basename(svmName)    
    
                if doInhAllexcEqexc[0] == 1:
                    if loadWeights==1:
                        Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'w_data_inh'])
                    else:
                        Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh'])
                                            
                elif doInhAllexcEqexc[1] == 1:
                    if loadWeights==1:                    
                        Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc'])
                    else:
                        Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc'])

                elif doInhAllexcEqexc[2] == 1:
                    if loadWeights==1:                                        
                        Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'w_data_exc'])
                    else:
                        Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc'])
            
        ###%%             
        perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh')
        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh')
        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh') 
        if loadWeights==1:
            w_data_inh = Datai.pop('w_data_inh') 
        
        perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc')
        perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc')
        perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc')   
        if loadWeights==1:
            w_data_allExc = Dataae.pop('w_data_allExc') 
        
        perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_data_exc')    
        perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_exc')
        perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_exc')
        if loadWeights==1:
            w_data_exc = Datae.pop('w_data_exc') 
        
       
    
        #%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.

        if loadWeights==1:            
            perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
            perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
            perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
            
            perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
            perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
            perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
            
            perClassErrorTest_data_exc = setTo50classErr(perClassErrorTest_data_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
            perClassErrorTest_shfl_exc = setTo50classErr(perClassErrorTest_shfl_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
            perClassErrorTest_chance_exc = setTo50classErr(perClassErrorTest_chance_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
            #% Get number of inh and exc        
            numInh[iday] = w_data_inh.shape[1]
            numAllexc[iday] = w_data_allExc.shape[1]
        
        
        #%% Get class error values for a specific time point
        
        perClassErrorTest_data_inh = perClassErrorTest_data_inh[:,eventI_ds+time2an].squeeze() # numSamps
        perClassErrorTest_shfl_inh = perClassErrorTest_shfl_inh[:,eventI_ds+time2an].squeeze() # numSamps
        perClassErrorTest_chance_inh = perClassErrorTest_chance_inh[:,eventI_ds+time2an].squeeze() # numSamps
        
        perClassErrorTest_data_allExc = perClassErrorTest_data_allExc[:,eventI_ds+time2an].squeeze() # numSamps
        perClassErrorTest_shfl_allExc = perClassErrorTest_shfl_allExc[:,eventI_ds+time2an].squeeze() # numSamps
        perClassErrorTest_chance_allExc = perClassErrorTest_chance_allExc[:,eventI_ds+time2an].squeeze() # numSamps
        
        perClassErrorTest_data_exc = perClassErrorTest_data_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
        perClassErrorTest_shfl_exc = perClassErrorTest_shfl_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
        perClassErrorTest_chance_exc = perClassErrorTest_chance_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
    
    
        #%% Append vars for all days
        
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
        if loadWeights==1: 
            del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc
        else:
            del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    
    
    
    ######################################################################################################################################################               
    ######################################################################################################################################################        
    #%% Done with all days
    
    eventI_allDays = eventI_allDays.astype('int')
    
    perClassErrorTest_data_inh_all = np.array(perClassErrorTest_data_inh_all)
    perClassErrorTest_shfl_inh_all = np.array(perClassErrorTest_shfl_inh_all)
    perClassErrorTest_chance_inh_all = np.array(perClassErrorTest_chance_inh_all)
    perClassErrorTest_data_allExc_all = np.array(perClassErrorTest_data_allExc_all)
    perClassErrorTest_shfl_allExc_all = np.array(perClassErrorTest_shfl_allExc_all)
    perClassErrorTest_chance_allExc_all = np.array(perClassErrorTest_chance_allExc_all)
    perClassErrorTest_data_exc_all = np.array(perClassErrorTest_data_exc_all) # numShufflesExc x numSamples x numFrames
    perClassErrorTest_shfl_exc_all = np.array(perClassErrorTest_shfl_exc_all)
    perClassErrorTest_chance_exc_all = np.array(perClassErrorTest_chance_exc_all)
    
    
    #%% Average and std of class accuracies across CV samples ... for each day
    
    numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
    numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]
    
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
    
    
    #%% Keep values of all mice (cvSample-averaged)
    
    av_test_data_inh_allMice.append(av_test_data_inh)
    sd_test_data_inh_allMice.append(sd_test_data_inh)
    av_test_data_exc_allMice.append(av_test_data_exc)
    sd_test_data_exc_allMice.append(sd_test_data_exc)
    av_test_data_allExc_allMice.append(av_test_data_allExc)
    sd_test_data_allExc_allMice.append(sd_test_data_allExc)
    
    av_test_shfl_inh_allMice.append(av_test_shfl_inh)
    sd_test_shfl_inh_allMice.append(sd_test_shfl_inh)
    av_test_shfl_exc_allMice.append(av_test_shfl_exc)
    sd_test_shfl_exc_allMice.append(sd_test_shfl_exc)
    av_test_shfl_allExc_allMice.append(av_test_shfl_allExc)
    sd_test_shfl_allExc_allMice.append(sd_test_shfl_allExc)

    av_test_chance_inh_allMice.append(av_test_chance_inh)
    sd_test_chance_inh_allMice.append(sd_test_chance_inh)
    av_test_chance_exc_allMice.append(av_test_chance_exc)
    sd_test_chance_exc_allMice.append(sd_test_chance_exc)
    av_test_chance_allExc_allMice.append(av_test_chance_allExc)
    sd_test_chance_allExc_allMice.append(sd_test_chance_allExc)

    
    
    #%%
    ######################## PLOTS ########################
    
    #%% Plot class accuracy in the frame before the choice onset for each session
    
    plt.figure(figsize=(4.5,3))
    plt.errorbar(range(numDays), av_test_data_inh, sd_test_data_inh, label='inh', color='r')
    plt.errorbar(range(numDays), av_test_data_exc, sd_test_data_exc, label='exc', color='b')
    plt.errorbar(range(numDays), av_test_data_allExc, sd_test_data_allExc, label='allExc', color='k')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
    plt.xlabel('Days')
    plt.ylabel('Classification accuracy (%)')
    makeNicePlots(plt.gca())
    plt.xlim([-.2,len(days)-1+.2])
    
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_eachDay'
        else:
            dd = 'stAl_eachDay'
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
    _,pcorrtrace = stats.ttest_ind(av_test_data_inh, av_test_data_exc) # p value of class accuracy being different from 50
    print pcorrtrace     
    
    

######################################################################################################################################################        
######################################################################################################################################################        
#%%
av_test_data_inh_allMice = np.array(av_test_data_inh_allMice)
sd_test_data_inh_allMice = np.array(sd_test_data_inh_allMice)
av_test_data_exc_allMice = np.array(av_test_data_exc_allMice)
sd_test_data_exc_allMice = np.array(sd_test_data_exc_allMice)
av_test_data_allExc_allMice = np.array(av_test_data_allExc_allMice)
sd_test_data_allExc_allMice = np.array(sd_test_data_allExc_allMice)

av_test_shfl_inh_allMice = np.array(av_test_shfl_inh_allMice)
sd_test_shfl_inh_allMice = np.array(sd_test_shfl_inh_allMice)
av_test_shfl_exc_allMice = np.array(av_test_shfl_exc_allMice)
sd_test_shfl_exc_allMice = np.array(sd_test_shfl_exc_allMice)
av_test_shfl_allExc_allMice = np.array(av_test_shfl_allExc_allMice)
sd_test_shfl_allExc_allMice = np.array(sd_test_shfl_allExc_allMice)

av_test_chance_inh_allMice = np.array(av_test_chance_inh_allMice)
sd_test_chance_inh_allMice = np.array(sd_test_chance_inh_allMice)
av_test_chance_exc_allMice = np.array(av_test_chance_exc_allMice)
sd_test_chance_exc_allMice = np.array(sd_test_chance_exc_allMice)
av_test_chance_allExc_allMice = np.array(av_test_chance_allExc_allMice)
sd_test_chance_allExc_allMice = np.array(sd_test_chance_allExc_allMice)


#%% Average classErr across sessions for each mouse

av_av_test_data_inh_allMice = np.array([np.nanmean(av_test_data_inh_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_data_exc_allMice = np.array([np.nanmean(av_test_data_exc_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_data_allExc_allMice = np.array([np.nanmean(av_test_data_allExc_allMice[im], axis=0) for im in range(len(mice))])

sd_av_test_data_inh_allMice = np.array([np.nanstd(av_test_data_inh_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_data_exc_allMice = np.array([np.nanstd(av_test_data_exc_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_data_allExc_allMice = np.array([np.nanstd(av_test_data_allExc_allMice[im], axis=0) for im in range(len(mice))])

av_av_test_shfl_inh_allMice = np.array([np.nanmean(av_test_shfl_inh_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_shfl_exc_allMice = np.array([np.nanmean(av_test_shfl_exc_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_shfl_allExc_allMice = np.array([np.nanmean(av_test_shfl_allExc_allMice[im], axis=0) for im in range(len(mice))])

sd_av_test_shfl_inh_allMice = np.array([np.nanstd(av_test_shfl_inh_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_shfl_exc_allMice = np.array([np.nanstd(av_test_shfl_exc_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_shfl_allExc_allMice = np.array([np.nanstd(av_test_shfl_allExc_allMice[im], axis=0) for im in range(len(mice))])

av_av_test_chance_inh_allMice = np.array([np.nanmean(av_test_chance_inh_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_chance_exc_allMice = np.array([np.nanmean(av_test_chance_exc_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_chance_allExc_allMice = np.array([np.nanmean(av_test_chance_allExc_allMice[im], axis=0) for im in range(len(mice))])

sd_av_test_chance_inh_allMice = np.array([np.nanstd(av_test_chance_inh_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_chance_exc_allMice = np.array([np.nanstd(av_test_chance_exc_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_chance_allExc_allMice = np.array([np.nanstd(av_test_chance_allExc_allMice[im], axis=0) for im in range(len(mice))])


#%% Plot session-averaged classErr for each mouse

plt.figure(figsize=(2,3))

plt.errorbar(range(len(mice)), av_av_test_data_inh_allMice, sd_av_test_data_inh_allMice, fmt='o', label='inh', color='r')
plt.errorbar(range(len(mice)), av_av_test_data_exc_allMice, sd_av_test_data_exc_allMice, fmt='o', label='exc', color='b')
plt.errorbar(range(len(mice)), av_av_test_data_allExc_allMice, sd_av_test_data_allExc_allMice, fmt='o', label='allExc', color='k')
'''
plt.errorbar(range(len(mice)), av_av_test_shfl_inh_allMice, sd_av_test_shfl_inh_allMice, color='r', fmt='o')
plt.errorbar(range(len(mice)), av_av_test_shfl_exc_allMice, sd_av_test_shfl_exc_allMice, color='b', fmt='o')
plt.errorbar(range(len(mice)), av_av_test_shfl_allExc_allMice, sd_av_test_shfl_allExc_allMice, color='k', fmt='o')
'''

plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
plt.xlabel('Mice', fontsize=11)
plt.ylabel('Classification accuracy (%)', fontsize=11)
plt.xlim([-.2,len(mice)-1+.2])
plt.xticks(range(len(mice)),mice)
ax = plt.gca()
makeNicePlots(ax)


if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_' + '_'.join(mice) #'chAl_allMice'
    else:
        dd = 'stAl_' + '_'.join(mice) #'stAl_allMice'
    
    if loadInhAllexcEqexc==0:        
        dd = dd+'_inhRois'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow0, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)





