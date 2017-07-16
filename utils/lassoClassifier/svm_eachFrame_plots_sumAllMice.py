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
mice = 'fni16', 'fni17', 'fni18'


time2an = -1; # relative to eventI, look at classErr in what time stamp.
chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
savefigs = 0
superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
#loadWeights = 1

# following will be needed for 'fni18': #set one of the following to 1:
allDays = 0# all 7 days will be used (last 3 days have z motion!)
noZmotionDays = 1 # 4 days that dont have z motion will be used.
noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

doPlots = 0 #1 # plot c path of each day 
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
    
dnow0 = '/classAccurTraces_eachFrame/frame'+str(time2an)+'/'
       


#%% Define common funcitons

execfile("defFuns.py")


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins=3):
    import glob

    if chAl==1:
        al = 'chAl'
    else:
        al = 'stAl'

    if trialHistAnalysis:
        svmn = 'svmPrevChoice_eachFrame_%s_ds%d_*' %(al,regressBins)
    else:
        svmn = 'svmCurrChoice_eachFrame_%s_ds%d_*' %(al,regressBins)
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    
#    if len(svmName)>0:
#        svmName = svmName[0] # get the latest file
#    else:
#        svmName = ''
#    
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
    
    
#%% function to set best c and get the class error values at best C

def setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, doPlots):
    
    numSamples = perClassErrorTest.shape[0] 
    numFrs = perClassErrorTest.shape[2]
    
    #%% Compute average of class errors across numSamples
    
    classErr_bestC_train_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_shfl = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_chance = np.full((numSamples, numFrs), np.nan)
    cbestFrs = np.full((numFrs), np.nan)
#    numNon0SampData = np.full((numFrs), np.nan)       
        
    for ifr in range(numFrs):
                
        meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
        semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        
        #%% Identify best c       
        
        # Use all range of c... it may end up a value at which all weights are 0.
        ix = np.argmin(meanPerClassErrorTest)
        if smallestC==1:
            cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria
            cbestAll = cbest
        else:
            cbestAll = cvect[ix]
        print 'best c = %.10f' %cbestAll
        
        
        
        #### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        if regType == 'l1': # in l2, we don't really have 0 weights!
            sys.exit('Needs work! below wAllC has to be for 1 frame') 
            
            a = abs(wAllC)>eps # non-zero weights
            b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
            c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
            cvectnow = cvect[c1stnon0:]
            
            meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:,ifr], axis = 0);
            semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:,ifr], axis = 0)/np.sqrt(numSamples);
            ix = np.argmin(meanPerClassErrorTestnow)
            if smallestC==1:
                cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria    
            else:
                cbest = cvectnow[ix]
    
            print 'best c (at least 1 non-0 weight) = ', cbest
            
        else:
            cbest = cbestAll            
        
    
        cbestFrs[ifr] = cbest
        
        
        #%% Set the decoder and class errors at best c 
        
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)
        
    #            w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
    #            b_bestc_data = bAllC[:,indBestC,ifr]            
        classErr_bestC_train_data[:,ifr] = perClassErrorTrain[:,indBestC,ifr].squeeze() # numSamps           
        classErr_bestC_test_data[:,ifr] = perClassErrorTest[:,indBestC,ifr].squeeze()
        classErr_bestC_test_shfl[:,ifr] = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
        classErr_bestC_test_chance[:,ifr] = perClassErrorTest_chance[:,indBestC,ifr].squeeze()        
    
       # check the number of non-0 weights
    #       for ifr in range(numFrs):
    #        w_data = wAllC[:,indBestC,:,ifr].squeeze()
    #        print np.sum(w_data > eps,axis=1)
    #        ada = np.sum(w_data > eps,axis=1) < thNon0Ws # samples w fewer than 2 non-0 weights        
    #        ada = ~ada # samples w >=2 non0 weights
    #        numNon0SampData[ifr] = ada.sum() # number of cv samples with >=2 non-0 weights
    
        
        #%% Plot C path           
        
        if doPlots:
    #            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure()
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
        #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
        #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
        
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            
            cl = 'r' if ifr==eventI_ds else 'k' # show frame number of eventI in red :)
            plt.title('Frame %d' %(ifr), color=cl)
            plt.tight_layout()          
           

    return classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs
    
    

#%%
av_train_data_allMice = []
sd_train_data_allMice = []

av_test_data_allMice = []
sd_test_data_allMice = []

av_test_shfl_allMice = []
sd_test_shfl_allMice = []

av_test_chance_allMice = []
sd_test_chance_allMice = []


#%%
for im in range(len(mice)):
        
    #%%            
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    execfile("svm_plots_setVars_n.py")      
#    execfile("svm_plots_setVars.py")      
    
    
    #%%     
    dnow = '/classAccurTraces_eachFrame/frame'+str(time2an)+'/'+mousename+'/'
              
                
    #%% Loop over days    
    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    classErr_bestC_train_data_all = [] # np.full((numSamp*nRandCorrSel, len(days)), np.nan)
    classErr_bestC_test_data_all = []
    classErr_bestC_test_shfl_all = []
    classErr_bestC_test_chance_all = []
    cbestFrs_all = []        
#    if loadWeights==1:            
#        numInh = np.full((len(days)), np.nan)
#        numAllexc = np.full((len(days)), np.nan)    
       
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
        
        
        
        #%%
        svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins) # for chAl: the latest file is with soft norm; earlier file is 
        
        svmName = svmName[0]
        print os.path.basename(svmName)


        #%% Load SVM vars
            
        Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl'])
        
        regType = Data.pop('regType').astype('str')
        cvect = Data.pop('cvect').squeeze()
    #    trsExcluded_svr = Data.pop('trsExcluded_svr').astype('bool').squeeze()                    
        perClassErrorTrain = Data.pop('perClassErrorTrain') # numSamples x len(cvect) x nFrames
        perClassErrorTest = Data.pop('perClassErrorTest')
        perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
        perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')
    #    Data = scio.loadmat(svmName, variable_names=['wAllC']) #,'bAllC'])
    #    wAllC = Data.pop('wAllC')
    #        bAllC = Data.pop('bAllC')     
        #'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'trsTrainedTestedInds':trsTrainedTestedInds, 'trsRemCorrInds':trsRemCorrInds,
    #    Data = scio.loadmat(svmName, variable_names=['trsTrainedTestedInds','trsRemCorrInds'])
    #    trsTrainedTestedInds = Data.pop('trsTrainedTestedInds')
    #    trsRemCorrInds = Data.pop('trsRemCorrInds')                                 
        numSamples = perClassErrorTest.shape[0] 
        numFrs = perClassErrorTest.shape[2]
            
            
        #%% Set class error values at best C (also find bestc for each frame and plot c path)
    
        classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs = setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, doPlots)    
    
        
        #%% Get class error values for a specific time point
        
        classErr_bestC_train_data = classErr_bestC_train_data[:,eventI_ds+time2an].squeeze() # numSamps
        classErr_bestC_test_data = classErr_bestC_test_data[:,eventI_ds+time2an].squeeze() # numSamps
        classErr_bestC_test_shfl = classErr_bestC_test_shfl[:,eventI_ds+time2an].squeeze() # numSamps
        classErr_bestC_test_chance = classErr_bestC_test_chance[:,eventI_ds+time2an].squeeze() # numSamps
    
    
        #%% Keep vars for all days
               
        # Delete vars before starting the next day           
        del perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance
            
        classErr_bestC_train_data_all.append(classErr_bestC_train_data) 
        classErr_bestC_test_data_all.append(classErr_bestC_test_data)
        classErr_bestC_test_shfl_all.append(classErr_bestC_test_shfl)
        classErr_bestC_test_chance_all.append(classErr_bestC_test_chance)
        cbestFrs_all.append(cbestFrs)    
    

    
    ######################################################################################################################################################        
    ######################################################################################################################################################        
    #%% Done with all days
    
    eventI_allDays = eventI_allDays.astype('int')
    
    classErr_bestC_train_data_all = np.array(classErr_bestC_train_data_all) # days x samps
    classErr_bestC_test_data_all = np.array(classErr_bestC_test_data_all)
    classErr_bestC_test_shfl_all = np.array(classErr_bestC_test_shfl_all)
    classErr_bestC_test_chance_all = np.array(classErr_bestC_test_chance_all)
    
    
    #%% Average and std of class accuracies across CV samples ... for each day
    
    av_train_data = 100-np.mean(classErr_bestC_train_data_all, axis=1) # average across cv samples
    sd_train_data = np.std(classErr_bestC_train_data_all, axis=1) / np.sqrt(numSamples)

    av_test_data = 100-np.mean(classErr_bestC_test_data_all, axis=1)
    sd_test_data = np.std(classErr_bestC_test_data_all, axis=1) / np.sqrt(numSamples)
        
    av_test_shfl = 100-np.mean(classErr_bestC_test_shfl_all, axis=1)
    sd_test_shfl = np.std(classErr_bestC_test_shfl_all, axis=1) / np.sqrt(numSamples)

    av_test_chance = 100-np.mean(classErr_bestC_test_chance_all, axis=1)
    sd_test_chance = np.std(classErr_bestC_test_chance_all, axis=1) / np.sqrt(numSamples)
    
    
    
    #%% Keep values of all mice (cvSample-averaged)
    
    av_train_data_allMice.append(av_train_data)
    sd_train_data_allMice.append(sd_train_data)
    
    av_test_data_allMice.append(av_test_data)
    sd_test_data_allMice.append(sd_test_data)

    av_test_shfl_allMice.append(av_test_shfl)
    sd_test_shfl_allMice.append(sd_test_shfl)

    av_test_chance_allMice.append(av_test_chance)
    sd_test_chance_allMice.append(sd_test_chance)

    
    _,pcorrtrace = stats.ttest_ind(av_test_data, av_test_shfl) # p value of class accuracy being different from 50
    print 'p value =', pcorrtrace     
    
    
    #%%
    ######################## PLOTS ########################
    
    #%% Plot class accuracy in the frame before the choice onset for each session
    
    plt.figure(figsize=(4.5,3))
    plt.errorbar(range(numDays), av_train_data, sd_train_data, label='train', color='r')
    plt.errorbar(range(numDays), av_test_data, sd_test_data, label='test', color='b')
    plt.errorbar(range(numDays), av_test_shfl, sd_test_shfl, label='shfl', color='k')
    plt.errorbar(range(numDays), av_test_chance, sd_test_chance, label='chance', color=[.6,.6,.6])
    
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
        
    

######################################################################################################################################################        
######################################################################################################################################################        
#%%
av_train_data_allMice = np.array(av_train_data_allMice) # for each mouse, size: numDays
sd_train_data_allMice = np.array(sd_train_data_allMice)

av_test_data_allMice = np.array(av_test_data_allMice)
sd_test_data_allMice = np.array(sd_test_data_allMice)

av_test_shfl_allMice = np.array(av_test_shfl_allMice)
sd_test_shfl_allMice = np.array(sd_test_shfl_allMice)

av_test_chance_allMice = np.array(av_test_chance_allMice)
sd_test_chance_allMice = np.array(sd_test_chance_allMice)
    

#%% Average classErr across sessions for each mouse

av_av_train_data_allMice = np.array([np.nanmean(av_train_data_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_data_allMice = np.array([np.nanmean(av_test_data_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_shfl_allMice = np.array([np.nanmean(av_test_shfl_allMice[im], axis=0) for im in range(len(mice))])
av_av_test_chance_allMice = np.array([np.nanmean(av_test_chance_allMice[im], axis=0) for im in range(len(mice))])

sd_av_train_data_allMice = np.array([np.nanstd(av_train_data_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_data_allMice = np.array([np.nanstd(av_test_data_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_shfl_allMice = np.array([np.nanstd(av_test_shfl_allMice[im], axis=0) for im in range(len(mice))])
sd_av_test_chance_allMice = np.array([np.nanstd(av_test_chance_allMice[im], axis=0) for im in range(len(mice))])


#%% Plot session-averaged classErr for each mouse

plt.figure(figsize=(2,3))

plt.errorbar(range(len(mice)), av_av_train_data_allMice, sd_av_train_data_allMice, fmt='o', label='train', color='r')
plt.errorbar(range(len(mice)), av_av_test_data_allMice, sd_av_test_data_allMice, fmt='o', label='test', color='b')
plt.errorbar(range(len(mice)), av_av_test_shfl_allMice, sd_av_test_shfl_allMice, fmt='o', label='shfl', color='k')
plt.errorbar(range(len(mice)), av_av_test_chance_allMice, sd_av_test_chance_allMice, fmt='o', label='chance', color=[.6,.6,.6])

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
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow0, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




