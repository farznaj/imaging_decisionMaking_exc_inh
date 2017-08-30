# -*- coding: utf-8 -*-
"""
Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
It will also call eventTimeDist.py to plot dist of event times and compare it with class acuur traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""

#%% Change the following vars:

mousename = 'fni18' #'fni17'
ch_st_goAl = [0,1,0] # whether do analysis on traces aligned on choice, stim or go tone.
if mousename == 'fni18':
    allDays = 1 # all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
if mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   
    
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")
execfile("svm_plots_setVars_n.py")  

#chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
doPlots = 0 #1 # plot c path of each day 
savefigs = 1

#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.


#%% 
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow = '/classAccurTraces_eachFrame_timeDists/'+mousename+'/'

smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.


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
'''
#####################################################################################################################################################   
#####################################################################################################################################################
'''
            
#%% Loop over days    

eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
classErr_bestC_train_data_all = [] # np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_test_data_all = []
classErr_bestC_test_shfl_all = []
classErr_bestC_test_chance_all = []
cbestFrs_all = []

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


    #%%
    svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins) # for chAl: the latest file is with soft norm; earlier file is 

    svmName = svmName[0]
    print os.path.basename(svmName)    


    #%% Set eventI
    
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


    #%% Once done with all frames, save vars for all days
           
    # Delete vars before starting the next day           
    del perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance
        
    classErr_bestC_train_data_all.append(classErr_bestC_train_data) # each day: samps x numFrs
    classErr_bestC_test_data_all.append(classErr_bestC_test_data)
    classErr_bestC_test_shfl_all.append(classErr_bestC_test_shfl)
    classErr_bestC_test_chance_all.append(classErr_bestC_test_chance)
    cbestFrs_all.append(cbestFrs)            
            

eventI_allDays = eventI_allDays.astype('int')
cbestFrs_all = np.array(cbestFrs_all)    
#classErr_bestC_train_data_all = np.array(classErr_bestC_train_data_all)
#classErr_bestC_test_data_all = np.array(classErr_bestC_test_data_all)
#classErr_bestC_test_shfl_all = np.array(classErr_bestC_test_shfl_all)
#classErr_bestC_test_chance_all = np.array(classErr_bestC_test_chance_all)



#%%    
######################################################################################################################################################    
######################################################################################################################################################          

#%% Average and std of class accuracies across samples (all CV samples) ... for each day

av_l2_train_d = np.array([100-np.nanmean(classErr_bestC_train_data_all[iday], axis=0) for iday in range(len(days))]) # numDays; each day has size numFrs
sd_l2_train_d = np.array([np.nanstd(classErr_bestC_train_data_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_l2_test_d = np.array([100-np.nanmean(classErr_bestC_test_data_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_test_d = np.array([np.nanstd(classErr_bestC_test_data_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_l2_test_s = np.array([100-np.nanmean(classErr_bestC_test_shfl_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_test_s = np.array([np.nanstd(classErr_bestC_test_shfl_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_l2_test_c = np.array([100-np.nanmean(classErr_bestC_test_chance_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_test_c = np.array([np.nanstd(classErr_bestC_test_chance_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

#_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl, nan_policy = 'omit')
#p


#%% Keep vars for chAl and stAl

if chAl==1:
    
    eventI_allDays_ch = eventI_allDays + 0
    
    av_l2_train_d_ch = av_l2_train_d + 0
    sd_l2_train_d_ch = sd_l2_train_d + 0
    
    av_l2_test_d_ch = av_l2_test_d + 0
    sd_l2_test_d_ch = sd_l2_test_d + 0
    
    av_l2_test_s_ch = av_l2_test_s + 0
    sd_l2_test_s_ch = sd_l2_test_s + 0
    
    av_l2_test_c_ch = av_l2_test_c + 0
    sd_l2_test_c_ch = sd_l2_test_c + 0
    
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
    



#%% PLOTS
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
######################## PLOTS ##########################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Plot class accur trace for all days. Separate subplots for testing data, shuffle, chance and training data

# define a colormap
from matplotlib import cm
from numpy import linspace
start = 0.0
stop = 1.0
number_of_lines= len(days)
cm_subsection = linspace(start, stop, number_of_lines) 
colors = [ cm.jet(x) for x in cm_subsection ]


plt.figure()
for iday in range(len(days)):    
#    plt.figure()
    nPre = eventI_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
    nPost = (len(av_l2_test_d[iday]) - eventI_allDays[iday] - 1)    
    a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
    b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
    time_al = np.concatenate((a,b))
    
    plt.subplot(221)
    plt.errorbar(time_al, av_l2_test_d[iday], yerr = sd_l2_test_d[iday], label=' ')    
#    plt.title(days[iday]) # Uncomment this if you a separate plot for each day.
#    plt.plot([0,0], [50,100], color='k', linestyle=':')
    #plt.errorbar(range(len(av_l2_test_d[iday])), av_l2_test_d[iday], yerr = sd_l2_test_d[iday])
#    plt.plot(time_al, av_l2_test_d[iday]-av_l2_test_s[iday], color=colors[iday])
    
    plt.subplot(222)
    plt.errorbar(time_al, av_l2_test_s[iday], yerr = sd_l2_test_s[iday], label='')

    plt.subplot(223)
    plt.errorbar(time_al, av_l2_train_d[iday], yerr = sd_l2_train_d[iday], label='')

    plt.subplot(224)
    plt.errorbar(time_al, av_l2_test_c[iday], yerr = sd_l2_test_c[iday], label=' ')    
#    plt.show() # Uncomment this if you a separate plot for each day.
    
#plt.subplot(224)
#plt.legend(loc='center left', bbox_to_anchor=(.7, .7)) 


##%%
plt.subplot(221)
plt.plot([0,0], [40,100], color='k', linestyle=':')
plt.title('Test')
if chAl==1:
    plt.xlabel('Time since choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time since stim onset (ms)', fontsize=13)
plt.ylabel('Classification accuracy (%)', fontsize=13)
ax = plt.gca()
makeNicePlots(ax,1)

plt.subplot(222)
plt.plot([0,0], [40,100], color='k', linestyle=':')
plt.title('Test-shfl')
ax = plt.gca()
makeNicePlots(ax,1)

plt.subplot(223)
plt.plot([0,0], [40,100], color='k', linestyle=':')
plt.title('Train')
ax = plt.gca()
makeNicePlots(ax,1)

plt.subplot(224)
plt.plot([0,0], [40,100], color='k', linestyle=':')
plt.title('Test-chance')
ax = plt.gca()
makeNicePlots(ax,1)

plt.subplots_adjust(hspace=0.65)


##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_testTrain_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_testTrain_' + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
'''
# plot test minus test_shuffle
plt.figure()
for iday in range(len(days)):    
#    plt.figure()
    nPre = eventI_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
    nPost = (len(av_l2_test_d[iday]) - eventI_allDays[iday] - 1)    
    a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
    b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
    time_al = np.concatenate((a,b))
    
#    plt.subplot(221)
#    plt.errorbar(time_al, av_l2_test_d[iday], yerr = sd_l2_test_d[iday], label=' ')    
    plt.title(days[iday])
#    plt.plot([0,0], [50,100], color='k', linestyle=':')
    #plt.errorbar(range(len(av_l2_test_d[iday])), av_l2_test_d[iday], yerr = sd_l2_test_d[iday])
    plt.plot(time_al, av_l2_test_d[iday]-av_l2_test_s[iday], color=colors[iday])
#    plt.plot(time_al, av_l2_test_d[iday]-av_l2_test_s[iday], color=colors[iday])
'''    



#%%
##################################################################################################
############## Align class accur traces of all days to make a final average trace ##############
################################################################################################## 
 
 
##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (len(av_l2_test_d[iday]) - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

av_l2_train_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
av_l2_test_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
av_l2_test_s_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
av_l2_test_c_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan

for iday in range(numDays):
    av_l2_train_d_aligned[:, iday] = av_l2_train_d[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    av_l2_test_d_aligned[:, iday] = av_l2_test_d[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    av_l2_test_s_aligned[:, iday] = av_l2_test_s[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    av_l2_test_c_aligned[:, iday] = av_l2_test_c[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    
        
#%% Average and STD across days (each day includes the average class accuracy across samples.)

av_l2_train_d_aligned_ave = np.mean(av_l2_train_d_aligned, axis=1)
av_l2_train_d_aligned_std = np.std(av_l2_train_d_aligned, axis=1)

av_l2_test_d_aligned_ave = np.mean(av_l2_test_d_aligned, axis=1)
av_l2_test_d_aligned_std = np.std(av_l2_test_d_aligned, axis=1)

av_l2_test_s_aligned_ave = np.mean(av_l2_test_s_aligned, axis=1)
av_l2_test_s_aligned_std = np.std(av_l2_test_s_aligned, axis=1)

av_l2_test_c_aligned_ave = np.mean(av_l2_test_c_aligned, axis=1)
av_l2_test_c_aligned_std = np.std(av_l2_test_c_aligned, axis=1)


_,pcorrtrace0 = stats.ttest_1samp(av_l2_test_d_aligned.transpose(), 50) # p value of class accuracy being different from 50

_,pcorrtrace = stats.ttest_ind(av_l2_test_d_aligned.transpose(), av_l2_test_s_aligned.transpose()) # p value of class accuracy being different from 50
        


#%%
##################################################################################################
################ Plots of aligned traces #########################################################
##################################################################################################
        
#%% Plot the average of aligned traces across all days

plt.figure(figsize=(4.5,3))

# test_shfl
#plt.fill_between(time_aligned, av_l2_test_s_aligned_ave - av_l2_test_s_aligned_std, av_l2_test_s_aligned_ave + av_l2_test_s_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
#plt.plot(time_aligned, av_l2_test_s_aligned_ave, 'g')

# test_chance
plt.fill_between(time_aligned, av_l2_test_c_aligned_ave - av_l2_test_c_aligned_std, av_l2_test_c_aligned_ave + av_l2_test_c_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, av_l2_test_c_aligned_ave, 'k')

# data
plt.fill_between(time_aligned, av_l2_test_d_aligned_ave - av_l2_test_d_aligned_std, av_l2_test_d_aligned_ave + av_l2_test_d_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, av_l2_test_d_aligned_ave, 'r')

if chAl==1:
    plt.xlabel('Time since choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time since stim onset (ms)', fontsize=13)
plt.ylabel('Classification accuracy (%)', fontsize=13)
plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
plt.legend()
ax = plt.gca()
#if trialHistAnalysis:
#    plt.xticks(np.arange(-400,600,200))
#else:    
#    plt.xticks(np.arange(0,1400,400))
makeNicePlots(ax)

# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()

pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')


##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_aveDays_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_aveDays_' + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




#%% Plot class accur in the last time window for each day

time2an = -1; # 1 bin before eventI
if chAl==1: # chAl: frame before choice
    a = av_l2_test_d_aligned[nPreMin,:] - av_l2_test_s_aligned[nPreMin,:]
    yl = 'class accuracy (testing data - shfl) %dms before choice' %(round(regressBins*frameLength))
else: # stAl: last frame
    a = av_l2_test_d_aligned[-1,:] - av_l2_test_s_aligned[-1,:]
    yl = 'class accuracy (testing data - shfl) (last %d bin)' %(round(regressBins*frameLength))
    
plt.figure(figsize=(5.5,3))
plt.plot(range(len(days)), a, marker='o')
plt.xticks(range(len(days)), [days[i][0:6] for i in range(len(days))], rotation='vertical')
plt.ylabel(yl)

makeNicePlots(plt.gca())

##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_time' + str(time2an) + '_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_time' + str(time2an) + '_' + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




#%% Plot all days testing real data - shuffle

# change color order to jet 
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

plt.figure(figsize=(5.5,10))

plt.subplot(311)
lineObjects = plt.plot(time_aligned, av_l2_test_d_aligned - av_l2_test_s_aligned, label=days)
plt.plot([0,0], [0,50], color='k', linestyle=':')
plt.legend(iter(lineObjects), (days), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Class accuracy (testing data - shfl)', fontsize=13)
if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
makeNicePlots(plt.gca())


##%%
#plt.figure(figsize=(5.5,6))
plt.subplot(312)
lineObjects = plt.plot(time_aligned, av_l2_test_d_aligned, label=days)
plt.plot([0,0], [50,100], color='k', linestyle=':')
plt.ylabel('Class accuracy (data)', fontsize=13)
#plt.legend(iter(lineObjects), (days), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.ylabel('class accuracy (testing data - shfl)', fontsize=13)
#if chAl==1:
#    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
#else:
#    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
makeNicePlots(plt.gca())


plt.subplot(313)
lineObjects = plt.plot(time_aligned, av_l2_test_s_aligned, label=days)
plt.plot([0,0], [50,100], color='k', linestyle=':')
plt.ylabel('Class accuracy (shfl)', fontsize=13)
if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
makeNicePlots(plt.gca())

plt.subplots_adjust(wspace=1, hspace=.5)


##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_testMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_testMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



#%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% # For each day plot dist of event times as well as class accur traces. Do it for both stim-aligned and choice-aligned data 
# For this section you need to run eventTimesDist.py; also run the codes above once with chAl=0 and once with chAl=1.
no
if 'eventI_allDays_ch' in locals() and 'eventI_allDays_st' in locals():
    
    #%% Set time vars needed below and Plot time dist of trial events for all days (pooled trials)
    
    execfile("eventTimesDist.py")
    
    
    #%% Plot class accur and event time dists for each day
    
    for iday in range(len(days)):    
    
        #%% Set event time vars
    
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
        
        
        #%% stAl; Plot class accur trace and timeDists for each day
        
        plt.figure()
        
        ############### class accuracy
        colors = 'k','r','b','m'
        nPre = eventI_allDays_st[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d_st[iday]) - eventI_allDays_st[iday] - 1)
        
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
        plt.subplot(223)
        plt.errorbar(time_al, av_l2_test_d_st[iday], yerr = sd_l2_test_d_st[iday])    
    
        # mark event times relative to stim onset
        plt.plot([0, 0], [50, 100], color='g')
        plt.plot([timeStimOffset0_all_relOn, timeStimOffset0_all_relOn], [50, 100], color=colors[0])
        plt.plot([timeStimOffset_all_relOn, timeStimOffset_all_relOn], [50, 100], color=colors[1])
        plt.plot([timeCommitCL_CR_Gotone_all_relOn, timeCommitCL_CR_Gotone_all_relOn], [50, 100], color=colors[2])
        plt.plot([time1stSideTry_all_relOn, time1stSideTry_all_relOn], [50, 100], color=colors[3])
        plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('Classification accuracy (%)') #, fontsize=13)
        
        ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
        
        
        
        
        ################# Plot hist of event times
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
        bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value
        
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
    
    
    
        #%% chAl; plot class accuracy and timeDists
    
        ############### class accuracy
        colors = 'g','k','r','b'
        nPre = eventI_allDays_ch[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d_ch[iday]) - eventI_allDays_ch[iday] - 1)
        
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
        plt.subplot(224)
        plt.errorbar(time_al, av_l2_test_d_ch[iday], yerr = sd_l2_test_d_ch[iday])  
        
        # mark event times relative to choice onset
        plt.plot([timeStimOnset_all_relCh, timeStimOnset_all_relCh], [50, 100], color=colors[0])    
        plt.plot([timeStimOffset0_all_relCh, timeStimOffset0_all_relCh], [50, 100], color=colors[1])
        plt.plot([timeStimOffset_all_relCh, timeStimOffset_all_relCh], [50, 100], color=colors[2])
        plt.plot([timeCommitCL_CR_Gotone_all_relCh, timeCommitCL_CR_Gotone_all_relCh], [50, 100], color=colors[3])
        plt.plot([0, 0], [50, 100], color='m')
        plt.xlabel('Time relative to choice onset (ms)')
        ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
        
        
        
        ################## Plot hist of event times
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
        bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value
        
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
        


        ##############%% Save the figure for each day 
        if savefigs:
            dd = 'timeDists_' + days[iday]
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
        """
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
        """
    
        
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
