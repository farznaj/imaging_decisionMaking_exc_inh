# -*- coding: utf-8 -*-
"""
Plots of outputs of file svm_eachFrame.py
Plot class accuracy for each frame .. .svm trained to decode choice on choice-aligned traces... bestc found for each frame separately.

Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""


mousename = 'fni16' #'fni17'

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("svm_plots_setVars.py")  

savefigs = 0
doPlots = 0 #1 # plot c path of each day 

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # 100ms # set to nan if you don't want to downsample.

dnow = '/classAccurTraces_eachFrame/'+mousename+'/'


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, regressBins=3):
    import glob

    if trialHistAnalysis:
        svmn = 'svmPrevChoice_eachFrame_chAl_ds%d_*' %(regressBins)
    else:
        svmn = 'svmCurrChoice_eachFrame_chAl_ds%d_*' %(regressBins)
    
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
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
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
    svmName = setSVMname(pnevFileName, trialHistAnalysis, regressBins) # latest file is with soft norm; earlier file is 

    if len(svmName)<2:
        sys.exit('<2 mat files; needs work!')            
    else:
        svmName = svmName[0]
        print os.path.basename(svmName)    



    #%% Set eventI
    
    # Load 1stSideTry-aligned traces, frames, frame of event of interest
    # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
#    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
    time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    
    print(np.shape(time_aligned_1stSide))
      
    
    #%% Downsample traces: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)
    
    if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
        print 'Downsampling traces ....'    
            
        T1 = time_aligned_1stSide.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X            
        time_aligned_1stSide = time_aligned_1stSide[0:regressBins*tt]
    #    print time_aligned_1stSide_d.shape
        time_aligned_1stSide = np.round(np.mean(np.reshape(time_aligned_1stSide, (regressBins, tt), order = 'F'), axis=0), 2)
        print time_aligned_1stSide.shape
    
        eventI_ch_ds = np.argwhere(np.sign(time_aligned_1stSide)>0)[0] # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
    
    else:
        print 'Now downsampling traces ....'
        
        eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
        eventI_ch_ds = eventI_ch
    
    eventI_allDays[iday] = eventI_ch_ds
    
    
    #%% Load SVM vars
        
    Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl'])
    
    regType = Data.pop('regType').astype('str')
    cvect = Data.pop('cvect').squeeze()
#    trsExcluded_svr = Data.pop('trsExcluded_svr').astype('bool').squeeze()                    
    perClassErrorTrain = Data.pop('perClassErrorTrain') # numSamples x len(cvect) x nFrames
    perClassErrorTest = Data.pop('perClassErrorTest')
    perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
    perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')            

#        Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
#        wAllC = Data.pop('wAllC')
#        bAllC = Data.pop('bAllC')
#     
    #'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'trsTrainedTestedInds':trsTrainedTestedInds, 'trsRemCorrInds':trsRemCorrInds,
#    Data = scio.loadmat(svmName, variable_names=['trsTrainedTestedInds','trsRemCorrInds'])
#    trsTrainedTestedInds = Data.pop('trsTrainedTestedInds')
#    trsRemCorrInds = Data.pop('trsRemCorrInds')                
                 
    numSamples = perClassErrorTest.shape[0] 
    numFrs = perClassErrorTest.shape[2]     
    

    #%% Find bestc for each frame, and plot c path

    classErr_bestC_train_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_shfl = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_chance = np.full((numSamples, numFrs), np.nan)
    cbestFrs = np.full((numFrs), np.nan)
        
    for ifr in range(numFrs):
        
        #%% Compute average of class errors across numSamples
        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
        semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        
        #%% Identify best c
        
        smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
        if smallestC==1:
            print 'bestc = smallest c whose cv error is less than 1se of min cv error'
        else:
            print 'bestc = c that gives min cv error'
        #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
        # Use all range of c... it may end up a value at which all weights are 0.
        ix = np.argmin(meanPerClassErrorTest)
        if smallestC==1:
            cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria
            cbestAll = cbest
        else:
            cbestAll = cvect[ix]
        print 'best c = ', cbestAll
        
        
        
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
        
        
        #%% Plot C path           
        
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
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
            
            plt.title('Frame %d' %(ifr))
            plt.tight_layout()
            
            
    #%% Once done with all frames, save vars for all days
    
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

#%% Average and std across shuffles ... for each day

av_l2_train_d = np.array([100-np.nanmean(classErr_bestC_train_data_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_train_d = np.array([np.nanstd(classErr_bestC_train_data_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_l2_test_d = np.array([100-np.nanmean(classErr_bestC_test_data_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_test_d = np.array([np.nanstd(classErr_bestC_test_data_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_l2_test_s = np.array([100-np.nanmean(classErr_bestC_test_shfl_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_test_s = np.array([np.nanstd(classErr_bestC_test_shfl_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

av_l2_test_c = np.array([100-np.nanmean(classErr_bestC_test_chance_all[iday], axis=0) for iday in range(len(days))]) # numDays
sd_l2_test_c = np.array([np.nanstd(classErr_bestC_test_chance_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

#_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl, nan_policy = 'omit')
#p


#%%
######################## PLOTS ########################

#%% Plot class accur for each day

for iday in range(len(days)):
    
    nPre = eventI_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
    nPost = (len(av_l2_test_d[iday]) - eventI_allDays[iday] - 1)
    
    a = -(np.asarray(frameLength) * range(nPre+1)[::-1])
    b = (np.asarray(frameLength) * range(1, nPost+1))
    time_al = np.concatenate((a,b))
    
    plt.subplot(221)
    plt.errorbar(time_al, av_l2_test_d[iday], yerr = sd_l2_test_d[iday])    
#    plt.title(days[iday])
    #plt.errorbar(range(len(av_l2_test_d[iday])), av_l2_test_d[iday], yerr = sd_l2_test_d[iday])

    plt.subplot(222)
    plt.errorbar(time_al, av_l2_test_s[iday], yerr = sd_l2_test_s[iday])

    plt.subplot(223)
    plt.errorbar(time_al, av_l2_train_d[iday], yerr = sd_l2_train_d[iday])

    plt.subplot(224)
    plt.errorbar(time_al, av_l2_test_c[iday], yerr = sd_l2_test_c[iday])


plt.subplot(221)
plt.title('test')
plt.subplot(222)
plt.title('test-shfl')
plt.subplot(223)
plt.title('train')
plt.subplot(224)
plt.title('test-chance')

plt.subplots_adjust(hspace=0.5)




#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (len(av_l2_test_d[iday]) - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
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
    

        
#%% Average across days

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
        
       
#%% Plot the average traces across all days

plt.figure(figsize=(4.5,3))

plt.fill_between(time_aligned, av_l2_test_c_aligned_ave - av_l2_test_c_aligned_std, av_l2_test_c_aligned_ave + av_l2_test_c_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, av_l2_test_c_aligned_ave, 'k')

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
        dd = 'chAl_eachFr'
    else:
        dd = 'stAl_eachFr'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)









#%%
#################################################################
#################### Testing data: corr, inorr #########################
#################################################################
#%% Plot average across samples for each day

# L2;
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1,10)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-5])
#ax = plt.subplot(121)
#plt.errorbar(range(numDays), av_l2_test_d, yerr = sd_l2_test_d, color='g', label='CV')
#plt.errorbar(range(numDays), av_l2_remCorr_d, yerr = sd_l2_remCorr_d, color='c', label='remCorr')
plt.errorbar(range(numDays), av_l2_both_d, yerr = sd_l2_both_d, color='c', label='Data')
plt.errorbar(range(numDays), av_l2_both_s, yerr = sd_l2_both_s, color='k', label='Chance')
plt.errorbar(range(numDays), av_l2_both_sh, yerr = sd_l2_both_sh, color=[.5,.5,.5], label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)   
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.45), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
ax = plt.subplot(gs[-5:-3])
x =[0,1]
labels = ['Data', 'Shfl']
plt.errorbar(x, [np.nanmean(av_l2_test_d), np.nanmean(av_l2_test_s)], yerr = [np.nanstd(av_l2_test_d), np.nanstd(av_l2_test_s)], marker='o', fmt=' ', color='k')
#x =[0,1,2]
#labels = ['Corr', 'Incorr', 'Shfl']
#plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_both_d), np.mean(np.concatenate((av_l2_test_s,av_l2_both_s)))], yerr = [np.std(av_l2_test_d), np.std(av_l2_both_d), np.std(np.concatenate((av_l2_test_s,av_l2_both_s)))], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
ax.axes.get_yaxis().set_visible(False)
#ax.yaxis.label.set_color('white')
ax.spines['left'].set_color('white')
#ax.set_axis_off()
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)
    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_eachFr'
    else:
        dd = 'stAl_eachFr'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    

    