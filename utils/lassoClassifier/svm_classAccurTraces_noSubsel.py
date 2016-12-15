# -*- coding: utf-8 -*-

"""
# plot class accuracy traces (ave,sd across days) for L1, no subselect data.

# remember this is including all trials (no cross validation)... also at each time point we test the decoder (not on average ep)

Created on Tue Dec 13 15:14:54 2016
@author: farznaj
"""

# Go to svm_plots_setVars and define vars!
execfile("svm_plots_setVars.py")    

dnow = '/classAccurTraces_L1'

#%%
'''
########################################################################################################################################################       
#################### Classification accuracy at all times ##############################################################################################    
########################################################################################################################################################    
'''

#%% Loop over days

corrClass_ave_allDays = []
corrClassShfl_ave_allDays = []
eventI_allDays = np.full([numDays,1], np.nan).flatten().astype('int')

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
   
    #%% Load eventI (we need it for the final alignment of corrClass traces of all days)
    
    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
#        traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
#        time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')
    
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
#        traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
#        time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
        
    eventI_allDays[iday] = eventI        
   
        
    #%%
    svmNameAll = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName, []) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).
#    svmNameAll = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName, 0) # go w latest for 500 shuffles and bestc including at least one non0 w.

    for i in [0]: #range(len(svmNameAll)):
        svmName = svmNameAll[i]
        print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        
        if abs(w.sum()) < eps:  # I think it is wrong to do this. When ws are all 0, it means there was no decoder for that day, ie no info about the choice.
            print '\tAll weights in the allTrials-trained decoder are 0 ... '
#            
#        else:    
        ##%% Load corrClass
        Data = scio.loadmat(svmName, variable_names=['corrClass', 'corrClassShfl'])
        corrClass = Data.pop('corrClass') # frames x trials
        corrClassShfl = Data.pop('corrClassShfl') # frames x trials x numSamples
        
        # Compute average across trials (also shuffles for corrClassShfl)
        corrClass_ave = np.mean(corrClass, axis=1) # numFrames x 1
        corrClassShfl_ave = np.mean(corrClassShfl, axis=(1,2)) # numFrames x 1
    
    
    #%% Pool corrClass_ave (average of corrClass across all rounds) of all days    
    corrClass_ave_allDays.append(corrClass_ave)
    corrClassShfl_ave_allDays.append(corrClassShfl_ave)
    


#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(corrClass_ave_allDays[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

corrClass_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
corrClassShfl_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    corrClass_aligned[:, iday] = corrClass_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    corrClassShfl_aligned[:, iday] = corrClassShfl_ave_allDays[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        corrClass_aligned0 = corrClass_aligned

    elif iTiFlg==1:
        corrClass_aligned1 = corrClass_aligned
        
        
#%% Average across days

corrClass_aligned_ave = np.mean(corrClass_aligned, axis=1) * 100
corrClass_aligned_std = np.std(corrClass_aligned, axis=1) * 100

corrClassShfl_aligned_ave = np.mean(corrClassShfl_aligned, axis=1) * 100
corrClassShfl_aligned_std = np.std(corrClassShfl_aligned, axis=1) * 100

_,pcorrtrace0 = stats.ttest_1samp(corrClass_aligned.transpose(), 50) # p value of class accuracy being different from 50

_,pcorrtrace = stats.ttest_ind(corrClass_aligned.transpose(), corrClassShfl_aligned.transpose()) # p value of class accuracy being different from 50
        
       
#%% Plot the average traces across all days
#ep_ms_allDays
plt.figure(figsize=(4.5,3))

plt.fill_between(time_aligned, corrClassShfl_aligned_ave - corrClassShfl_aligned_std, corrClassShfl_aligned_ave + corrClassShfl_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, corrClassShfl_aligned_ave, 'k')

plt.fill_between(time_aligned, corrClass_aligned_ave - corrClass_aligned_std, corrClass_aligned_ave + corrClass_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, corrClass_aligned_ave, 'r')

plt.xlabel('Time since stim onset (ms)', fontsize=13)
plt.ylabel('Classification accuracy (%)', fontsize=13)
plt.legend()
ax = plt.gca()
if trialHistAnalysis:
    plt.xticks(np.arange(-400,600,200))
else:    
    plt.xticks(np.arange(0,1400,400))

makeNicePlots(ax)

# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()

pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

#plt.xticks(np.arange(-400,1400,200))
#win=[time_aligned[0], 0]
#plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
#plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])
    
# Plot lines for the training epoch
if 'ep_ms_allDays' in locals():
    win = np.mean(ep_ms_allDays, axis=0)
    plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
    plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])


##%% Save the figure    
if savefigs:
    for i in [0]:#range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'corrClassTrace'
        fign = os.path.join(svmdir+dnow, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_inches='tight')#, bbox_extra_artists=(lgd,))



    