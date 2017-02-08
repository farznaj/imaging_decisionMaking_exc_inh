# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:36:50 2017

@author: farznaj
"""

# Go to svm_plots_setVars and define vars!
execfile("svm_plots_setVars.py")    

eps = sys.float_info.epsilon #2.2204e-16


#%%
dnow = '/stability'

if trialHistAnalysis:
    dp = '/previousChoice'
else:
    dp = '/currentChoice'


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
def setSVMname(pnevFileName, trialHistAnalysis, ntName, itiName='all'):
    import glob
    
    if trialHistAnalysis:
    #     ep_ms = np.round((ep-eventI)*frameLength)
        th_stim_dur = []
        svmn = 'svmPrevChoice_allN_allFrsTrained_' 
    else:
        svmn = 'svmCurrChoice_allN_allFrsTrained_' 
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName
    

#%% ANALYSIS

angle_choice_sh_all = []
angle_choice_all = []
angle_stim_all = []
angle_stim_sh_all = []

for iday in range(len(days)):

    print '___________________'
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    #%% Set .mat file names
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
    
    # from setImagingAnalysisNamesP import *
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))
    
    # Load inhibitRois
#    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
#    inhibitRois = Data.pop('inhibitRois')[0,:]

    
    #%% Load vars
    svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName)    
    print os.path.basename(svmName)
        
    ##%% Load vars
    Data = scio.loadmat(svmName, variable_names=['w', 'wsh', 'w_sr', 'wsh_sr'])
    w = Data.pop('w') # frames x neurons x nrandtr
    wsh = Data.pop('wsh') # frames x neurons x nrandtr x nshfl
    w_sr = Data.pop('w_sr') # frames x neurons x nrandtr
    wsh_sr = Data.pop('wsh_sr') # frames x neurons x nrandtr x nshfl
    
    ##%% Load vars (w, etc)
#    Data = scio.loadmat(svmName, variable_names=['NsExcluded'])
#    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')

    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
#    inhRois = inhibitRois[~NsExcluded]        
    
    
###############################################################################
###############################################################################                
    #%% normalize weights (ie the w vector at each frame must have length 1)
    
    # data
    normw = np.linalg.norm(w, axis=1) # frames x nrandtr; 2-norm of weights 
    w_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
    if sum(normw<=eps).sum()!=0:
        print 'take care of this; you need to reshape w_normed first'
    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
    
    # shuffle
    normwsh = np.linalg.norm(wsh, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
    wsh_normed = np.transpose(np.transpose(wsh,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
    if sum(normwsh<=eps).sum()!=0:
        print 'take care of this'
        wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
    
    
    #%% average ws across all trial subselects, then again normalize them so they have length 1 (Bagging)
    
    # data
    w_nb = np.mean(w_normed, axis=(2)) # frames x neurons 
    nw = np.linalg.norm(w_nb, axis=1) # frames; 2-norm of weights 
    w_n2b = np.transpose(np.transpose(w_nb,(1,0))/nw, (1,0)) # frames x neurons
    
    # shuffle
    wsh_nb = np.mean(wsh_normed, axis=(2)) # frames x neurons x nshfl
    nwsh = np.linalg.norm(wsh_nb, axis=1) # frames x nshfl; 2-norm of weights 
    wsh_n2b = np.transpose(np.transpose(wsh_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl
    
    
    #%% compute angle between ws at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)
    
    # some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!
    
    # we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
    angle_choice = np.arccos(abs(np.dot(w_n2b, w_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
    
    nShflSamp = w.shape[-1]
    angle_choice_sh = np.full((np.shape(w)[0],np.shape(w)[0],nShflSamp), np.nan) # frames x frames x nshfl    
    for ish in range(nShflSamp):
        angle_choice_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_n2b[:,:,ish], wsh_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl
    
    
    # plot angle between decoders at different time points
    
    # data
    plt.figure()
    plt.imshow(angle_choice, cmap='jet_r')
    plt.colorbar()
    
    # shuffle (average of angles across shuffles)
    plt.figure()
    plt.imshow(np.nanmean(angle_choice_sh, axis=2), cmap='jet_r') # average across shuffles
    plt.colorbar()
    
 
 
###############################################################################
###############################################################################
    #%% normalize weights (ie the w vector at each frame must have length 1)
    
    #eps = sys.float_info.epsilon #2.2204e-16
    # data
    normw = np.linalg.norm(w_sr, axis=1) # frames x nrandtr; 2-norm of weights 
    w_sr_normed = np.transpose(np.transpose(w_sr,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
    if sum(normw<=eps).sum()!=0:
        print 'take care of this; you need to reshape w_normed first'
    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
    
    # shuffle
    normwsh = np.linalg.norm(wsh_sr, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
    wsh_sr_normed = np.transpose(np.transpose(wsh_sr,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
    if sum(normwsh<=eps).sum()!=0:
        print 'take care of this'
        wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
    
    
    #%% average ws across all trial subselects, then again normalize them so they have length 1 (Bagging)
    
    # data
    w_sr_nb = np.mean(w_sr_normed, axis=(2)) # frames x neurons 
    nw = np.linalg.norm(w_sr_nb, axis=1) # frames; 2-norm of weights 
    w_sr_n2b = np.transpose(np.transpose(w_sr_nb,(1,0))/nw, (1,0)) # frames x neurons
    
    # shuffle
    wsh_sr_nb = np.mean(wsh_sr_normed, axis=(2)) # frames x neurons x nshfl
    nwsh = np.linalg.norm(wsh_sr_nb, axis=1) # frames x nshfl; 2-norm of weights 
    wsh_sr_n2b = np.transpose(np.transpose(wsh_sr_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl
    
    
    #%% compute angle between ws at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)
    
    # some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!
    
    # we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
    angle_stim = np.arccos(abs(np.dot(w_sr_n2b, w_sr_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
    
    angle_stim_sh = np.full((np.shape(w)[0],np.shape(w)[0],nShflSamp), np.nan) # frames x frames x nshfl
    for ish in range(nShflSamp):
        angle_stim_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_sr_n2b[:,:,ish], wsh_sr_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl
    
    
    # plot angle between decoders at different time points
    
    # data
    plt.figure()
    plt.imshow(angle_stim, cmap='jet_r')
    plt.colorbar()
    
    # shuffle (average of angles across shuffles)
    plt.figure()
    plt.imshow(np.nanmean(angle_stim_sh, axis=2), cmap='jet_r') # average across shuffles
    plt.colorbar()
    
###############################################################################
###############################################################################


    #%%
    angle_choice_all.append(angle_choice)
    angle_choice_sh_all.append(angle_choice_sh)
    angle_stim_all.append(angle_stim)
    angle_stim_sh_all.append(angle_stim_sh)




angle_choice_all = np.array(angle_choice_all)
angle_choice_sh_all = np.array(angle_choice_sh_all)
angle_stim_all = np.array(angle_stim_all)
angle_stim_sh_all = np.array(angle_stim_sh_all)


#%%
        
#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(angle_choice_all[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align angles of all days on the common eventI

r = nPreMin + nPostMin + 1
angle_choice_aligned = np.ones((r,r, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)
angle_choice_sh_aligned = np.ones((r,r,nShflSamp, numDays)) + np.nan # frames x frames x nShfl x days, aligned on common eventI (equals nPreMin)
angle_stim_aligned = np.ones((r,r, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)
angle_stim_sh_aligned = np.ones((r,r,nShflSamp, numDays)) + np.nan # frames x frames x nShfl x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
#    inds = eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1
    inds = np.arange(eventI_allDays[iday] - nPreMin,  eventI_allDays[iday] + nPostMin + 1)
    
    angle_choice_aligned[:,:,iday] = angle_choice_all[iday][inds][:,inds]
    angle_choice_sh_aligned[:,:,:,iday] = angle_choice_sh_all[iday][inds][:,inds,:]
    angle_stim_aligned[:,:,iday] = angle_stim_all[iday][inds][:,inds]
    angle_stim_sh_aligned[:,:,:,iday] = angle_stim_sh_all[iday][inds][:,inds,:]


#%% Average across days
ang_choice_av = np.mean(angle_choice_aligned, axis=-1)
ang_choice_sh_av = np.mean(angle_choice_sh_aligned, axis=-1)
ang_stim_av = np.mean(angle_stim_aligned, axis=-1)
ang_stim_sh_av = np.mean(angle_stim_sh_aligned, axis=-1)


#%% Plot averages across days

# plot angle between decoders at different time points

# choice
# data
plt.figure(figsize=(8,8))
plt.subplot(221)
plt.imshow(ang_choice_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('choice')

# shuffle (average of angles across shuffles)
#plt.figure()
plt.subplot(222)
plt.imshow(np.nanmean(ang_choice_sh_av, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
makeNicePlots(plt.gca())
plt.title('choice-shuffle')

# stimulus
# data
#plt.figure()
plt.subplot(223)
plt.imshow(ang_stim_av, cmap='jet_r')
plt.colorbar()
plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
makeNicePlots(plt.gca())
plt.title('stimulus')

# shuffle (average of angles across shuffles)
#plt.figure()
plt.subplot(224)
plt.imshow(np.nanmean(ang_stim_sh_av, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
makeNicePlots(plt.gca())
plt.title('stimulus-shuffle')

plt.subplots_adjust(hspace=.3)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp)
#    d = os.path.join(svmdir+dnow,mousename+'/bestC'+dp) #, dgc, dta)
#    d = os.path.join(svmdir+dnow,mousename+'/bestC'+dp)
#    d = os.path.join(svmdir+dnow+'/bestC',mousename+dp)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
    
    fign = os.path.join(d, suffn[0:5]+'angles.'+fmt[0])
#    fign = os.path.join(d, suffn[0:5]+'classErrVsC_'+days[iday][0:6]+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')


    
#%% Plot each day    
    
    
# plot angle between decoders at different time points

for iday in range(numDays):
    
    # choice
    # data
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.imshow(angle_choice_aligned[:,:,iday], cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    #ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
    plt.title('choice')
    
    # shuffle (average of angles across shuffles)
    #plt.figure()
    plt.subplot(222)
    plt.imshow(np.nanmean(angle_choice_sh_aligned[:,:,iday], axis=2), cmap='jet_r') # average across shuffles
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    plt.title('choice-shuffle')
    
    # stimulus
    # data
    #plt.figure()
    plt.subplot(223)
    plt.imshow(angle_stim_aligned[:,:,iday], cmap='jet_r')
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    plt.title('stimulus')
    
    # shuffle (average of angles across shuffles)
    #plt.figure()
    plt.subplot(224)
    plt.imshow(np.nanmean(angle_stim_sh_aligned[:,:,iday], axis=2), cmap='jet_r') # average across shuffles
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    plt.title('stimulus-shuffle')
    
    plt.subplots_adjust(hspace=.3)

    