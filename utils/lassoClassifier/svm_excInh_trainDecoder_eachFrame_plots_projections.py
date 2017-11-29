# -*- coding: utf-8 -*-
"""
Project raw neural responses onto SVM decoder and plot HR-trial and LR-trial averaged projections


Created on Mon Oct  2 19:44:08 2017

@author: farznaj
"""


#%% Change the following vars:

mousename = 'fni17' #'fni17'
lastTimeBinMissed = 0 #1# if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
#days = [151101_1] --> you picked this day!
savefigs = 0
useB = [1,1] # useB_useGuassianB if 0, no intercept will be added to the decoder projections. 
# If [1,0] b will be averaged across samples, and then added to x.w to make svm projections.
# If [1,1] b will be compouted using the threshold that separates two guassian dists (svm projs for each trial category).
# conclusions: aveDecoder is not good. gaussian is reasonable. noB is also good, except it gives mirror images for exc and inh (the same but reverse projections!)
doPlots = [1,0] # make svm projection plots for aveTrs and single trs.
corrTrained = 1
doIncorr = 0

if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
elif mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   
else:
    import numpy as np
    allDays = np.nan
    noZmotionDays = np.nan
    noZmotionDays_strict = np.nan
    noExtraStimDays = np.nan


trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 

execfile("defFuns.py")
#execfile("svm_plots_setVars_n.py")  
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)

strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found
neuronType = 2
thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.

if useB==0:
    bn = 'noB'
elif useB==[1,0]:
    bn = 'aveDecoderB'
elif useB==[1,1]:
    bn = 'gaussB'


#%% 
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/proj_eachDay/'+bn+'/'


#%% Plot ave-hr, ave-lr svm projections, also each lr and each hr svm projection

def setSVMpoj(x, w, b, useB=0):
    T, N, C = x.shape
    
    # normalize decoder of each frame to its magnitude
    w = w / sci.linalg.norm(w, axis=0);   # numNeurons x numFrames
    
    
    # now project neural responses of all trials for each frame onto the decoder of that frame    
    X_w_allFrs = [] # frs x trials (svm projections for trial, the time course.)    
    for ifr in range(T):
        if useB:
            X_w = np.dot(x[ifr,:,:].T, w[:,ifr]) + b[ifr]; # nTrials (svm projections for all trials for frame ifr)
        else:  # dont add the intercept
            X_w = np.dot(x[ifr,:,:].T, w[:,ifr])
        
        X_w_allFrs.append(X_w)
    X_w_allFrs = np.array(X_w_allFrs) # frames x trials
        
    return X_w_allFrs
        

#%%        
def plotSVMpoj(X_w_allFrs, time_trace, hr_trs, lr_trs, doPlots, lab):
    
    if doPlots[0]==1:        ##%% Plot HR, LR - averaged svm projections    
        # choice-aligned projections and raw average
        plt.figure(figsize=(3,3))
    #    plt.subplot(1,2,1)
        tr1 = np.nanmean(X_w_allFrs[:, hr_trs],  axis = 1)
        tr1_se = np.nanstd(X_w_allFrs[:, hr_trs],  axis = 1) / np.sqrt(sum(hr_trs));
        tr0 = np.nanmean(X_w_allFrs[:, lr_trs],  axis = 1)
        tr0_se = np.nanstd(X_w_allFrs[:, lr_trs],  axis = 1) / np.sqrt(sum(lr_trs));
        plt.fill_between(time_trace, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='g', facecolor='g')#, linewidth=0)
        plt.fill_between(time_trace, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor=[1,0,1], facecolor=[1,0,1])#, linewidth=0)
        plt.plot(time_trace, tr1, 'g', label = 'high-rate choice')#, linestyle='-')
        plt.plot(time_trace, tr0, color=[1,0,1], label = 'low-rate choice')#, linestyle='--')
        plt.xlabel('Time relative to choice (ms)')
        plt.ylabel('SVM projections (a.u.)')
        yl = plt.gca().get_ylim()
        plt.vlines(0, yl[0], yl[1], color='k', linestyle=':')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
        makeNicePlots(plt.gca(), 1,1)    
    
        if savefigs:#% Save the figure
            if chAl==1:
                dd = days[iday][0:6] + '_' + lab + '_' + suffn[0:5] + 'chAl_aveHR_LR'
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
        
    if doPlots[1]==1:        #### Plot individual trials    
        ############# all HR trials
        hri = np.argwhere(hr_trs)
        r = sum(hr_trs)    
        plt.figure(figsize=(1.5, r))
        
        for isp in range(sum(hr_trs)):
            trac = X_w_allFrs[:, hri[isp]]
            plt.subplot(r,1,isp+1)
            plt.plot(time_trace, trac, color='b')
            plt.vlines(0, min(trac), max(trac))
            plt.hlines(0, min(time_trace), max(time_trace))
    #        plt.axis('off')
            makeNicePlots(plt.gca(),1,1)
            plt.gca().xaxis.set_visible(False) 
            plt.ylabel('HR %d' %(hri[isp]))
    
        if savefigs:#% Save the figure
            if chAl==1:
                dd = days[iday][0:6] + '_' + lab + '_' + suffn[0:5] + 'chAl_eachHR'
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            
        ############ all LR trials
        lri = np.argwhere(lr_trs)
        r = sum(lr_trs)    
        plt.figure(figsize=(1.5, r))
        
        for isp in range(sum(lr_trs)):
            trac = X_w_allFrs[:, lri[isp]]
            plt.subplot(r,1,isp+1)
            plt.plot(time_trace, trac, color='r')
            plt.vlines(0, min(trac), max(trac))
            plt.hlines(0, min(time_trace), max(time_trace))
    #        plt.axis('off')
            makeNicePlots(plt.gca(),1,1)
            plt.gca().xaxis.set_visible(False) 
            plt.ylabel('LR %d' %(lri[isp]))
    
    
        if savefigs:#% Save the figure
            if chAl==1:
                dd = days[iday][0:6] + '_' + lab + '_' + suffn[0:5] + 'chAl_eachLR'
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            
            
#%% Loop over days

X_w_inh_allD = []   
X_w_exc_allD = []

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


    ###########################################################################################################################################
    #%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
    #     - traces are set in set_aligned_traces.m matlab script.
    
    ################# Load outcomes and choice (allResp_HR_LR) for the current trial
    # if trialHistAnalysis==0:
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    # choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    #print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))
    
    
    if trialHistAnalysis:
        # Load trialHistory structure to get choice vector of the previous trial
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
    
    
        
    ################## Set trials strength and identify trials with stim strength of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
        stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
        cb = np.array(Data.pop('cb')).flatten().astype('float')
    
        s = stimrate-cb; # how far is the stimulus rate from the category boundary?
        if strength2ana == 'easy':
            str2ana = (abs(s) >= (max(abs(s)) - thStimStrength));
        elif strength2ana == 'hard':
            str2ana = (abs(s) <= thStimStrength);
        elif strength2ana == 'medium':
            str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength))); 
        else:
            str2ana = np.full((1, np.shape(outcomes)[0]), True, dtype=bool).flatten();
    
        print 'Number of trials with stim strength of interest = %i' %(str2ana.sum())
        print 'Stim rates for training = {}'.format(np.unique(stimrate[str2ana]))
    
        '''
        # Set to nan those trials in outcomes and allRes that are nan in traces_al_stim
        I = (np.argwhere((~np.isnan(traces_al_stim).sum(axis=0)).sum(axis=1)))[0][0] # first non-nan neuron
        allTrs2rmv = np.argwhere(sum(np.isnan(traces_al_stim[:,I,:])))
        print(np.shape(allTrs2rmv))
    
        outcomes[allTrs2rmv] = np.nan
        allResp_HR_LR[allTrs2rmv] = np.nan
        '''
    
    
    
    #%% Load inhibitRois
    
    Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
    inhibitRois = Data.pop('inhibitRois_pix')[0,:]    
    print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
    
        
    ###########################################################################################################################################
    #%% Set choiceVec0  (Y: the response vector)
        
    if trialHistAnalysis:
        choiceVec0 = choiceVec0All[:,iTiFlg] # choice on the previous trial for short (or long or all) ITIs
        choiceVec0S = choiceVec0All[:,0]
        choiceVec0L = choiceVec0All[:,1]
    else: # set choice for the current trial
        choiceVec0 = allResp_HR_LR;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        # choiceVec0 = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        if outcome2ana == 'corr':
            choiceVec0[outcomes!=1] = np.nan; # analyze only correct trials.
        elif outcome2ana == 'incorr':
            choiceVec0[outcomes!=0] = np.nan; # analyze only incorrect trials.   
        
        choiceVec0[~str2ana] = np.nan   
        # Y = choiceVec0
        # print(choiceVec0.shape)
    print '%d correct trials; %d incorrect trials' %((outcomes==1).sum(), (outcomes==0).sum())
    
    
    
    #%% Load spikes and time traces to set X for training SVM
    
    if chAl==1:    #%% Use choice-aligned traces 
        # Load 1stSideTry-aligned traces, frames, frame of event of interest
        # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
        Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
        traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
        time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
        eventI = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
        #eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
        # print(np.shape(traces_al_1stSide))
        
        trsExcluded = (np.isnan(np.sum(traces_al_1stSide, axis=(0,1))) + np.isnan(choiceVec0)) != 0
        
        X_svm = traces_al_1stSide[:,:,~trsExcluded]  
    #    Y_svm = choiceVec0[~trsExcluded];
    #    print 'frs x units x trials', X_svm.shape
    #    print Y_svm.shape
        
        time_trace = time_aligned_1stSide
    
    
    print 'frs x units x trials', X_svm.shape    
    
    
    
    #%% Set Y for training SVM   
    
    Y_svm = choiceVec0[~trsExcluded]
    #len(Y_svm)
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y_svm==1)
    lr_trs = (Y_svm==0)
        
    print '%d HR trials; %d LR trials' %(sum(hr_trs), sum(lr_trs))
    

    #%% I think we need at the very least 3 trials of each class to train SVM. So exit the analysis if this condition is not met!
    
    if min((Y_svm==1).sum(), (Y_svm==0).sum()) < 3:
        sys.exit('Too few trials to do SVM training! HR=%d, LR=%d' %((Y_svm==1).sum(), (Y_svm==0).sum()))
        
        
    #%% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)
    
    X_svm, time_trace, eventI_ds = downsampXsvmTime(X_svm, time_trace, eventI, regressBins, lastTimeBinMissed)    
    
    
    #%% Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
    # thAct and thTrsWithSpike are parameters that you can play with.
    
    if softNorm==0:    
        if trialHistAnalysis==0:
            # X_svm
            T1, N1, C1 = X_svm.shape
            X_svm_N = np.reshape(X_svm.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
            
            # Define NsExcluded as neurons with low stdX
            stdX_svm = np.std(X_svm_N, axis = 0);
            NsExcluded = stdX_svm < thAct
            # np.sum(stdX < thAct)
            
        print '%d = Final # non-active neurons' %(sum(NsExcluded))
        # a = size(spikeAveEp0,2) - sum(NsExcluded);
        print 'Using %d out of %d neurons; Fraction excluded = %.2f\n' %(np.shape(X_svm)[1]-sum(NsExcluded), np.shape(X_svm)[1], sum(NsExcluded)/float(np.shape(X_svm)[1]))
        
        """
        print '%i, %i, %i: #original inh, excit, unsure' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
        # Check what fraction of inhibitRois are excluded, compare with excitatory neurons.
        if neuronType==2:    
            print '%i, %i, %i: #excluded inh, excit, unsure' %(np.sum(inhibitRois[NsExcluded]==1), np.sum(inhibitRois[NsExcluded]==0), np.sum(np.isnan(inhibitRois[NsExcluded])))
            print '%.2f, %.2f, %.2f: fraction excluded inh, excit, unsure\n' %(np.sum(inhibitRois[NsExcluded]==1)/float(np.sum(inhibitRois==1)), np.sum(inhibitRois[NsExcluded]==0)/float(np.sum(inhibitRois==0)), np.sum(np.isnan(inhibitRois[NsExcluded]))/float(np.sum(np.isnan(inhibitRois))))
        """
        ##%% Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)
        
        X_svm = X_svm[:,~NsExcluded,:]
        print 'choice-aligned traces: ', np.shape(X_svm)
        
    else:
        print 'Using soft normalization; so all neurons will be included!'
        NsExcluded = np.zeros(np.shape(X_svm)[1]).astype('bool')  
    
    
    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
    if neuronType==2:
        inhRois = inhibitRois[~NsExcluded]
    #    print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
    #    print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)
    
        
    #%% 
    
#    numDataPoints = X_svm.shape[2] 
#    print '# data points = %d' %numDataPoints
    # numTrials = (~trsExcluded).sum()
    # numNeurons = (~NsExcluded).sum()
    numTrials, numNeurons = X_svm.shape[2], X_svm.shape[1]
    print '%d trials; %d neurons' %(numTrials, numNeurons)
    # print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape    
        
    
    #%% Keep a copy of X_svm before normalization
    
    X_svm00 = X_svm + 0
    
    
    #%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 
    
    # Normalize all frames to the same value (problem: some frames FRs may span a much larger range than other frames, hence svm solution will become slow)
    """
    # X_svm
    T1, N1, C1 = X_svm.shape
    X_svm_N = np.reshape(X_svm.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
        
    meanX_svm = np.mean(X_svm_N, axis = 0);
    stdX_svm = np.std(X_svm_N, axis = 0);
    
    # normalize X
    X_svm_N = (X_svm_N - meanX_svm) / stdX_svm;
    X_svm = np.reshape(X_svm_N, (T1, C1, N1), order = 'F').transpose(0 ,2 ,1)
    """
    
    # Normalize each frame separately (do soft normalization)
    X_svm_N = np.full(np.shape(X_svm), np.nan)
    meanX_fr = []
    stdX_fr = []
    for ifr in range(np.shape(X_svm)[0]):
        m = np.mean(X_svm[ifr,:,:], axis=1)
        s = np.std(X_svm[ifr,:,:], axis=1)   
        meanX_fr.append(m) # frs x neurons
        stdX_fr.append(s)       
        
        if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
            s = s+thAct     
    
        X_svm_N[ifr,:,:] = ((X_svm[ifr,:,:].T - m) / s).T
    
    meanX_fr = np.array(meanX_fr) # frames x neurons
    stdX_fr = np.array(stdX_fr) # frames x neurons
    
    
    X_svm = X_svm_N
    
    
    #%% Look at min and max of each neuron at each frame ... this is to decide for normalization
    """
    mn = []
    mx = []
    for t in range(X_svm.shape[0]):
        x = X_svm[t,:,:].squeeze().T # trails x units
        mn.append(np.min(x,axis=0))
        mx.append(np.max(x,axis=0))
    
    mn = np.array(mn)
    mx = np.array(mx)
    
    
    plt.figure(figsize=(4,4))
    #fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.imshow(mn, cmap='jet', interpolation='nearest', aspect='auto') # frames x units
    plt.colorbar()
    #im = axes[0].imshow(mn, cmap='jet', aspect='auto') # frames x units
    #clim=im.properties()['clim']
    #clim=im.properties()['clim']
    #axes[1].imshow(my_image2, clim=clim)
    #plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    
    plt.figure(figsize=(4,4))
    plt.imshow(mx, cmap='jet', interpolation='nearest', aspect='auto') # frames x units
    plt.colorbar()
    """
    
    
    #%% Plot
    '''
    if doPlots:
        plt.figure
        plt.subplot(2,2,1)
    #    plt.plot(np.mean(meanX_fr,axis=0)) # average across frames for each neuron
        plt.hist(meanX_fr.reshape(-1,))
        plt.xlabel('meanX \n(mean of trials)')
        plt.ylabel('frames x neurons')
        plt.title('min = %.6f' %(np.min(np.mean(meanX_fr,axis=0))))
    
        plt.subplot(2,2,3)
    #    plt.plot(np.mean(stdX_fr,axis=0))
        plt.hist(stdX_fr.reshape(-1,))
        plt.xlabel('stdX \n(std of trials)')
        plt.ylabel('frames x neurons')
        plt.title('min = %.6f' %(np.min(np.mean(stdX_fr,axis=0))))
    
        plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
        # plt.subplots_adjust(hspace=.5)
    
    
        
    # choice-aligned: classes: choices
    # Plot stim-aligned averages after centering and normalization
    if doPlots:
    #    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    #    hr_trs = (Y_svm==1)
    #    lr_trs = (Y_svm==0)
    
        plt.figure()
        plt.subplot(1,2,1)
        a1 = np.nanmean(X_svm[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
        tr1 = np.nanmean(a1,  axis = 1)
        tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
        a0 = np.nanmean(X_svm[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
        tr0 = np.nanmean(a0,  axis = 1)
        tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials);
        mn = np.concatenate([tr1,tr0]).min()
        mx = np.concatenate([tr1,tr0]).max()
    #    plt.plot([win[0], win[0]], [mn, mx], 'g-.') # mark the begining and end of training window
    #    plt.plot([win[-1], win[-1]], [mn, mx], 'g-.')
        plt.fill_between(time_trace, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
        plt.fill_between(time_trace, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_trace, tr1, 'b', label = 'high rate')
        plt.plot(time_trace, tr0, 'r', label = 'low rate')
        # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
        # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    #    plt.xlabel('time aligned to stimulus onset (ms)')
        plt.title('Population average - raw')
        plt.legend()
    '''
    
    
    #%% Load weights from the svm file
    
    for idi in [0,2]:#range(3):
    
        doInhAllexcEqexc = np.full((3), False)
        doInhAllexcEqexc[idi] = True 
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins)        
        svmName = svmName[0] # which svm file to use in case the same analysis was run multiple times # [0] will be the latest file
        print os.path.basename(svmName)    
    
        if doInhAllexcEqexc[0] == 1:
            if useB:
                Datai = scio.loadmat(svmName, variable_names=['w_data_inh', 'b_data_inh'])
            else:
                Datai = scio.loadmat(svmName, variable_names=['w_data_inh'])
                
    #    elif doInhAllexcEqexc[1] == 1:
    #        Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc'])
        elif doInhAllexcEqexc[2] == 1:
            if useB:
                Datae = scio.loadmat(svmName, variable_names=['w_data_exc', 'b_data_exc', 'excNsEachSamp'])                                                 
            else:
                Datae = scio.loadmat(svmName, variable_names=['w_data_exc', 'excNsEachSamp'])
            
            
        
    ###%%             
    #    perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh')
    #    perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh')
    #    perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh') 
    w_data_inh = Datai.pop('w_data_inh')  # numSamples x numNeurons x numFrames
    if useB:
        b_data_inh = Datai.pop('b_data_inh')
    
    #    perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc')
    #    perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc')
    #    perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc')   
    #    w_data_allExc = Dataae.pop('w_data_allExc') 
    
    #    perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_data_exc')    
    #    perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_exc')
    #    perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_exc')
    w_data_exc = Datae.pop('w_data_exc')  # numShufflesExc x numSamples x numNeurons x numFrames
    excNsEachSamp = Datae.pop('excNsEachSamp')
    numShufflesExc = w_data_exc.shape[0]
    if useB:
        b_data_exc = Datae.pop('b_data_exc')        

    
    ## Sanity check
    if sum(inhRois==1) != w_data_inh.shape[1]:
        sys.exit('svmFile outputs do not match postFile outputs')
        
    if X_svm.shape[0] != w_data_inh.shape[2]:
        sys.exit('svmFile outputs do not match postFile outputs')
        

    #%% Load weights of all neurons

    svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained) # for chAl: the latest file is with soft norm; earlier file is 
    svmName = svmName[0]
    print os.path.basename(svmName)    

    _, _, _, _, _, w_data_allN, b_data_allN = loadSVM_allN(svmName, 0, doIncorr, 2) # numSamples x numNeurons x numFrames

        
    #%% Normalize weights for each samp
    
    normw = sci.linalg.norm(w_data_inh, axis=1);   # numSamps x numFrames
    w_data_inh_normed = np.transpose(np.transpose(w_data_inh,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1

    normw = sci.linalg.norm(w_data_exc, axis=2);   # numShufflesExc x numSamps x numFrames
    w_data_exc_normed = np.transpose(np.transpose(w_data_exc,(2,0,1,3))/normw, (1,2,0,3)) # numShufflesExc x numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
    
    normw = sci.linalg.norm(w_data_allN, axis=1);   # numSamps x numFrames
    w_data_allN_normed = np.transpose(np.transpose(w_data_allN,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
    
    
    #%% Set SVM projections

    ######################## All Neurons ########################
    
    ##%% Average normalized decoders of all samples (ie trial resamples)   
    # we will once more normalize wInh and wExc in setSVMpoj
       
    wAllN = np.mean(w_data_allN_normed, axis=0)  # numNeurons x numFrames
    if useB:
        bAllN = np.mean(b_data_allN, axis=0)  # numFrames
    else:
        bAllN = np.nan
    

    XallN = X_svm


    

    #%% #%% Set and plot ave-hr, ave-lr svm projections, also each lr and each hr svm projection     

    # Set the intercept for all samp decoders ... (matt's way in NN2014) classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...    
    
    if useB == [1,1]:
        
        # get the projections without the intercept
        X_w_allN = setSVMpoj(XallN, wAllN, bAllN, 0) # frames x trials 
        
        hrp = X_w_allN[:,hr_trs]    
        lrp = X_w_allN[:,lr_trs]    
        mu0 = np.mean(hrp, axis=1)
        mu1 = np.mean(lrp, axis=1)
        sd0 = np.std(hrp, axis=1)
        sd1 = np.std(lrp, axis=1)
        thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
        b_gauss = thresh
    
        X_w_allN = setSVMpoj(XallN, wAllN, b_gauss, useB) # frames x trials 
        
    else:    # no intercept, or ave-decoder intercept
    
        X_w_allN = setSVMpoj(XallN, wAllN, bAllN, useB) # frames x trials   
            


    #%% Plot SVM proj
    
    plotSVMpoj(X_w_allN, time_trace, hr_trs, lr_trs, doPlots, lab='allN') # frames x trials   

    # plot svm proj for one of the samps
#    X_w_inh0 = setSVMpoj(Xinh, w_data_inh_normed[20,:,:], b_data_inh[20], hr_trs, lr_trs, useB) # frames x trials   
#    plotSVMpoj(X_w_inh0, time_trace, hr_trs, lr_trs, doPlots, lab='inh') # frames x trials   
    

    #%%######################## INH ########################
    
    ##%% Average normalized decoders of all samples (ie trial resamples)   
    # we will once more normalize wInh and wExc in setSVMpoj
       
    wInh = np.mean(w_data_inh_normed, axis=0)  # numNeurons x numFrames
    if useB:
        bInh = np.mean(b_data_inh, axis=0)  # numFrames
    else:
        bInh = np.nan
    

    Xinh = X_svm[:, inhRois==1,:]       


    

    #%% #%% Set and plot ave-hr, ave-lr svm projections, also each lr and each hr svm projection     

    # Set the intercept for all samp decoders ... (matt's way in NN2014) classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...    
    
    if useB == [1,1]:
        
        # get the projections without the intercept
        X_w_inh = setSVMpoj(Xinh, wInh, bInh, 0) # frames x trials 
        
        hrp = X_w_inh[:,hr_trs]    
        lrp = X_w_inh[:,lr_trs]    
        mu0 = np.mean(hrp, axis=1)
        mu1 = np.mean(lrp, axis=1)
        sd0 = np.std(hrp, axis=1)
        sd1 = np.std(lrp, axis=1)
        thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
        b_gauss = thresh
    
        X_w_inh = setSVMpoj(Xinh, wInh, b_gauss, useB) # frames x trials 
        
    else:    # no intercept, or ave-decoder intercept
    
        X_w_inh = setSVMpoj(Xinh, wInh, bInh, useB) # frames x trials   
            


    #%% Plot SVM proj
    
    plotSVMpoj(X_w_inh, time_trace, hr_trs, lr_trs, doPlots, lab='inh') # frames x trials   

    # plot svm proj for one of the samps
#    X_w_inh0 = setSVMpoj(Xinh, w_data_inh_normed[20,:,:], b_data_inh[20], hr_trs, lr_trs, useB) # frames x trials   
#    plotSVMpoj(X_w_inh0, time_trace, hr_trs, lr_trs, doPlots, lab='inh') # frames x trials   
    
    
    #%%######################## EXC ########################

    ##%% Average normalized decoders of all samples (ie trial resamples)   
    # we will once more normalize wInh and wExc in setSVMpoj
        
    wExc = np.mean(w_data_exc_normed, axis=1)  # numShufflesExc x numNeurons x numFrames
    if useB:
        bExc = np.mean(b_data_exc, axis=1)  # numShufflesExc x numFrames
    else:
        bExc = np.full((numShufflesExc), np.nan)
    # exc: average across all exc and trial samps
#    wExc = np.mean(w_data_exc_normed, axis=(0,1))  # numNeurons x numFrames
#    bExc = np.mean(b_data_exc, axis=(0,1))  # numFrames    

    #### Set Xexc of each excShfl
    lenInh = (inhRois==1).sum()
    excI = np.argwhere(inhRois==0)        
    XexcEq = []
    #excNsEachSamp = []
    for ii in range(numShufflesExc): 
        en = excNsEachSamp[ii].squeeze() # n randomly selected exc neurons.    
        Xexc = X_svm[:, en,:]
        XexcEq.append(Xexc)    
    #    excNsEachSamp.append(en) # indeces of exc neurons (our of X_svm) used for svm training in each exc shfl (below).... you need this if you want to get svm projections for a particular exc shfl (eg w_data_exc[nExcShfl,:,:,:])
    XexcEq = np.array(XexcEq) # nExcShfl x frames x units x trials
    
    # since in svmfile you haven't saved the indeces of exc neurons in each exc shfl, we cannot project each excShfl decoder onto its on neural responses.... 
    # to get around this issue, I'm taking average of decoders across all excShfl, also average of XexcEq across all excShfl.   
#    XexcEq_aveExcShfl = np.mean(XexcEq, axis=0) # frames x units x trials
    
        
    #%% Set svm projections for each excShfl, then compute average across excShfls
        
    X_w_exc = [] # Set projections for each excShfl
    for ii in range(numShufflesExc): 

        if useB == [1,1]:
            
            # get the projections without the intercept
            X_w_exc_eachExcShfl = setSVMpoj(XexcEq[ii], wExc[ii], bExc[ii], 0) # frames x trials 
            
            hrp = X_w_exc_eachExcShfl[:,hr_trs]    
            lrp = X_w_exc_eachExcShfl[:,lr_trs]    
            mu0 = np.mean(hrp, axis=1)
            mu1 = np.mean(lrp, axis=1)
            sd0 = np.std(hrp, axis=1)
            sd1 = np.std(lrp, axis=1)
            thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
            b_gauss = thresh
        
            X_w_exc_eachExcShfl = setSVMpoj(XexcEq[ii], wExc[ii], b_gauss, useB) # frames x trials 
            
        else:    # no intercept, or ave-decoder intercept
    
            X_w_exc_eachExcShfl = setSVMpoj(XexcEq[ii], wExc[ii], bExc[ii], useB)

        X_w_exc.append(X_w_exc_eachExcShfl)
    X_w_exc = np.array(X_w_exc) # nExcShfl x frs x trs
    
    
    # average svm projs of all excShfl
    X_w_exc_ave = np.mean(X_w_exc, axis=0) # frs x trials

    
    #%% Plot ave-hr, ave-lr svm projections, also each lr and each hr svm projection    
    # averaged across excShfl
    
    plotSVMpoj(X_w_exc_ave, time_trace, hr_trs, lr_trs, doPlots, lab='aveExc')    

 
    #%% Plot individual exc shfls (instead of their ave)

    # use below to find good shuffles
    '''
    for ii in range(numShufflesExc): 
    
        plotSVMpoj(X_w_exc[ii], time_trace, hr_trs, lr_trs, doPlots, lab='exc_shfl %d' %(ii))
        plt.title('exc shfl %d' %(ii))
        
    '''
    # then plot and save them here!                
    
    ii = 41  #     ii = rng.permutation(numShufflesExc)[0]    
    plotSVMpoj(X_w_exc[ii], time_trace, hr_trs, lr_trs, doPlots, lab='exc_shfl %d' %(ii))
    plt.title('exc shfl %d' %(ii))   
    

    
    
    #%%
    X_w_inh_allD.append(X_w_inh)    
    X_w_exc_allD.append(X_w_exc)    
    
    
#X_w_inh_allD = np.array(X_w_inh_allD)    
#X_w_exc_allD = np.array(X_w_exc_allD)        
    


    