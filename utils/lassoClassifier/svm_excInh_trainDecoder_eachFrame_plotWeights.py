# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:26:41 2017

@author: farznaj
"""

# -*- coding: utf-8 -*-
"""
Plots histograms and time course (during the trial) of weights for exc, inh (svm trained on non-overlapping time windows)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""

#%%
mice = 'fni16', 'fni17', 'fni18', 'fni19'

savefigs = 0
useAllNdecoder = 1 # if 1: use the decoder that was trained on all neurons; # if 0: Use the decoder that was trained only on inh or only exc.
normWeights = 0 # if 1, weights will be normalized to unity length.

doPlots = 1 #1 # plot hists of w per mouse
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
corrTrained = 1
doIncorr = 0
time2an = -1 # relative to eventI, look at classErr in what time stamp.
lastTimeBinMissed = 1 # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
#poolDaysOfMice = 0 # for plotting w traces; if 0, average and sd across mice (each mouse has session-averaged weights); if 1, average and sd across all sessions of all mice

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
    
smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
    
#dnow0 = '/excInh_trainDecoder_eachFrame_weights/frame'+str(time2an)+'/'
dnow0 = '/excInh_trainDecoder_eachFrame_weights/'
       
execfile("defFuns.py")  

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')
#dp = 'time' + str(time2an) + '_' + nowStr    
dp = 'time' + str(time2an)    

if useAllNdecoder:
    dect = 'allN_decoder'
else:
    dect = 'excInh_decoder'

if normWeights:
    nl = 'normedW'
else:
    nl = 'notNormedW'

if chAl==1:
    cha = 'chAl_'
else:
    cha = 'stAl_'   
    
colors = ['b','r']  #        colors = ['k','r']                   


#%%  Define function to plot exc, inh distribution of weights, all sessions pooled on 1 subplot, and individual sessions on a different subplot.
    
def plotHistErrBarWsEI(we,wi, lab, ylab, doIndivDays=0, fig=np.nan,h1=[],h2=[]):
    # we and wi have numDays element
    lab1 = 'exc'
    lab2 = 'inh'
#    colors = ['b','r']  #        colors = ['k','r']               
    
    #### set vars: hist and P val for all days pooled (exc vs inh)
#        lab = 'w'
#        ylab = 'Fract neurons (all days)'
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    
    a = np.concatenate((we)) #np.reshape(wexc,(-1,)) 
    b = np.concatenate((wi)) #np.reshape(winh,(-1,))
    
    print np.sum(abs(a)<=eps), ' exc ws < eps'
    print np.sum(abs(b)<=eps), ' inh ws < eps'
    print [len(a), len(b)]
    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    
    
    ################%% Plot hist
    if np.isnan(fig):            
        plt.figure(figsize=(5,5))    
        gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
        h1 = gs[0,0:2]
        h2 = gs[0,2:3]
    ax1,_ = histerrbar(h1,h2,a,b,binEvery,p,lab,colors,ylab,lab1,lab2)
    #plt.xlabel(lab)
#        ax1.set_xlim([-.05, .05])
    yl = ax1.get_ylim()
    ax1.set_ylim([yl[0]-np.diff(yl)/20, yl[1]])
    xl = ax1.get_xlim()
    ax1.set_xlim([xl[0]-np.diff(xl)/20, xl[1]])
    makeNicePlots(ax1,1,1)
    
    
    ################%% plot error bars for individual days
    if doIndivDays:
        a = we
        b = wi
        # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
        p = []
        for i in range(len(a)):
            _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
            p.append(p0)
        p = np.array(p)    

        ax2,_ = errbarAllDays(a,b,p,gs,colors,lab1,lab2)
        
        makeNicePlots(ax2,1,1)
        
        
        if savefigs:#% Save the figure
        
            d = os.path.join(svmdir+dnow)
            
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
            
            fign = os.path.join(d, suffn[0:5]+cha+'dist_'+dp+'_'+lab+'_excVSinh_'+dpm+'.'+fmt[0])

            plt.savefig(fign, bbox_inches='tight')
        
        return ax1,ax2
    else:
        return ax1
            
            
#%% 
'''
#####################################################################################################################################################   
#####################################################################################################################################################
'''

#%%

winhAve_all_allMice = []
wexcAve_all_allMice = []
winhAve_all2_allMice = []
wexcAve_all2_allMice = []
cbestFrs_all_allMice = []
numDaysAll = np.full(len(mice), np.nan)
numDaysAll_good = np.full(len(mice), np.nan)
normw_allN_allD_allMice = []
normw_inh_allD_allMice = []
normw_exc_allD_allMice = []
dpmAllm = []
eventI_allDays_allMice = []
eventI_ds_allDays_allMice = []
corr_hr_lr_allMice = []

#%%
for im in range(len(mice)):
        
    #%%            
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

#    execfile("svm_plots_setVars_n.py")      
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
    numDaysAll[im] = len(days)        

#    dnow = '/excInh_trainDecoder_eachFrame_weights/'+mousename+'/'+dect
#    dnow = os.path.join('/excInh_trainDecoder_eachFrame_weights',mousename,dect,nl)            
    dnow = os.path.join(dnow0,mousename,dect,nl)   
    dpm = 'days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr   
    dpmAllm.append(dpm)
            
            
    #%% 
    cbestFrs_all = []
    winhAve_all = []
    wexcAve_all = []
    eventI_allDays = np.full((len(days)), np.nan)  # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    eventI_ds_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
    normw_allN_allD = []
    normw_inh_allD = [] # 2-norm of decoders, for each day: samps x frames
    normw_exc_allD = []    
    
    
    #%% Loop over days   
    
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
        svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained) # for chAl: the latest file is with soft norm; earlier file is 
        svmName = svmName[0]
        print os.path.basename(svmName)    
        

        #%% Set number of hr, lr trials that were used for svm training
        
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
                

        #%% Set eventI_ds (downsampled eventI)
    
        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds    

        
        #%% Load SVM vars            
        
        if useAllNdecoder: # use the decoder that was trained using all neurons       
            
            #%% Load inhibitRois            
            Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
            inhRois = Data.pop('inhibitRois_pix')[0,:]    
    #        print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
            
            
            ######                        
            Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTest']) #,'perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl'])
            
            regType = Data.pop('regType').astype('str')
            cvect = Data.pop('cvect').squeeze()
            perClassErrorTest = Data.pop('perClassErrorTest')
            Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
            wAllC = Data.pop('wAllC') # numSamples x len(cvect) x nNeurons x nFrames
            bAllC = Data.pop('bAllC') # numSamples x len(cvect) x nFrames                      
                
            ####%% Find bestc for each frame (funcion is defined in defFuns.py)            
            cbestFrs = findBestC(perClassErrorTest, cvect, regType, smallestC) # nFrames                	
        	
            #####%% Set class error values at best C (if desired plot c path) (funcion is defined in defFuns.py)    
            doPlotscpath = 0;
            _, _, _, _, w_bestc_data, b_bestc_data = setClassErrAtBestc(cbestFrs, cvect, doPlotscpath, np.nan, np.nan, np.nan, np.nan, wAllC, bAllC)
            del perClassErrorTest, wAllC, bAllC    
        
            
            if normWeights: ##%% normalize weights of each sample            
                normw = sci.linalg.norm(w_bestc_data, axis=1);   # numSamps x numFrames #2-norm of each decoder (particular sample and frame)
                w_bestc_data = np.transpose(np.transpose(w_bestc_data,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
                normw_allN_allD.append(normw)
                
            # average decoders across samps
            w_bestc_dataAve = np.mean(w_bestc_data, axis=0) # nNeurons x nFrames
            
            if normWeights: # normalize averaged decoder
                normw = sci.linalg.norm(w_bestc_dataAve, axis=0)
                w_bestc_dataAve = w_bestc_dataAve / normw
            
            # Now separate decoder weights into exc and inh weights
            winhAve = w_bestc_dataAve[inhRois==1,:] # nNeurons x nFrames
            wexcAve = w_bestc_dataAve[inhRois==0,:]
            ##%% Set w for inh and exc                   
#            winh0 = w_bestc_data[:,inhRois==1,:] # numSamples x nNeurons x nFrames
#            wexc0 = w_bestc_data[:,inhRois==0,:]

        else:  
            #%% Use the other svm file in which SVM was trained using only inh or exc.
            _,_,_,_,_,_,_,_,_, winh0, wexc0, w_data_exc, _, _, _, svmName_excInh, svmName_allN = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, 0, 1, 0)
            # weights are samps x neurons x frames

            if normWeights: ##%% normalize weights of each sample        
                normw = sci.linalg.norm(winh0, axis=1);   # numSamps x numFrames
                winh0 = np.transpose(np.transpose(winh0,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
                normw_inh_allD.append(normw)
                
                normw = sci.linalg.norm(wexc0, axis=1);   # numSamps x numFrames
                wexc0 = np.transpose(np.transpose(wexc0,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
                normw_exc_allD.append(normw)
                

            ##%% Take average across samples (this is like getting a single decoder across all trial subselects... so I guess bagging)... (not ave of abs... the idea is that ave of shuffles represents the neurons weights for the decoder given the entire dataset (all trial subselects)... if a w switches signs across samples then it means this neuron is not very consistently contributing to the choice... so i think we should average its w and not its abs(w))
            winhAve = np.mean(winh0, axis=0) # neurons x frames
            wexcAve = np.mean(wexc0, axis=0)
                
            if normWeights: ##%% normalize averaged weights
    #        if useAllNdecoder==0:
                normw = sci.linalg.norm(winhAve, axis=0)
                winhAve = winhAve / normw
                        
                normw = sci.linalg.norm(wexcAve, axis=0)
                wexcAve = wexcAve / normw

        
        #%% Once done with all frames, keep vars for all days
               
        # Delete vars before starting the next day           
    #    del perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, wAllC, bAllC
            
        winhAve_all.append(winhAve) # aveSamps len: numdays; each day: (nNeurons x nFrames)
        wexcAve_all.append(wexcAve)

        # use weights of each samp... I think it is completely fine to work the ave-samp weights... this ave decoder is the one that best represents each neurons weight                
        '''
        winh_all.append(winh0) # each samp (numSamples x nNeurons x nFrames)
        wexc_all.append(wexc0)     
        '''    
        # time course of weights for each day            
        '''
        plt.figure()
        plt.plot(np.mean(wexcAve,axis=0),'b'); 
        plt.plot(np.mean(winhAve,axis=0),'r'); 
        plt.title(days[iday])
        '''
        
    #%%
    ######################################################################    
    ######################### DONE WITH ALL DAYS #########################
        
    #%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
    # th for min number of trs of each class
    
    mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
    print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
    print np.array(days)[mn_corr < thTrained]
    
    numDaysAll_good[im] = sum(mn_corr>=thTrained)
#    days = np.array(days)[mn_corr>=thTrained]
    
        
    #%% Once done with all days keep vars for all days
    
    eventI_allDays = eventI_allDays.astype('int')
    eventI_ds_allDays = eventI_ds_allDays.astype('int')
    cbestFrs_all = np.array(cbestFrs_all)    
    numDaysAll = numDaysAll.astype(int)
    numDaysAll_good = numDaysAll_good.astype(int)
    

    #%% Set weights for frame -1 (at the end of this page you compute inha and exca, which are a general version of _all2 vars.... ie they include w values for all frames instead of just frame -1.)
    # only include days with enough svm-trained trials    

    winhAve_all2 = np.array([winhAve_all[iday][:, eventI_ds_allDays[iday]+time2an] for iday in range(len(days)) if mn_corr[iday]>=thTrained]) #numGoodDays; each day:nNeurons
    wexcAve_all2 = np.array([wexcAve_all[iday][:, eventI_ds_allDays[iday]+time2an] for iday in range(len(days)) if mn_corr[iday]>=thTrained])
    

    #%% Keep values of all mice
    
    # aveSamps, all time bins
    winhAve_all_allMice.append(winhAve_all) # nMice; each mouse: nAllDays (not just good days); each day: nNeurons x n Frs (ie weights at all times)
    wexcAve_all_allMice.append(wexcAve_all)    

    # aveSamps, time bin -1
    # only include days with enough svm-trained trials    
    winhAve_all2_allMice.append(winhAve_all2)  # nMice; each mouse: nGoodDays; each day: nNeurons (ie weights at time -1)
    wexcAve_all2_allMice.append(wexcAve_all2)    
    
    eventI_allDays_allMice.append(eventI_allDays)
    eventI_ds_allDays_allMice.append(eventI_ds_allDays)
    corr_hr_lr_allMice.append(corr_hr_lr)    
#    cbestFrs_all_allMice.append(cbestFrs_all)
    
    normw_allN_allD_allMice.append(normw_allN_allD)
    normw_inh_allD_allMice.append(normw_inh_allD)
    normw_exc_allD_allMice.append(normw_exc_allD)
    
    
    #%%    
    ####################### PLOTS for each mouse #########################
    ###### Histogram of w and absW at time bin -1 for each mouse #####################
    ######################################################################
    # Plot hists and error bars for exc and inh weights; only days with enough svm-trained trials are included in ws below
            
    if doPlots:
        ##### weights        
        lab = 'w'
        ylab = 'Fract neurons (all days)'
        plotHistErrBarWsEI(wexcAve_all2, winhAve_all2, lab, ylab, doIndivDays=1)  
        
        ##### abs weights
        lab = 'abs w'
        ylab = 'Fract neurons (all days)'        
        plotHistErrBarWsEI(abs(wexcAve_all2), abs(winhAve_all2), lab, ylab, doIndivDays=1)  




#%%        
##############################################################################################################################################################
###############    PLOTS OF ALL MICE   ############################################################################################################################
##############################################################################################################################################################
#######################################################################################################################################################
##############################################################################################################################################################
 
dnowAM = os.path.join(dnow0,'allMice',dect,nl)

 
#%% PLOTS OF ALL MICE

# number of exc and inh neurons per mouse
numExcEachMouse = np.array([(np.concatenate((wexcAve_all2_allMice[im]))).shape[0] for im in range(len(mice))])
numInhEachMouse = np.array([(np.concatenate((winhAve_all2_allMice[im]))).shape[0] for im in range(len(mice))])
print 'each mouse: numExc:', numExcEachMouse
print 'each mouse: numInh:', numInhEachMouse
print 'each mouse: numExc/numInh:', numExcEachMouse/numInhEachMouse.astype(float)
print 'All mice: totExc:', sum(numExcEachMouse)
print 'All mice: totInh:', sum(numInhEachMouse)  
    
#dp = 'time' + str(time2an) #+ '_' + nowStr    
dpp = '_'.join(mice) + '_' + nowStr   


#%% Define function to plot Hist of w at frame -1 --> at the end of this page you compute inha and exca, which are a general version of _all2 vars.... ie they include w values for all frames instead of just frame -1.
        
def plotHistErrBarWsEI_allMice(we_allM, wi_allM, ne, ni, lab, colors):
    
    ######### dist (all mice pooled)
#    lab = 'w'
    ylab = 'Fract neurons (all days,mice pooled)'
    ax1 = plotHistErrBarWsEI(np.concatenate((we_allM)), np.concatenate((wi_allM)), lab, ylab, 0)  
    ax1.set_title('num exc = %d; inh = %d' %(sum(ne), sum(ni)))
    
    
    ########## individual mouse
    wexc_aveM = np.array([np.mean(np.concatenate((we_allM[im]))) for im in range(len(mice))])        
    winh_aveM = np.array([np.mean(np.concatenate((wi_allM[im]))) for im in range(len(mice))])
    
    wexc_sdM = np.array([np.std(np.concatenate((we_allM[im])))/np.sqrt(ne[im]) for im in range(len(mice))])        
    winh_sdM = np.array([np.std(np.concatenate((wi_allM[im])))/np.sqrt(ni[im]) for im in range(len(mice))])
    
    #plt.figure(figsize=(2,3))
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    ax = plt.subplot(gs[1,0:2])
    plt.errorbar(range(len(mice)), wexc_aveM, wexc_sdM, fmt='o', label='exc', color=colors[0])
    plt.errorbar(range(len(mice)), winh_aveM, winh_sdM, fmt='o', label='inh', color=colors[1])
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
    plt.xlabel('Mice', fontsize=11)
    plt.ylabel('Classifier weights', fontsize=11)
    plt.xlim([-.2,len(mice)-1+.2])
    plt.xticks(range(len(mice)),mice)
    ax = plt.gca()
    makeNicePlots(ax,0,1)
    
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnowAM)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+cha+'dist_'+dp+'_'+lab+'_excVSinh_'+dpp+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')
        
        

#%% Hist of w at frame -1 --> at the end of this page you compute inha and exca, which are a general version of _all2 vars.... ie they include w values for all frames instead of just frame -1.

##### weights        
lab = 'w'
plotHistErrBarWsEI_allMice(wexcAve_all2_allMice, winhAve_all2_allMice, numExcEachMouse, numInhEachMouse, lab, colors)

##### Abs weights        
lab = 'abs w'
plotHistErrBarWsEI_allMice([abs(wexcAve_all2_allMice[im]) for im in range(len(mice))], [abs(winhAve_all2_allMice[im]) for im in range(len(mice))], numExcEachMouse, numInhEachMouse, lab, colors)


#%% Plot time course of w (above we just plotted time -1) 
####################################################################################    
####################################################################################      
####################################################################################      
     
#%%  Compute average weights across neurons for each frame
# input: winhAve_all_allMice: nMice; each mouse: nAllDays (not just good days); each day: nNeurons x n Frs (ie weights at all times)  
# output: winhAve_all_allMice_eachFr: nMice; each mouse: nAllDays (not just good days); each day: n Frs (ie neuron_averaged weights at all times)          
     
# average weights across neurons for each frame ... for each session of each mouse    
winhAve_all_allMice_eachFr = []
wexcAve_all_allMice_eachFr = []
winhAbsAve_all_allMice_eachFr = []
wexcAbsAve_all_allMice_eachFr = []
for im in range(len(mice)):
    # average across neurons for each frame
    winhAve_all_allMice_eachFr.append(np.array([np.mean(winhAve_all_allMice[im][iday], axis=0) for iday in range(numDaysAll[im])]))
    wexcAve_all_allMice_eachFr.append(np.array([np.mean(wexcAve_all_allMice[im][iday], axis=0) for iday in range(numDaysAll[im])]))    
     # ABS w: average across neurons for each frame
    winhAbsAve_all_allMice_eachFr.append(np.array([np.mean(abs(winhAve_all_allMice[im][iday]), axis=0) for iday in range(numDaysAll[im])]))
    wexcAbsAve_all_allMice_eachFr.append(np.array([np.mean(abs(wexcAve_all_allMice[im][iday]), axis=0) for iday in range(numDaysAll[im])]))    

winhAve_all_allMice_eachFr = np.array(winhAve_all_allMice_eachFr) # nMice; each mouse: nAllDays (not just good days); each day: n Frs (ie neuron-averaged weights at all times)   
wexcAve_all_allMice_eachFr = np.array(wexcAve_all_allMice_eachFr)
winhAbsAve_all_allMice_eachFr = np.array(winhAbsAve_all_allMice_eachFr)
wexcAbsAve_all_allMice_eachFr = np.array(wexcAbsAve_all_allMice_eachFr)



#%% Now aligned the traces created above (ie neuron-averaged time course of weights)

############## FOR ALL MICE: Align weights at all frames of all days to make a final average trace ##############
##%% 
##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.

##### nPreMin : index of eventI on aligned traces ####

nPostAll = []
for im in range(len(mice)):
    numDays = numDaysAll[im]
    nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
    for iday in range(numDays):
        nPost[iday] = (len(winhAve_all_allMice_eachFr[im][iday]) - eventI_ds_allDays_allMice[im][iday] - 1)
    nPostAll.append(nPost)
    
nPreMin = np.min(np.concatenate((eventI_ds_allDays_allMice))) #min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(np.concatenate((nPostAll)))
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces
'''
if corrTrained==0: # remember below has issues...
    a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
    b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
    time_aligned = np.concatenate((a,b))
else:
'''
totLen = nPreMin + nPostMin +1
time_aligned = set_time_al(totLen, np.min(np.concatenate((eventI_allDays_allMice))), lastTimeBinMissed)

print time_aligned


#%% Align neuron-averaged weight traces of all days on the common eventI
# only include days with enough svm-trained trials    

# input: winhAve_all_allMice_eachFr: nMice; each mouse: nAllDays (not just good days); each day: n Frs (ie neuron_averaged weights at all times)        
# output: winhAve_all_allMice_eachFr_aligned_all: # nMice; each mouse is nAlignedFrames x days(all, but bad is set to nan)

##### average-neuron weights
winhAve_all_allMice_eachFr_aligned_all = [] # nMice; each mouse is nAlignedFrames x days(all, but bad is set to nan)
wexcAve_all_allMice_eachFr_aligned_all = []
winhAbsAve_all_allMice_eachFr_aligned_all = [] 
wexcAbsAve_all_allMice_eachFr_aligned_all = []    
for im in range(len(mice)):
    numDays = numDaysAll[im]
    
    winhAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
    wexcAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)    
    winhAbsAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
    wexcAbsAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)    
    
    # set to nan days with too few svm-trained trials    
    corr_hr_lr = corr_hr_lr_allMice[im]
    mn_corr = np.min(corr_hr_lr,axis=1)        
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained:
            winhAve_all_allMice_eachFr_aligned[:, iday] = winhAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
            wexcAve_all_allMice_eachFr_aligned[:, iday] = wexcAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
            winhAbsAve_all_allMice_eachFr_aligned[:, iday] = winhAbsAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
            wexcAbsAve_all_allMice_eachFr_aligned[:, iday] = wexcAbsAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]

    winhAve_all_allMice_eachFr_aligned_all.append(winhAve_all_allMice_eachFr_aligned) # nMice; each mouse is nAlignedFrames x days(all, but bad is set to nan)
    wexcAve_all_allMice_eachFr_aligned_all.append(wexcAve_all_allMice_eachFr_aligned)    
    winhAbsAve_all_allMice_eachFr_aligned_all.append(winhAbsAve_all_allMice_eachFr_aligned)
    wexcAbsAve_all_allMice_eachFr_aligned_all.append(wexcAbsAve_all_allMice_eachFr_aligned)    


    
#%% Average across sessions for each mouse:
    # Set average and se of neuron-averaged weights across sessions (only good sessions) for each mouse ... 
    # these vars will be used if poolDaysOfMice = 0 
# input: winhAve_all_allMice_eachFr_aligned_all: # nMice; each mouse is nAlignedFrames x days(all, but bad is set to nan)
# output: winhAveDay_aligned: # numMouse x numFrames
    
# ave
winhAveDay_aligned = np.array([np.nanmean(winhAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames
wexcAveDay_aligned = np.array([np.nanmean(wexcAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames

winhAbsAveDay_aligned = np.array([np.nanmean(winhAbsAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames
wexcAbsAveDay_aligned = np.array([np.nanmean(wexcAbsAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames

# se
winhSdDay_aligned = np.array([np.nanstd(winhAve_all_allMice_eachFr_aligned_all[im], axis=1)/np.sqrt(numDaysAll_good[im]) for im in range(len(mice))]) # numMouse x numFrames
wexcSdDay_aligned = np.array([np.nanstd(wexcAve_all_allMice_eachFr_aligned_all[im], axis=1)/np.sqrt(numDaysAll_good[im]) for im in range(len(mice))]) # numMouse x numFrames

winhAbsSdDay_aligned = np.array([np.nanstd(winhAbsAve_all_allMice_eachFr_aligned_all[im], axis=1)/np.sqrt(numDaysAll_good[im]) for im in range(len(mice))]) # numMouse x numFrames
wexcAbsSdDay_aligned = np.array([np.nanstd(wexcAbsAve_all_allMice_eachFr_aligned_all[im], axis=1)/np.sqrt(numDaysAll_good[im]) for im in range(len(mice))]) # numMouse x numFrames


#%% Pool sessions of all mice
    # Pool neuron-average weights across sessions and mice ... 
    # these vars will be used if poolDaysOfMice = 1
# input: winhAve_all_allMice_eachFr_aligned_all: # nMice; each mouse is nAlignedFrames x days(all, but bad is set to nan)
# output: winhAllDaysMice_aligned: (pooled sessions of all mice) x frames

if len(mice)>1:
    winhAllDaysMouse = [] # (numFrs x numMice) ... first all frames of mouse 0, then all frames of mouse 1, then all frames of mouse 2
    wexcAllDaysMouse = []
    winhAbsAllDaysMouse = [] # (numFrs x numMice) ... first all frames of mouse 0, then all frames of mouse 1, then all frames of mouse 2
    wexcAbsAllDaysMouse = []
    for im in range(len(mice)):
        for ifr in range(len(time_aligned)):
            winhAllDaysMouse.append(winhAve_all_allMice_eachFr_aligned_all[im][ifr]) # weights for all sessions of mouse im, and frame ifr
            wexcAllDaysMouse.append(wexcAve_all_allMice_eachFr_aligned_all[im][ifr])
            winhAbsAllDaysMouse.append(winhAbsAve_all_allMice_eachFr_aligned_all[im][ifr]) # weights for all sessions of mouse im, and frame ifr
            wexcAbsAllDaysMouse.append(wexcAbsAve_all_allMice_eachFr_aligned_all[im][ifr])    
#    if len(mice)==1:
#        winhAllDaysMouse = [winhAllDaysMouse]
#        wexcAllDaysMouse = [wexcAllDaysMouse]
            
    # now pool sessions of all mice for each frame; so we get : (sessions of all mice) x frames            
    a = np.reshape(winhAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    winhAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    a = np.reshape(wexcAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    wexcAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    
    a = np.reshape(winhAbsAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    winhAbsAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    a = np.reshape(wexcAbsAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    wexcAbsAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    

#%% Plot the final average and sd across mice (poolDaysOfMice=0) also across pooled sessions (poolDaysOfMice=1)
# remember neurons of each sessions are already averaged for each frame

#############%% Define function ot plot the average of aligned traces across all days
def plotwTraces(ye,ee, yi, ei, p, ylab):

    plt.fill_between(time_aligned, ye - ee, ye + ee, alpha=0.5, edgecolor=colors[0], facecolor=colors[0])
    plt.plot(time_aligned, ye, colors[0], label='exc')
    
    plt.fill_between(time_aligned, yi - ei, yi + ei, alpha=0.5, edgecolor=colors[1], facecolor=colors[1])
    plt.plot(time_aligned, yi, colors[1], label='inh')
        
    if chAl==1:
        plt.xlabel('Time since choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    plt.ylabel(ylab, fontsize=11)
    
    #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)     
    ax = plt.gca()        
    makeNicePlots(ax,1,1)
    
    # Plot a dot for significant time points
    ymin, ymax = ax.get_ylim()    
    plt.vlines(0,ymin,ymax, color='k',linestyle=':')        
    pp = p+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    plt.plot(time_aligned, pp, color='k')
        

#%% call the above function

def call_plotwTraces(av_wexc, sd_wexc, av_winh, sd_winh, pcorrtrace,av_wexc_abs, sd_wexc_abs, av_winh_abs, sd_winh_abs, pAbscorrtrace, dnowAM, dpp):
    ############### PLOT ###############                
    fig = plt.figure(figsize=(3,4))
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:3]
    h2 = gs[1,0:3]
    
    ######## Weights
    plt.subplot(h1)          
    ylab = 'Classifier weights'
    plotwTraces(av_wexc, sd_wexc, av_winh, sd_winh, pcorrtrace, ylab)

    ######## Abs Weights
    plt.subplot(h2)
    ylab = 'Abs (classifier weights)'
    plotwTraces(av_wexc_abs, sd_wexc_abs, av_winh_abs, sd_winh_abs, pAbscorrtrace, ylab)        
    
    plt.subplots_adjust(wspace=1, hspace=.8)
    
    
    ##%% Save the figure    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnowAM)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        if poolDaysOfMice==0:
            fign = os.path.join(d, suffn[0:5]+cha+'timeCourse_w_absw_aveMice_aveDays_aveNs_excVSinh_'+dpp+'.'+fmt[0])
        else:
            fign = os.path.join(d, suffn[0:5]+cha+'timeCourse_w_absw_avePooledMiceDays_aveNs_excVSinh_'+dpp+'.'+fmt[0])    
    
        fig.savefig(fign, bbox_inches='tight')
    
        
#%%
############### Set vars ###############
nMice = len(mice)
nTotGoodSess = sum(numDaysAll_good)

for im in range(len(mice)):        
#    execfile("svm_plots_setVars_n.py")  # to get days
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    dnow = os.path.join(dnow0,mousename,dect,nl)   
    dpm = 'days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr   
#        dpmAllm.append(dpm)
                
    #### weights            
    av_winh = winhAveDay_aligned.flatten() # nFrames (average across sessions; each session: averaged across neurons)
    sd_winh = winhSdDay_aligned.flatten()
    
    av_wexc = wexcAveDay_aligned.flatten()
    sd_wexc = wexcSdDay_aligned.flatten()
    
    _,pcorrtrace = stats.ttest_ind(winhAve_all_allMice_eachFr_aligned_all[im].T, wexcAve_all_allMice_eachFr_aligned_all[im].T, nan_policy='omit') # p value of class accuracy being different from 50

    #### Abs
    av_winh_abs = winhAbsAveDay_aligned.flatten()
    sd_winh_abs = winhAbsSdDay_aligned.flatten()
    
    av_wexc_abs = wexcAbsAveDay_aligned.flatten()
    sd_wexc_abs = wexcAbsSdDay_aligned.flatten()
    
    _,pAbscorrtrace = stats.ttest_ind(winhAbsAve_all_allMice_eachFr_aligned_all[im].T, wexcAbsAve_all_allMice_eachFr_aligned_all[im].T, nan_policy='omit')
    
    poolDaysOfMice=0
    call_plotwTraces(av_wexc, sd_wexc, av_winh, sd_winh, pcorrtrace,av_wexc_abs, sd_wexc_abs, av_winh_abs, sd_winh_abs, pAbscorrtrace, dnow, dpm)
    
    
######################## Averages of all mice ########################        
if len(mice)>1:
    for poolDaysOfMice in [0,1]: # for plotting w traces; if 0, average and sd across mice (each mouse has session-averaged weights); if 1, average and sd across all sessions of all mice           
        if poolDaysOfMice==0:
            # average and sd across mice (each mouse has session-averaged weights)
            a = winhAveDay_aligned # numMouse x numFrames
            b = wexcAveDay_aligned
            aa = winhAbsAveDay_aligned
            bb = wexcAbsAveDay_aligned
            n = nMice
        else:
            # average and sd across all sessions of all mice
            a = winhAllDaysMice_aligned # (pooled sessions of all mice) x frames
            b = wexcAllDaysMice_aligned
            aa = winhAbsAllDaysMice_aligned
            bb = wexcAbsAllDaysMice_aligned
            n = nTotGoodSess        
        #a = winh_fractnon0
        #b = wexc_fractnon0
        
        ################# set ave and se ################           
        #### weights            
        av_winh = np.nanmean(a, axis=0)
        sd_winh = np.nanstd(a, axis=0)/ np.sqrt(n)
        
        av_wexc = np.nanmean(b, axis=0)
        sd_wexc = np.nanstd(b, axis=0)/ np.sqrt(n)
        
        _,pcorrtrace = stats.ttest_ind(a, b, nan_policy='omit') # p value of class accuracy being different from 50

        #### Abs
        av_winh_abs = np.nanmean(aa, axis=0)
        sd_winh_abs = np.nanstd(aa, axis=0)/ np.sqrt(n)
        
        av_wexc_abs = np.nanmean(bb, axis=0)
        sd_wexc_abs = np.nanstd(bb, axis=0)/ np.sqrt(n)
        
        _,pAbscorrtrace = stats.ttest_ind(aa, bb, nan_policy='omit')
    

    call_plotwTraces(av_wexc, sd_wexc, av_winh, sd_winh, pcorrtrace,av_wexc_abs, sd_wexc_abs, av_winh_abs, sd_winh_abs, pAbscorrtrace, dnowAM, dpp)

        


#%% HERE we do not average neurons of each session anymore!!... we work with single neurons!

############################################################################################
############################################################################################
####################################################################################      
##########%%  Align all neuron's trace of weights
####################################################################################
# Above you took average of weights across neurons and sessions and mice; 
# Here we take each neuron's weight (without any averaging) and look into the histograms of weights.       
####################################################################################    

#%% Get values of w (for all neurons) at different time points from the aligned traces; (comapre w _all2 to make sure eveything is fine)

# input: winhAve_all_allMice: nMice; each mouse: nAllDays (not just good days); each day: nNeurons x nFrs (ie weights at all times)  
# output: winhAve_all_allMice_eachFr_aligned_allN: nMice; each mouse is nGoodDays; each day is: neurons x alignedFrames

# winhAve_all_allMice_eachFr_aligned_allN is same as winhAve_all_allMice except it is aligned on the common eventI. Also it includes only good days.
# (it includes all neurons' weights at all frames for all days and mice)

###### Align traces of all days on the common eventI
##### all neurons' weights    --->    Align traces of all days on the common eventI
winhAve_all_allMice_eachFr_aligned_allN = [] # nMice; each mouse is nGoodDays; each day is: neurons x alignedFrames
wexcAve_all_allMice_eachFr_aligned_allN = []

for im in range(len(mice)):
    winhAve_all_allMice_eachFr_aligned = [] # each day is neurons x alignedFrames, aligned on common eventI (equals nPreMin)
    wexcAve_all_allMice_eachFr_aligned = [] 
    
    numDays = numDaysAll[im]        
    corr_hr_lr = corr_hr_lr_allMice[im]
    mn_corr = np.min(corr_hr_lr,axis=1)
    
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained:
            winhAve_all_allMice_eachFr_aligned.append(winhAve_all_allMice[im][iday][:, eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1])
            wexcAve_all_allMice_eachFr_aligned.append(wexcAve_all_allMice[im][iday][:, eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1])

    winhAve_all_allMice_eachFr_aligned_allN.append(winhAve_all_allMice_eachFr_aligned)
    wexcAve_all_allMice_eachFr_aligned_allN.append(wexcAve_all_allMice_eachFr_aligned)    

    
    
    
    
#%% Set fraction non-zero weights for each time bin... to plot the timecourse
# intput: winhAve_all_allMice_eachFr_aligned_allN: nMice; each mouse is nGoodDays; each day is: neurons x alignedFrames    
# output: winh_fractnon0: nGoodDays x nAlignedFrames
    
#    eps = 1e-10
eps = sys.float_info.epsilon

winh_fractnon0 = [] # daysOfAllMice x alignedFrames
wexc_fractnon0 = []
for im in range(len(mice)):
    corr_hr_lr = corr_hr_lr_allMice[im]
    mn_corr = np.min(corr_hr_lr,axis=1)
#    inh0 = []
#    exc0 = []
    for iday in range(numDaysAll_good[im]):            
#        inh0.append(np.mean(abs(winhAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) # nFrames # fraction non0 weights for each frame
#        exc0.append(np.mean(abs(wexcAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) 
        winh_fractnon0.append(np.mean(abs(winhAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) # nGoodDays x nFrames # fraction non0 weights for each frame
        wexc_fractnon0.append(np.mean(abs(wexcAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) 
#    winh_fractnon0.append(inh0)
#    wexc_fractnon0.append(exc0)    
    

########## Set ave and se of fract non-0 weights across sessions
a = winh_fractnon0
b = wexc_fractnon0
n = sum(numDaysAll_good) # np.shape(a)[0]
    
av_winh = np.mean(a, axis=0)
sd_winh = np.std(a, axis=0)/ np.sqrt(n)
av_wexc = np.mean(b, axis=0)
sd_wexc = np.std(b, axis=0)/ np.sqrt(n)

_,pcorrtrace = stats.ttest_ind(a, b) # p value of class accuracy being different from 50

       
##### Plot the average fract non-0 ws of aligned traces across all days    
fig = plt.figure(figsize=(3,4))
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:3]
h2 = gs[1,0:3]

plt.subplot(h1)          
ylab = 'Fract non-0 classifier weights'
plotwTraces(av_wexc, sd_wexc, av_winh, sd_winh, pcorrtrace, ylab)


##%% Save the figure    
if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnowAM)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+cha+'fractNon0w_timeCourse_aveMiceDays_aveNs_excVSinh_'+dpp+'.'+fmt[0])    
    fig.savefig(fign, bbox_inches='tight')

    

    
#######################################################        
#######################################################        
#%% Get values of winhAve_all_allMice_eachFr_aligned_allN for each frame    
    
# input: winhAve_all_allMice_eachFr_aligned_allN: nMice; each mouse is nGoodDays; each day is: neurons x alignedFrames
# output: winhAve_allFrs_allMice: nMice x alignedFrames; each element: nGoodDays; each day: nNeurons
    
# np.shape(winhAve_allFrs_allMice): nMice x alignedFrames
# np.shape(winhAve_allFrs_allMice[0]): alignedFrames x nGoodDays
# np.shape(winhAve_allFrs_allMice[0][0]): nGoodDays
# np.shape(winhAve_allFrs_allMice[0][0][0]): nNeurons
winhAve_allFrs_allMice = [] 
wexcAve_allFrs_allMice = [] 
for im in range(len(mice)):       
    winhAve_allFrs = []
    wexcAve_allFrs = []
    for fr in range(len(time_aligned)):
#        nPreMin+fr2an
        winhAve_allFrs.append(np.array([winhAve_all_allMice_eachFr_aligned_allN[im][iday][:,fr] for iday in range(numDaysAll_good[im])]))
        wexcAve_allFrs.append(np.array([wexcAve_all_allMice_eachFr_aligned_allN[im][iday][:,fr] for iday in range(numDaysAll_good[im])]))
    
    winhAve_allFrs_allMice.append(winhAve_allFrs)
    wexcAve_allFrs_allMice.append(wexcAve_allFrs)    

    
'''    
# get values at time point fr2an    
# inha is exactly like wexcAve_all2_allMice when fr2an = time2an
# winhAve_all2_allMice.append(winhAve_all2)  # nMice; each mouse: nGoodDays; each day: nNeurons (ie weights at time -1)
fr2an = -1
inha = []
exca = []
for im in range(len(mice)):    
    inha.append(np.array([winhAve_allFrs_allMice[im][nPreMin+fr2an][iday] for iday in range(numDaysAll_good[im])]))
    exca.append(np.array([wexcAve_allFrs_allMice[im][nPreMin+fr2an][iday] for iday in range(numDaysAll_good[im])]))        
'''    


#%% Plot weight traces again, this time pool all neurons of all days and all mice, then make ave and se...
# above we first averaged neurons of each session then made plots.

def plotAllPooled(dnowAM, dpp):        
    ############### SET VARS ###############       
         
    wExcTraceAveall = np.full((len(time_aligned)), np.nan)
    wInhTraceAveall = np.full((len(time_aligned)), np.nan)
    wExcTraceSdall = np.full((len(time_aligned)), np.nan)
    wInhTraceSdall = np.full((len(time_aligned)), np.nan)    
    pAll = np.full((len(time_aligned)), np.nan)    
    wAbsExcTraceAveall = np.full((len(time_aligned)), np.nan)
    wAbsInhTraceAveall = np.full((len(time_aligned)), np.nan)
    wAbsExcTraceSdall = np.full((len(time_aligned)), np.nan)
    wAbsInhTraceSdall = np.full((len(time_aligned)), np.nan)    
    pAbsAll = np.full((len(time_aligned)), np.nan)        
    
    for fr in range(len(time_aligned)):
        # inha: nMice; each mouse: nGoodDays; each day: nNeurons (for a specific time point)
        # just like : # winhAve_all2_allMice.append(winhAve_all2)         
        inha = []
        exca = []
        for im in range(len(mice)):    
            inha.append(np.array([winhAve_allFrs_allMice[im][fr][iday] for iday in range(numDaysAll_good[im])]))
            exca.append(np.array([wexcAve_allFrs_allMice[im][fr][iday] for iday in range(numDaysAll_good[im])]))
            
        ## pooling all neurons of all days of all mice for a particular frame            
        # w
        e = np.concatenate((np.concatenate((exca)))) # sum(numExcEachMouse) 
        i = np.concatenate((np.concatenate((inha)))) # sum(numInhEachMouse)
        wExcTraceAveall[fr] = np.mean(e)
        wInhTraceAveall[fr] = np.mean(i)
        wExcTraceSdall[fr] = np.std(e)/ np.sqrt(sum(numExcEachMouse))
        wInhTraceSdall[fr] = np.std(i)/ np.sqrt(sum(numInhEachMouse))        
        _, pAll[fr] = stats.ttest_ind(e, i)#, nan_policy='omit')
        
        # abs w
        e = np.concatenate((abs(np.concatenate((exca)))))
        i = np.concatenate((abs(np.concatenate((inha)))))
        wAbsExcTraceAveall[fr] = np.mean(e)
        wAbsInhTraceAveall[fr] = np.mean(i)
        wAbsExcTraceSdall[fr] = np.std(e)/ np.sqrt(sum(numExcEachMouse))
        wAbsInhTraceSdall[fr] = np.std(i)/ np.sqrt(sum(numInhEachMouse))        
        _, pAbsAll[fr] = stats.ttest_ind(e, i)#, nan_policy='omit')
        
#    wExcTraceSdall = wExcTraceSdall / np.sqrt(sum(numExcEachMouse))
#    wInhTraceSdall = wInhTraceSdall / np.sqrt(sum(numInhEachMouse))
#    wAbsExcTraceSdall = wAbsExcTraceSdall / np.sqrt(sum(numExcEachMouse))
#    wAbsInhTraceSdall = wAbsInhTraceSdall / np.sqrt(sum(numInhEachMouse))
    
    
    ############### PLOT ###############      
          
    fig = plt.figure(figsize=(3,4))
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:3]
    h2 = gs[1,0:3]
    
    plt.subplot(h1)
    plotwTraces(wExcTraceAveall , wExcTraceSdall , wInhTraceAveall , wInhTraceSdall , pAll, ylab='Classifier weights')
    
    plt.subplot(h2)
    plotwTraces(wAbsExcTraceAveall , wAbsExcTraceSdall , wAbsInhTraceAveall , wAbsInhTraceSdall , pAbsAll, ylab='Abs (classifier weights)')
    
    plt.subplots_adjust(wspace=1, hspace=.8)
    
    
    ##%% Save the figure    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnowAM)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'timeCourse_w_absw_avePooledMiceDaysNs_excVSinh_'+dpp+'.'+fmt[0])
        
        plt.savefig(fign, bbox_inches='tight')


if len(mice)==1:            
    for im in range(len(mice)):        
#        execfile("svm_plots_setVars_n.py")  # to get days
        days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
        mousename = mice[im] # mousename = 'fni16' #'fni17'
        dnow = os.path.join(dnow0,mousename,dect,nl)   
        dpm = 'days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
        
        plotAllPooled(dnow, dpm)        

else: # pool all mice
    plotAllPooled(dnowAM, dpp)        
    

#%% Plot hist of w and abs(w) for exc, inh for each time point relative to eventI (on the aligned trace)
# (pooled across all sessions and all mice and all neurons)

plt.figure(figsize=(5,70))    
gs = gridspec.GridSpec(2*len(time_aligned), 3)#, width_ratios=[2, 1]) 
   

for fr in range(len(time_aligned)):
    # inha: nMice; each mouse: nGoodDays; each day: nNeurons (for a specific time point)
    # just like : # winhAve_all2_allMice.append(winhAve_all2)         
    inha = []
    exca = []
    for im in range(len(mice)):    
        inha.append(np.array([winhAve_allFrs_allMice[im][fr][iday] for iday in range(numDaysAll_good[im])]))
        exca.append(np.array([wexcAve_allFrs_allMice[im][fr][iday] for iday in range(numDaysAll_good[im])]))

    
    ### PLOT HISTS               
    ##### weights        
    h1 = gs[fr*2,0:2]
    h2 = gs[fr*2,2:3]            
    lab = 'w'
    ylab = 'Fract neurons (all days,mice)'        
    ax1 = plotHistErrBarWsEI(np.concatenate((exca)), np.concatenate((inha)), lab, ylab, 0,1,h1,h2)  
    ax1.set_title('%.0f ms' %(time_aligned[fr]))
    
    ##### abs weights
    h1 = gs[fr*2+1,0:2]
    h2 = gs[fr*2+1,2:3]
    lab = 'abs w'                
    ax1 = plotHistErrBarWsEI(np.concatenate(([abs(exca[im]) for im in range(len(mice))])), np.concatenate(([abs(inha[im]) for im in range(len(mice))])), lab, '', 0,1,h1,h2)
    ax1.set_title('%.0f ms' %(time_aligned[fr]))            
#        ax1.set_title('')
    
plt.subplots_adjust(wspace=1, hspace=.8)    
    

##%% Save the figure    
if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnowAM)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
    
    fign = os.path.join(d, suffn[0:5]+cha+'dist_allFrs_w_absw_excVSinh_'+dpp+'.'+fmt[0])
    
    plt.savefig(fign, bbox_inches='tight')
    
            
    
#%% Plot 2-norm of decoders       
########################################################################################        
########################################################################################

if normWeights:        
    #####%% average 2-norm ws across samps and align 2-norm traces of all days of all mice
    
    def alAv2NormW(normw_allN_allD_allMice):
        
        ##%% Average 2-norm across samps
                
        normw_allM = []
        
        for im in range(len(mice)):    
            normw_allD = [np.mean(normw_allN_allD_allMice[im][iday], axis=0) for iday in range(numDaysAll[im])] 
            normw_allM.append(normw_allD)  #nMice; each mouse: nAllDays; each day: nFrs (decoders 2-norm for each frame)
        
        
        ##%% Align 2-norm traces of all days on the common eventI
        # input: # nMice; each mouse is nGoodDays; each day is: nFrames
        # output: # nMice; each mouse is nGoodDays x alignedFrames
        
        normw_allM_aligned = [] 
        
        for im in range(len(mice)):    
            normw_aligned = np.full((numDaysAll[im], len(time_aligned)), np.nan)
            mn_corr = np.min(corr_hr_lr_allMice[im], axis=1)
            for iday in range(numDaysAll[im]):
                if mn_corr[iday]>=thTrained:
                    normw_aligned[iday,:] = normw_allM[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1] 
            normw_allM_aligned.append(normw_aligned)
            
        
        return normw_allM_aligned
    
    
    
    #%% Make plots of 2norm
    
    def plot2normw(normw_allM_aligned, lab='_allN_'):
        
        ##%% Plot 2-norm of decoder for each frame for each day ... each mouse    
        step = 5
        x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
                   np.arange(np.argwhere(time_aligned>=0)[0], len(time_aligned)+.5, step))))).astype(int)
        
        for im in range(len(mice)):    
            plt.figure()
            plt.imshow(normw_allM_aligned[im])
            plt.xticks(x, np.round(time_aligned[x]).astype(int))
            plt.xlabel('Time relative to choice onset')    
            plt.ylabel('Days')
            plt.colorbar(label='2-Norm of decoder')
            makeNicePlots(plt.gca())
            
            ##%% Save the figure    
            if savefigs:#% Save the figure
                mousename = mice[im] # mousename = 'fni16' #'fni17'
                dnow = os.path.join(dnow0,mousename,dect,nl)   
                
                d = os.path.join(svmdir+dnow)
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                
                fign = os.path.join(d, suffn[0:5]+cha+'normW'+lab+dpmAllm[im]+'.'+fmt[0])            
                plt.savefig(fign, bbox_inches='tight')       
        
        
        #######%% Plot average norm-w across days for each mouse        
        plt.figure()
        for im in range(len(mice)):        
            plt.errorbar(time_aligned, np.nanmean(normw_allM_aligned[im],axis=0), np.nanstd(normw_allM_aligned[im],axis=0), label=mice[im])   
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)     
        plt.xlabel('Time relative to choice onset')    
        plt.ylabel('2-Norm of decoder \nave +/- sd across days')
        makeNicePlots(plt.gca())
           
        ##%% Save the figure    
        if savefigs:#% Save the figure
            d = os.path.join(svmdir+dnowAM)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
            
            fign = os.path.join(d, suffn[0:5]+cha+'normW'+lab+'aveDays_'+dpp+'.'+fmt[0])            
            plt.savefig(fign, bbox_inches='tight')   
    
     
    #%%
    
    if useAllNdecoder:
        normw_allM_aligned = alAv2NormW(normw_allN_allD_allMice)
        plot2normw(normw_allM_aligned)
    else:
        # exc
        normw_allM_aligned = alAv2NormW(normw_exc_allD_allMice)
        plot2normw(normw_allM_aligned, '_exc_')
    
        # inh
        normw_allM_aligned = alAv2NormW(normw_inh_allD_allMice)
        plot2normw(normw_allM_aligned, '_inh_')
        
    