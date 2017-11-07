# -*- coding: utf-8 -*-
"""
Set vars in svm_stability_plots_excInhEachFrame_setVars.py

Created on Thu Nov  2 10:03:50 2017

@author: farznaj
"""


#%% Set the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19' # 'fni17',

doPlotsEachMouse = 0 # make plots for each mouse
savefigs = 0

doExcSamps = 1 # if 1, for exc use vars defined by averaging trial subselects first for each exc samp and then averaging across exc samps. This is better than pooling all trial samps and exc samps ...    
#excludeLowTrDays = 1 # remove days with too few trials
#nsh = 1000 # number of times to shuffle the decoders to get null distribution of angles
# quantile of stim strength - cb to use         
# 0: hard (min <= sr < 25th percentile of stimrate-cb); 
# 1: medium hard (25th<= sr < 50th); 
# 2: medium easy (50th<= sr < 75th); 
# 3: easy (75th<= sr <= max);
#thQStimStrength = 3 # 0 to 3 : hard to easy # # set to nan if you want to include all strengths in computing behaioral performance

doAllN = 1 # plot allN, instead of allExc
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
corrTrained = 1
doIncorr = 0
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces.  #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
#useEqualTrNums = 1
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'  
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
dnow0 = '/stability/'

import glob
stabLab0_data = 'Instability (average angle with other decoders)'    
stabLab0_shfl = 'Instability (shfl; average angle with other decoders)'    
stabLab0 = 'Stability (degrees away from full mis-alignment)'    
stabLab1 = 'Stability (number of aligned decoders)' # with the decoder of interest (ie the decoder at the specific timepont we are looking at)            

if chAl==1:
    xl = 'Time since choice onset (ms)'
else:
    xl = 'Time since stimulus onset (ms)'

execfile("defFuns.py")


#%%
#cmap='jet_r'; #cmap='hot'
def plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap='jet', cblab=''):

    totLen = len(time_aligned) 
    step = 4
    x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
               np.arange(np.argwhere(time_aligned>=0)[0], totLen, step))))).astype(int)
    #x = np.arange(0,totLen,step)
               
               
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


#%% Plot Angles averaged across days for data, shfl, shfl-data (plot angle between decoders at different time points)        

def plotAngsAll(nPreMin, time_aligned, angInh_av, angExc_av, angAllExc_av, angExc_excsh_av, angInhS_av, angExcS_av, angAllExcS_av, angExc_excshS_av, fna, nowStr):
    
    totLen = len(time_aligned) 
    
    ######## Plot angles, averaged across days
    # data
    if doExcSamps:
        ex = angExc_excsh_av
    else:
        ex = angExc_av        
    plt.figure(figsize=(8,8))
    cmin = np.floor(np.min([np.nanmin(angInh_av), np.nanmin(ex), np.nanmin(angAllExc_av)]))
    cmax = 90
    cmap = 'jet_r' # lower angles: more aligned: red
    cblab = 'Angle between decoders'
    lab = 'inh (data)'; top = angInh_av; plt.subplot(221); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = 'exc (data)'; top = ex; plt.subplot(223); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = labAll+' (data)'; top = angAllExc_av; plt.subplot(222); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    plt.subplots_adjust(hspace=.2, wspace=.3)
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
        if chAl==1:
            dd = 'chAl_anglesAveDays_inhExcAllExc_data_' + fna + '_' + nowStr
        else:
            dd = 'stAl_anglesAveDays_inhExcAllExc_data_' + fna + '_' + nowStr       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
            
    
    # shfl
    if doExcSamps:
        ex = angExc_excshS_av
    else:
        ex = angExcS_av        
    plt.figure(figsize=(8,8))
    cmin = np.floor(np.min([np.nanmin(angInhS_av), np.nanmin(ex), np.nanmin(angAllExcS_av)]))
    cmax = 90
    cmap = 'jet_r' # lower angles: more aligned: red
    cblab = 'Angle between decoders'
    lab = 'inh (shfl)'; top = angInhS_av; plt.subplot(221); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = 'exc (shfl)'; top = ex; plt.subplot(223); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = labAll+' (shfl)'; top = angAllExcS_av; plt.subplot(222); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    plt.subplots_adjust(hspace=.2, wspace=.3)
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
        if chAl==1:
            dd = 'chAl_anglesAveDays_inhExcAllExc_shfl_' + fna + '_' + nowStr
        else:
            dd = 'stAl_anglesAveDays_inhExcAllExc_shfl_' + fna + '_' + nowStr       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
    
    
    ##### Most useful plot:
    # shfl-real 
    if doExcSamps:
        ex = angExc_excshS_av - angExc_excsh_av
    else:
        ex = angExcS_av - angExc_av         
    plt.figure(figsize=(8,8))
    cmin = np.floor(np.min([np.nanmin(angInhS_av-angInh_av), np.nanmin(ex), np.nanmin(angAllExcS_av-angAllExc_av)]))
    cmax = np.floor(np.max([np.nanmax(angInhS_av-angInh_av), np.nanmax(ex), np.nanmax(angAllExcS_av-angAllExc_av)]))
    cmap = 'jet' # larger value: lower angle for real: more aligned: red
    cblab = 'Alignment rel. shuffle'
    lab = 'inh (shfl-data)'; 
    top = angInhS_av - angInh_av; 
#    top[np.triu_indices(len(top),k=0)] = np.nan # plot only the lower triangle (since values are rep)
    plt.subplot(221); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = labAll+'exc (shfl-data)'; 
    top = ex; 
#    top[np.triu_indices(len(top),k=0)] = np.nan
    plt.subplot(223); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = labAll+' (shfl-data)'; 
    top = angAllExcS_av - angAllExc_av; 
#    top[np.triu_indices(len(top),k=0)] = np.nan
    plt.subplot(222); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    
    # last subplot: inh(shfl-real) - exc(shfl-real)
    cmin = np.nanmin((angInhS_av - angInh_av) - (ex)); 
    cmax = np.nanmax((angInhS_av - angInh_av) - (ex))
    cmap = 'jet' # more diff: higher inh: lower angle for inh: inh more aligned: red
    cblab = 'Inh-Exc: angle between decoders'
    lab = 'inh-exc'; 
    top = ((angInhS_av - angInh_av) - (ex)); 
#    top[np.triu_indices(len(top),k=0)] = np.nan
    plt.subplot(224); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    
    # Is inh and exc stability different? 
    # for each population (inh,exc) subtract shuffle-averaged angles from real angles... do ttest across days for each pair of time points
    _,pei = stats.ttest_ind(angleInh_aligned - angleInhS_aligned_avSh , angleExc_aligned - angleExcS_aligned_avSh, axis=-1) 
#    pei[np.triu_indices(len(pei),k=0)] = np.nan
    # mark sig points by a dot:
    for f1 in range(totLen):
        for f2 in range(totLen): #np.delete(range(totLen),f1): #
            if pei[f1,f2]<.05:
                plt.plot(f2,f1, marker='*', color='b', markersize=5)
    
    plt.subplots_adjust(hspace=.2, wspace=.3)
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
        if chAl==1:
            dd = 'chAl_anglesAveDays_inhExcAllExc_dataMshfl_' + fna + '_' + nowStr
        else:
            dd = 'stAl_anglesAveDays_inhExcAllExc_dataMshfl_' + fna + '_' + nowStr       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        



           
#%% 
numDaysAll = np.full(len(mice), np.nan, dtype=int)
time_trace_all_allMice = []

angleInh_aligned_allMice = []
angleExc_aligned_allMice = []
angleAllExc_aligned_allMice = []
angleExc_excsh_aligned_allMice = []
angleExc_excsh_aligned_avExcsh_allMice = []
angleExc_excsh_aligned_sdExcsh_allMice = []   

angleInhS_aligned_avSh_allMice = []
angleExcS_aligned_avSh_allMice = []
angleAllExcS_aligned_avSh_allMice = []
angleExc_excshS_aligned_avSh_allMice = []
angleExc_excshS_aligned_avExcsh_allMice = []
angleExc_excshS_aligned_sdExcsh_allMice = []

sigAngInh_allMice = []
sigAngExc_allMice = []
sigAngAllExc_allMice = []
sigAngExc_excsh_allMice = []
sigAngExc_excsh_avExcsh_allMice = []
sigAngExc_excsh_sdExcsh_allMice = []

stabScoreInh0_data_allMice = []
stabScoreExc0_data_allMice = []
stabScoreAllExc0_data_allMice = []
stabScoreExc_excsh0_data_allMice = []
stabScoreExc_excsh0_data_avExcsh_allMice = []
stabScoreExc_excsh0_data_sdExcsh_allMice = []

stabScoreInh0_shfl_allMice = []
stabScoreExc0_shfl_allMice = []
stabScoreAllExc0_shfl_allMice = []
stabScoreExc_excsh0_shfl_allMice = []
stabScoreExc_excsh0_shfl_avExcsh_allMice = []
stabScoreExc_excsh0_shfl_sdExcsh_allMice = []

stabScoreInh0_allMice = []
stabScoreExc0_allMice = []
stabScoreAllExc0_allMice = []
stabScoreExc_excsh0_allMice = []
stabScoreExc_excsh0_avExcsh_allMice = []
stabScoreExc_excsh0_sdExcsh_allMice = []

stabScoreInh1_allMice = []
stabScoreExc1_allMice = []
stabScoreAllExc1_allMice = []
stabScoreExc_excsh1_allMice = []
stabScoreExc_excsh1_avExcsh_allMice = []
stabScoreExc_excsh1_sdExcsh_allMice = []


angInh_av_allMice = []
angExc_av_allMice = []
angAllExc_av_allMice = []
angExc_excsh_av_allMice = []

angInhS_av_allMice = []
angExcS_av_allMice = []
angAllExcS_av_allMice = []
angExc_excshS_av_allMice = []

sigAngInh_av_allMice = []
sigAngExc_av_allMice = []
sigAngAllExc_av_allMice = []
sigAngExc_excsh_av_allMice = []

stabScoreInh0_data_av_allMice = []
stabScoreExc0_data_av_allMice = []
stabScoreAllExc0_data_av_allMice = []
stabScoreExc_excsh0_data_av_allMice = []

stabScoreInh0_shfl_av_allMice = []
stabScoreExc0_shfl_av_allMice = []
stabScoreAllExc0_shfl_av_allMice = []
stabScoreExc_excsh0_shfl_av_allMice = []
     
stabScoreInh0_av_allMice = []
stabScoreExc0_av_allMice = []
stabScoreAllExc0_av_allMice = []
stabScoreExc_excsh0_av_allMice = []         

stabScoreInh1_av_allMice = []
stabScoreExc1_av_allMice = []
stabScoreAllExc1_av_allMice = []
stabScoreExc_excsh1_av_allMice = []         


classAccurTMS_inh_allMice = []
classAccurTMS_exc_allMice = []
classAccurTMS_allExc_allMice = []
     
behCorr_allMice = [] 
behCorrHR_allMice = [] 
behCorrLR_allMice = [] 

corr_hr_lr_allMice = []
days_allMice = [] 
eventI_allDays_allMice = []
eventI_ds_allDays_allMice = []
nowStr_allMice= []
     
#%%
# im = 0     
for im in range(len(mice)):
    
    #%%            
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    if mousename == 'fni18': #set one of the following to 1:
        allDays = 1# all 7 days will be used (last 3 days have z motion!)
        noZmotionDays = 0 # 4 days that dont have z motion will be used.
        noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    if mousename == 'fni19':    
        allDays = 1
        noExtraStimDays = 0   
        
    execfile("svm_plots_setVars_n.py")      
#    execfile("svm_plots_setVars.py")      
    days_allMice.append(days)
    numDaysAll[im] = len(days)
   
    dnow = dnow0+mousename+'/'


    #%% Set svm_stab mat file name

    imagingDir = setImagingAnalysisNamesP(mousename)

    fname = os.path.join(imagingDir, 'analysis')    
    if not os.path.exists(fname):
        print 'creating folder'
        os.makedirs(fname)    
        
    finame = os.path.join(fname, 'svm_stability_*.mat')

    stabName = glob.glob(finame)
    stabName = sorted(stabName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    stabName = stabName[0]
    nowStr = stabName[-17:-4]
    nowStr_allMice.append(nowStr)


    #### set svmStab mat file name that contains behavioral and class accuracy vars
    finame = os.path.join(fname, 'svm_stabilityBehCA_*.mat')

    stabBehName = glob.glob(finame)
    stabBehName = sorted(stabBehName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    stabBehName = stabBehName[0]




    #%% Load stability vars

    Data = scio.loadmat(stabName) #, variable_names=['eventI_ds_allDays'])   
    
    angleInh_aligned = Data.pop('angleInh_aligned')
    angleExc_aligned = Data.pop('angleExc_aligned')
    angleAllExc_aligned = Data.pop('angleAllExc_aligned')
    angleExc_excsh_aligned = Data.pop('angleExc_excsh_aligned')
    angleExc_excsh_aligned_avExcsh = Data.pop('angleExc_excsh_aligned_avExcsh')
    angleExc_excsh_aligned_sdExcsh = Data.pop('angleExc_excsh_aligned_sdExcsh')
    angleInhS_aligned_avSh = Data.pop('angleInhS_aligned_avSh')
    angleExcS_aligned_avSh = Data.pop('angleExcS_aligned_avSh')
    angleAllExcS_aligned_avSh = Data.pop('angleAllExcS_aligned_avSh')
    angleExc_excshS_aligned_avSh = Data.pop('angleExc_excshS_aligned_avSh')
    angleExc_excshS_aligned_avExcsh = Data.pop('angleExc_excshS_aligned_avExcsh')
    angleExc_excshS_aligned_sdExcsh = Data.pop('angleExc_excshS_aligned_sdExcsh')
    sigAngInh = Data.pop('sigAngInh')
    sigAngExc = Data.pop('sigAngExc')
    sigAngAllExc = Data.pop('sigAngAllExc')
    sigAngExc_excsh = Data.pop('sigAngExc_excsh')
    sigAngExc_excsh_avExcsh = Data.pop('sigAngExc_excsh_avExcsh')
    sigAngExc_excsh_sdExcsh = Data.pop('sigAngExc_excsh_sdExcsh')
    stabScoreInh0_data = Data.pop('stabScoreInh0_data')
    stabScoreExc0_data = Data.pop('stabScoreExc0_data')
    stabScoreAllExc0_data = Data.pop('stabScoreAllExc0_data')
    stabScoreExc_excsh0_data = Data.pop('stabScoreExc_excsh0_data')
    stabScoreExc_excsh0_data_avExcsh = Data.pop('stabScoreExc_excsh0_data_avExcsh')
    stabScoreExc_excsh0_data_sdExcsh = Data.pop('stabScoreExc_excsh0_data_sdExcsh')
    stabScoreInh0_shfl = Data.pop('stabScoreInh0_shfl')
    stabScoreExc0_shfl = Data.pop('stabScoreExc0_shfl')
    stabScoreAllExc0_shfl = Data.pop('stabScoreAllExc0_shfl')
    stabScoreExc_excsh0_shfl = Data.pop('stabScoreExc_excsh0_shfl')
    stabScoreExc_excsh0_shfl_avExcsh = Data.pop('stabScoreExc_excsh0_shfl_avExcsh')
    stabScoreExc_excsh0_shfl_sdExcsh = Data.pop('stabScoreExc_excsh0_shfl_sdExcsh')
    stabScoreInh0 = Data.pop('stabScoreInh0')
    stabScoreExc0 = Data.pop('stabScoreExc0')
    stabScoreAllExc0 = Data.pop('stabScoreAllExc0')
    stabScoreExc_excsh0 = Data.pop('stabScoreExc_excsh0')
    stabScoreExc_excsh0_avExcsh = Data.pop('stabScoreExc_excsh0_avExcsh')
    stabScoreExc_excsh0_sdExcsh = Data.pop('stabScoreExc_excsh0_sdExcsh')
    stabScoreInh1 = Data.pop('stabScoreInh1')
    stabScoreExc1 = Data.pop('stabScoreExc1')
    stabScoreAllExc1 = Data.pop('stabScoreAllExc1')
    stabScoreExc_excsh1 = Data.pop('stabScoreExc_excsh1')
    stabScoreExc_excsh1_avExcsh = Data.pop('stabScoreExc_excsh1_avExcsh')
    stabScoreExc_excsh1_sdExcsh = Data.pop('stabScoreExc_excsh1_sdExcsh')
    angInh_av = Data.pop('angInh_av')
    angExc_av = Data.pop('angExc_av')
    angAllExc_av = Data.pop('angAllExc_av')
    angExc_excsh_av = Data.pop('angExc_excsh_av')
    angInhS_av = Data.pop('angInhS_av')
    angExcS_av = Data.pop('angExcS_av')
    angAllExcS_av = Data.pop('angAllExcS_av')
    angExc_excshS_av = Data.pop('angExc_excshS_av')
    sigAngInh_av = Data.pop('sigAngInh_av')
    sigAngExc_av = Data.pop('sigAngExc_av')
    sigAngAllExc_av = Data.pop('sigAngAllExc_av')
    sigAngExc_excsh_av = Data.pop('sigAngExc_excsh_av')
    stabScoreInh0_data_av = Data.pop('stabScoreInh0_data_av').flatten()
    stabScoreExc0_data_av = Data.pop('stabScoreExc0_data_av').flatten()
    stabScoreAllExc0_data_av = Data.pop('stabScoreAllExc0_data_av').flatten()
    stabScoreExc_excsh0_data_av = Data.pop('stabScoreExc_excsh0_data_av').flatten()
    stabScoreInh0_shfl_av = Data.pop('stabScoreInh0_shfl_av').flatten()
    stabScoreExc0_shfl_av = Data.pop('stabScoreExc0_shfl_av').flatten()
    stabScoreAllExc0_shfl_av = Data.pop('stabScoreAllExc0_shfl_av').flatten()
    stabScoreExc_excsh0_shfl_av = Data.pop('stabScoreExc_excsh0_shfl_av').flatten()
    stabScoreInh0_av = Data.pop('stabScoreInh0_av').flatten()
    stabScoreExc0_av = Data.pop('stabScoreExc0_av').flatten()
    stabScoreAllExc0_av = Data.pop('stabScoreAllExc0_av').flatten()
    stabScoreExc_excsh0_av = Data.pop('stabScoreExc_excsh0_av').flatten()
    stabScoreInh1_av = Data.pop('stabScoreInh1_av').flatten()
    stabScoreExc1_av = Data.pop('stabScoreExc1_av').flatten()
    stabScoreAllExc1_av = Data.pop('stabScoreAllExc1_av').flatten()
    stabScoreExc_excsh1_av = Data.pop('stabScoreExc_excsh1_av').flatten()
    corr_hr_lr = Data.pop('corr_hr_lr')
    eventI_allDays = Data.pop('eventI_allDays').flatten()
    eventI_ds_allDays = Data.pop('eventI_ds_allDays').flatten()
    
    ####### Load beh and CA vars
    Data = scio.loadmat(stabBehName)
    
    behCorr_all = Data.pop('behCorr_all').flatten() # the following comment is for the above mat file: I didnt save _all vars... so what is saved is only for one day! ... have to reset these 3 vars again here!
    behCorrHR_all = Data.pop('behCorrHR_all').flatten()
    behCorrLR_all = Data.pop('behCorrLR_all').flatten()
    classAccurTMS_inh = Data.pop('classAccurTMS_inh') # the following comment is for the above mat file: there was also problem in setting these vars, so you need to reset these 3 vars here
    classAccurTMS_exc = Data.pop('classAccurTMS_exc')
    classAccurTMS_allExc = Data.pop('classAccurTMS_allExc')


         
    #%%
    ######################## PLOTS of each mouse ########################
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    if doPlotsEachMouse==1:
        
        mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
        print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
        print np.array(days)[mn_corr < thTrained]
        
        numGoodDays = sum(mn_corr>=thTrained)    
        numOrigDays = numDaysAll[im].astype(int)
        
        dayinds = np.arange(numOrigDays)
        dayinds = np.delete(dayinds, np.argwhere(mn_corr < thTrained))

    
        #%% Set time_aligned
       
        nPreMin = np.nanmin(eventI_ds_allDays[mn_corr >= thTrained]).astype('int') # number of frames before the common eventI, also the index of common eventI.     
        totLen = classAccurTMS_allExc.shape[1] #nPreMin + nPostMin +1        
    
        # Get downsampled time trace, without using the non-downsampled eventI
        # you need nPreMin and totLen
        a = frameLength*np.arange(-regressBins*nPreMin,0)
        b = frameLength*np.arange(0,regressBins*(totLen-nPreMin))
        aa = np.mean(np.reshape(a,(regressBins,nPreMin), order='F'), axis=0)
        bb = np.mean(np.reshape(b,(regressBins,totLen-nPreMin), order='F'), axis=0)
        time_aligned = np.concatenate((aa,bb))


   
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        #%% Plot min number of HR/LR trials per day
        
        plt.figure(figsize=(4.5,3))
        plt.plot(mn_corr)
        plt.xlabel('Days')
        plt.ylabel('Trial number (min HR,LR)')
        makeNicePlots(plt.gca())
        
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnow) #,mousename)       
            daysnew = (np.array(days))[dayinds]
            if chAl==1:
                dd = 'chAl_ntrs_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_ntrs_' + daysnew[0][0:6] + '-to-' + daysnew[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0]) 
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            
            
        #%% Plot Angles averaged across days for data, shfl, shfl-data (plot angle between decoders at different time points)        
        
        fna = days[0][0:6] + '-to-' + days[-1][0:6]
        
        plotAngsAll(nPreMin, time_aligned, angInh_av, angExc_av, angAllExc_av, angExc_excsh_av, angInhS_av, angExcS_av, angAllExcS_av, angExc_excshS_av, fna, nowStr)

        '''
#        totLen = len(time_aligned) # 
        step = 4
        x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
                   np.arange(np.argwhere(time_aligned>=0)[0], totLen, step))))).astype(int)
        #x = np.arange(0,totLen,step)
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
        if doExcSamps:
            ex = angExc_excsh_av
        else:
            ex = angExc_av        
        plt.figure(figsize=(8,8))
        cmin = np.floor(np.min([np.nanmin(angInh_av), np.nanmin(ex), np.nanmin(angAllExc_av)]))
        cmax = 90
        cmap = 'jet_r' # lower angles: more aligned: red
        cblab = 'Angle between decoders'
        lab = 'inh (data)'; top = angInh_av; plt.subplot(221); plotAng(top, lab, cmin, cmax, cmap, cblab)
        lab = 'exc (data)'; top = ex; plt.subplot(223); plotAng(top, lab, cmin, cmax, cmap, cblab)
        lab = labAll+' (data)'; top = angAllExc_av; plt.subplot(222); plotAng(top, lab, cmin, cmax, cmap, cblab)
        plt.subplots_adjust(hspace=.2, wspace=.3)
        
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnow) #,mousename)       
            if chAl==1:
                dd = 'chAl_anglesAveDays_inhExcAllExc_data_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_anglesAveDays_inhExcAllExc_data_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
                
        
        # shfl
        if doExcSamps:
            ex = angExc_excshS_av
        else:
            ex = angExcS_av        
        plt.figure(figsize=(8,8))
        cmin = np.floor(np.min([np.nanmin(angInhS_av), np.nanmin(ex), np.nanmin(angAllExcS_av)]))
        cmax = 90
        cmap = 'jet_r' # lower angles: more aligned: red
        cblab = 'Angle between decoders'
        lab = 'inh (shfl)'; top = angInhS_av; plt.subplot(221); plotAng(top, lab, cmin, cmax, cmap, cblab)
        lab = 'exc (shfl)'; top = ex; plt.subplot(223); plotAng(top, lab, cmin, cmax, cmap, cblab)
        lab = labAll+' (shfl)'; top = angAllExcS_av; plt.subplot(222); plotAng(top, lab, cmin, cmax, cmap, cblab)
        plt.subplots_adjust(hspace=.2, wspace=.3)
        
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnow) #,mousename)       
            if chAl==1:
                dd = 'chAl_anglesAveDays_inhExcAllExc_shfl_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_anglesAveDays_inhExcAllExc_shfl_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
        
        
        ##### Most useful plot:
        # shfl-real 
        if doExcSamps:
            ex = angExc_excshS_av - angExc_excsh_av
        else:
            ex = angExcS_av - angExc_av         
        plt.figure(figsize=(8,8))
        cmin = np.floor(np.min([np.nanmin(angInhS_av-angInh_av), np.nanmin(ex), np.nanmin(angAllExcS_av-angAllExc_av)]))
        cmax = np.floor(np.max([np.nanmax(angInhS_av-angInh_av), np.nanmax(ex), np.nanmax(angAllExcS_av-angAllExc_av)]))
        cmap = 'jet' # larger value: lower angle for real: more aligned: red
        cblab = 'Alignment rel. shuffle'
        lab = 'inh (shfl-data)'; 
        top = angInhS_av - angInh_av; 
        top[np.triu_indices(len(top),k=0)] = np.nan # plot only the lower triangle (since values are rep)
        plt.subplot(221); plotAng(top, lab, cmin, cmax, cmap, cblab)
        lab = labAll+'exc (shfl-data)'; 
        top = ex; 
        top[np.triu_indices(len(top),k=0)] = np.nan
        plt.subplot(223); plotAng(top, lab, cmin, cmax, cmap, cblab)
        lab = 'allExc (shfl-data)'; 
        top = angAllExcS_av - angAllExc_av; 
        top[np.triu_indices(len(top),k=0)] = np.nan
        plt.subplot(222); plotAng(top, lab, cmin, cmax, cmap, cblab)
        
        # last subplot: inh(shfl-real) - exc(shfl-real)
        cmin = np.nanmin((angInhS_av - angInh_av) - (ex)); 
        cmax = np.nanmax((angInhS_av - angInh_av) - (ex))
        cmap = 'jet' # more diff: higher inh: lower angle for inh: inh more aligned: red
        cblab = 'Inh-Exc: angle between decoders'
        lab = 'inh-exc'; 
        top = ((angInhS_av - angInh_av) - (ex)); 
        top[np.triu_indices(len(top),k=0)] = np.nan
        plt.subplot(224); plotAng(top, lab, cmin, cmax, cmap, cblab)
        
        # Is inh and exc stability different? 
        # for each population (inh,exc) subtract shuffle-averaged angles from real angles... do ttest across days for each pair of time points
        _,pei = stats.ttest_ind(angleInh_aligned - angleInhS_aligned_avSh , angleExc_aligned - angleExcS_aligned_avSh, axis=-1) 
        pei[np.triu_indices(len(pei),k=0)] = np.nan
        # mark sig points by a dot:
        for f1 in range(totLen):
            for f2 in range(totLen): #np.delete(range(totLen),f1): #
                if pei[f1,f2]<.05:
                    plt.plot(f2,f1, marker='*', color='b', markersize=5)
        
        plt.subplots_adjust(hspace=.2, wspace=.3)
        
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnow) #,mousename)       
            if chAl==1:
                dd = 'chAl_anglesAveDays_inhExcAllExc_dataMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_anglesAveDays_inhExcAllExc_dataMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        '''        
        #%% Plot heatmaps of stability and class accuracy for all days
        
        ### decide which one of the following (for score0: data, shfl, shfl-data) you want to plot below
        '''
        # 0: data
        ssi = stabScoreInh0_data, stabScoreInh1
        sse = stabScoreExc0_data, stabScoreExc1
    #    sse_excsh = stabScoreExc_excsh0, stabScoreExc_excsh1
        sse_excsh = stabScoreExc_excsh0_data_avExcsh, stabScoreExc_excsh1_avExcsh
        ssa = stabScoreAllExc0_data, stabScoreAllExc1        
        ssl = stabLab0_data, stabLab1
        
        # 0: shfl
        ssi = stabScoreInh0_shfl, stabScoreInh1
        sse = stabScoreExc0_shfl, stabScoreExc1
    #    sse_excsh = stabScoreExc_excsh0, stabScoreExc_excsh1
        sse_excsh = stabScoreExc_excsh0_shfl_avExcsh, stabScoreExc_excsh1_avExcsh
        ssa = stabScoreAllExc0_shfl, stabScoreAllExc1        
        ssl = stabLab0_shfl, stabLab1
        '''
        # 0: ave(shfl) - data
        ssi = stabScoreInh0, stabScoreInh1
        sse = stabScoreExc0, stabScoreExc1
    #    sse_excsh = stabScoreExc_excsh0, stabScoreExc_excsh1
        sse_excsh = stabScoreExc_excsh0_avExcsh, stabScoreExc_excsh1_avExcsh
        ssa = stabScoreAllExc0, stabScoreAllExc1        
        ssl = stabLab0, stabLab1
        
        
        ##%%
        ########################################
        #plt.plot(stabScore.T);
        #plt.legend(days, loc='center left', bbox_to_anchor=(1, .7))
        
        step = 4
        x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
                   np.arange(np.argwhere(time_aligned>=0)[0], totLen, step))))).astype(int)
        #cmins,cmaxs, cminc,cmaxc
        asp = 'auto' #2 #
        
        def plotStabScore(top, lab, cmins, cmaxs, cmap='jet', cblab='', ax=plt):
        #    img = ax.imshow(top, cmap)
            img = ax.imshow(top, cmap, vmin=cmins, vmax=cmaxs)
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
            return img
          
          
        #################### stability ####################
        for meas in range(2): #range(3): # which one of the measures below to use            
            stabScoreInh = ssi[meas]
            if doExcSamps:
                stabScoreExc = sse_excsh[meas]
            else:
                stabScoreExc = sse[meas]
            stabScoreAllExc = ssa[meas]
            stabLab = ssl[meas]
             
            _,ps = stats.ttest_ind(stabScoreInh, stabScoreExc) 
            _,pc = stats.ttest_ind(classAccurTMS_inh, classAccurTMS_exc) 
            
            _,psa = stats.ttest_ind(stabScoreInh, stabScoreAllExc) 
            _,pca = stats.ttest_ind(classAccurTMS_inh, classAccurTMS_allExc) 
                         
            cmins = np.floor(np.min([np.nanmin(stabScoreInh), np.nanmin(stabScoreExc), np.nanmin(stabScoreAllExc)]))
            cmaxs = np.floor(np.max([np.nanmax(stabScoreInh), np.nanmax(stabScoreExc), np.nanmax(stabScoreAllExc)]))
    
              
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7,5))
            
            lab = 'inh'
            top = stabScoreInh
            cmap ='jet'
            cblab = stabLab
            ax = axes.flat[0]
            img = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
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
            
            
            lab = labAll
            top = stabScoreAllExc
            cmap ='jet'
            cblab = stabLab
            ax = axes.flat[2]
            img = plotStabScore(top, lab, cmins, cmaxs, cmap, cblab, ax)
            #plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
            #add_colorbar(img)
            pos1 = ax.get_position()
            cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72], adjustable='box-forced')
            #cb_ax = fig.add_axes([0.92, np.prod([pos1.y0,asp]), 0.02, pos1.width])
            cbar = fig.colorbar(img, cax=cb_ax, label=cblab)
            #cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95)
            #fig.colorbar(img, ax=axes.ravel().tolist())
            # Make an axis for the colorbar on the right side
            #from mpl_toolkits.axes_grid1 import make_axes_locatable
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #fig.colorbar(img, cax=cax)
            #### mark sig timepoints (all exc sig diff from inh)
            y=-1; pp = np.full(ps.shape, np.nan); pp[psa<=.05] = y
            ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
            ax.set_aspect(asp)
            #fig.tight_layout()   
            
            plt.subplots_adjust(wspace=.4)
            
            ##%% Save the figure    
            if savefigs:
                d = os.path.join(svmdir+dnow) #,mousename)       
#                daysnew = (np.array(days))[dayinds]
                if meas==0:
                    nn = 'ang'
                elif meas==1:
                    nn = 'nsig'                
                
                if chAl==1:
                    dd = 'chAl_stab_'+nn+'_eachDay_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                else:
                    dd = 'chAl_stab_'+nn+'_eachDay_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)            
                fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        

            #%% P value (inh vs exc as well as inh vs allExc) for stability and class accurr across days for each time bin
            
            plt.figure()
            plt.subplot(221); plt.plot(ps); plt.ylabel('p_stability'); makeNicePlots(plt.gca())
            plt.title('exc vs inh'); 
            plt.xticks(x, np.round(time_aligned[x]).astype(int))
            plt.axhline(y=.05, c='r', lw=1)
            
            plt.subplot(223); plt.plot(pc); plt.ylabel('p_class accur'); makeNicePlots(plt.gca())
            plt.xticks(x, np.round(time_aligned[x]).astype(int))
            plt.axhline(y=.05, c='r', lw=1)
            plt.xlabel('Time (ms')
            
            plt.subplot(222); plt.plot(psa); plt.ylabel('p_stability'); makeNicePlots(plt.gca())
            plt.title('all exc vs inh')
            plt.xticks(x, np.round(time_aligned[x]).astype(int))
            plt.axhline(y=.05, c='r', lw=1)
            yl = plt.gca().get_ylim(); plt.ylim([-.005, yl[1]])
            
            plt.subplot(224); plt.plot(pca); plt.ylabel('p_class accur'); makeNicePlots(plt.gca())
            plt.xticks(x, np.round(time_aligned[x]).astype(int))
            plt.axhline(y=.05, c='r', lw=1)
            plt.xlabel('Time (ms')
            yl = plt.gca().get_ylim(); plt.ylim([-.005, yl[1]])
            
            plt.subplots_adjust(hspace=.4, wspace=.5)
            
            
            ##%% Save the figure    
            if savefigs:
                d = os.path.join(svmdir+dnow) #,mousename)       
#                daysnew = (np.array(days))[dayinds]
                if meas==0:
                    nn = 'ang'
                elif meas==1:
                    nn = 'nsig'    
                
                if chAl==1:
                    dd = 'chAl_stab_'+nn+'_p_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                else:
                    dd = 'chAl_stab_'+nn+'_p_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
                    
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)            
                fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            
            
            #%% stab vs days  
            
            nbin = 1 #3 : Average stability score in the last n bins before the choice, and compare it between exc and inh
            rr = np.arange(nPreMin-nbin, nPreMin)
            ssbefchExc = np.mean(stabScoreExc[:, rr], axis=1)
            ssbefchInh = np.mean(stabScoreInh[:, rr], axis=1)
            ssbefchAllExc = np.mean(stabScoreAllExc[:, rr], axis=1)
            
            plt.figure(figsize=(4.5,3))
            plt.plot(ssbefchAllExc, 'k', label=labAll)
            plt.plot(ssbefchInh, 'r', label='inh')
            plt.plot(ssbefchExc, 'b', label='exc')
            plt.xlabel('Days')
            plt.ylabel(stabLab)
            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
            makeNicePlots(plt.gca())
            
            
            ##%% Save the figure    
            if savefigs:
                d = os.path.join(svmdir+dnow) #,mousename)       
#                daysnew = (np.array(days))[dayinds]
                if meas==0:
                    nn = 'ang'
                elif meas==1:
                    nn = 'nsig'                    
                n = '_aveLast%dbins' %(nbin)

                if chAl==1:
                    dd = 'chAl_stab_'+nn+n+'_inhExc'+labAll+'_'  + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                else:
                    dd = 'chAl_stab_'+nn+n+'_inhExc'+labAll+'_'  + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       

                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)            
                fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
                    
        
        #%% ######################## class accuracy (data - shfl) ####################
        
        cminc = np.floor(np.min([np.nanmin(classAccurTMS_inh), np.nanmin(classAccurTMS_exc), np.nanmin(classAccurTMS_allExc)]))
        cmaxc = np.floor(np.max([np.nanmax(classAccurTMS_inh), np.nanmax(classAccurTMS_exc), np.nanmax(classAccurTMS_allExc)]))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7,5))
        
        lab = 'inh'
        top = classAccurTMS_inh
        cmap ='jet'
        cblab = 'Class accuracy (data-shfl)'
        ax = axes.flat[0]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax)
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
        
        
        lab = labAll
        top = classAccurTMS_allExc
        cmap ='jet'
        cblab = 'Class accuracy (data-shfl)'
        ax = axes.flat[2]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax)
        #plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
        #add_colorbar(img)
        cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
        cbar = fig.colorbar(img, cax=cb_ax, label=cblab)
        #cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95)
        #fig.colorbar(img, ax=axes.ravel().tolist())
        # Make an axis for the colorbar on the right side
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(img, cax=cax)
        y=-1; pp = np.full(ps.shape, np.nan); pp[pca<=.05] = y
        ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
        ax.set_aspect(asp)
        
        plt.subplots_adjust(wspace=.4)
        
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnow) #,mousename)       
#            daysnew = (np.array(days))[dayinds]
            if chAl==1:
                dd = 'chAl_classAcc_eachDay_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_classAcc_eachDay_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
        #%% Average class accuracy in the last n bins before the choice, and compare it between exc and inh
            
#        nbin = 1 #3
        rr = np.arange(nPreMin-nbin, nPreMin)
        ssbefchExc = np.mean(classAccurTMS_exc[:, rr], axis=1)
        ssbefchInh = np.mean(classAccurTMS_inh[:, rr], axis=1)
        ssbefchAllExc = np.mean(classAccurTMS_allExc[:, rr], axis=1)
        
        plt.figure(figsize=(4.5,3))
        plt.plot(ssbefchAllExc, 'k', label=labAll)
        plt.plot(ssbefchInh, 'r', label='inh')
        plt.plot(ssbefchExc, 'b', label='exc')
        plt.xlabel('Days')
        plt.ylabel('Class accuracy (data-shfl)')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
        makeNicePlots(plt.gca())
        
        
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnow) #,mousename)       
#                daysnew = (np.array(days))[dayinds]

            n = '_aveLast%dbins' %(nbin)

            if chAl==1:
                dd = 'chAl_classAcc'+n+'_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_classAcc'+n+'_inhExc'+labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       

            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
                
                

    #%% Keep vars of all mice     
    ####################################################################################################
    ####################################################################################################


    angleInh_aligned_allMice.append(angleInh_aligned) # each mouse: frs x frs x days
    angleExc_aligned_allMice.append(angleExc_aligned)
    angleAllExc_aligned_allMice.append(angleAllExc_aligned)
    # keep vars for individual exc samps
    angleExc_excsh_aligned_allMice.append(angleExc_excsh_aligned) # each mouse: frs x frs x excSamps x days ... we dont save the vars of each excSamp, instead we save their ave and se across excSamps
    angleExc_excsh_aligned_avExcsh_allMice.append(angleExc_excsh_aligned_avExcsh)
    angleExc_excsh_aligned_sdExcsh_allMice.append(angleExc_excsh_aligned_sdExcsh)    
    
    ### shfl
    angleInhS_aligned_avSh_allMice.append(angleInhS_aligned_avSh) # each mouse: frs x frs x days
    angleExcS_aligned_avSh_allMice.append(angleExcS_aligned_avSh)
    angleAllExcS_aligned_avSh_allMice.append(angleAllExcS_aligned_avSh)
    # keep vars for individual exc samps    
    angleExc_excshS_aligned_avSh_allMice.append(angleExc_excshS_aligned_avSh) # each mouse: frs x frs x excSamps x days
    angleExc_excshS_aligned_avExcsh_allMice.append(angleExc_excshS_aligned_avExcsh)
    angleExc_excshS_aligned_sdExcsh_allMice.append(angleExc_excshS_aligned_sdExcsh)
    
    ### significant or not
    sigAngInh_allMice.append(sigAngInh) # each mouse: frs x frs x days
    sigAngExc_allMice.append(sigAngExc)
    sigAngAllExc_allMice.append(sigAngAllExc)
    sigAngExc_excsh_allMice.append(sigAngExc_excsh) # each mouse: frs x frs x excSamps x days
    sigAngExc_excsh_avExcsh_allMice.append(sigAngExc_excsh_avExcsh)
    sigAngExc_excsh_sdExcsh_allMice.append(sigAngExc_excsh_sdExcsh)    
    
    
    ################ measures of stability computed by averaging decoders of each time point with all other time points
    # instab_data
    stabScoreInh0_data_allMice.append(stabScoreInh0_data)  # each mouse: days x frs 
    stabScoreExc0_data_allMice.append(stabScoreExc0_data)
    stabScoreAllExc0_data_allMice.append(stabScoreAllExc0_data)
    stabScoreExc_excsh0_data_allMice.append(stabScoreExc_excsh0_data)  # each mouse: days  x excSamps x frs
    stabScoreExc_excsh0_data_avExcsh_allMice.append(stabScoreExc_excsh0_data_avExcsh)
    stabScoreExc_excsh0_data_sdExcsh_allMice.append(stabScoreExc_excsh0_data_sdExcsh)
    
    # instab_shfl
    stabScoreInh0_shfl_allMice.append(stabScoreInh0_shfl)  # each mouse: days x frs 
    stabScoreExc0_shfl_allMice.append(stabScoreExc0_shfl)
    stabScoreAllExc0_shfl_allMice.append(stabScoreAllExc0_shfl)
    stabScoreExc_excsh0_shfl_allMice.append(stabScoreExc_excsh0_shfl)  # each mouse: days  x excSamps x frs
    stabScoreExc_excsh0_shfl_avExcsh_allMice.append(stabScoreExc_excsh0_shfl_avExcsh)
    stabScoreExc_excsh0_shfl_sdExcsh_allMice.append(stabScoreExc_excsh0_shfl_sdExcsh)
    
    # stab (shfl-data)
    stabScoreInh0_allMice.append(stabScoreInh0)  # each mouse: days x frs 
    stabScoreExc0_allMice.append(stabScoreExc0)
    stabScoreAllExc0_allMice.append(stabScoreAllExc0)
    stabScoreExc_excsh0_allMice.append(stabScoreExc_excsh0)  # each mouse: days  x excSamps x frs
    stabScoreExc_excsh0_avExcsh_allMice.append(stabScoreExc_excsh0_avExcsh)
    stabScoreExc_excsh0_sdExcsh_allMice.append(stabScoreExc_excsh0_sdExcsh)
    
    # num aligned decoder
    stabScoreInh1_allMice.append(stabScoreInh1)  # each mouse: days x frs 
    stabScoreExc1_allMice.append(stabScoreExc1)
    stabScoreAllExc1_allMice.append(stabScoreAllExc1)
    stabScoreExc_excsh1_allMice.append(stabScoreExc_excsh1)   # each mouse: days  x excSamps x frs
    stabScoreExc_excsh1_avExcsh_allMice.append(stabScoreExc_excsh1_avExcsh)
    stabScoreExc_excsh1_sdExcsh_allMice.append(stabScoreExc_excsh1_sdExcsh)    


    ################ ave across days    
    angInh_av_allMice.append(angInh_av) 
    angExc_av_allMice.append(angExc_av)
    angAllExc_av_allMice.append(angAllExc_av)
    angExc_excsh_av_allMice.append(angExc_excsh_av)

    
    angInhS_av_allMice.append(angInhS_av)     
    angExcS_av_allMice.append(angExcS_av)    
    angAllExcS_av_allMice.append(angAllExcS_av)
    angExc_excshS_av_allMice.append(angExc_excshS_av) # frs x frs --> average of excSamp averges across days

    
    sigAngInh_av_allMice.append(sigAngInh_av)
    sigAngExc_av_allMice.append(sigAngExc_av)
    sigAngAllExc_av_allMice.append(sigAngAllExc_av)    
    sigAngExc_excsh_av_allMice.append(sigAngExc_excsh_av)
    
    stabScoreInh0_data_av_allMice.append(stabScoreInh0_data_av)
    stabScoreExc0_data_av_allMice.append(stabScoreExc0_data_av)
    stabScoreAllExc0_data_av_allMice.append(stabScoreAllExc0_data_av)
    stabScoreExc_excsh0_data_av_allMice.append(stabScoreExc_excsh0_data_av)

    stabScoreInh0_shfl_av_allMice.append(stabScoreInh0_shfl_av)
    stabScoreExc0_shfl_av_allMice.append(stabScoreExc0_shfl_av)
    stabScoreAllExc0_shfl_av_allMice.append(stabScoreAllExc0_shfl_av)
    stabScoreExc_excsh0_shfl_av_allMice.append(stabScoreExc_excsh0_shfl_av)
    # shfl-data
    stabScoreInh0_av_allMice.append(stabScoreInh0_av)
    stabScoreExc0_av_allMice.append(stabScoreExc0_av)
    stabScoreAllExc0_av_allMice.append(stabScoreAllExc0_av)
    stabScoreExc_excsh0_av_allMice.append(stabScoreExc_excsh0_av)
     
    stabScoreInh1_av_allMice.append(stabScoreInh1_av)
    stabScoreExc1_av_allMice.append(stabScoreExc1_av)
    stabScoreAllExc1_av_allMice.append(stabScoreAllExc1_av)
    stabScoreExc_excsh1_av_allMice.append(stabScoreExc_excsh1_av)


    ###############
    corr_hr_lr_allMice.append(corr_hr_lr)
    eventI_allDays_allMice.append(eventI_allDays.astype('int'))        
    eventI_ds_allDays_allMice.append(eventI_ds_allDays.astype('int'))        
#    time_trace_all_allMice.append(np.array(time_trace_all))
    
    behCorr_allMice.append(np.array(behCorr_all))
    behCorrHR_allMice.append(np.array(behCorrHR_all))
    behCorrLR_allMice.append(np.array(behCorrLR_all))   
#    numDaysAll = numDaysAll.astype(int)    

    classAccurTMS_inh_allMice.append(classAccurTMS_inh)
    classAccurTMS_exc_allMice.append(classAccurTMS_exc)
    classAccurTMS_allExc_allMice.append(classAccurTMS_allExc)

        
#%%        

angInh_av_allMice = np.array(angInh_av_allMice) # each mouse: frs x frs
angExc_av_allMice = np.array(angExc_av_allMice)
angAllExc_av_allMice = np.array(angAllExc_av_allMice)
angExc_excsh_av_allMice = np.array(angExc_excsh_av_allMice)

angInhS_av_allMice = np.array(angInhS_av_allMice)
angExcS_av_allMice = np.array(angExcS_av_allMice)
angAllExcS_av_allMice = np.array(angAllExcS_av_allMice)
angExc_excshS_av_allMice = np.array(angExc_excshS_av_allMice)

sigAngInh_av_allMice = np.array(sigAngInh_av_allMice)
sigAngExc_av_allMice = np.array(sigAngExc_av_allMice)
sigAngAllExc_av_allMice = np.array(sigAngAllExc_av_allMice)
sigAngExc_excsh_av_allMice = np.array(sigAngExc_excsh_av_allMice)

stabScoreInh0_allMice = np.array(stabScoreInh0_allMice)
stabScoreExc0_allMice = np.array(stabScoreExc0_allMice)
stabScoreAllExc0_allMice = np.array(stabScoreAllExc0_allMice)
stabScoreExc_excsh0_allMice = np.array(stabScoreExc_excsh0_allMice)
stabScoreExc_excsh0_avExcsh_allMice = np.array(stabScoreExc_excsh0_avExcsh_allMice)
stabScoreExc_excsh0_sdExcsh_allMice = np.array(stabScoreExc_excsh0_sdExcsh_allMice)

stabScoreInh1_allMice = np.array(stabScoreInh1_allMice)
stabScoreExc1_allMice = np.array(stabScoreExc1_allMice)
stabScoreAllExc1_allMice = np.array(stabScoreAllExc1_allMice)
stabScoreExc_excsh1_allMice = np.array(stabScoreExc_excsh1_allMice)
stabScoreExc_excsh1_avExcsh_allMice = np.array(stabScoreExc_excsh1_avExcsh_allMice)
stabScoreExc_excsh1_sdExcsh_allMice = np.array(stabScoreExc_excsh1_sdExcsh_allMice)

######
corr_hr_lr_allMice = np.array(corr_hr_lr_allMice)
eventI_ds_allDays_allMice = np.array(eventI_ds_allDays_allMice)
eventI_allDays_allMice = np.array(eventI_allDays_allMice)

behCorr_allMice = np.array(behCorr_allMice)
behCorrHR_allMice = np.array(behCorrHR_allMice)
behCorrLR_allMice = np.array(behCorrLR_allMice)

classAccurTMS_inh_allMice = np.array(classAccurTMS_inh_allMice)
classAccurTMS_exc_allMice = np.array(classAccurTMS_exc_allMice)
classAccurTMS_allExc_allMice = np.array(classAccurTMS_allExc_allMice)



no
#%% 



#%%################ Plots for all mice ################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#%% 

mice2an = np.arange(len(mice)) #np.array([0,1,3]) # which mice to analyze #range(len(mice))
meas = 0  # what measure of stability to use --> 0: ave angle ; 1 : num sig angles
nbin = 1 # how many time bins before the choice to average (on downsampled traces)

from datetime import datetime
nowStrAM = datetime.now().strftime('%y%m%d-%H%M%S')


#%% Define days to be analyzed (Identify days with too few trials (to exclude them!))    

daysGood_allMice = []
dayinds_allMice = []
mn_corr_allMice = []

for im in range(len(mice)):    
    print '________________',mice[im],'________________' 
    mn_corr = np.min(corr_hr_lr_allMice[im], axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.

    print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
    print np.array(days_allMice[im])[mn_corr < thTrained]
    
    numGoodDays = sum(mn_corr>=thTrained)    
    numOrigDays = numDaysAll[im].astype(int)
    
    dayinds = np.arange(numOrigDays)
    dayinds = np.delete(dayinds, np.argwhere(mn_corr < thTrained))

    dayinds_allMice.append(dayinds)
    daysGood_allMice.append(np.array(days_allMice[im])[dayinds])        
    mn_corr_allMice.append(mn_corr)
    
dayinds_allMice = np.array(dayinds_allMice)
daysGood_allMice = np.array(daysGood_allMice)    
mn_corr_allMice = np.array(mn_corr_allMice)


#%% behav vs svm performance
        
dnowb = dnow0#+'/behavior_svm/'
# remove low trial days from beh vars
behCorr_allMice = [behCorr_allMice[im][dayinds_allMice[im]] for im in range(len(mice))]
behCorrHR_allMice = [behCorrHR_allMice[im][dayinds_allMice[im]] for im in range(len(mice))]
behCorrLR_allMice = [behCorrLR_allMice[im][dayinds_allMice[im]] for im in range(len(mice))]



#%% Set time_aligned and nPreMin for aligning traces of all mice

time_aligned_final, nPreMin_final, nPostMin_final, nPreMin_allMice = set_nprepost_allMice(classAccurTMS_exc_allMice, eventI_ds_allDays_allMice, corr_hr_lr_allMice)


#%% now align the traces of all mice

angInh_av_allMice_al = alTrace_frfr(angInh_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # nMice x frs x frs
angExc_av_allMice_al = alTrace_frfr(angExc_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angAllExc_av_allMice_al = alTrace_frfr(angAllExc_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angExc_excsh_av_allMice_al = alTrace_frfr(angExc_excsh_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)

angInhS_av_allMice_al = alTrace_frfr(angInhS_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angExcS_av_allMice_al = alTrace_frfr(angExcS_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angAllExcS_av_allMice_al = alTrace_frfr(angAllExcS_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angExc_excshS_av_allMice_al = alTrace_frfr(angExc_excshS_av_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)


#%% Average above matrices across mice

angInh_av_allMice_al_av = np.mean(angInh_av_allMice_al, axis=0) # frs x frs
angExc_av_allMice_al_av = np.mean(angExc_av_allMice_al, axis=0)
angAllExc_av_allMice_al_av = np.mean(angAllExc_av_allMice_al, axis=0)
angExc_excsh_av_allMice_al_av = np.mean(angExc_excsh_av_allMice_al, axis=0)

angInhS_av_allMice_al_av = np.mean(angInhS_av_allMice_al, axis=0)
angExcS_av_allMice_al_av = np.mean(angExcS_av_allMice_al, axis=0)
angAllExcS_av_allMice_al_av = np.mean(angAllExcS_av_allMice_al, axis=0)
angExc_excshS_av_allMice_al_av = np.mean(angExc_excshS_av_allMice_al, axis=0)



#%% Plot average of angles across mice

fna = '_'.join(mice) # (np.array(mice)[mice2an_all])

plotAngsAll(nPreMin_final, time_aligned_final, angInh_av_allMice_al_av, angExc_av_allMice_al_av, angAllExc_av_allMice_al_av, angExc_excsh_av_allMice_al_av, angInhS_av_allMice_al_av, angExcS_av_allMice_al_av, angAllExcS_av_allMice_al_av, angExc_excshS_av_allMice_al_av)




#%% Align stab scores of all days of each mouse on the final common eventI (nPreMin_final) so all traces of all mice are aligned.

def alAllDays4AllMice(x):
    x_al = []
    for im in range(len(mice)):
        ev = nPreMin_allMice[im]*np.full((x[im].shape[0]),1,dtype=int)
        x_al.append(np.array(alTrace(x[im], ev, nPreMin_final, nPostMin_final))) # days x miceAlignedFrames
    x_al = np.array(x_al)
    return x_al


# meas 0 (ave angs)
stabScoreInh0_allMice_al = alAllDays4AllMice(stabScoreInh0_allMice)
stabScoreExc0_allMice_al = alAllDays4AllMice(stabScoreExc0_allMice)
stabScoreAllExc0_allMice_al = alAllDays4AllMice(stabScoreAllExc0_allMice)
stabScoreExc_excsh0_allMice_al = alAllDays4AllMice([np.transpose(stabScoreExc_excsh0_allMice[im], (0,2,1)) for im in range(len(mice))]) # days x miceAlignedFrames x excSamps  (remember input was # days x excSamps x miceAlignedFrames, but you transposed it here)
stabScoreExc_excsh0_avExcsh_allMice_al = alAllDays4AllMice(stabScoreExc_excsh0_avExcsh_allMice)
stabScoreExc_excsh0_sdExcsh_allMice_al = alAllDays4AllMice(stabScoreExc_excsh0_sdExcsh_allMice)


# meas 1 (num sig angs)
stabScoreInh1_allMice_al = alAllDays4AllMice(stabScoreInh1_allMice)
stabScoreExc1_allMice_al = alAllDays4AllMice(stabScoreExc1_allMice)
stabScoreAllExc1_allMice_al = alAllDays4AllMice(stabScoreAllExc1_allMice)
stabScoreExc_excsh1_allMice_al = alAllDays4AllMice([np.transpose(stabScoreExc_excsh1_allMice[im], (0,2,1)) for im in range(len(mice))]) # days x miceAlignedFrames x excSamps  (remember input was # days x excSamps x miceAlignedFrames, but you transposed it here)
stabScoreExc_excsh1_avExcsh_allMice_al = alAllDays4AllMice(stabScoreExc_excsh1_avExcsh_allMice)
stabScoreExc_excsh1_sdExcsh_allMice_al = alAllDays4AllMice(stabScoreExc_excsh1_sdExcsh_allMice)


if doExcSamps:
    stabScoreExc0_al = stabScoreExc_excsh0_avExcsh_allMice_al
    stabScoreExc1_al = stabScoreExc_excsh1_avExcsh_allMice_al
    # try only one exc samp
    '''
    iexc = rng.permutation(stabScoreExc_excsh1_allMice_al[im].shape[2])[0] # pick a random exc samp
    stabScoreExc0_al = [stabScoreExc_excsh0_allMice_al[im][:,:,iexc] for im in range(len(mice))]
    stabScoreExc1_al = [stabScoreExc_excsh1_allMice_al[im][:,:,iexc] for im in range(len(mice))]
    '''
else:
    stabScoreExc0_al = stabScoreExc0_allMice_al
    stabScoreExc1_al = stabScoreExc1_allMice_al



#%% Align angles for each mouse (individual days) fr x fr x day

angleInh_aligned_allMice_al = alTrace_frfr(angleInh_aligned_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # each mouse: frs x frs x days
angleExc_aligned_allMice_al = alTrace_frfr(angleExc_aligned_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angleAllExc_aligned_allMice_al = alTrace_frfr(angleAllExc_aligned_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angleExc_excsh_aligned_allMice_al = alTrace_frfr(angleExc_excsh_aligned_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # each mouse: frs x frs x excSamps x days
angleExc_excsh_aligned_avExcsh_allMice_al = alTrace_frfr(angleExc_excsh_aligned_avExcsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angleExc_excsh_aligned_sdExcsh_allMice_al = alTrace_frfr(angleExc_excsh_aligned_sdExcsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)

### shfl
angleInhS_aligned_avSh_allMice_al = alTrace_frfr(angleInhS_aligned_avSh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # each mouse: frs x frs x days
angleExcS_aligned_avSh_allMice_al = alTrace_frfr(angleExcS_aligned_avSh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angleAllExcS_aligned_avSh_allMice_al = alTrace_frfr(angleAllExcS_aligned_avSh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angleExc_excshS_aligned_avSh_allMice_al = alTrace_frfr(angleExc_excshS_aligned_avSh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # each mouse: frs x frs x excSamps x days
angleExc_excshS_aligned_avExcsh_allMice_al = alTrace_frfr(angleExc_excshS_aligned_avExcsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
angleExc_excshS_aligned_sdExcsh_allMice_al = alTrace_frfr(angleExc_excshS_aligned_sdExcsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)


#%% Align sig matrices for each mouse

sigAngInh_allMice_al = alTrace_frfr(sigAngInh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # each mouse: frs x frs x days
sigAngExc_allMice_al = alTrace_frfr(sigAngExc_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
sigAngAllExc_allMice_al = alTrace_frfr(sigAngAllExc_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
sigAngExc_excsh_allMice_al = alTrace_frfr(sigAngExc_excsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # each mouse: frs x frs x excSamps x days
sigAngExc_excsh_avExcsh_allMice_al = alTrace_frfr(sigAngExc_excsh_avExcsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
sigAngExc_excsh_sdExcsh_allMice_al = alTrace_frfr(sigAngExc_excsh_sdExcsh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)

    
#%% Set non-sig angles to nan so we get angles only for those decoders that are significantly different from shuffled control

def setAngOnlySig(xang, xsig):
    
    x_onlySig = []    
    for im in range(len(mice)):
        angSigs = xang[im] + 0
        angSigs[xsig[im] == 0] = np.nan # set non-sig angles to nan
    
        x_onlySig.append(angSigs) # each mouse: frs x frs x days
    
    return x_onlySig


# data
angleInh_aligned_allMice_onlySig_data = setAngOnlySig(angleInh_aligned_allMice_al, sigAngInh_allMice_al)  # each mouse: frs x frs x days
angleExc_aligned_allMice_onlySig_data = setAngOnlySig(angleExc_aligned_allMice_al, sigAngExc_allMice_al)   # each mouse: frs x frs x days
angleAllExc_aligned_allMice_onlySig_data = setAngOnlySig(angleAllExc_aligned_allMice_al, sigAngAllExc_allMice_al)  
angleExc_excsh_aligned_allMice_onlySig_data = setAngOnlySig(angleExc_excsh_aligned_allMice_al, sigAngExc_excsh_allMice_al) # each mouse: frs x frs x excSamps x days
angleExc_excsh_aligned_avExcsh_allMice_onlySig_data = setAngOnlySig(angleExc_excsh_aligned_avExcsh_allMice_al, sigAngExc_excsh_avExcsh_allMice_al)
angleExc_excsh_aligned_sdExcsh_allMice_onlySig_data = setAngOnlySig(angleExc_excsh_aligned_sdExcsh_allMice_al, sigAngExc_excsh_sdExcsh_allMice_al)

# shfl - data
import operator
angleInh_aligned_allMice_onlySig = setAngOnlySig(map(operator.sub, angleInhS_aligned_avSh_allMice_al , angleInh_aligned_allMice_al), sigAngInh_allMice_al)  # each mouse: frs x frs x days
angleExc_aligned_allMice_onlySig = setAngOnlySig(map(operator.sub, angleExcS_aligned_avSh_allMice_al , angleExc_aligned_allMice_al), sigAngExc_allMice_al)   # each mouse: frs x frs x days
angleAllExc_aligned_allMice_onlySig = setAngOnlySig(map(operator.sub, angleAllExcS_aligned_avSh_allMice_al , angleAllExc_aligned_allMice_al), sigAngAllExc_allMice_al)  
angleExc_excsh_aligned_allMice_onlySig = setAngOnlySig(map(operator.sub, angleExc_excshS_aligned_avSh_allMice_al , angleExc_excsh_aligned_allMice_al), sigAngExc_excsh_allMice_al) # each mouse: frs x frs x excSamps x days
angleExc_excsh_aligned_avExcsh_allMice_onlySig = setAngOnlySig(map(operator.sub, angleExc_excshS_aligned_avExcsh_allMice_al , angleExc_excsh_aligned_avExcsh_allMice_al), sigAngExc_excsh_avExcsh_allMice_al)
angleExc_excsh_aligned_sdExcsh_allMice_onlySig = setAngOnlySig(map(operator.sub, angleExc_excshS_aligned_sdExcsh_allMice_al , angleExc_excsh_aligned_sdExcsh_allMice_al), sigAngExc_excsh_sdExcsh_allMice_al)


#%% For each mouse average onlySig angles across days

angleInh_aligned_allMice_onlySig_avD = [np.nanmean(angleInh_aligned_allMice_onlySig[im], axis=-1) for im in range(len(mice))] # mice x frs x frs
angleExc_aligned_allMice_onlySig_avD = [np.nanmean(angleExc_aligned_allMice_onlySig[im], axis=-1) for im in range(len(mice))]
angleAllExc_aligned_allMice_onlySig_avD = [np.nanmean(angleAllExc_aligned_allMice_onlySig[im], axis=-1) for im in range(len(mice))]
angleExc_excsh_aligned_allMice_onlySig_avD = [np.nanmean(angleExc_excsh_aligned_allMice_onlySig[im], axis=-1) for im in range(len(mice))] # mice x frs x frs x excSamps
angleExc_excsh_aligned_avExcsh_allMice_onlySig_avD = [np.nanmean(angleExc_excsh_aligned_avExcsh_allMice_onlySig[im], axis=-1) for im in range(len(mice))]
angleExc_excsh_aligned_sdExcsh_allMice_onlySig_avD = [np.nanmean(angleExc_excsh_aligned_sdExcsh_allMice_onlySig[im], axis=-1) for im in range(len(mice))]


#%% Average above matrices across mice

angleInh_aligned_allMice_onlySig_av = np.nanmean(angleInh_aligned_allMice_onlySig_avD, axis=0) # frs x frs
angleExc_aligned_allMice_onlySig_av = np.nanmean(angleExc_aligned_allMice_onlySig_avD, axis=0)
angleAllExc_aligned_allMice_onlySig_av = np.nanmean(angleAllExc_aligned_allMice_onlySig_avD, axis=0)
angleExc_excsh_aligned_allMice_onlySig_av = np.nanmean(angleExc_excsh_aligned_allMice_onlySig_avD, axis=0) # frs x frs x excSamps
angleExc_excsh_aligned_avExcsh_allMice_onlySig_av = np.nanmean(angleExc_excsh_aligned_avExcsh_allMice_onlySig_avD, axis=0)
angleExc_excsh_aligned_sdExcsh_allMice_onlySig_av = np.nanmean(angleExc_excsh_aligned_sdExcsh_allMice_onlySig_avD, axis=0)


#%% Now plot these onlySig ang averages computed above

def plotAngOnlySig(nPreMin, time_aligned, angInh_av, angExc_av, angAllExc_av, angExc_excsh_av, fna, nowStrAM):

    totLen = len(time_aligned) 
    step = 4
    x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
               np.arange(np.argwhere(time_aligned>=0)[0], totLen, step))))).astype(int)
    #x = np.arange(0,totLen,step)
    
    ######## Plot angles, averaged across days
    # data
    if doExcSamps:
        ex = angExc_excsh_av
    else:
        ex = angExc_av        
        
    plt.figure(figsize=(8,8))
    cmin = np.floor(np.min([np.nanmin(angInh_av), np.nanmin(ex), np.nanmin(angAllExc_av)]))
    cmax = 90
    cmap = 'jet_r' # lower angles: more aligned: red
    cblab = 'Alignment rel. shuffle' #'Angle between decoders'
    lab = 'inh (data)'; top = angInh_av; plt.subplot(221); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = 'exc (data)'; top = ex; plt.subplot(223); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    lab = labAll+' (data)'; top = angAllExc_av; plt.subplot(222); plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    plt.subplots_adjust(hspace=.2, wspace=.3)
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
        if chAl==1:
            dd = 'chAl_anglesAveDays_inhExcAllExc_onlySig_' + fna + '_' + nowStr
        else:
            dd = 'stAl_anglesAveDays_inhExcAllExc_onlySig_' + fna + '_' + nowStr       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            

#%%            

fna = '_'.join(mice) # (np.array(mice)[mice2an_all])

plotAngOnlySig(nPreMin_final, time_aligned_final, angleInh_aligned_allMice_onlySig_av, angleExc_aligned_allMice_onlySig_av, angleAllExc_aligned_allMice_onlySig_av, angleExc_excsh_aligned_avExcsh_allMice_onlySig_av, fna, nowStrAM)
    

#%% Set meas 2

######## data : for each timebin average angle of the decoder with other decoders (of all other timepoints)

stabScoreInh2_allMice_al = []
for im in range(len(mice)): # days x frs
    stabScoreInh2_allMice_al.append(np.array([np.nanmean(angleInh_aligned_allMice_onlySig[im][:,:,iday] , axis=0) for iday in range(angleInh_aligned_allMice_onlySig[im].shape[2])])) # days x frs

stabScoreExc2_allMice_al = []
for im in range(len(mice)): # days x frs
    stabScoreExc2_allMice_al.append(np.array([np.nanmean(angleExc_aligned_allMice_onlySig[im][:,:,iday] , axis=0) for iday in range(angleExc_aligned_allMice_onlySig[im].shape[2])])) # days x frs

stabScoreAllExc2_allMice_al = []
for im in range(len(mice)): # days x frs
    stabScoreAllExc2_allMice_al.append(np.array([np.nanmean(angleAllExc_aligned_allMice_onlySig[im][:,:,iday] , axis=0) for iday in range(angleAllExc_aligned_allMice_onlySig[im].shape[2])])) # days x frs


stabScoreExc_excsh2_allMice_al = []
for im in range(len(mice)):  # days x excSamps x frs
    nd = angleExc_excsh_aligned_allMice_onlySig[im].shape[-1]
    numExcSamples = angleExc_excsh_aligned_allMice_onlySig[im].shape[-2]
    nf = angleExc_excsh_aligned_allMice_onlySig[im].shape[0]
    a = np.full((nd, numExcSamples, nf), np.nan) # days x excSamps x frs
    for iday in range(nd):
        a[iday,:,:] = np.array([np.nanmean(angleExc_excsh_aligned_allMice_onlySig[im][:,:,iesh,iday] , axis=0) for iesh in range(numExcSamples)]) 
    stabScoreExc_excsh2_allMice_al.append(a)
# average and std across excSamps for each day  (and each mouse)   
stabScoreExc_excsh2_avExcsh = [np.nanmean(stabScoreExc_excsh2_allMice_al[im] , axis=1) for im in range(len(mice))] # days x frs    
stabScoreExc_excsh2_sdExcsh = [np.nanstd(stabScoreExc_excsh2_allMice_al[im], axis=1) for im in range(len(mice))] # days x frs  
    
stabLab2_onlySig = 'Stability (only sig angles: degrees away from full mis-alignment)' 



if doExcSamps:
    stabScoreExc2_al = stabScoreExc_excsh2_avExcsh
else:
    stabScoreExc2_al = stabScoreExc2_allMice_al
    
    
    
#%%
meas = 1

if meas==0:
    nn = 'ang'
elif meas==1:
    nn = 'nsig'            
elif meas==2:
    nn = 'angOfSig'

    
#%% average of mice time course: Plot time course of whatever var u r interested in for exc and inh: take average across days for each mouse; then plot ave+/-se across mice

colors = 'b','r', 'k' 
labs = 'exc', 'inh', 'allN' 
#lab = 'Stability (deg.)'

daysDim = 0
doErrbar = 0
alph = .5
fnam = 'timeCourse_'+nn +'_aveMice_aveDays_inhExc'+labAll+'_'  + '_'.join(mice) + '_' + nowStrAM

if meas==0:
    stab = stabScoreExc0_al, stabScoreInh0_allMice_al, stabScoreAllExc0_allMice_al
    lab = stabLab0
elif meas==1:
    stab = stabScoreExc1_al, stabScoreInh1_allMice_al, stabScoreAllExc1_allMice_al
    lab = stabLab1
elif meas==2:
    stab = stabScoreExc2_al, stabScoreInh2_allMice_al, stabScoreAllExc2_allMice_al
    lab = stabLab2_onlySig
       

plt.figure(figsize=(3,2))
plotTimecourse_avMice(time_aligned_final, stab, daysDim, colors, labs, lab, alph, doErrbar, dnow0, fnam)       

       
       
#%% each mouse time course: Plot time course of whatever var u r interested in for exc and inh      

daysDim = 0       

linstyl = '-'
alph = 1

for im in range(len(mice)):

    if meas==0:
        stab = stabScoreExc0_al[im], stabScoreInh0_allMice_al[im], stabScoreAllExc0_allMice_al[im]
    elif meas==1:
        stab = stabScoreExc1_al[im], stabScoreInh1_allMice_al[im], stabScoreAllExc1_allMice_al[im]
    elif meas==2:
        stab = stabScoreExc2_al[im], stabScoreInh2_allMice_al[im], stabScoreAllExc2_allMice_al[im]
        
    days = daysGood_allMice[im]
    dnow = dnow0+mice[im]+'/'
    fnam = 'timeCourse_'+nn +'_aveDays_inhExc'+labAll+'_'  + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr_allMice[im]
    
    plt.figure(figsize=(3,2))            
    plotTimecourse_eachMouse(time_aligned_final, stab, daysDim, colors, labs, lab, linstyl, alph, dnow, fnam)#, col='b', lab='exc')#, linstyl=linstyls[trtype], alph=alphas[trtype]) # avFRexc: alignedFrames x days (each element was averaged across all neurons and trials))
           
       



#%% Set averages across mice: stability vs. day, for exc,inh,allN

def plotssca(ssbefchExc_allMice, ssbefchInh_allMice, ssbefchAllExc_allMice, mnLenDays, sl, savl, mice2an_manyDays, mice2an_all):
    
    nowStr = nowStrAM
    ############ get data for the first mnLenDays days
    sseAMf = [ssbefchExc_allMice[i][0:mnLenDays] for i in mice2an_manyDays] # nMice x mnLenDays
    ssiAMf = [ssbefchInh_allMice[i][0:mnLenDays] for i in mice2an_manyDays]
    ssaAMf = [ssbefchAllExc_allMice[i][0:mnLenDays] for i in mice2an_manyDays]
    
    # average stability across mice from day 1 to mnLenDays
    sseAv_f = np.mean(sseAMf, axis=0)
    ssiAv_f = np.mean(ssiAMf, axis=0)
    ssaAv_f = np.mean(ssaAMf, axis=0)
    sseSd_f = np.std(sseAMf, axis=0)/np.sqrt(np.shape(sseAMf)[0])
    ssiSd_f = np.std(ssiAMf, axis=0)/np.sqrt(np.shape(sseAMf)[0])
    ssaSd_f = np.std(ssaAMf, axis=0)/np.sqrt(np.shape(sseAMf)[0])
    _,pei_f = stats.ttest_ind(sseAv_f, ssiAv_f)
    # pool all days of all mice, then compute p for inh vs exc
    _,p = stats.ttest_ind(np.reshape(sseAMf,(-1,)), np.reshape(ssiAMf,(-1,)))
    _,pei_f_ed = stats.ttest_ind(sseAMf, ssiAMf, axis=0)
    
    
    
    ############ get data for the last mnLenDays days
    sseAMl = [ssbefchExc_allMice[i][-mnLenDays:] for i in mice2an_manyDays] # nMice x mnLenDays
    ssiAMl = [ssbefchInh_allMice[i][-mnLenDays:] for i in mice2an_manyDays]
    ssaAMl = [ssbefchAllExc_allMice[i][-mnLenDays:] for i in mice2an_manyDays]
    
    # average stability across mice starting from last day (going back to first day)
    sseAv_l = np.mean(sseAMl, axis=0)
    ssiAv_l = np.mean(ssiAMl, axis=0)
    ssaAv_l = np.mean(ssaAMl, axis=0)
    sseSd_l = np.std(sseAMl, axis=0)/np.sqrt(np.shape(sseAMf)[0])
    ssiSd_l = np.std(ssiAMl, axis=0)/np.sqrt(np.shape(sseAMf)[0])
    ssaSd_l = np.std(ssaAMl, axis=0)/np.sqrt(np.shape(sseAMf)[0])   
    _,pei_l = stats.ttest_ind(sseAv_l, ssiAv_l)
    
    
    
    #########################%% Plot average stability across mice for each day of training, 
    #################### Start from the first training day ####################
    
    plt.figure(figsize=(4.5,2.5))
    #plt.subplot(211)
    plt.fill_between(range(mnLenDays), sseAv_f-sseSd_f, sseAv_f+sseSd_f, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(range(mnLenDays), ssiAv_f-ssiSd_f, ssiAv_f+ssiSd_f, alpha=0.5, edgecolor='r', facecolor='r')
    #plt.fill_between(range(mnLenDays), ssaAv_f-ssaSd_f, ssaAv_f+ssaSd_f, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(sseAv_f, color='b', label='exc')
    plt.plot(ssiAv_f, color='r', label='inh')
    plt.plot(ssaAv_f, color='k', label=labAll)
    #plt.errorbar(range(mnLenDays), sseAv_f, sseSd_f, color='b', label='exc')
    #plt.errorbar(range(mnLenDays), ssiAv_f, ssiSd_f, color='r', label='inh')
    #plt.errorbar(range(mnLenDays), ssaAv_f, ssaSd_f, color='k', label=labAll)
    plt.title('P_exc,inh = %.3f' %(pei_f))   
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
    plt.xlabel('Days from 1st imaging session')
    plt.ylabel(sl)
    makeNicePlots(plt.gca())
    '''
    y = np.full(np.shape(pei_f_ed), np.nan)
    mx = plt.gca().get_ylim()[1].astype(int)
    y[pei_f_ed<=.05] = mx
    plt.plot(range(len(y)), y)
    '''
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow0) #,mousename)           
    
        if meas==0:
            nn = 'ang'
        elif meas==1:
            nn = 'nsig'                    
        n = '_aveLast%dbins' %(nbin)
    
        if chAl==1:
            dd = 'chAl_'+savl+'_'+nn+n+'_firstDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_manyDays]) + '_' + nowStr
        else:
            dd = 'chAl_'+savl+'_'+nn+n+'_firstDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_manyDays]) + '_' + nowStr
    
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
    
    #########################%% Plot average stability across mice for each day of training
    #################### Start from the last training day ####################
    
    plt.figure(figsize=(4.5,2.5))
    #plt.subplot(211)
    plt.fill_between(range(mnLenDays), sseAv_l-sseSd_l, sseAv_l+sseSd_l, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(range(mnLenDays), ssiAv_l-ssiSd_l, ssiAv_l+ssiSd_l, alpha=0.5, edgecolor='r', facecolor='r')
    #plt.fill_between(range(mnLenDays), ssaAv_l-ssaSd_l, ssaAv_l+ssaSd_l, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(sseAv_l, color='b', label='exc')
    plt.plot(ssiAv_l, color='r', label='inh')
    plt.plot(ssaAv_l, color='k', label=labAll)
    #plt.errorbar(range(mnLenDays), sseAv_l, sseSd_l, color='b', label='exc')
    #plt.errorbar(range(mnLenDays), ssiAv_l, ssiSd_l, color='r', label='inh')
    #plt.errorbar(range(mnLenDays), ssaAv_l, ssaSd_l, color='k', label=labAll)
    x = np.array(range(mnLenDays))[np.arange(0,mnLenDays,5)]    
    plt.xticks(np.arange(mnLenDays-1,0,-5), x) #[::-1] # we do mnLenDays-1 bc the index of the last day is this... it's python so indexing starts from 0
    plt.title('P_exc,inh = %.3f' %(pei_l))   
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
    plt.xlabel('Days back from last imaging session')
    plt.ylabel(sl)
    makeNicePlots(plt.gca())
    '''
    y = np.full(np.shape(pei_l_ed), np.nan)
    mx = plt.gca().get_ylim()[1].astype(int)
    y[pei_l_ed<=.05] = mx
    plt.plot(range(len(y)), y)
    '''
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow0) #,mousename)           
        
        if meas==0:
            nn = 'ang'
        elif meas==1:
            nn = 'nsig'                    
        n = '_aveLast%dbins' %(nbin)
    
        if chAl==1:
            dd = 'chAl_'+savl+'_'+nn+n+'_lastDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_manyDays]) + '_' + nowStr
        else:
            dd = 'chAl_'+savl+'_'+nn+n+'_lastDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_manyDays]) + '_' + nowStr
    
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    

    ##########################################################################                       
    ##########################################################################                       
    #########################%% For each mouse, average stability in the first and last 5 days of trianing (stab is already averaged in the last 3 bins before the choice)
    # for this plot we show all mice (bc we are averaging only over 5 days)
    
    thd = 5
    sse_avd_f = np.mean([ssbefchExc_allMice[im][0:thd] for im in mice2an_all], axis=1) # average across the 5 days for each mouse
    ssi_avd_f = np.mean([ssbefchInh_allMice[im][0:thd] for im in mice2an_all], axis=1)
    ssa_avd_f = np.mean([ssbefchAllExc_allMice[im][0:thd] for im in mice2an_all], axis=1)
    # sd of the 1st 5 days
    sse_sdd_f = np.std([ssbefchExc_allMice[im][0:thd] for im in mice2an_all], axis=1)/np.sqrt(thd) # sd across the 5 days for each mouse
    ssi_sdd_f = np.std([ssbefchInh_allMice[im][0:thd] for im in mice2an_all], axis=1)/np.sqrt(thd)
    ssa_sdd_f = np.std([ssbefchAllExc_allMice[im][0:thd] for im in mice2an_all], axis=1)/np.sqrt(thd)
#    print sse_avd_f, ssi_avd_f, ssa_avd_f
    
    lastd = -1
    sse_avd_l = np.mean([ssbefchExc_allMice[im][lastd-thd:lastd] for im in mice2an_all], axis=1) # -thd:
    ssi_avd_l = np.mean([ssbefchInh_allMice[im][lastd-thd:lastd] for im in mice2an_all], axis=1)
    ssa_avd_l = np.mean([ssbefchAllExc_allMice[im][lastd-thd:lastd] for im in mice2an_all], axis=1)
    # sd of the last 5 days
    sse_sdd_l = np.std([ssbefchExc_allMice[im][lastd-thd:lastd] for im in mice2an_all], axis=1)/np.sqrt(thd) # -thd:
    ssi_sdd_l = np.std([ssbefchInh_allMice[im][lastd-thd:lastd] for im in mice2an_all], axis=1)/np.sqrt(thd)
    ssa_sdd_l = np.std([ssbefchAllExc_allMice[im][lastd-thd:lastd] for im in mice2an_all], axis=1)/np.sqrt(thd)
#    print sse_avd_l, ssi_avd_l, ssa_avd_l
    
    
    plt.figure(figsize=(4.5,2.5))
    index = np.arange(len(mice2an_all))
    bar_width = 0.15
    
    # first days of training
    errd = dict(ecolor='gray') #, lw=2, capsize=5, capthick=2)
    opacity = 0.5
    plt.bar(index, sse_avd_f, bar_width, alpha=opacity,  color='b', label=('exc_1st%dDays' %(thd)), yerr=sse_sdd_f, error_kw=errd)
    plt.bar(index+1*bar_width, ssi_avd_f, bar_width, alpha=opacity,  color='r', label=('inh_1st%dDays' %(thd)), yerr=ssi_sdd_f, error_kw=errd)
    #plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
    #plt.xticks(index+bar_width, np.array(mice)[mice2an_all])
    #plt.ylabel(sl)
    #makeNicePlots(plt.gca())
    
    
    # last days of training
    #plt.figure(figsize=(4.5,2.5))
    opacity = 0.8
    plt.bar(index+2*bar_width, sse_avd_l, bar_width, alpha=opacity,  color='b', label=('exc_last%dDays' %(thd)), yerr=sse_sdd_l, error_kw=errd)
    plt.bar(index+3*bar_width, ssi_avd_l, bar_width, alpha=opacity,  color='r', label=('inh_last%dDays' %(thd)), yerr=ssi_sdd_l, error_kw=errd)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
    plt.xticks(index+2*bar_width, np.array(mice)[mice2an_all])
    plt.ylabel(sl)
    makeNicePlots(plt.gca())
    
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow0) #,mousename)                                    
        n = '_aveLast%dbins' %(nbin)
    
        if chAl==1:
            dd = 'chAl_'+savl+n+'_ave1st&lastDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_all]) + '_' + nowStr
        else:
            dd = 'chAl_'+savl+nn+n+'_ave1st&lastDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_all]) + '_' + nowStr
    
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




    ##########################################################################                       
    ##########################################################################                       
    #########################%% For each mouse, average stability across all days of trianing (stab is already averaged in the last 3 bins before the choice)
    # for this plot we show all mice 
    
    sse_avd_f = [np.mean(ssbefchExc_allMice[im]) for im in mice2an_all] # average across the 5 days for each mouse
    ssi_avd_f = [np.mean(ssbefchInh_allMice[im]) for im in mice2an_all]
    ssa_avd_f = [np.mean(ssbefchAllExc_allMice[im]) for im in mice2an_all]
    # se
    sse_sdd_f = [np.std(ssbefchExc_allMice[im]) / np.sqrt(len(ssbefchExc_allMice[im])) for im in mice2an_all]
    ssi_sdd_f = [np.std(ssbefchInh_allMice[im]) / np.sqrt(len(ssbefchExc_allMice[im])) for im in mice2an_all]
    ssa_sdd_f = [np.std(ssbefchAllExc_allMice[im]) / np.sqrt(len(ssbefchExc_allMice[im])) for im in mice2an_all]

    plt.figure(figsize=(4.5,2.5))
    index = np.arange(len(mice2an_all))
    bar_width = 0.13
    
    plt.errorbar(index, sse_avd_f, yerr=sse_sdd_f, color='b', label='exc', fmt='o')
    plt.errorbar(index+1*bar_width, ssi_avd_f, yerr=ssi_sdd_f, color='r', label='inh', fmt='o')
    plt.errorbar(index+2*bar_width, ssa_avd_f, yerr=ssa_sdd_f, color='k', label=labAll, fmt='o')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)
    plt.xticks(index+1*bar_width, np.array(mice)[mice2an_all])
    plt.ylabel(sl)
    plt.xlim([-2*bar_width, len(mice)+bar_width])
    makeNicePlots(plt.gca())
    
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow0) #,mousename)                                    
        n = '_aveLast%dbins' %(nbin)
    
        if chAl==1:
            dd = 'chAl_'+savl+n+'_aveAllDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_all]) + '_' + nowStr
        else:
            dd = 'chAl_'+savl+nn+n+'_aveAllDays_inhExc'+labAll+'_' + '_'.join(np.array(mice)[mice2an_all]) + '_' + nowStr
    
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        

#%% Get stability in the last time bin for plots
# sability :
#    meas=0: average angle of each decoder with other decoders
#    meas=1: number of significantly aligned decoders with each decoder at each time point

# Below you can average stability score in the some bins (instead of using only the last time bin before the choice).
# (average in the last n bins before the choice, and compare it between exc and inh)


for meas in [0,1]: # make all plots for both measures of stability
    
    ssbefchExc_allMice = []
    ssbefchInh_allMice = []
    ssbefchAllExc_allMice = []
    
    for im in range(len(mice)):
    
        if meas==0:     # using measure 0 for stab (ave angles)
            if doExcSamps:
                stabScoreExc = stabScoreExc_excsh0_avExcsh_allMice[im]
            else:
                stabScoreExc = stabScoreExc0_allMice[im]
            stabScoreInh = stabScoreInh0_allMice[im]
            stabScoreAllExc = stabScoreAllExc0_allMice[im]   
            sl = stabLab0
        
        elif meas==1:    # using measure 1 for stab (num sig angles)
            if doExcSamps:
                stabScoreExc = stabScoreExc_excsh1_avExcsh_allMice[im]
            else:
                stabScoreExc = stabScoreExc1_allMice[im]
            stabScoreInh = stabScoreInh1_allMice[im]
            stabScoreAllExc = stabScoreAllExc1_allMice[im]   
            sl = stabLab1
        
        
        nPreMin = np.nanmin(eventI_ds_allDays_allMice[im][mn_corr_allMice[im] >= thTrained]).astype('int') 
        print nPreMin
    
        rr = np.arange(nPreMin-nbin, nPreMin)
        
        # average stability score in the last n bins before the choice for each day (stab is already computed as average of angle of decoders in each time bin with all other decoders at other times)
        ssbefchExc = np.mean(stabScoreExc[:, rr], axis=1) # numDays
        ssbefchInh = np.mean(stabScoreInh[:, rr], axis=1)
        ssbefchAllExc = np.mean(stabScoreAllExc[:, rr], axis=1)
        
        ssbefchExc_allMice.append(ssbefchExc)
        ssbefchInh_allMice.append(ssbefchInh)    
        ssbefchAllExc_allMice.append(ssbefchAllExc)
       
    ssbefchExc_allMice = np.array(ssbefchExc_allMice)
    ssbefchInh_allMice = np.array(ssbefchInh_allMice)
    ssbefchAllExc_allMice = np.array(ssbefchAllExc_allMice)
    
    
    #%% Plots: Set averages across mice: stability vs. day, for exc,inh,allN.. . also show summary plots of all mice (stab of time -1 (averaged angle with all other time points))
    
    mice2an_manyDays = np.array([0,1,3])  ##### NOTE: exclude fni18 from this analysis because you didn't image him throughout training
    mice2an_all = np.arange(len(mice))
    mnLenDays = np.min([len(ssbefchExc_allMice[i]) for i in mice2an_manyDays]) # How many of the days to show in the plot?
    #mnLenDays = 30
    
    savl = 'stab'
    if meas==0:
        nn = 'ang'
    elif meas==1:
        nn = 'nsig'      
    savl = savl+'_'+nn
    
    plotssca(ssbefchExc_allMice, ssbefchInh_allMice, ssbefchAllExc_allMice, mnLenDays, sl, savl, mice2an_manyDays, mice2an_all)
    
    
    
    #%% Same as above figures, now for class accuracy
    #############################################
    #############################################
    ##%% 
    
    mice2an = np.arange(len(mice))
    
    # average a in the last 3 bins before the choice
    def avLast3(a,im):
        an = a[im]#[d[im]]
        # average stability score in the last 3 bins before the choice
        ana = np.mean(an[:, rr], axis=1) # numDays
        return ana
    
    
    
    cabefchExc_allMice = []
    cabefchInh_allMice = []
    cabefchAllExc_allMice = []
    
    for im in mice2an:
        cabefchExc = avLast3(classAccurTMS_exc_allMice, im)
        cabefchInh = avLast3(classAccurTMS_inh_allMice, im)
        cabefchAllExc = avLast3(classAccurTMS_allExc_allMice, im)
    
        cabefchExc_allMice.append(cabefchExc)
        cabefchInh_allMice.append(cabefchInh)    
        cabefchAllExc_allMice.append(cabefchAllExc)
       
    cabefchExc_allMice = np.array(cabefchExc_allMice)
    cabefchInh_allMice = np.array(cabefchInh_allMice)
    cabefchAllExc_allMice = np.array(cabefchAllExc_allMice)
    
    
    #%% Plots
    
    savl = 'classAccur'
    plotssca(cabefchExc_allMice, cabefchInh_allMice, cabefchAllExc_allMice, mnLenDays, 'Class accuracy (data-shfl)', savl, mice2an_manyDays, mice2an_all)
    
    
    
            
    #%%
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    #%% SVM performance (allN; ave last nbins before the choice) vs. behav performance
    ### REMEMBER: these plots get saved to a different directory than stability plots
    ### This is because they are behavioral performance vs both svm class accuracy and svm stability
    # so they dont just belong to stability.
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    #%% Plots of single mice
    #############################################################################################
    
    #%% Plot performance of svm and behavior vs days for each mouse
    
    for im in range(len(mice2an)):
        
        nowStr = nowStr_allMice[im]
        
        plt.figure(figsize=(3,4))
    
        plt.subplot(311)
        plt.plot(behCorr_allMice[im], label='allTrs')
        plt.plot(behCorrHR_allMice[im], label='HR')
        plt.plot(behCorrLR_allMice[im], label='LR')
        plt.ylabel('beh corr fract')#('Behavior (fraction correct)')    
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
        plt.title(mice[im])
        makeNicePlots(plt.gca(),0,1)
        
        plt.subplot(312)
        plt.plot(cabefchAllExc_allMice[im])       
        plt.ylabel('svm accur %')#('Classificaiton accuracy of SVM decoder')    
        makeNicePlots(plt.gca(),0,1)
        
        plt.subplot(313)
        plt.plot(ssbefchAllExc_allMice[im])    
        plt.ylabel('svm stab degree')#('Stability of SVM decoder')    
        plt.xlabel('Days')
        makeNicePlots(plt.gca(),0,1)    
        
        plt.subplots_adjust(hspace=.4)
        
    
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnowb+mice[im]) #,mousename)           
            
            if meas==0:
                nn = 'ang'
            elif meas==1:
                nn = 'nsig'                    
            n = '_aveLast%dbins' %(nbin)
        
            if chAl==1:
                dd = 'chAl_beh_classAccur_'+nn+n+'_'+labAll+'_' + daysGood_allMice[im][0][0:6] + '-to-' + daysGood_allMice[im][-1][0:6] + '_' + nowStr    
            else:
                dd = 'chAl_beh_classAccur_'+nn+n+'_'+labAll+'_' + daysGood_allMice[im][0][0:6] + '-to-' + daysGood_allMice[im][-1][0:6] + '_' + nowStr    
        
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
                
    
    #%% scatter plots: svm performance vs behavioral performance
    
    for im in range(len(mice2an)):
        
        nowStr = nowStr_allMice[im]
        
        
        plt.figure(figsize=(7,1.5))
        y = cabefchAllExc_allMice[im] 
        y2 = ssbefchAllExc_allMice[im]
        
        # plot vs. beh performance on all trials (half hr, half lr)
        plt.subplot(131)
        x = behCorr_allMice[im]; 
        plt.plot(x, y, 'b.', label='class accuracy (% correct testing trials)')
        plt.plot(x, y2, 'g.', label='stability (degrees away from orthogonal)')    
        plt.xlabel('beh (fract corr, all trs)') #'behavior (Fraction correct, all trials)'
        plt.ylabel('SVM')    
    #    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1)     
        ccc = np.corrcoef(x,y)[0, 1]
        ccs = np.corrcoef(x,y2)[0, 1]
        plt.title('%s: accur,beh = %.2f\nstab,beh = %.2f' %(mice[im], ccc, ccs))
        makeNicePlots(plt.gca(),1,1)
        
        
        # plot vs. beh performance on hr trials
        plt.subplot(132)
        x = behCorrHR_allMice[im];
        plt.plot(x, y, 'b.', label='class accuracy (% correct testing trials)')
        plt.plot(x, y2, 'g.', label='stability (degrees away from orthogonal)')    
        plt.xlabel('beh (fract corr, HR trs)')
        plt.ylabel('SVM')    
    #    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1) 
        ccc = np.corrcoef(x,y)[0, 1]
        ccs = np.corrcoef(x,y2)[0, 1]
        plt.title('accur,beh = %.2f\nstab,beh = %.2f' %(ccc, ccs))
        makeNicePlots(plt.gca(),1,1)
    
    
        # plot vs. beh performance on lr trials
        plt.subplot(133)
        x = behCorrLR_allMice[im];
        plt.plot(x, y, 'b.', label='SVM class accuracy (% correct testing trials)')
        plt.plot(x, y2, 'g.', label='SVM stability (degrees away from orthogonal)')    
        plt.xlabel('beh (fract corr, LR trs)')
        plt.ylabel('SVM')    
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1) 
        ccc = np.corrcoef(x,y)[0, 1]
        ccs = np.corrcoef(x,y2)[0, 1]
        plt.title('accur,beh = %.2f\nstab,beh = %.2f' %(ccc, ccs))
        makeNicePlots(plt.gca(),1,1)    
        
        plt.subplots_adjust(wspace=1, hspace=1)
        
          
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnowb+mice[im]) #,mousename)           
            
            if meas==0:
                nn = 'ang'
            elif meas==1:
                nn = 'nsig'                    
            n = '_aveLast%dbins' %(nbin)
        
            if chAl==1:
                dd = 'chAl_scatter_behVSsvm_'+nn+n+'_'+labAll+'_'+ daysGood_allMice[im][0][0:6] + '-to-' + daysGood_allMice[im][-1][0:6] + '_' + nowStr #  + '_'.join(np.array(mice)[mice2an])
            else:
                dd = 'chAl_scatter_behVSsvm_'+nn+n+'_'+labAll+'_'+ daysGood_allMice[im][0][0:6] + '-to-' + daysGood_allMice[im][-1][0:6] + '_' + nowStr
        
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            
    
    ##################################################################################################
    ##################################################################################################
    #%% All mice: pool all sessions,  plot svm vs. beh performance on all trials (half hr, half lr)
    
    x = np.concatenate(([behCorr_allMice[im] for im in mice2an])) 
    y = np.concatenate(([cabefchAllExc_allMice[im] for im in mice2an]))
    y2 = np.concatenate(([ssbefchAllExc_allMice[im] for im in mice2an]))
    
    plt.figure(figsize=(3,3))
    
    plt.plot(x, y, 'b.', label='class accuracy (% correct testing trials)')
    plt.plot(x, y2, 'g.', label='stability (degrees away from orthogonal)')    
    plt.xlabel('beh (fract corr, all trs)') #'behavior (Fraction correct, all trials)'
    plt.ylabel('SVM')    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1)     
    ccc = np.corrcoef(x,y)[0, 1]
    ccs = np.corrcoef(x,y2)[0, 1]
    plt.title('accur,beh = %.2f\nstab,beh = %.2f' %(ccc, ccs))
    makeNicePlots(plt.gca(),1,1)
    
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnowb) #,mousename)           
        
        if meas==0:
            nn = 'ang'
        elif meas==1:
            nn = 'nsig'                    
        n = '_aveLast%dbins' %(nbin)
    
        if chAl==1:
            dd = 'chAl_scatter_behVSsvm_'+nn+n+'_'+labAll+'_' + '_'.join(np.array(mice)[mice2an]) + '_' + nowStrAM   
        else:
            dd = 'chAl_scatter_behVSsvm_'+nn+n+'_'+labAll+'_' + '_'.join(np.array(mice)[mice2an]) + '_' + nowStrAM
    
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            
    
    
    #%% Scatter plot: stab vs class accur, allmice, all sessions
    
    y = np.concatenate(([cabefchAllExc_allMice[im] for im in mice2an]))
    y2 = np.concatenate(([ssbefchAllExc_allMice[im] for im in mice2an]))
    
    plt.figure(figsize=(3,3))
    
    plt.plot(y, y2, 'b.')#, label='class accuracy (% correct testing trials)')
    plt.xlabel('SVM class accuracy (%)') #'behavior (Fraction correct, all trials)'
    plt.ylabel('SVM stability (degrees away from orthogonal)')    
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1)     
    ccc = np.corrcoef(y,y2)[0, 1]
    plt.title('accur,stab= %.2f' %(ccc))
    makeNicePlots(plt.gca(),1,1)
    
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow0) #,mousename)           
        
        if meas==0:
            nn = 'ang'
        elif meas==1:
            nn = 'nsig'                    
        n = '_aveLast%dbins' %(nbin)
    
        if chAl==1:
            dd = 'chAl_scatter_svmCAvsStab_'+nn+n+'_'+labAll+'_' + '_'.join(np.array(mice)[mice2an]) + '_' + nowStrAM   
        else:
            dd = 'chAl_scatter_svmCAvsStab_'+nn+n+'_'+labAll+'_' + '_'.join(np.array(mice)[mice2an]) + '_' + nowStrAM
    
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
    #%% scatter plots, each mouse svm performance vs stability
    
    for im in range(len(mice2an)):
        
        nowStr = nowStr_allMice[im]
        
        plt.figure(figsize=(3,3))
        y = cabefchAllExc_allMice[im] 
        y2 = ssbefchAllExc_allMice[im]
        
        plt.plot(y, y2, 'b.')#, label='class accuracy (% correct testing trials)')
        plt.xlabel('SVM class accuracy (%)') #'behavior (Fraction correct, all trials)'
        plt.ylabel('SVM stability (degrees away from orthogonal)')    
        
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1)     
        ccc = np.corrcoef(y,y2)[0, 1]
        plt.title('accur,stab= %.2f' %(ccc))
        makeNicePlots(plt.gca(),1,1)
    
          
        ##%% Save the figure    
        if savefigs:
            d = os.path.join(svmdir+dnowb+mice[im]) #,mousename)           
            
            if meas==0:
                nn = 'ang'
            elif meas==1:
                nn = 'nsig'                    
            n = '_aveLast%dbins' %(nbin)
        
            if chAl==1:
                dd = 'chAl_scatter_svmCAvsStab_'+nn+n+'_'+labAll+'_'+ daysGood_allMice[im][0][0:6] + '-to-' + daysGood_allMice[im][-1][0:6] + '_' + nowStr #  + '_'.join(np.array(mice)[mice2an])
            else:
                dd = 'chAl_scatter_svmCAvsStab_'+nn+n+'_'+labAll+'_'+ daysGood_allMice[im][0][0:6] + '-to-' + daysGood_allMice[im][-1][0:6] + '_' + nowStr
        
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
        