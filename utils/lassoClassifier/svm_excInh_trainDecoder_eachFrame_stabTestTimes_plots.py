# -*- coding: utf-8 -*-
"""
vars are created in svm_excInh_trainDecoder_eachFrame_plots_stabTestTimes_setVars_post.py


Created on Thu Nov 16 16:15:20 2017

@author: farznaj
"""


#%%

mice = 'fni16', 'fni17', 'fni18', 'fni19'

saveDir_allMice = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice'

savefigs = 0
corrTrained = 1
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
useEqualTrNums = 1
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")

regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.


#### vars needed for loading class accuracies of decoders trained and tested on the same time bin
loadWeights = 0
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
doAllN = 1 # plot allN, instead of allExc
doIncorr = 0
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'
if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'    
    
dir0 = '/stability_decoderTestedAllTimes/'
set_save_p_samps = 0 # set and save p value across samples per day (takes time to be done !!)

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')


#%% Set days (all days and good days) for each mouse

days_allMice, numDaysAll, daysGood_allMice, dayinds_allMice, mn_corr_allMice = svm_setDays_allMice(mice, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, regressBins, useEqualTrNums, shflTrsEachNeuron, thTrained)
numDaysGood = np.array([len(daysGood_allMice[im]) for im in range(len(mice))])



#%% Define function for all plots below. Heatmaps of time vs time... showing either CA, or diff from max CA, etc... for exc,inh,allN

def plotAngInhExcAllN(topi0, tope0, topa0, cblab, namf, do4, doCA=0, doAv=0):
    # for the 4th subplot below when plotting stab measure
    def plotstabsp(stab_inhm, stab_excm, stab_allNm, yls, ax):        
        ax.plot(time_aligned_allMice[im], stab_inhm, color='r')
        ax.plot(time_aligned_allMice[im], stab_excm, color='b')
        ax.plot(time_aligned_allMice[im], stab_allNm, color='k')          
        plt.colorbar(ax=ax)    
        ax.set_xlim(s1[im].get_xlim())
        ax.set_xlabel('Testing t (ms)')
        ax.set_ylabel(yls) # Duration(ms) w decoders close to optimal decoder at time t      
    #    s4[im].set_xticks(s3[im].get_xticks())
    #    makeNicePlots(s4[im], 1, 1)      

    def plotstabsp_sd(stab_inhm, stab_excm, stab_allNm, yls, ax, stab_inh_sdm, stab_exc_sdm, stab_allN_sdm):
        ax.fill_between(time_aligned_allMice[im], stab_inhm-stab_inh_sdm, stab_inhm+stab_inh_sdm, color='r', alpha=.7)
        ax.fill_between(time_aligned_allMice[im], stab_excm-stab_exc_sdm, stab_excm+stab_exc_sdm, color='b', alpha=.7)
        ax.fill_between(time_aligned_allMice[im], stab_allNm-stab_allN_sdm, stab_allNm+stab_allN_sdm, color='k', alpha=.7)
        plt.colorbar(ax=ax)    
        ax.set_xlim(s1[im].get_xlim())
        ax.set_xlabel('Testing t (ms)')
        ax.set_ylabel(yls)        
    
    #linstyl = '-','-.',':'
    cols = 'black', 'gray', 'silver'
#    colsi = 'mediumblue', 'blue', 'cyan'
#    colse = 'red', 'tomato', 'lightsalmon'

    s1 = []; s2 = []; s3 = []; s4 = []
    
    for im in range(len(mice)):    
        
        if doAv:
            topi = np.nanmean(topi0[im], axis=0) 
            tope = np.nanmean(tope0[im], axis=0) 
            top = np.nanmean(topa0[im], axis=0) 
            topsd = np.nanstd(topa0[im], axis=0) / np.sqrt(numDaysGood[im])
        else: # inputs are already averaged across days
            topi = topi0[im]
            tope = tope0[im]
            top = topa0[im]
            
        cmap = 'jet' #    extent = setExtent_imshow(time_aligned)
        
        cmin = np.nanmin(np.concatenate((topi,tope, top)))
        cmax = np.nanmax(np.concatenate((topi,tope, top)))
    
        time_aligned = time_aligned_allMice[im]
        nPreMin = nPreMin_allMice[im]
    
       
        plt.figure(figsize=(7,6));           
        
        ################# inh
        plt.subplot(221);     
        lab = 'inh' 
    #    cblab = '' # 'Class accuracy (%)'    
        yl = 'Decoder training t (ms)'
        xl = 'Testing t (ms)'  
        
        plotAng(topi, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.plot([time_aligned[-1],time_aligned[0]], [time_aligned[-1],time_aligned[0]]) # mark the diagonal
        plt.xlim(xlim)        
        plt.ylim(ylim)        
        s1.append(plt.gca());
#        print plt.gca().get_position()
#        sz = plt.gcf().get_size_inches()*fig.dpi
    

        ############### exc
        plt.subplot(223);    
        lab = 'exc' 
    #    cblab = 'Class accuracy (%)'    
#        yl = 'Decoder trained at t (ms)'
#        xl = 'Decoder tested at t (ms)'  
        
        plotAng(tope, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
        plt.plot([time_aligned[-1],time_aligned[0]], [time_aligned[-1],time_aligned[0]]) # mark the diagonal
        plt.xlim(xlim)        
        plt.ylim(ylim)
        s3.append(plt.gca());
        
        
        ################# allN
        plt.subplot(222);     
        lab = 'allN'    
#        cmin = np.nanmin(top)
#        cmax = np.nanmax(top)
    #    cblab = 'Class accuracy (%)'    
#        yl = '' #'Decoder trained at t (ms)'
#        xl = 'Decoder tested at t (ms)'          
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
        plt.plot([time_aligned[-1],time_aligned[0]], [time_aligned[-1],time_aligned[0]]) # mark the diagonal
        plt.xlim(xlim)        
        plt.ylim(ylim)
        s2.append(plt.gca());
        
        if doCA==1: # Mark a few time points (their CA will be plotted in subplot 224)
            fr2ana = [nPreMin-1, nPreMin-7, nPreMin+5]
            cnt = -1
            for fr2an in fr2ana:               
                cnt = cnt+1
                ##### If below you plot decoder trained at fr2an, see how it does on all other times
                plt.axhline(time_aligned[fr2an],np.min(time_aligned),np.max(time_aligned), color=cols[cnt])
                ##### If below you plot decoders tested at fr2an, see how well decoders trained at different times do
#                plt.axvline(time_aligned[fr2an],np.min(time_aligned),np.max(time_aligned), color=cols[cnt])
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
                
            
            
        #################### allN: show example traces for a few time points
        if do4: 
            plt.subplot(224);
            s4.append(plt.gca());
            if doCA==1:
#                top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im] # nTrainedFrs x nTestingFrs   # plot the actual class accuracies instead of the drop in CA
#                topsd = classAcc_allN_allDays_alig_avSamps_sdDays_allMice[im] / np.sqrt(numDaysGood[im])
                cnt = -1    
                # plot class accuracy tested at all times (x axis) when decoder was trained at time fr2an
                for fr2an in fr2ana:       
                    cnt = cnt+1                   
                    ##### Decoder trained at fr2an, see how it does on all other times
                    plt.fill_between(time_aligned, top[fr2an] - topsd[fr2an], top[fr2an] + topsd[fr2an], color=cols[cnt], alpha=.7)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.xlabel('Decoder testing t (ms)')                    
                    ##### Decoders tested at fr2an, see how well decoders trained at different times do
#                    plt.fill_between(time_aligned, top[:,fr2an] - topsd[:,fr2an], top[:,fr2an] + topsd[:,fr2an], color=cols[cnt], alpha=.5)
#                    plt.xlabel('Decoder training t (ms)')    
                    
            #        plt.plot(time_aligned, topi[fr2an], color=colsi[cnt])#, linestyle=linstyl[cnt])
            #        plt.plot(time_aligned, tope[fr2an], color=colse[cnt])#, linestyle=linstyl[cnt])        
                    plt.vlines(time_aligned[fr2an],np.min(top),np.max(top), color=cols[cnt])#, linestyle=linstyl[cnt])
                plt.ylabel(cblab)
                plt.xlim(xlim)
                plt.colorbar()                
#                print plt.gca().get_position()                         
            else: # plot stab measures                    
                if doCA==0:                
                    plotstabsp(stab_inh[im], stab_exc[im], stab_allN[im], yls, plt.gca())
                elif doCA==-1: # computed across samps for each day... so we have sd
                    plotstabsp_sd(stab_inh[im], stab_exc[im], stab_allN[im], yls, plt.gca(), stab_inh_sd[im], stab_exc_sd[im], stab_allN_sd[im])
                # mark time 0 in the above plots
                s1[im].axhline(0,0,len(time_aligned))
                s2[im].axhline(0,0,len(time_aligned))
                s3[im].axhline(0,0,len(time_aligned))    
#                plt.plot(time_aligned, stab_inh[im], color='r')
#                plt.plot(time_aligned, stab_exc[im], color='b')
#                plt.plot(time_aligned, stab_allN[im], color='k')          
#                plt.colorbar()
#                plt.xlabel('Testing t (ms)')
#                plt.ylabel('Stability duration (ms)') # Duration(ms) w decoders close to optimal decoder at time t
            makeNicePlots(plt.gca(), 1, 1)        
        
        plt.subplots_adjust(hspace=.2, wspace=.8)
    

        ############%% Save figure for each mouse
        if savefigs:#% Save the figure
            mousename = mice[im]        
            dnow = dir0 + mousename + '/'
            days = daysGood_allMice[im]
            
            if chAl==1:
                dd = 'chAl_testTrainDiffTimes_' + namf + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr_allMice[im] + '_' + nowStr
            else:
                dd = 'stAl_testTrainDiffTimes_' + namf + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr_allMice[im] + '_' + nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


    return s1,s2,s3,s4



#%% Load matlab vars

################################################################################
################################################################################
    
#%% Load class accur vars (averages of samples)

cnam = glob.glob(os.path.join(saveDir_allMice, 'stability_decoderTestedAllTimes_' + '_'.join(mice) + '_*'))[0]

data = scio.loadmat(cnam, variable_names=['eventI_ds_allDays_allMice',
                    'classAcc_allN_allDays_alig_avSamps_allMice',
                    'classAcc_inh_allDays_alig_avSamps_allMice',
                    'classAcc_exc_allDays_alig_avSamps_allMice',
                    'classAcc_allN_shfl_allDays_alig_avSamps_allMice',
                    'classAcc_inh_shfl_allDays_alig_avSamps_allMice',
                    'classAcc_exc_shfl_allDays_alig_avSamps_allMice',
                    'classAcc_allN_allDays_alig_sdSamps_allMice',
                    'classAcc_inh_allDays_alig_sdSamps_allMice',
                    'classAcc_exc_allDays_alig_sdSamps_allMice',
                    'classAcc_allN_shfl_allDays_alig_sdSamps_allMice',
                    'classAcc_inh_shfl_allDays_alig_sdSamps_allMice',
                    'classAcc_exc_shfl_allDays_alig_sdSamps_allMice'])

                    
eventI_ds_allDays_allMice = data.pop('eventI_ds_allDays_allMice').flatten()                    
# average of samps
classAcc_allN_allDays_alig_avSamps_allMice = data.pop('classAcc_allN_allDays_alig_avSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_avSamps_allMice = data.pop('classAcc_inh_allDays_alig_avSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_avSamps_allMice = data.pop('classAcc_exc_allDays_alig_avSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps
classAcc_allN_shfl_allDays_alig_avSamps_allMice = data.pop('classAcc_allN_shfl_allDays_alig_avSamps_allMice').flatten()
classAcc_inh_shfl_allDays_alig_avSamps_allMice = data.pop('classAcc_inh_shfl_allDays_alig_avSamps_allMice').flatten()
classAcc_exc_shfl_allDays_alig_avSamps_allMice = data.pop('classAcc_exc_shfl_allDays_alig_avSamps_allMice').flatten()
# sd of samps
classAcc_allN_allDays_alig_sdSamps_allMice = data.pop('classAcc_allN_allDays_alig_sdSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_sdSamps_allMice = data.pop('classAcc_inh_allDays_alig_sdSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_sdSamps_allMice = data.pop('classAcc_exc_allDays_alig_sdSamps_allMice').flatten() # nGoodDays x nTrainedFrs x nTestingFrs x nExcSamps
classAcc_allN_shfl_allDays_alig_sdSamps_allMice = data.pop('classAcc_allN_shfl_allDays_alig_sdSamps_allMice').flatten()
classAcc_inh_shfl_allDays_alig_sdSamps_allMice = data.pop('classAcc_inh_shfl_allDays_alig_sdSamps_allMice').flatten()
classAcc_exc_shfl_allDays_alig_sdSamps_allMice = data.pop('classAcc_exc_shfl_allDays_alig_sdSamps_allMice').flatten()

                    
#%% Load mat file: p value across samples per day (takes time to be done !!)
# the measure is OK except we dont know what to do with exc bc it has exc samps... ... there is no easy way to set a threshold for p value that works for both inh and exc... given that exc is averaged across exc samps!
# you can perhaps do some sort of bootstrap!

pnam = glob.glob(os.path.join(saveDir_allMice, 'stability_decoderTestedAllTimes_pval_' + '_'.join(mice) + '_*'))[0] #glob.glob(os.path.join(svmdir+dir0, '*pval_*_svm_testEachDecoderOnAllTimes*'))[0]
data = scio.loadmat(pnam)

p_inh_samps_allMice = data.pop('p_inh_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
p_exc_samps_allMice = data.pop('p_exc_samps_allMice').flatten() # nMice; each mouse: days x frs x frs x excSamps
p_allN_samps_allMice = data.pop('p_allN_samps_allMice').flatten() # nMice; each mouse: days x frs x frs


#%% 
numExcSamps = np.shape(classAcc_exc_allDays_alig_avSamps_allMice[0])[-1]
numSamps = 50 # this is the usual value, get it from classAcc_allN_allDays_alig_allMice[im].shape[-1]


#%% Set time_aligned for all mice

time_aligned_allMice = []
nPreMin_allMice = []   
 
for im in range(len(mice)):  
    time_aligned, nPreMin, nPostMin = set_nprepost(classAcc_allN_allDays_alig_avSamps_allMice[im], eventI_ds_allDays_allMice[im], mn_corr_allMice[im], thTrained, regressBins, traceAl=1)    
    
    time_aligned_allMice.append(time_aligned)
    nPreMin_allMice.append(nPreMin)


lenTraces_AllMice = np.array([len(time_aligned_allMice[im]) for im in range(len(mice))])


#%% Set averages of CA
################################################################################
################################################################################

#%% exc: average across excSamps for each day (of each mouse)

classAcc_exc_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_allDays_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_shfl_allDays_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs


#%% Average across days for each mouse

classAcc_allN_allDays_alig_avSamps_avDays_allMice = []
classAcc_inh_allDays_alig_avSamps_avDays_allMice = []
classAcc_exc_allDays_alig_avSamps_avDays_allMice = []

classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice = []
classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice = []
classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice = []

#AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    # shfl
    classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice.append(np.mean(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    


######%% SD across days for each mouse

classAcc_allN_allDays_alig_avSamps_sdDays_allMice = []
classAcc_inh_allDays_alig_avSamps_sdDays_allMice = []
classAcc_exc_allDays_alig_avSamps_sdDays_allMice = []

classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice = []
classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice = []
classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice = []

#AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

for im in range(len(mice)):
    
    classAcc_allN_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    # shfl
    classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs
    classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice.append(np.std(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0)) # nTrainedFrs x nTestingFrs





#%% Compute change in CA when tested at time ts and trained at time tr relative to its max value (ie when trained on ts (t same as testing time))

alph = .05

def CAchangeWhenTrainedAtOtherTimes(top, p_samps): # drop; in CA when trained at a different time than the testing time
    topd = np.full((len(top),len(top)), np.nan)
    for ifr in range(len(top)):
        if p_samps[ifr,ifr] <= alph: # do not include time points whose max performance (at ts) is chance, it doesnt make sense to see how good decoders trained at other time points can decode choice at ts.
            topd[:,ifr] = top[:,ifr] - top[ifr,ifr]                    
    return topd


def CAfractChangeWhenTrainedAtOtherTimes(top, p_samps): # drop measured as fraction of max CA; in CA when trained at a different time than the testing time
    topd = np.full((len(top),len(top)), np.nan)
    for ifr in range(len(top)):
        if p_samps[ifr,ifr] <= alph: # do not include time points whose max performance (at ts) is chance, it doesnt make sense to see how good decoders trained at other time points can decode choice at ts.
    #        topd[:,ifr] = (top[:,ifr] - top[ifr,ifr]) / top[ifr,ifr]                
            topd[:,ifr] = top[:,ifr] / top[ifr,ifr]  # this is easier to understand ... show fraction of max performance!                  
    return topd


######### change    
classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_allDays_alig_avSamps_allMice)
classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_allDays_alig_avSamps_allMice)
classAcc_changeFromMax_exc_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_allDays_alig_avSamps_allMice)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice[im][iday] = CAchangeWhenTrainedAtOtherTimes(classAcc_allN_allDays_alig_avSamps_allMice[im][iday], p_allN_samps_allMice[im][iday])
        classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice[im][iday] = CAchangeWhenTrainedAtOtherTimes(classAcc_inh_allDays_alig_avSamps_allMice[im][iday], p_inh_samps_allMice[im][iday])
        for iexc in range(numExcSamps):
            classAcc_changeFromMax_exc_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAchangeWhenTrainedAtOtherTimes(classAcc_exc_allDays_alig_avSamps_allMice[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday,:,:,iexc])

# exc: average across exc samps for each day
classAcc_changeFromMax_exc_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_changeFromMax_exc_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])



######### fract max 
classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_allDays_alig_avSamps_allMice)
classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_allDays_alig_avSamps_allMice)
classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_allDays_alig_avSamps_allMice)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_allN_allDays_alig_avSamps_allMice[im][iday], p_allN_samps_allMice[im][iday])
        classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_inh_allDays_alig_avSamps_allMice[im][iday], p_inh_samps_allMice[im][iday])
        for iexc in range(numExcSamps):
            classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_exc_allDays_alig_avSamps_allMice[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday,:,:,iexc])

# exc: average across exc samps for each day
classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])




#%% PLOTS
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
   
    
#%% Plot heatmaps of CA, CA-max(CA), CA/max(CA) when decoder is trained and tested at all time points

cblab012 = 'Class accuracy (%)', 'Class accuracy - max (%)', 'Class accuracy / max' #'Fract change in max class accuracy' 
figns = 'classAccur', 'classAccurMinusMax', 'classAccurFractMax'

# plot class accuracies
plotAngInhExcAllN(classAcc_inh_allDays_alig_avSamps_allMice, classAcc_exc_allDays_alig_avSamps2_allMice, classAcc_allN_allDays_alig_avSamps_allMice, cblab012[0], figns[0], 1, 1, 1)


# plot class accur as a fraction of max class accuracy (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice, classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice, classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice, cblab012[2], figns[2], 1, 1, 1)


# plot change in class accuracies rel2 max (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice, classAcc_changeFromMax_exc_allDays_alig_avSamps2_allMice, classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice, cblab012[1], figns[1], 1, 1, 1)



#%%
###########################################
########### P value, data vs shfl ###########
###########################################

#%% Set p value (data vs shfl) for each pair of time points across days: across all days is CA of decoder trained at time t_i and tested at time t_j different from the trial-label shuffled case?

whichP = 2 # which statistical test to use: 0: wilcoxon, 1: MW, 2: ttest) 

p_inh_allMice = []
p_exc_allMice = []
p_allN_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    p_inh = np.full((nfrs,nfrs), np.nan)
    p_exc = np.full((nfrs,nfrs), np.nan)
    p_allN = np.full((nfrs,nfrs), np.nan)
    
    for ifr1 in range(nfrs):
        for ifr2 in range(nfrs):
            # inh
            a = classAcc_inh_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]
            b = classAcc_inh_shfl_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]       
            p_inh[ifr1,ifr2] = pwm(a,b)[whichP]
    
            # exc
            a = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr2]
            b = classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im][:,ifr1,ifr2]       
            p_exc[ifr1,ifr2] = pwm(a,b)[whichP]

            # allN
            a = classAcc_allN_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]
            b = classAcc_allN_shfl_allDays_alig_avSamps_allMice[im][:,ifr1,ifr2]       
            p_allN[ifr1,ifr2] = pwm(a,b)[whichP]

            
    p_inh_allMice.append(p_inh)
    p_exc_allMice.append(p_exc)
    p_allN_allMice.append(p_allN)



################################################ 
##%% Plots of p value: data vs shfl, across days

plot_CA_diff_fractDiff = 0
cblab = 'P (data vs shfl)'
namf = 'p_data_shfl'
plotAngInhExcAllN(p_inh_allMice, p_exc_allMice, p_allN_allMice, cblab, namf, 0)


################################################ 
# Now set the significancy matrix ... CA at which time points is siginificantly different from shuffle.

alph = .001 # p value for significancy

p_inh_allMice_01 = [x <= alph for x in p_inh_allMice]
p_exc_allMice_01 = [x <= alph for x in p_exc_allMice]
p_allN_allMice_01 = [x <= alph for x in p_allN_allMice]

cblab = 'Sig (data vs shfl)'
namf = 'sig_p'+str(alph)[2:]+'_data_shfl'
plotAngInhExcAllN(p_inh_allMice_01, p_exc_allMice_01, p_allN_allMice_01, cblab, namf, 0)





#%%
#################################################################  
########### P value : CA(trained and tested at ts) vs ########### 
###########  CA(tested at ts but trained at tr) ###########
#################################################################  


#%% Do ttest between max CA(trained and tested at ts) and CA(tested at ts but trained at tr)
    # do this across days

alph = .05 # only if CA(ts,ts) is above chance we run ttest, otherwise we set p value to nan.

p_inh_diag_allMice = []
p_exc_diag_allMice = []
p_allN_diag_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    p_inh = np.full((nfrs,nfrs), np.nan)
    p_exc = np.full((nfrs,nfrs), np.nan)
    p_allN = np.full((nfrs,nfrs), np.nan)
    
    for ts in range(nfrs): # testing time point

        # run p value against the CA(trained and tested at ts)
        bi = classAcc_inh_allDays_alig_avSamps_allMice[im][:,ts,ts] # tested and trained on the same time point
        be = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,ts,ts]               
        b = classAcc_allN_allDays_alig_avSamps_allMice[im][:,ts,ts]       
        
        for tr in range(nfrs): # training time point
            # inh
            if p_inh_allMice[im][ts,ts] <= alph: # only if CA at ts,ts is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                
                a = classAcc_inh_allDays_alig_avSamps_allMice[im][:,tr,ts]           
#                # we dont care if CA increases in another time compared to diagonal CA... I think this is bc diagonal CA has trial subselection as cross validation, but other times CA dont have that (since they are already cross validated by testing the decoder at other times)
#                m = a<=bi0
#                bi = bi0[m]
#                a = a[m]                

                p_inh[tr,ts] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(bi)[~np.isnan(bi)])[1]
    #            p_inh[tr,ts] = ttest2(a,bi,tail='left') # a < b
            
            # exc
            if p_exc_allMice[im][ts,ts] <= alph: # only if CA at ts,ts is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.

                a = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,tr,ts]
#                # we dont care if CA increases in another time compared to diagonal CA... I think this is bc diagonal CA has trial subselection as cross validation, but other times CA dont have that (since they are already cross validated by testing the decoder at other times)
#                m = a<=be0
#                be = be0[m]
#                a = a[m]
                
                p_exc[tr,ts] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(be)[~np.isnan(be)])[1]
    #            p_exc[tr,ts] = ttest2(a,be,tail='left') # a < b
            
            # allN
            if p_allN_allMice[im][ts,ts] <= alph: # only if CA at ts,ts is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                a = classAcc_allN_allDays_alig_avSamps_allMice[im][:,tr,ts]            
#                # we dont care if CA increases in another time compared to diagonal CA... I think this is bc diagonal CA has trial subselection as cross validation, but other times CA dont have that (since they are already cross validated by testing the decoder at other times)
#                m = a<=b0
#                b = b0[m]
#                a = a[m]
                
                p_allN[tr,ts] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
    #            p_allN[tr,ts] = ttest2(a,b,tail='left') # a < b
            
            
    p_inh_diag_allMice.append(p_inh)
    p_exc_diag_allMice.append(p_exc)
    p_allN_diag_allMice.append(p_allN)
    


#%% Plot above computed p values (between max CA(trained and tested at ts) and CA(tested at ts but trained at tr))

cblab = 'p (CA diagonal vs t)'
namf = 'p_maxCA_otherTimes'
plotAngInhExcAllN(p_inh_diag_allMice, p_exc_diag_allMice, p_allN_diag_allMice, cblab, namf, 0)


################################################ 
# Now set the significancy matrix ... 
################################################
alph = .05 # p value for significancy

p_inh_diag_allMice_01 = [x <= alph for x in p_inh_diag_allMice]
p_exc_diag_allMice_01 = [x <= alph for x in p_exc_diag_allMice]
p_allN_diag_allMice_01 = [x <= alph for x in p_allN_diag_allMice]


cblab = 'sig (CA diagonal vs t)'
namf = 'sig_maxCA_otherTimes'
plotAngInhExcAllN(p_inh_diag_allMice_01, p_exc_diag_allMice_01, p_allN_diag_allMice_01, cblab, namf, 0)



    
#%%
######################################################
########### Decoder generalizes to other times? ###########
######################################################
    
#%% Is CA(trained at tr, tested at ts) within 1 std of CA(trained and tested at ts) 
    # do this across days (ie ave and std across days)

#alph = .05 # if diagonal CA is sig (comapred to shfl), then see if it generalizes or not... othereise dont use that time point for analysis.
fractStd = 1

############## for each testing time point (ts), we see what trained decoders do similarly good as the optimal decoder for that time point (ie the decoder trained on ts)

sameCA_inh_diag_allMice = []
sameCA_exc_diag_allMice = []
sameCA_allN_diag_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    sameCA_inh = np.full((nfrs,nfrs), np.nan) # if 1 then CA(trained at t but tested at t') < 1 std of CA(trained and tested at t) 
    sameCA_exc = np.full((nfrs,nfrs), np.nan)
    sameCA_allN = np.full((nfrs,nfrs), np.nan)
    
    for ts in range(nfrs): # testing time point
        
        # Set average and sd across days
        maxcai = classAcc_inh_allDays_alig_avSamps_avDays_allMice[im][ts,ts], fractStd*classAcc_inh_allDays_alig_avSamps_sdDays_allMice[im][ts,ts] # tested and trained on the same time point
        maxcae = classAcc_exc_allDays_alig_avSamps_avDays_allMice[im][ts,ts], fractStd*classAcc_exc_allDays_alig_avSamps_sdDays_allMice[im][ts,ts]
        maxca = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im][ts,ts], fractStd*classAcc_allN_allDays_alig_avSamps_sdDays_allMice[im][ts,ts]
        # set mean - 1sd
        maxca_inh = maxcai[0]-maxcai[1]
        maxca_exc = maxcae[0]-maxcae[1]
        maxca_allN = maxca[0]-maxca[1]
        
        for tr in range(nfrs): # training time point
            # inh
#            if p_inh_allMice[im][ts,ts] <= alph: # only if CA at tr,tr is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
            a = classAcc_inh_allDays_alig_avSamps_allMice[im][:,tr,ts].mean()           
            sameCA_inh[tr,ts] = (a >= maxca_inh) #+ (a>=bi[0]+bi[1]) # sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(bi)[~np.isnan(bi)])[1]
            
            # exc
#            if p_exc_allMice[im][ts,ts] <= alph:
            a = classAcc_exc_allDays_alig_avSamps2_allMice[im][:,tr,ts].mean()           
            sameCA_exc[tr,ts] = (a >= maxca_exc) #+ (a>=be[0]+be[1]) # sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(be)[~np.isnan(be)])[1]
            
            # allN
#            if p_allN_allMice[im][ts,ts] <= alph:
            a = classAcc_allN_allDays_alig_avSamps_allMice[im][:,tr,ts].mean()                       
            sameCA_allN[tr,ts] = (a >= maxca_allN) #+ (a>=b[0]+b[1]) # sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
            
            
    sameCA_inh_diag_allMice.append(sameCA_inh)
    sameCA_exc_diag_allMice.append(sameCA_exc)
    sameCA_allN_diag_allMice.append(sameCA_allN)


#%% Plot of the above analysis : is CA(trained at tr, tested at ts) within 1 std of CA(trained and tested at ts) ... done across days

cblab = 'CA(tr,ts) with 1sd of CA(ts,ts)' #'Diagonal decoder generalizes?',
namf = 'classAccurWin1sdMax'

# For how long each decoder is stable (ie duration over which decoder can be trained to acheive CA within 1sd of max performance on the testing time
        # for the following you need to sum over axis=1... but not sure if it makes sense given that above you do the analysis for each column (ie comparing CA(:,ts) w CA(ts,ts))
        ######################## For how long each decoder is stable (ie its CA when tested on a different time point is within 1 std)
stab_inh = regressBins*frameLength*np.array([np.sum(sameCA_inh_diag_allMice[im], axis=0) for im in range(len(mice))])
stab_exc = regressBins*frameLength*np.array([np.sum(sameCA_exc_diag_allMice[im], axis=0) for im in range(len(mice))])
stab_allN = regressBins*frameLength*np.array([np.sum(sameCA_allN_diag_allMice[im], axis=0) for im in range(len(mice))])
yls = 'Trained decoder stab dur (ms)'

s1,s2,s3,s4 = plotAngInhExcAllN(sameCA_inh_diag_allMice, sameCA_exc_diag_allMice, sameCA_allN_diag_allMice, cblab, namf, 1)


"""
# Get the upper triangle, ie before the testing time point, for how long trained decoders will perform well
# not sure how useful... ofcourse at the upper left corner there is only 1 point so dur will be 100ms
stab_inh = regressBins*frameLength*np.array([np.sum(np.triu(sameCA_inh_diag_allMice[im], k=0), axis=0) for im in range(len(mice))])
stab_exc = regressBins*frameLength*np.array([np.sum(np.triu(sameCA_exc_diag_allMice[im], k=0), axis=0) for im in range(len(mice))])
stab_allN = regressBins*frameLength*np.array([np.sum(np.triu(sameCA_allN_diag_allMice[im], k=0), axis=0) for im in range(len(mice))])
yls = 'Trained decoder stab dur (bef testing t, ms)'

s1,s2,s3,s4 = plotAngInhExcAllN(sameCA_inh_diag_allMice, sameCA_exc_diag_allMice, sameCA_allN_diag_allMice, cblab, namf, 1)

for im in range(len(mice)):
    s1[im].axhline(0,0,len(sameCA_inh_diag_allMice[im]))
    s2[im].axhline(0,0,len(sameCA_inh_diag_allMice[im]))
    s3[im].axhline(0,0,len(sameCA_inh_diag_allMice[im]))    
    # plot stab measures    
    plotstabsp(stab_inh[im], stab_exc[im], stab_allN[im], yls)

    

# Get the lower triangle, ie after hte testing time point for how long trained decoders will perform well
# not sure how useful... ofcourse at the lower right corner there is only 1 point so dur will be 100ms
stab_inh = regressBins*frameLength*np.array([np.sum(np.tril(sameCA_inh_diag_allMice[im], k=0), axis=0) for im in range(len(mice))])
stab_exc = regressBins*frameLength*np.array([np.sum(np.tril(sameCA_exc_diag_allMice[im], k=0), axis=0) for im in range(len(mice))])
stab_allN = regressBins*frameLength*np.array([np.sum(np.tril(sameCA_allN_diag_allMice[im], k=0), axis=0) for im in range(len(mice))])
yls = 'Trained decoder stab dur (after testing t, ms)'

s1,s2,s3,s4 = plotAngInhExcAllN(sameCA_inh_diag_allMice, sameCA_exc_diag_allMice, sameCA_allN_diag_allMice, cblab, namf, 1)

for im in range(len(mice)):
    s1[im].axhline(0,0,len(sameCA_inh_diag_allMice[im]))
    s2[im].axhline(0,0,len(sameCA_inh_diag_allMice[im]))
    s3[im].axhline(0,0,len(sameCA_inh_diag_allMice[im]))    
    # plot stab measures    
    plotstabsp(stab_inh[im], stab_exc[im], stab_allN[im], yls)
    
"""



#%% Same as above but done for each day, across samps... ie is CA at other times points similar (ie within mean - 1 se across samps) of the diagonal CA. This is done for each day, then an average is formed across days

fractStd = 1
alph = .05 # if diagonal CA is sig (comapred to shfl), then see if it generalizes or not... othereise dont use that time point for analysis.

sameCA_inh_samps_allMice = []
sameCA_allN_samps_allMice = []
sameCA_exc_samps_allMice = []
sameCA_exc_samps2_allMice = []

for im in range(len(mice)):

    nfrs = len(time_aligned_allMice[im])    

    sameCA_inh_samps_allDays = []
    sameCA_allN_samps_allDays = []
    sameCA_exc_samps_allDays = []
    sameCA_exc_samps2_allDays = []
        
    for iday in range(numDaysGood[im]):
        sameCA_inh = np.full((nfrs,nfrs), np.nan)
        sameCA_exc = np.full((nfrs,nfrs,numExcSamps), np.nan)
        sameCA_allN = np.full((nfrs,nfrs), np.nan)
        
        for ts in range(nfrs): # testing time point
            
            maxcai = classAcc_inh_allDays_alig_avSamps_allMice[im][iday,ts,ts], fractStd*classAcc_inh_allDays_alig_sdSamps_allMice[im][iday,ts,ts]        # tested and trained on the same time point                
            maxca = classAcc_allN_allDays_alig_avSamps_allMice[im][iday,ts,ts], fractStd*classAcc_allN_allDays_alig_sdSamps_allMice[im][iday,ts,ts]       
            # do for each excSamp
            maxcaea = np.full((2, numExcSamps), np.nan)
            for iexc in range(numExcSamps):
                maxcaea[:, iexc] = classAcc_exc_allDays_alig_avSamps_allMice[im][iday,ts,ts,iexc], fractStd*classAcc_exc_allDays_alig_sdSamps_allMice[im][iday,ts,ts,iexc]
            # average (across excSamps) the ave and sd across samps
#            be = np.mean(bea, axis=1)
            
                
            for tr in range(nfrs): # training time point
                # inh
                if p_inh_samps_allMice[im][iday,ts,ts] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                    a = classAcc_inh_allDays_alig_avSamps_allMice[im][iday,tr,ts]
                    sameCA_inh[tr,ts] = (a >= (maxcai[0]-maxcai[1])) 
                
                # allN
                if p_allN_samps_allMice[im][iday,ts,ts] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                    a = classAcc_allN_allDays_alig_avSamps_allMice[im][iday,tr,ts]
                    sameCA_allN[tr,ts] = (a >= (maxca[0]-maxca[1])) 
                
                # exc ... compute p value across (trial) samps for each excSamp
                for iexc in range(numExcSamps):
                    if p_exc_samps_allMice[im][iday,ts,ts,iexc] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                        a = classAcc_exc_allDays_alig_avSamps_allMice[im][iday,tr,ts,iexc]                    
                        sameCA_exc[tr,ts,iexc] = (a >= (maxcaea[0,iexc]-maxcaea[1,iexc]))    
                # exc: average across exc samps
                sameCA_exc_avSamps = np.nanmean(sameCA_exc,axis=-1)
                
        # keep vars of all days
        sameCA_inh_samps_allDays.append(sameCA_inh)
        sameCA_allN_samps_allDays.append(sameCA_allN)
        sameCA_exc_samps_allDays.append(sameCA_exc)
        sameCA_exc_samps2_allDays.append(sameCA_exc_avSamps)        
    
    # keep vars of all mice
    sameCA_inh_samps_allMice.append(sameCA_inh_samps_allDays)
    sameCA_allN_samps_allMice.append(sameCA_allN_samps_allDays)
    sameCA_exc_samps_allMice.append(sameCA_exc_samps_allDays)    
    sameCA_exc_samps2_allMice.append(sameCA_exc_samps2_allDays)

sameCA_inh_samps_allMice = np.array(sameCA_inh_samps_allMice)
sameCA_allN_samps_allMice = np.array(sameCA_allN_samps_allMice)
sameCA_exc_samps_allMice = np.array(sameCA_exc_samps_allMice)
sameCA_exc_samps2_allMice = np.array(sameCA_exc_samps2_allMice)
 
   
#%% Average of above values across days [whether for each day CA(tr,ts) is within 1se of CA(ts,ts) across samps]

sameCA_inh_samps_aveDays_allMice = [np.nanmean(sameCA_inh_samps_allMice[im],axis=0) for im in range(len(mice))]
sameCA_allN_samps_aveDays_allMice = [np.nanmean(sameCA_allN_samps_allMice[im],axis=0) for im in range(len(mice))]
# exc: average across days and exc samps
#sameCA_exc_samps_aveDays_allMice = [np.nanmean(sameCA_exc_samps_allMice[im],axis=(0,-1)) for im in range(len(mice))]
sameCA_exc_samps_aveDays_allMice = [np.nanmean(sameCA_exc_samps2_allMice[im],axis=0) for im in range(len(mice))]
# use one random exc samp
#r = rng.permutation(numExcSamps)[0]
#sameCA_exc_samps_aveDays_allMice = [np.nanmean(np.array(sameCA_exc_samps_allMice[im])[:,:,:,r],axis=0) for im in range(len(mice))]


# Average of fract days with stable decoders across time points (ie its CA when tested on a different time point is within 1 std)
#stab_inh = np.array([np.nanmean(sameCA_inh_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_exc = np.array([np.nanmean(sameCA_exc_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_allN = np.array([np.nanmean(sameCA_allN_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])



# For how long trained decoders have CA similar to max CA
            # Compute for each day, at each time point, the duration of stability (ie to how many other time points the decoder generalizes)
stab_inh_eachDay_allMice = []
stab_exc_eachDay_allMice = []
stab_allN_eachDay_allMice = []

for im in range(len(mice)):
    stab_inh_eachDay = []
    stab_exc_eachDay = []
    stab_allN_eachDay = []
    
    for iday in range(numDaysGood[im]):
        stab_inh_eachDay.append(regressBins*frameLength * np.sum(sameCA_inh_samps_allMice[im][iday], axis=0)) # days x trainedFrs        
        stab_allN_eachDay.append(regressBins*frameLength * np.sum(sameCA_allN_samps_allMice[im][iday], axis=0)) # days x trainedFrs        
#        stab_exc_eachDay.append(np.sum(sameCA_exc_samps_allMice[im][iday], axis=0)) # days x trainedFrs x excSamps
        stab_exc_eachDay.append(regressBins*frameLength * np.sum(sameCA_exc_samps2_allMice[im][iday], axis=0)) # days x trainedFrs
        
    stab_inh_eachDay_allMice.append(np.array(stab_inh_eachDay))
    stab_allN_eachDay_allMice.append(np.array(stab_allN_eachDay))
    stab_exc_eachDay_allMice.append(np.array(stab_exc_eachDay))

stab_inh_eachDay_allMice = np.array(stab_inh_eachDay_allMice)
stab_allN_eachDay_allMice = np.array(stab_allN_eachDay_allMice)
stab_exc_eachDay_allMice = np.array(stab_exc_eachDay_allMice)

'''
for iday in range(numDaysGood[im]): 
    plt.figure()
    plt.imshow(classAcc_inh_allDays_alig_avSamps_allMice[im][iday]); plt.colorbar()
    plt.imshow(sameCA_exc_samps2_allMice[im][iday])
'''

#%% Plot of the above analysis [whether for each day CA(tr,ts) is within 1se of CA(ts,ts) across samps]

# On average across days, for how long each decoder generalizes.
stab_inh = np.array([np.nanmean(stab_inh_eachDay_allMice[im],axis=0) for im in range(len(mice))])
stab_allN = np.array([np.nanmean(stab_allN_eachDay_allMice[im],axis=0) for im in range(len(mice))])
# exc: average across days and exc samps
#stab_exc = np.array([np.nanmean(stab_exc_eachDay_allMice[im],axis=(0,-1)) for im in range(len(mice))])
stab_exc = np.array([np.nanmean(stab_exc_eachDay_allMice[im],axis=0) for im in range(len(mice))])
# use one random exc samp
#r = rng.permutation(numExcSamps)[0]
#stab_exc = np.array([np.nanmean(np.array(stab_exc_eachDay_allMice[im])[:,:,r],axis=0) for im in range(len(mice))])

# sd
stab_inh_sd = np.array([np.nanstd(stab_inh_eachDay_allMice[im],axis=0)/np.sqrt(numDaysGood[im]) for im in range(len(mice))])
stab_allN_sd = np.array([np.nanstd(stab_allN_eachDay_allMice[im],axis=0)/np.sqrt(numDaysGood[im]) for im in range(len(mice))])
stab_exc_sd = np.array([np.nanstd(stab_exc_eachDay_allMice[im],axis=0)/np.sqrt(numDaysGood[im]) for im in range(len(mice))])


cblab = 'Fract days with high stability'
namf = 'classAccurWin1sdMax_samps'
plotAngInhExcAllN(sameCA_inh_samps_aveDays_allMice, sameCA_exc_samps_aveDays_allMice, sameCA_allN_samps_aveDays_allMice, cblab, namf, 1, -1)





#%%
####################################################################################
####################################################################################

#%% Plots of stability vs day

#%% Look at the stability duration (duration that a decoder can be trained to perform within 1se of CA(ts,ts)) in time -1 for each day, and plot it across days!

plt.figure()
for im in range(len(mice)):
    plt.subplot(2,2,im+1)
    a = stab_inh_eachDay_allMice[im][:, nPreMin_allMice[im]-1]    
    plt.plot(a)
    plt.xlabel('Days')
    if np.in1d(im,[0,1]):
        plt.ylabel(yls)
    makeNicePlots(plt.gca())
plt.subplots_adjust(hspace=0.5, wspace=.5)



#%% Look at CA(tr,ts) / CA(ts,ts) (ie max_CA at ts)  across days.... try a bunch of points but eg the following makes sense: go with training at time -1, and testing at -6 

# trt and tst are relative to time right before the choice
def trts(trt,tst):
    tr_1_ts_3 = []
    for im in range(len(mice)):
        tr = max(0, nPreMin_allMice[im] + trt)
        ts = min(lenTraces_AllMice[im]-1, nPreMin_allMice[im] + tst)
    
        fractMax = [classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im][iday][tr, ts] for iday in range(numDaysGood[im])]
        tr_1_ts_3.append(fractMax)
    
    tr_1_ts_3 = np.array(tr_1_ts_3)
    
    
    #####
    plt.figure()
    for im in range(len(mice)):
        plt.subplot(2,2,im+1)
        plt.plot(tr_1_ts_3[im])
        plt.xlabel('Days')
        if np.in1d(im,[0,1]):
            plt.ylabel('classAccurFractMax')
        makeNicePlots(plt.gca())
    plt.subplots_adjust(hspace=0.5, wspace=.5)


    ############%% Save figure for each mouse
    if savefigs:#% Save the figure   
    
        namf = 'classAccurFractMax_vs_days_'
        trsn = 'tr'+str(trt)+'ts'+str(tst)+'_'

        if chAl==1:
            dd = 'chAl_' + namf + trsn + '_'.join(mice) + '_' + nowStr
        else:
            dd = 'stAl_' + namf + trsn + '_'.join(mice) + '_' + nowStr
            
        d = os.path.join(svmdir+dir0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dir0, suffn[0:5]+dd+'.'+fmt[0])
        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


#%% times are relative to choice



trt = -10 # -5
for tst in [-2, 0, 2, 6]:
    trts(trt,tst)


trt = -2
for tst in [0, 2, 6]:
    trts(trt,tst)


trt = 0
for tst in [2, 6]:
    trts(trt,tst)


trt = 2
for tst in [6]:
    trts(trt,tst)



# train ~600ms before choice, test 100ms before choice


    




#%%    
'''
plt.figure(figsize=(numDaysGood[im]*2,2))

for iday in range(numDaysGood[im]):
    plt.subplot(1,numDaysGood[im], iday+1)
    plt.imshow(sameCA_allN_samps_allMice[im][iday])
'''    
    
    
#%% p value across samples per day (takes time to be done !!)
# it is fine except we dont know what to do with exc bc it has exc samps... ... there is no easy way to set a threshold for p value that works for both inh and exc... given that exc is averaged across exc samps!
# you can perhaps do some sort of bootstrap!

"""
if set_save_p_samps == 0 :
    pnam = glob.glob(os.path.join(svmdir+dir0, '*pval_*_svm_testEachDecoderOnAllTimes*'))[0]
    data = scio.loadmat(pnam)

    p_inh_samps_allMice = data.pop('p_inh_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
    p_exc_samps_allMice = data.pop('p_exc_samps_allMice').flatten() # nMice; each mouse: days x frs x frs x excSamps
    p_allN_samps_allMice = data.pop('p_allN_samps_allMice').flatten() # nMice; each mouse: days x frs x frs

else:
    p_inh_samps_allMice = []
    p_exc_samps_allMice = []
    p_allN_samps_allMice = []
    
    for im in range(len(mice)):
        print 'mouse ', im
        nfrs = len(time_aligned_allMice[im])    
    
        p_inh_samps_allDays = []
        p_exc_samps_allDays = []
        p_allN_samps_allDays = []
            
        for iday in range(numDaysGood[im]):
            print 'day ', iday
            p_inh = np.full((nfrs,nfrs), np.nan)
            p_exc = np.full((nfrs,nfrs,numExcSamps), np.nan)
            p_allN = np.full((nfrs,nfrs), np.nan)
            
            for ifr1 in range(nfrs):
                for ifr2 in range(nfrs):
                    
                    # inh
                    a = classAcc_inh_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    b = classAcc_inh_shfl_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    p_inh[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
            
                    # allN
                    a = classAcc_allN_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    b = classAcc_allN_shfl_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    p_allN[ifr1,ifr2] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
                    
                    # exc ... compute p value across (trial) samps for each excSamp
                    for iexc in range(numExcSamps):
    #                    print iexc
                        a = classAcc_exc_allDays_alig_allMice[im][iexc,iday,ifr1,ifr2,:]
                        b = classAcc_exc_shfl_allDays_alig_allMice[im][iexc,iday,ifr1,ifr2,:]
                        p_exc[ifr1,ifr2,iexc] = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])[1]
        
            # keep vars of all days
            p_inh_samps_allDays.append(p_inh)
            p_exc_samps_allDays.append(p_exc)
            p_allN_samps_allDays.append(p_allN)
        
        # keep vars of all mice
        p_inh_samps_allMice.append(p_inh_samps_allDays)
        p_exc_samps_allMice.append(p_exc_samps_allDays)
        p_allN_samps_allMice.append(p_allN_samps_allDays)
                
    
    #from datetime import datetime
    #nowStr = datetime.now().strftime('%y%m%d-%H%M%S')            
    pnam = os.path.join(svmdir+dir0, 'pval_' + '_'.join(mice) + '_' + os.path.basename(stabTestDecodeName))
    scio.savemat(pnam, {'p_inh_samps_allMice':p_inh_samps_allMice, 'p_exc_samps_allMice':p_exc_samps_allMice, 'p_allN_samps_allMice':p_allN_samps_allMice})


#%%
### exc: median p value across excSamps
p_exc_samps2_allMice = [np.nanmedian(p_exc_samps_allMice[im], axis=-1) for im in range(len(mice))] # nMice; each mouse: days x frs x frs
# pick one random exc samp instead of above
iexc = rng.permutation(numExcSamps)[0]
p_exc_samps2_allMice = [p_exc_samps_allMice[im][:,:,:,iexc] for im in range(len(mice))]
    

### median of p value (computed across samps for each day) across days
p_allN_samps_avDays_allMice = [np.mean(p_allN_samps_allMice[im], axis=0) for im in range(len(mice))] # nMice; each mouse: frs x frs
p_inh_samps_avDays_allMice = [np.mean(p_inh_samps_allMice[im], axis=0) for im in range(len(mice))]
p_exc_samps_avDays_allMice = [np.mean(p_exc_samps2_allMice[im], axis=0) for im in range(len(mice))]

plt.figure(); plt.imshow(p_allN_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(p_inh_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(p_exc_samps_avDays_allMice[0]); plt.colorbar()



alph = .05

p_allN_samps_allMice_01 = [x <= alph for x in p_allN_samps_avDays_allMice]
p_inh_samps_allMice_01 = [x <= alph for x in p_inh_samps_avDays_allMice]
p_exc_samps_allMice_01 = [x <= alph for x in p_exc_samps_avDays_allMice]

plt.figure(); plt.imshow(p_allN_samps_allMice_01[0]); plt.colorbar()
plt.figure(); plt.imshow(p_inh_samps_allMice_01[0]); plt.colorbar()
plt.figure(); plt.imshow(p_exc_samps_allMice_01[0]); plt.colorbar()



### fraction of days with sig values of p for each pair of time points
# average sig value (computed across samps for each day) across days
sig_allN_samps_avDays_allMice = [np.nanmean(p_allN_samps_allMice[im]<=alph, axis=0) for im in range(len(mice))] # nMice; each mouse: frs x frs
sig_inh_samps_avDays_allMice = [np.nanmean(p_inh_samps_allMice[im]<=alph, axis=0) for im in range(len(mice))]
sig_exc_samps_avDays_allMice = [np.nanmean(p_exc_samps2_allMice[im]<=alph, axis=0) for im in range(len(mice))]

plt.figure(); plt.imshow(sig_allN_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(sig_inh_samps_avDays_allMice[0]); plt.colorbar()
plt.figure(); plt.imshow(sig_exc_samps_avDays_allMice[0]); plt.colorbar()

# show time point in which >80% of the days are significant
for im in range(len(mice)):
    plt.figure(); plt.imshow(sig_allN_samps_avDays_allMice[im]>.9); plt.colorbar()
    plt.figure(); plt.imshow(sig_inh_samps_avDays_allMice[im]>.9); plt.colorbar()
    plt.figure(); plt.imshow(sig_exc_samps_avDays_allMice[im]>.9); plt.colorbar()




plt.imshow(p_allN_samps_allMice_01[0]); plt.colorbar()

for im in range(len(mice)):
    plt.figure(); plt.imshow(p_allN_allMice_01[im]); plt.colorbar()

"""


#%%
##compute only continues ones
#    
#stab = np.full(len(time_aligned), np.nan)    
#for fr2an in range(len(time_aligned)):
#    stab[fr2an] = sum(top[fr2an] >= (top[fr2an,fr2an] - topsd[fr2an,fr2an]))
#
#
#classAcc_allN_allDays_alig_avSamps_allMice[im]
#    
#    

#%%
"""
'''
for ifr in range(18):
    d = (top[ifr,ifr]-top[ifr,:])/top[ifr,ifr]; 
    plt.figure()
    plt.plot(d); 
    plt.vlines(ifr,d.min(),d.max())
    plt.hlines(.1,0,18)    
'''

di_allMice = []
de_allMice = []    
d_allMice = []    

for im in range(len(mice)):    
    
    # data class accuracies
    topi = classAcc_inh_allDays_alig_avSamps_avDays_allMice[im]    
    tope = classAcc_exc_allDays_alig_avSamps_avDays_allMice[im]    
    top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im]       
   
   
    di = [(topi[ifr,ifr]-topi[ifr,:]) / topi[ifr,ifr] for ifr in range(len(top))]
    de = [(tope[ifr,ifr]-tope[ifr,:]) / tope[ifr,ifr] for ifr in range(len(top))]
    d = [(top[ifr,ifr]-top[ifr,:]) / top[ifr,ifr] for ifr in range(len(top))]
    
    
    di_allMice.append(di)
    de_allMice.append(de)
    d_allMice.append(d)
    


plt.imshow(d_allMice, cmap='jet_r'); plt.colorbar(); 
"""

#%% Old stuff

'''
    #%% For each day average class accur across samps
    
    # subtract 100 to turn it into class accur
    classAcc_allN_aveSamps = 100-np.array([np.mean(classErr_allN_allDays_allMice[im][iday], axis=1) for iday in range(numDaysAll[im])]) # numDays; each day: trainedFrs x testingFrs
    classAcc_inh_aveSamps = 100-np.array([np.mean(classErr_inh_allDays_allMice[im][iday], axis=1) for iday in range(numDaysAll[im])]) # numDays; each day: trainedFrs x testingFrs
    classAcc_exc_aveSamps = 100-np.array([([np.mean(classErr_exc_allDays_allMice[im][iday][iexc], axis=1) for iexc in range(numExcSamps)]) for iday in range(numDaysAll[im])]) # numDays x nExcSamps; each day: trainedFrs x testingFrs



    ############### Align samp-averaged traces of all days ###############

    #%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(classAcc_allN_aveSamps, eventI_ds_allDays_allMice[im], mn_corr_allMice[im], thTrained, regressBins) # av_test_data_inh

    eventI_ds_allMice.append(nPreMin) # you need this var to align traces of all mice
    nFrs = len(time_aligned) #classAcc_allN_aveSamps_alig_avD.shape[0]
    
    
    #%% Align traces of all days

    classAcc_allN_aveSamps_alig = alTrace_frfr(classAcc_allN_aveSamps, eventI_ds_allDays_allMice[im], nPreMin, nPostMin) # days x trainedAlFrs x testedAlFrs
    classAcc_inh_aveSamps_alig = alTrace_frfr(classAcc_inh_aveSamps, eventI_ds_allDays_allMice[im], nPreMin, nPostMin) # days x trainedAlFrs x testedAlFrs
    classAcc_exc_aveSamps_alig = np.array([alTrace_frfr(classAcc_exc_aveSamps[:,iexc], eventI_ds_allDays_allMice[im], nPreMin, nPostMin) for iexc in range(numExcSamps)]) # excSamps x days x alFrs x alFrs
    
    
    #%% Set diagonal to nan (for the diagonal elements decoder is trained using 90% of the trials and tested on all trials... so they are not informative... accuracy is close to 90-100% for all time points)
    
    [np.fill_diagonal(classAcc_allN_aveSamps_alig[iday,:,:], np.nan) for iday in range(numDaysAll[im])];
    [np.fill_diagonal(classAcc_inh_aveSamps_alig[iday,:,:], np.nan) for iday in range(numDaysAll[im])];
    for iday in range(numDaysAll[im]):
        [np.fill_diagonal(classAcc_exc_aveSamps_alig[iexc,iday,:,:], np.nan) for iexc in range(numExcSamps)];

    
    #%% exc: average across exc samps for each day

    classAcc_exc_aveSamps_alig_avSamps = np.mean(classAcc_exc_aveSamps_alig, axis=0)   # days x trainedAlFrs x testedAlFrs
      
      
    #%% Average above aligned traces across days

    classAcc_allN_aveSamps_alig_avD = np.mean(classAcc_allN_aveSamps_alig, axis=0)   # trainedAlFrs x testedAlFrs
    classAcc_inh_aveSamps_alig_avD = np.mean(classAcc_inh_aveSamps_alig, axis=0)   # trainedAlFrs x testedAlFrs
    classAcc_exc_aveSamps_alig_avD = np.mean(classAcc_exc_aveSamps_alig_avSamps, axis=0) # already averaged across exc samps for each day  # trainedAlFrs x testedAlFrs
    
    
    #%% Plot CA tested at all time points
    
    plt.figure(figsize=(6,6))
    
    cmap='jet'
#    extent = setExtent_imshow(time_aligned)

    ##### inh
    plt.subplot(221)
 
    lab = 'inh' 
    top = classAcc_inh_aveSamps_alig_avD
    cmin = np.nanmin(top)
    cmax = np.nanmax(top)
    cblab = '' # 'Class accuracy (%)'    
    yl = 'Decoder trained at t (ms)'
    xl = 'Decoder tested at t (ms)'  
    
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    


    ##### exc
    plt.subplot(223)
 
    lab = 'exc' 
    top = classAcc_exc_aveSamps_alig_avD
    cmin = np.nanmin(top)
    cmax = np.nanmax(top)
    cblab = 'Class accuracy (%)'    
    yl = 'Decoder trained at t (ms)'
    xl = 'Decoder tested at t (ms)'  
    
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    



    ##### allN
    plt.subplot(222)
 
    lab = 'allN'
    top = classAcc_allN_aveSamps_alig_avD
    cmin = np.nanmin(top)
    cmax = np.nanmax(top)
    cblab = 'Class accuracy (%)'    
    yl = '' #'Decoder trained at t (ms)'
    xl = 'Decoder tested at t (ms)'  
    
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
    
    plt.subplots_adjust(hspace=.2, wspace=.5)
'''    





#%%
        """            
        # Data class accuracies        
        if plot_CA_diff_fractDiff==0:
            cblab = cblab012[0] #'Class accuracy (%)'   
            cmap='jet' #    extent = setExtent_imshow(time_aligned)
    
            topi = topi0[im] + 0  
            tope = tope0[im] + 0    
            top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im] + 0     
            topsd = classAcc_allN_allDays_alig_avSamps_sdDays_allMice[im] / np.sqrt(numDaysGood[im])


        # Change in class accuracy relative to max class accuracy (as a result of testing the decoder at a different time point) 
        elif plot_CA_diff_fractDiff==1:
            cblab = cblab012[1] #'Class accuracy rel2 max (%)'   
            cmap='jet' #'jet_r'
            
            # average across days
            topi = np.mean(classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice[im], axis=0) 
            tope = np.mean(classAcc_changeFromMax_exc_allDays_alig_avSamps2_allMice[im], axis=0) 
            top = np.mean(classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice[im], axis=0) 
            topsd = np.std(classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice[im], axis=0) / np.sqrt(numDaysGood[im]) 
    
            '''
            topid = np.full((len(top),len(top)), np.nan) # nTrainedFrs x nTestingFrs ... how well can trainedFr decoder predict choice at testingFrs, relative to max (optimal) decode accuracy at those testingFrs which is obtained by training decoders at frs = testingFrs
            toped = np.full((len(top),len(top)), np.nan)
            topd = np.full((len(top),len(top)), np.nan)
            for ifr in range(len(top)):
                topid[:,ifr] = topi[:,ifr] - topi[ifr,ifr]  # topi_diff[ifr0, ifr]: how well decoder trained at ifr0 generalize to time ifr # should be same as: topi_diff[ifr,:] = topi[ifr,ifr] - topi[ifr,:]
                toped[:,ifr] = tope[:,ifr] - tope[ifr,ifr]
                topd[:,ifr] = top[:,ifr] - top[ifr,ifr]                
#                topid[ifr,:] = topi[ifr,:] - topi[ifr,ifr]  # topid(ifr,:) shows how well decoder trained at ifr does at all other times, relative to its performance on ifr (ie when it tested and trained on ifr).
#                toped[ifr,:] = tope[ifr,:] - tope[ifr,ifr]
#                topd[ifr,:] = top[ifr,:] - top[ifr,ifr]                
            topi = topid
            tope = toped
            top = topd
            '''

            '''            
            decoderGeneralize_inh = np.full((len(top),len(top)), np.nan) # decoderGeneralizing(ifr1,ifr2) shows how much ifr1-trained decoder generalizes to ifr2
            decoderGeneralize_exc = np.full((len(top),len(top)), np.nan) # decoderGeneralizing(ifr1,ifr2) shows how much ifr1-trained decoder generalizes to ifr2
            decoderGeneralize_allN = np.full((len(top),len(top)), np.nan) # decoderGeneralizing(ifr1,ifr2) shows how much ifr1-trained decoder generalizes to ifr2
            
            for ifr2 in range(len(top)): # testingFr
                # inh
                decoderGeneralize_inh[:,ifr2] = topi[:,ifr2] - topi[ifr2,ifr2] # this meas shows how well ifr1-trained decoder can do at ifr2 taking into account the max performance that it could reach at ifr2 (ie acquired by training on ifr2)                                 
                # exc
                decoderGeneralize_exc[:,ifr2] = tope[:,ifr2] - tope[ifr2,ifr2]
                # allN
                decoderGeneralize_allN[:,ifr2] = top[:,ifr2] - top[ifr2,ifr2]
            # extended version of above... same though
            for ifr1 in range(len(top)): # trainingFr
                for ifr2 in range(len(top)): # testingFr
                    # inh
                    canow = topi[ifr1,ifr2]
                    maxcafortestingfr = topi[ifr2,ifr2] # max is when it's trained on the same time-point (ifr2) that it's being tested at.
                    cadevfrommax = canow - maxcafortestingfr # how much ca is different from max if decoder was trained on ifr1
                    decoderGeneralize_inh[ifr1,ifr2] = cadevfrommax # this meas shows how well ifr1-trained decoder can do at ifr2 taking into account the max performance that it could reach at ifr2 (ie acquired by training on ifr2)                   
                    
                    # exc
                    decoderGeneralize_exc[ifr1,ifr2] = tope[ifr1,ifr2] - tope[ifr2,ifr2]

                    # allN
                    decoderGeneralize_allN[ifr1,ifr2] = top[ifr1,ifr2] - top[ifr2,ifr2]
            
            topi = decoderGeneralize_inh
            tope = decoderGeneralize_exc
            top = decoderGeneralize_allN
            cblab = cblab012[2] #'Fract change in max class accuracy'   
            cmap='jet' #'jet_r'        
            '''
            
        # Fraction of max class accuracy that is changed after testing the decoder at a different time point     
        elif plot_CA_diff_fractDiff==2:
            cblab = cblab012[2] #'Fract change in max class accuracy'   
            cmap='jet' #'jet_r' 

            # average across days
            topi = np.mean(classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice[im], axis=0) 
            tope = np.mean(classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice[im], axis=0) 
            top = np.mean(classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im], axis=0) 
            topsd = np.std(classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im], axis=0) / np.sqrt(numDaysGood[im]) 
            
            '''
            topid = np.full((len(top),len(top)), np.nan) # nTrainedFrs x nTestingFrs ... how well can trainedFr decoder predict choice at testingFrs, relative to max (optimal) decode accuracy at those testingFrs which is obtained by training decoders at frs = testingFrs
            toped = np.full((len(top),len(top)), np.nan)
            topd = np.full((len(top),len(top)), np.nan)
            for ifr in range(len(top)):
                topid[:,ifr] = (topi[:,ifr] - topi[ifr,ifr]) / topi[ifr,ifr] # topi_diff[ifr0, ifr]: how well decoder trained at ifr0 generalize to time ifr # should be same as: topi_diff[ifr,:] = topi[ifr,ifr] - topi[ifr,:]
                toped[:,ifr] = (tope[:,ifr] - tope[ifr,ifr]) / tope[ifr,ifr]
                topd[:,ifr] = (top[:,ifr] - top[ifr,ifr]) / top[ifr,ifr]
            topi = topid
            tope = toped
            top = topd
            '''

        
        '''    
        thdrop = .1
        topi = [x<=thdrop for x in topi]    
        tope = [x<=thdrop for x in tope]    
        top = [x<=thdrop for x in top]    
        cblab = '% drop in class accuracy < 0.1'   
        cmap='jet'
        '''
        """        
        
        
#%%

#%% same as classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice
#############################
"""
alph = .05

fractMax_inh_allMice = []
fractMax_exc_allMice = []
fractMax_allN_allMice = []

for im in range(len(mice)):
    
    nfrs = len(time_aligned_allMice[im])    
    fractMax_inh = []
    fractMax_exc = []
    fractMax_allN = []
    
    for iday in range(numDaysGood[im]):
        
        fractMax_i = np.full((nfrs,nfrs), np.nan)
        fractMax_e = np.full((nfrs,nfrs,numExcSamps), np.nan)
        fractMax_a = np.full((nfrs,nfrs), np.nan)
        
        for tr in range(nfrs):
            # inh
            trainedPerf = classAcc_inh_allDays_alig_avSamps_allMice[im][iday,tr,:] # each row
            maxPerf = np.diagonal(classAcc_inh_allDays_alig_avSamps_allMice[im][iday,:,:]) # diagonal
            fractMax_i[tr,:] = trainedPerf / maxPerf
            fractMax_i[tr, np.diagonal(p_inh_samps_allMice[im][iday,:,:]) > alph] = np.nan # set to nan if maxPerf is non-sig relative to shfl... if max performance at ts is chance, it doesnt make sense to see how good decoders trained at other time points can decode choice at ts.
                        
            # exc : do it for each excSamp
            for iexc in range(numExcSamps):
                trainedPerf = classAcc_exc_allDays_alig_avSamps_allMice[im][iexc,iday,tr,:] # each row
                maxPerf = np.diagonal(classAcc_exc_allDays_alig_avSamps_allMice[im][iexc,iday,:,:]) # diagonal
                fractMax_e[tr,:,iexc] = trainedPerf / maxPerf            
                fractMax_e[tr, np.diagonal(p_exc_samps_allMice[im][iday,:,:,iexc]) > alph, iexc] = np.nan # set to nan if maxPerf is non-sig relative to shfl... if max performance at ts is chance, it doesnt make sense to see how good decoders trained at other time points can decode choice at ts.
                        
            # allN
            trainedPerf = classAcc_allN_allDays_alig_avSamps_allMice[im][iday,tr,:] # each row
            maxPerf = np.diagonal(classAcc_allN_allDays_alig_avSamps_allMice[im][iday,:,:]) # diagonal
            fractMax_a[tr,:] = trainedPerf / maxPerf            
            fractMax_a[tr, np.diagonal(p_allN_samps_allMice[im][iday,:,:]) > alph] = np.nan # set to nan if maxPerf is non-sig relative to shfl... if max performance at ts is chance, it doesnt make sense to see how good decoders trained at other time points can decode choice at ts.
                     
                        
        fractMax_inh.append(fractMax_i)
        fractMax_exc.append(fractMax_e)
        fractMax_allN.append(fractMax_a)

    fractMax_inh_allMice.append(fractMax_inh) # days x frs x frs
    fractMax_exc_allMice.append(fractMax_exc) # days x frs x frs x excSamps
    fractMax_allN_allMice.append(fractMax_allN) # days x frs x frs
        

# Averages across days
        
fractMax_inh_avDays_allMice = [np.nanmean(fractMax_inh_allMice[im],axis=0) for im in range(len(mice))]
fractMax_allN_avDays_allMice = [np.nanmean(fractMax_allN_allMice[im],axis=0) for im in range(len(mice))]
# exc: average across days and exc samps
fractMax_exc_avDays_allMice = [np.nanmean(fractMax_exc_allMice[im],axis=(0,-1)) for im in range(len(mice))]        


cblab = 'Fract days with generalized diagonal decoder'
plotAngInhExcAllN(fractMax_inh_avDays_allMice, fractMax_exc_avDays_allMice, fractMax_allN_avDays_allMice, cblab, namf, 0)
"""

        