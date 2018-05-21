#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SVM trained on correct trials. Compare its performance on correct vs. incorrect trials. Done for exc, inh, all neurons.
vars are created in svm_excInh_trainDecoder_eachFrame_plots_stabTestTimes_setVars_post.py

classAcc_allN_allDays_... --> testing incorrect
classAccTest_data_allN_... --> testing correct


Created on Mon Apr 30 22:33:08 2018
@author: farznaj
"""

#%%

mice = 'fni16', 'fni17', 'fni18', 'fni19'

saveDir_allMice = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice'
          
startDayFni19 = 16 #1 # this will only affect the plot of aveSe across days # what day is the start day for mouse fni19: amazing changes during learning. I believe he was more posterior. In early days he represents stimulus category, in later days he represents choice. So we go with his choice days for this summary.

savefigs = 0
set_save_p_samps = 0 # set it to 0 if p vals are already saved for the stab mat file under study # set and save p value across samples per day (takes time to be done !!)

doTestingTrs = 0 # if 1 compute classifier performance only on testing trials; otherwise on all trials
normWeights = 0 #1 # if 1, weights will be normalized to unity length. ### NOTE: you figured if you do np.dot(x,w) using normalized w, it will not match the output of svm (perClassError)

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
testIncorr = 1
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
    
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'

if testIncorr:
    corrn = 'corrIncorr_'

dir0 = '/excInh_trainDecoder_eachFrame/'

if doTestingTrs:
    nts = 'testingTrs_'
else:
    nts = ''

if normWeights:
    nw = 'wNormed_'
else:
    nw = 'wNotNormed_'    
   
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')


    
#%% Set days (all days and good days) for each mouse

days_allMice, numDaysAll, daysGood_allMice, dayinds_allMice, mn_corr_allMice = svm_setDays_allMice(mice, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, regressBins, useEqualTrNums, shflTrsEachNeuron, thTrained)
numDaysGood = np.array([len(daysGood_allMice[im]) for im in range(len(mice))])


#%% Load classAcc vars that were set in svm_excInh_trainDecoder_eachFrame_stabTestTimes_setVars_post.py and set their averages across days

snn = 'decoderTestedIncorrTrs_'
execfile("svm_excInh_load_av_classAcc.py")




#%%
###############################################################################################
############################################ PLOTS ############################################
###############################################################################################

#%% For each population (inh, exc, allN) compare SVM performance on correct vs incorrect trials.

##### incorr
# ave across days
topallav_incorr = classAcc_inh_allDays_alig_avSamps_avDays_allMice, classAcc_exc_allDays_alig_avSamps_avDays_allMice, classAcc_allN_allDays_alig_avSamps_avDays_allMice # classAcc_inh_allDays_alig_avSamps_allMice, classAcc_exc_allDays_alig_avSamps2_allMice, classAcc_allN_allDays_alig_avSamps_allMice
topallav_incorrs = classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice, classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice, classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice# classAcc_inh_shfl_allDays_alig_avSamps_allMice, classAcc_exc_shfl_allDays_alig_avSamps2_allMice, classAcc_allN_shfl_allDays_alig_avSamps_allMice
# se across days
topallse_incorr = classAcc_inh_allDays_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAcc_exc_allDays_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAcc_allN_allDays_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood)
topallse_incorrs = classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood)

##### testing corr
# ave across days
topallav_corr = classAccTest_data_inh_alig_avSamps_avDays_allMice, classAccTest_data_exc_alig_avSamps_avDays_allMice, classAccTest_data_allN_alig_avSamps_avDays_allMice #classAccTest_data_inh_alig_avSamps_allMice, classAccTest_data_exc_alig_avSamps2_allMice, classAccTest_data_allN_alig_avSamps_allMice
topallav_corrs = classAccTest_shfl_inh_alig_avSamps_avDays_allMice, classAccTest_shfl_exc_alig_avSamps_avDays_allMice, classAccTest_shfl_allN_alig_avSamps_avDays_allMice #classAccTest_shfl_inh_alig_avSamps_allMice, classAccTest_shfl_exc_alig_avSamps2_allMice, classAccTest_shfl_allN_alig_avSamps_allMice
# se across days
topallse_corr = classAccTest_data_inh_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAccTest_data_exc_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAccTest_data_allN_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood)
topallse_corrs = classAccTest_shfl_inh_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAccTest_shfl_exc_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood), classAccTest_shfl_allN_alig_avSamps_sdDays_allMice / np.sqrt(numDaysGood) 

cols = 'r', 'b', 'k'
colss = cols #'lightsalmon', 'cyan', 'gray'
alphci = [.7, .5]
alphcis = [.4, .2]
    
for im in range(len(mice)):    
    time_aligned = time_aligned_allMice[im]
    nPreMin = nPreMin_allMice[im]
    
    ###### data    
    # incorr
    topi = topallav_incorr[0][im] # np.nanmean(topall[0][im], axis=0) 
    tope = topallav_incorr[1][im] # np.nanmean(topall[1][im], axis=0) 
    top = topallav_incorr[2][im] # np.nanmean(topall[2][im], axis=0) 
    topisd = topallse_incorr[0][im] # np.nanstd(topall[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesd = topallse_incorr[1][im] # np.nanstd(topall[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsd = topallse_incorr[2][im] # np.nanstd(topall[2][im], axis=0) / np.sqrt(numDaysGood[im])
    # corr
    topi_corr = topallav_corr[0][im] # np.nanmean(topall_corr[0][im], axis=0) 
    tope_corr = topallav_corr[1][im] # np.nanmean(topall_corr[1][im], axis=0) 
    top_corr = topallav_corr[2][im] # np.nanmean(topall_corr[2][im], axis=0) 
    topisd_corr = topallse_corr[0][im] # np.nanstd(topall_corr[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesd_corr = topallse_corr[1][im] # np.nanstd(topall_corr[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsd_corr = topallse_corr[2][im] # np.nanstd(topall_corr[2][im], axis=0) / np.sqrt(numDaysGood[im])
    ###### shfl
    # incorr
    topis = topallav_incorrs[0][im] # np.nanmean(topall[0][im], axis=0) 
    topes = topallav_incorrs[1][im] # np.nanmean(topall[1][im], axis=0) 
    tops = topallav_incorrs[2][im] # np.nanmean(topall[2][im], axis=0) 
    topisds = topallse_incorrs[0][im] # np.nanstd(topall[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesds = topallse_incorrs[1][im] # np.nanstd(topall[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsds = topallse_incorrs[2][im] # np.nanstd(topall[2][im], axis=0) / np.sqrt(numDaysGood[im])
    # corr
    topi_corrs = topallav_corrs[0][im] # np.nanmean(topall_corrs[0][im], axis=0) 
    tope_corrs = topallav_corrs[1][im] # np.nanmean(topall_corr[1][im], axis=0) 
    top_corrs = topallav_corrs[2][im] # np.nanmean(topall_corr[2][im], axis=0) 
    topisd_corrs = topallse_corrs[0][im] # np.nanstd(topall_corr[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesd_corrs = topallse_corrs[1][im] # np.nanstd(topall_corr[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsd_corrs = topallse_corrs[2][im] # np.nanstd(topall_corr[2][im], axis=0) / np.sqrt(numDaysGood[im])
    

   
    plt.figure(figsize=(3,8))
    
    ######### inh
    plt.subplot(311)
    cnt = 0    
    ### data
    # corr
    plt.fill_between(time_aligned, topi_corr - topisd_corr, topi_corr + topisd_corr, color=cols[cnt], alpha=alphci[0], label='corr')
    plt.plot(time_aligned, topi_corr , color=cols[cnt])
    # incorr
    plt.fill_between(time_aligned, topi - topisd, topi + topisd, color=cols[cnt], alpha=alphci[1], label='incorr')
    plt.plot(time_aligned, topi , color=cols[cnt])
    ### shfl
    # corr
    plt.fill_between(time_aligned, topi_corrs - topisd_corrs, topi_corrs + topisd_corrs, color=colss[cnt], alpha=alphcis[0])
    plt.plot(time_aligned, topi_corrs , color=colss[cnt], alpha=alphcis[0])
    # incorr
    plt.fill_between(time_aligned, topis - topisds, topis + topisds, color=colss[cnt], alpha=alphcis[1])
    plt.plot(time_aligned, topis , color=colss[cnt], alpha=alphcis[1])
    makeNicePlots(plt.gca(), 1, 1) 
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')
    
    
    
    ######### exc
    plt.subplot(312)
    cnt = 1
    ### data
    # corr        
    plt.fill_between(time_aligned, tope_corr - topesd_corr, tope_corr + topesd_corr, color=cols[cnt], alpha=alphci[0], label='corr')
    plt.plot(time_aligned, tope_corr , color=cols[cnt])
    # incorr
    plt.fill_between(time_aligned, tope - topesd, tope + topesd, color=cols[cnt], alpha=alphci[1], label='incorr')
    plt.plot(time_aligned, tope , color=cols[cnt])
    ### shfl
    # corr
    plt.fill_between(time_aligned, tope_corrs - topesd_corrs, tope_corrs + topesd_corrs, color=colss[cnt], alpha=alphcis[0])
    plt.plot(time_aligned, tope_corrs , color=colss[cnt], alpha=alphcis[0])
    # incorr
    plt.fill_between(time_aligned, topes - topesds, topes + topesds, color=colss[cnt], alpha=alphcis[1])
    plt.plot(time_aligned, topes , color=colss[cnt], alpha=alphcis[1])    
    makeNicePlots(plt.gca(), 1, 1) 
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')
    
    ######### allN
    plt.subplot(313)
    cnt = 2
    ### data
    # corr            
    plt.fill_between(time_aligned, top_corr - topsd_corr, top_corr + topsd_corr, color=cols[cnt], alpha=alphci[0], label='corr')
    plt.plot(time_aligned, top_corr , color=cols[cnt])
    # incorr
    plt.fill_between(time_aligned, top - topsd, top + topsd, color=cols[cnt], alpha=alphci[1], label='incorr')
    plt.plot(time_aligned, top , color=cols[cnt])    
    ### shfl
    # corr
    plt.fill_between(time_aligned, top_corrs - topsd_corrs, top_corrs + topsd_corrs, color=colss[cnt], alpha=alphcis[0])
    plt.plot(time_aligned, top_corrs , color=colss[cnt], alpha=alphcis[0])
    # incorr
    plt.fill_between(time_aligned, tops - topsds, tops + topsds, color=colss[cnt], alpha=alphcis[1])
    plt.plot(time_aligned, tops , color=colss[cnt], alpha=alphcis[1])    
    makeNicePlots(plt.gca(), 1, 1)    
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')

    plt.subplots_adjust(hspace=.4)

        
    ##%% Save the figure    
    if savefigs:
        days = daysGood_allMice[im]
        dnow = dir0 + mice[im] + '/'
        
        if chAl==1:
            dd = 'chAl_' + corrn + 'aveSeDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'aveSeDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') 
        
 
#%% Plot corr and incorr in different subplots, and for each compare exc, inh, allN

for im in range(len(mice)):    
    time_aligned = time_aligned_allMice[im]
    nPreMin = nPreMin_allMice[im]
    
    ###### data    
    # incorr
    topi = topallav_incorr[0][im] # np.nanmean(topall[0][im], axis=0) 
    tope = topallav_incorr[1][im] # np.nanmean(topall[1][im], axis=0) 
    top = topallav_incorr[2][im] # np.nanmean(topall[2][im], axis=0) 
    topisd = topallse_incorr[0][im] # np.nanstd(topall[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesd = topallse_incorr[1][im] # np.nanstd(topall[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsd = topallse_incorr[2][im] # np.nanstd(topall[2][im], axis=0) / np.sqrt(numDaysGood[im])
    # corr
    topi_corr = topallav_corr[0][im] # np.nanmean(topall_corr[0][im], axis=0) 
    tope_corr = topallav_corr[1][im] # np.nanmean(topall_corr[1][im], axis=0) 
    top_corr = topallav_corr[2][im] # np.nanmean(topall_corr[2][im], axis=0) 
    topisd_corr = topallse_corr[0][im] # np.nanstd(topall_corr[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesd_corr = topallse_corr[1][im] # np.nanstd(topall_corr[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsd_corr = topallse_corr[2][im] # np.nanstd(topall_corr[2][im], axis=0) / np.sqrt(numDaysGood[im])
    ###### shfl
    # incorr
    topis = topallav_incorrs[0][im] # np.nanmean(topall[0][im], axis=0) 
    topes = topallav_incorrs[1][im] # np.nanmean(topall[1][im], axis=0) 
    tops = topallav_incorrs[2][im] # np.nanmean(topall[2][im], axis=0) 
    topisds = topallse_incorrs[0][im] # np.nanstd(topall[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesds = topallse_incorrs[1][im] # np.nanstd(topall[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsds = topallse_incorrs[2][im] # np.nanstd(topall[2][im], axis=0) / np.sqrt(numDaysGood[im])
    # corr
    topi_corrs = topallav_corrs[0][im] # np.nanmean(topall_corrs[0][im], axis=0) 
    tope_corrs = topallav_corrs[1][im] # np.nanmean(topall_corr[1][im], axis=0) 
    top_corrs = topallav_corrs[2][im] # np.nanmean(topall_corr[2][im], axis=0) 
    topisd_corrs = topallse_corrs[0][im] # np.nanstd(topall_corr[0][im], axis=0) / np.sqrt(numDaysGood[im])
    topesd_corrs = topallse_corrs[1][im] # np.nanstd(topall_corr[1][im], axis=0) / np.sqrt(numDaysGood[im])
    topsd_corrs = topallse_corrs[2][im] # np.nanstd(topall_corr[2][im], axis=0) / np.sqrt(numDaysGood[im])
    

    plt.figure(figsize=(6,2.5))
    
    ############################## incorr ##############################     
    plt.subplot(121) 
    plt.title('Incorrect trials')
    
    ######### inh
    cnt = 0    
    ### data
    # incorr
    plt.fill_between(time_aligned, topi - topisd, topi + topisd, color=cols[cnt], alpha=alphci[1])
    plt.plot(time_aligned, topi , color=cols[cnt], label='inh')
    ### shfl
    # incorr
    plt.fill_between(time_aligned, topis - topisds, topis + topisds, color=colss[cnt], alpha=alphcis[1])
    plt.plot(time_aligned, topis , color=colss[cnt], alpha=alphcis[1])
#    makeNicePlots(plt.gca(), 1, 1) 
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')
    
    
    ######### exc
    cnt = 1
    ### data
    # incorr
    plt.fill_between(time_aligned, tope - topesd, tope + topesd, color=cols[cnt], alpha=alphci[1])
    plt.plot(time_aligned, tope , color=cols[cnt], label='exc')
    ### shfl
    # incorr
    plt.fill_between(time_aligned, topes - topesds, topes + topesds, color=colss[cnt], alpha=alphcis[1])
    plt.plot(time_aligned, topes , color=colss[cnt], alpha=alphcis[1])    
#    makeNicePlots(plt.gca(), 1, 1) 
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')


    ######### allN
    cnt = 2
    ### data
    # incorr
    plt.fill_between(time_aligned, top - topsd, top + topsd, color=cols[cnt], alpha=alphci[1])
    plt.plot(time_aligned, top , color=cols[cnt], label='allN')    
    ### shfl
    # incorr
    plt.fill_between(time_aligned, tops - topsds, tops + topsds, color=colss[cnt], alpha=alphcis[1])
    plt.plot(time_aligned, tops , color=colss[cnt], alpha=alphcis[1])    
    makeNicePlots(plt.gca(), 1, 1)    
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')



    ############################## corr ##############################
    plt.subplot(122)
    plt.title('Correct trials')
    
    ######### inh    
    cnt = 0    
    ### data
    # corr
    plt.fill_between(time_aligned, topi_corr - topisd_corr, topi_corr + topisd_corr, color=cols[cnt], alpha=alphci[0])#, label='corr')
    plt.plot(time_aligned, topi_corr , color=cols[cnt])
    ### shfl
    # corr
    plt.fill_between(time_aligned, topi_corrs - topisd_corrs, topi_corrs + topisd_corrs, color=colss[cnt], alpha=alphcis[0])
    plt.plot(time_aligned, topi_corrs , color=colss[cnt], alpha=alphcis[0])
#    makeNicePlots(plt.gca(), 1, 1) 
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')
    
    ######### exc
    cnt = 1
    ### data
    # corr        
    plt.fill_between(time_aligned, tope_corr - topesd_corr, tope_corr + topesd_corr, color=cols[cnt], alpha=alphci[0])#, label='corr')
    plt.plot(time_aligned, tope_corr , color=cols[cnt])
    ### shfl
    # corr
    plt.fill_between(time_aligned, tope_corrs - topesd_corrs, tope_corrs + topesd_corrs, color=colss[cnt], alpha=alphcis[0])
    plt.plot(time_aligned, tope_corrs , color=colss[cnt], alpha=alphcis[0])
#    makeNicePlots(plt.gca(), 1, 1) 
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')
    
    ######### allN
    cnt = 2
    ### data
    # corr            
    plt.fill_between(time_aligned, top_corr - topsd_corr, top_corr + topsd_corr, color=cols[cnt], alpha=alphci[0])#, label='corr')
    plt.plot(time_aligned, top_corr , color=cols[cnt])
    ### shfl
    # corr
    plt.fill_between(time_aligned, top_corrs - topsd_corrs, top_corrs + topsd_corrs, color=colss[cnt], alpha=alphcis[0])
    plt.plot(time_aligned, top_corrs , color=colss[cnt], alpha=alphcis[0])
    makeNicePlots(plt.gca(), 1, 1)    
    plt.xlabel('Time since choice onset (ms)')               
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    yl = plt.gca().get_ylim()
    plt.vlines(0, yl[0], yl[1], 'gray', linestyle=':')
    
    
    plt.subplots_adjust(wspace=.4)
        
    ##%% Save the figure    
    if savefigs:
        
        days = daysGood_allMice[im]
        dnow = dir0 + mice[im] + '/'        
        
        if chAl==1:
            dd = 'chAl_' + 'corrIncorr_aveSeDays_sepPlots_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + 'corrIncorr_aveSeDays_sepPlots_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') 
        

#%% Heatmaps of class accuracy showing all days; svm performance on incorr trials, show shfl, data, data-shfl

sepCB = 1 # if 1: plot inh and exc on a separate c axis scale than allN    
asp = 'auto' #2
cmap ='jet'
ncols = 3

for im in range(len(mice)):
    
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(7,15))
    
    for iplot in range(3): # iplot = 0              
        if iplot==0: ### shfl: plot class accuracy for each day
            topall = classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], classAcc_allN_shfl_allDays_alig_avSamps_allMice[im]
            namp = 'classAccur_'
            cblab = 'Class accuracy (%)\nshfl'
            
        if iplot==1: ### data: plot class accuracy for each day
            topall = classAcc_inh_allDays_alig_avSamps_allMice[im], classAcc_exc_allDays_alig_avSamps2_allMice[im], classAcc_allN_allDays_alig_avSamps_allMice[im]
            namp = 'classAccur_'
            cblab = 'Class accuracy (%)\ndata'
     
        if iplot==2: ### data-shfl: plot class accuracy for each day
            topall = classAcc_inh_allDays_alig_avSamps_allMice[im] - classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], classAcc_exc_allDays_alig_avSamps2_allMice[im] - classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], classAcc_allN_allDays_alig_avSamps_allMice[im] - classAcc_allN_shfl_allDays_alig_avSamps_allMice[im]
            namp = 'classAccur_'
            cblab = 'Class accuracy (%)\ndata-shfl'
            
            
        topi = topall[0]
        tope = topall[1]
        topa = topall[2]
        
        
        if sepCB: # plot inh and exc on a different c axis scale
            cminc = np.floor(np.min([np.nanmin(topi), np.nanmin(tope)]))
            cmaxc = np.floor(np.max([np.nanmax(topi), np.nanmax(tope)]))
            cbn = 'sepColorbar_'
        else:
            cminc = np.floor(np.min([np.nanmin(topi), np.nanmin(tope), np.nanmin(topa)]))
            cmaxc = np.floor(np.max([np.nanmax(topi), np.nanmax(tope), np.nanmax(topa)]))            
            cbn = ''
        
        lab = 'inh'
        top = topi
        ax = axes.flat[ncols*(iplot+1)-3]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax, '')
        ax.set_aspect(asp)
        makeNicePlots(ax,1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()        
     
       
        lab = 'exc'
        top = tope
        ax = axes.flat[ncols*(iplot+1)-2]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax)
    #    y=-1; pp = np.full(ps.shape, np.nan); pp[pc<=.05] = y
    #    ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
        ax.set_aspect(asp)
        makeNicePlots(ax,1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)        
        if sepCB: # plot inh and exc on a different c axis scale
            pos = ax.get_position()
            cb_ax = fig.add_axes([0, pos.y0, 0.02, pos.height])
#            cb_ax = fig.add_axes([0, 0.15, 0.02, 0.72])
            cbar = fig.colorbar(img, cax=cb_ax, label='')
        
        
        
        ### allN
        if sepCB:
            cminc = np.floor(np.min([np.nanmin(topi), np.nanmin(tope), np.nanmin(topa)]))
            cmaxc = np.floor(np.max([np.nanmax(topi), np.nanmax(tope), np.nanmax(topa)]))
        
        lab = labAll
        top = topa
        ax = axes.flat[ncols*(iplot+1)-1]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax, '')
        #plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
        #add_colorbar(img)
        pos = ax.get_position()
        cb_ax = fig.add_axes([0.92, pos.y0, 0.02, pos.height])
#        cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
        cbar = fig.colorbar(img, cax=cb_ax, label=cblab)
        ax.set_aspect(asp)
        makeNicePlots(ax,1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
     
        
        plt.subplots_adjust(wspace=.4, hspace=.4)
        
        
        ##%% Save the figure    
        if savefigs:
            days = daysGood_allMice[im]
            dnow = dir0 + mice[im] + '/'
        
            d = os.path.join(svmdir+dnow) #,mousename)       
    #            daysnew = (np.array(days))[dayinds]
            if chAl==1:
                dd = 'chAl_' + corrn + 'eachDay_heatmap_' + cbn + namp + labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'eachDay_heatmap_' + cbn + namp + labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        


###########################################################################
###########################################################################
            
#%% Plot classErr (aver and se across all sessions) for each mouse

# set ave and se across days
            
days2an_allM = [(range(numDaysGood[im])) for im in range(len(mice))]
days2an_allM[3] = np.arange(startDayFni19, numDaysGood[3])

# corr testing data
av_allN = [np.mean(classAccTest_data_allN_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1]) for im in range(len(mice))] # days  # corr
av_inh = [np.mean(classAccTest_data_inh_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1]) for im in range(len(mice))] # days  # corr
av_exc = [np.mean(classAccTest_data_exc_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1]) for im in range(len(mice))] # days  # corr

se_allN = [np.std(classAccTest_data_allN_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1])/np.sqrt(len(days2an_allM[im])) for im in range(len(mice))] # days  # corr
se_inh = [np.std(classAccTest_data_inh_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1])/np.sqrt(len(days2an_allM[im])) for im in range(len(mice))] # days  # corr
se_exc = [np.std(classAccTest_data_exc_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1])/np.sqrt(len(days2an_allM[im])) for im in range(len(mice))] # days  # corr

# incorr
av_allN_incorr = [np.mean(classAcc_allN_allDays_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1]) for im in range(len(mice))] # days  # corr
av_inh_incorr = [np.mean(classAcc_inh_allDays_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1]) for im in range(len(mice))] # days  # corr
av_exc_incorr = [np.mean(classAcc_exc_allDays_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1]) for im in range(len(mice))] # days  # corr

se_allN_incorr = [np.std(classAcc_allN_allDays_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1])/np.sqrt(len(days2an_allM[im])) for im in range(len(mice))] # days  # corr
se_inh_incorr = [np.std(classAcc_inh_allDays_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1])/np.sqrt(len(days2an_allM[im])) for im in range(len(mice))] # days  # corr
se_exc_incorr = [np.std(classAcc_exc_allDays_alig_avSamps_allMice[im][days2an_allM[im],nPreMin_allMice[im]-1])/np.sqrt(len(days2an_allM[im])) for im in range(len(mice))] # days  # corr


#%%

plt.figure(figsize=(4,3))

plt.subplot(121)
# corr
ac = plt.errorbar(range(len(mice)), av_allN, se_allN, fmt='o', label=labAll, color='k', markeredgecolor='k', markersize=4)
#plt.errorbar(range(len(mice)), av_inh, se_inh, fmt='o', label='inh', color='r')
#plt.errorbar(range(len(mice)), av_exc, se_exc, fmt='o', label='exc', color='b')

# incorr
ai = plt.errorbar(range(len(mice)), av_allN_incorr, se_allN_incorr, fmt='o', label=labAll, color='k', alpha=.3, markeredgecolor='k', markersize=4)
#plt.errorbar(range(len(mice)), av_inh_incorr, se_inh, fmt='o', label='inh', color='r', alpha=.3)
#plt.errorbar(range(len(mice)), av_exc_incorr, se_exc, fmt='o', label='exc', color='b', alpha=.3)

#plt.legend([ac,ai],['corr','incorr'], loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
plt.xlabel('Mice', fontsize=11)
plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. choice', fontsize=11)
plt.xlim([-.2,len(mice)-1+.2])
plt.xticks(range(len(mice)),mice)
ax = plt.gca()
makeNicePlots(ax)
yl = ax.get_ylim()
plt.ylim([yl[0]-2, yl[1]])



plt.subplot(122)
gp = .3
# corr
#ac = plt.errorbar(range(len(mice)), av_allN, se_allN, fmt='o', label=labAll, color='k')
ac = plt.errorbar(range(len(mice)), av_inh, se_inh, fmt='o', label='inh', color='r', markeredgecolor='r', markersize=4)
plt.errorbar(np.arange(len(mice))+gp, av_exc, se_exc, fmt='o', label='exc', color='b', markeredgecolor='b', markersize=4)

# incorr
#ai = plt.errorbar(range(len(mice)), av_allN_incorr, se_allN, fmt='o', label=labAll, color='k', alpha=.3)
ai = plt.errorbar(range(len(mice)), av_inh_incorr, se_inh_incorr, fmt='o', label='inh', color='r', alpha=.3, markeredgecolor='r', markersize=4)
plt.errorbar(np.arange(len(mice))+gp, av_exc_incorr, se_exc_incorr, fmt='o', label='exc', color='b', alpha=.3, markeredgecolor='b', markersize=4)

plt.legend([ac,ai],['corr','incorr'], loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
plt.xlabel('Mice', fontsize=11)
#plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. choice', fontsize=11)
plt.xlim([-.2,len(mice)-1+.2+gp])
plt.xticks(range(len(mice)),mice)
ax = plt.gca()
makeNicePlots(ax)
yl = ax.get_ylim()
plt.ylim([yl[0]-2, yl[1]])


plt.subplots_adjust(wspace=.5)



if savefigs:#% Save the figure
    if shflTrsEachNeuron:
        sn = 'shflTrsPerN_'
    else:
        sn = ''
        
    if chAl==1:
        dd = 'chAl_' + corrn + 'aveSeDays_time-1_' + labAll + '_' + sn + '_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_' + corrn + 'aveSeDays_time-1_' + labAll + '_' + sn + '_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dir0)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


 
#%% Compare CA (data-shfl) at time -1 between corr and incorr, plot dists. Do it both across days, and across samples of all days. Also show p value (corr vs incorr) for each day.

#startDayFni19 = 16 # what days is the start day for mouse fni19: amazing changes during learning. I believe he was more posterior. In early days he represents stimulus category, in later days he represents choice. So we go with his choice days for this summary.
subtractShfl = 0 # if 1, plot data-shfl; otherwise plot only data.
alsoPooledSamps = 0 # if 1, plot pooled samps of all days
            
if subtractShfl:
    lab = '% Class accur (data-shfl)'
else:
    lab = '% Class accur (data)'
    
lab1 = 'corr'
lab2 = 'incorr'
colors = ['g','orchid'] #['g','k']


# make plots for allN, inh, exc    
for aie in [0,1,2]:#[0,1,2]: 
    
    if aie==0:
        labN = labAll
    elif aie==1:
        labN = 'inh'
    elif aie==2:
        labN = 'exc'
          
        
    for im in range(len(mice)):    
    
        days2an = range(numDaysGood[im])
        if im==3: # mouse fni19: amazing changes during learning. I believe he was more posterior. In early days he represents stimulus category, in later days he represents choice. So we go with his choice days for this summary.
            days2an = np.arange(startDayFni19,numDaysGood[im])
            
        if alsoPooledSamps:
            plt.figure(figsize=(7,4.5))    
            gs = gridspec.GridSpec(2, 5)#, width_ratios=[2, 1])     
            rn = 1
        else:
            plt.figure(figsize=(6,2))    
            gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1])     
            rn = 0
    
    
        ###### days, ave samps
        if aie==0: # allN
            # corr
            a = classAccTest_data_allN_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1] # days
            if subtractShfl:
                a = a - classAccTest_shfl_allN_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1] # days    
            # incorr
            b = classAcc_allN_allDays_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1]
            if subtractShfl:
                b = b - classAcc_allN_shfl_allDays_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1]
                
        elif aie==1: # inh
            # corr
            a = classAccTest_data_inh_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1] # days
            if subtractShfl:
                a = a - classAccTest_shfl_inh_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1] # days    
            # incorr
            b = classAcc_inh_allDays_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1]
            if subtractShfl:
                b = b - classAcc_inh_shfl_allDays_alig_avSamps_allMice[im][days2an,nPreMin_allMice[im]-1]
                
        elif aie==2: # exc
            # corr
            a = classAccTest_data_exc_alig_avSamps2_allMice[im][days2an,nPreMin_allMice[im]-1] # days
            if subtractShfl:
                a = a - classAccTest_shfl_exc_alig_avSamps2_allMice[im][days2an,nPreMin_allMice[im]-1] # days    
            # incorr
            b = classAcc_exc_allDays_alig_avSamps2_allMice[im][days2an,nPreMin_allMice[im]-1]
            if subtractShfl:
                b = b - classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im][days2an,nPreMin_allMice[im]-1]            
            
    
        r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
        binEvery = r/float(15)    
        _, p = stats.ttest_ind(a.flatten(), b.flatten(), nan_policy='omit')
          
        h1 = gs[rn,0:2]
        h2 = gs[rn,2:3]
    #    lab = '% Class accur (data-shfl)'
        ax1, ax2 = histerrbar(h1,h2,a,b,binEvery,p,lab,colors,ylab='Fraction days',lab1=lab1,lab2=lab2,plotCumsum=0)
        ax1.legend('', frameon=False)
        makeNicePlots(ax1,1)
        
        
        
        ###### all samps of all days
        if alsoPooledSamps:
            if aie==0: # allN
                # corr
                a = classAccTest_data_allN_alig_allMice[im][days2an,nPreMin_allMice[im]-1] # days x samps
                if subtractShfl:
                    a = a - classAccTest_shfl_allN_alig_allMice[im][days2an,nPreMin_allMice[im]-1] # days x samps
                # incorr
                b = classAcc_allN_allDays_alig_allMice[im][days2an,nPreMin_allMice[im]-1]
                if subtractShfl:
                    b = b - classAcc_allN_shfl_allDays_alig_allMice[im][days2an,nPreMin_allMice[im]-1]
        
            if aie==1: # inh
                # corr
                a = classAccTest_data_inh_alig_allMice[im][days2an,nPreMin_allMice[im]-1] # days x samps
                if subtractShfl:
                    a = a - classAccTest_shfl_inh_alig_allMice[im][days2an,nPreMin_allMice[im]-1] # days x samps
                # incorr
                b = classAcc_inh_allDays_alig_allMice[im][days2an,nPreMin_allMice[im]-1]
                if subtractShfl:
                    b = b - classAcc_inh_shfl_allDays_alig_allMice[im][days2an,nPreMin_allMice[im]-1]
                
            if aie==2: # exc
                # corr
                a = classAccTest_data_exc_alig_allMice[im][days2an,nPreMin_allMice[im]-1] # days x samps
                if subtractShfl:
                    a = a - classAccTest_shfl_exc_alig_allMice[im][days2an,nPreMin_allMice[im]-1] # days x samps
                # incorr
                b = classAcc_exc_allDays_alig_allMice[im][days2an,nPreMin_allMice[im]-1]
                if subtractShfl:
                    b = b - classAcc_exc_shfl_allDays_alig_allMice[im][days2an,nPreMin_allMice[im]-1]
                    
                    
            r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
            binEvery = r/float(15)    
            _, p = stats.ttest_ind(a.flatten(), b.flatten(), nan_policy='omit')
              
            h1 = gs[0,0:2]
            h2 = gs[0,2:3]
        #    lab = '% Class accur (data-shfl)'
            ax1, ax2 = histerrbar(h1,h2,a,b,binEvery,p,lab,colors,ylab='Fraction days & samps',lab1=lab1,lab2=lab2,plotCumsum=0)
            ax1.legend(bbox_to_anchor=(.6, 1.1), frameon=False)
            makeNicePlots(ax1,1)    
        
        #    plt.subplots_adjust(wspace=1.05, hspace=.7)
            
            
            ####### p value for each day (across samps)    
            h1 = gs[0,3:5]
            t2,p2  = sci.stats.ttest_ind(a, b, axis=1)
            
            pp = p2+0
            pp[p2>.05] = np.nan
            
            ppc = pp+0
            ppc[t2<0] = np.nan # sig days with corr CA > incorr
            ppi = pp+0
            ppi[t2>0] = np.nan # sig days with incorr CA > corr
            
            pp_f = np.mean(~np.isnan(pp)) # fract sig days, ie. days with different CA for corr and incorr 
            ppc_f = np.mean(~np.isnan(ppc)) # fract sig days that have corr > incorr
            ppi_f = np.mean(~np.isnan(ppi)) # fract sig days that have incorr > corr
            
                
            ax1 = plt.subplot(h1) 
            plt.plot(p2)
            plt.plot(pp, 'ro')
            plt.title('Fract sig days, big corr, big incorr\n%.2f, %.2f, %.2f' %(pp_f, ppc_f, ppi_f))
            makeNicePlots(ax1,1,1)    
            plt.xlabel('Days')
            plt.ylabel('P (ttest; corr vs incorr)')
    
        plt.subplots_adjust(wspace=1.8, hspace=.7)
    
            
        ##%% Save the figure    
        if savefigs:
            days = daysGood_allMice[im]
            dnow = dir0 + mice[im] + '/'
            
            d = os.path.join(svmdir+dnow) #,mousename)       
            if chAl==1:
                dd = 'chAl_' + corrn + 'time-1_dist_classAccur_' + labN + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'time-1_dist_classAccur_' + labN + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)







    
    
