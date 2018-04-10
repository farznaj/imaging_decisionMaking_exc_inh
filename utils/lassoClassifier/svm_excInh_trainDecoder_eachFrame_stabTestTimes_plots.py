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
set_save_p_samps = 0 # set it to 0 if p vals are already saved for the stab mat file under study # set and save p value across samples per day (takes time to be done !!)

doTestingTrs = 1 # if 1 compute classifier performance only on testing trials; otherwise on all trials
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

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

if doTestingTrs:
    nts = 'testingTrs_'
else:
    nts = ''

if normWeights:
    nw = 'wNormed_'
else:
    nw = 'wNotNormed_'
    
    
#%% Set days (all days and good days) for each mouse

days_allMice, numDaysAll, daysGood_allMice, dayinds_allMice, mn_corr_allMice = svm_setDays_allMice(mice, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, regressBins, useEqualTrNums, shflTrsEachNeuron, thTrained)
numDaysGood = np.array([len(daysGood_allMice[im]) for im in range(len(mice))])



#%% Define function for all plots below. Heatmaps of time vs time... showing either CA, or diff from max CA, etc... for exc,inh,allN

# if doAv=1; average will be made across days that is provided in top; if 0, inputs are already averaged across days

def plotAngInhExcAllN(topi0, tope0, topa0, cblab, namf, do4, doCA=0, doAv=0):
    # for the 4th subplot below when plotting stab measure
    def plotstabsp(stab_inhm, stab_excm, stab_allNm, yls, ax):        
        ax.plot(time_aligned_allMice[im], stab_inhm, color='r')
        ax.plot(time_aligned_allMice[im], stab_excm, color='b')
        ax.plot(time_aligned_allMice[im], stab_allNm, color='k')          
#        plt.colorbar(ax=ax)    
        ax.set_xlim(s1[im].get_xlim())
        ax.set_xlabel('Testing t (ms)')
        ax.set_ylabel(yls) # Duration(ms) w decoders close to optimal decoder at time t      
    #    s4[im].set_xticks(s3[im].get_xticks())
    #    makeNicePlots(s4[im], 1, 1)      

    def plotstabsp_sd(stab_inhm, stab_excm, stab_allNm, yls, ax, stab_inh_sdm, stab_exc_sdm, stab_allN_sdm):
        ax.fill_between(time_aligned_allMice[im], stab_inhm-stab_inh_sdm, stab_inhm+stab_inh_sdm, color='r', alpha=.5)
        ax.fill_between(time_aligned_allMice[im], stab_excm-stab_exc_sdm, stab_excm+stab_exc_sdm, color='b', alpha=.5)
        ax.fill_between(time_aligned_allMice[im], stab_allNm-stab_allN_sdm, stab_allNm+stab_allN_sdm, color='k', alpha=.5)
        ax.plot(time_aligned_allMice[im], stab_inhm, color='r')
        ax.plot(time_aligned_allMice[im], stab_excm, color='b')
        ax.plot(time_aligned_allMice[im], stab_allNm, color='k')
#        plt.colorbar(ax=ax)    
        ax.set_xlim(s1[im].get_xlim())
        ax.set_xlabel('Testing t (ms)')
        ax.set_ylabel(yls)        
    
    #linstyl = '-','-.',':'
    cols = 'black', 'gray', 'silver'
#    cols = 'g', 'm', 'orange'
#    colsi = 'mediumblue', 'blue', 'cyan'
#    colse = 'red', 'tomato', 'lightsalmon'

    s1 = []; s2 = []; s3 = []; s4 = []; s5 = []; s6 = []
    
    for im in range(len(mice)):    
        
        if doAv:
            topi = np.nanmean(topi0[im], axis=0) 
            tope = np.nanmean(tope0[im], axis=0) 
            top = np.nanmean(topa0[im], axis=0) 
            # se
            topisd = np.nanstd(topi0[im], axis=0) / np.sqrt(numDaysGood[im])
            topesd = np.nanstd(tope0[im], axis=0) / np.sqrt(numDaysGood[im])
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
    
       
        plt.figure(figsize=(10,10)); # 7,6
        
        ################# inh
        plt.subplot(334);     
        lab = 'inh' 
    #    cblab = '' # 'Class accuracy (%)'    
        yl = 'Decoder training t (ms)'
        xl = 'Testing t (ms)'  
        
        plotAng(topi, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
#        plt.plot([time_aligned[-1],time_aligned[0]], [time_aligned[-1],time_aligned[0]]) # mark the diagonal
        plt.xlim(xlim)        
        plt.ylim(ylim)        
        s1.append(plt.gca());
#        print plt.gca().get_position()
#        sz = plt.gcf().get_size_inches()*fig.dpi
        if doCA==1: # Mark a few time points (their CA will be plotted in subplot 224)
            fr2ana = [nPreMin-1, nPreMin-7, nPreMin+5]
            time2ana = np.round(regressBins*frameLength*np.array([-1,-7,5]))
            cnt = -1
            for fr2an in fr2ana:               
                cnt = cnt+1
                ##### If below you plot decoder trained at fr2an, see how it does on all other times
                plt.axhline(time_aligned[fr2an],np.min(time_aligned),np.max(time_aligned), color=cols[cnt])        
    

        ############### exc
        plt.subplot(337);    
        lab = 'exc' 
    #    cblab = 'Class accuracy (%)'    
#        yl = 'Decoder trained at t (ms)'
#        xl = 'Decoder tested at t (ms)'  
        
        plotAng(tope, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
#        plt.plot([time_aligned[-1],time_aligned[0]], [time_aligned[-1],time_aligned[0]]) # mark the diagonal
        plt.xlim(xlim)        
        plt.ylim(ylim)
        s3.append(plt.gca());
        if doCA==1: # Mark a few time points (their CA will be plotted in subplot 224)
            fr2ana = [nPreMin-1, nPreMin-7, nPreMin+5]
            cnt = -1
            for fr2an in fr2ana:               
                cnt = cnt+1
                ##### If below you plot decoder trained at fr2an, see how it does on all other times
                plt.axhline(time_aligned[fr2an],np.min(time_aligned),np.max(time_aligned), color=cols[cnt])        
        
        ################# allN
        plt.subplot(331);     
        lab = 'allN'    
#        cmin = np.nanmin(top)
#        cmax = np.nanmax(top)
    #    cblab = 'Class accuracy (%)'    
#        yl = '' #'Decoder trained at t (ms)'
#        xl = 'Decoder tested at t (ms)'          
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab, xl, yl)    
#        plt.plot([time_aligned[-1],time_aligned[0]], [time_aligned[-1],time_aligned[0]]) # mark the diagonal
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
                
            
            
        #################### show example traces for a few time points
        if do4: 
            ax4 = plt.subplot(332);
            s4.append(plt.gca());
            if doCA==1:
                pos1 = ax4.get_position() # get the original position 
                pos2 = [pos1.x0, pos1.y0 + 0.1,  pos1.width / 1.7, pos1.height / 1.7] 
                ax4.set_position(pos2) # set a new position
                ##### allN #####
#                top = classAcc_allN_allDays_alig_avSamps_avDays_allMice[im] # nTrainedFrs x nTestingFrs   # plot the actual class accuracies instead of the drop in CA
#                topsd = classAcc_allN_allDays_alig_avSamps_sdDays_allMice[im] / np.sqrt(numDaysGood[im])
                cnt = -1    
                # plot class accuracy tested at all times (x axis) when decoder was trained at time fr2an
                for fr2an in fr2ana:       
                    cnt = cnt+1                   
                    ##### Decoder trained at fr2an, see how it does on all other times
                    plt.fill_between(time_aligned, top[fr2an] - topsd[fr2an], top[fr2an] + topsd[fr2an], color=cols[cnt], alpha=.7)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.plot(time_aligned, top[fr2an] , color=cols[cnt])
                    plt.xlabel('Decoder testing t (ms)')               
                    plt.vlines(time_aligned[fr2an],np.min(top),np.max(top), color=cols[cnt])#, linestyle=linstyl[cnt])                         
                    
                    # shift times, so 0 means testing time RELATIVE TO training time
#                    plt.fill_between(time_aligned - time_aligned[fr2an], top[fr2an] - topsd[fr2an], top[fr2an] + topsd[fr2an], color=cols[cnt], alpha=.7)                    
#                    plt.plot(time_aligned - time_aligned[fr2an], top[fr2an] , color=cols[cnt])
#                    plt.xlabel('Decoder testing t rel. training t (ms)')                    
                    
                    ##### Decoders tested at fr2an, see how well decoders trained at different times do
#                    plt.fill_between(time_aligned, top[:,fr2an] - topsd[:,fr2an], top[:,fr2an] + topsd[:,fr2an], color=cols[cnt], alpha=.5)
#                    plt.xlabel('Decoder training t (ms)')                    
            #        plt.plot(time_aligned, topi[fr2an], color=colsi[cnt])#, linestyle=linstyl[cnt])
            #        plt.plot(time_aligned, tope[fr2an], color=colse[cnt])#, linestyle=linstyl[cnt])                            
                plt.ylabel(cblab)
                plt.xlim(xlim)
                plt.title('allN')
#                plt.colorbar()                
                makeNicePlots(plt.gca(), 1, 1)                  

                
                ##### inh #####
                ax5 = plt.subplot(335);
                s5.append(plt.gca());
                pos1 = ax5.get_position() # get the original position 
                pos2 = [pos1.x0, pos1.y0 + 0.1,  pos1.width / 1.7, pos1.height / 1.7] 
                ax5.set_position(pos2) # set a new position                
                cnt = -1    
                # plot class accuracy tested at all times (x axis) when decoder was trained at time fr2an
                for fr2an in fr2ana:       
                    cnt = cnt+1                   
                    ##### Decoder trained at fr2an, see how it does on all other times
                    plt.fill_between(time_aligned, topi[fr2an] - topisd[fr2an], topi[fr2an] + topisd[fr2an], color=cols[cnt], alpha=.7)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.plot(time_aligned, topi[fr2an] , color=cols[cnt])
                    plt.xlabel('Decoder testing t (ms)')                    
                    plt.vlines(time_aligned[fr2an],np.min(topi),np.max(topi), color=cols[cnt])#, linestyle=linstyl[cnt])
                    
                    # shift times, so 0 means testing time RELATIVE TO training time
#                    plt.fill_between(time_aligned - time_aligned[fr2an], topi[fr2an] - topisd[fr2an], topi[fr2an] + topisd[fr2an], color=cols[cnt], alpha=.7)                    
#                    plt.plot(time_aligned - time_aligned[fr2an], topi[fr2an] , color=cols[cnt])
#                    plt.xlabel('Decoder testing t rel. training t (ms)')                    
                plt.ylabel(cblab)
                plt.xlim(xlim)
                plt.title('inh')
#                plt.colorbar()                
                makeNicePlots(plt.gca(), 1, 1)  


                ##### exc #####
                ax6 = plt.subplot(338);
                s6.append(plt.gca());    
                pos1 = ax6.get_position() # get the original position 
                pos2 = [pos1.x0, pos1.y0 + 0.1,  pos1.width / 1.7, pos1.height / 1.7] 
                ax6.set_position(pos2) # set a new position                            
                cnt = -1    
                # plot class accuracy tested at all times (x axis) when decoder was trained at time fr2an
                for fr2an in fr2ana:       
                    cnt = cnt+1                   
                    ##### Decoder trained at fr2an, see how it does on all other times
                    plt.fill_between(time_aligned, tope[fr2an] - topesd[fr2an], tope[fr2an] + topesd[fr2an], color=cols[cnt], alpha=.7)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.plot(time_aligned, tope[fr2an] , color=cols[cnt])
                    plt.xlabel('Decoder testing t (ms)')                    
                    plt.vlines(time_aligned[fr2an],np.min(tope),np.max(tope), color=cols[cnt])#, linestyle=linstyl[cnt])
                    
                    # shift times, so 0 means testing time RELATIVE TO training time
#                    plt.fill_between(time_aligned - time_aligned[fr2an], tope[fr2an] - topesd[fr2an], tope[fr2an] + topesd[fr2an], color=cols[cnt], alpha=.7)                    
#                    plt.plot(time_aligned - time_aligned[fr2an], tope[fr2an] , color=cols[cnt])
#                    plt.xlabel('Decoder testing t rel. training t (ms)')                                        
                plt.ylabel(cblab)
                plt.xlim(xlim)
                plt.title('exc')
#                plt.colorbar()                                    
                makeNicePlots(plt.gca(), 1, 1)                  

                
                ######## Now inh,exc,allN superimposed ########
                spn = 333, 336, 339
                for ispn in [1,0,2]:
                    plt.subplot(spn[ispn]);
                    fr2an = fr2ana[ispn] ##### Decoder trained at fr2an, see how it does on all other times
                    # allN 
                    plt.fill_between(time_aligned, top[fr2an] - topsd[fr2an], top[fr2an] + topsd[fr2an], color='k', alpha=.5)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.plot(time_aligned, top[fr2an] , color='k', label='allN')
                    # inh 
                    plt.fill_between(time_aligned, topi[fr2an] - topisd[fr2an], topi[fr2an] + topisd[fr2an], color='r', alpha=.5)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.plot(time_aligned, topi[fr2an] , color='r', label='inh')
                    # exc 
                    plt.fill_between(time_aligned, tope[fr2an] - topesd[fr2an], tope[fr2an] + topesd[fr2an], color='b', alpha=.5)              #        plt.errorbar(time_aligned, top[fr2an], topsd[fr2an], color=cols[cnt])#, linestyle=linstyl[cnt])
                    plt.plot(time_aligned, tope[fr2an] , color='b', label='exc')                
                    plt.xlabel('Decoder testing t (ms)')                    
                    plt.vlines(time_aligned[fr2an],np.min(top[fr2an]),np.max(top[fr2an]), color=cols[1])#, linestyle=linstyl[cnt])
                    plt.ylabel(cblab)
                    plt.xlim(xlim)
                    plt.title('Training t %d ms' %(time2ana[ispn]))
    #                plt.colorbar()                                    
                    makeNicePlots(plt.gca(), 1, 1)         
                    plt.legend(loc=0, frameon=False) #loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
                
            else: # plot stab measures                    
                if doCA==0:                
                    plotstabsp(stab_inh[im], stab_exc[im], stab_allN[im], yls, plt.gca())
                    # mark time 0 in the above plots
#                    s1[im].axhline(0,0,len(time_aligned))
#                    s2[im].axhline(0,0,len(time_aligned))
#                    s3[im].axhline(0,0,len(time_aligned))  
                    
                elif doCA==-1: # computed across samps for each day... so we have sd
                    plotstabsp_sd(stab_inh[im], stab_exc[im], stab_allN[im], yls, plt.gca(), stab_inh_se[im], stab_exc_se[im], stab_allN_se[im])
                
                makeNicePlots(plt.gca(), 1, 1)        
        
        plt.subplots_adjust(hspace=.4, wspace=.6)
    

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


#%% Define function to plot error bars of stability duration showing summary of all mice

def plotErrBarStabDur(doSe=0):
    
    # average stabDur across all frames
    stab_inh_avFrs = [np.nanmean(stab_inh[im]) for im in range(len(mice))]
    stab_exc_avFrs = [np.nanmean(stab_exc[im]) for im in range(len(mice))]
    stab_allN_avFrs = [np.nanmean(stab_allN[im]) for im in range(len(mice))]
    # stabDur at time -1
    stab_inhM1 = [stab_inh[im][nPreMin_allMice[im]-1] for im in range(len(mice))]
    stab_excM1 = [stab_exc[im][nPreMin_allMice[im]-1] for im in range(len(mice))]
    stab_allNM1 = [stab_allN[im][nPreMin_allMice[im]-1] for im in range(len(mice))]
    
    gp = .1
    plt.figure(figsize=(5,3))
    
    plt.subplot(121) # average across frames
    plt.errorbar(range(len(mice)), stab_inh_avFrs, marker='o',color='r', fmt='.')
    plt.errorbar(np.arange(len(mice))+gp, stab_exc_avFrs, marker='o',color='b', fmt='.')
    plt.errorbar(np.arange(len(mice))+gp*2, stab_allN_avFrs, marker='o',color='k', fmt='.')
    plt.ylabel('%s\naverage times' %(yls))
    plt.xticks(range(len(mice)), mice)
    plt.xlim([-.5, len(mice)-.5])    
    makeNicePlots(plt.gca())
    
    plt.subplot(122) # time -1
    plt.errorbar(range(len(mice)), stab_inhM1, marker='o',color='r', fmt='.')
    plt.errorbar(np.arange(len(mice))+gp, stab_excM1, marker='o',color='b', fmt='.')
    plt.errorbar(np.arange(len(mice))+gp*2, stab_allNM1, marker='o',color='k', fmt='.')
    if doSe: # se across days; this is for the case 'classAccurWin1sdMax_samps' 
        stab_inh_seM1 = [stab_inh_se[im][nPreMin_allMice[im]-1] for im in range(len(mice))]
        stab_exc_seM1 = [stab_exc_se[im][nPreMin_allMice[im]-1] for im in range(len(mice))]
        stab_allN_seM1 = [stab_allN_se[im][nPreMin_allMice[im]-1] for im in range(len(mice))]
            
        plt.errorbar(range(len(mice)), stab_inhM1, stab_inh_seM1, marker='o',color='r', fmt='.')
        plt.errorbar(np.arange(len(mice))+gp, stab_excM1, stab_exc_seM1, marker='o',color='b', fmt='.')
        plt.errorbar(np.arange(len(mice))+gp*2, stab_allNM1, stab_allN_seM1, marker='o',color='k', fmt='.')
    
    plt.ylabel('time -1')
    plt.xticks(range(len(mice)), mice)
    plt.xlim([-.5, len(mice)-.5])    
    makeNicePlots(plt.gca())
    
    plt.subplots_adjust(wspace=.5)
    
    if savefigs:
        if chAl:
            cha = 'chAl_'
        else:
            cha = 'stAl_'
        
        fnam = 'testTrainDiffTimes_sum_' + namf + '_' + labAll + '_' + '_'.join(mice) + '_' + nowStr
        d = os.path.join(svmdir+dir0) #,mousename)       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
            
        fign = os.path.join(d, suffn[0:5]+cha+fnam+'.'+fmt[0])    
        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



#%% Load matlab vars

################################################################################
################################################################################
    
##%% Load class accur vars (averages of samples)

cnam = glob.glob(os.path.join(saveDir_allMice, 'stability_decoderTestedAllTimes_' + nts + nw + '_'.join(mice) + '_*'))
cnam = sorted(cnam, key=os.path.getmtime) # sort pnevFileNames by date (descending)
cnam = cnam[::-1][0] # so the latest file is the 1st one.
print cnam

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
eventI_ds_allDays_allMice = [eventI_ds_allDays_allMice[im].flatten() for im in range(len(mice))]

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

      
classAcc_allN_allDays_alig_avSamps_allMice0 = classAcc_allN_allDays_alig_avSamps_allMice
classAcc_inh_allDays_alig_avSamps_allMice0 = classAcc_inh_allDays_alig_avSamps_allMice
classAcc_exc_allDays_alig_avSamps_allMice0 = classAcc_exc_allDays_alig_avSamps_allMice


#%% Set some vars

numExcSamps = np.shape(classAcc_exc_allDays_alig_avSamps_allMice[0])[-1]
numSamps = 50 # this is the usual value, get it from classAcc_allN_allDays_alig_allMice[im].shape[-1]

nowStr_allMice = [cnam[-17:-4]] * len(mice)


#%% Set time_aligned for all mice

time_aligned_allMice = []
nPreMin_allMice = []   
 
for im in range(len(mice)):  
    time_aligned, nPreMin, nPostMin = set_nprepost(classAcc_allN_allDays_alig_avSamps_allMice[im], eventI_ds_allDays_allMice[im], mn_corr_allMice[im], thTrained, regressBins, traceAl=1)    
    
    time_aligned_allMice.append(time_aligned)
    nPreMin_allMice.append(nPreMin)


lenTraces_AllMice = np.array([len(time_aligned_allMice[im]) for im in range(len(mice))])


              
#%% Set or load p values: across samples per day: is data different froms shfl (done for each pair of testing and training frames.)
# (takes time to be done !!)
# the measure is OK except we dont know what to do with exc bc it has exc samps... ... there is no easy way to set a threshold for p value that works for both inh and exc... given that exc is averaged across exc samps!
# you can perhaps do some sort of bootstrap!

tail = 'right' # tail = '': both tails  # right-tail ttest: data > shfl

if tail=='right':
    nta = 'rTail_'
else:
    nta = ''
    

if set_save_p_samps == 0 :
    print 'Loading p values ....'
    
    ######### data vs shfl for each time pair
    pnam = glob.glob(os.path.join(saveDir_allMice, 'pval_' + nta + 'stability_decoderTestedAllTimes_' + nts + nw + '_'.join(mice) + '_*'))[0] #glob.glob(os.path.join(svmdir+dir0, '*pval_*_svm_testEachDecoderOnAllTimes*'))[0]
    print pnam
    
    data = scio.loadmat(pnam)    
    p_inh_samps_allMice = data.pop('p_inh_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
    p_exc_samps_allMice = data.pop('p_exc_samps_allMice').flatten() # nMice; each mouse: days x frs x frs x excSamps
    p_allN_samps_allMice = data.pop('p_allN_samps_allMice').flatten() # nMice; each mouse: days x frs x frs

    ######### each time pair vs diagonal
    pnam = glob.glob(os.path.join(saveDir_allMice, 'pval_vsDiag_' + 'stability_decoderTestedAllTimes_' + nts + nw + '_'.join(mice) + '_*'))[0] #glob.glob(os.path.join(svmdir+dir0, '*pval_*_svm_testEachDecoderOnAllTimes*'))[0]
    print pnam
    
    data = scio.loadmat(pnam)    
    p_inh_diag_samps_allMice = data.pop('p_inh_diag_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
    p_exc_diag_samps_allMice = data.pop('p_exc_diag_samps_allMice').flatten() # nMice; each mouse: days x frs x frs x excSamps
    p_allN_diag_samps_allMice = data.pop('p_allN_diag_samps_allMice').flatten() # nMice; each mouse: days x frs x frs
    
else:
    
    data = scio.loadmat(cnam, variable_names=['classAcc_inh_allDays_alig_allMice', 'classAcc_allN_allDays_alig_allMice', 'classAcc_exc_allDays_alig_allMice',
                                              'classAcc_inh_shfl_allDays_alig_allMice', 'classAcc_allN_shfl_allDays_alig_allMice', 'classAcc_exc_shfl_allDays_alig_allMice'])    
    
    classAcc_inh_allDays_alig_allMice = data.pop('classAcc_inh_allDays_alig_allMice').flatten() 
    classAcc_allN_allDays_alig_allMice = data.pop('classAcc_allN_allDays_alig_allMice').flatten() 
    classAcc_exc_allDays_alig_allMice = data.pop('classAcc_exc_allDays_alig_allMice').flatten() 
    classAcc_inh_shfl_allDays_alig_allMice = data.pop('classAcc_inh_shfl_allDays_alig_allMice').flatten() 
    classAcc_allN_shfl_allDays_alig_allMice = data.pop('classAcc_allN_shfl_allDays_alig_allMice').flatten() 
    classAcc_exc_shfl_allDays_alig_allMice = data.pop('classAcc_exc_shfl_allDays_alig_allMice').flatten()     


    ###########################################################################
    ##%% ttest (each day, across samples) between data and shuffled for each time pair
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
                    p_inh[ifr1,ifr2] = ttest2(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)], tail=tail)
            
                    # allN
                    a = classAcc_allN_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    b = classAcc_allN_shfl_allDays_alig_allMice[im][iday,ifr1,ifr2,:]
                    p_allN[ifr1,ifr2] = ttest2(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)], tail=tail)
                    
                    # exc ... compute p value across (trial) samps for each excSamp
                    for iexc in range(numExcSamps):
    #                    print iexc
                        a = classAcc_exc_allDays_alig_allMice[im][iday,ifr1,ifr2,iexc,:]
                        b = classAcc_exc_shfl_allDays_alig_allMice[im][iday,ifr1,ifr2,iexc,:]
                        p_exc[ifr1,ifr2,iexc] = ttest2(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)], tail=tail)
        
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
    pnam = os.path.join(saveDir_allMice, 'pval_' + nta + os.path.basename(cnam))
    scio.savemat(pnam, {'p_inh_samps_allMice':p_inh_samps_allMice, 'p_exc_samps_allMice':p_exc_samps_allMice, 'p_allN_samps_allMice':p_allN_samps_allMice})



    ###########################################################################
    ## Do ttest between max CA(trained and tested at ts) and CA(tested at ts but trained at tr)
        # do this for each day, across samps
    
    alph = .05 # only if CA(ts,ts) is above chance we run ttest, otherwise we set p value to nan.
    
    p_inh_diag_samps_allMice = []
    p_allN_diag_samps_allMice = []
    p_exc_diag_samps_allMice = []
    p_exc_diag_samps2_allMice = []
    
    for im in range(len(mice)):
        
        nfrs = len(time_aligned_allMice[im])    
        
        p_inh_samps_allDays = []
        p_allN_samps_allDays = []
        p_exc_samps_allDays = []
        p_exc_samps2_allDays = []
            
        for iday in range(numDaysGood[im]):
            p_inh = np.full((nfrs,nfrs), np.nan)
            p_exc = np.full((nfrs,nfrs,numExcSamps), np.nan)
            p_allN = np.full((nfrs,nfrs), np.nan)
                
            for ts in range(nfrs): # testing time point
        
                # run p value against the CA(trained and tested at ts)
                bi = classAcc_inh_allDays_alig_allMice[im][iday,ts,ts] # tested and trained on the same time point            
                b = classAcc_allN_allDays_alig_allMice[im][iday,ts,ts]       
                be = np.full((numSamps, numExcSamps), np.nan)
                for iexc in range(numExcSamps):
                    be[:,iexc] = classAcc_exc_allDays_alig_allMice[im][iday,ts,ts,iexc]               
                    
                for tr in range(nfrs): # training time point
                    ##### inh #####
                    if p_inh_samps_allMice[im][iday][ts,ts] <= alph: # only if CA at ts,ts is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                        
                        a = classAcc_inh_allDays_alig_allMice[im][iday,tr,ts]           
                        p_inh[tr,ts] = ttest2(np.array(a)[~np.isnan(a)], np.array(bi)[~np.isnan(bi)])
            #            p_inh[tr,ts] = ttest2(a,bi,tail='left') # a < b
                                   
                    ##### allN #####
                    if p_allN_samps_allMice[im][iday][ts,ts] <= alph: # only if CA at ts,ts is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                        a = classAcc_allN_allDays_alig_allMice[im][iday,tr,ts]            
                        p_allN[tr,ts] = ttest2(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])
            #            p_allN[tr,ts] = ttest2(a,b,tail='left') # a < b
    
    
                    ##### exc #####
                    for iexc in range(numExcSamps):
                        if p_exc_samps_allMice[im][iday][ts,ts,iexc] <= alph: # only if CA at ts,ts is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.   
                            a = classAcc_exc_allDays_alig_allMice[im][iday,tr,ts, iexc]
                            p_exc[tr,ts,iexc] = ttest2(np.array(a)[~np.isnan(a)], np.array(be[:,iexc])[~np.isnan(be[:,iexc])])
                #            p_exc[tr,ts] = ttest2(a,be,tail='left') # a < b
                    p_exc_avSamps = np.nanmean(p_exc,axis=-1)
                    
            # keep vars of all days
            p_inh_samps_allDays.append(p_inh)
            p_allN_samps_allDays.append(p_allN)
            p_exc_samps_allDays.append(p_exc)
            p_exc_samps2_allDays.append(p_exc_avSamps)        
        
        # keep vars of all mice
        p_inh_diag_samps_allMice.append(p_inh_samps_allDays)
        p_allN_diag_samps_allMice.append(p_allN_samps_allDays)
        p_exc_diag_samps_allMice.append(p_exc_samps_allDays)    
        p_exc_diag_samps2_allMice.append(p_exc_samps2_allDays)
    
    p_inh_diag_samps_allMice = np.array(p_inh_diag_samps_allMice)
    p_allN_diag_samps_allMice = np.array(p_allN_diag_samps_allMice)
    p_exc_diag_samps_allMice = np.array(p_exc_diag_samps_allMice)
    p_exc_diag_samps2_allMice = np.array(p_exc_diag_samps2_allMice)
    
    
    pnam = os.path.join(saveDir_allMice, 'pval_vsDiag_' + os.path.basename(cnam))
    scio.savemat(pnam, {'p_inh_diag_samps_allMice':p_inh_diag_samps_allMice, 'p_exc_diag_samps_allMice':p_exc_diag_samps_allMice, 'p_allN_diag_samps_allMice':p_allN_diag_samps_allMice})



#%% Set averages of CA
################################################################################
################################################################################

##%% exc: average across excSamps for each day (of each mouse)

classAcc_exc_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_allDays_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps2_allMice = [np.mean(classAcc_exc_shfl_allDays_alig_avSamps_allMice[im], axis=-1) for im in range(len(mice))] # nGoodDays x nTrainedFrs x nTestingFrs


#%% Average CA across days for each mouse

#AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

classAcc_allN_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
# shfl
classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice = np.array([np.mean(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
    


######%% SD across days for each mouse

# AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]
    
classAcc_allN_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_allN_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_inh_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_exc_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
# shfl
classAcc_allN_shfl_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_inh_shfl_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_exc_shfl_allDays_alig_avSamps_sdDays_allMice = np.array([np.std(classAcc_exc_shfl_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs


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

"""
def CAfractChangeWhenTrainedAtOtherTimes(top, p_samps): # drop measured as fraction of max CA; in CA when trained at a different time than the testing time
    topcopy = top+0
    topd = np.full((len(top),len(top)), np.nan)
    for ifr in range(len(top)):
        if p_samps[ifr,ifr] <= alph: # do not include time points whose max performance (at ts) is chance, it doesnt make sense to see how good decoders trained at other time points can decode choice at ts.
            for ifr0 in range(len(top)):
                if p_samps[ifr0,ifr] > alph:
#                    print topcopy[ifr0,ifr] 
                    topcopy[ifr0,ifr] = 50
#                    print topcopy[ifr0,ifr]                     
                topd[ifr0,ifr] = topcopy[ifr0,ifr] / topcopy[ifr,ifr]  # this is easier to understand ... show fraction of max performance!                  
    return topd
"""

################## CA - max CA ##################   
#### data
classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_allDays_alig_avSamps_allMice)
classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_allDays_alig_avSamps_allMice)
classAcc_changeFromMax_exc_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_allDays_alig_avSamps_allMice)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice[im][iday] = CAchangeWhenTrainedAtOtherTimes(classAcc_allN_allDays_alig_avSamps_allMice[im][iday], p_allN_samps_allMice[im][iday])
        classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice[im][iday] = CAchangeWhenTrainedAtOtherTimes(classAcc_inh_allDays_alig_avSamps_allMice[im][iday], p_inh_samps_allMice[im][iday])
        for iexc in range(numExcSamps):
            classAcc_changeFromMax_exc_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAchangeWhenTrainedAtOtherTimes\
            (classAcc_exc_allDays_alig_avSamps_allMice[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday][:,:,iexc])

# exc: average across exc samps for each day
classAcc_changeFromMax_exc_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_changeFromMax_exc_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])
    
"""
#### shfl
classAcc_changeFromMax_allN_shfl_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_shfl_allDays_alig_avSamps_allMice)
classAcc_changeFromMax_inh_shfl_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_shfl_allDays_alig_avSamps_allMice)
classAcc_changeFromMax_exc_shfl_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_shfl_allDays_alig_avSamps_allMice)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_changeFromMax_allN_shfl_allDays_alig_avSamps_allMice[im][iday] = CAchangeWhenTrainedAtOtherTimes(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im][iday], p_allN_samps_allMice[im][iday])
        classAcc_changeFromMax_inh_shfl_allDays_alig_avSamps_allMice[im][iday] = CAchangeWhenTrainedAtOtherTimes(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im][iday], p_inh_samps_allMice[im][iday])
        for iexc in range(numExcSamps):
            classAcc_changeFromMax_exc_shfl_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAchangeWhenTrainedAtOtherTimes\
            (classAcc_exc_shfl_allDays_alig_avSamps_allMice[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday][:,:,iexc])

# exc: average across exc samps for each day
classAcc_changeFromMax_exc_shfl_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_changeFromMax_exc_shfl_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])
"""


################## fract max ##################
#### data
classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_allDays_alig_avSamps_allMice0)
classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_allDays_alig_avSamps_allMice0)
classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_allDays_alig_avSamps_allMice0)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_allN_allDays_alig_avSamps_allMice0[im][iday], p_allN_samps_allMice[im][iday])
        classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_inh_allDays_alig_avSamps_allMice0[im][iday], p_inh_samps_allMice[im][iday])
        for iexc in range(numExcSamps):
            classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAfractChangeWhenTrainedAtOtherTimes\
            (classAcc_exc_allDays_alig_avSamps_allMice0[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday][:,:,iexc])

# exc: average across exc samps for each day
classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])

"""
#### shfl
classAcc_fractChangeFromMax_allN_shfl_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_shfl_allDays_alig_avSamps_allMice)
classAcc_fractChangeFromMax_inh_shfl_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_shfl_allDays_alig_avSamps_allMice)
classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_shfl_allDays_alig_avSamps_allMice)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_fractChangeFromMax_allN_shfl_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_allN_shfl_allDays_alig_avSamps_allMice[im][iday], p_allN_samps_allMice[im][iday])
        classAcc_fractChangeFromMax_inh_shfl_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes(classAcc_inh_shfl_allDays_alig_avSamps_allMice[im][iday], p_inh_samps_allMice[im][iday])
        for iexc in range(numExcSamps):
            classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAfractChangeWhenTrainedAtOtherTimes\
            (classAcc_exc_shfl_allDays_alig_avSamps_allMice[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday][:,:,iexc])

# exc: average across exc samps for each day
classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])


#### data - shfl
classAcc_fractChangeFromMax_allN_dms_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_allN_allDays_alig_avSamps_allMice)
classAcc_fractChangeFromMax_inh_dms_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_inh_allDays_alig_avSamps_allMice)
classAcc_fractChangeFromMax_exc_dms_allDays_alig_avSamps_allMice = copy.deepcopy(classAcc_exc_allDays_alig_avSamps_allMice)
# outputs are: days x frs x frs  
for im in range(len(mice)):
    for iday in range(numDaysGood[im]):
        classAcc_fractChangeFromMax_allN_dms_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes\
            (classAcc_allN_allDays_alig_avSamps_allMice[im][iday] - classAcc_allN_shfl_allDays_alig_avSamps_allMice[im][iday] , p_allN_samps_allMice[im][iday])
        
        classAcc_fractChangeFromMax_inh_dms_allDays_alig_avSamps_allMice[im][iday] = CAfractChangeWhenTrainedAtOtherTimes\
            (classAcc_inh_allDays_alig_avSamps_allMice[im][iday] - classAcc_inh_shfl_allDays_alig_avSamps_allMice[im][iday] , p_inh_samps_allMice[im][iday])
            
        for iexc in range(numExcSamps):
            classAcc_fractChangeFromMax_exc_dms_allDays_alig_avSamps_allMice[im][iday,:,:,iexc] = CAfractChangeWhenTrainedAtOtherTimes\
            (classAcc_exc_allDays_alig_avSamps_allMice[im][iday][:,:,iexc] - classAcc_exc_shfl_allDays_alig_avSamps_allMice[im][iday][:,:,iexc], p_exc_samps_allMice[im][iday][:,:,iexc])

# exc: average across exc samps for each day
classAcc_fractChangeFromMax_exc_dms_allDays_alig_avSamps2_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_exc_dms_allDays_alig_avSamps_allMice[im],axis=-1) for im in range(len(mice))])
"""


#%% Average fract max CA across days for each mouse

# AVERAGE only the last 10 days! in case naive vs trained makes a difference!
#[np.max([-10,-len(days_allMice[im])]):]

# change
classAcc_changeFromMax_allN_allDays_alig_avSamps_avDays_allMice = np.array([np.nanmean(classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_changeFromMax_inh_allDays_alig_avSamps_avDays_allMice = np.array([np.nanmean(classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_changeFromMax_exc_allDays_alig_avSamps_avDays_allMice = np.array([np.nanmean(classAcc_changeFromMax_exc_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs

# fract max
classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_avDays_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_avDays_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs
classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_avDays_allMice = np.array([np.nanmean(classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice[im], axis=0) for im in range(len(mice))]) # nTrainedFrs x nTestingFrs



#%% PLOTS of each mouse
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
   
    
#%% Plot day-averaged heatmaps of CA, CA-max(CA), CA/max(CA) when decoder is trained and tested at all time points (only plot data)

cblab012 = 'Class accuracy (%)', 'Class accuracy - max (%)', 'Class accuracy / max' #'Fract change in max class accuracy' 
#### Data
figns = 'classAccur', 'classAccurMinusMax', 'classAccurFractMax'

# plot class accuracies
plotAngInhExcAllN(classAcc_inh_allDays_alig_avSamps_allMice, classAcc_exc_allDays_alig_avSamps2_allMice, classAcc_allN_allDays_alig_avSamps_allMice, cblab012[0], figns[0], 1, 1, 1)


# plot class accur as a fraction of max class accuracy (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice, classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice, classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice, cblab012[2], figns[2], 1, 1, 1)


# plot change in class accuracies rel2 max (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_changeFromMax_inh_allDays_alig_avSamps_allMice, classAcc_changeFromMax_exc_allDays_alig_avSamps2_allMice, classAcc_changeFromMax_allN_allDays_alig_avSamps_allMice, cblab012[1], figns[1], 1, 1, 1)


"""
#### Shfl
figns = 'classAccur_shfl', 'classAccurMinusMax_shfl', 'classAccurFractMax_shfl'

# plot class accuracies
plotAngInhExcAllN(classAcc_inh_shfl_allDays_alig_avSamps_allMice, classAcc_exc_shfl_allDays_alig_avSamps2_allMice, classAcc_allN_shfl_allDays_alig_avSamps_allMice, cblab012[0], figns[0], 1, 1, 1)


# plot class accur as a fraction of max class accuracy (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_fractChangeFromMax_inh_shfl_allDays_alig_avSamps_allMice, classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps2_allMice, classAcc_fractChangeFromMax_allN_shfl_allDays_alig_avSamps_allMice, cblab012[2], figns[2], 1, 1, 1)


# plot change in class accuracies rel2 max (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_changeFromMax_inh_shfl_allDays_alig_avSamps_allMice, classAcc_changeFromMax_exc_shfl_allDays_alig_avSamps2_allMice, classAcc_changeFromMax_allN_shfl_allDays_alig_avSamps_allMice, cblab012[1], figns[1], 1, 1, 1)



#### data - shfl
figns = 'classAccurFractMax_dms'

# plot class accur as a fraction of max class accuracy (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_fractChangeFromMax_inh_dms_allDays_alig_avSamps_allMice, classAcc_fractChangeFromMax_exc_dms_allDays_alig_avSamps2_allMice, classAcc_fractChangeFromMax_allN_dms_allDays_alig_avSamps_allMice, cblab012[2], figns[2], 1, 1, 1)



#### data / shfl
figns = 'classAccurFractMax_ddivs'
# plot class accur as a fraction of max class accuracy (trained on the same t as tested)
# non-sig CAs (rel2 shfl) are set to nan, and not included in the plot.
plotAngInhExcAllN(classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_allMice / classAcc_fractChangeFromMax_inh_shfl_allDays_alig_avSamps_allMice, \
                  classAcc_fractChangeFromMax_exc_allDays_alig_avSamps2_allMice / classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps2_allMice, \
                  classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice / classAcc_fractChangeFromMax_allN_shfl_allDays_alig_avSamps_allMice, cblab012[2], figns[2], 1, 1, 1)
"""


#%%
#############################################################################
########### P value, across days, data vs shfl for each time pair ###########
#############################################################################

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
# Now set the significancy matrix ... CA at which time points is siginificantly different from shuffle.

alph = .001 # p value for significancy

p_inh_allMice_01 = [x <= alph for x in p_inh_allMice]
p_exc_allMice_01 = [x <= alph for x in p_exc_allMice]
p_allN_allMice_01 = [x <= alph for x in p_allN_allMice]


#%% Plots of p value: data vs shfl, across days

# pval
plot_CA_diff_fractDiff = 0
cblab = 'P (data vs shfl)'
namf = 'p_data_shfl'
plotAngInhExcAllN(p_inh_allMice, p_exc_allMice, p_allN_allMice, cblab, namf, 0)

# sig or not
cblab = 'Sig (data vs shfl)'
namf = 'sig_p'+str(alph)[2:]+'_data_shfl'
plotAngInhExcAllN(p_inh_allMice_01, p_exc_allMice_01, p_allN_allMice_01, cblab, namf, 0)




#%%
########################################################################
########### Quantify: Does decoder generalize to other times? ###########
########################################################################
    

#%%
#################################################################  
########### P value (across days): CA(trained and tested at ts) vs ########### 
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
    

################################################ 
# Now set the significancy matrix ... 
################################################
alph = .05 # p value for significancy

# if vars below are 0 it means CA(tested at ts but trained at tr) is NOT significantly different from max CA(trained and tested at ts) => stable
# if vars below are 1 it means CA(tested at ts but trained at tr) IS significantly different from max CA(trained and tested at ts) => unstable
p_inh_diag_allMice_01 = [x <= alph for x in p_inh_diag_allMice]
p_exc_diag_allMice_01 = [x <= alph for x in p_exc_diag_allMice]
p_allN_diag_allMice_01 = [x <= alph for x in p_allN_diag_allMice]


#%% Plot above computed p values (between max CA(trained and tested at ts) and CA(tested at ts but trained at tr))

#### Plot p vals
cblab = 'p (CA diagonal vs t)'
namf = 'p_maxCA_otherTimes'
plotAngInhExcAllN(p_inh_diag_allMice, p_exc_diag_allMice, p_allN_diag_allMice, cblab, namf, 0)

#### Plot significant or not
cblab = 'sig (CA diagonal vs t)'
namf = 'sig_maxCA_otherTimes'
plotAngInhExcAllN(p_inh_diag_allMice_01, p_exc_diag_allMice_01, p_allN_diag_allMice_01, cblab, namf, 0)


#%% Errorbar of the above analysis at time -1: sig_maxCA_otherTimes

# For how long each decoder is stable (ie duration over which decoder can be trained to acheive CA that is NOT significantly different from maxCA
stab_inh = regressBins*frameLength*np.array([np.sum(p_inh_diag_allMice_01[im]==0, axis=0) for im in range(len(mice))])
stab_exc = regressBins*frameLength*np.array([np.sum(p_exc_diag_allMice_01[im]==0, axis=0) for im in range(len(mice))])
stab_allN = regressBins*frameLength*np.array([np.sum(p_allN_diag_allMice_01[im]==0, axis=0) for im in range(len(mice))])
yls = 'Trained decoder stab dur (ms)'
namf = 'sig_maxCA_otherTimes'

# set the sum computed above to nan if p value was nan (otherwise it will be 0)
for im in range(len(mice)):
    stab_inh[im][np.sum(~np.isnan(p_inh_diag_allMice[im]), axis=0)==0] = np.nan
    stab_exc[im][np.sum(~np.isnan(p_exc_diag_allMice[im]), axis=0)==0] = np.nan
    stab_allN[im][np.sum(~np.isnan(p_allN_diag_allMice[im]), axis=0)==0] = np.nan


plotErrBarStabDur()



#%%
############################################################################################  
########### P value (across samps for each day): CA(trained and tested at ts) vs ########### 
###########  CA(tested at ts but trained at tr) ############################################
############################################################################################  

# These p vals are loaded above.

#%%    
################################################ 
# Now set the significancy matrix ... 
################################################
alph = .05 # p value for significancy

# if vars below are 0 it means (for each day) CA(tested at ts but trained at tr) is NOT significantly different from max CA(trained and tested at ts) => stable
# if vars below are 1 it means (for each day) CA(tested at ts but trained at tr) IS significantly different from max CA(trained and tested at ts) => unstable
p_inh_diag_samps_allMice_01 = []
p_allN_diag_samps_allMice_01 = []
p_exc_diag_samps_allMice_01 = []

for im in range(len(mice)):
    p_inh_diag_samps_allMice_01m = np.full(np.shape(p_inh_diag_samps_allMice[im]), np.nan)
    p_allN_diag_samps_allMice_01m = np.full(np.shape(p_allN_diag_samps_allMice[im]), np.nan)
    p_exc_diag_samps_allMice_01m = np.full(np.shape(p_exc_diag_samps_allMice[im]), np.nan)
    
    for iday in range(numDaysGood[im]):
        p_inh_diag_samps_allMice_01m[iday] = [x <= alph for x in p_inh_diag_samps_allMice[im][iday]]
        p_allN_diag_samps_allMice_01m[iday] = [x <= alph for x in p_allN_diag_samps_allMice[im][iday]]
        p_exc_diag_samps_allMice_01m[iday] = [x <= alph for x in p_exc_diag_samps_allMice[im][iday]]
        # set to nan those that are nan in p
        p_inh_diag_samps_allMice_01m[iday][np.isnan(p_inh_diag_samps_allMice[im][iday])] = np.nan
        p_allN_diag_samps_allMice_01m[iday][np.isnan(p_allN_diag_samps_allMice[im][iday])] = np.nan
        p_exc_diag_samps_allMice_01m[iday][np.isnan(p_exc_diag_samps_allMice[im][iday])] = np.nan
        
    p_inh_diag_samps_allMice_01.append(p_inh_diag_samps_allMice_01m)
    p_allN_diag_samps_allMice_01.append(p_allN_diag_samps_allMice_01m)
    p_exc_diag_samps_allMice_01.append(p_exc_diag_samps_allMice_01m)


#%% Average of above values across days [whether for each day CA(tr,ts) is within 1se of CA(ts,ts) across samps]

p_inh_diag_samps_01_aveDays_allMice = [np.nanmean(p_inh_diag_samps_allMice_01[im],axis=0) for im in range(len(mice))]
p_allN_diag_samps_01_aveDays_allMice = [np.nanmean(p_allN_diag_samps_allMice_01[im],axis=0) for im in range(len(mice))]
# exc: average across days and exc samps
p_exc_diag_samps_01_aveDays_allMice = [np.nanmean(p_exc_diag_samps_allMice_01[im],axis=(0,-1)) for im in range(len(mice))]
#sameCA_exc_samps_aveDays_allMice = [np.nanmean(sameCA_exc_samps2_allMice[im],axis=0) for im in range(len(mice))]
# use one random exc samp
#r = rng.permutation(numExcSamps)[0]
#sameCA_exc_samps_aveDays_allMice = [np.nanmean(np.array(sameCA_exc_samps_allMice[im])[:,:,:,r],axis=0) for im in range(len(mice))]

'''
for iday in range(numDaysGood[im]): 
    plt.figure()
    plt.imshow(classAcc_inh_allDays_alig_avSamps_allMice[im][iday]); plt.colorbar()
    plt.imshow(sameCA_exc_samps2_allMice[im][iday])
'''

#%% For how long trained decoders have CA similar to max CA
            # Compute for each day, at each time point, the duration of stability (ie to how many other time points the decoder generalizes)
stab_p_inh_eachDay_allMice = []
stab_p_exc_eachDay_allMice = []
stab_p_allN_eachDay_allMice = []

for im in range(len(mice)):
    stab_inh_eachDay = np.full((numDaysGood[im], np.shape(p_inh_diag_samps_allMice[im])[1]), np.nan)    
    stab_allN_eachDay = np.full((numDaysGood[im], np.shape(p_inh_diag_samps_allMice[im])[1]), np.nan)
    stab_exc_eachDay = np.full((numDaysGood[im], np.shape(p_inh_diag_samps_allMice[im])[1], numExcSamps), np.nan)
    
    for iday in range(numDaysGood[im]):
        # sum across training frames
        stab_inh_eachDay[iday] = regressBins*frameLength * np.sum(p_inh_diag_samps_allMice_01[im][iday]==0, axis=0) # days x testingFrs        
        stab_allN_eachDay[iday] = regressBins*frameLength * np.sum(p_allN_diag_samps_allMice_01[im][iday]==0, axis=0) # days x testingFrs        
        stab_exc_eachDay[iday] = regressBins*frameLength * np.sum(p_exc_diag_samps_allMice_01[im][iday]==0, axis=0) # days x testingFrs x excSamps
        # set to nan those that are nan in p
        stab_inh_eachDay[iday][np.sum(~np.isnan(p_inh_diag_samps_allMice[im][iday]), axis=0)==0] = np.nan
        stab_allN_eachDay[iday][np.sum(~np.isnan(p_allN_diag_samps_allMice[im][iday]), axis=0)==0] = np.nan
        stab_exc_eachDay[iday][np.sum(~np.isnan(p_exc_diag_samps_allMice[im][iday]), axis=0)==0] = np.nan
        
    stab_p_inh_eachDay_allMice.append(stab_inh_eachDay)
    stab_p_allN_eachDay_allMice.append(stab_allN_eachDay)
    stab_p_exc_eachDay_allMice.append(stab_exc_eachDay)

stab_p_inh_eachDay_allMice = np.array(stab_p_inh_eachDay_allMice)
stab_p_allN_eachDay_allMice = np.array(stab_p_allN_eachDay_allMice)
stab_p_exc_eachDay_allMice = np.array(stab_p_exc_eachDay_allMice) # days x testingFrs x excSamps

# Average of fract days with stable decoders across time points (ie its CA when tested on a different time point is within 1 std)
#stab_inh = np.array([np.nanmean(sameCA_inh_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_exc = np.array([np.nanmean(sameCA_exc_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_allN = np.array([np.nanmean(sameCA_allN_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])

    
#%% Errorbar of the above analysis at time -1: stability duration, allmice, ave and se across days. (duration that a decoder can be trained to perform significantly not different from CA(ts,ts)) in time -1 for each day, and plot it across days!

# On average across days, for how long each decoder generalizes.
stab_inh = np.array([np.nanmean(stab_p_inh_eachDay_allMice[im],axis=0) for im in range(len(mice))])
stab_allN = np.array([np.nanmean(stab_p_allN_eachDay_allMice[im],axis=0) for im in range(len(mice))])
# exc: average across days and exc samps
stab_exc = np.array([np.nanmean(stab_p_exc_eachDay_allMice[im],axis=(0,-1)) for im in range(len(mice))])
# exc: average across exc samps
stab_exc_aveExcSamps = np.array([np.nanmean(stab_p_exc_eachDay_allMice[im],axis=-1) for im in range(len(mice))])
# use one random exc samp
#r = rng.permutation(numExcSamps)[0]
#stab_exc = np.array([np.nanmean(np.array(stab_exc_eachDay_allMice[im])[:,:,r],axis=0) for im in range(len(mice))])
"""
nValidDays_inh = np.full(len(mice), np.nan)
nValidDays_allN = np.full(len(mice), np.nan)
nValidDays_exc = np.full((numExcSamps,len(mice)), np.nan)
for im in range(len(mice)):
    a = stab_p_inh_eachDay_allMice[im][:, nPreMin_allMice[im]-1]   
    nValidDays_inh[im] = sum(~np.isnan(a))
    #
    a = stab_p_allN_eachDay_allMice[im][:, nPreMin_allMice[im]-1]   
    nValidDays_allN[im] = sum(~np.isnan(a))
    #
    a = stab_p_exc_eachDay_allMice[im][:, nPreMin_allMice[im]-1]   
    nValidDays_exc[:,im] = sum(~np.isnan(a))    
# average across excSamps
nValidDays_exc = np.round(np.mean(nValidDays_exc, axis=0))
"""

#nPreMin_allMice[im]-1
nValidDays_inh = []
nValidDays_allN = []
nValidDays_exc = []

for im in range(len(mice)):
    nValidDays_inhm = []
    nValidDays_allNm = []
    nValidDays_excm = []

    for ifr in range(stab_p_inh_eachDay_allMice[im].shape[1]):
        
        a = stab_p_inh_eachDay_allMice[im][:, ifr]   
        nValidDays_inhm.append(sum(~np.isnan(a)))
        #
        a = stab_p_allN_eachDay_allMice[im][:, ifr]   
        nValidDays_allNm.append(sum(~np.isnan(a)))
        #
        a = stab_exc_aveExcSamps[im][:, ifr]   
        nValidDays_excm.append(sum(~np.isnan(a)))

    nValidDays_inh.append(nValidDays_inhm)
    nValidDays_allN.append(nValidDays_allNm)
    nValidDays_exc.append(nValidDays_excm)
    
    
# Standard error across days
stab_inh_se = np.array([np.nanstd(stab_p_inh_eachDay_allMice[im],axis=0)/np.sqrt(nValidDays_inh[im]) for im in range(len(mice))])
stab_allN_se = np.array([np.nanstd(stab_p_allN_eachDay_allMice[im],axis=0)/np.sqrt(nValidDays_allN[im]) for im in range(len(mice))])
#stab_exc_se = np.array([np.nanstd(stab_p_exc_eachDay_allMice[im],(axis=0,-1))/np.sqrt(nValidDays_exc[im]) for im in range(len(mice))])
stab_exc_se = np.array([np.nanstd(stab_exc_aveExcSamps[im],axis=0)/np.sqrt(nValidDays_exc[im]) for im in range(len(mice))])

namf = 'sig_maxCA_otherTimes_samps'
plotErrBarStabDur(doSe=1)


#%% Plot of the above analysis [whether for each day CA(tr,ts) is significantly different from CA(ts,ts) across samps]

#### Plot significant or not
cblab = 'Fract sig (CA diagonal vs t) across days'
namf = 'sig_maxCA_otherTimes_samps'
plotAngInhExcAllN(p_inh_diag_samps_01_aveDays_allMice, p_exc_diag_samps_01_aveDays_allMice, p_allN_diag_samps_01_aveDays_allMice, cblab, namf, 1, -1)
    



#%%
###########################################################
########### Is CA(trained at tr, tested at ts) within #####
###### 2 std of CA(trained and tested at ts)?  ############
###########################################################
    
#%% Is CA(trained at tr, tested at ts) within 1 std of CA(trained and tested at ts) 
    # do this across days (ie ave and std across days)

#alph = .05 # if diagonal CA is sig (comapred to shfl), then see if it generalizes or not... othereise dont use that time point for analysis.
fractStd = 2

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

s1,s2,s3,s4 = plotAngInhExcAllN(sameCA_inh_diag_allMice, sameCA_exc_diag_allMice, sameCA_allN_diag_allMice, cblab, namf, 1)


#%% Errorbar of the above analysis at time -1: stability duration

# For how long each decoder is stable (ie duration over which decoder can be trained to acheive CA within 1sd of max performance on the testing time
        # for the following you need to sum over axis=1... but not sure if it makes sense given that above you do the analysis for each column (ie comparing CA(:,ts) w CA(ts,ts))
        ######################## For how long each decoder is stable (ie its CA when tested on a different time point is within 1 std)
stab_inh = regressBins*frameLength*np.array([np.sum(sameCA_inh_diag_allMice[im], axis=0) for im in range(len(mice))])
stab_exc = regressBins*frameLength*np.array([np.sum(sameCA_exc_diag_allMice[im], axis=0) for im in range(len(mice))])
stab_allN = regressBins*frameLength*np.array([np.sum(sameCA_allN_diag_allMice[im], axis=0) for im in range(len(mice))])
yls = 'Trained decoder stab dur (ms)'
namf = 'classAccurWin1sdMax'

plotErrBarStabDur()


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

    

# Get the lower triangle, ie after the testing time point for how long trained decoders will perform well
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

fractStd = 2 #1
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
                if p_inh_samps_allMice[im][iday][ts,ts] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                    a = classAcc_inh_allDays_alig_avSamps_allMice[im][iday,tr,ts]
                    sameCA_inh[tr,ts] = (a >= (maxcai[0]-maxcai[1])) 
                
                # allN
                if p_allN_samps_allMice[im][iday][ts,ts] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
                    a = classAcc_allN_allDays_alig_avSamps_allMice[im][iday,tr,ts]
                    sameCA_allN[tr,ts] = (a >= (maxca[0]-maxca[1])) 
                
                # exc ... 
                for iexc in range(numExcSamps):
                    if p_exc_samps_allMice[im][iday][ts,ts,iexc] <= alph: # only if CA at ifr1,ifr1 is significantly diff from shuffle, we evaluate whether it generalizes to other times as well.
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

'''
for iday in range(numDaysGood[im]): 
    plt.figure()
    plt.imshow(classAcc_inh_allDays_alig_avSamps_allMice[im][iday]); plt.colorbar()
    plt.imshow(sameCA_exc_samps2_allMice[im][iday])
'''


#%% For how long trained decoders have CA similar to max CA
            # Compute for each day, at each time point, the duration of stability (ie to how many other time points the decoder generalizes)
stab_inh_eachDay_allMice = []
stab_allN_eachDay_allMice = []
stab_exc_eachDay_allMice = []

for im in range(len(mice)):
    stab_inh_eachDay = []
    stab_exc_eachDay = []
    stab_allN_eachDay = []
    
    for iday in range(numDaysGood[im]):
        # sum across training frames
        stab_inh_eachDay.append(regressBins*frameLength * np.sum(sameCA_inh_samps_allMice[im][iday], axis=0)) # days x testingFrs        
        stab_allN_eachDay.append(regressBins*frameLength * np.sum(sameCA_allN_samps_allMice[im][iday], axis=0)) # days x testingFrs        
#        stab_exc_eachDay.append(np.sum(sameCA_exc_samps_allMice[im][iday], axis=0)) # days x testingFrs x excSamps
        stab_exc_eachDay.append(regressBins*frameLength * np.sum(sameCA_exc_samps2_allMice[im][iday], axis=0)) # days x testingFrs
        
    stab_inh_eachDay_allMice.append(np.array(stab_inh_eachDay))
    stab_allN_eachDay_allMice.append(np.array(stab_allN_eachDay))
    stab_exc_eachDay_allMice.append(np.array(stab_exc_eachDay))

stab_inh_eachDay_allMice = np.array(stab_inh_eachDay_allMice)
stab_allN_eachDay_allMice = np.array(stab_allN_eachDay_allMice)
stab_exc_eachDay_allMice = np.array(stab_exc_eachDay_allMice)

# Average of fract days with stable decoders across time points (ie its CA when tested on a different time point is within 1 std)
#stab_inh = np.array([np.nanmean(sameCA_inh_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_exc = np.array([np.nanmean(sameCA_exc_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])
#stab_allN = np.array([np.nanmean(sameCA_allN_samps_aveDays_allMice[im], axis=1) for im in range(len(mice))])


#%% Errorbar of the above analysis at time -1: stability duration, allmice, ave and se across days. (duration that a decoder can be trained to perform within 1se of CA(ts,ts)) in time -1 for each day, and plot it across days!

# On average across days, for how long each decoder generalizes.
stab_inh = np.array([np.nanmean(stab_inh_eachDay_allMice[im],axis=0) for im in range(len(mice))]) # nMice; # each mouse: nFrames
stab_allN = np.array([np.nanmean(stab_allN_eachDay_allMice[im],axis=0) for im in range(len(mice))])
# exc: average across days and exc samps
#stab_exc = np.array([np.nanmean(stab_exc_eachDay_allMice[im],axis=(0,-1)) for im in range(len(mice))])
stab_exc = np.array([np.nanmean(stab_exc_eachDay_allMice[im],axis=0) for im in range(len(mice))])
# use one random exc samp
#r = rng.permutation(numExcSamps)[0]
#stab_exc = np.array([np.nanmean(np.array(stab_exc_eachDay_allMice[im])[:,:,r],axis=0) for im in range(len(mice))])

#nPreMin_allMice[im]-1
nValidDays_inh = []
nValidDays_allN = []
nValidDays_exc = []

for im in range(len(mice)):
    nValidDays_inhm = []
    nValidDays_allNm = []
    nValidDays_excm = []
    for ifr in range(stab_inh_eachDay_allMice[im].shape[1]):
        
        a = stab_inh_eachDay_allMice[im][:, ifr]   
        nValidDays_inhm.append(sum(~np.isnan(a)))
        #
        a = stab_allN_eachDay_allMice[im][:, ifr]   
        nValidDays_allNm.append(sum(~np.isnan(a)))
        #
        a = stab_exc_eachDay_allMice[im][:, ifr]   
        nValidDays_excm.append(sum(~np.isnan(a)))

    nValidDays_inh.append(nValidDays_inhm)
    nValidDays_allN.append(nValidDays_allNm)
    nValidDays_exc.append(nValidDays_excm)
    
    
# Standard error across days
stab_inh_se = np.array([np.nanstd(stab_inh_eachDay_allMice[im],axis=0)/np.sqrt(nValidDays_inh[im]) for im in range(len(mice))]) # nMice; # each mouse: nFrames
stab_allN_se = np.array([np.nanstd(stab_allN_eachDay_allMice[im],axis=0)/np.sqrt(nValidDays_allN[im]) for im in range(len(mice))])
stab_exc_se = np.array([np.nanstd(stab_exc_eachDay_allMice[im],axis=0)/np.sqrt(nValidDays_exc[im]) for im in range(len(mice))])

yls = 'Trained decoder stab dur (ms)'
namf = 'classAccurWin'+str(fractStd)+'sdMax_samps'
plotErrBarStabDur(doSe=1)


#%% Plot of the above analysis [whether for each day CA(tr,ts) is within 1se of CA(ts,ts) across samps]

cblab = 'Fract days with high stability'
namf = 'classAccurWin'+str(fractStd)+'sdMax_samps'
plotAngInhExcAllN(sameCA_inh_samps_aveDays_allMice, sameCA_exc_samps_aveDays_allMice, sameCA_allN_samps_aveDays_allMice, cblab, namf, 1, -1)




#%% Plots of stability vs day
####################################################################################
####################################################################################
####################################################################################

#%% Look at the stability duration (duration that a decoder can be trained to perform within 1se of CA(ts,ts)) in time -1 for each day, and plot it across days!

plt.figure()
for im in range(len(mice)):
    a = stab_inh_eachDay_allMice[im][:, nPreMin_allMice[im]-1]   
    
    plt.subplot(2,2,im+1)     
    plt.plot(a, marker='.')
    plt.xlabel('Days')
    plt.xlim([-1, len(a)])
    if np.in1d(im,[0,1]):
        plt.ylabel(yls)
    plt.title(mice[im])
    makeNicePlots(plt.gca(),0,1)
    
plt.subplots_adjust(hspace=0.7, wspace=.5)

############%% Save figure for each mouse
if savefigs:#% Save the figure   

    namf = 'stabDur_classAccurWin'+str(fractStd)+'sdMax_samps'+'_vs_days_'
    
    if chAl==1:
        dd = 'chAl_' + namf + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_' + namf + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dir0)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dir0, suffn[0:5]+dd+'.'+fmt[0])
    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)

    

#%% Define function to look at CA(tr,ts) / CA(ts,ts) (ie max_CA at ts)  across days.... try a bunch of points but eg the following makes sense: go with training at time -1, and testing at -6 

# trt and tst are relative to time right before the choice
def trts(trt,tsta):
    
    plt.figure(figsize=(10, 2.2*len(tsta)))
    for i in range(len(tsta)):
        
        tst = tsta[i]
        
        tr_1_ts_3 = []
        for im in range(len(mice)):
            tr = max(0, nPreMin_allMice[im] + trt)
            ts = min(lenTraces_AllMice[im]-1, nPreMin_allMice[im] + tst)
        
            fractMax = [classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_allMice[im][iday][tr, ts] for iday in range(numDaysGood[im])]
            tr_1_ts_3.append(fractMax)
        
        tr_1_ts_3 = np.array(tr_1_ts_3)
        
        
        #####        
        for im in range(len(mice)):
            plt.subplot(len(tsta), len(mice), len(mice)*i + im+1)
            plt.plot(tr_1_ts_3[im], marker='.')
            plt.xlabel('Days')
            plt.title(mice[im])
            plt.xlim([-1, len(tr_1_ts_3[im])])
            if im==0: #np.in1d(im,[0,1]):
                plt.ylabel('classAccurFractMax\ntrt %d, tst %d' %(trt, tst))
            makeNicePlots(plt.gca(),0,1)
        plt.subplots_adjust(hspace=1, wspace=.5)


    ############%% Save figure for each mouse
    if savefigs:#% Save the figure   
    
        namf = 'classAccurFractMax_vs_days_'
        trsn = 'tr'+str(trt)+'ts'+str(tsta)+'_'

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


#%% Plot CA(tr,ts) / CA(ts,ts) (ie max_CA at ts)  across days.... try a bunch of points but eg the following makes sense: go with training at time -1, and testing at -6  

# times are relative to choice

trt = -7 # -5
tsta = [-3, 0, 2, 6]    
trts(trt,tsta)


trt = -3
tsta = [-1, 0, 2, 6]
trts(trt,tsta)


trt = 0
tsta  = [2, 6]
trts(trt,tsta)


trt = 3
tsta = [6]
trts(trt,tsta)

# train ~600ms before choice, test 100ms before choice




#%% PLOTS of all mice
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

########################## Align traces across all mice ##########################

#%% Set time_aligned and nPreMin for aligning traces of all mice

time_aligned_final, nPreMin_final, nPostMin_final, nPreMin_allMice = set_nprepost_allMice(classAcc_inh_allDays_alig_avSamps_allMice, eventI_ds_allDays_allMice, mn_corr_allMice)


############################ Classification accuracies ############################

#%% Align day-averaged CA traces 

classAcc_allN_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_allN_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
classAcc_inh_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_inh_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
classAcc_exc_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_exc_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
# shfl
classAcc_allN_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_allN_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
classAcc_inh_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_inh_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
classAcc_exc_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_exc_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 


#%% Average day-averaged traces across mice

classAcc_allN_alig_avSamps_avDays_avMice = np.mean(classAcc_allN_alig_avSamps_avDays_allMice_al, axis=0) # frs x frs
classAcc_inh_alig_avSamps_avDays_avMice = np.mean(classAcc_inh_alig_avSamps_avDays_allMice_al, axis=0) 
classAcc_exc_alig_avSamps_avDays_avMice = np.mean(classAcc_exc_alig_avSamps_avDays_allMice_al, axis=0)                    
# shfl
classAcc_allN_shfl_alig_avSamps_avDays_avMice = np.mean(classAcc_allN_shfl_alig_avSamps_avDays_allMice_al, axis=0) # frs x frs
classAcc_inh_shfl_alig_avSamps_avDays_avMice = np.mean(classAcc_inh_shfl_alig_avSamps_avDays_allMice_al, axis=0) 
classAcc_exc_shfl_alig_avSamps_avDays_avMice = np.mean(classAcc_exc_shfl_alig_avSamps_avDays_allMice_al, axis=0)                    

# set se across mice
classAcc_allN_alig_avSamps_avDays_sdMice = np.std(classAcc_allN_alig_avSamps_avDays_allMice_al, axis=0) / np.sqrt(len(mice)) # frs x frs
classAcc_inh_alig_avSamps_avDays_sdMice = np.std(classAcc_inh_alig_avSamps_avDays_allMice_al, axis=0) / np.sqrt(len(mice))
classAcc_exc_alig_avSamps_avDays_sdMice = np.std(classAcc_exc_alig_avSamps_avDays_allMice_al, axis=0) / np.sqrt(len(mice))


#%% Plot heatmaps: mice-averaged of CAs (frs x frs) 

# ttest across mice (day-averaged traces)            
_,pei = stats.ttest_ind(classAcc_inh_alig_avSamps_avDays_allMice_al - classAcc_inh_shfl_alig_avSamps_avDays_allMice_al , classAcc_exc_alig_avSamps_avDays_allMice_al - classAcc_exc_shfl_alig_avSamps_avDays_allMice_al , axis=0) 

fnam = figns[0]+'_AveMice_AveDays_inhExc'+labAll+'_' + '_'.join(mice) + '_' + nowStr   # (np.array(mice)[mice2an_all])

plotAngsAll(nPreMin_final, time_aligned_final, classAcc_inh_alig_avSamps_avDays_avMice, [], classAcc_allN_alig_avSamps_avDays_avMice, classAcc_exc_alig_avSamps_avDays_avMice, 
            classAcc_inh_shfl_alig_avSamps_avDays_avMice, [], classAcc_allN_shfl_alig_avSamps_avDays_avMice, classAcc_exc_shfl_alig_avSamps_avDays_avMice, pei, dir0, fnam, cblab012[0], CA=1)



#%% Plot CA traces of a specific horizontal slice of the heatmap (decoder trained at time t, how does it perform when tested at all other times)

fr2an = nPreMin_final-1

plt.figure(figsize=(3*3,2))  

plt.subplot(131)
top = classAcc_allN_alig_avSamps_avDays_avMice[fr2an]
topsd = classAcc_allN_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='k', alpha=.5)
plt.plot(time_aligned_final, top, color='k')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel(cblab012[0])
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1, 1)


plt.subplot(132)
top = classAcc_inh_alig_avSamps_avDays_avMice[fr2an]
topsd = classAcc_inh_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='r', alpha=.5)
plt.plot(time_aligned_final, top, color='r')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel(cblab012[0])
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1)


plt.subplot(133)
top = classAcc_exc_alig_avSamps_avDays_avMice[fr2an]
topsd = classAcc_exc_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='b', alpha=.5)
plt.plot(time_aligned_final, top, color='b')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel(cblab012[0])
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1)

plt.subplots_adjust(wspace=.6)


################## save the figure ##################
if savefigs:
    if chAl:
        cha = 'chAl_'
    else:
        cha = 'stAl_'
        
    fnam = figns[0]+'_timeBin-1_AveMice_AveDays_inhExc'+labAll+'_' + '_'.join(mice) + '_' + nowStr
    d = os.path.join(svmdir+dir0) #,mousename)       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
        
    fign = os.path.join(d, suffn[0:5]+cha+fnam+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)

    
'''
# show individual mice
a = [classAcc_allN_alig_avSamps_avDays_allMice_al[im][fr2an] for im in range(len(mice))]
for im in range(len(mice)):
    top = a[im]
    plt.plot(time_aligned_final, top, color='k')
'''




#%%
############################ Same as above, now for fract max CA ############################

##%% Align day-averaged CA traces 

classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_allN_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_inh_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_exc_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
# shfl
#classAcc_fractChangeFromMax_allN_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_allN_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
#classAcc_fractChangeFromMax_inh_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_inh_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
#classAcc_fractChangeFromMax_exc_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 


#%% Average day-averaged traces across mice

classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_avMice = np.nanmean(classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_allMice_al, axis=0) # frs x frs
classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_avMice = np.nanmean(classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_allMice_al, axis=0) 
classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_avMice = np.nanmean(classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_allMice_al, axis=0)                    
# shfl
#classAcc_fractChangeFromMax_allN_shfl_alig_avSamps_avDays_avMice = np.mean(classAcc_fractChangeFromMax_allN_shfl_alig_avSamps_avDays_allMice_al, axis=0) # frs x frs
#classAcc_fractChangeFromMax_inh_shfl_alig_avSamps_avDays_avMice = np.mean(classAcc_fractChangeFromMax_inh_shfl_alig_avSamps_avDays_allMice_al, axis=0) 
#classAcc_fractChangeFromMax_exc_shfl_alig_avSamps_avDays_avMice = np.mean(classAcc_fractChangeFromMax_exc_shfl_alig_avSamps_avDays_allMice_al, axis=0)                    

# set se across mice
classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_sdMice = np.std(classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_allMice_al, axis=0) / np.sqrt(len(mice)) # frs x frs
classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_sdMice = np.std(classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_allMice_al, axis=0) / np.sqrt(len(mice))
classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_sdMice = np.std(classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_allMice_al, axis=0) / np.sqrt(len(mice))


#%% Plot heatmaps: mice-averaged of CAs (frs x frs) 

# ttest across mice (day-averaged traces)            
#_,pei = stats.ttest_ind(classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_allMice_al - classAcc_fractChangeFromMax_inh_shfl_alig_avSamps_avDays_allMice_al , classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_allMice_al - classAcc_fractChangeFromMax_exc_shfl_alig_avSamps_avDays_allMice_al , axis=0) 
_,pei = stats.ttest_ind(classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_allMice_al , classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_allMice_al, axis=0) 

fnam = figns[2]+'_AveMice_AveDays_inhExc'+labAll+'_' + '_'.join(mice) + '_' + nowStr   # (np.array(mice)[mice2an_all])

plotAngsAll(nPreMin_final, time_aligned_final, classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_avMice, [], classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_avMice, classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_avMice, 
            0, [], 0, 0, pei, dir0, fnam, cblab012[2], CA=1)



#%% #%% Plot CA traces of a specific horizontal slice of the heatmap (decoder trained at time t, how does it perform when tested at all other times) 
# Now for fract max: Plot CA traces of a specific horizontal slice of the heatmap (decoder trained at time t, how does it perform when tested at all other times)

fr2an = nPreMin_final-1

plt.figure(figsize=(3*3,2))  

plt.subplot(131)
top = classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_avMice[fr2an]
topsd = classAcc_fractChangeFromMax_allN_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='k', alpha=.5)
plt.plot(time_aligned_final, top, color='k')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel(cblab012[2])
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1, 1)


plt.subplot(132)
top = classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_avMice[fr2an]
topsd = classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='r', alpha=.5)
plt.plot(time_aligned_final, top, color='r')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel(cblab012[2])
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1)


plt.subplot(133)
top = classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_avMice[fr2an]
topsd = classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='b', alpha=.5)
plt.plot(time_aligned_final, top, color='b')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel(cblab012[2])
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1)

plt.subplots_adjust(wspace=.6)


################## save the figure ##################
if savefigs:
    if chAl:
        cha = 'chAl_'
    else:
        cha = 'stAl_'
        
    fnam = figns[2]+'_timeBin-1_AveMice_AveDays_inhExc'+labAll+'_' + '_'.join(mice) + '_' + nowStr
    d = os.path.join(svmdir+dir0) #,mousename)       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
        
    fign = os.path.join(d, suffn[0:5]+cha+fnam+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)

    
'''
# show individual mice
a = [classAcc_allN_alig_avSamps_avDays_allMice_al[im][fr2an] for im in range(len(mice))]
for im in range(len(mice)):
    top = a[im]
    plt.plot(time_aligned_final, top, color='k')
'''





#%%
############################ Same as above, now for stability [whether for each day CA(tr,ts) is within 1se of CA(ts,ts) across samps] ############################

##%% Align day-averaged traces 

sameCA_allN_samps_aveDays_allMice_al = alTrace_frfr(sameCA_allN_samps_aveDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
sameCA_inh_samps_aveDays_allMice_al = alTrace_frfr(sameCA_inh_samps_aveDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
sameCA_exc_samps_aveDays_allMice_al = alTrace_frfr(sameCA_exc_samps_aveDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
# shfl
#classAcc_fractChangeFromMax_allN_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_allN_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
#classAcc_fractChangeFromMax_inh_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_inh_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 
#classAcc_fractChangeFromMax_exc_shfl_alig_avSamps_avDays_allMice_al = alTrace_frfr(classAcc_fractChangeFromMax_exc_shfl_allDays_alig_avSamps_avDays_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final) # mice x frs x frs 


#%% Average day-averaged traces across mice

sameCA_allN_samps_alig_avSamps_avDays_avMice = np.nanmean(sameCA_allN_samps_aveDays_allMice_al, axis=0) # frs x frs
sameCA_inh_samps_alig_avSamps_avDays_avMice = np.nanmean(sameCA_inh_samps_aveDays_allMice_al, axis=0) 
sameCA_exc_samps_alig_avSamps_avDays_avMice = np.nanmean(sameCA_exc_samps_aveDays_allMice_al, axis=0)                    

# set se across mice
sameCA_allN_samps_alig_avSamps_avDays_sdMice = np.std(sameCA_allN_samps_aveDays_allMice_al, axis=0) / np.sqrt(len(mice)) # frs x frs
sameCA_inh_samps_alig_avSamps_avDays_sdMice = np.std(sameCA_inh_samps_aveDays_allMice_al, axis=0) / np.sqrt(len(mice))
sameCA_exc_samps_alig_avSamps_avDays_sdMice = np.std(sameCA_exc_samps_aveDays_allMice_al, axis=0) / np.sqrt(len(mice))


#%% Plot heatmaps: mice-averaged of stabilities (frs x frs) 

### First define function to plot heatmaps
def plotStabAll(nPreMin, time_aligned, angInh_av, angAllExc_av, angExc_excsh_av, pei, dnow, fnam, cblab):
       
    totLen = len(time_aligned) 

    if chAl:
        cha = 'chAl_'
    else:
        cha = 'stAl_'
    
    cmap = 'jet'       
    xl = 'Testing t (ms)'
    yl = 'Decoder training t (ms)'
    cmax = np.ceil(np.max([np.nanmax(angInh_av), np.nanmax(angExc_excsh_av), np.nanmax(angAllExc_av)]))

        
    plt.figure(figsize=(8,8))  
    rows = 2; cols = 2; 
    sp1 = 1; sp3 = 3; sp2 = 2; sp4 = 4        # for data        
    
    ex = angExc_excsh_av    
    cmin = np.floor(np.min([np.nanmin(angInh_av), np.nanmin(ex), np.nanmin(angAllExc_av)]))
    
    plt.subplot(rows,cols,sp1); lab = 'inh (data)'; 
    top = angInh_av; 
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    plt.xlabel(xl); plt.ylabel(yl)
    
    plt.subplot(rows,cols,sp3); lab = 'exc (data)'; 
    top = ex; 
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    plt.xlabel(xl); plt.ylabel(yl)
    
    plt.subplot(rows,cols,sp2); lab = labAll+' (data)'; 
    top = angAllExc_av; 
    plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
    plt.xlabel(xl); plt.ylabel(yl)
    
    plt.subplots_adjust(hspace=.2, wspace=.5)
    
    #### save the figure ####
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
            
        fign = os.path.join(d, suffn[0:5]+cha+fnam+'.'+fmt[0])    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        

###########################################################################################
# ttest across mice (day-averaged traces)            
#_,pei = stats.ttest_ind(classAcc_fractChangeFromMax_inh_alig_avSamps_avDays_allMice_al - classAcc_fractChangeFromMax_inh_shfl_alig_avSamps_avDays_allMice_al , classAcc_fractChangeFromMax_exc_alig_avSamps_avDays_allMice_al - classAcc_fractChangeFromMax_exc_shfl_alig_avSamps_avDays_allMice_al , axis=0) 
_,pei = stats.ttest_ind(sameCA_inh_samps_aveDays_allMice_al , sameCA_exc_samps_aveDays_allMice_al, axis=0) 
#plt.imshow(pei<alph); plt.colorbar()

fnam = 'classAccurWin'+str(fractStd)+'sdMax_samps'+'_AveMice_AveDays_inhExc'+labAll+'_' + '_'.join(mice) + '_' + nowStr   # (np.array(mice)[mice2an_all])

plotStabAll(nPreMin_final, time_aligned_final, sameCA_inh_samps_alig_avSamps_avDays_avMice, sameCA_allN_samps_alig_avSamps_avDays_avMice, sameCA_exc_samps_alig_avSamps_avDays_avMice, 
            pei, dir0, fnam, 'CA within 2SD of CA(ts,ts)')



#%% #%% Plot CA traces of a specific horizontal slice of the heatmap (decoder trained at time t, how does it perform when tested at all other times) 
# Now for fract max: Plot CA traces of a specific horizontal slice of the heatmap (decoder trained at time t, how does it perform when tested at all other times)

fr2an = nPreMin_final-1

plt.figure(figsize=(3*3,2))  

plt.subplot(131)
top = sameCA_allN_samps_alig_avSamps_avDays_avMice[fr2an]
topsd = sameCA_allN_samps_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='k', alpha=.5)
plt.plot(time_aligned_final, top, color='k')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel('CA within 2SD of CA(ts,ts)')
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1, 1)


plt.subplot(132)
top = sameCA_inh_samps_alig_avSamps_avDays_avMice[fr2an]
topsd = sameCA_inh_samps_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='r', alpha=.5)
plt.plot(time_aligned_final, top, color='r')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel('CA within 2SD of CA(ts,ts)')
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1)


plt.subplot(133)
top = sameCA_exc_samps_alig_avSamps_avDays_avMice[fr2an]
topsd = sameCA_exc_samps_alig_avSamps_avDays_sdMice[fr2an]
plt.fill_between(time_aligned_final, top - topsd, top + topsd, color='b', alpha=.5)
plt.plot(time_aligned_final, top, color='b')
plt.xlabel('Decoder testing t (ms)') 
plt.ylabel('CA within 2SD of CA(ts,ts)')
plt.vlines(time_aligned_final[fr2an],np.min(top),np.max(top), color='k', linestyle=':') # mark the training time point
makeNicePlots(plt.gca(), 1)

plt.subplots_adjust(wspace=.6)


################## save the figure ##################
if savefigs:
    if chAl:
        cha = 'chAl_'
    else:
        cha = 'stAl_'
        
    fnam = 'classAccurWin'+str(fractStd)+'sdMax_samps'+'_timeBin-1_AveMice_AveDays_inhExc'+labAll+'_' + '_'.join(mice) + '_' + nowStr
    d = os.path.join(svmdir+dir0) #,mousename)       
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)            
        
    fign = os.path.join(d, suffn[0:5]+cha+fnam+'.'+fmt[0])    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)

    
'''
# show individual mice
a = [classAcc_allN_alig_avSamps_avDays_allMice_al[im][fr2an] for im in range(len(mice))]
for im in range(len(mice)):
    top = a[im]
    plt.plot(time_aligned_final, top, color='k')
'''






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
##%%
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

        