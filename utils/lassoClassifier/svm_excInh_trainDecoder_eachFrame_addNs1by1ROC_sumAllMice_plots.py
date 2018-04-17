#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:56:47 2018

@author: farznaj
"""

#%%

mice = 'fni16', 'fni17', 'fni18', 'fni19'

perc_thb = [20,80] # perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.
savefigs = 0


thTrained = 10
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
   
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

execfile("defFuns.py")

dnow = '/excInh_trainDecoder_eachFrame_addNs1by1ROC/'


#%%

dav_allExc_allM = []
dav_inh_allM = []
dav_exc_av_allM = []
days2an_heatmap_allM = []
behCorr_all_allM = []
av_test_data_allExc_allM = []
av_test_data_inh_allM = []
av_test_data_exc_allM = []
av_test_data_exc_excSamp_allM = []
mn_corr_allM = []
av_test_shfl_allExc_allM = []
av_test_shfl_inh_allM = []
av_test_shfl_exc_allM = []
av_test_shfl_exc_excSamp_allM = []

for im in range(len(mice)):
    mousename = mice[im]
    
    imagingDir = setImagingAnalysisNamesP(mousename)
    fname = os.path.join(imagingDir, 'analysis')            
    finame = os.path.join(fname, 'svm_classErr_shflTrsEachN_*.mat')    
    data = scio.loadmat(glob.glob(finame)[0])
    
    eventI_ds_allDays = np.array(data.pop('eventI_ds_allDays')).flatten().astype('float')
    mn_corr = np.array(data.pop('mn_corr')).flatten().astype('float')
    perClassErrorTest_data_allExc_alln = np.array(data.pop('perClassErrorTest_data_allExc_alln')).flatten()
    perClassErrorTest_data_inh_alln = np.array(data.pop('perClassErrorTest_data_inh_alln')).flatten()
    perClassErrorTest_data_exc_alln = np.array(data.pop('perClassErrorTest_data_exc_alln')).flatten()
    av_test_data_allExc = np.array(data.pop('av_test_data_allExc')).flatten()
    av_test_data_inh = np.array(data.pop('av_test_data_inh')).flatten()
    av_test_data_exc = np.array(data.pop('av_test_data_exc')).flatten()
    perClassErrorTest_data_allExc_all_shflTrsEachNn = np.array(data.pop('perClassErrorTest_data_allExc_all_shflTrsEachNn')).flatten()
    perClassErrorTest_data_inh_all_shflTrsEachNn = np.array(data.pop('perClassErrorTest_data_inh_all_shflTrsEachNn')).flatten()
    perClassErrorTest_data_exc_all_shflTrsEachNn = np.array(data.pop('perClassErrorTest_data_exc_all_shflTrsEachNn')).flatten()
    av_test_data_allExc_shflTrsEachN = np.array(data.pop('av_test_data_allExc_shflTrsEachN')).flatten()
    av_test_data_inh_shflTrsEachN = np.array(data.pop('av_test_data_inh_shflTrsEachN')).flatten()
    av_test_data_exc_shflTrsEachN = np.array(data.pop('av_test_data_exc_shflTrsEachN')).flatten()
    
    finame = os.path.join(fname, 'svm_shflClassErr_*.mat')    
    finame = glob.glob(finame)
    finame = sorted(finame, key=os.path.getmtime)
    finame = finame[::-1][0] # so the latest file is the 1st one.    
    data = scio.loadmat(finame)
    av_test_shfl_allExc = np.array(data.pop('av_test_shfl_allExc')).flatten()
    av_test_shfl_inh = np.array(data.pop('av_test_shfl_inh')).flatten()
    av_test_shfl_exc = np.array(data.pop('av_test_shfl_exc')).flatten()
    perClassErrorTest_shfl_exc_alln = np.array(data.pop('perClassErrorTest_shfl_exc_alln')).flatten()
    
    numD = len(eventI_ds_allDays)
    
    ###%% Average CAs across cv samps, do this separately for each exc samp
    av_test_data_exc_shflTrsEachN_excSamp = []
    av_test_data_exc_excSamp = []
    av_test_shfl_exc_excSamp = []
    for iday in range(numD):
        # numShufflesExc x number of neurons in the decoder x nFrs 
        av_test_data_exc_shflTrsEachN_excSamp.append(100 - np.mean(perClassErrorTest_data_exc_all_shflTrsEachNn[iday], axis=2).T) # numShufflesExc x number of neurons in the decoder
        av_test_data_exc_excSamp.append(100 - np.mean(perClassErrorTest_data_exc_alln[iday], axis=2).T) # numShufflesExc x number of neurons in the decoder
        av_test_shfl_exc_excSamp.append(100 - np.mean(perClassErrorTest_shfl_exc_alln[iday], axis=2).T) # numShufflesExc x number of neurons in the decoder
    
    numExcSamples = perClassErrorTest_data_exc_all_shflTrsEachNn[0].shape[1]
    #numDaysGood = sum(mn_corr>=thTrained)
    mn_corr0 = mn_corr + 0 # will be used for the heatmaps of CA for all days; We need to exclude 151023 from fni16, this day had issues! ...  in the mat file of stabTestTrainTimes, its class accur is very high ... and in the mat file of excInh_trainDecoder_eachFrame it is very low ...ANYWAY I ended up removing it from the heatmap of CA for all days!!! but it is included in other analyses!!
#    if mousename=='fni16':
#        mn_corr0[days.index('151023_1')] = thTrained - 1 # see it will be excluded from analysis!    
    days2an_heatmap = mn_corr0 >= thTrained 
       
    # load behavioral performance
    imagingDir = setImagingAnalysisNamesP(mousename)
    fname = os.path.join(imagingDir, 'analysis')    
    # set svmStab mat file name that contains behavioral and class accuracy vars
    finame = os.path.join(fname, 'svm_stabilityBehCA_*.mat')
    stabBehName = glob.glob(finame)
    stabBehName = sorted(stabBehName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    stabBehName = stabBehName[0]
    
    # load beh vars    
    Data = scio.loadmat(stabBehName)
    behCorr_all = Data.pop('behCorr_all').flatten() # the following comment is for the above mat file: I didnt save _all vars... so what is saved is only for one day! ... have to reset these 3 vars again here!
    #    behCorrHR_all = Data.pop('behCorrHR_all').flatten()
    #    behCorrLR_all = Data.pop('behCorrLR_all').flatten()
            
    
    ##%%
    av_test_data_allExc_allM.append(av_test_data_allExc)
    av_test_data_inh_allM.append(av_test_data_inh)
    av_test_data_exc_allM.append(av_test_data_exc)
    av_test_data_exc_excSamp_allM.append(av_test_data_exc_excSamp)
    mn_corr_allM.append(mn_corr)
    av_test_shfl_allExc_allM.append(av_test_shfl_allExc)
    av_test_shfl_inh_allM.append(av_test_shfl_inh)
    av_test_shfl_exc_allM.append(av_test_shfl_exc)
    av_test_shfl_exc_excSamp_allM.append(av_test_shfl_exc_excSamp)
     
       
    ############################## Compute the change in CA from the actual case to shflTrsEachNeuron case ... aveaged across those population sizes that are significantly different between actual and shflTrsEachN
    # for each day do ttest across samples for each of the population sizes to see if shflTrsEachN is differnt fromt eh eactual case (ie neuron numbers in the decoder)
    dav_allExc, dav_inh, dav_exc_av, dav_exc = changeCA_shflTrsEachN()
    
    
    dav_allExc_allM.append(dav_allExc)
    dav_inh_allM.append(dav_inh)
    dav_exc_av_allM.append(dav_exc_av)
    days2an_heatmap_allM.append(days2an_heatmap)
    behCorr_all_allM.append(behCorr_all)

no




#%%
#################################################################################################################################
#################################### Plots of change in CA after breaking noise correaltions ####################################
#################################################################################################################################

#%%

############ Plot change in CA vs training day ############

plt.figure(figsize=(3, 8))

for im in range(len(mice)):
    
    dav_allExc = dav_allExc_allM[im]
    dav_inh = dav_inh_allM[im]
    dav_exc_av = dav_exc_av_allM[im]
#    days2an_heatmap = days2an_heatmap_allM[im]
#    behCorr_all = behCorr_all_allM[im]
    
    plt.subplot(len(mice), 1, im+1)
    
    plt.plot(dav_allExc, 'k.-', label='allExc')
    plt.plot(dav_inh, 'r.-', label='inh')
    plt.plot(dav_exc_av, 'b.-', label='exc')
    makeNicePlots(plt.gca())
    plt.ylabel('Change in CA', fontsize=11)        
    plt.xlabel('Training day', fontsize=11)        
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1, frameon=False)         
    #    plt.hist(dav_exc[~np.isnan(dav_exc)])
    #    plt.hist(dav_inh[~np.isnan(dav_inh)])
    
plt.subplots_adjust(hspace=.6, wspace=.5)


if savefigs:
    if chAl==1:
        dd = 'chAl_VSshflTrsEachN_CAchange_vsDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_VSshflTrsEachN_CAchange_vsDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



############%% Plot ave and se of change in CA across days... compare exc, inh, allExc ############   
     
plt.figure(figsize=(4, 2))

for im in range(len(mice)):
    
    dav_allExc = dav_allExc_allM[im]
    dav_inh = dav_inh_allM[im]
    dav_exc_av = dav_exc_av_allM[im]
#    days2an_heatmap = days2an_heatmap_allM[im]
#    behCorr_all = behCorr_all_allM[im]

    aa = np.nanmean(dav_allExc)
    ai = np.nanmean(dav_inh)
    ae = np.nanmean(dav_exc_av)
    # se across days
    sa = np.nanstd(dav_allExc) / np.sqrt(sum(~np.isnan(dav_allExc)))
    si = np.nanstd(dav_inh) / np.sqrt(sum(~np.isnan(dav_inh)))
    se = np.nanstd(dav_exc_av) / np.sqrt(sum(~np.isnan(dav_exc_av)))       

    
    plt.subplot(1, len(mice), im+1)
    
    plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k', markeredgecolor='k')
    plt.errorbar(1, ai, si, fmt='o', label='inh', color='r', markeredgecolor='r')
    plt.errorbar(2, ae, se, fmt='o', label='exc', color='b', markeredgecolor='b')        
    if im==len(mice)-1:
        plt.legend(loc=0, frameon=False, numpoints=1, bbox_to_anchor=(1, .73))    
    plt.xticks(range(3), ['allExc','inh','exc'], rotation=70)    
    plt.xlim([-.5, 2+.5])            
#    ax = plt.gca();    yl = ax.get_ylim();     plt.ylim([yl[0]-2, yl[1]])
    plt.ylim([-1.7, 11])
#    if im>0:
#        plt.gca().spines['left'].set_color('white')
#        plt.gca().yaxis.set_visible(False)
    if im==0:       
        plt.ylabel('Change in CA', fontsize=11)        
    makeNicePlots(plt.gca())

plt.subplots_adjust(hspace=.5, wspace=.5)        
        

if savefigs:
    if chAl==1:
        dd = 'chAl_VSshflTrsEachN_CAchange_avSeDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_VSshflTrsEachN_CAchange_avSeDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
    
################## Behavior: compare change in CA (after removing noise corr) for days with low vs high behavioral performance ##################

plotAll = 1 # if 0, plot only allExc; otherwise plot exc and inh too
    
if plotAll:
    labs = 'allExc', 'exc', 'inh'    
    plt.figure(figsize=(4, 2))
else:
    labs = 'allExc', 
    plt.figure(figsize=(3, 2))


for im in range(len(mice)):
    
    dav_allExc = dav_allExc_allM[im]
    dav_inh = dav_inh_allM[im]
    dav_exc_av = dav_exc_av_allM[im]
    days2an_heatmap = days2an_heatmap_allM[im]
    behCorr_all = behCorr_all_allM[im]

    
    a = behCorr_all[days2an_heatmap]    
    thb = np.percentile(a, perc_thb)     # thb = np.percentile(a, [10,90])
    
    loBehCorrDays = (a <= thb[0])    
    hiBehCorrDays = (a >= thb[1])
    print sum(loBehCorrDays), sum(hiBehCorrDays), ': num days with low and high beh performance'
    
    aa = dav_allExc[days2an_heatmap]
    bb = dav_inh[days2an_heatmap]
    cc = dav_exc_av[days2an_heatmap]
    a = aa[loBehCorrDays]; ah = aa[hiBehCorrDays]
    b = bb[loBehCorrDays]; bh = bb[hiBehCorrDays]
    c = cc[loBehCorrDays]; ch = cc[hiBehCorrDays]    
    
    ahs = np.nanstd(ah)/ np.sqrt(sum(~np.isnan(ah)))        
    chs = np.nanstd(ch)/ np.sqrt(sum(~np.isnan(ch)))
    bhs = np.nanstd(bh)/ np.sqrt(sum(~np.isnan(bh)))
    as0 = np.nanstd(a) / np.sqrt(sum(~np.isnan(a)))
    cs0 = np.nanstd(c) / np.sqrt(sum(~np.isnan(c)))
    bs0 = np.nanstd(b) / np.sqrt(sum(~np.isnan(b)))
    
    
    plt.subplot(1, len(mice), im+1)
    
    x = range(len(labs))    
    plt.errorbar(x[0], np.nanmean(ah), ahs, marker='o', color='k', fmt='.', markeredgecolor='k', label='high beh')
    plt.errorbar(x[0], np.nanmean(a), as0, marker='o', color='gray', fmt='.', markeredgecolor='gray', label='low beh')        
    if plotAll:
        plt.errorbar(x[1], np.nanmean(ch), chs, marker='o', color='b', fmt='.', markeredgecolor='b')
        plt.errorbar(x[1], np.nanmean(c), cs0, marker='o', color='lightblue', fmt='.', markeredgecolor='lightblue')       
        plt.errorbar(x[2], np.nanmean(bh), bhs, marker='o',color='r', fmt='.', markeredgecolor='r')        
        plt.errorbar(x[2], np.nanmean(b), bs0, marker='o',color='lightsalmon', fmt='.', markeredgecolor='lightsalmon')    
    plt.xlim([x[0]-.5, x[-1]+.5]) # plt.xlim([-1,3])    
    plt.xticks(x, [mice[im]], rotation=70) # 'vertical'    
    if im==len(mice)-1:
        plt.legend(loc=0, frameon=False, numpoints=1, bbox_to_anchor=(.7, .73))    
    if im>0:
        plt.gca().spines['left'].set_color('white')
        plt.gca().yaxis.set_visible(False)
    else:
        plt.ylabel('Change in CA (%)')            
    if plotAll:
        plt.ylim([-2, 13])
    else:
        plt.ylim([4, 13])        
#    yl = plt.gca().get_ylim();     r = np.diff(yl);     plt.ylim([yl[0], yl[1]+r/10.])       
    makeNicePlots(plt.gca(),0,1)
    
plt.subplots_adjust(hspace=.5, wspace=.5)


if savefigs:
    sn = '_'.join(labs) + '_'
    
    if chAl==1:
        dd = 'chAl_VSshflTrsEachN_CAchange_avSeDaysBeh_' + sn + 'addNsROC_' + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_VSshflTrsEachN_CAchange_avSeDaysBeh_addNsROCBeh_' + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
        



#%%
####################################################################################
####################################################################################
####################################################################################
    
#%% Compute number of neurons to reach plateau for each day

nNsContHiCA = 1 #3 # if there is a gap of >=3 continuous n Ns with high CA (CA > thp percentile of CA across all Ns), call the first number of N as the point of plateuou.
alph = .05 # only set plateau if CA is sig different from chance
#thpa = 80 #10 # CAs above thp percentile are considered high CAs (where plataue happens)
#thpi = 80 #50
#thpe = 80 #80

platN_allExc_allM = []
platN_inh_allM = []
platN_exc_allM = []
platN_exc_excSamp_allM = []

for im  in range(len(mice)):
    
    av_test_data_allExc = av_test_data_allExc_allM[im]
    av_test_data_inh = av_test_data_inh_allM[im]
    av_test_data_exc = av_test_data_exc_allM[im]
    av_test_data_exc_excSamp = av_test_data_exc_excSamp_allM[im]
    av_test_shfl_allExc = av_test_shfl_allExc_allM[im]
    av_test_shfl_inh = av_test_shfl_inh_allM[im]
    av_test_shfl_exc = av_test_shfl_exc_allM[im]
    av_test_shfl_exc_excSamp = av_test_shfl_exc_excSamp_allM[im]
    numD = len(av_test_data_allExc)
    mn_corr = mn_corr_allM[im]
    
   
    platN_allExc, platN_inh, platN_exc, platN_exc_excSamp = numNeurPlateau()
    
    platN_allExc_allM.append(platN_allExc)
    platN_inh_allM.append(platN_inh)
    platN_exc_allM.append(platN_exc)
    platN_exc_excSamp_allM.append(platN_exc_excSamp)
    
    

#%%
#################################################################################################################################
#################################### Plots of number of neurons to reach plateau ####################################
#################################################################################################################################
    
#%%

mnp = 0
mxp = 85
mn_allExc = np.min([np.nanpercentile(platN_allExc_allM[im], [mnp,mxp]) for im in range(len(mice))])
mx_allExc = np.max([np.nanpercentile(platN_allExc_allM[im], [mnp,mxp]) for im in range(len(mice))])
mn_inh = np.min([np.nanpercentile(platN_inh_allM[im], [mnp,mxp]) for im in range(len(mice))])
mx_inh = np.max([np.nanpercentile(platN_inh_allM[im], [mnp,mxp]) for im in range(len(mice))])
mn_exc = np.min([np.nanpercentile(platN_exc_allM[im], [mnp,mxp]) for im in range(len(mice))])
mx_exc = np.max([np.nanpercentile(platN_exc_allM[im], [mnp,mxp]) for im in range(len(mice))])


###################### Plot number of neurons to plateau vs. training day ####################
f1 = plt.figure(1,figsize=(3,8))    
for im  in range(len(mice)):
    platN_allExc = platN_allExc_allM[im]
    platN_inh = platN_inh_allM[im]
    platN_exc = platN_exc_allM[im]
    
    plt.figure(1)
    
    plt.subplot(len(mice),1,im+1)
    plt.plot(platN_allExc, 'k.-', label='allExc')    
    plt.plot(platN_inh, 'r.-', label='inh')
    plt.plot(platN_exc, 'b.-', label='exc')
    makeNicePlots(plt.gca())
    plt.ylabel('# neurons to plateau', fontsize=11)        
    plt.xlabel('Training day', fontsize=11)        
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1, frameon=False)  
    
plt.subplots_adjust(wspace=.5, hspace=.5)
    
    
if savefigs:
    if chAl==1:
        dd = 'chAl_numNeurPlateau_vsDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_numNeurPlateau_vsDays__addNsROC_' + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)       



##################### Box plots #####################
f2 = plt.figure(2,figsize=(3,6))
for im  in range(len(mice)):
    platN_allExc = platN_allExc_allM[im]
    platN_inh = platN_inh_allM[im]
    platN_exc = platN_exc_allM[im]
        
    plt.figure(2)
    
    ###### allExc ######
    plt.subplot(3, len(mice), im+1)    
    bp = plt.boxplot(platN_allExc[~np.isnan(platN_allExc)], 0, showfliers=False, labels=['allExc'])#, widths=.05)#, positions=[0])
    bp['boxes'][0].set(color='k')
    bp['medians'][0].set(color='k')
    bp['whiskers'][0].set(linewidth=0); bp['whiskers'][1].set(linewidth=0)
    bp['caps'][0].set(linewidth=0); bp['caps'][1].set(linewidth=0)
    makeNicePlots(plt.gca())    
#    plt.ylim([mn_allExc, mx_allExc])
    plt.ylim([-2, mx_allExc])
    if im>0:
        plt.gca().spines['left'].set_color('white')
        plt.gca().yaxis.set_visible(False)
    else:
        plt.ylabel('# neurons to plateau', fontsize=11)
    
    
    ###### inh and exc ######
    plt.subplot(3, len(mice), len(mice)+im+1)
    # inh
    bp = plt.boxplot(platN_inh[~np.isnan(platN_inh)], 0, showfliers=False)#, positions=[1])
    bp['boxes'][0].set(color='r')
    bp['medians'][0].set(color='r')
    bp['whiskers'][0].set(linewidth=0); bp['whiskers'][1].set(linewidth=0)
    bp['caps'][0].set(linewidth=0); bp['caps'][1].set(linewidth=0)
    # exc
    bp = plt.boxplot(platN_exc[~np.isnan(platN_exc)], 0, showfliers=False, labels=[mice[im]])#, labels=['inh'])#, positions=[2])
    bp['boxes'][0].set(color='b')
    bp['medians'][0].set(color='b')
    bp['whiskers'][0].set(linewidth=0); bp['whiskers'][1].set(linewidth=0)
    bp['caps'][0].set(linewidth=0); bp['caps'][1].set(linewidth=0)
    makeNicePlots(plt.gca(), 0,1)    
#    plt.ylim([min(mn_inh, mn_exc), max(mx_inh, mx_exc)])
    plt.ylim([-2, max(mx_inh, mx_exc)])
    if im>0:
        plt.gca().spines['left'].set_color('white')
        plt.gca().yaxis.set_visible(False)
    else:
        plt.ylabel('# neurons to plateau', fontsize=11)


    ###### exc - inh ######
    plt.subplot(3, len(mice), 2*len(mice)+im+1)    
    a = platN_exc - platN_inh
    bp = plt.boxplot(a[~np.isnan(a)], 0, showfliers=False, labels=['exc-inh'])
    bp['boxes'][0].set(color='m')
    bp['medians'][0].set(color='m')
    bp['whiskers'][0].set(linewidth=0); bp['whiskers'][1].set(linewidth=0)
    bp['caps'][0].set(linewidth=0); bp['caps'][1].set(linewidth=0)    
    makeNicePlots(plt.gca())  

plt.subplots_adjust(wspace=.5, hspace=.5)
    
    
if savefigs:
    if chAl==1:
        dd = 'chAl_numNeurPlateau_boxPlot_addNsROC_' + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_numNeurPlateau_boxPlot__addNsROC_' + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)   

    
    
##################### Error bars : ave and se across days ##################### 
f3 = plt.figure(3,figsize=(3,6))
for im  in range(len(mice)):
    platN_allExc = platN_allExc_allM[im]
    platN_inh = platN_inh_allM[im]
    platN_exc = platN_exc_allM[im]

    aa = np.nanmean(platN_allExc)
    ai = np.nanmean(platN_inh)
    ae = np.nanmean(platN_exc)
    sa = np.nanstd(platN_allExc) / np.sqrt(sum(~np.isnan(platN_allExc)))
    si = np.nanstd(platN_inh) / np.sqrt(sum(~np.isnan(platN_inh)))
    se = np.nanstd(platN_exc) / np.sqrt(sum(~np.isnan(platN_exc)))
    aeid = np.nanmean(platN_exc - platN_inh)
    seid = np.nanstd(platN_exc - platN_inh) / np.sqrt(sum(~np.isnan(platN_exc - platN_inh)))
    
    
    plt.figure(3)
    labs = 'allExc', #'allExc', 'exc', 'inh'
    x = range(len(labs))

    ###### allExc ######
    plt.subplot(3, len(mice), im+1)    
    plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k', markeredgecolor='k')
    if im==0: #len(mice)-1:
        plt.ylabel('# neurons to plateau', fontsize=11)
    plt.xlim([x[0]-.5, x[-1]+.5]) # plt.xlim([-1,3])    
    plt.xticks(x, (labs), rotation=70) # 'vertical'
    ax = plt.gca()
    makeNicePlots(ax)
    yl = ax.get_ylim()
    plt.ylim([yl[0]-2, yl[1]])

    
    ###### inh and exc ######
    plt.subplot(3, len(mice), len(mice)+im+1)    
    plt.errorbar(0, ai, si, fmt='o', label='inh', color='r', markeredgecolor='r')
    plt.errorbar(0, ae, se, fmt='o', label='exc', color='b', markeredgecolor='b')    
    if im==0: #len(mice)-1:
#        plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
        plt.ylabel('# neurons to plateau', fontsize=11)
    plt.xlim([x[0]-.5, x[-1]+.5]) # plt.xlim([-1,3])    
    plt.xticks(x, '', rotation=70) # 'vertical'
    ax = plt.gca()
    makeNicePlots(ax)
    yl = ax.get_ylim()
    plt.ylim([yl[0]-2, yl[1]])

    
    ###### exc - inh ######
    plt.subplot(3, len(mice), 2*len(mice)+im+1)    
    plt.errorbar(0, aeid, seid, fmt='o', label='exc-inh', color='m', markeredgecolor='m')
    if im==0: #len(mice)-1:
        plt.ylabel('# neurons to plateau', fontsize=11)
    plt.xlim([x[0]-.5, x[-1]+.5]) # plt.xlim([-1,3])    
    plt.xticks(x, '', rotation=70) # 'vertical'
    ax = plt.gca()
    makeNicePlots(ax)
    yl = ax.get_ylim()
    plt.ylim([yl[0]-2, yl[1]])

plt.subplots_adjust(wspace=.5, hspace=.5)


if savefigs:
    if chAl==1:
        dd = 'chAl_numNeurPlateau_avSeDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'stAl_numNeurPlateau_avSeDays_addNsROC_' + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)   
    
    
    
 
    
        