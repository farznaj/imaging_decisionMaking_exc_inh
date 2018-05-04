#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Use vars of class accur to plot the following for all mice (summary quantifications) for low/high behavioral performance
    1) Classifier accuracy
    2) choice signal onset
    

Created on Sun Apr 15 23:56:47 2018
@author: farznaj
"""

#%%

mice = 'fni16', 'fni17', 'fni19' #'fni16', 'fni17', 'fni18', 'fni19'

corrTrained = 0
perc_thb = [20,80] # [10,90] #perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.
savefigs = 1


thTrained = 10
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
   
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

execfile("defFuns.py")

dnow = '/excInh_trainDecoder_eachFrame/'

if corrTrained==0:
    corrn0 = 'allOutcome_'
else:
    corrn0 = 'corr_'
    
    
#%%
    
eventI_ds_allDays_allM = []
mn_corr_allM = []
behCorr_all_allM = []
days2an_heatmap_allM = []
av_test_data_allExc_aligned_allM = []
av_test_data_exc_aligned_allM = []
av_test_data_inh_aligned_allM = []
firstFrame_allN_allM = []
firstFrame_inh_allM = []
firstFrame_exc_randSamp_allM = []
firstFrame_exc_avSamps_allM = []
choicePref_exc_allM = []
choicePref_inh_allM = []

for im in range(len(mice)):
    mousename = mice[im]
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl)
    
    dataPath = setImagingAnalysisNamesP(mousename, [], [])    
    dirn = os.path.join(dataPath,'analysis') #    dirn = os.path.join(dataPath+mousename,'imaging','analysis')


    ############ Load ROC vars ############    
    # outcome analyzed for the ROC:
    al = 'chAl'
    o2aROC = '_allOutcome'; #'_corr'; # '_incorr';
    thStimSt = 0; namz = '';
    namv = 'ROC_curr_%s%s_stimstr%d%s_%s_*.mat' %(al, o2aROC, thStimSt, namz, mousename)
    a = glob.glob(os.path.join(dirn,namv)) #[0] 
    dirROC = sorted(a, key=os.path.getmtime)[::-1][0] # sort so the latest file is the 1st one. # use the latest saved file
    namtf = os.path.basename(dirROC)
    print namtf
    
   
    Data = scio.loadmat(dirROC, variable_names=['choicePref_all_alld_exc', 'choicePref_all_alld_inh'])  
    choicePref_all_alld_exc = Data.pop('choicePref_all_alld_exc').flatten() # nDays; each day: frames x neurons
    choicePref_all_alld_inh = Data.pop('choicePref_all_alld_inh').flatten()
    
    # Compute abs deviation of AUC from chance (.5); we want to compute |AUC-.5|, since choice pref = 2*(auc-.5), therefore |auc-.5| = 1/2 * |choice pref|
    # % now ipsi is positive bc we do minus, in the original vars (load above) contra was positive
    choicePref_exc = .5*abs(-choicePref_all_alld_exc)
    choicePref_inh = .5*abs(-choicePref_all_alld_inh)
    
    ############## keep vars of all mice
    choicePref_exc_allM.append(choicePref_exc)
    choicePref_inh_allM.append(choicePref_inh)
    
    
    #%% Load class accuracy vars and onset of choice signal vars
    
    imagingDir = setImagingAnalysisNamesP(mousename)
    fname = os.path.join(imagingDir, 'analysis')            
    finame = os.path.join(fname,  'svm_' + corrn0 + 'classAcc_nNsPlateau_*.mat')    
    finame = glob.glob(finame)[0]
    print finame
    data = scio.loadmat(finame)
    
    eventI_ds_allDays = np.array(data.pop('eventI_ds_allDays')).flatten().astype('float')
    mn_corr = np.array(data.pop('mn_corr')).flatten().astype('float')
    
    av_test_data_allExc_aligned = np.array(data.pop('av_test_data_allExc_aligned'))
    av_test_data_exc_aligned = np.array(data.pop('av_test_data_exc_aligned'))
    av_test_data_inh_aligned = np.array(data.pop('av_test_data_inh_aligned'))
    firstFrame_allN = np.array(data.pop('firstFrame_allN')).flatten()
    firstFrame_inh = np.array(data.pop('firstFrame_inh')).flatten()
    firstFrame_exc_randSamp = np.array(data.pop('firstFrame_exc_randSamp')).flatten()
    firstFrame_exc_avSamps = np.array(data.pop('firstFrame_exc_avSamps')).flatten()
    
    
    mn_corr0 = mn_corr + 0 # will be used for the heatmaps of CA for all days; We need to exclude 151023 from fni16, this day had issues! ...  in the mat file of stabTestTrainTimes, its class accur is very high ... and in the mat file of excInh_trainDecoder_eachFrame it is very low ...ANYWAY I ended up removing it from the heatmap of CA for all days!!! but it is included in other analyses!!
    if mousename=='fni16': #np.logical_and(mousename=='fni16', corrTrained==1):
        mn_corr0[days.index('151023_1')] = thTrained - 1 # see it will be excluded from analysis!    
    days2an_heatmap = mn_corr0 >= thTrained        
    
    
    ##%% Only get good days    
    firstFrame_allN = firstFrame_allN[days2an_heatmap] # ('150903_1') ... it is one of the early days and it is significant all over the trial!!! (rerun svm for this day!)        
    firstFrame_inh = firstFrame_inh[days2an_heatmap]
    firstFrame_exc_randSamp = firstFrame_exc_randSamp[days2an_heatmap]
    firstFrame_exc_avSamps = firstFrame_exc_avSamps[days2an_heatmap]
    
    ##%% NOTE: do you want to do the following??    
    if np.logical_and(mousename=='fni16', corrTrained==1):
        firstFrame_allN[days.index('150903_1')] = np.nan # ('150903_1') ... it is one of the early days and it is significant all over the trial!!! (rerun svm for this day!)        
        firstFrame_inh[days.index('150903_1')] = np.nan
        firstFrame_exc_randSamp[days.index('150903_1')] = np.nan
        firstFrame_exc_avSamps[days.index('150903_1')] = np.nan

    
    #%% Load behavioral performance
    
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
            
    
    #%% Keep vars of all mice
    
    eventI_ds_allDays_allM.append(eventI_ds_allDays)
    mn_corr_allM.append(mn_corr)
    behCorr_all_allM.append(behCorr_all)
    days2an_heatmap_allM.append(days2an_heatmap)    
    
    av_test_data_allExc_aligned_allM.append(av_test_data_allExc_aligned)
    av_test_data_exc_aligned_allM.append(av_test_data_exc_aligned)
    av_test_data_inh_aligned_allM.append(av_test_data_inh_aligned)
    
    firstFrame_allN_allM.append(firstFrame_allN)
    firstFrame_inh_allM.append(firstFrame_inh)
    firstFrame_exc_randSamp_allM.append(firstFrame_exc_randSamp)
    firstFrame_exc_avSamps_allM.append(firstFrame_exc_avSamps)

#no



#%%

#################################################################################################################################
##################################################### Plots of class accuracy ###################################################
#################################################################################################################################

#perc_thb = [20,80] # [10,90] #perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.

#%% ################## Behavior: compare CA for days with low vs high behavioral performance ##################

######################################## Set vars #######################################
   
a_allM = []
ah_allM = []
b_allM = []
bh_allM = []
c_allM = []
ch_allM = []

for im in range(len(mice)):

    eventI_ds_allDays = eventI_ds_allDays_allM[im]
    mn_corr = mn_corr_allM[im]
    behCorr_all = behCorr_all_allM[im]
    days2an_heatmap = days2an_heatmap_allM[im]
    
    av_test_data_allExc_aligned = av_test_data_allExc_aligned_allM[im]
    av_test_data_exc_aligned = av_test_data_exc_aligned_allM[im]
    av_test_data_inh_aligned = av_test_data_inh_aligned_allM[im]
    
    nPreMin = np.nanmin(eventI_ds_allDays[mn_corr >= thTrained]).astype(int)
    
    #################################### Set CA for days with low vs high behavioral performance ###############
    a = behCorr_all[days2an_heatmap]    
    thb = np.percentile(a, perc_thb)
#    mn = np.min(a);     mx = np.max(a);     thb = mn+.1, mx-.1    
    
    loBehCorrDays = (a <= thb[0])    
    hiBehCorrDays = (a >= thb[1])
    print sum(loBehCorrDays), sum(hiBehCorrDays), ': num days with low and high beh performance'
    
    aa = av_test_data_allExc_aligned[nPreMin-1, days2an_heatmap]
    bb = av_test_data_inh_aligned[nPreMin-1, days2an_heatmap]
    cc = av_test_data_exc_aligned[nPreMin-1, days2an_heatmap]
    a = aa[loBehCorrDays]; ah = aa[hiBehCorrDays]
    b = bb[loBehCorrDays]; bh = bb[hiBehCorrDays]
    c = cc[loBehCorrDays]; ch = cc[hiBehCorrDays]    
    
    a_allM.append(a)
    ah_allM.append(ah)
    b_allM.append(b)
    bh_allM.append(bh)
    c_allM.append(c)
    ch_allM.append(ch)
    
 
#%% 
######################################## Plots #######################################
    
mxH_allExc = max(np.array([ah_allM[im].mean() + ah_allM[im].std() / np.sqrt(len(ah_allM[im])) for im in range(len(mice))]))   
labs = 'allN', 'exc', 'inh'    
plt.figure(figsize=(4.5, 2))

for im in range(len(mice)):
    
    a = a_allM[im]
    ah = ah_allM[im]
    b = b_allM[im]
    bh = bh_allM[im]
    c = c_allM[im]
    ch = ch_allM[im]
    
    ####### errorbar: mean and st error: # compare CA for days with low vs high behavioral performance        
    plt.subplot(1, len(mice), im+1)
    x = range(len(labs))  

    plt.errorbar([0], [ah.mean()], [ah.std()/np.sqrt(len(ah))], marker='o', color='k', fmt='.', markeredgecolor='k', label='high beh', markersize=4)
    plt.errorbar([0], [a.mean()], [a.std()/np.sqrt(len(a))], marker='o', color='gray', fmt='.', markeredgecolor='gray', label='low beh', markersize=4)    
    plt.errorbar([1], [ch.mean()], [ch.std()/np.sqrt(len(ch))], marker='o', color='b', fmt='.', markeredgecolor='b', markersize=4)
    plt.errorbar([1], [c.mean()], [c.std()/np.sqrt(len(c))], marker='o', color='lightblue', fmt='.', markeredgecolor='lightblue', markersize=4)       
    plt.errorbar([2], [bh.mean()], [bh.std()/np.sqrt(len(bh))], marker='o',color='r', fmt='.', markeredgecolor='r', markersize=4)        
    plt.errorbar([2], [b.mean()], [b.std()/np.sqrt(len(b))], marker='o',color='lightsalmon', fmt='.', markeredgecolor='lightsalmon', markersize=4)        
    
    plt.xlim([x[0]-.5, x[-1]+.5]) # plt.xlim([-1,3])    
    plt.xticks([x[1]], [mice[im]], rotation=70) # 'vertical'
    plt.ylim([48, mxH_allExc])
    if im==len(mice)-1:
        plt.gca().legend(loc=0, frameon=False, numpoints=1, bbox_to_anchor=(.85, .93), bbox_transform=plt.gcf().transFigure)    
#    if im>0:
#        plt.gca().spines['left'].set_color('white')
#        plt.gca().yaxis.set_visible(False)
    if im==0:
        plt.ylabel('Class accur (%)')                    

    makeNicePlots(plt.gca(),0,1)
    
plt.subplots_adjust(hspace=.5, wspace=1)

#
if savefigs:
    sn = '_'.join(labs) + '_'
    
    if chAl==1:
        dd = 'chAl_' + corrn0 + 'classAccur_avSeDaysBeh_' + sn + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'chAl_' + corrn0 + 'classAccur_avSeDaysBeh_' + sn + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



#%%
#################################################################################################################################
##################################################### Plots of choice signal onset ###################################################
#################################################################################################################################

#perc_thb = [20,80] # [10,90] #perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.
    
#%% ################## Behavior: compare choice signal onset for days with low vs high behavioral performance ##################

######################################## Set vars #######################################

exc_av = 1 # if 1 use firstFrame_exc_avSamps # use firstFrame found for a random exc samp or found for each exc samp and then averaged!
    
a_allM = []
ah_allM = []
b_allM = []
bh_allM = []
c_allM = []
ch_allM = []

for im in range(len(mice)):

    behCorr_all = behCorr_all_allM[im]
    days2an_heatmap = days2an_heatmap_allM[im]    
    firstFrame_allN = firstFrame_allN_allM[im]
    firstFrame_inh = firstFrame_inh_allM[im]
    firstFrame_exc_randSamp = firstFrame_exc_randSamp_allM[im]
    firstFrame_exc_avSamps = firstFrame_exc_avSamps_allM[im]
    if exc_av:
        firstFrame_exc = firstFrame_exc_avSamps
    else:
        firstFrame_exc = firstFrame_exc_randSamp
        
    
    #########%% Compare choice signal onset for days with low vs high behavioral performance #########%%      
    #### allN
    a = behCorr_all[days2an_heatmap][~np.isnan(firstFrame_allN)] 
    thb = np.percentile(a, perc_thb)
#    mn = np.min(a);    mx = np.max(a);    thb = mn+.1, mx-.1        
    
    loBehCorrDays_allN = (a <= thb[0])     # bc different days are valid for allN, inh and exc, we have to find low and high beh performance days separately for each population.
    hiBehCorrDays_allN = (a >= thb[1])
    print sum(loBehCorrDays_allN), sum(hiBehCorrDays_allN), ': num days with low and high beh performance'

    #### inh
    a = behCorr_all[days2an_heatmap][~np.isnan(firstFrame_inh)]     
    thb = np.percentile(a, perc_thb)
    
    loBehCorrDays_inh = (a <= thb[0])    
    hiBehCorrDays_inh = (a >= thb[1])
    print sum(loBehCorrDays_inh), sum(hiBehCorrDays_inh), ': num days with low and high beh performance'

    #### exc
    a = behCorr_all[days2an_heatmap][~np.isnan(firstFrame_exc)] 
    thb = np.percentile(a, perc_thb)
    
    loBehCorrDays_exc = (a <= thb[0])    
    hiBehCorrDays_exc = (a >= thb[1])
    print sum(loBehCorrDays_exc), sum(hiBehCorrDays_exc), ': num days with low and high beh performance'

    
    aa = firstFrame_allN[~np.isnan(firstFrame_allN)]
    bb = firstFrame_inh[~np.isnan(firstFrame_inh)]
    cc = firstFrame_exc[~np.isnan(firstFrame_exc)]
    a = aa[loBehCorrDays_allN]; ah = aa[hiBehCorrDays_allN]
    b = bb[loBehCorrDays_inh]; bh = bb[hiBehCorrDays_inh]
    c = cc[loBehCorrDays_exc]; ch = cc[hiBehCorrDays_exc]      
    
    a_allM.append(a)
    ah_allM.append(ah)
    b_allM.append(b)
    bh_allM.append(bh)
    c_allM.append(c)
    ch_allM.append(ch)
    
 
#%% 
######################################## Plots #######################################
    
mx_allExc = max(np.array([a_allM[im].mean() + a_allM[im].std() / np.sqrt(len(a_allM[im])) for im in range(len(mice))]))   
mx_exc = max(np.array([c_allM[im].mean() + c_allM[im].std() / np.sqrt(len(c_allM[im])) for im in range(len(mice))]))   
mx_inh = max(np.array([b_allM[im].mean() + b_allM[im].std() / np.sqrt(len(b_allM[im])) for im in range(len(mice))]))   
mxH_allExc = max(np.array([ah_allM[im].mean() + ah_allM[im].std() / np.sqrt(len(ah_allM[im])) for im in range(len(mice))]))   
mxH_exc = max(np.array([ch_allM[im].mean() + ch_allM[im].std() / np.sqrt(len(ch_allM[im])) for im in range(len(mice))]))   
mxH_inh = max(np.array([bh_allM[im].mean() + bh_allM[im].std() / np.sqrt(len(bh_allM[im])) for im in range(len(mice))]))   
mx = np.ceil(max(mx_allExc, mx_exc, mx_inh, mxH_allExc, mxH_exc, mxH_inh)+70)

mn_allExc = min(np.array([a_allM[im].mean() - a_allM[im].std() / np.sqrt(len(a_allM[im])) for im in range(len(mice))]))   
mn_exc = min(np.array([c_allM[im].mean() - c_allM[im].std() / np.sqrt(len(c_allM[im])) for im in range(len(mice))]))   
mn_inh = min(np.array([b_allM[im].mean() - b_allM[im].std() / np.sqrt(len(b_allM[im])) for im in range(len(mice))]))   
mnH_allExc = min(np.array([ah_allM[im].mean() - ah_allM[im].std() / np.sqrt(len(ah_allM[im])) for im in range(len(mice))]))   
mnH_exc = min(np.array([ch_allM[im].mean() - ch_allM[im].std() / np.sqrt(len(ch_allM[im])) for im in range(len(mice))]))   
mnH_inh = min(np.array([bh_allM[im].mean() - bh_allM[im].std() / np.sqrt(len(bh_allM[im])) for im in range(len(mice))]))   
mn = np.floor(min(mn_allExc, mn_exc, mn_inh, mnH_allExc, mnH_exc, mnH_inh)-40)


labs = 'allN', 'exc', 'inh'    
plt.figure(figsize=(5.5, 2))

for im in range(len(mice)):
    
    a = a_allM[im]
    ah = ah_allM[im]
    b = b_allM[im]
    bh = bh_allM[im]
    c = c_allM[im]
    ch = ch_allM[im]
    
    ####### errorbar: mean and st error: # compare CA for days with low vs high behavioral performance        
    plt.subplot(1, len(mice), im+1)
    x = range(len(labs))  

    plt.errorbar([0], [ah.mean()], [ah.std()/np.sqrt(len(ah))], marker='o', color='k', fmt='.', markeredgecolor='k', label='high beh', markersize=4)
    plt.errorbar([0], [a.mean()], [a.std()/np.sqrt(len(a))], marker='o', color='gray', fmt='.', markeredgecolor='gray', label='low beh', markersize=4)    
    plt.errorbar([1], [ch.mean()], [ch.std()/np.sqrt(len(ch))], marker='o', color='b', fmt='.', markeredgecolor='b', markersize=4)
    plt.errorbar([1], [c.mean()], [c.std()/np.sqrt(len(c))], marker='o', color='lightblue', fmt='.', markeredgecolor='lightblue', markersize=4)       
    plt.errorbar([2], [bh.mean()], [bh.std()/np.sqrt(len(bh))], marker='o',color='r', fmt='.', markeredgecolor='r', markersize=4)        
    plt.errorbar([2], [b.mean()], [b.std()/np.sqrt(len(b))], marker='o',color='lightsalmon', fmt='.', markeredgecolor='lightsalmon', markersize=4)        
    
    plt.xlim([x[0]-.5, x[-1]+.5]) # plt.xlim([-1,3])    
    plt.xticks([x[1]], [mice[im]], rotation=70) # 'vertical'
    plt.ylim([mn, mx])
    if im==len(mice)-1:
        plt.gca().legend(frameon=False, numpoints=1, bbox_to_anchor=(1.2, .93), bbox_transform=plt.gcf().transFigure)    
#    if im>0:
#        plt.gca().spines['left'].set_color('white')
#        plt.gca().yaxis.set_visible(False)
    if im==0:
        plt.ylabel('Choice signal onset (ms)\nrel. animal choice')

    makeNicePlots(plt.gca(),0,1)
    
plt.subplots_adjust(hspace=.5, wspace=1.5)

#
if savefigs:
    sn = '_'.join(labs) + '_'
    
    if chAl==1:
        dd = 'chAl_' + corrn0 + 'onsetChoice_avSeDaysBeh_' + sn + '_'.join(mice) + '_' + nowStr
    else:
        dd = 'chAl_' + corrn0 + 'onsetChoice_avSeDaysBeh_' + sn + '_'.join(mice) + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



        