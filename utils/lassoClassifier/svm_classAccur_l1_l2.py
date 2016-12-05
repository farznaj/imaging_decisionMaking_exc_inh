# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:26:33 2016

@author: farznaj
"""


# First run the beginning of mainSVM_notebook_plots
# Also run the class accuracy section there to get class accur for subselect, l1

#%%
'''
#####################################################################################################################################################   
############################ Classification accuracy (testing data) ###################################################################################################     
#####################################################################################################################################################
'''

#%% Loop over days

numSamp = 100 # number of shuffles
l1_err_test_data = np.full((numSamp, len(days)), np.nan)
l1_err_test_shfl = np.full((numSamp, len(days)), np.nan)
l2_err_test_data = np.full((numSamp, len(days)), np.nan)
l2_err_test_shfl = np.full((numSamp, len(days)), np.nan)
l1_err_train_data = np.full((numSamp, len(days)), np.nan)
l1_err_train_shfl = np.full((numSamp, len(days)), np.nan)
l2_err_train_data = np.full((numSamp, len(days)), np.nan)
l2_err_train_shfl = np.full((numSamp, len(days)), np.nan)
l1_wnon0_all = np.full((1, len(days)), np.nan).flatten()
l2_wnon0_all = np.full((1, len(days)), np.nan).flatten()

for iday in range(len(days)):
    
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
    
    #%%
    svmNameAll = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName, []) # latest is l1, then l2. we use [] to get both files.
    
    for i in range(len(svmNameAll)):
        svmName = svmNameAll[i]
        print os.path.basename(svmName)
            
        Data = scio.loadmat(svmName, variable_names=['w'])
        w = Data.pop('w')[0,:]
        
#        if abs(w.sum()) < eps:  # I think it is wrong to do this. When ws are all 0, it means there was no decoder for that day, ie no info about the choice.
#            print '\tIn round %d all weights are 0 ... not analyzing' %(roundi)
#            
#        else:    
        ##%% Load the class-loss array for the testing dataset (actual and shuffled) (length of array = number of samples, usually 100) 
        Data = scio.loadmat(svmName, variable_names=['regType', 'perClassErrorTest_data', 'perClassErrorTest_shfl', 'perClassErrorTrain_data', 'perClassErrorTrain_shfl'])
        regType = Data.pop('regType').astype(str)
        perClassErrorTest_data = Data.pop('perClassErrorTest_data')[0,:] # numSamples
        perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')[0,:]
        perClassErrorTrain_data = Data.pop('perClassErrorTrain_data')[0,:] # numSamples
        perClassErrorTrain_shfl = Data.pop('perClassErrorTrain_shfl')[0,:]
            
        if regType=='l1':
            l1_err_test_data[:, iday] = perClassErrorTest_data #it will be nan for rounds with all-0 weights
            l1_err_test_shfl[:, iday] = perClassErrorTest_shfl
            l1_err_train_data[:, iday] = perClassErrorTrain_data #it will be nan for rounds with all-0 weights
            l1_err_train_shfl[:, iday] = perClassErrorTrain_shfl
            l1_wnon0_all[iday] = (w!=0).mean()
    
        elif regType=='l2':
            l2_err_test_data[:, iday] = perClassErrorTest_data
            l2_err_test_shfl[:, iday] = perClassErrorTest_shfl
            l2_err_train_data[:, iday] = perClassErrorTrain_data
            l2_err_train_shfl[:, iday] = perClassErrorTrain_shfl
            l2_wnon0_all[iday] = (w!=0).mean()
            
'''
#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        err_test_data_ave_allDays0 = err_test_data_ave_allDays
        err_test_shfl_ave_allDays0 = err_test_shfl_ave_allDays

    elif iTiFlg==1:
        err_test_data_ave_allDays1 = err_test_data_ave_allDays
        err_test_shfl_ave_allDays1 = err_test_shfl_ave_allDays
'''        

#%% Average and std across shuffles
av_l1_test_d = 100-np.nanmean(l1_err_test_data, axis=0) # numDays
av_l1_test_s = 100-np.nanmean(l1_err_test_shfl, axis=0) 
sd_l1_test_d = np.nanstd(l1_err_test_data, axis=0) / np.sqrt(numSamp) 
sd_l1_test_s = np.nanstd(l1_err_test_shfl, axis=0) / np.sqrt(numSamp)  
av_l2_test_d = 100-np.nanmean(l2_err_test_data, axis=0) # numDays
av_l2_test_s = 100-np.nanmean(l2_err_test_shfl, axis=0) 
sd_l2_test_d = np.nanstd(l2_err_test_data, axis=0) / np.sqrt(numSamp)  
sd_l2_test_s = np.nanstd(l2_err_test_shfl, axis=0) / np.sqrt(numSamp)  
av_l1_train_d = 100-np.nanmean(l1_err_train_data, axis=0) # numDays
av_l1_train_s = 100-np.nanmean(l1_err_train_shfl, axis=0) 
sd_l1_train_d = np.nanstd(l1_err_train_data, axis=0) / np.sqrt(numSamp) 
sd_l1_train_s = np.nanstd(l1_err_train_shfl, axis=0) / np.sqrt(numSamp) 
av_l2_train_d = 100-np.nanmean(l2_err_train_data, axis=0) # numDays
av_l2_train_s = 100-np.nanmean(l2_err_train_shfl, axis=0) 
sd_l2_train_d = np.nanstd(l2_err_train_data, axis=0) / np.sqrt(numSamp) 
sd_l2_train_s = np.nanstd(l2_err_train_shfl, axis=0) / np.sqrt(numSamp) 

_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl)
p





######################## PLOTS ########################

#################################################################
#################### Testing data #########################
#################################################################
#%% Plot average across rounds for each day
### L1;
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-2])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), av_l1_test_d, yerr = sd_l1_test_d, color='g', label='L1 Data')
plt.errorbar(range(numDays), av_l1_test_s, yerr = sd_l1_test_s, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[-2:-1])
plt.errorbar(x, [np.mean(av_l1_test_d), np.mean(av_l1_test_s)], yerr = [np.std(av_l1_test_d), np.std(av_l1_test_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)

if savefigs:#% Save the figure
    #dnow = '/l1_l2_subsel_comparison'
    dnow = '/shortLongITI_afterSFN'
    fign = os.path.join(svmdir+dnow, suffn+'test_L1'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')  



### L2;
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-2])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), av_l2_test_d, yerr = sd_l2_test_d, color='g', label='L2 Data')
plt.errorbar(range(numDays), av_l2_test_s, yerr = sd_l2_test_s, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[-2:-1])
plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_test_s)], yerr = [np.std(av_l2_test_d), np.std(av_l2_test_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)

if savefigs:#% Save the figure
    #dnow = '/l1_l2_subsel_comparison'
    dnow = '/shortLongITI_afterSFN'
    fign = os.path.join(svmdir+dnow, suffn+'test_L2'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')


#%% All on the same figure: L1, L2, subsel
plt.figure(figsize=(8,2.5))
gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), ave_test_d, yerr = sd_test_d, color='g', label='Subsel L1')
plt.errorbar(range(numDays), av_l1_test_d, yerr = sd_l1_test_d, color='b', label='L1')
plt.errorbar(range(numDays), av_l2_test_d, yerr = sd_l2_test_d, color='k', label='L2')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
makeNicePlots(ax)


##%% Average across days
# data
x =[0,1,2]
labels = ['Subsel', 'L1', 'L2']
ax = plt.subplot(gs[3:4])
plt.errorbar(x, [np.mean(ave_test_d), np.mean(av_l1_test_d), np.mean(av_l2_test_d)], 
                 yerr = [np.std(ave_test_d), np.std(av_l1_test_d), np.std(av_l2_test_d)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.title('Data')
makeNicePlots(ax)


# shuffle
x =[0,1,2]
labels = ['Subsel', 'L1', 'L2']
ax = plt.subplot(gs[4:5])
plt.errorbar(x, [np.mean(ave_test_s), np.mean(av_l1_test_s), np.mean(av_l2_test_s)], 
                 yerr = [np.std(ave_test_s), np.std(av_l1_test_s), np.std(av_l2_test_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
plt.title('Shuffle')
makeNicePlots(ax)


plt.subplots_adjust(wspace=.7)

if savefigs:#% Save the figure
    #dnow = '/l1_l2_subsel_comparison'
    dnow = '/shortLongITI_afterSFN'
    fign = os.path.join(svmdir+dnow, suffn+'test_all'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')





#################################################################
#################### Training data #########################
#################################################################
#%% Plot average across rounds for each day
### L1;
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-2])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), av_l1_train_d, yerr = sd_l1_train_d, color='g', label='L1 Data')
plt.errorbar(range(numDays), av_l1_train_s, yerr = sd_l1_train_s, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(training data)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[-2:-1])
plt.errorbar(x, [np.mean(av_l1_train_d), np.mean(av_l1_train_s)], yerr = [np.std(av_l1_train_d), np.std(av_l1_train_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)

if savefigs:#% Save the figure
    #dnow = '/l1_l2_subsel_comparison'
    dnow = '/shortLongITI_afterSFN'
    fign = os.path.join(svmdir+dnow, suffn+'train_L1'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')  



### L2;
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-2])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), av_l2_train_d, yerr = sd_l2_train_d, color='g', label='L2 Data')
plt.errorbar(range(numDays), av_l2_train_s, yerr = sd_l2_train_s, color='k', label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(training data)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.25), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
x =[0,1]
labels = ['Data', 'Shfl']
ax = plt.subplot(gs[-2:-1])
plt.errorbar(x, [np.mean(av_l2_train_d), np.mean(av_l2_train_s)], yerr = [np.std(av_l2_train_d), np.std(av_l2_train_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)

if savefigs:#% Save the figure
        #dnow = '/l1_l2_subsel_comparison'
    dnow = '/shortLongITI_afterSFN'
    fign = os.path.join(svmdir+dnow, suffn+'test_L2'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')

        

#%% L1, L2, subsel
plt.figure(figsize=(8,4.5))
gs = gridspec.GridSpec(2, 5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0,0:3])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), ave_train_d, yerr = sd_train_d, color='g', label='Subsel L1') # , marker='o', fmt=' '
plt.errorbar(range(numDays), av_l1_train_d, yerr = sd_l1_train_d, color='b', label='L1')
plt.errorbar(range(numDays), av_l2_train_d, yerr = sd_l2_train_d, color='k', label='L2')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(training data)', fontsize=13, labelpad=10)
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.7), frameon=False)
makeNicePlots(ax)


##%% Average across days
x =[0,1,2]
labels = ['Subsel', 'L1', 'L2']
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, [np.mean(ave_train_d), np.mean(av_l1_train_d), np.mean(av_l2_train_d)], yerr = [np.std(ave_train_d), np.std(av_l1_train_d), np.std(av_l2_train_d)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
makeNicePlots(ax)


# shuffle
x =[0,1,2]
labels = ['Subsel', 'L1', 'L2']
ax = plt.subplot(gs[0,4:5])
plt.errorbar(x, [np.mean(ave_train_s), np.mean(av_l1_train_s), np.mean(av_l2_train_s)], 
                 yerr = [np.std(ave_train_s), np.std(av_l1_train_s), np.std(av_l2_train_s)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[-1]+1])
#plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.subplots_adjust(wspace=1)
plt.title('Shuffle')
makeNicePlots(ax)



# plot of fract non-0 w for each day (when using all trials for training, not when doing kfold)
ax = plt.subplot(gs[1,0:3])
plt.plot(l1_wnon0_all, label='L1', color='b')    
plt.plot(l2_wnon0_all, label='L2', color='k')
plt.ylim([0, 1.1])
plt.xlim([-1, len(days)])
plt.xlabel('Days')
plt.ylabel('Fract non-0 w')
plt.legend(loc=0, frameon=False)
makeNicePlots(ax)


plt.subplots_adjust(wspace=1, hspace=.4)

if savefigs:#% Save the figure
        #dnow = '/l1_l2_subsel_comparison'
    dnow = '/shortLongITI_afterSFN'
    fign = os.path.join(svmdir+dnow, suffn+'test_all'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')


