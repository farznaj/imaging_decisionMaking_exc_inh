# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:06:12 2018

@author: farznaj
"""


#%%
regType = 'l2' # 'l2' : regularization type
kfold = 10
numSamples = 10 #100; # number of iterations for finding the best c (inverse of regularization parameter)
thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.

analysFolders = 'V1_Cux2_StaticGratings', 'V1_Rbp4_StaticGratings', 'V2_Cux2_StaticGratings', 'V2_Rbp4_StaticGratings' #, 'Natural Scenes'


import scipy as sci
import scipy.io as scio
import numpy as np   
import numpy.random as rng
from crossValidateModel import *
from linearSVM import *
from matplotlib import pyplot as plt
import h5py
import glob
import os



#%% Function to only show left and bottom axes of plots, make tick directions outward, remove every other tick label if requested.

def makeNicePlots(ax, rmv2ndXtickLabel=0, rmv2ndYtickLabel=0):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # Make tick directions outward    
    ax.tick_params(direction='out')    
    # Tweak spacing between subplots to prevent labels from overlapping
    #plt.subplots_adjust(hspace=0.5)
#    ymin, ymax = ax.get_ylim()

    # Remove every other tick label
    if rmv2ndXtickLabel:
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[::2]]
        
    if rmv2ndYtickLabel:
        [label.set_visible(False) for label in ax.yaxis.get_ticklabels()[::2]]
    
    # gap between tick labeles and axis
#    ax.tick_params(axis='x', pad=30)

#    plt.xticks(x, labels, rotation='vertical')
    #ax.xaxis.label.set_color('red')    
#    plt.gca().spines['left'].set_color('white')
    #plt.gca().yaxis.set_visible(False)


#%% Define perClassError: percent difference between Y and Yhat, ie classification error

def perClassError(Y, Yhat):
    import numpy as np
    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
    return perClassEr

    
#%% Loop over folders (V1,V2,different mouse lines)
    
for ifold in np.arange(1,len(analysFolders)): #range(len(analysFolders)): #
    
    dirData = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal', analysFolders[ifold])
    allFiles = glob.glob(os.path.join(dirData,'*.h5')) 
       
    
    #%% Loop over days
    
    wAllC_allDays = []
    bAllC_allDays = []
    perClassErrorTrain_allDays = []
    perClassErrorTest_allDays = []
    perClassErrorTest_shfl_allDays = []
    perClassErrorTest_chance_allDays = []
        
    for iday in range(len(allFiles)):
            
        #%% If files are mat format
        '''    
        data = scio.loadmat('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal/VISpRbp4Cre_TrialsByNeurons.mat')
        X = data.pop('VISpRbp4CreKL100ori')
        np.shape(X) # trials x neurons
        
        data = scio.loadmat('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal/VISpRbp4Cre_StimulusVector.mat')
        Y = data.pop('stim_vector').flatten()
        np.shape(Y)
        '''
        
        #%% If files are h5 format
        
        #f = h5py.File('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal/VISp_Cux2-CreERT2_ori_data.h5', 'r')
        print allFiles[iday]
        f = h5py.File(allFiles[iday], 'r')
        
        #print("Keys: %s" % f.keys())
        
        a_group_key = list(f.keys())[1]
        X = list(f[a_group_key]) # trials x neurons
        
        a_group_key = list(f.keys())[2]
        Y = list(f[a_group_key])
        
        X = np.array(X)
        Y = np.array(Y)
        
        print np.shape(X), np.shape(Y)
    
    
        #%%
        Y[Y==90] = 1
    
    
        #%% Keep a copy of X_svm before normalization
        
        X000 = X # + 0
        #X00 = X + 0
        print np.min(X000), np.max(X000)
    
    
        #%% Average over the last frame of the stimulus and the 1 frame after that
    
        X00 = np.mean(X000[:,:,6:9], 2)
    #    np.shape(X00)
    
    
        #%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 
        
        ##### set mean and sd of neural activity across trials
        m = np.mean(X00, axis=0)
        s = np.std(X00, axis=0)   
        
        #s = s+thAct     
        
        ##### do normalization
        X_N = (X00 - m) / s
            
        X = X_N
        print np.min(X), np.max(X)
        
            
        #%% Do SVM
            
        cvect = [.01]#,.001] #10**(np.arange(-6, 6,0.2))/Y.shape[0]
        #10**(np.arange(3, 5,.1))/Y.shape[0]
    #    print 'try the following regularization values: \n', cvect
        nCvals = len(cvect)
        
        
        wAllC = np.ones((numSamples, nCvals, X.shape[1]))+np.nan;
        bAllC = np.ones((numSamples, nCvals))+np.nan;
        
        perClassErrorTrain = np.ones((numSamples, nCvals))+np.nan;
        perClassErrorTest = np.ones((numSamples, nCvals))+np.nan;
        
        perClassErrorTest_shfl = np.ones((numSamples, nCvals))+np.nan
        perClassErrorTest_chance = np.ones((numSamples, nCvals))+np.nan
        
        no = Y.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)    
            
        for s in range(numSamples): # permute trials to get numSamples different sets of training and testing trials.
        
            print 'Iteration %d' %(s)
        
            permIxs = rng.permutation(len_test)  
        
            a_corr = np.zeros(len_test)
            if rng.rand()>.5:
                b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
            else:
                b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
            a_corr[b] = 1
            
            for i in range(nCvals): # train SVM using different values of regularization parameter             
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = 1)
                
                wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i] = np.squeeze(summary.model.intercept_);
                
                # classification errors                    
                perClassErrorTrain[s,i] = summary.perClassErrorTrain;
                perClassErrorTest[s,i] = summary.perClassErrorTest;                
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                ypredict = summary.model.predict(summary.XTest)
                perClassErrorTest_shfl[s,i] = perClassError(summary.YTest[permIxs], ypredict);
                perClassErrorTest_chance[s,i] = perClassError(a_corr, ypredict);        
                
           
       
        #%% Keep vars of all days
       
        wAllC_allDays.append(wAllC.squeeze())
        bAllC_allDays.append(bAllC.squeeze())
        perClassErrorTrain_allDays.append(perClassErrorTrain.squeeze())
        perClassErrorTest_allDays.append(perClassErrorTest.squeeze())
        perClassErrorTest_shfl_allDays.append(perClassErrorTest_shfl.squeeze())
        perClassErrorTest_chance_allDays.append(perClassErrorTest_chance.squeeze())
    
    
    #%% Save vars for each folder
    
    dirAnalys = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal_SVM', analysFolders[ifold])
    if not os.path.exists(dirAnalys):
        os.makedirs(dirAnalys)
    
    saveName = os.path.join(dirAnalys , 'svm_linear_l2') #os.path.basename(allFiles[iday])[0:-3])
        
    scio.savemat(saveName, {'perClassErrorTrain_allDays':perClassErrorTrain_allDays,
        'perClassErrorTest_allDays':perClassErrorTest_allDays,
        'perClassErrorTest_shfl_allDays':perClassErrorTest_shfl_allDays,
        'perClassErrorTest_chance_allDays':perClassErrorTest_chance_allDays})    
    
    
    #%% Average and standar error across samples, for each day and turn class error to accuracy
        
    perClassErrorTest_allDays_avSamps = 100-np.mean(perClassErrorTest_allDays, axis=1)
    perClassErrorTest_shfl_allDays_avSamps = 100-np.mean(perClassErrorTest_shfl_allDays, axis=1)
    
    perClassErrorTest_allDays_seSamps = np.std(perClassErrorTest_allDays, axis=1)/np.sqrt(numSamples)
    perClassErrorTest_shfl_allDays_seSamps = np.std(perClassErrorTest_shfl_allDays, axis=1)/np.sqrt(numSamples)
    
        
    #%% Plot class accur vs day
        
    plt.figure(figsize=(4.5,3))
    
    plt.errorbar(np.arange(len(allFiles))+1, perClassErrorTest_allDays_avSamps, perClassErrorTest_allDays_seSamps, label='data')    
    plt.errorbar(np.arange(len(allFiles))+1, perClassErrorTest_shfl_allDays_avSamps, perClassErrorTest_shfl_allDays_seSamps, label='shuffled')    
    plt.xlabel('Session #')    
    plt.ylabel('Classification accuracy (%)\n (cross-validated)')
    plt.xlim([0, len(allFiles)+2])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    makeNicePlots(plt.gca(),0,0)        
    
    
    plt.savefig(saveName, bbox_inches='tight') 


#%%

### V1_Cux2
dirAnalys = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal_SVM', analysFolders[0])
saveName = os.path.join(dirAnalys , 'svm_linear_l2')
    
data = scio.loadmat(saveName)
perClassErrorTest_allDays = data.pop('perClassErrorTest_allDays')
perClassErrorTest_shfl_allDays = data.pop('perClassErrorTest_shfl_allDays')
# Average and standar error across samples, for each day and turn class error to accuracy    
perClassErrorTest_allDays_avSamps_V1_Cux2 = 100-np.mean(perClassErrorTest_allDays, axis=1)
perClassErrorTest_shfl_allDays_avSamps_V1_Cux2 = 100-np.mean(perClassErrorTest_shfl_allDays, axis=1)

perClassErrorTest_allDays_seSamps_V1_Cux2 = np.std(perClassErrorTest_allDays, axis=1)/np.sqrt(numSamples)
perClassErrorTest_shfl_allDays_seSamps_V1_Cux2 = np.std(perClassErrorTest_shfl_allDays, axis=1)/np.sqrt(numSamples)



### V1_Rbp4
dirAnalys = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal_SVM', analysFolders[1])
saveName = os.path.join(dirAnalys , 'svm_linear_l2')
    
data = scio.loadmat(saveName)
perClassErrorTest_allDays = data.pop('perClassErrorTest_allDays')
perClassErrorTest_shfl_allDays = data.pop('perClassErrorTest_shfl_allDays')
# Average and standar error across samples, for each day and turn class error to accuracy    
perClassErrorTest_allDays_avSamps_V1_Rbp4 = 100-np.mean(perClassErrorTest_allDays, axis=1)
perClassErrorTest_shfl_allDays_avSamps_V1_Rbp4 = 100-np.mean(perClassErrorTest_shfl_allDays, axis=1)

perClassErrorTest_allDays_seSamps_V1_Rbp4 = np.std(perClassErrorTest_allDays, axis=1)/np.sqrt(numSamples)
perClassErrorTest_shfl_allDays_seSamps_V1_Rbp4 = np.std(perClassErrorTest_shfl_allDays, axis=1)/np.sqrt(numSamples)


### V2_Cux2
dirAnalys = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal_SVM', analysFolders[2])
saveName = os.path.join(dirAnalys , 'svm_linear_l2')
    
data = scio.loadmat(saveName)
perClassErrorTest_allDays = data.pop('perClassErrorTest_allDays')
perClassErrorTest_shfl_allDays = data.pop('perClassErrorTest_shfl_allDays')
# Average and standar error across samples, for each day and turn class error to accuracy    
perClassErrorTest_allDays_avSamps_V2_Cux2 = 100-np.mean(perClassErrorTest_allDays, axis=1)
perClassErrorTest_shfl_allDays_avSamps_V2_Cux2 = 100-np.mean(perClassErrorTest_shfl_allDays, axis=1)

perClassErrorTest_allDays_seSamps_V2_Cux2 = np.std(perClassErrorTest_allDays, axis=1)/np.sqrt(numSamples)
perClassErrorTest_shfl_allDays_seSamps_V2_Cux2 = np.std(perClassErrorTest_shfl_allDays, axis=1)/np.sqrt(numSamples)


### V2_Rbp4
dirAnalys = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal_SVM', analysFolders[3])
saveName = os.path.join(dirAnalys , 'svm_linear_l2')
    
data = scio.loadmat(saveName)
perClassErrorTest_allDays = data.pop('perClassErrorTest_allDays')
perClassErrorTest_shfl_allDays_V2 = data.pop('perClassErrorTest_shfl_allDays')
# Average and standar error across samples, for each day and turn class error to accuracy    
perClassErrorTest_allDays_avSamps_V2_Rbp4 = 100-np.mean(perClassErrorTest_allDays, axis=1)
perClassErrorTest_shfl_allDays_avSamps_V2_Rbp4 = 100-np.mean(perClassErrorTest_shfl_allDays, axis=1)

perClassErrorTest_allDays_seSamps_V2_Rbp4 = np.std(perClassErrorTest_allDays, axis=1)/np.sqrt(numSamples)
perClassErrorTest_shfl_allDays_seSamps_V2_Rbp4 = np.std(perClassErrorTest_shfl_allDays, axis=1)/np.sqrt(numSamples)



#%% Plot class accur vs day
    
plt.figure(figsize=(4.5,3))

plt.subplot(121)
plt.errorbar([1,2], [perClassErrorTest_allDays_avSamps_V1_Cux2.mean(), perClassErrorTest_allDays_avSamps_V2_Cux2.mean()] , [perClassErrorTest_allDays_avSamps_V1_Cux2.std(), perClassErrorTest_allDays_avSamps_V2_Cux2.std()], label='data', marker='o',color='k', fmt='.')    
plt.errorbar([1,2], [perClassErrorTest_shfl_allDays_avSamps_V1_Cux2.mean(), perClassErrorTest_shfl_allDays_avSamps_V2_Cux2.mean()] , [perClassErrorTest_shfl_allDays_avSamps_V1_Cux2.std(), perClassErrorTest_shfl_allDays_avSamps_V2_Cux2.std()], label='shuffled', marker='o',color='g', fmt='.')    
plt.xlim([0, 3])
plt.ylim([45, 80])
#plt.errorbar(2, perClassErrorTest_allDays_avSamps_V2_Cux2.mean() , perClassErrorTest_allDays_avSamps_V2_Cux2.std(), label='data(V2_Cux2)', color='k')    
#plt.errorbar(2, perClassErrorTest_shfl_allDays_avSamps_V2_Cux2.mean() , perClassErrorTest_shfl_allDays_avSamps_V2_Cux2.std(), label='shuffled(V2_Cux2)', color='g')    

plt.ylabel('Classification accuracy (%)\n (cross-validated)')
plt.xticks([1,2], ('V1_Cux2', 'V2_Cux2'), rotation=20)
#plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
makeNicePlots(plt.gca(),0,0)        


############ Rbp4
plt.subplot(122)
plt.errorbar([1,2], [perClassErrorTest_allDays_avSamps_V1_Rbp4.mean(), perClassErrorTest_allDays_avSamps_V2_Rbp4.mean()] , [perClassErrorTest_allDays_avSamps_V1_Rbp4.std(), perClassErrorTest_allDays_avSamps_V2_Rbp4.std()], label='data', marker='o',color='k', fmt='.')    
plt.errorbar([1,2], [perClassErrorTest_shfl_allDays_avSamps_V1_Rbp4.mean(), perClassErrorTest_shfl_allDays_avSamps_V2_Rbp4.mean()] , [perClassErrorTest_shfl_allDays_avSamps_V1_Rbp4.std(), perClassErrorTest_shfl_allDays_avSamps_V2_Rbp4.std()], label='shuffled', marker='o',color='g', fmt='.')    
plt.xlim([0, 3])
plt.ylim([45, 80])

#plt.ylabel('Classification accuracy (%)\n (cross-validated)')
plt.xticks([1,2], ('V1_Rbp4', 'V2_Rbp4'), rotation=20)#, rotation='vertical')
plt.legend(loc='center left', bbox_to_anchor=(.8, .7), frameon=False, numpoints=1)    
makeNicePlots(plt.gca(),0,0)        


saveName = os.path.join('/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/AllenProposal_SVM', 'classAccur_aveDays_Cux_Rbp_V1_V2.pdf')
plt.savefig(saveName, bbox_inches='tight') 
    
    

#%% Find bestc for each frame, and plot the c path
'''
# set best c 

smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.


######%% Compute average of class errors across numSamples        
meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:], axis = 0);
semPerClassErrorTrain = np.std(perClassErrorTrain[:,:], axis = 0)/np.sqrt(numSamples);        
meanPerClassErrorTest = np.mean(perClassErrorTest[:,:], axis = 0);
semPerClassErrorTest = np.std(perClassErrorTest[:,:], axis = 0)/np.sqrt(numSamples);        
meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:], axis = 0);
semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:], axis = 0)/np.sqrt(numSamples);        
#meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:], axis = 0);
#semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:], axis = 0)/np.sqrt(numSamples);


######%% Identify best c :        

# Use all range of c... it may end up a value at which all weights are 0.
ix = np.argmin(meanPerClassErrorTest)
if smallestC==1:
    cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
    cbest = cbest[0]; # best regularization term based on minError+SE criteria
    cbestAll = cbest
else:
    cbestAll = cvect[ix]

cbest = cbestAll
    
print cbest

#######%% Set the decoder and class errors at best c (for data)
"""
# you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
# we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
indBestC = np.in1d(cvect, cbest)

w_bestc_data = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
b_bestc_data = bAllC[:,indBestC]

classErr_bestC_train_data = perClassErrorTrain[:,indBestC].squeeze()

classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze()
classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC].squeeze()
classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC].squeeze()
"""


#%%######################%% plot C path           

#        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
plt.figure()
plt.subplot(1,2,1)
plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
#    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        

plt.plot(cvect, meanPerClassErrorTrain, 'ko', label = 'training')
plt.plot(cvect, meanPerClassErrorTest, 'ro', label = 'testing')
#plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
plt.plot(cvect, meanPerClassErrorTest_shfl, 'yo', label = 'cv-shfl')            
plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'ro')

if len(cvect)>1:
    plt.xlim([cvect[1], cvect[-1]])
    plt.xscale('log')
plt.xlabel('c (inverse of regularization parameter)')
plt.ylabel('classification error (%)')
plt.legend(loc='center left', bbox_to_anchor=(1, .7))

plt.tight_layout()




#%%
############################# Set the parameters below at bestc (for all samples):
#nFrs = x.shape[0]   

perClassErrorTrain_data = np.full((numSamples), np.nan)
perClassErrorTest_data = np.full((numSamples), np.nan)
perClassErrorTestShfl = np.full((numSamples), np.nan)
perClassErrorTestChance = np.full((numSamples), np.nan)
w_data = np.full((numSamples, X.shape[1]), np.nan)
b_data = np.full((numSamples), np.nan)
  
indBestC = np.in1d(cvect, cbest)
perClassErrorTrain_data[:] = perClassErrorTrain[:,indBestC].squeeze()
perClassErrorTest_data[:] = perClassErrorTest[:,indBestC].squeeze()
perClassErrorTestShfl[:] = perClassErrorTest_shfl[:,indBestC].squeeze()
perClassErrorTestChance[:] = perClassErrorTest_chance[:,indBestC].squeeze()
w_data[:,:] = wAllC[:,indBestC,:].squeeze()
b_data[:] = bAllC[:,indBestC].squeeze()

# plot average across samples
plt.figure()
#    plt.plot(time_trace, np.mean(perClassErrorTrain_data,axis=0), 'k')
plt.plot(0, np.mean(perClassErrorTest_data,axis=0), 'ro')
#plt.plot(0, np.mean(perClassErrorTest_data_incorr,axis=0), color=[.5,0,0])
plt.plot(0, np.mean(perClassErrorTestShfl,axis=0), 'yo')
    
'''
    
    
   
   