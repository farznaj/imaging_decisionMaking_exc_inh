# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:59:12 2016

@author: farznaj

This is Farzaneh's first Python code :-) She is very happy and pleased about it :D

example call:

mousename = 'fni17'
imagingFolder = '151102'
mdfFileNumber = (1,2)

# optional inputs:
signalCh = [2]
pnev2load = []

from setImagingAnalysisNamesP import *

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load)
imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber)

"""

#%%
def setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, **options):

    #%%
    import numpy as np
    import platform
    import glob
    import os.path
    
    
    if options.get('signalCh'):
        signalCh = options.get('signalCh');    
    else:
        signalCh = []
        
    if options.get('pnev2load'):
        pnev2load = options.get('pnev2load');    
    else:
        pnev2load = []
        
    if len(pnev2load)==0:
        pnev2load = 0;
            
    #%%
    if platform.system()=='Linux':
        dataPath = '/home/farznaj/Shares/Churchland/data/'
    # Gamal's dir needed here.
        
    #%%        
    tifFold = dataPath+mousename+'/imaging/'+imagingFolder+'/'
    r = '%03d-'*len(mdfFileNumber)
    r = r[:-1]
    rr = r % (mdfFileNumber)
    
    date_major = imagingFolder+'_'+rr
    imfilename = tifFold+date_major+'.mat'
    
    #%%
    if len(signalCh)>0:
        pnevFileName = date_major+'_ch'+str(signalCh)+'-Pnev*'
        pnevFileName = glob.glob(tifFold+pnevFileName)   
        
        array = []
        for idx in range(0, len(pnevFileName)):
            array.append(os.path.getmtime(pnevFileName[idx]))
            
        #%%
        if len(pnevFileName)==0:
            c = ("No Pnev file was found"); print("%s\n" % c)
            pnevFileName = ''
        else:
            inds = np.argsort(array)
            inds = inds[::-1]
                
        pnevFileName = pnevFileName[inds[pnev2load]]
    else:
        pnevFileName = ''
    
    #%%
    return imfilename, pnevFileName
    
    
#%%
#imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh, pnev2load)

