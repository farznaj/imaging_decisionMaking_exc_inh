# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:57:41 2017

@author: farznaj
"""

inum = 100;
n = 1000
ang = np.full((inum), np.nan)
angperm = np.full((inum), np.nan)

for i in range(inum):
    w = np.concatenate((rng.rand(n*3.5/4), -rng.rand(n*.5/4))) # neurons
    nw = np.linalg.norm(w) # frames; 2-norm of weights 
    w_n = w/nw 
    
    
    w = np.concatenate((rng.rand(n*3.5/4), -rng.rand(n*.5/4)))  # neurons
    nw = np.linalg.norm(w) # frames; 2-norm of weights 
    w_n2 = w/nw 

    nord = rng.permutation(n) # shuffle neurons 
    nord1 = rng.permutation(n)
    
    ang[i] = np.arccos(abs(np.dot(w_n[nord].transpose(), w_n2[nord1])))*180/np.pi 

    angperm[i] = np.arccos(abs(np.dot(w_n[nord].transpose(), w_n[nord1])))*180/np.pi 
   
ang.mean(), angperm.mean()

   
#%%


