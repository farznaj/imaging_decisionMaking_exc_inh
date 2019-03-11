#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:05:00 2018

@author: farznaj
"""

import numpy as np
import matplotlib.pyplot as plt
#import rng

s = 3 # sparsitiy level
# d >> n # dimension >> number of samples
n = 50
d = 256

# d = 45

# y = A*x

x = np.zeros(d)
x[[20, 30, 40]] = np.random.randint(4, 10)


x = np.random.randn(d)

A = np.random.randn(n,d)

y = np.dot(A,x)


np.shape(x), np.shape(y)
np.shape(A)

y / A


x_est = np.dot(np.linalg.pinv(A), y)

x_est.shape
x_est[[20, 30, 40]] 

plt.plot(np.abs(x - x_est))


#%% OMP (orthogonal matching pursuit) --> to find the best dimensions ... picking the features ...  L1
# https://imgur.com/a/2sXQX42

import scipy as sci
import sys
eps = sys.float_info.epsilon 


#%%
eps = 10**-10
# A: nxd
x0 = 0 
r0 = y #n
s0 = set()

rall = []
for i in range(100):
    
    rall.append(r0)
    
    n = np.argmax(np.dot(r0, A))    
    s0.add(n)    
    
    x_est = np.dot(np.linalg.pinv(A[:, list(s0)]), y)
    
    r0 = y - np.dot(A[:, list(s0)], x_est)
        
    err = np.linalg.norm(r0)
    print err
    
    if np.logical_or(err < eps , i==d):
        sys.exit('converged')
        
    if i>1 and np.linalg.norm(rall[-1]) > np.linalg.norm(rall[-2]):
        sys.exit('converged')
    
    
    

