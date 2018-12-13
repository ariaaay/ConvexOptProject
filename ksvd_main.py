#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:15:57 2018

@author: hwehry
"""

import time
import numpy as np
import pickle
import spams
import sys
import pdb
from gensim.models import KeyedVectors
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt
from sklearn import decomposition
from util import *

dtype = np.float64

def ksvd_dictupdate(X,A,D,L):
    '''
    # K-SVD:
    # (dictionary update)
    '''
    # for every l-th atom in D...
    # update the dictionary atoms so that it minimizes errors obtained from compressed A
    for l in range(L):
        # indicies of the signals in X whose representations use Dl
        I = A[:,l] != 0 
        #if there are no contributions made by this atom to Ax, then continue
        if np.sum(I) == 0:
            continue
        
        #only signals containing nonzero elements are used            
        A_copy =A[I,:].copy()
        A_copy[:,l] = 0
        D_copy = D.copy()
        D_copy[l,:] = 0
        error = X[I,:] - np.dot(A_copy, D_copy)

        #produce a SVD decomp of the error matrix
        U,S,V = np.linalg.svd(error, full_matrices=True)
        D[l,:] = V[0,:]

        #update only the picked non-zero elements of Ax 
        A[I,l] = S[0] * U[:,0]
    
    diff = (X - np.dot(A, D))**2
    loss = np.mean(diff.sum(axis=1))
        
    return A,D,loss 

 

def ksvd_joint(X, Y, w, Axsim,Aysim, Sim=True,L=100, l1penalty=0.025):
    X = X-np.mean(X,axis=1,keepdims=True)
    Y = Y-np.mean(Y,axis=1,keepdims=True)

    nx,px = X.shape
    ny,py = Y.shape
    Ax = np.zeros((nx,L),dtype=dtype) #
    Ay = np.zeros((ny,L),dtype=dtype) # first w rows will be equal
    Dx = np.random.normal(size=(L,px)).astype(dtype)
    Dx = Dx / np.linalg.norm(Dx,ord=2,axis=1,keepdims=True).astype(dtype)
    Dy = np.random.normal(size=(L,py)).astype(dtype)
    Dy = Dy / np.linalg.norm(Dy,ord=2,axis=1,keepdims=True).astype(dtype)
    
    params = {'transform_alpha':l1penalty,'transform_n_nonzero_coefs':None,\
              'positive_code':True, 'transform_algorithm':'lasso_lars', \
              'split_sign':False, 'n_jobs':None}
    
    lossesx = []
    lossesy = []
    if Sim:
        lossesAx = []
        lossesAy = []
    
    output_fname = '../data/ksvd_Losses_edit_3.txt'  
    f = open(output_fname,'a+')
    f.write('\t'.join(['Step','Lx','Ly', 'LAx','LAy','\n']))
    f.close()
    
    converged = False
    steps = 0
    tol = 1e-6
    curr_obj = 1e6

    while not converged and steps < 100:
        prev_obj = curr_obj
        print('Iteration: %s' % steps)
        
        # Update shared A
        sprscoder = decomposition.SparseCoder(np.hstack((Dx,Dy)), **params)
        Aw = sprscoder.transform(np.hstack((X[:w,:],Y[:w,:])))
        Ax[:w,:] = Aw
        Ay[:w,:] = Aw
        # Update Ax not in Ay
        sprscoder = decomposition.SparseCoder(Dx, **params)
        Ax[w:,:] = sprscoder.transform(X[w:,:])       
        # Update Ay not in Ax
        sprscoder = decomposition.SparseCoder(Dy, **params)
        Ay[w:,:] = sprscoder.transform(Y[w:,:])
 
        Ax,Dx,lossx = ksvd_dictupdate(X,Ax,Dx,L)
        Ay[:w,:] = Ax[:w,:]
        Ay,Dy,lossy = ksvd_dictupdate(Y,Ay,Dy,L)
        Ax[:w,:] = Ay[:w,:]
        lossesx.append(lossx)
        print('Loss of X: %f' % lossx)
        lossesy.append(lossy)
        print('Loss of Y: %f' % lossy)
        
        ## Initialize sparse Ax
#        if steps == 0:
#            v
#            Ax = sprscoder.transform(X)
#            sprscoder = decomposition.SparseCoder(Dy, **params)
#            Ay = sprscoder.transform(Y)
#        
#        Ax,Dx,lossx = ksvd_dictupdate(X,Ax,Dx,L)
#        lossesx.append(lossx)
#        print('Loss of X: %f' % lossx)
#
#        sprscoder = decomposition.SparseCoder(Dx, **params)
#        Ax = sprscoder.transform(X)        
#        Ay[:w,:] = Ax[:w,:]
#        
#        Ay,Dy,_ = ksvd_dictupdate(Y,Ay,Dy,L)
#        sprscoder = decomposition.SparseCoder(Dy, **params)
#        Ay = sprscoder.transform(Y)
#        lossy = np.mean(np.sum((Y- Ay @ Dy)**2,axis=1))
#        lossesy.append(lossy)
#        print('Loss of Y: %f' % lossy)
#        
#        Awdiff = np.mean(np.sum((Ax[:w,:] - Ay[:w,:])**2,axis=1))
#        Awdiffs.append(Awdiff)        
#        Ax[:w,:] = Ay[:w,:]
        
        if Sim:
            lossAx = np.mean(np.sum((Ax - Axsim)**2,axis=1))
            lossesAx.append(lossAx)
            print('Loss of Ax: %f' % lossAx)

            lossAy = np.mean(np.sum((Ay - Aysim)**2,axis=1))
            lossesAy.append(lossAy)
            print('Loss of Ay: %f' % lossAy)

        
        f = open(output_fname,'a+')
        f.write('\t'.join([np.str(steps),np.str(lossx), np.str(lossy),np.str(lossAx),np.str(lossAy),'\n']))
        f.close()
        
        steps+=1
        curr_obj = lossx + lossy
        converged = np.abs(prev_obj - curr_obj) <= tol
    
    return lossesx,lossesy,lossesAx,lossesAy,Awdiffs

if __name__ == '__main__':
    brain_data_path = "../data/S1_PPA_LH.npy"
    brain_labels_path = "../data/image_category.p"
    obj_embedding_path = "../data/pix2vec_200.model"

    # Brain data is provided as a single numpy array, labels as a pickled
    # Python list
    brain_data = np.load(brain_data_path)
    brain_labels = pickle.load(open(brain_labels_path, 'rb'))
    
    # Object embeddings are read from a gensim model file.
    wv_model = KeyedVectors.load(obj_embedding_path, mmap='r')
    obj_vectors = wv_model.vectors
    obj_labels = list(wv_model.vocab)
    brain_data_unique, brain_labels_unique = takeout_repeated_brain_trials(brain_data, brain_labels)
    X, Y, w = extract_common_objs(brain_data_unique, brain_labels_unique, obj_vectors, obj_labels)
    
    nx,px = X.shape
    ny,py = Y.shape
    X, Y, Axsim, Aysim = simulate_data(w, nx-w, ny-w, l=100, c=200)

    Lx,Ly,LAx,LAy,Awdiffs = ksvd_joint(X, Y, w, Axsim,Aysim, Sim=True,L=100, l1penalty=0.025)
    plt.plot(Lx)
    plt.plot(Ly)
    np.save('../data/ksvd_Lx.npy', Lx)
    pdb.set_trace()
    
#%%

data_array = np.loadtxt('../data/ksvd_Losses_edit_3.txt',skiprows=1)
plt.semilogy(data_array[:,1],label='X Loss')
plt.semilogy(data_array[:,2],label='Y Loss')
plt.xlabel('Iteration, k')
plt.ylabel('Loss')
plt.title('k-SVD does not return the true joint embedding')
plt.legend()
plt.show
    #%%
##%%
#    nx,px = X.shape
#    ny,py = Y.shape
#    X,Y = simulate_data(w, nx, ny, l=100, c=200)
#    nx,px = X.shape
#    ny,py = Y.shape
#    L = 100
#    iters = 10#000
#    l1penalty = 0.025
#    tol = 1e-8
#    #x_nonzero_coefs = .045*nx #~ l1 penalty
#    #y_nonzero_coefs = .045*ny
#    
#    Ax = np.zeros((nx,L),dtype=dtype) #
#    Ay = np.zeros((ny,L),dtype=dtype) # first w rows will be equal
#    Dx = np.random.normal(size=(L,px)).astype(dtype)
#    Dx = Dx / np.linalg.norm(Dx,ord=2,axis=1,keepdims=True).astype(dtype)
#    Dy = np.random.normal(size=(L,py)).astype(dtype)
#    Dy = Dy / np.linalg.norm(Dy,ord=2,axis=1,keepdims=True).astype(dtype)
#    
#    params = {'transform_alpha':l1penalty,'transform_n_nonzero_coefs':None,\
#              'positive_code':True, 'transform_algorithm':'lasso_lars', \
#              'split_sign':False, 'n_jobs':None}
#    
#    print(np.sum((X - np.dot(Ax, Dx))**2))
#
#    lossesA = []
#    lossesX = []
#    for it in range(iters):
#        print(it)
#        Dx_old = Dx.copy()
#        
#        '''
#        # Sparse Coding Stage:
#        # (sparse representation update, given D)
#        '''
#        #Use any pursuit algorithm to compute 
#        #∀i : Γi := Argmin_γ || xi − Dγ||_2  s.t. ||γ||_0 ≤ K
#        #for i = 1, 2, . . . , N
#        sprscoder = decomposition.SparseCoder(Dx, **params)
#        Ax = sprscoder.transform(X)
##        for i in range(nx):
##            gx = sprscoder.transform(X[i,:].reshape(1, -1))
##            Ax[i,:] = gx
#        
#        '''
#        # K-SVD:
#        # (dictionary update)
#        '''
#        Ax,Dx,loss = ksvd_dictupdate(X,Ax,Dx,L)
#        
#        print(loss)
#        lossesX.append(loss)
#        
#    nplosses = np.asarray(lossesX)
#    #np.save('../data/ksvd_losses.npy', nplosses)
#    plt.plot(nplosses)
    
#%%
