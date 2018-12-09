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

#def extract_common_objs(brain_data, brain_labels, obj_vectors, obj_labels):
#    """
#    This functions takes in NON-repeated brain data.
#    Outputs:
#    - X: object embedding matrix of dimension w1-by-c. First w rows of X corresponds to objects
#    that are also present in the brain data, in the same order as it is present in the brain data.
#    - Xlabels: labels corresponding to the rows in X.
#    - Y: the brain data matrix of dimension w2-by-v. 
#    - Ylabels: labels corresponding to the rows in Y.
#    - w: number of words that overlap
#    """
#    br_overlap_idx = []
#    obj_overlap_idx = []
#
#    for i, obj_lab in enumerate(obj_labels):
#        for j, br_lab in enumerate(brain_labels):
#            if obj_lab == br_lab:
#                obj_overlap_idx.append(i)
#                br_overlap_idx.append(j)
#    assert(len(obj_overlap_idx) == len(br_overlap_idx))
#    w = len(obj_overlap_idx)
#
#    br_nonoverlap_idx = np.setdiff1d(np.arange(len(brain_labels)), br_overlap_idx)
#    obj_nonoverlap_idx = np.setdiff1d(np.arange(len(obj_labels)), obj_overlap_idx)
#
#    X = np.vstack((obj_vectors[obj_overlap_idx,:], obj_vectors[obj_nonoverlap_idx,:]))
#    Y = np.vstack((brain_data[br_overlap_idx,:], brain_data[br_nonoverlap_idx,:]))
#
#    return X, Y, w
#
#def avg_repeated_brain_trials(brain_data, brain_labels):
#    """
#    Average the duplicated trials in the brain data
#    """
#    brain_labels_unique = []
#    label2trial = dict()
#    for i, l in enumerate(brain_labels):
#        if l not in brain_labels_unique:
#            brain_labels_unique.append(l)
#            label2trial[l] = [i]
#        else:
#            label2trial[l].append(i)
#    
#    brain_data_unique = np.zeros((len(brain_labels_unique), brain_data.shape[1]))
#    for j, lab in enumerate(brain_labels_unique):
#        if len(label2trial[lab]) == 1:
#            brain_data_unique[j,:] = brain_data[label2trial[lab],:]
#        else:
#            brain_data_unique[j,:] = np.mean([brain_data[idx,:] for idx in label2trial[lab]])
#    return brain_data_unique, brain_labels_unique
#    
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
    X,Y = simulate_data(w, nx, ny, l=100, c=200)
    nx,px = X.shape
    ny,py = Y.shape
    L = 100
    iters = 10#000
    l1penalty = 0.025
    tol = 1e-8
    #x_nonzero_coefs = .045*nx #~ l1 penalty
    #y_nonzero_coefs = .045*ny
    
    Ax = np.zeros((nx,L),dtype=dtype) #
    Ay = np.zeros((ny,L),dtype=dtype) # first w rows will be equal
    Dx = np.random.normal(size=(L,px)).astype(dtype)
    Dx = Dx / np.linalg.norm(Dx,ord=2,axis=1,keepdims=True).astype(dtype)
    Dy = np.random.normal(size=(L,py)).astype(dtype)
    Dy = Dy / np.linalg.norm(Dy,ord=2,axis=1,keepdims=True).astype(dtype)
    
    params = {'transform_alpha':l1penalty,'transform_n_nonzero_coefs':None,\
              'positive_code':True, 'transform_algorithm':'lasso_lars', \
              'split_sign':False, 'n_jobs':None}
    
    print(np.sum((X - np.dot(Ax, Dx))**2))

#%%
    lossesA = []
    lossesX = []
    for it in range(iters):
        print(it)
        Dx_old = Dx.copy()
        
        '''
        # Sparse Coding Stage:
        # (sparse representation update, given D)
        '''
        #Use any pursuit algorithm to compute 
        #∀i : Γi := Argmin_γ || xi − Dγ||_2  s.t. ||γ||_0 ≤ K
        #for i = 1, 2, . . . , N
        sprscoder = decomposition.SparseCoder(Dx, **params)
        Ax = sprscoder.transform(X)
#        for i in range(nx):
#            gx = sprscoder.transform(X[i,:].reshape(1, -1))
#            Ax[i,:] = gx
        
        '''
        # K-SVD:
        # (dictionary update)
        '''
        # for every l-th atom in D...
        # update the dictionary atoms so that it minimizes errors obtained from compressed A
        for l in range(L):
            # indicies of the signals in X whose representations use Dl
            I = Ax[:,l] != 0 
            #if there are no contributions made by this atom to Ax, then continue
            if np.sum(I) == 0:
                continue
            
            #only signals containing nonzero elements are used            
            Ax_copy =Ax[I,:].copy()
            Ax_copy[:,l] = 0
            Dx_copy = Dx.copy()
            Dx_copy[l,:] = 0
            error = X[I,:] - np.dot(Ax_copy, Dx_copy)
    
            #produce a SVD decomp of the error matrix
            U,S,V = np.linalg.svd(error, full_matrices=True)
            Dx[l,:] = V[0,:]

            #update only the picked non-zero elements of Ax 
            Ax[I,l] = S[0] * U[:,0]
                    

        error = (X - np.dot(Ax, Dx))**2
        loss = np.mean(error.sum(axis=1))
        print(loss)
        lossesX.append(loss)
#%%
    nplosses = np.asarray(lossesX)
    #np.save('../data/ksvd_losses.npy', nplosses)
    plt.plot(nplosses)