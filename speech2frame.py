#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:07:52 2018

@author: leon
"""

from mm import Bunch, generateFace, exportObj, importObj, rotMat2angle
import os, json
import numpy as np
import librosa
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
from mayavi import mlab
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
from skimage import io
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, issparse
import networkx as nx

def speechProc(siroFile, siroNumFrames, siroFPS, kuroFile, kuroNumFrames):
    # Load audio tracks, pre-emphasize, and create feature vectors from mfcc, rmse, and deltas of mfcc
    nfft = 1024
    hopSamples = 512
    
    wav_kuro, fs_kuro = librosa.load(kuroFile, sr=44100)
    wav_kuro = np.r_[wav_kuro[0], wav_kuro[1:] - 0.97 * wav_kuro[:-1]]
    mfcc_kuro = librosa.feature.mfcc(y = wav_kuro, sr = fs_kuro, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
    mfcc_kuro[0, :] = librosa.feature.rmse(y = wav_kuro, n_fft = nfft, hop_length = hopSamples)
    delta_kuro = librosa.feature.delta(mfcc_kuro)
    mfcc_kuro = np.r_[mfcc_kuro, delta_kuro]
    
    wav_siro, fs_siro = librosa.load(siroFile, sr=44100)
    wav_siro = np.r_[wav_siro[0], wav_siro[1:] - 0.97 * wav_siro[:-1]]
    mfcc_siro = librosa.feature.mfcc(y = wav_siro, sr = fs_siro, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
    mfcc_siro[0, :] = librosa.feature.rmse(y = wav_siro, n_fft = nfft, hop_length = hopSamples)
    delta_siro = librosa.feature.delta(mfcc_siro)
    mfcc_siro = np.r_[mfcc_siro, delta_siro]
    
    # Find mfccs that are nearest to video frames in time
    t_video = np.linspace(0, siroNumFrames / siroFPS, siroNumFrames)
    
    t_audio_siro = np.linspace(0, mfcc_siro.shape[1] * hopSamples / fs_siro, mfcc_siro.shape[1])
    t_audio_kuro = np.linspace(0, mfcc_kuro.shape[1] * hopSamples / fs_kuro, mfcc_kuro.shape[1])
    
    NN = NearestNeighbors(n_neighbors = 1, metric = 'l1')
    
    NN.fit(t_audio_siro.reshape(-1, 1))
    distance, ind = NN.kneighbors(t_video.reshape(-1, 1))
    mfcc_siro_sampled = mfcc_siro[:, ind.squeeze()]
    
    NN.fit(t_audio_kuro.reshape(-1, 1))
    distance, ind = NN.kneighbors(t_video[:kuroNumFrames].reshape(-1, 1))
    mfcc_kuro_sampled = mfcc_kuro[:, ind.squeeze()]
    
    return mfcc_siro_sampled, mfcc_kuro_sampled, t_video

if __name__ == "__main__":

    os.chdir('/home/leon/f2f-fitting/obama/')
    numFramesSiro = 2882 #3744 #2260
    numFramesKuro = 2041
    
    siroAudioVec, kuroAudioVec, t_video = speechProc('siro.wav', numFramesSiro, 24, 'kuro.wav', numFramesKuro)
    
    # Find siro samples that are nearest to each kuro sample
    k = 20
    NN = NearestNeighbors(n_neighbors = k, metric = 'l2')
    
    NN.fit(siroAudioVec.T)
    distance, ind = NN.kneighbors(kuroAudioVec.T)
    
    # Calculate edge weights for candidate frames
    scaler = StandardScaler()
    
    param = np.load('paramRTS2Orig.npy')
    expCoef = scaler.fit_transform(param[:, 80: 76+80])
#    expCoef = param[:, 80: 76+80]
    angles = param[:, 76+80: 76+80+3]
    trans = scaler.fit_transform(param[:, 76+80+3: 76+80+5])
    R = np.empty((numFramesSiro, 3, 3))
    for i in range(numFramesSiro):
        R[i, ...] = rotMat2angle(angles[i, :])
    
    lmPairs = np.array([[42, 47], [43, 46], [44, 45], [30, 36], [42, 45], [44, 47], [25, 29], [26, 28], [19, 23], [20, 22]])
    lm = np.empty((numFramesSiro, 70, 2))
    for i in range(numFramesSiro):
        with open('landmark/' + '{:0>5}'.format(i+1) + '.json', 'r') as fd:
            lm[i, ...] = np.array([l[0] for l in json.load(fd)], dtype = int).squeeze()[:, :2]
            
    # Get landmark locations on 3DMM
    # Load 3DMM
    m = Bunch(np.load('../models/bfm2017.npz'))
    m.idEvec = m.idEvec[:, :, :80]
    m.idEval = m.idEval[:80]
    m.expEvec = m.expEvec[:, :, :76]
    m.expEval = m.expEval[:76]
    m.texEvec = m.texEvec[:, :, :80]
    m.texEval = m.texEval[:80]
    
    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
    sourceLmPairs = sourceLandmarkInds[lmPairs]
    uniqueSourceLm, uniqueInv = np.unique(sourceLmPairs, return_inverse = True)
    
    mouthIdx = np.load('../bfmMouthIdx.npy')
    mouthVertices = np.load('mouthVertices.npy')
    mouthVertices = mouthVertices.reshape((numFramesSiro, mouthIdx.size, 3))
    
    # Enforce similarity in similarity transform parameters from candidate frames to original video frames
    Dp = np.empty((numFramesKuro, k))
    for q in range(numFramesKuro):
        c = ind[q, :]
        Dp[q, :] = np.linalg.norm(trans[q, :] - trans[c, :], axis = 1) + np.linalg.norm(R[q, ...] - R[c, ...], axis = (1, 2))
        
    # Transition between candidate frames should have similar 3DMM landmarks and expression parameters
    mmLm = np.empty((numFramesSiro, 3, uniqueSourceLm.size))
    for t in range(numFramesSiro):
        mmLm[t] = generateFace(param[t, :], m, ind = uniqueSourceLm)
    mmLm = mmLm[..., uniqueInv[::2]] - mmLm[..., uniqueInv[1::2]]
    mmLmNorm = np.linalg.norm(mmLm, axis = 1)
    
    Dm = np.empty((numFramesKuro - 1, k, k))
    weights = np.empty((numFramesKuro - 1, k, k))
    for t in range(numFramesKuro - 1):
        for c1 in range(k):
            Dm[t, c1] = np.linalg.norm(mmLmNorm[ind[t, c1], :] - mmLmNorm[ind[t+1, :], :], axis = 1) + np.linalg.norm(expCoef[ind[t, c1]] - expCoef[ind[t+1, :], :], axis = 1)
            
#            np.exp(-np.fabs(t_video[ind[t, c1]] - t_video[ind[t+1, :]])**2)
            
            weights[t, c1] = Dm[t, c1] + Dp[t, c1] + Dp[t+1, :]
    
    # Create DAG and assign edge weights from distance matrix
    G = nx.DiGraph()
    for i in range(numFramesKuro - 1):
        left = np.arange(i*k, (i+1)*k)
        right = np.arange((i+1)*k, (i+2)*k)
        G.add_nodes_from(left)
        G.add_nodes_from(right)
        G.add_weighted_edges_from((u, v, weights[i, u - i*k, v - (i+1)*k]) for u in left for v in right)
    
    # Use A* shortest path algorithms
    astarLength = np.empty((k, k))
    for s in range(k):
        for t in range(k):
            astarLength[s, t] = nx.astar_path_length(G, s, right[t])
    
    s, t = np.unravel_index(astarLength.argmin(), (k, k))
    optPath = nx.astar_path(G, s, right[t])
    optPath = np.unravel_index(optPath, (numFramesKuro, k))
    optPath = ind[optPath[0], optPath[1]]
    