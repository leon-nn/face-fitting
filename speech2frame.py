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

if __name__ == "__main__":

    os.chdir('/home/leon/f2f-fitting/obama/')
    numFramesSiro = 2882 #3744 #2260
    numFramesKuro = 2041
    fps = 24
    
    # Load audio tracks, pre-emphasize, and create feature vectors from mfcc, rmse, and deltas of mfcc
    nfft = 1024
    hopSamples = 512
    
    wav_kuro, fs_kuro = librosa.load('kuro.wav', sr=44100)
    wav_kuro = np.r_[wav_kuro[0], wav_kuro[1:] - 0.97 * wav_kuro[:-1]]
    mfcc_kuro = librosa.feature.mfcc(y = wav_kuro, sr = fs_kuro, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
    mfcc_kuro[0, :] = librosa.feature.rmse(y = wav_kuro, n_fft = nfft, hop_length = hopSamples)
    delta_kuro = librosa.feature.delta(mfcc_kuro)
    mfcc_kuro = np.r_[mfcc_kuro, delta_kuro]
    
    wav_siro, fs_siro = librosa.load('siro.wav', sr=44100)
    wav_siro = np.r_[wav_siro[0], wav_siro[1:] - 0.97 * wav_siro[:-1]]
    mfcc_siro = librosa.feature.mfcc(y = wav_siro, sr = fs_siro, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
    mfcc_siro[0, :] = librosa.feature.rmse(y = wav_siro, n_fft = nfft, hop_length = hopSamples)
    delta_siro = librosa.feature.delta(mfcc_siro)
    mfcc_siro = np.r_[mfcc_siro, delta_siro]
    
    # Find mfccs that are nearest to video frames in time
    t_video = np.linspace(0, numFramesSiro / fps, numFramesSiro)
    
    t_audio_siro = np.linspace(0, mfcc_siro.shape[1] * hopSamples / fs_siro, mfcc_siro.shape[1])
    t_audio_kuro = np.linspace(0, mfcc_kuro.shape[1] * hopSamples / fs_kuro, mfcc_kuro.shape[1])
    
    NN = NearestNeighbors(n_neighbors = 1, metric = 'l1')
    
    NN.fit(t_audio_siro.reshape(-1, 1))
    distance, ind = NN.kneighbors(t_video.reshape(-1, 1))
    mfcc_siro_sampled = mfcc_siro[:, ind.squeeze()]
    
    NN.fit(t_audio_kuro.reshape(-1, 1))
    distance, ind = NN.kneighbors(t_video[:2041].reshape(-1, 1))
    mfcc_kuro_sampled = mfcc_kuro[:, ind.squeeze()]
    
    # Find siro samples that are nearest to each kuro sample
    k = 20
    NN = NearestNeighbors(n_neighbors = k, metric = 'l2')
    
    NN.fit(mfcc_siro_sampled.T)
    distance, ind = NN.kneighbors(mfcc_kuro_sampled.T)
    
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
    
    mouthIdx = np.load('../bfmMouthIdx.npy')
    mouthVertices = np.load('mouthVertices.npy')
    mouthVertices = mouthVertices.reshape((numFramesSiro, mouthIdx.size, 3))
    
    # Enforce similarity in similarity transform parameters from candidate frames to original video frames
    Dp = np.empty((k, numFramesKuro))
    for q in range(numFramesKuro):
        c = ind[q, :]
        Dp[:, q] = np.linalg.norm(trans[q, :] - trans[c, :], axis = 1) + np.linalg.norm(R[q, ...] - R[c, ...], axis = (1, 2))
        
        # Add another term for pixel landmark proximity
        
    # Transition between candidate frames should have similar 3DMM landmarks and expression parameters
    
    
    # Create DAG and assign edge weights from distance matrix
    G = nx.DiGraph()
    for i in range(numFramesKuro - 1):
        left = np.arange(i*k, (i+1)*k)
        right = np.arange((i+1)*k, (i+2)*k)
        G.add_nodes_from(left)
        G.add_nodes_from(right)
        G.add_edges_from((u, v) for u in left for v in right)
    
    # Use Dijkstra shortest path 