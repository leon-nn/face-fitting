#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:02:36 2017

@author: leon
"""
from vol2mesh import Bunch, generateFace, exportObj
import os
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

if __name__ == "__main__":

    os.chdir('/home/leon/f2f-fitting/obama/')
    numFrames = 2882 #3744 #2260
#    fps = 24
#    
#    # Load audio tracks, pre-emphasize, and create feature vectors from mfcc, rmse, and deltas of mfcc
#    nfft = 1024
#    hopSamples = 512
#    
#    wav_kuro, fs_kuro = librosa.load('kuro.wav', sr=44100)
#    wav_kuro = np.r_[wav_kuro[0], wav_kuro[1:] - 0.97 * wav_kuro[:-1]]
#    mfcc_kuro = librosa.feature.mfcc(y = wav_kuro, sr = fs_kuro, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
#    mfcc_kuro[0, :] = librosa.feature.rmse(y = wav_kuro, n_fft = nfft, hop_length = hopSamples)
#    delta_kuro = librosa.feature.delta(mfcc_kuro)
#    mfcc_kuro = np.r_[mfcc_kuro, delta_kuro]
#    
#    wav_siro, fs_siro = librosa.load('siro.wav', sr=44100)
#    wav_siro = np.r_[wav_siro[0], wav_siro[1:] - 0.97 * wav_siro[:-1]]
#    mfcc_siro = librosa.feature.mfcc(y = wav_siro, sr = fs_siro, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
#    mfcc_siro[0, :] = librosa.feature.rmse(y = wav_siro, n_fft = nfft, hop_length = hopSamples)
#    delta_siro = librosa.feature.delta(mfcc_siro)
#    mfcc_siro = np.r_[mfcc_siro, delta_siro]
#    
#    # Find mfccs that are nearest to video frames in time
#    t_video = np.linspace(0, numFrames / fps, numFrames)
#    
#    t_audio_siro = np.linspace(0, mfcc_siro.shape[1] * hopSamples / fs_siro, mfcc_siro.shape[1])
#    t_audio_kuro = np.linspace(0, mfcc_kuro.shape[1] * hopSamples / fs_kuro, mfcc_kuro.shape[1])
#    
#    NN = NearestNeighbors(n_neighbors = 1, metric = 'l1')
#    
#    NN.fit(t_audio_siro.reshape(-1, 1))
#    distance, ind = NN.kneighbors(t_video.reshape(-1, 1))
#    mfcc_siro_sampled = mfcc_siro[:, ind.squeeze()]
#    
#    NN.fit(t_audio_kuro.reshape(-1, 1))
#    distance, ind = NN.kneighbors(t_video[:2041].reshape(-1, 1))
#    mfcc_kuro_sampled = mfcc_kuro[:, ind.squeeze()]
#    
#    # Cluster mfccs. Use 40 clusters -- 39 clusterable phonemes in American English
#    M = 40
#    #X = np.c_[mfcc_siro, mfcc_kuro].T
#    X = mfcc_siro.T
#    gmmObs = GaussianMixture(n_components = M, covariance_type = 'diag')
#    gmmObs.fit(X)
#    mfcc_classes = gmmObs.means_
#    
#    obsLabels_siro = gmmObs.predict(mfcc_siro_sampled.T)
#    obsLabels_kuro = gmmObs.predict(mfcc_kuro_sampled.T)
#    
#    # Find and cluster the features of the video in model-space
#    m = Bunch(np.load('../models/bfm2017.npz'))
#    m.idEvec = m.idEvec[:, :, :80]
#    m.idEval = m.idEval[:80]
#    m.expEvec = m.expEvec[:, :, :76]
#    m.expEval = m.expEval[:76]
#    m.texEvec = m.texEvec[:, :, :80]
#    m.texEval = m.texEval[:80]
#    
#    param = np.load('paramRTS2Orig.npy')
#    #for frame in np.arange(1, numFrames + 1):
#    #    shape = generateFace(np.r_[param[frame, :-7], np.zeros(6), 1], m)
#    
#    #tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1))
#    #view = mlab.view()
#    
#    N = 150
#    X = param[:, 80: -7]
#    kShapes = KMeans(n_clusters = N)
#    kShapes.fit(X)
#    
#    stateShapes = kShapes.cluster_centers_
#    stateLabels = kShapes.labels_
#    
#    # Calculate initial state probabilities for states
#    states, stateCounts = np.unique(stateLabels, return_counts = True)
#    pi = stateCounts / stateLabels.size
#    
#    # Calculate transition probabilities using known clusters
#    transition, transitionCounts = np.unique(np.c_[stateLabels[:-1], stateLabels[1:]], return_counts = True, axis = 0)
#    A = np.zeros((N, N))
#    A[transition[:, 0], transition[:, 1]] = transitionCounts
#    A /= A.sum(1)[:, np.newaxis]
#    
#    # Calculate emission probabilities using known clusters
#    B = np.zeros((N, M))
#    for state in range(N):
#        obsClass, classCount = np.unique(obsLabels_siro[stateLabels == state], return_counts = True)
#        B[state, obsClass] = classCount
#    B /= B.sum(1)[:, np.newaxis]
#    
#    # HMM stuff
#    model = hmm.MultinomialHMM(n_components = N)
#    model.startprob_ = pi
#    model.transmat_ = A
#    model.emissionprob_ = B
#    
#    # Try to reproduce siro
#    stateSeq_siro = model.predict(obsLabels_siro.reshape(-1, 1))
#    
#    # Kuro
#    stateSeq_kuro = model.predict(obsLabels_kuro.reshape(-1, 1))
#    
#    np.save('siroGroundTruth', stateLabels)
#    np.save('siroStateSequence', stateSeq_siro)
#    np.save('kuroStateSequence', stateSeq_kuro)
#    np.save('shapeStateParams', stateShapes)
#    
#    # Render and save pics
#    if not os.path.exists('stateShapes'):
#        os.makedirs('stateShapes')
#    for shape in range(N):
#        fName = '{:0>5}'.format(shape + 1)
#        exportObj(generateFace(np.r_[param[-1, :80], stateShapes[stateSeq_siro[shape], :], np.zeros(6), 1], m), f = m.face, fNameOut = 'stateShapes/' + fName)
#    
#    selectedFrames = np.zeros(stateSeq_kuro.size, dtype = int)
#    scaler = StandardScaler()
#    normalizedRTS = scaler.fit_transform(param[:, -7:])
#    for i in range(stateSeq_kuro.size):
#        # Find the video frames that match to the current shape state
#        frames = np.argwhere(stateLabels == stateSeq_kuro[i]).squeeze()
#        
#        # From these frames, find the frame that is closest to the i-th video frame in terms of rotation, translation, and scale
#        candidateFramesRTS = normalizedRTS[frames, :]
#        currentFrameRTS = normalizedRTS[i, :]
#        
#        NN = NearestNeighbors(n_neighbors = 1, metric = 'l2')
#        
#        NN.fit(candidateFramesRTS)
#        distance, ind = NN.kneighbors(currentFrameRTS.reshape(1, -1))
#        
#        selectedFrames[i] = frames[ind.squeeze()]
#    np.save('kuroSelectedFrames', selectedFrames)
    
    """
    2nd HMM for temporal consistency
    """
    
    N = 150
    stateLabels = np.load('siroGroundTruth.npy')
    stateSeq_siro = np.load('siroStateSequence.npy')
    stateSeq_kuro = np.load('kuroStateSequence.npy')
    
    
    
    # Calculate pairwise difference between the frames as transition probabilities
#    videoVec = np.empty((numFrames, mpimg.imread('orig/00001.png').size//3))
#    for i in range(numFrames):
#        fName = '{:0>5}'.format(i + 1)
#        videoVec[i, :] = io.imread('orig/' + fName + '.png', as_grey = True).flatten()
        
#    frameDifferences = metrics.pairwise.euclidean_distances(videoVec)
    frameDifferences = np.load('frameDistanceMat.npy')
    A = np.exp(-0.1*frameDifferences)
    A /= A.sum(1)[:, np.newaxis]
    
    # Find the frames that match to each clustered 3DMM state and set a uniform PDF for these frames as the emission probabilities
#    state2frame = [None] * N
    B = np.zeros((numFrames, N))
#    for i in range(N):
#        state2frame[i] = np.nonzero(stateLabels == i)[0].tolist()
#        B[state2frame[i], i] = 1. / len(state2frame[i])
    
    B[np.arange(numFrames), stateLabels] = 1
    
    B = np.ones((numFrames, N)) / N
    
    # Use a uniform PDF over all frames as the initial distribution
    pi = np.ones(numFrames) / numFrames
    
    # Set up the HMM
    model = hmm.MultinomialHMM(n_components = numFrames)
    model.startprob_ = pi
    model.transmat_ = A
    model.emissionprob_ = B
    
    
    frames = model.predict(stateLabels.reshape(-1, 1))
