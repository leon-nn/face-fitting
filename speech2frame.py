#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:07:52 2018

@author: leon
"""

from mm import Bunch, generateFace, importObj, rotMat2angle
import glob, os, json
import numpy as np
import librosa
from sklearn.neighbors import NearestNeighbors
from mayavi import mlab
from sklearn.preprocessing import StandardScaler
import networkx as nx

def animate(v, f, saveDir, t = None, alpha = 1):
    # Create the save directory for the images if it doesn't exist
    if not saveDir.endswith('/'):
        saveDir += '/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    # Render the mesh
    if t is None:
        tmesh = mlab.triangular_mesh(v[0, 0, :], v[0, 1, :], v[0, 2, :], f-1, scalars = np.arange(v.shape[2]), color = (1, 1, 1))
    
    # Add texture if given
    else:
        tmesh = mlab.triangular_mesh(v[0, 0, :], v[0, 1, :], v[0, 2, :], f-1, scalars = np.arange(v.shape[2]))
        if t.shape[1] is not 3:
            t = t.T
        tmesh.module_manager.scalar_lut_manager.lut.table = np.c_[(t * 255), alpha * 255 * np.ones(v.shape[2])].astype(np.uint8)
#        tmesh.actor.pro2perty.lighting = False
        
    # Change viewport to x-y plane and enforce orthographic projection
    mlab.view(0, 0, 'auto', 'auto')
    
    mlab.gcf().scene.parallel_projection = True
    
    # Save the first frame, then loop through the rest and save them
    mlab.savefig(saveDir + '00001.png', figure = mlab.gcf())
    tms = tmesh.mlab_source
    for i in range(1, v.shape[0]):
        fName = '{:0>5}'.format(i + 1)
        tms.set(x = v[i, 0, :], y = v[i, 1, :], z = v[i, 2, :])
        mlab.savefig(saveDir + fName + '.png', figure = mlab.gcf())
        
def speechProc(fName, numFrames, fps, kuro = False, return_time_vec = False):
    # Load audio tracks, pre-emphasize, and create feature vectors from mfcc, rmse, and deltas of mfcc
    nfft = 1024
    hopSamples = 512
    
    wav, fs = librosa.load(fName, sr=44100)
    wav = np.r_[wav[0], wav[1:] - 0.97 * wav[:-1]]
    mfcc = librosa.feature.mfcc(y = wav, sr = fs, n_mfcc = 13, n_fft = nfft, hop_length = hopSamples)
    mfcc[0, :] = librosa.feature.rmse(y = wav, n_fft = nfft, hop_length = hopSamples)
    mfccDelta = librosa.feature.delta(mfcc)
    audioFeatures = np.r_[mfcc, mfccDelta]
    
    # Find audio features that are nearest to video frames in time
    timeVecVideo = np.linspace(0, numFrames / fps, numFrames)
    
    timeVecAudioFeatures = np.linspace(0, audioFeatures.shape[1] * hopSamples / fs, audioFeatures.shape[1])
    
    NN = NearestNeighbors(n_neighbors = 1, metric = 'l1')
    NN.fit(timeVecAudioFeatures.reshape(-1, 1))
    
    if kuro:
        numFrames = np.ceil(audioFeatures.shape[1] * hopSamples / fs * fps).astype(np.int_)
        distance, ind = NN.kneighbors(timeVecVideo[:numFrames].reshape(-1, 1))
    else:
        distance, ind = NN.kneighbors(timeVecVideo.reshape(-1, 1))
    
    audioFeaturesSampled = audioFeatures[:, ind.squeeze()]
    
    if return_time_vec:
        return audioFeaturesSampled, timeVecVideo
    else:
        return audioFeaturesSampled

if __name__ == "__main__":

    os.chdir('/home/leon/f2f-fitting/obama/')
    fNameSiro = 'siroNorm.wav'
    numFramesSiro = 2882 #3744 #2260
    fpsSiro = 24
    
    siroAudioVec, timeVecVideo = speechProc(fNameSiro, numFramesSiro, fpsSiro, return_time_vec = True)
    
    # Create a kNN fitter to find the k closest siro audio features
    k = 20
    NN = NearestNeighbors(n_neighbors = k, metric = 'l2')
    NN.fit(siroAudioVec.T)
    
    # Initialize
    # Load 3DMM
    m = Bunch(np.load('../models/bfm2017.npz'))
    m.idEvec = m.idEvec[:, :, :80]
    m.idEval = m.idEval[:80]
    m.expEvec = m.expEvec[:, :, :76]
    m.expEval = m.expEval[:76]
    
    # Load 3DMM parameters for the siro video, scaling some for a distance measure
    scaler = StandardScaler()
    param = np.load('paramRTS2Orig.npy')
    expCoef = scaler.fit_transform(param[:, m.idEval.size: m.idEval.size + m.expEval.size])
    angles = param[:, m.idEval.size + m.expEval.size: m.idEval.size + m.expEval.size + 3]
    trans = scaler.fit_transform(param[:, m.idEval.size + m.expEval.size + 3: m.idEval.size + m.expEval.size + 5])
    R = np.empty((numFramesSiro, 3, 3))
    for i in range(numFramesSiro):
        R[i, ...] = rotMat2angle(angles[i, :])
    
    # Load OpenPose 2D landmarks for the siro video
    lmPairs = np.array([[42, 47], [43, 46], [44, 45], [30, 36], [42, 45], [44, 47], [25, 29], [26, 28], [19, 23], [20, 22]])
    lm = np.empty((numFramesSiro, 70, 2))
    for i in range(numFramesSiro):
        with open('landmark/' + '{:0>5}'.format(i+1) + '.json', 'r') as fd:
            lm[i, ...] = np.array([l[0] for l in json.load(fd)], dtype = int).squeeze()[:, :2]
            
    # Get corresponding landmark locations on 3DMM
    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
    sourceLmPairs = sourceLandmarkInds[lmPairs]
    uniqueSourceLm, uniqueInv = np.unique(sourceLmPairs, return_inverse = True)
    
    # Load mouth region from 3DMM for animation
    mouthIdx = np.load('../bfmMouthIdx.npy')
    mouthVertices = np.load('mouthVertices.npy')
    mouthFace = importObj('mouth.obj', dataToImport = ['f'])[0]
    
    # Loop through each siro target audio file
    for fNameKuro in glob.glob('condition_enhanced/cleaned/*.wav'):
        fNameKuro = 'condition_enhanced/cleaned/7_EJF101_ESPBOBAMA1_00101_V01_T01.wav'
        kuroAudioVec = speechProc(fNameKuro, numFramesSiro, fpsSiro, kuro = True)
        numFramesKuro = kuroAudioVec.shape[1]
        
        distance, ind = NN.kneighbors(kuroAudioVec.T)
        
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
                
    #            np.exp(-np.fabs(timeVecVideo[ind[t, c1]] - timeVecVideo[ind[t+1, :]])**2)
                
                weights[t, c1] = Dm[t, c1] + Dp[t, c1] + Dp[t+1, :] + distance[t, c1] + distance[t+1, :]
        
        # Create DAG and assign edge weights from distance matrix
        G = nx.DiGraph()
        for i in range(numFramesKuro - 1):
            left = np.arange(i*k, (i+1)*k)
            right = np.arange((i+1)*k, (i+2)*k)
            G.add_nodes_from(left)
            G.add_nodes_from(right)
            G.add_weighted_edges_from((u, v, weights[i, u - i*k, v - (i+1)*k]) for u in left for v in right)
        
        # Use A* shortest path algorithm to find the distances from each of the k source nodes to each of the k terminal nodes
        astarLength = np.empty((k, k))
        for s in range(k):
            for t in range(k):
                astarLength[s, t] = nx.astar_path_length(G, s, right[t])
        
        # Find the optimal path with the minimum distance of the k^2 paths calculated above
        s, t = np.unravel_index(astarLength.argmin(), (k, k))
        optPath = nx.astar_path(G, s, right[t])
        optPath = np.unravel_index(optPath, (numFramesKuro, k))
        optPath = ind[optPath[0], optPath[1]]
        audioMinDistancePath = ind[:, 0]
        break
        if not os.path.exists('graphOptPath'):
            os.makedirs('graphOptPath')
        if not os.path.exists('minDistancePath'):
            os.makedirs('minDistancePath')
        np.save('graphOptPath/' + os.path.splitext(os.path.basename(fNameKuro))[0], optPath)
        np.save('minDistancePath/' + os.path.splitext(os.path.basename(fNameKuro))[0], audioMinDistancePath)
        
        # Animate
        v = mouthVertices.reshape((numFramesSiro, 3, mouthIdx.size), order = 'F')
        animate(v[optPath], mouthFace, 'graphAnimate/' + os.path.splitext(os.path.basename(fNameKuro))[0], m.texMean[:, mouthIdx])
        mlab.close(all = True)
        animate(v[audioMinDistancePath], mouthFace, 'minDistanceAnimate/' + os.path.splitext(os.path.basename(fNameKuro))[0], m.texMean[:, mouthIdx])
        mlab.close(all = True)