#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:41:03 2018

@author: leon
"""

import numpy as np



"""
For vol2mesh (optimization functions for fitting VRN depth map to image)
"""

def initialShapeCost(P, target, m, sourceLandmarkInds, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Landmark fitting cost
    source = generateFace(P, m, ind = sourceLandmarkInds)
    
    rlan = (source - target.T).flatten('F')
    Elan = np.dot(rlan, rlan) / sourceLandmarkInds.size
    
    # Regularization cost
    Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    return w[0] * Elan + w[1] * Ereg

def initialShapeGrad(P, target, m, sourceLandmarkInds, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = P[m.idEval.size + m.expEval.size:][:3]
    R = rotMat2angle(angles)
    t = P[m.idEval.size + m.expEval.size:][3: 6]
    s = P[m.idEval.size + m.expEval.size:][6]
    
    # The eigenmodel, before rigid transformation and scaling
    model = m.idMean[:, sourceLandmarkInds] + np.tensordot(m.idEvec[:, sourceLandmarkInds, :], idCoef, axes = 1) + np.tensordot(m.expEvec[:, sourceLandmarkInds, :], expCoef, axes = 1)
    
    # After rigid transformation and scaling
    source = s*np.dot(R, model) + t[:, np.newaxis]
    
    rlan = (source - target.T).flatten('F')
        
    drV_dalpha = s*np.tensordot(R, m.idEvec[:, sourceLandmarkInds, :], axes = 1)
    drV_ddelta = s*np.tensordot(R, m.expEvec[:, sourceLandmarkInds, :], axes = 1)
    drV_dpsi = s*np.dot(dR_dpsi(angles), model)
    drV_dtheta = s*np.dot(dR_dtheta(angles), model)
    drV_dphi = s*np.dot(dR_dphi(angles), model)
    drV_dt = np.tile(np.eye(3), [sourceLandmarkInds.size, 1])
    drV_ds = np.dot(R, model)
    
    Jlan = np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
    
    return 2 * (w[0] * np.dot(Jlan.T, rlan) / sourceLandmarkInds.size + w[1] * np.r_[idCoef / m.idEval, expCoef / m.expEval, np.zeros(7)])

def shapeCost(P, m, target, targetLandmarks, sourceLandmarkInds, NN, w = (1, 1, 1), calcID = True):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Transpose target if necessary
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # After rigid transformation and scaling
    source = generateFace(P, m)
    
    # Find the nearest neighbors of the target to the source vertices
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
    
    # Calculate resisduals
    rver = (source - targetNN).flatten('F')
    rlan = (source[:, sourceLandmarkInds] - targetLandmarks).flatten('F')
    
    # Calculate costs
    Ever = np.dot(rver, rver) / m.numVertices
    Elan = np.dot(rlan, rlan) / sourceLandmarkInds.size
    
    if calcID:
        
        Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    else:
        
        Ereg = np.sum(expCoef ** 2 / m.expEval)
    
    return w[0] * Ever + w[1] * Elan + w[2] * Ereg

def shapeGrad(P, m, target, targetLandmarks, sourceLandmarkInds, NN, w = (1, 1, 1), calcID = True):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = P[m.idEval.size + m.expEval.size:][:3]
    R = rotMat2angle(angles)
    t = P[m.idEval.size + m.expEval.size:][3: 6]
    s = P[m.idEval.size + m.expEval.size:][6]
    
    # Transpose if necessary
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # The eigenmodel, before rigid transformation and scaling
    model = m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)
    
    # After rigid transformation and scaling
    source = s*np.dot(R, model) + t[:, np.newaxis]
    
    # Find the nearest neighbors of the target to the source vertices
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
    
    # Calculate resisduals
    rver = (source - targetNN).flatten('F')
    rlan = (source[:, sourceLandmarkInds] - targetLandmarks).flatten('F')
        
    drV_ddelta = s*np.tensordot(R, m.expEvec, axes = 1)
    drV_dpsi = s*np.dot(dR_dpsi(angles), model)
    drV_dtheta = s*np.dot(dR_dtheta(angles), model)
    drV_dphi = s*np.dot(dR_dphi(angles), model)
    drV_dt = np.tile(np.eye(3), [m.numVertices, 1])
    drV_ds = np.dot(R, model)
    
    if calcID:
        
        drV_dalpha = s*np.tensordot(R, m.idEvec, axes = 1)
        
        Jver = np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
        
        Jlan = np.c_[drV_dalpha[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, idCoef.size), order = 'F'), drV_ddelta[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')]
        
        return 2 * (w[0] * np.dot(Jver.T, rver) / m.numVertices + w[1] * np.dot(Jlan.T, rlan) / sourceLandmarkInds.size + w[2] * np.r_[idCoef / m.idEval, expCoef / m.expEval, np.zeros(7)])
    
    else:
        
        Jver = np.c_[drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
        
        Jlan = np.c_[drV_ddelta[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')]
        
        return 2 * (np.r_[np.zeros(idCoef.size), w[0] * np.dot(Jver.T, rver) / m.numVertices] + np.r_[np.zeros(idCoef.size), w[1] * np.dot(Jlan.T, rlan) / sourceLandmarkInds.size] + w[2] * np.r_[np.zeros(idCoef.size), expCoef / m.expEval, np.zeros(7)])