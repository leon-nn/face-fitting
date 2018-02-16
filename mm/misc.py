#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:43:14 2018

@author: leon
"""
import numpy as np

def getTexture(dirName, pose = 0):
    """
    Get the RGB values referenced by the texture vertices in the .obj file for each tester. There are 20 poses to choose from from 0 to 19.
    """
    # Import the texture vertices
    textV = importObj(dirName, pose, dataToImport = ['vt'], pose = 20)[0]
    textV[:, 0] *= 639  # Multiply by the max column index
    textV[:, 1] *= 479  # Multiply by the max row index
    
    # Generate array of row and column indices for interpolation function
    r = np.arange(0, 480)   # Array of row indices
    c = np.arange(0, 640)   # Array of column indices
    
    numTesters = 150
    
    # Initialize array to store RGB values of each texture vertex for all testers
    interpRGB = np.empty((numTesters, 11558, 3))
    
    for i in range(numTesters):
        # Load the RGB image of the pose
        fName = dirName + 'Tester_' + str(i + 1) + '/TrainingPose/pose_' + str(pose) + '.png'
        img = mpimg.imread(fName)
        
        # Do linear interpolation to find the RGB values at the texture vertices given the image data
        interpRGB[i, :, :] = np.c_[intp.interpn((r, c), img[:, :, 0], textV[:, ::-1]), intp.interpn((r, c), img[:, :, 1], textV[:, ::-1]), intp.interpn((r, c), img[:, :, 2], textV[:, ::-1])]
    
    return interpRGB

def gaussNewton(P, m, target, targetLandmarks, sourceLandmarkInds, NN, jacobi = True, calcId = True):
    """
    Energy function to be minimized for fitting.
    """
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
#    start = clock()
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
#    print('NN: %f' % (clock() - start))
    
    # Calculate resisduals
    rVert = targetNN - source
    rLand = targetLandmarks - source[:, sourceLandmarkInds]
    rAlpha = idCoef ** 2 / m.idEval
    rDelta = expCoef ** 2 / m.expEval
    
    # Calculate costs
    Ever = np.linalg.norm(rVert, axis = 0).sum() / m.numVertices
    Elan = np.linalg.norm(rLand, axis = 0).sum() / sourceLandmarkInds.size
    Ereg = np.sum(rAlpha) + np.sum(rDelta)
    
    if jacobi:
#        start = clock()
        
        drV_dalpha = -s*np.tensordot(R, m.idEvec, axes = 1)
        drV_ddelta = -s*np.tensordot(R, m.expEvec, axes = 1)
        drV_dpsi = -s*np.dot(dR_dpsi(angles), model)
        drV_dtheta = -s*np.dot(dR_dtheta(angles), model)
        drV_dphi = -s*np.dot(dR_dphi(angles), model)
        drV_dt = -np.tile(np.eye(3), [source.shape[1], 1])
        drV_ds = -np.dot(R, model)
        
        drR_dalpha = np.diag(2*idCoef / m.idEval)
        drR_ddelta = np.diag(2*expCoef / m.expEval)
        
        # Calculate Jacobian
        if calcId:
            
            r = np.r_[rVert.flatten('F'), rLand.flatten('F'), rAlpha, rDelta]
        
            J = np.r_[np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')], np.c_[drV_dalpha[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, idCoef.size), order = 'F'), drV_ddelta[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')], np.c_[drR_dalpha, np.zeros((idCoef.size, expCoef.size + 7))], np.c_[np.zeros((expCoef.size, idCoef.size)), drR_ddelta, np.zeros((expCoef.size, 7))]]
            
            # Parameter update (Gauss-Newton)
            dP = -np.linalg.inv(np.dot(J.T, J)).dot(J.T).dot(r)
        
        else:
            
            r = np.r_[rVert.flatten('F'), rLand.flatten('F'), rDelta]
            
            J = np.r_[np.c_[drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')], np.c_[drV_ddelta[:, sourceLandmarkInds, :].reshape((np.prod(targetLandmarks.shape), expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')], np.c_[drR_ddelta, np.zeros((expCoef.size, 7))]]
            
            # Parameter update (Gauss-Newton)
            dP = np.r_[np.zeros(m.idEval.size), -np.linalg.inv(np.dot(J.T, J)).dot(J.T).dot(r)]
        
#        print('GN: %f' % (clock() - start))
        
        return Ever + Elan + Ereg, dP
    
    return Ever + Elan + Ereg

def shBasis(alb, n):
    """
    SH basis functions                               lm
        1/np.sqrt(4*np.pi)                          Y00
        np.sqrt(3/(4*np.pi))*nz                     Y10
        np.sqrt(3/(4*np.pi))*nx                     Y11e
        np.sqrt(3/(4*np.pi))*ny                     Y11o
        1/2*np.sqrt(5/(4*np.pi))*(3*nz^2 - 1)       Y20
        3*np.sqrt(5/(12*np.pi))*nx*nz               Y21e
        3*np.sqrt(5/(12*np.pi))*ny*nz               Y21o
        3/2*np.sqrt(5/(12*np.pi))*(nx^2 - ny^2)     Y22e
        3*np.sqrt(5/(12*np.pi))*nx*ny               Y22o
    
    For a sphere, the Lambertian kernel has most of its energy in the first three bands of the spherical harmonic basis functions (above). This implies that Lambertian reflectance functions can be well-approximated by these low-order SH bases.
    """
    
    # Nine delta function locations (el, az) for point light sources to create positive lighting
    lsph = np.array([[0, 0], [68, -90], [74, 108], [80, 52], [85, -42], [85, -137], [85, 146], [85, -4], [51, 67]]) * np.pi / 180
#    lsph = np.array([[0, 0], [49, 17], [-68, 0], [73, -18], [77, 37], [-84, 47], [-84, -47], [82, -56], [-50, -84]]) * np.pi / 180
    
    # Transform to Cartesian coordinates
    lx, ly, lz = sph2cart(lsph[:, 0], lsph[:, 1])
    
    # Evaluate spherical harmonics at these point light source locations
    H = sh9(lx, ly, lz)
    
    # Evaluate spherical harmonics at face shape normals
    B = sh9(n[:, 0], n[:, 1], n[:, 2])
    
    I = np.empty((alb.shape[0], 9, alb.shape[1]))
    for c in range(alb.shape[0]):
        I[c, :, :] = np.dot(H.T, B * alb[c, :])
    
#    b = np.empty((alb.shape[0], alb.shape[1], 9))
#    b[:, :, 0] = np.pi * 1/np.sqrt(4*np.pi) * alb
#    b[:, :, 1] = 2*np.pi/3 * np.sqrt(3/(4*np.pi)) * n[:, 2] * alb
#    b[:, :, 2] = 2*np.pi/3 * np.sqrt(3/(4*np.pi)) * n[:, 0] * alb
#    b[:, :, 3] = 2*np.pi/3 * np.sqrt(3/(4*np.pi)) * n[:, 1] * alb
#    b[:, :, 4] = np.pi/4 * 1/2*np.sqrt(5/(4*np.pi)) * (3*np.square(n[:, 2]) - 1) * alb
#    b[:, :, 5] = np.pi/4 * 3*np.sqrt(5/(12*np.pi)) * n[:, 0] * n[:, 2] * alb
#    b[:, :, 6] = np.pi/4 * 3*np.sqrt(5/(12*np.pi)) * n[:, 1] * n[:, 2] * alb
#    b[:, :, 7] = np.pi/4 * 3/2*np.sqrt(5/(12*np.pi)) * (np.square(n[:, 0]) - np.square(n[:, 1])) * alb
#    b[:, :, 8] = np.pi/4 * 3*np.sqrt(5/(12*np.pi)) * n[:, 0] * n[:, 1] * alb
    
    return I

