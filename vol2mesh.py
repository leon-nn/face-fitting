#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:52:46 2017

@author: leon
"""

from time import clock
import glob, os, re, json
import numpy as np
from scipy.interpolate import interpn
#from scipy.optimize import minimize
from scipy.optimize import check_grad
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mayavi import mlab
#import visvis as vv
from pylab import savefig

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def onpick3(event):
    """
    Interactive clicking for pyplot scatter plots.
    Example:
    fig, ax = plt.subplots()
    x = ...
    y = ...
    col = ax.scatter(x, y, s = 1, picker = True)
    fig.canvas.mpl_connect('pick_event', onpick3)
    """
    ind = event.ind
    print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))

def exportObj(v, c = None, vt = None, f = None, fNameIn = None, fNameOut = 'test.obj'):
    """
    Write vertices to an .obj file.
    """
    # Make sure x, y, z vertex coordinates are along the columns
    if v.shape[1] != 3:
        v = v.T
    
    # Add the .obj extension if necessary
    if not fNameOut.endswith('.obj'):
        fNameOut += '.obj'
    
    # If user provides an .obj file for the face specification
    if fNameIn is not None:
        if not fNameIn.endswith('.obj'):
            fNameIn += '.obj'
            
        with open(fNameIn, 'r') as fi:
            with open(fNameOut, 'w') as fo:
                # Initialize counter for the vertex index
                vertexInd = 0
                
                # Iterate through the lines in the template file
                for line in fi:
                    # If a line starts with 'v ', then we replace it with the vertex coordinate with the index corresponding to the line number
                    if line.startswith('v '):
                        # Keep 6 digits after the decimal to follow the formatting in the template file
                        fo.write('v ' + '{:.6f}'.format(v[vertexInd, 0]) + ' ' + '{:.6f}'.format(v[vertexInd, 1]) + ' ' + '{:.6f}'.format(v[vertexInd, 2]) + '\n')
                        
                        # Increment the vertex index counter
                        vertexInd += 1
                    
                    elif line.startswith('vn'):
                        continue
                    
                    # Else, the line must end with 'vt' or 'f', so we copy from the template file because those coordinates are in correspondence
                    elif line.startswith('vt'):
                        fo.write(line)
                    
                    elif line.startswith('f'):
                        f = [int(ind) for ind in re.split('/| ', line[2:])]
                        fo.write('f ' + str(f[0]) + '/' + str(f[1]) + '/0 ' + str(f[3]) + '/' + str(f[4]) + '/0 ' + str(f[6]) + '/' + str(f[7]) + '/0 ' + str(f[9]) + '/' + str(f[10]) + '/0\n')
                    
                    else:
                        continue
    # If user provides a NumPy array for the face specification, or doesn't provide one at all (just to write out the vertices)
    else:
        with open(fNameOut, 'w') as fo:
            if c is None:
                for vertex in v:
                    fo.write('v ' + '{:.6f}'.format(vertex[0]) + ' ' + '{:.6f}'.format(vertex[1]) + ' ' + '{:.6f}'.format(vertex[2]) + '\n')
            else:
                if c.shape[1] != 3:
                    c = c.T
                for i in range(v.shape[0]):
                    fo.write('v ' + '{:.6f}'.format(v[i, 0]) + ' ' + '{:.6f}'.format(v[i, 1]) + ' ' + '{:.6f}'.format(v[i, 2]) + ' ' + '{:.6f}'.format(c[i, 0]) + ' ' + '{:.6f}'.format(c[i, 1]) + ' ' + '{:.6f}'.format(c[i, 2]) + '\n')
            
            if vt is not None:
                for text in vt:
                    fo.write('vt ' + '{:.6f}'.format(text[0]) + ' ' + '{:.6f}'.format(text[1]) + '\n')
            
            if f is not None:
                # obj face indexing starts at 1
                if np.min(f) == 0:
                    f = f + 1
                if f.shape[1] == 4:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n')
                elif f.shape[1] == 3:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')

def generateFace(P, m, ind = None):
    """
    Generate vertices based off of eigenmodel and vector of parameters
    """
    # Shape eigenvector coefficients
    idCoef = P[: m.idEvec.shape[2]]
    expCoef = P[m.idEvec.shape[2]: m.idEvec.shape[2] + m.expEvec.shape[2]]
    
    # Rotation Euler angles, translation vector, scaling factor
    R = rotMat2angle(P[m.idEvec.shape[2] + m.expEvec.shape[2]:][:3])
    t = P[m.idEvec.shape[2] + m.expEvec.shape[2]:][3: 6]
    s = P[m.idEvec.shape[2] + m.expEvec.shape[2]:][6]
    
    # The eigenmodel, before rigid transformation and scaling
    if ind is None:
        model = m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)
    else:
        model = m.idMean[:, ind] + np.tensordot(m.idEvec[:, ind, :], idCoef, axes = 1) + np.tensordot(m.expEvec[:, ind, :], expCoef, axes = 1)
    
    # After rigid transformation and scaling
    return s*np.dot(R, model) + t[:, np.newaxis]

def rotMat2angle(R):
    """
    Conversion between 3x3 rotation matrix and Euler angles psi, theta, and phi in radians (rotations about the x, y, and z axes, respectively). If the input is 3x3, then the output will return a size-3 array containing psi, theta, and phi. If the input is a size-3 array, then the output will return the 3x3 rotation matrix.
    """
    if R.shape == (3, 3):
        if abs(R[2, 0]) != 1:
            theta = -np.arcsin(R[2, 0])
            psi = np.arctan2(R[2, 1]/np.cos(theta), R[2, 2]/np.cos(theta))
            phi = np.arctan2(R[1, 0]/np.cos(theta), R[0, 0]/np.cos(theta))
        else:
            phi = 0
            if R[2, 0] == -1:
                theta = np.pi/2
                psi = np.arctan2(R[0, 1], R[0, 2])
            else:
                theta = -np.pi/2
                psi = np.arctan2(-R[0, 1], -R[0, 2])
        
        return np.array([psi, theta, phi])
    
    elif R.shape == (3,):
        psi, theta, phi = R
        Rx = np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)], [0, np.sin(psi), np.cos(psi)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        
        return np.dot(Rz, np.dot(Ry, Rx))

def initialRegistration(A, B):
    """
    Find the rotation matrix R, translation vector t, and scaling factor s to reconstruct the 3D vertices of the target B from the source A as B' = s*R*A.T + t.
    """
    
    # Make sure the x, y, z vertex coordinates are along the columns
    if A.shape[0] == 3:
        A = A.T
    if B.shape[0] == 3:
        B = B.T
    
    # Find centroids of A and B landmarks and move them to the origin
    muA = np.mean(A, axis = 0)
    muB = np.mean(B, axis = 0)
    A = A - muA
    B = B - muB
    
    # Calculate the rotation matrix R. Note that the returned V is actually V.T.
    U, V = np.linalg.svd(np.dot(A.T, B))[::2]
    R = np.dot(V.T, U.T)
    
    # Flip sign on the third column of R if it is a reflectance matrix
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        
    # Find scale factor
    s = np.trace(np.dot(B.T, np.dot(A, R.T))) / np.trace(np.dot(A.T, A))
    
    # Find the translation vector
    t = -s*np.dot(R, muA) + muB
    
    # Find Euler angles underlying rotation matrix
    angle = rotMat2angle(R)
    
    return np.r_[angle, t, s]

def initialShape(P, target, m, sourceLandmarkInds):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEvec.shape[2]]
    expCoef = P[m.idEvec.shape[2]: m.idEvec.shape[2] + m.expEvec.shape[2]]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = P[m.idEvec.shape[2] + m.expEvec.shape[2]:][:3]
    R = rotMat2angle(angles)
    t = P[m.idEvec.shape[2] + m.expEvec.shape[2]:][3: 6]
    s = P[m.idEvec.shape[2] + m.expEvec.shape[2]:][6]
    
    # Landmark fitting cost
    source = s*R.dot(m.idMean[:, sourceLandmarkInds] + np.tensordot(m.idEvec[:, sourceLandmarkInds, :], idCoef, axes = 1) + np.tensordot(m.expEvec[:, sourceLandmarkInds, :], expCoef, axes = 1)) + t[:, np.newaxis]
    
    Elan = np.linalg.norm(target - source.T, axis = 1).sum() / sourceLandmarkInds.size
    
    # Regularization cost
    Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    return Elan + Ereg

def dR_dpsi(angles):
    """
    Derivative of the rotation matrix with respect to the x-axis rotation.
    """
    psi, theta, phi = angles
    return np.array([[0, np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi), np.cos(psi)*np.sin(phi) - np.cos(psi)*np.sin(theta)*np.cos(phi)], [0, -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.cos(phi) - np.sin(psi)*np.sin(theta)*np.sin(phi)], [0, np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(theta)]])

def dR_dtheta(angles):
    """
    Derivative of the rotation matrix with respect to the y-axis rotation.
    """
    psi, theta, phi = angles
    return np.array([[-np.sin(theta)*np.cos(phi), np.sin(psi)*np.cos(theta)*np.cos(phi), np.cos(psi)*np.cos(theta)*np.cos(phi)], [-np.sin(theta)*np.sin(phi), np.sin(psi)*np.cos(theta)*np.sin(phi), np.cos(psi)*np.cos(theta)*np.sin(phi)], [-np.cos(theta), -np.sin(psi)*np.sin(theta), -np.cos(psi)*np.sin(theta)]])

def dR_dphi(angles):
    """
    Derivative of the rotation matrix with respect to the z-axis rotation.
    """
    psi, theta, phi = angles
    return np.array([[-np.cos(theta)*np.sin(phi), -np.cos(psi)*np.cos(phi) - np.sin(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.cos(phi) - np.cos(psi)*np.sin(theta)*np.sin(phi)], [np.cos(theta)*np.cos(phi), -np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi), np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)], [0, 0, 0]])

def shapeCost(P, m, target, targetLandmarks, sourceLandmarkInds, NN):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Rotation matrix, translation vector, scaling factor
    R = rotMat2angle(P[m.idEval.size + m.expEval.size:][:3])
    t = P[m.idEval.size + m.expEval.size:][3: 6]
    s = P[m.idEval.size + m.expEval.size:][6]
    
    # Transpose if necessary
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # After rigid transformation and scaling
    source = s*np.dot(R, m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)) + t[:, np.newaxis]
    
    # Find the nearest neighbors of the target to the source vertices
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
    
    # Calculate resisduals
    rver = (source - targetNN).flatten('F')
    rlan = (source[:, sourceLandmarkInds] - targetLandmarks).flatten('F')
    
    # Calculate costs
    Ever = np.dot(rver, rver) / m.numVertices
    Elan = np.dot(rlan, rlan) / sourceLandmarkInds.size
    Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    return Ever + Elan + Ereg

def shapeGrad(P, m, target, targetLandmarks, sourceLandmarkInds, NN):
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
        
    drV_dalpha = s*np.tensordot(R, m.idEvec, axes = 1)
    drV_ddelta = s*np.tensordot(R, m.expEvec, axes = 1)
    drV_dpsi = s*np.dot(dR_dpsi(angles), model)
    drV_dtheta = s*np.dot(dR_dtheta(angles), model)
    drV_dphi = s*np.dot(dR_dphi(angles), model)
    drV_dt = np.tile(np.eye(3), [m.numVertices, 1])
    drV_ds = np.dot(R, model)
    
    Jver = np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')]
    
    Jlan = np.c_[drV_dalpha[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, idCoef.size), order = 'F'), drV_ddelta[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')]
    
    return 2 * (np.dot(Jver.T, rver) / m.numVertices + np.dot(Jlan.T, rlan) / sourceLandmarkInds.size + np.r_[idCoef / m.idEval, expCoef / m.expEval, np.zeros(7)])
    
#    drR_dalpha = np.diag(2*idCoef / m.idEval)
#    drR_ddelta = np.diag(2*expCoef / m.expEval)
#            
#    J = np.r_[2 * np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')] / m.numVertices, 2 * np.c_[drV_dalpha[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, idCoef.size), order = 'F'), drV_ddelta[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')] / sourceLandmarkInds.size, np.c_[drR_dalpha, np.zeros((idCoef.size, expCoef.size + 7))], np.c_[np.zeros((expCoef.size, idCoef.size)), drR_ddelta, np.zeros((expCoef.size, 7))]]
#    
#    return J.T.dot(np.r_[rver, rlan, idCoef ** 2 / m.idEval, expCoef ** 2 / m.expEval])

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

if __name__ == "__main__":
    
    os.chdir('/home/leon/f2f-fitting/trump2/volume/')
    numFrames = 3744 #2260 #2882
    
    # Load 3DMM
    m = Bunch(np.load('../../models/bfm2017.npz'))
    m.idEvec = m.idEvec[:, :, :80]
    m.idEval = m.idEval[:80]
    m.expEvec = m.expEvec[:, :, :76]
    m.expEval = m.expEval[:76]
    m.texEvec = m.texEvec[:, :, :80]
    m.texEval = m.texEval[:80]
    
    targetLandmarkInds = np.array([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
    
    #initRegInds = np.array([0, 1, 2, 4, 6, 7, 8, 12, 18, 21, 24, 27, 30, 33, 36, 39])
    
    #fName = 'IMG_00041'
    #fNameImgScaled = '/home/leon/vrn/examples/scaled/' + fName + '.jpg'
    #fNameImgOrig = '/home/leon/vrn/examples/' + fName + '.jpg'
    #fNameVol = '/home/leon/vrn/output/' + fName + '.raw'
    #fNameLandmarks = '/home/leon/vrn/examples/' + fName + '.txt'
    
    #fNameImgScaled = 'kobayashi/scaled.jpg'
    #fNameImgOrig = 'kobayashi/orig.jpg'
    #fNameVol = 'kobayashi/volume.raw'
    #fNameLandmarks = 'kobayashi/landmarks.txt'
    
    #os.chdir('/home/leon/f2f-fitting/kao/volume/')
    #for file in glob.glob('*.raw'):
    #    fName = os.path.splitext(file)[0]
    #    fNameImgOrig = '../orig/' + fName + '.jpg'
    #    fNameVol = '../volume/' + fName + '.raw'
    #    fNameLandmarks = '../landmarks/' + fName + '.txt'
    #    
    #    '''
    #    Preprocess landmark locations: map to cropped/scaled version of image
    #    '''
    #    
    #    # Read the landmarks generated by VRN and the original JSON landmarks
    #    if fNameLandmarks.endswith('.txt'):
    #        with open(fNameLandmarks, 'r') as fd:
    #            lm = []
    #            for l in fd:
    #                lm.append([int(coord) for coord in l.split(',')])
    #        lm = np.array(lm)
    #    elif fNameLandmarks.endswith('.json'):
    #        with open(fNameLandmarks, 'r') as fd:
    #            lm = json.load(fd)
    #        lm = np.array([l[0] for l in lm], dtype = int).squeeze()[:, :3]
    #        lmConf = lm[:, -1]
    #        lm = lm[:, :2]
    #    
    #    # Map the landmarks from the original image to the scaled and cropped image
    #    imgOrig = mpimg.imread(fNameImgOrig)
    #    
    #    lmRange = np.ptp(lm, axis = 0)
    #    scale = 90 / np.sqrt(lmRange.prod())
    #    cropCorner = (np.min(lm, axis = 0) - lmRange / 2) * scale
    #    
    #    scaledImgDim = np.array(imgOrig.shape[1::-1]) * scale
    #    
    #    # Case 1: The cropped picture is contained within the scaled image
    #    if (cropCorner >= 0).all() and ((192 + cropCorner) < scaledImgDim).all():
    #        lmScaled = lm * scale - cropCorner
    #    
    #    # Case 2: The crop corner is outside of the scaled image, but the extent of the cropped picture is within the bounds of the scaled image
    #    elif (cropCorner < 0).any() and ((192 + cropCorner) < scaledImgDim).all():
    #        lmScaled = lm * scale - cropCorner * (cropCorner > 0) - cropCorner * (cropCorner < 0) / 2
    #    
    #    # Case 3: The crop corner is outside of the scaled image, and the extent of the cropped picture is beyond the bounds of the scaled image
    #    elif (cropCorner < 0).any() and ((192 + cropCorner) > scaledImgDim).any():
    #        lmScaled = lm * scale - cropCorner * (cropCorner > 0) + (192 - (scaledImgDim - cropCorner * (cropCorner > 0))) / 2
    #    
    #    vol = np.fromfile(fNameVol, dtype = np.int8)
    #    vol = vol.reshape((200, 192, 192))
    #    target = np.argmax(vol[::-1, :, :] > 0, axis = 0)
    #    target = target / 2
    #    
    #    np.save('../landmarks_scaled/' + fName, lmScaled)
    #    np.save('../depth/' + fName, target)
    
#    plt.ioff()
    param = np.zeros((numFrames, m.idEval.size + m.expEval.size + 7))
    TS2orig = np.zeros((numFrames, 4))
    
#    for frame, file in enumerate(glob.glob('*.raw')):
#        fName = os.path.splitext(file)[0]
#        print(fName)
#        frame -= 1
#        fNameImgOrig = '../orig/' + fName + '.png'
#        fNameVol = '../volume/' + fName + '.raw'
#        fNameLandmarks = '../landmark/' + fName + '.json'
    for frame in np.arange(1, numFrames + 1):
        print(frame)
    #    fName = '{:0>5}'.format(frame * 10)
        fName = '{:0>5}'.format(frame)
        fNameImgScaled = '../scaled/' + fName + '.png'
        fNameImgOrig = '../orig/' + fName + '.png'
        fNameVol = '../volume/' + fName + '.raw'
#        fNameLandmarks = 'obama/landmark/' + fName + '.txt'
        fNameLandmarks = '../landmark/' + fName + '.json'
        
        '''
        Preprocess landmark locations: map to cropped/scaled version of image
        '''
        
        # Read the landmarks generated by VRN and the original JSON landmarks
        if fNameLandmarks.endswith('.txt'):
            with open(fNameLandmarks, 'r') as fd:
                lm = []
                for l in fd:
                    lm.append([int(coord) for coord in l.split(',')])
            lm = np.array(lm)
        elif fNameLandmarks.endswith('.json'):
            with open(fNameLandmarks, 'r') as fd:
                lm = json.load(fd)
            lm = np.array([l[0] for l in lm], dtype = int).squeeze()[:, :3]
            lmConf = lm[:, -1]
            lm = lm[:, :2]
        
        # Map the landmarks from the original image to the scaled and cropped image
        imgOrig = mpimg.imread(fNameImgOrig)
#        plt.figure()
#        plt.imshow(imgOrig)
#        plt.scatter(lm[:, 0], lm[:, 1], s = 2)
#        plt.title(fName)
#        plt.close()
        
        lmRange = np.ptp(lm, axis = 0)
        scale = 90 / np.sqrt(lmRange.prod())
        cropCorner = (np.min(lm, axis = 0) - lmRange / 2) * scale
        
        scaledImgDim = np.array(imgOrig.shape[1::-1]) * scale
        
        # Case 1: The cropped picture is contained within the scaled image
        if (cropCorner >= 0).all() and ((192 + cropCorner) < scaledImgDim).all():
            lmScaled = lm * scale - cropCorner
        
        # Case 2: The crop corner is outside of the scaled image, but the extent of the cropped picture is within the bounds of the scaled image
        elif (cropCorner < 0).any() and ((192 + cropCorner) < scaledImgDim).all():
            lmScaled = lm * scale - cropCorner * (cropCorner > 0) - cropCorner * (cropCorner < 0) / 2
        
        # Case 3: The crop corner is outside of the scaled image, and the extent of the cropped picture is beyond the bounds of the scaled image
        elif (cropCorner < 0).any() and ((192 + cropCorner) > scaledImgDim).any():
            lmScaled = lm * scale - cropCorner * (cropCorner > 0) + (192 - (scaledImgDim - cropCorner * (cropCorner > 0))) / 2
        
        # Plot scaled landmarks
#        imgScaled = mpimg.imread(fNameImgScaled)
#        fig, ax = plt.subplots()
#        plt.imshow(imgScaled)
#        plt.hold(True)
#        x = lmScaled[:, 0]
#        y = lmScaled[:, 1]
#        ax.scatter(x, y, s = 2, c = 'b', picker = True)
#        fig.canvas.mpl_connect('pick_event', onpick3)
        
#        plt.figure()
#        plt.imshow(imgScaled)
#        plt.scatter(lmScaled[:, 0], lmScaled[:, 1], s = 2)
#        plt.title(fName)
#        if not os.path.exists('../landmarkPic'):
#            os.makedirs('../landmarkPic')
#        savefig('../landmarkPic/' + fName + '.png', bbox_inches='tight')
#        plt.close('all')
        
        '''
        Initial registration
        '''
        
        # Import volume and rescale the z-axis by 1/2
        vol = np.fromfile(fNameVol, dtype = np.int8)
        vol = vol.reshape((200, 192, 192))
        
        # Interpolate the max volume values at the landmarks
        depth = np.argmax(vol[::-1, :, :] > 0, axis = 0) / 2
        lmScaled = lmScaled[targetLandmarkInds, :]
        targetLandmarks = np.c_[lmScaled, interpn((np.arange(0, 192), np.arange(0, 192)), depth, lmScaled[:, ::-1], method = 'nearest')]
        nzd = targetLandmarks[:, 2] != 0
        targetLandmarks = targetLandmarks[nzd, :]
    
        # Initialize shape coefficients
        # Find initial guess of the rigid transformation (rotation, translation, scale) based off of the mean face of the 3DMM
        
        if frame == 1:
            rho = initialRegistration(m.idMean[:, sourceLandmarkInds[nzd]], targetLandmarks)
            P = np.r_[np.zeros(m.idEval.size + m.expEval.size), rho]
#        else:
#            rho = initialRegistration(generateFace(np.r_[P[:m.idEval.size + m.expEval.size], np.zeros(6), 1], m, sourceLandmarkInds[nzd]), targetLandmarks)
#            P[-7:] = rho
            
            # Further refine this initial guess while retrieving an initial guess of the shape parameters
#            initFit = minimize(initialShape, P, args = (targetLandmarks, m, sourceLandmarkInds[nzd]))
#            P = initFit['x']
        
        '''
        Optimization
        '''
        
        # Nearest neighbors fitting from scikit-learn to form correspondence between target vertices and source vertices during optimization
        xv, yv = np.meshgrid(np.arange(192), np.arange(192))
        target = np.c_[xv.flatten(), yv.flatten(), depth.flatten()][np.flatnonzero(depth), :]
        NN = NearestNeighbors(n_neighbors = 1, metric = 'l2')
        NN.fit(target)
        
        
#        grad = check_grad(shapeCost, shapeGrad, P, m, target, targetLandmarks, sourceLandmarkInds[nzd], NN)
#        break
    
#        optFit = minimize(shapeCost, P, args = (m, target, targetLandmarks, sourceLandmarkInds[nzd], NN), method = 'cg', jac = shapeGrad)
        
        if frame <= 20:
            
            cost = np.empty((50))
            for i in range(cost.size):
        #        print('Iteration %d' % i)
                cost[i], dP = gaussNewton(P, m, target, targetLandmarks, sourceLandmarkInds[nzd], NN, calcId = True)
                
                P += dP
        
        else:
            cost = np.empty((25))
            for i in range(cost.size):
        #        print('Iteration %d' % i)
                cost[i], dP = gaussNewton(P, m, target, targetLandmarks, sourceLandmarkInds[nzd], NN, calcId = False)
                
                P += dP
        
        """
        Transform translate and scale parameters for original image
        """
        
        # Save the parameters for the cropped/scaled image
        param[frame - 1, :] = P
        
        # Re-scale to original input image
        TS2orig[frame - 1, -1] = P[-1] / scale
        
        # Translate to account for original image dimensions
        # Case 1: The cropped picture is contained within the scaled image
        if (cropCorner >= 0).all() and ((192 + cropCorner) < scaledImgDim).all():
            TS2orig[frame - 1, :2] = (P[-4: -2] + cropCorner) / scale
        
        # Case 2: The crop corner is outside of the scaled image, but the extent of the cropped picture is within the bounds of the scaled image
        elif (cropCorner < 0).any() and ((192 + cropCorner) < scaledImgDim).all():
            TS2orig[frame - 1, :2] = (P[-4: -2] + cropCorner * (cropCorner > 0) + cropCorner * (cropCorner < 0) / 2) / scale
        
        # Case 3: The crop corner is outside of the scaled image, and the extent of the cropped picture is beyond the bounds of the scaled image
        elif (cropCorner < 0).any() and ((192 + cropCorner) > scaledImgDim).any():
            TS2orig[frame - 1, :2] = (P[-4: -2] + cropCorner * (cropCorner > 0) - (192 - (scaledImgDim - cropCorner * (cropCorner > 0))) / 2) / scale
#        
#    np.save('../param', param)
#    np.save('../paramRTS2Orig', np.c_[param[:, :m.idEval.size + m.expEval.size + 3], TS2orig])
#    np.save('../paramWithoutRTS', np.c_[param[:, :m.idEval.size + m.expEval.size], np.zeros((numFrames, 6)), np.ones(numFrames)])
#    np.save('../RTS', TS2orig)
#    
##    source = generateFace(P, m)
#    #exportObj(generateFace(np.r_[np.zeros(m.idEval.size + m.expEval.size), rho], m), f = m.face, fNameOut = 'initReg')
#    #exportObj(source, f = m.face, fNameOut = 'source')
##    exportObj(source[:, sourceLandmarkInds], fNameOut = 'sourceLandmarks')
#    #exportObj(target, fNameOut = 'target')
#    #exportObj(targetLandmarks, fNameOut = 'targetLandmarks')
#    

#
#    if not os.path.exists('../shapes'):
#        os.makedirs('../shapes')
#    for shape in range(numFrames):
#        fName = '{:0>5}'.format(shape + 1)
#        exportObj(generateFace(np.r_[param[shape, :m.idEval.size + m.expEval.size], np.zeros(6), 1], m), f = m.face, fNameOut = '../shapes/' + fName)
    
#    param = np.load('../paramWithoutRTS.npy')
    
    
#    shape = generateFace(np.r_[param[0, :m.idEval.size + m.expEval.size], np.zeros(6), 1], m)
#    tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1))
#    view = mlab.view()
#    
#    if not os.path.exists('../shapePic'):
#        os.makedirs('../shapePic')
#    for frame in range(21):
#        fName = '{:0>5}'.format(frame + 1)
#        shape = generateFace(np.r_[param[frame, :m.idEval.size + m.expEval.size], np.zeros(6), 1], m)
#        
##        mlab.options.offscreen = True
#        tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1))
#        mlab.view(view[0], view[1], view[2], view[3])
#        mlab.savefig('../shapePic/' + fName + '.png', figure = mlab.gcf())
#        mlab.close(all = True)
    
    
    
    ##    shape = np.load('/home/leon/f2f-fitting/kao3/depth/meshPoints/Adam_Freier_0001.npy')
##    depth = np.load('/home/leon/f2f-fitting/kao3/depth/Adam_Freier_0001.npy')
    
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    xv, yv = np.meshgrid(np.arange(192), np.arange(192))
#    ax.scatter(xv, yv, depth, s = 0.1, c = 'b')
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
    
    #im = vv.imread(fNameImgScaled)
    #
    #t = vv.imshow(im)
    #t.interpolate = True # interpolate pixels
    #
    ## volshow will use volshow3 and rendering the isosurface if OpenGL
    ## version is >= 2.0. Otherwise, it will show slices with bars that you
    ## can move (much less useful).
    #volRGB = np.stack(((vol > 1) * im[:,:,0],
    #                   (vol > 1) * im[:,:,1],
    #                   (vol > 1) * im[:,:,2]), axis=3)
    #
    #v = vv.volshow(vol > 1, renderStyle='iso')
    #
    #l0 = vv.gca()
    #l0.light0.ambient = 0.9 # 0.2 is default for light 0
    #l0.light0.diffuse = 1.0 # 1.0 is default
    #
    #a = vv.gca()
    #a.camera.fov = 0 # orthographic
    #
    #vv.use().Run()