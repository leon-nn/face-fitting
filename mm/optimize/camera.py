#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:42:30 2018

@author: leon
"""

import numpy as np
from ..utils.transform import rotMat2angle
from scipy.linalg import rq
from scipy.optimize import least_squares
    
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

def estCamMat(lm2D, lm3D, cam = 'perspective'):
    """
    Direct linear transform / "Gold Standard Algorithm" to estimate camera matrix from 2D-3D landmark correspondences. The input 2D and 3D landmark NumPy arrays have XY and XYZ coordinates in each row, respectively. For an orthographic camera, the algebraic and geometric errors are equivalent, so there is no need to do the least squares step at the end. The orthographic camera returns a 2x4 camera matrix, since the last row is just [0, 0, 0, 1].
    """
    # Normalize landmark coordinates; preconditioning
    numLandmarks = lm2D.shape[0]
    
    c2D = np.mean(lm2D, axis = 0)
    uvCentered = lm2D - c2D
    s2D = np.linalg.norm(uvCentered, axis = 1).mean()
    
    
    c3D = np.mean(lm3D, axis = 0)
    xyzCentered = lm3D - c3D
    s3D = np.linalg.norm(xyzCentered, axis = 1).mean()
    X = np.c_[xyzCentered / s3D * np.sqrt(3), np.ones(numLandmarks)]
    
    # Similarity transformations for normalization
    Tinv = np.array([[s2D, 0, c2D[0]], [0, s2D, c2D[1]], [0, 0, 1]])
    U = np.linalg.inv([[s3D, 0, 0, c3D[0]], [0, s3D, 0, c3D[1]], [0, 0, s3D, c3D[2]], [0, 0, 0, 1]])
    
    if cam == 'orthographic':
        x = uvCentered / s2D * np.sqrt(2)
        
        # Build linear system of equations in 8 unknowns of projection matrix
        A = np.zeros((2 * numLandmarks, 8))
        
        A[0: 2*numLandmarks - 1: 2, :4] = X
        A[1: 2*numLandmarks: 2, 4:] = X
        
        # Solve linear system and de-normalize
        p8 = np.linalg.lstsq(A, x.flatten())[0].reshape(2, 4)
        
#        K, R = rq(p8[:, :3], mode = 'economic')
#        R = np.vstack((R[0, :], R[1, :], np.cross(R[0, :], R[1, :])))
#        angles = rotMat2angle(R)
#        param = np.r_[K[0, 0], K[0, 1], K[1, 1], angles, p8[:, 3]]
#        
#        def orthographicCamMatLS(param, x, X, w):
#            # Reconstruct the camera matrix P from the RQ decomposition
#            K = np.array([[param[0], param[1]], [0 , param[2]]])
#            R = rotMat2angle(param[3: 6])[:2, :]
#            P = np.c_[K.dot(R), param[6:]]
#            
#            # Calculate resisduals of landmark correspondences
#            r = x.flatten() - np.dot(X, P.T).flatten()
#    
#            # Calculate residuals for constraints
#            rscale = np.fabs(param[0] - param[2])
#            rskew = param[1]
#    
#            return np.r_[w[0] * r, w[1] * rscale, w[2] * rskew]
#            
#        def orthographicCamMat(param, x, X, w):
#            # Reconstruct the camera matrix P from the RQ decomposition
#            K = np.array([[param[0], param[1]], [0 , param[2]]])
#            R = rotMat2angle(param[3: 6])[:2, :]
#            P = np.c_[K.dot(R), param[6:]]
#            
#            # Calculate resisduals of landmark correspondences
#            r = x.flatten() - np.dot(X, P.T).flatten()
#    
#            # Calculate costs
#            Elan = np.dot(r, r)
#            Escale = np.square(np.fabs(param[0]) - np.fabs(param[2]))
#            Eskew = np.square(param[1])
#    
#            return w[0] * Elan + w[1] * Escale + w[2] * Eskew
#        
#        param = minimize(orthographicCamMat, param, args = (x, X, (5, 1, 1)))
#        param = least_squares(orthographicCamMatLS, param, args = (x, X, (1, 1, 1)), bounds = (np.r_[0, 0, 0, -np.inf*np.ones(5)], np.inf))
#        K = np.array([[param.x[0], param.x[1]], [0 , param.x[2]]])
#        R = rotMat2angle(param.x[3: 6])[:2, :]
#        p8 = np.c_[K.dot(R), param.x[6:]]
        
        Pnorm = np.vstack((p8, np.array([0, 0, 0, 1])))
        P = Tinv.dot(Pnorm).dot(U)
        
        return P[:2, :]
    
    elif cam == 'perspective':
        x = np.c_[uvCentered / s2D * np.sqrt(2), np.ones(numLandmarks)]
        
        # Matrix for homogenous system of equations to solve for camera matrix
        A = np.zeros((2 * numLandmarks, 12))
        
        A[0: 2*numLandmarks - 1: 2, 0: 4] = X
        A[0: 2*numLandmarks - 1: 2, 8:] = -x[:, 0, np.newaxis] * X
        
        A[1: 2*numLandmarks: 2, 4: 8] = -X
        A[1: 2*numLandmarks: 2, 8:] = x[:, 1, np.newaxis] * X
        
        # Take the SVD and take the last row of V', which corresponds to the lowest eigenvalue, as the homogenous solution
        V = np.linalg.svd(A, full_matrices = 0)[-1]
        Pnorm = np.reshape(V[-1, :], (3, 4))
        
        # Further nonlinear LS to minimize error between 2D landmarks and 3D projections onto 2D plane.
        def cameraProjectionResidual(M, x, X):
            """
            min_{P} sum_{i} || x_i - PX_i ||^2
            """
            return x.flatten() - np.dot(X, M.reshape((3, 4)).T).flatten()
        
        Pgold = least_squares(cameraProjectionResidual, Pnorm.flatten(), args = (x, X))
        
        # Denormalize P
        P = Tinv.dot(Pgold.x.reshape(3, 4)).dot(U)
        
        return P

def splitCamMat(P, cam = 'perspective'):
    """
    """
    if cam == 'orthographic':
        # Extract params from orthographic projection matrix
        R1 = P[0, 0: 3]
        R2 = P[1, 0: 3]
        st = np.r_[P[0, 3], P[1, 3]]
        
        s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
        r1 = R1 / np.linalg.norm(R1)
        r2 = R2 / np.linalg.norm(R2)
        r3 = np.cross(r1, r2)
        R = np.vstack((r1, r2, r3))
        
        # Set R to closest orthogonal matrix to estimated rotation matrix
        U, V = np.linalg.svd(R)[::2]
        R = U.dot(V)
        
        # Determinant of R must = 1
        if np.linalg.det(R) < 0:
            U[2, :] = -U[2, :]
            R = U.dot(V)
        
        # Remove scale from translations
        t = st / s
        
        angle = rotMat2angle(R)
        
        return s, angle, st
    
    elif cam == 'perspective':
        # Get inner parameters from projection matrix via RQ decomposition
        K, R = rq(P[:, :3], mode = 'economic')
        angle = rotMat2angle(R)
        t = np.linalg.inv(K).dot(P[:, -1])
        
        return K, angle, t