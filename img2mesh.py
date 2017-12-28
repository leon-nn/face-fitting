#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:24:40 2017

@author: leon
"""

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
from scipy.optimize import minimize, check_grad, least_squares
from scipy.linalg import rq
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

def orthographicEst(lm2D, lm3D):
    numLandmarks = lm2D.shape[0]
    
    # Normalize landmark coordinates; preconditioning
    c2D = np.mean(lm2D, axis = 0)
    uvCentered = lm2D - c2D
    s2D = np.linalg.norm(uvCentered, axis = 1).mean()
    x = uvCentered / s2D * np.sqrt(2)
    
    c3D = np.mean(lm3D, axis = 0)
    xyzCentered = lm3D - c3D
    s3D = np.linalg.norm(xyzCentered, axis = 1).mean()
    X = xyzCentered / s3D * np.sqrt(3)
    
    # Similarity transformations for normalization
    Tinv = np.array([[s2D, 0, c2D[0]], [0, s2D, c2D[1]], [0, 0, 1]])
    U = np.linalg.inv([[s3D, 0, 0, c3D[0]], [0, s3D, 0, c3D[1]], [0, 0, s3D, c3D[2]], [0, 0, 0, 1]])

    # Build linear system of equations in 8 unknowns of projection matrix
    A = np.zeros((2 * numLandmarks, 8))
    
    A[0: 2*numLandmarks - 1: 2, :3] = X
    A[0: 2*numLandmarks - 1: 2, 3] = 1
    
    A[1: 2*numLandmarks: 2, 4: 7] = X
    A[1: 2*numLandmarks: 2, 7] = 1
    
    # Solve linear system and de-normalize
    p8 = np.linalg.lstsq(A, x.flatten())[0]
    
    Pnorm = np.r_[p8, 0, 0, 0, 1].reshape(3, 4)
    P = Tinv.dot(Pnorm).dot(U)
    
    return P[:2, :]

def perspectiveEst(lm2D, lm3D):
    # Direct linear transform / "Gold Standard Algorithm"
    # Normalize landmark coordinates; preconditioning
    numLandmarks = lm2D.shape[0]
    
    c2D = np.mean(lm2D, axis = 0)
    uvCentered = lm2D - c2D
    s2D = np.linalg.norm(uvCentered, axis = 1).mean()
    x = np.c_[uvCentered / s2D * np.sqrt(2), np.ones(numLandmarks)]
    
    c3D = np.mean(lm3D, axis = 0)
    xyzCentered = lm3D - c3D
    s3D = np.linalg.norm(xyzCentered, axis = 1).mean()
    X = np.c_[xyzCentered / s3D * np.sqrt(3), np.ones(numLandmarks)]
    
    # Similarity transformations for normalization
    Tinv = np.array([[s2D, 0, c2D[0]], [0, s2D, c2D[1]], [0, 0, 1]])
    U = np.linalg.inv([[s3D, 0, 0, c3D[0]], [0, s3D, 0, c3D[1]], [0, 0, s3D, c3D[2]], [0, 0, 0, 1]])
    
    # Create matrix for homogenous system of equations to solve for camera matrix
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

def estCamMat(lm2D, lm3D, cam = 'perspective'):
    """
    Direct linear transform / "Gold Standard Algorithm" to estimate camera matrix from 2D-3D landmark correspondences. The input 2D and 3D landmark NumPy arrays have XY and XYZ coordinates in each row, respectively. For an orthographic camera, the algebraic and geometric errors are equivalent, so there is no need to do the least squares step at the end. The orthographic camera returns a 2x4 camera matrix, since the last row is just [0, 0, 0, 1].
    """
    # Normalize landmark coordinates; preconditioning
    numLandmarks = lm2D.shape[0]
    
    c2D = np.mean(lm2D, axis = 0)
    uvCentered = lm2D - c2D
    s2D = np.linalg.norm(uvCentered, axis = 1).mean()
    x = np.c_[uvCentered / s2D * np.sqrt(2), np.ones(numLandmarks)]
    
    c3D = np.mean(lm3D, axis = 0)
    xyzCentered = lm3D - c3D
    s3D = np.linalg.norm(xyzCentered, axis = 1).mean()
    X = np.c_[xyzCentered / s3D * np.sqrt(3), np.ones(numLandmarks)]
    
    # Similarity transformations for normalization
    Tinv = np.array([[s2D, 0, c2D[0]], [0, s2D, c2D[1]], [0, 0, 1]])
    U = np.linalg.inv([[s3D, 0, 0, c3D[0]], [0, s3D, 0, c3D[1]], [0, 0, s3D, c3D[2]], [0, 0, 0, 1]])
    
    if cam == 'orthographic':
        # Build linear system of equations in 8 unknowns of projection matrix
        A = np.zeros((2 * numLandmarks, 8))
        
        A[0: 2*numLandmarks - 1: 2, :4] = X
        A[1: 2*numLandmarks: 2, 4:] = X
        
        # Solve linear system and de-normalize
        p8 = np.linalg.lstsq(A, x.flatten())[0]
        
        Pnorm = np.r_[p8, 0, 0, 0, 1].reshape(3, 4)
        P = Tinv.dot(Pnorm).dot(U)
        
        return P[:2, :]
    
    elif cam == 'perspective':
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
        K, R = rq(P[:, :3])
        angle = rotMat2angle(R)
        t = np.linalg.inv(K).dot(P[:, -1])
        
        return K, angle, t
    
def camWithShape(param, m, lm2d, lm3dInd, cam):
    """
    Minimize L2-norm of landmark fitting residuals and regularization terms for shape parameters
    """
    if cam == 'orthographic':
        P = param[:8]
        P = np.vstack((P.reshape((2, 4)), np.array([0, 0, 0, 1])))
        idCoef = param[8: 8 + m.idEval.size]
        expCoef = param[8 + m.idEval.size:]
    
    elif cam == 'perspective':
        P = param[:12]
        P = P.reshape((3, 4))
        idCoef = param[12: 12 + m.idEval.size]
        expCoef = param[12 + m.idEval.size:]
    
    # Convert to homogenous coordinates
    numLandmarks = lm3dInd.size
    
    lm3d = generateFace(np.r_[idCoef, expCoef, np.zeros(6), 1], m, ind = lm3dInd).T
    
    xlan = np.c_[lm2d, np.ones(numLandmarks)]
    Xlan = np.dot(np.c_[lm3d, np.ones(numLandmarks)], P.T)
    
    # Energy of landmark residuals
    rlan = (Xlan - xlan).flatten('F')
    Elan = np.dot(rlan, rlan)
    
    # Energy of shape regularization terms
    Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    return Elan + Ereg

def dR_dpsi(angles):
    """
    Derivative of the rotation matrix with respect to the x-axis rotation.
    """
    psi, theta, phi = angles
    return np.array([[0, np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi), np.cos(psi)*np.sin(phi) - np.sin(psi)*np.sin(theta)*np.cos(phi)], [0, -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.cos(phi) - np.sin(psi)*np.sin(theta)*np.sin(phi)], [0, np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(theta)]])

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

if __name__ == "__main__":
    
    os.chdir('/home/leon/f2f-fitting/obama/orig/')
    numFrames = 2882 #2260 #3744
    
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
    
#    plt.ioff()
    param = np.zeros((numFrames, m.idEval.size + m.expEval.size + 7))
    cam = 'perspective'
    
    for frame in np.arange(1, 1 + 1):
        print(frame)
        fName = '{:0>5}'.format(frame)
        fNameImgOrig = '../orig/' + fName + '.png'
        fNameLandmarks = '../landmark/' + fName + '.json'
        
        '''
        Preprocess landmark locations: map to cropped/scaled version of image
        '''
        
        with open(fNameLandmarks, 'r') as fd:
            lm = json.load(fd)
        lm = np.array([l[0] for l in lm], dtype = int).squeeze()[:, :3]
        lmConf = lm[targetLandmarkInds, -1]
        lm = lm[targetLandmarkInds, :2]
        
        # Plot the landmarks on the image
        img = mpimg.imread(fNameImgOrig)
#        plt.figure()
#        plt.imshow(img)
#        plt.scatter(lm[:, 0], lm[:, 1], s = 2)
#        plt.title(fName)
#        if not os.path.exists('../landmarkPic'):
#            os.makedirs('../landmarkPic')
#        savefig('../landmarkPic/' + fName + '.png', bbox_inches='tight')
#        plt.close('all')
#        plt.close()
        
#        fig, ax = plt.subplots()
#        plt.imshow(img)
#        plt.hold(True)
#        x = lm[:, 0]
#        y = lm[:, 1]
#        ax.scatter(x, y, s = 2, c = 'b', picker = True)
#        fig.canvas.mpl_connect('pick_event', onpick3)
        
        '''
        Initial registration
        '''
        
        if frame == 1:
            idCoef = np.zeros(m.idEval.size)
            expCoef = np.zeros(m.expEval.size)
            texCoef = np.zeros(m.texEval.size)
            param = np.r_[np.zeros(m.idEval.size + m.expEval.size + 6), 1]
            
        lm3D = generateFace(param, m, ind = sourceLandmarkInds).T
        
        P = estCamMat(lm, lm3D, cam)
        
        # Even more minimization with projection matrix to get initial shape parameters
        initCamShape = minimize(camWithShape, np.r_[P.flatten(), idCoef, expCoef], args = (m, lm, sourceLandmarkInds, cam))
        
        # Separate variates in parameter vector
        P = initCamShape.x[:12].reshape((3, 4))
        idCoef = initCamShape.x[12: 12 + m.idEval.size]
        expCoef = initCamShape.x[12 + m.idEval.size:]
        
        K, angles, t = splitCamMat(P, cam)
        
        # Project 3D model into 2D plane
        param = np.r_[idCoef, expCoef, angles, t, 1]
        modelBeforeProj = generateFace(param, m)
        fitting = K.dot(modelBeforeProj)
        
        # Plot the projected 3D model on top of the input RGB image
        plt.figure()
        plt.imshow(img)
#        plt.scatter(fitting[0, :], fitting[1, :], s = 0.1, c = 'g')
        plt.scatter(fitting[0, sourceLandmarkInds], fitting[1, sourceLandmarkInds], s = 3, c = 'b')
#        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'r')
        
#        tmesh = mlab.triangular_mesh(modelBeforeProj[0, :], modelBeforeProj[1, :], modelBeforeProj[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1))
        
        break
        # Z-buffer: smaller z is closer to image plane (e.g. the nose should have relatively small z values)
        vertex2pixel = fitting[:2, :].T.astype(int)
        pixelInd, ind, counts = np.unique(vertex2pixel, return_index = True, return_counts = True, axis = 0)
        zBuffer = np.empty(ind.size, dtype = int)
        for i in range(ind.size):
            if counts[i] == 1:
                zBuffer[i] = ind[i]
            else:
                inds = np.where((vertex2pixel[:, 0] == pixelInd[i, 0]) & (vertex2pixel[:, 1] == pixelInd[i, 1]))[0]
                zBuffer[i] = inds[np.argmin(modelBeforeProj[2, inds])]
        
        #mask = np.zeros(img.shape[:2], dtype = bool)
        #mask.flat[np.ravel_multi_index(vertex2pixel[zBuffer, ::-1].T, img.shape[:2])] = True
        #plt.figure()
        #plt.imshow(mask)
        """
        """
        def textureCost(texCoef, x, mask, m):
            """
            Energy formulation for fitting texture
            """
            # Photo-consistency
            Xcol = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1)).T
            
            r = (Xcol - x).flatten()
            
            Ecol = np.dot(r, r) / mask.size
            
            # Statistical regularization
            Ereg = np.sum(texCoef ** 2 / m.texEval)
            
            return Ecol + Ereg
        
        def textureGrad(texCoef, x, mask, m):
            """
            Jacobian for texture energy
            """
            Xcol = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1)).T
            
            r = (Xcol - x).flatten()
            
            # Jacobian
            J = m.texEvec[:, mask, :].reshape((m.texMean[:, mask].size, m.texEval.size), order = 'F')
            
            return 2 * (np.dot(J.T, r) / mask.size + texCoef / m.texEval)
        
        def fitter(param, x, vis, m, lm2d, lm3d):
            """
            Energy formulation for fitting 3D face model to 2D image
            """
            K = np.reshape(param[:6], (2, 3))
            angle = param[6: 9]
            R = rotMat2angle(angle)
            t = param[9: 12]
            shCoef = param[12: 39]
            idCoef = param[39: 39 + m.idEval.size]
            expCoef = param[39 + m.idEval.size: 39 + m.idEval.size + m.expEval.size]
            texCoef = param[39 + m.idEval.size + m.expEval.size:]
            
            # Photo-consistency
            shape = R.dot(m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)) + t[:, np.newaxis]
            
            texture = m.texMean[:, vis] + np.tensordot(m.texEvec[:, vis, :], texCoef, axes = 1)
            
            normals = calcNormals(R, m, idCoef, expCoef)
            shBases = shBasis(texture, normals)
            
            texture[0, :] = np.tensordot(shBases[0, :, :], shCoef[:9], axes = 1)
            texture[1, :] = np.tensordot(shBases[1, :, :], shCoef[9: 18], axes = 1)
            texture[2, :] = np.tensordot(shBases[2, :, :], shCoef[18: 27], axes = 1)
            
            Ecol = np.linalg.norm(x - texture[:, ind].T, axis = 1).sum() / ind.size
            
            # Feature alignment (landmarks)
            Xlan = K.dot(R.dot(m.idMean[:, lm3d] + np.tensordot(m.idEvec[:, lm3d, :], idCoef, axes = 1) + np.tensordot(m.expEvec[:, lm3d, :], expCoef, axes = 1)) + t[:, np.newaxis])
            Elan = np.linalg.norm(lm2d - Xlan.T, axis = 1).sum() / lm3d.size
            
            # Statistical regularization
            Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval) + np.sum(texCoef ** 2 / m.texEval)
            
            return Ecol + 10*Elan + Ereg
        
        x = np.reshape(img, (np.prod(img.shape[:2]), 3))
        x = x[np.ravel_multi_index(vertex2pixel[zBuffer, ::-1].T, img.shape[:2]), :]
        param2 = minimize(textureCost, texCoef, args = (x, zBuffer, m), method = 'cg', jac = textureGrad)
        check_grad(textureCost, textureGrad, texCoef, x, zBuffer, m)
        #param2 = minimize(fitter, np.r_[K[:2, :].flatten(), angles, t, idCoef, expCoef, texCoef], args = (x, m, landmarkPixelInd, landmarkInds3D))
        
        texCoef = param2['x']
        #K2 = np.reshape(param2.x[:6], (2, 3))
        #angles2 = param2.x[6: 9]
        #R2 = rotMat2angle(angles2)
        #t2 = param2.x[9: 12]
        #idCoef2 = param2.x[12: 12 + m.idEval.size]
        #expCoef2 = param2.x[12 + m.idEval.size: 12 + m.idEval.size + m.expEval.size]
        #texCoef = param2.x[12 + m.idEval.size + m.expEval.size:]
        
        # Project 3D model into 2D plane
        #fitting = K2.dot(R2.dot(m.idMean + np.tensordot(m.idEvec, idCoef2, axes = 1) + np.tensordot(m.expEvec, expCoef2, axes = 1)) + t2[:, np.newaxis])
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        #exportObj(shape.T, c = texture.T, f = m.face, fNameOut = 'texTest')
        tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices))
        tmesh.module_manager.scalar_lut_manager.lut.table = np.c_[(texture.T * 255), 255 * np.ones(m.numVertices)].astype(np.uint8)
        mlab.draw()
        
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
    
        optFit = minimize(shapeCost, P, args = (m, target, targetLandmarks, sourceLandmarkInds[nzd], NN), method = 'cg', jac = shapeGrad)
        P = optFit['x']
        
        source = generateFace(P, m)
        plt.figure()
        plt.imshow(imgScaled)
        plt.scatter(source[0, :], source[1, :], s = 1)
        break

#    np.save('../param', param)
#    np.save('../paramRTS2Orig', np.c_[param[:, :m.idEval.size + m.expEval.size + 3], TS2orig])
#    np.save('../paramWithoutRTS', np.c_[param[:, :m.idEval.size + m.expEval.size], np.zeros((numFrames, 6)), np.ones(numFrames)])
#    np.save('../RTS', TS2orig)
    
##    source = generateFace(P, m)
#    #exportObj(generateFace(np.r_[np.zeros(m.idEval.size + m.expEval.size), rho], m), f = m.face, fNameOut = 'initReg')
#    #exportObj(source, f = m.face, fNameOut = 'source')
##    exportObj(source[:, sourceLandmarkInds], fNameOut = 'sourceLandmarks')
#    #exportObj(target, fNameOut = 'target')
#    #exportObj(targetLandmarks, fNameOut = 'targetLandmarks')
#    

#    param = np.load('../paramRTS2Orig.npy')
#    if not os.path.exists('../shapesPrecise'):
#        os.makedirs('../shapesPrecise')
#    for shape in range(numFrames):
#        fName = '{:0>5}'.format(shape + 1)
#        exportObj(generateFace(np.r_[param[shape, :m.idEval.size + m.expEval.size], np.zeros(6), 1], m), f = m.face, fNameOut = '../shapesPrecise/' + fName)
    
#    param = np.load('../paramWithoutRTS.npy')
    
    
#    shape = generateFace(np.r_[param[0, :m.idEval.size + m.expEval.size], np.zeros(6), 1], m)
#    tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1))
##    view = mlab.view()
#    
#    if not os.path.exists('../shapePic'):
#        os.makedirs('../shapePic')
#    for frame in range(100):
#        fName = '{:0>5}'.format(frame + 1)
#        shape = generateFace(np.r_[param[frame, :m.idEval.size + m.expEval.size], np.zeros(6), 1], m)
#        
##        mlab.options.offscreen = True
#        tmesh = mlab.triangular_mesh(shape[0, :], shape[1, :], shape[2, :], m.face, scalars = np.arange(m.numVertices), color = (1, 1, 1))
#        mlab.view(view[0], view[1], view[2], view[3])
#        mlab.savefig('../shapePic/' + fName + '.png', figure = mlab.gcf())
#        mlab.close(all = True)