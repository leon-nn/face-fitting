#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:43:00 2017

@author: nguyen
"""
import numpy as np

# For computing eigenvalues and eigenvectors via eigsh
from scipy.sparse.linalg import eigsh

# Just-in-time compiler
from numba import jit

# Regex for extra help with parsing input .obj files
import re

# For plotting and importing PNG images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from mpl_toolkits.mplot3d import Axes3D
#import scipy.interpolate as intp

from time import clock
import os
from collections import defaultdict
from itertools import chain, compress

#from scipy.stats import mode

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq
from scipy.io import loadmat

import h5py

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
    
def importObj(dirName, shape = 0, dataToImport = ['v', 'vt', 'f'], pose = 20):
    """
    Return the geometric and texture vertices along with the quadrilaterials containing the geometric and texture indices for all 150 testers of FaceWarehouse for a certain shape/pose/expression. Input (1) a string for the directory name that contains the folders 'Tester_1' through 'Tester_150', (2) an int for the shape number, which is in the range [0, 46] (1 neutral + 46 expressions), and (3) a list containing strings to indicate what part of the .obj file to read ('v' = geometric vertices, 'vt' = texture vertices, 'f' = face quadrilaterals).
    """
    # Number of observations (people/testers) in the dataset
    numTesters = 150
    
    # If input is just a single .obj file
    if dirName.endswith('.obj'):
        singleFile = True
        fName = dirName
    else:
        singleFile = False
        
        # Make sure directory name has final forward slash
        if not dirName.endswith('/'):
            dirName += '/'
    
    for i in range(numTesters):
        if (not singleFile) and pose == 47:
            fName = dirName + 'Tester_' + str(i+1) + '/Blendshape/shape_' + str(shape) + '.obj'
        elif (not singleFile) and pose == 20:
            fName = dirName + 'Tester_' + str(i+1) + '/TrainingPose/pose_' + str(shape) + '.obj'
        
        with open(fName) as fd:
            # Initialize lists to store the data from the .obj files
            v = []      # Geometric vertices (x, y, z)
            vt = []     # Texture vertices (U, V)
            f = []      # Face quadrilaterals
            for line in fd:
                if line.startswith('v ') and 'v' in dataToImport:
                    v.append([float(num) for num in line[2:].split(' ')])
                elif line.startswith('vt') and 'vt' in dataToImport and i == 0:
                    vt.append([float(num) for num in line[3:].split(' ')])
                elif line.startswith('f') and 'f' in dataToImport and i == 0:
                    f.append([int(ind) for ind in re.split('/| ', line[2:])])
                else:
                    continue
        
        if i == 0:
            geoV = np.empty((numTesters, len(v), 3))
            textV = np.empty((len(vt), 2))
            quad = np.empty((2, len(f), 4), dtype = 'int')
            
        # Store the data for each shape
        if 'vt' in dataToImport and i == 0:
            textV[:, :] = np.array(vt)
        if 'f' in dataToImport and i == 0:
            quad[0, :, :] = np.array(f)[:, [0, 3, 6, 9]]
            quad[1, :, :] = np.array(f)[:, [1, 4, 7, 10]]
            
        if 'v' in dataToImport and not singleFile:
            geoV[i, :, :] = np.array(v)
        elif 'v' in dataToImport and singleFile:
            geoV = np.array(v)
            break
        else:
            break
    
    # Select which data to return based on the dataToImport input
    objToData = {'v': geoV, 'vt': textV, 'f': quad}
    return [objToData.get(key) for key in dataToImport]

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

@jit
def PCA(data, numPC = 80):
    """
    Return the top principle components of some data. Input (1) the data as a 2D NumPy array, where the observations are along the rows and the data elements of each observation are along the columns, and (2) the number of principle components (numPC) to keep.
    """
    # Number of observations
    M = data.shape[0]
    
    # Mean (not using np.mean for jit reasons)
    mean = data.sum(axis = 0)/M
    data = data - mean
    
    # Covariance (we don't remove the M scaling factor here to try to avoid floating point errors that could make C unsymmetric)
    C = np.dot(data.T, data)
    
    # Compute the top 'numPC' eigenvectors & eigenvalues of the covariance matrix. This uses the scipy.sparse.linalg version of eigh, which happens to be much faster for some reason than the nonsparse version for this case where k << N. Since we didn't remove the M scaling factor in C, the eigenvalues here are scaled by M.
    eigVal, eigVec = eigsh(C, k = numPC, which = 'LM')

    return eigVal[::-1]/M, eigVec[:, ::-1], mean

def subdivide(v, f):
    """
    Use Catmull-Clark subdivision to subdivide a quad-mesh, increasing the number of faces by 4 times. Input the vertices and the face-vertex index mapping.
    """
    # Make v 3D if it isn't, for my convenience
    if len(v.shape) != 3:
        v = v[np.newaxis, :, :]
    
    # Check to make sure f is 2D (only shape info) and indices start at 0
    if len(f.shape) != 2:
        f = f[0, :, :]
    if np.min(f) != 0:
        f = f - 1
        
    # Find the edges in the input face mesh
    edges = np.c_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 3]], f[:, [3, 0]]]
    edges = np.reshape(edges, (4*f.shape[0], 2))
    edges = np.sort(edges, axis = 1)
    edges, edgeInd = np.unique(edges, return_inverse = True, axis = 0)
    edges = [frozenset(edge) for edge in edges]
    
    # Map from face index to sets of edges connected to the face
    face2edge = [[frozenset(edge) for edge in np.c_[face[:2], face[1:3], face[2:4], face[[-1, 0]]].T] for face in f]
    
    # Map from sets of edges to face indices
    edge2face = defaultdict(list)
    for faceInd, edgesOnFace in enumerate(face2edge):
        for edge in edgesOnFace:
            edge2face[edge].append(faceInd)
    
    # Map from vertices to the faces they're connected to
    vertex2face = [np.where(np.isin(f, vertexInd).any(axis = 1))[0].tolist() for vertexInd in range(v.shape[1])]
    
    # Map from vertices to the edges they're connected to
    vertex2edge = [list(compress(edges, [vertexInd in edge for edge in edges])) for vertexInd in range(v.shape[1])]
    
    # Number of faces connected to each vertex (i.e. valence)
    nFaces = np.array([np.isin(f, vertexInd).any(axis = 1).sum() for vertexInd in range(v.shape[1])])
    
    # Number of edges connected to each vertex
    nEdges = np.array([len(vertex2edge[vertexInd]) for vertexInd in range(v.shape[1])])
    
    # Loop thru the vertices of each tester's face to find the new set of vertices
    for tester in range(v.shape[0]):
        print('Calculating new vertices for tester %d' % (tester + 1))
        # Face points: the mean of the vertices on a face
        facePt = np.array([np.mean(v[tester, vertexInd, :], axis = 0) for vertexInd in f])
        
        # Edge points
        edgePt = np.empty((len(edges), 3))
        for i, edge in enumerate(edges):
            # If an edge is only associated with one face, then it is on a border of the 3D model. The edge point is thus the midpoint of the vertices defining the edge.
            if len(edge2face[edge]) == 1:
                edgePt[i, :] = np.mean(v[tester, list(edge), :], axis = 0)
            
            # Else, the edge point is the mean of (1) the face points of the two faces adjacent to the edge and (2) the midpoint of the vertices defining the edge.
            else:
                edgePt[i, :] = np.mean(np.r_[facePt[edge2face[edge], :], v[tester, list(edge), :]], axis = 0)
        
        # New coordinates: loop thru each vertex P of the original vertices to calc
        newPt = np.empty(v.shape[1: ])
        for i, P in enumerate(v[tester, :, :]):
            # If P is not on the border
            if nFaces[i] == nEdges[i]:
                # Mean of the face points from the faces surrounding P
                F = np.mean(facePt[vertex2face[i], :], axis = 0)
                
                # Mean of the edge midpoints from the edges connected to P
                R = np.mean(v[tester, list(chain.from_iterable(vertex2edge[i])), :], axis = 0)
                
                # The new coordinates of P is a combination of F, R, and P
                newPt[i, :] = (F + 2*R + (nFaces[i] - 3)*P)/nFaces[i]
                
            # Otherwise, P is on the border
            else:
                # For the edges connected to P, find the edges on the border
                borderEdge = [len(edge2face[edge]) == 1 for edge in vertex2edge[i]]
                
                # The midpoints of these edges on the border
                R = v[tester, list(chain.from_iterable(compress(vertex2edge[i], borderEdge))), :]
                
                # The new coordinates of P is the mean of R and P
                newPt[i, :] = np.mean(np.r_[R, P[np.newaxis, :]], axis = 0)
        
        # Save the result
        if tester == 0:
            vNew = np.empty((v.shape[0], facePt.shape[0] + edgePt.shape[0] + newPt.shape[0], 3))
            
        vNew[tester, :, :] = np.r_[facePt, edgePt, newPt]
    
    # Form the new faces
    fNew = np.c_[f.flatten() + facePt.shape[0] + edgePt.shape[0], edgeInd + facePt.shape[0], np.repeat(np.arange(facePt.shape[0]), 4), edgeInd.reshape((edgeInd.shape[0]//4, 4))[:, [3, 0, 1, 2]].flatten() + facePt.shape[0]] + 1
    
    return vNew, fNew

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

@jit
def perspectiveTransformKinect(d, inverse = False):
    """
    Transformation between pixel indices (u, v) of depth map to real-world coordinates in mm (x, y) for Kinect v1 depth camera (640x480 resolution). Depth values z are in mm. In the forward direction, go from (u, v, z) to (x, y, z). In the inverse direction, go from (x, y, z) to (u, v, z).
    """
    # Mapping from (x, y, z) to (uz, vz, z)
    real2pixel = np.array([[580.606, 0, 314.758], [0, 580.885, 252.187], [0, 0, 1]])
    
    # Mapping from (uz, vz, z) to (x, y, z)
    pixel2real = np.linalg.inv(real2pixel)
    
    # Mark depth values that are non-zero
    nonZeroZ = d[:, 2] != 0
    
    if not inverse:
        uvz = d[nonZeroZ, :]
        uzvzz = np.c_[np.prod(uvz[:, ::2], axis = 1), np.prod(uvz[:, 1:], axis = 1), uvz[:, 2]]
        xyz = np.dot(pixel2real, uzvzz.T).T
        
        return xyz, nonZeroZ
    
    else:
        xyz = d[nonZeroZ, :]
        uzvzz = np.dot(real2pixel, xyz.T).T
        uvz = np.c_[uzvzz[:, 0] / xyz[:, 2], uzvzz[:, 1] / xyz[:, 2], xyz[:, 2]]
        
        return uvz, nonZeroZ

@jit
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

def processBFM2017(fName, fNameLandmarks):
    """
    Read the face models and landmarks from the Basel Face Model 2017 dataset. Input the filename of the .h5 file and the filename of a .txt file containing the text detailing the landmark locations.
    """
    data = h5py.File(fName, 'r')
    
    # Identity
    idMean = np.empty(data.get('/shape/model/mean').shape)
    data.get('/shape/model/mean').read_direct(idMean)
    idVar = np.empty(data.get('/shape/model/noiseVariance').shape)
    data.get('/shape/model/noiseVariance').read_direct(idVar)
    idEvec = np.empty(data.get('/shape/model/pcaBasis').shape)
    data.get('/shape/model/pcaBasis').read_direct(idEvec)
    idEval = np.empty(data.get('/shape/model/pcaVariance').shape)
    data.get('/shape/model/pcaVariance').read_direct(idEval)
    
    # Expression
    expMean = np.empty(data.get('/expression/model/mean').shape)
    data.get('/expression/model/mean').read_direct(expMean)
    expVar = np.empty(data.get('/expression/model/noiseVariance').shape)
    data.get('/expression/model/noiseVariance').read_direct(expVar)
    expEvec = np.empty(data.get('/expression/model/pcaBasis').shape)
    data.get('/expression/model/pcaBasis').read_direct(expEvec)
    expEval = np.empty(data.get('/expression/model/pcaVariance').shape)
    data.get('/expression/model/pcaVariance').read_direct(expEval)
    
    # Texture
    texMean = np.empty(data.get('/color/model/mean').shape)
    data.get('/color/model/mean').read_direct(texMean)
    texVar = np.empty(data.get('/color/model/noiseVariance').shape)
    data.get('/color/model/noiseVariance').read_direct(texVar)
    texEvec = np.empty(data.get('/color/model/pcaBasis').shape)
    data.get('/color/model/pcaBasis').read_direct(texEvec)
    texEval = np.empty(data.get('/color/model/pcaVariance').shape)
    data.get('/color/model/pcaVariance').read_direct(texEval)
    
    # Triangle face indices
    face = np.empty(data.get('/shape/representer/cells').shape, dtype = 'int')
    data.get('/shape/representer/cells').read_direct(face)
    
    # Find vertex indices corresponding to the 40 given landmark vertices
    points = np.empty(data.get('/shape/representer/points').shape)
    data.get('/shape/representer/points').read_direct(points)
    
    with open(fNameLandmarks, 'r') as fd:
        landmark = []
        for line in fd:
            landmark.append([x for x in line.split(' ')])
    
    landmark = np.array(landmark)
    landmarkName = landmark[:, 0].tolist()
    landmark = landmark[:, 2:].astype('float')
    
    NN = NearestNeighbors(n_neighbors = 1, metric = 'l2')
    NN.fit(points.T)
    landmarkInd = NN.kneighbors(landmark)[1].squeeze()
    
    # Reshape to be compatible with fitting code
    numVertices = idMean.size // 3
    idMean = idMean.reshape((3, numVertices), order = 'F')
    idEvec = idEvec.reshape((3, numVertices, 199), order = 'F')
    expMean = expMean.reshape((3, numVertices), order = 'F')
    expEvec = expEvec.reshape((3, numVertices, 100), order = 'F')
    texMean = texMean.reshape((3, numVertices), order = 'F')
    texEvec = texEvec.reshape((3, numVertices, 199), order = 'F')
    face = face.T + 1
    
    # Save into an .npz uncompressed file
    np.savez('./models/bfm2017', face = face, idMean = idMean, idEvec = idEvec, idEval = idEval, expMean = expMean, expEvec = expEvec, expEval = expEval, texMean = texMean, texEvec = texEvec, texEval = texEval, landmark = landmark, landmarkInd = landmarkInd, landmarkName = landmarkName, numVertices = numVertices)
    
def generateModels(dirName, saveDirName = './'):
    """
    Generate eigenmodels of face meshes for (1) the neutral face and (2) the expressions. Save eigenvectors, eigenvalues, and means into .npy arrays.
    """
    if not dirName.endswith('/'):
        dirName += '/'
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    # Neutral face
    print('Loading neutral faces')
    vNeu = importObj(dirName, shape = 0, dataToImport = ['v'])[0]
    vNeu = np.reshape(vNeu, (150, vNeu.shape[1]*3))
    start = clock()
    evalNeu, evecNeu, meanNeu = PCA(vNeu)
    print(clock() - start)
    
    np.save(saveDirName + 'idEval', evalNeu)
    np.save(saveDirName + 'idEvec', evecNeu)
    np.save(saveDirName + 'idMean', meanNeu)
    
    # Expressions (from the 46 expression blendshapes)
    vExp = np.empty((150*46, vNeu.shape[1]))
    for s in range(46):
        print('Loading expression %d' % (s+1))
        temp = importObj(dirName, shape = s+1, dataToImport = ['v'], pose = 47)[0]
        # Subtract the neutral shape from the expression shape for each test subject
        vExp[s*150: (s+1)*150, :] = np.reshape(temp, (150, vNeu.shape[1])) - vNeu
    
    start = clock()
    evalExp, evecExp = PCA(vExp, numPC = 76)[:2]
    print(clock() - start)
    
    np.save(saveDirName + 'expEval', evalExp)
    np.save(saveDirName + 'expEvec', evecExp)

def exportObj(v, c = None, vt = None, f = None, fNameIn = None, fNameOut = 'test.obj'):
    """
    Write vertices to an .obj file.
    """
    # Make sure x, y, z vertex coordinates are along the columns
    if (c is None) and (v.shape[1] != 3):
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
                for i in range(v.shape[0]):
                    fo.write('v ' + '{:.6f}'.format(v[i, 0]) + ' ' + '{:.6f}'.format(v[i, 1]) + ' ' + '{:.6f}'.format(v[i, 2]) + ' ' + '{:.6f}'.format(c[i, 0]) + ' ' + '{:.6f}'.format(c[i, 1]) + ' ' + '{:.6f}'.format(c[i, 2]) + '\n')
            
            if vt is not None:
                for text in vt:
                    fo.write('vt ' + '{:.6f}'.format(text[0]) + ' ' + '{:.6f}'.format(text[1]) + '\n')
            
            if f is not None:
                if f.shape[1] == 4:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n')
                elif f.shape[1] == 3:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')

def saveLandmarks(dirName, saveDirName = './'):
    """
    Read the landmarks from the TrainingPoses and save them in a .npy file.
    """
    if not dirName.endswith('/'):
        dirName += '/'
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    landmarks = np.empty((20, 150, 74, 2))
    for pose in range(20):
        for tester in range(150):
            fName = 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.land'
            with open(dirName + fName, 'r') as fd:
                next(fd)
                landmarks[pose, tester, :, :] = np.array([[float(num) for num in line.split(' ')] for line in fd])
    
    np.save(saveDirName + 'landmarksTrainingPoses', landmarks)

def saveDepthMaps(dirName, saveDirName = './'):
    """
    Read the depth information from the Kinect .poses files and save them in a .npy file.
    """
    if not dirName.endswith('/'):
        dirName += '/'
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    depth = np.empty((20, 150, 480, 640), dtype = 'uint16')
    for pose in range(20):
        for tester in range(150):
            fName = 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.poses'
            with open(dirName + fName, 'rb') as fd:
                # First 4 bytes contain the frame number, and the next 640*480*3 contain the RGB data, so skip these
                fd.seek(4 + 1*640*480*3)
                
                # Each depth map value is 2 bytes (short)
                d = fd.read(2*640*480)
            
            # Convert bytes to int
            depth[pose, tester, :, :] = np.array([int.from_bytes(bytes([x, y]), byteorder = 'little') for x, y in zip(d[0::2], d[1::2])]).reshape((480, 640))
            
    np.save(saveDirName + 'depthMaps', depth)

def saveMasks(dirName, saveDirName = './masks/', mask = 'faceMask.obj', poses = 20):
    """
    Loop through the original 3D head models in the directory defined by dirName and extract the facial area defined by mask .obj file, saving the facial 3D models of the original 3D heads into new .obj files in the directory defined by saveDirName.
    """
    if not saveDirName.endswith('/'):
        saveDirName += '/'
        
    # Loop through the poses/shapes
    for shape in range(poses):
        # The reference mask defining the facial region is based off of the first tester in pose/shape 0
        if shape == 0:
            v = importObj(dirName, shape, dataToImport = ['v'], pose = poses)[0]
            faceMask = importObj(mask, shape = 0)[0]
            idx = np.zeros(faceMask.shape[0], dtype = int)
            for i, vertex in enumerate(faceMask):
                idx[i] = np.where(np.equal(vertex, v[0, :, :]).all(axis = 1))[0]
        else:
            v = importObj(dirName, shape, dataToImport = ['v'], pose = poses)[0]
        
        v = v[:, idx, :]
        
        for tester in range(150):
            print('Processing shape %d for tester %d' % (shape+1, tester+1))
            if poses == 47:
                if not os.path.exists(saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/'):
                    os.makedirs(saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/')
                fName = saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/shape_' + str(shape) + '.obj'
            
            if poses == 20:
                if not os.path.exists(saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/'):
                    os.makedirs(saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/')
                fName = saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(shape) + '.obj'
            
            exportObj(v[tester, :, :], fNameIn = mask, fNameOut = fName)
            
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

def gaussNewton(P, target, targetLandmarks, sourceLandmarkInds, NN):
    """
    Energy function to be minimized for fitting.
    """
    # Load geometric models
    idEvec = np.load('./models/idEvec.npy')
    idEval = np.load('./models/idEval.npy')
    idEvec = np.reshape(idEvec, (3, idEvec.shape[0]//3, 80), order = 'F')
    idMean = np.load('./models/idMean.npy')
    idMean = np.reshape(idMean, (3, idMean.size//3), order = 'F')
    expEvec = np.load('./models/expEvec.npy')
    expEval = np.load('./models/expEval.npy')
    expEvec = np.reshape(expEvec, (3, expEvec.shape[0]//3, 76), order = 'F')
    
    # Shape eigenvector coefficients
    alpha = P[: idEvec.shape[2]]
    delta = P[idEvec.shape[2]: idEvec.shape[2] + expEvec.shape[2]]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = P[idEvec.shape[2] + expEvec.shape[2]:][:3]
    R = rotMat2angle(angles)
    t = P[idEvec.shape[2] + expEvec.shape[2]:][3: 6]
    s = P[idEvec.shape[2] + expEvec.shape[2]:][6]
    
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # The eigenmodel, before rigid transformation and scaling
    model = idMean + np.tensordot(idEvec, alpha, axes = 1) + np.tensordot(expEvec, delta, axes = 1)
    
    # After rigid transformation and scaling
    source = s*np.dot(R, model) + t[:, np.newaxis]
    
    # Find the nearest neighbors of the target to the source vertices
    start = clock()
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
    print('NN: %f' % (clock() - start))
    
    start = clock()
    # Calculate resisduals
    rVert = targetNN - source
    rLand = targetLandmarks - source[:, sourceLandmarkInds]
    rAlpha = alpha ** 2 / idEval
    rDelta = delta ** 2 / expEval
    r = np.r_[rVert.flatten('F'), rLand.flatten('F'), rAlpha, rDelta]
    
    # Calculate Jacobian
    drV_dalpha = -s*np.tensordot(R, idEvec, axes = 1)
    drV_ddelta = -s*np.tensordot(R, expEvec, axes = 1)
    drV_dpsi = -s*np.dot(dR_dpsi(angles), model)
    drV_dtheta = -s*np.dot(dR_dtheta(angles), model)
    drV_dphi = -s*np.dot(dR_dphi(angles), model)
    drV_dt = -np.tile(np.eye(3), [source.shape[1], 1])
    drV_ds = -np.dot(R, model)
    
    drL_dalpha = drV_dalpha[:, sourceLandmarkInds, :]
    drL_ddelta = drV_ddelta[:, sourceLandmarkInds, :]
    drL_dpsi = drV_dpsi[:, sourceLandmarkInds]
    drL_dtheta = drV_dtheta[:, sourceLandmarkInds]
    drL_dphi = drV_dphi[:, sourceLandmarkInds]
    drL_dt = -np.tile(np.eye(3), [sourceLandmarkInds.size, 1])
    drL_ds = drV_ds[:, sourceLandmarkInds]
    
    drR_dalpha = np.diag(2*alpha / idEval)
    drR_ddelta = np.diag(2*delta / expEval)
    
    J = np.r_[np.c_[drV_dalpha.reshape((np.prod(source.shape), alpha.size), order = 'F'), drV_ddelta.reshape((np.prod(source.shape), delta.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')], np.c_[drL_dalpha.reshape((np.prod(targetLandmarks.shape), alpha.size), order = 'F'), drL_ddelta.reshape((np.prod(targetLandmarks.shape), delta.size), order = 'F'), drL_dpsi.flatten('F'), drL_dtheta.flatten('F'), drL_dphi.flatten('F'), drL_dt, drL_ds.flatten('F')], np.c_[drR_dalpha, np.zeros((alpha.size, delta.size + 7))], np.c_[np.zeros((delta.size, alpha.size)), drR_ddelta, np.zeros((delta.size, 7))]]
    
    # Parameter update (Gauss-Newton)
    dP = -np.linalg.inv(np.dot(J.T, J)).dot(J.T).dot(r)
    
    print('GN: %f' % (clock() - start))
    
    # Calculate costs
    costVert = np.linalg.norm(rVert, axis = 0).sum()
    costLand = np.linalg.norm(targetLandmarks - source[:, sourceLandmarkInds], axis = 0).sum()
    
    totCost = costVert + costLand + np.sum(rAlpha) + np.sum(rDelta)
    
    return dP, totCost, target

def generateFace(P):
    """
    Generate vertices based off of eigenmodel and vector of parameters
    """
    # Load geometric models
    idEvec = np.load('./models/idEvec.npy')
    idEvec = np.reshape(idEvec, (3, idEvec.shape[0]//3, 80), order = 'F')
    idMean = np.load('./models/idMean.npy')
    idMean = np.reshape(idMean, (3, idMean.size//3), order = 'F')
    expEvec = np.load('./models/expEvec.npy')
    expEvec = np.reshape(expEvec, (3, expEvec.shape[0]//3, 76), order = 'F')
    
    # Shape eigenvector coefficients
    idCoef = P[: idEvec.shape[2]]
    expCoef = P[idEvec.shape[2]: idEvec.shape[2] + expEvec.shape[2]]
    
    # Rotation Euler angles, translation vector, scaling factor
    R = rotMat2angle(P[idEvec.shape[2] + expEvec.shape[2]:][:3])
    t = P[idEvec.shape[2] + expEvec.shape[2]:][3: 6]
    s = P[idEvec.shape[2] + expEvec.shape[2]:][6]
    
    # The eigenmodel, before rigid transformation and scaling
    model = idMean + np.tensordot(idEvec, idCoef, axes = 1) + np.tensordot(expEvec, expCoef, axes = 1)
    
    # After rigid transformation and scaling
    return s*np.dot(R, model) + t[:, np.newaxis]

dirName = '/home/nguyen/Documents/Data/facewarehouse/FaceWarehouse_Data_0/'
#saveDirName = '/home/nguyen/Documents/Data/facewarehouse/Models/'
        
#vt, f = importObj(dirName, shape = 0, dataToImport = ['vt', 'f'])
#plt.scatter(vt[:, 0], vt[:, 1], s = 1)

# Use a template mask to extract the facial region from the 3D models
#saveMasks(dirName, saveDirName = './masks2v2/', mask = 'mask2v2.obj', poses = 20)

# Subdivision
#v, f = importObj('./masks2v2/', shape = 0, dataToImport = ['v', 'f'])
#f = f[0, :, :] - 1
#vNew, fNew = subdivide(v, f)

## Load identity (neutral face) eigenmodel
#idEvec = np.load('./models/idEvec.npy')
#idEval = np.load('./models/idEval.npy')
#idMean = np.load('./models/idMean.npy')
##idEvec = np.reshape(idEvec, (idMean.size//3, 3, 80))
#idEvec = np.reshape(idEvec, (3, idEvec.shape[0]//3, 80), order = 'F')
##idMean = np.reshape(idMean, (idMean.size//3, 3))
#idMean = np.reshape(idMean, (3, idMean.size//3), order = 'F')
#
## Load expression eigenmodel
#expEvec = np.load('./models/expEvec.npy')
#expEval = np.load('./models/expEval.npy')
#expEvec = np.reshape(expEvec, (3, expEvec.shape[0]//3, 76), order = 'F')

#processBFM2017('/home/nguyen/Documents/Data/model2017-1_face12_nomouth.h5', '/home/nguyen/Documents/Data/bfm2017landmarks.txt')
m = Bunch(np.load('./models/bfm2017.npz'))

bfm2fw = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 21, 22, 23, 24, 25, 27, 29, 30, 32, 33, 34, 37, 38, 39])
fw2bfm = np.array([7, 59, 55, 62, 49, 39, 65, 34, 15, 18, 33, 31, 32, 52, 50, 45, 41, 40, 30, 21, 24, 29, 27, 28, 46, 48, 44, 37, 38])

idCoef = np.zeros(m.idEval.shape)
idCoef[0] = 1
expCoef = np.zeros(m.expEval.shape)
expCoef[0] = 1
texCoef = np.zeros(m.texEval.shape)
texCoef[0] = 1
#source = m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)
#sourceColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
#exportObj(source.T, c = sourceColor.T, f = m.face)
#exportObj(source[:, m.landmarkInd].T, fNameOut = 'landmarksBFM')

#data = loadmat('/home/nguyen/Downloads/PublicMM1/01_MorphableModel.mat', variable_names = ['shapeEV', 'shapeMU', 'shapePC', 'texEV', 'texMU', 'texPC', 'tl'])
#numVertices = data['shapeMU'].size // 3
#idMean = data['shapeMU'].reshape((numVertices, 3))
#texMean = data['texMU'].reshape((numVertices, 3))
#exportObj(idMean, c = texMean, f = data['tl'])

## Load 3D vertex indices of landmarks and find their vertices on the neutral face
#landmarkInds3D = np.load('./data/landmarkInds3D.npy')
landmarkInds3D = m.landmarkInd[bfm2fw]

pose = 7
tester = 7

# Gather 2D landmarks that correspond to manually chosen 3D landmarks
#landmarkInds2D = np.array([0, 1, 4, 7, 10, 13, 14, 27, 29, 31, 33, 46, 49, 52, 55, 65])
landmarkInds2D = fw2bfm
landmarks = np.load('./data/landmarks2D.npy')[pose, tester, landmarkInds2D, :]
landmarkPixelInd = (landmarks * np.array([639, 479])).astype(int)
landmarkPixelInd[:, 1] = 479 - landmarkPixelInd[:, 1]

### Get target 3D coordinates of depth maps at the 16 landmark locations
#depth = np.load('./data/depthMaps.npy')[pose, tester, :, :]
#
#targetLandmark, nonZeroDepth = perspectiveTransformKinect(np.c_[landmarkPixelInd[:, 0], landmarkPixelInd[:, 1], depth[landmarkPixelInd[:, 1], landmarkPixelInd[:, 0]]])
#
##target = perspectiveTransformKinect(np.c_[np.tile(np.arange(640), 480), np.repeat(np.arange(480), 640), depth.flatten()])[0]
#
##plt.imshow(depth[pose, tester, :, :])
#
##plt.figure()
##plt.scatter(targetLandmark[:, 0], targetLandmark[:, 1], s=1)
##plt.figure()
##plt.scatter(target[:, 0], target[:, 1], s=1)
#
## Initialize parameters
#idCoef = np.zeros(idEval.shape)
#idCoef[0] = 1
#expCoef = np.zeros(expEval.shape)
#expCoef[0] = 1

# Do initial registration between the 16 corresponding landmarks on the depth map and the face model
#source = idMean + np.tensordot(idEvec, idCoef, axes = 1) + np.tensordot(expEvec, expCoef, axes = 1)
#rho = initialRegistration(source[:, landmarkInds3D[nonZeroDepth]], targetLandmark)
#Rd = rotMat2angle(rho[:3])

#P = np.r_[idCoef, expCoef, rho]
#source = generateFace(P)

# Nearest neighbors fitting from scikit-learn to form correspondence between target vertices and source vertices during optimization
#NN = NearestNeighbors(n_neighbors = 1, metric = 'l1')
#NN.fit(target)
#NNparams = NN.get_params()
#
#cost = np.empty((100))
#for i in range(100):
#    print('Iteration %d' % i)
#    dP, cost[i], target = gaussNewton(P, target, targetLandmark.T, landmarkInds3D[nonZeroDepth], NN)
#    
#    P += dP
#
#source = generateFace(P)

#hist, bins = np.histogram(np.array(distances), bins=50)
#width = 0.7 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.show()

#exportObj(target, fNameOut = 'target.obj') 
#exportObj(targetLandmark, fNameOut = 'targetLandmark.obj')
#exportObj(source, fNameIn = './mask2v2.obj', fNameOut = 'source.obj')

# Direct linear transform / "Gold Standard Algorithm"
# Normalize landmark coordinates; preconditioning
numLandmarks = landmarkInds2D.size
c2D = np.mean(landmarkPixelInd, axis = 0)
uvCentered = landmarkPixelInd - c2D
s2D = np.linalg.norm(uvCentered, axis = 1).mean()
x = np.c_[uvCentered / s2D * np.sqrt(2), np.ones(numLandmarks)]

c3D = np.mean(m.idMean[:, landmarkInds3D].T, axis = 0)
xyzCentered = m.idMean[:, landmarkInds3D].T - c3D
s3D = np.linalg.norm(xyzCentered, axis = 1).mean()
X = np.c_[xyzCentered / s3D * np.sqrt(3), np.ones(numLandmarks)]

Tinv = np.array([[s2D, 0, c2D[0]], [0, s2D, c2D[1]], [0, 0, 1]])
U = np.linalg.inv([[s3D, 0, 0, c3D[0]], [0, s3D, 0, c3D[1]], [0, 0, s3D, c3D[2]], [0, 0, 0, 1]])

# Create matrix for homogenous system of equations to solve for camera matrix
A = np.empty((2*numLandmarks, 12))
for row in range(landmarkInds2D.size):
    A[row*2, :] = np.r_[X[row, :], 0, 0, 0, 0, -x[row, 0]*X[row, :]]
    A[row*2 + 1, :] = np.r_[0, 0, 0, 0, -X[row, :], x[row, 1]*X[row, :]]

# Take the SVD and take the last row of V' as the homogenous solution
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
P = np.dot(Tinv, Pgold.x.reshape(3, 4)).dot(U)

# Even more minimization with projection matrix to get initial shape parameters
def camWithShape(param, x, Xmu, Xid, Xexp, idEval, expEval):
    """
    Minimize L2-norm of landmark fitting residuals and regularization terms for shape parameters
    """
    P = param[:12]
    P = P.reshape((3, 4))
    idCoef = param[12: 12 + idEval.size]
    expCoef = param[12 + idEval.size:]
    
    X = np.dot(P, Xmu + np.tensordot(Xid, idCoef, axes = 1) + np.tensordot(Xexp, expCoef, axes = 1)).T
    residuals = x - X
    
    regularize = np.sum(idCoef ** 2 / idEval) + np.sum(expCoef ** 2 / expEval)
    
    return np.linalg.norm(residuals, axis = 1).sum() + regularize

Lmu = np.r_[m.idMean[:, landmarkInds3D], np.ones((1, landmarkInds3D.size))]
Lid = np.concatenate((m.idEvec[:, landmarkInds3D, :], np.zeros((1, landmarkInds3D.size, idCoef.size))), axis = 0)
Lexp = np.concatenate((m.expEvec[:, landmarkInds3D, :], np.zeros((1, landmarkInds3D.size, expCoef.size))), axis = 0)
x = np.c_[landmarkPixelInd, np.ones(landmarkInds3D.size)]

param = minimize(camWithShape, np.r_[P.flatten(), idCoef, expCoef], args = (x, Lmu, Lid, Lexp, m.idEval, m.expEval))

# Separate variates in parameter vector
P = param.x[:12].reshape((3, 4))
idCoef = param.x[12: 12 + idCoef.size]
expCoef = param.x[12 + idCoef.size:]

# Get inner parameters from projection matrix via RQ decomposition
K, R = rq(P[:, :3])
t = np.linalg.inv(K).dot(P[:, -1])

# Project 3D model into 2D plane
fitting = K[:2, :].dot(R.dot(m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)) + t[:, np.newaxis])

# Plot the projected 3D model on top of the input RGB image
fName = dirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.png'
img = mpimg.imread(fName)
plt.imshow(img)
plt.hold(True)
plt.scatter(fitting[0, :], fitting[1, :], s = 0.1, c = 'g')
plt.hold(True)
plt.scatter(fitting[0, landmarkInds3D], fitting[1, landmarkInds3D], s = 2, c = 'b')
plt.hold(True)
plt.scatter(landmarkPixelInd[:, 0], landmarkPixelInd[:, 1], s = 2, c = 'r')

"""
Some stuff for spherical harmonic illumination model that will be developed
"""
## Get triangle faces
#f = importObj('./masks2v2/', shape = 0, dataToImport = ['f'])[0][0, :, :] - 1
#f = np.r_[f[:, [0, 1, 2]], f[:, [0, 2, 3]]]
#
## Calculate normals of vertex points
#idMean = np.load('./models/idMean.npy')
#idMean = np.reshape(idMean, (idMean.size//3, 3))
#
#fNorm = np.cross(idMean[f[:, 0], :] - idMean[f[:, 1], :], idMean[f[:, 0], :] - idMean[f[:, 2], :])
#
#vertex2face = [np.where(np.isin(f, vertexInd).any(axis = 1))[0].tolist() for vertexInd in range(idMean.shape[0])]
#
#vNorm = np.array([np.sum(fNorm[faces, :], axis = 0) for faces in vertex2face])
#
#vNorm = normalize(vNorm)
#
#def SHbasis(v, n):
#    """
#    SH basis functions                               lm
#        1/np.sqrt(4*np.pi)                          Y00
#        np.sqrt(3/(4*np.pi))*nz                     Y10
#        np.sqrt(3/(4*np.pi))*nx                     Y11e
#        np.sqrt(3/(4*np.pi))*ny                     Y11o
#        1/2*np.sqrt(5/(4*np.pi))*(3*nz^2 - 1)       Y20
#        3*np.sqrt(5/(12*np.pi))*nx*nz               Y21e
#        3*np.sqrt(5/(12*np.pi))*ny*nz               Y21o
#        3/2*np.sqrt(5/(12*np.pi))*(nx^2 - ny^2)     Y22e
#        3*np.sqrt(5/(12*np.pi))*nx*ny               Y22o
#    
#    For a sphere, the Lambertian kernel has most of its energy in the first three bands of the spherical harmonic basis functions (above). This implies that Lambertian reflectance functions can be well-approximated by these low-order SH bases.
#    """
#    # Lambertian kernel coefficients indexed by l
##    k = np.array([np.sqrt(np.pi)/2, np.sqrt(np.pi/3), np.sqrt(5*np.pi)/8])
#    
#    # Normalization factor for SH
##    norm = np.array([np.sqrt((4*np.pi)), np.sqrt((4*np.pi)/3), np.sqrt((4*np.pi)/5)])
#    
#    b = np.empty((9, v.shape))
#    b[0, :, :] = 1/np.sqrt(4*np.pi) * v
#    b[1, :, :] = np.sqrt(3/(4*np.pi)) * n[:, 2] * v
#    b[2, :, :] = np.sqrt(3/(4*np.pi)) * n[:, 0] * v
#    b[3, :, :] = np.sqrt(3/(4*np.pi)) * n[:, 1] * v
#    b[4, :, :] = 1/2*np.sqrt(5/(4*np.pi)) * (3*np.power(n[:, 2]) - 1) * v
#    b[5, :, :] = 3*np.sqrt(5/(12*np.pi)) * n[:, 0] * n[:, 2] * v
#    b[6, :, :] = 3*np.sqrt(5/(12*np.pi)) * n[:, 1] * n[:, 2] * v
#    b[7, :, :] = 3/2*np.sqrt(5/(12*np.pi)) * (np.square(n[:, 0]) - np.square(n[:, 1])) * v
#    b[8, :, :] = 3*np.sqrt(5/(12*np.pi)) * n[:, 0] * n[:, 1] * v

# Load 3D landmarks and find their vertex indices
#v = importObj('./mask2v2.obj', shape = 0, dataToImport = ['v'])[0]
#landmarks3D = importObj('./landmarks.obj', shape = 0, dataToImport = ['v'])[0]
#landmarks3D = landmarks3D[[7, 4, 3, 2, 11, 12, 15, 5, 6, 13, 14, 0, 10, 9, 8, 1]]
#landmarkInds3D = [np.where(np.isin(v, landmark).any(axis = 1))[0][0] for landmark in landmarks3D]
#np.save('./landmarkInds3D', landmarkInds3D)

#fig, ax = plt.subplots()
#x = landmarks3D[:, 0]
#y = landmarks3D[:, 1]
#ax.scatter(x, y, s = 1, picker = True)
#fig.canvas.mpl_connect('pick_event', onpick3)

# Confirm that 16 2D landmarks are in correspondence across poses and testers
#pose = 0
#tester = 0
#landmarks = np.load('./data/landmarks2D.npy')[pose, tester, :, :]
#landmarkPixelInd = (landmarks * np.array([639, 479])).astype(int)
#landmarkPixelInd[:, 1] = 479 - landmarkPixelInd[:, 1]
#fName = dirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.png'
#img = mpimg.imread(fName)

#bfm2fw = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 21, 22, 23, 24, 25, 27, 29, 30, 32, 33, 34, 37, 38, 39]
#fw2bfm = [7, 59, 55, 62, 49, 39, 65, 34, 15, 18, 33, 31, 32, 52, 50, 45, 41, 40, 30, 21, 24, 29, 27, 28, 46, 48, 44, 37, 38]

#x = landmarks16[pose, tester, :, 0]
#y = landmarks16[pose, tester, :, 1]
#plt.scatter(x*640, (1-y)*480, s = 1)
#x = landmarkPixelInd[fw2bfm, 0]
#y = landmarkPixelInd[fw2bfm, 1]
#x = landmarkPixelInd[:, 0]
#y = landmarkPixelInd[:, 1]
#plt.scatter(x, y, s = 1, c = 'r')

#fig, ax = plt.subplots()
#plt.imshow(img)
#plt.hold(True)
#plt.imshow(depth[pose, tester, :, :].astype(float))
#plt.hold(True)
#x = landmarks16NN[pose, tester, :, 0]
#y = landmarks16NN[pose, tester, :, 1]
#ax.scatter(x, y, s = 1, c = 'r', picker = True)
#fig.canvas.mpl_connect('pick_event', onpick3)