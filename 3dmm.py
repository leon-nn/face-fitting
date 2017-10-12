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
import scipy.interpolate as intp
from time import clock
import os
from collections import defaultdict
from itertools import chain, compress

def importObj(dirName, shape = 0, dataToImport = ['v', 'vt', 'f']):
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
        if not singleFile:
#            fName = dirName + 'Tester_' + str(i+1) + '/Blendshape/shape_' + str(shape) + '.obj'
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
    textV = importObj(dirName, pose, dataToImport = ['vt'])[0]
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
    data -= mean
    
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
    for faceInd, edges in enumerate(face2edge):
        for edge in edges:
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
        print(tester)
        # Face points: the mean of the vertices on a face
        facePt = np.array([np.mean(v[tester, vertexInd, :], axis = 0) for vertexInd in f])
        
        # Edge points
        edgePt = np.empty((len(edges), 3))
        for i, edge in enumerate(edges):
            # If an edge is only associated with one face, then it is on a border of the 3D model. The edge point is thus the midpoint of the vertices defining the edge.
            if len(edge2face[edge]) == 1:
                edgePt[i, :] = np.mean(v[tester, list(edge), :], axis = 0)
            
            # Else, the edge point is the mean of (1) the face point of the two faces adjacent to the edge and (2) the midpoint of the vertices defining the edge.
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
            print(np.r_[facePt, edgePt, newPt].shape)
        vNew[tester, :, :] = np.r_[facePt, edgePt, newPt]
    
    # Form the new faces
    fNew = np.c_[f.flatten() + facePt.shape[0] + edgePt.shape[0], edgeInd + facePt.shape[0], np.repeat(np.arange(facePt.shape[0]), 4), edgeInd.reshape((edgeInd.shape[0]//4, 4))[:, [3, 0, 1, 2]].flatten() + facePt.shape[0]] + 1
    
    return vNew, fNew

@jit
def processShapes(dirName, saveDirName):
    """
    Go through all 150 testers in the dirName directory and find the mean shape, eigenvalues, and eigenvectors for each of the 47 expressions. Save these into .npy files in the saveDirName directory.
    """
    for shape in range(47):
        # Read the 'v' coordinates from an .obj file into a NumPy array
        geoV = importObj(dirName, shape, dataToImport = ['v'])[0]
        
        # Reshape the coordinates so that observations are along the rows
        geoV = np.reshape(geoV, (150, 11510*3))
        
        # Find the eigenvalues, eigenvectors, and mean shape of the vertices
        eigValS, eigVecS, meanS = PCA(geoV)
        
        # Save into .npy format
        np.save(saveDirName + 'shape' + str(shape) + 'eigValS', eigValS)
        np.save(saveDirName + 'shape' + str(shape) + 'eigVecS', eigVecS)
        np.save(saveDirName + 'shape' + str(shape) + 'meanS', meanS)

@jit
def processTextures(dirName, saveDirName):
    """
    Go through all 150 testers in the dirName directory and find the mean texture, eigenvalues, and eigenvectors for each of the 20 poses. Save these into .npy files in the saveDirName directory.
    """
    for pose in range(20):
        # Get the RGB values referenced by the texture vertices
        text = getTexture(dirName, pose)
        
        # Reshape the coordinates so that observations are along the rows
        text = np.reshape(text, (150, 11558*3))
        
        # Find the eigenvalues, eigenvectors, and mean shape of the vertices
        eigVal, eigVec, mean = PCA(text)
        
        # Save into .npy format
        np.save(saveDirName + 'pose' + str(pose) + 'eigValT', eigVal)
        np.save(saveDirName + 'pose' + str(pose) + 'eigVecT', eigVec)
        np.save(saveDirName + 'pose' + str(pose) + 'meanT', mean)

def writeShape(mean, eigVec, a = None, fNameOut = 'test.obj'):
    """
    Given a mean face shape, the eigenvectors, and eigenvector parameters a, write a new face shape out to an .obj file.
    """
    # Template .obj file to read from
    fNameIn = 'objTemplate.obj'
    
    # Add the .obj extension if necessary
    if not fNameOut.endswith('.obj'):
        fNameOut += '.obj'
    
    # Find the dimension N of the data and the number of eigenvectors M
    N, M = eigVec.shape
    
    # Make a random set of eigenvector parameters if not user-provided
    if a is None:
        a = np.random.rand(M)
    
    # Construct the new face shape S
    S = mean + np.dot(eigVec, a)
    
    # Reshape so that each vertex is in its own row
    S = np.reshape(S,(N//3, 3))
    
    with open(fNameIn, 'r') as fi:
        with open(fNameOut, 'w') as fo:
            # Initialize counter for the vertex index
            vertexInd = 0
            
            # Iterate through the lines in the template file
            for line in fi:
                # If a line starts with 'v ', then we replace it with the vertex coordinate with the index corresponding to the line number
                if line.startswith('v '):
                    # Keep 6 digits after the decimal to follow the formatting in the template file
                    fo.write('v ' + '{:.6f}'.format(S[vertexInd, 0]) + ' ' + '{:.6f}'.format(S[vertexInd, 1]) + ' ' + '{:.6f}'.format(S[vertexInd, 2]) + '\n')
                    
                    # Increment the vertex index counter
                    vertexInd += 1
                    
                # Else, the line must end with 'vt' or 'f', so we copy from the template file because those coordinates are in correspondence
                else:
                    fo.write(line)

def exportObj(v, vt = None, f = None, fNameIn = None, fNameOut = 'test.obj'):
    """
    Write vertices to an .obj file.
    """
    
    # Add the .obj extension if necessary
    if not fNameOut.endswith('.obj'):
        fNameOut += '.obj'
    
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
                    
                    else:
                        f = [int(ind) for ind in re.split('/| ', line[2:])]
                        fo.write('f ' + str(f[0]) + '/' + str(f[1]) + '/0 ' + str(f[3]) + '/' + str(f[4]) + '/0 ' + str(f[6]) + '/' + str(f[7]) + '/0 ' + str(f[9]) + '/' + str(f[10]) + '/0\n')
        
    else:
        with open(fNameOut, 'w') as fo:
            for vertex in v:
                fo.write('v ' + '{:.6f}'.format(vertex[0]) + ' ' + '{:.6f}'.format(vertex[1]) + ' ' + '{:.6f}'.format(vertex[2]) + '\n')
            
            if vt is not None:
                for text in vt:
                    fo.write('vt ' + '{:.6f}'.format(text[0]) + ' ' + '{:.6f}'.format(text[1]) + '\n')
            
            if f is not None:
                if vt is None:
                    for face in f:
                        fo.write('f ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n')
                        
def saveMasks(dirName, saveDirName = './masks', mask = 'faceMask.obj'):
    """
    Loop through the original 3D head models in the directory defined by dirName and extract the facial area defined by mask .obj file, saving the facial 3D models of the original 3D heads into new .obj files in the directory defined by saveDirName.
    """
    # Loop through the poses/shapes
    for shape in range(20):
        # The reference mask defining the facial region is based off of the first tester in pose/shape 0
        if shape == 0:
            v = importObj(dirName, shape, dataToImport = ['v'])[0]
            faceMask = importObj(mask, shape = 0)[0]
            idx = np.zeros(faceMask.shape[0], dtype = int)
            for i, vertex in enumerate(faceMask):
                idx[i] = np.where(np.equal(vertex, v[0, :, :]).all(axis = 1))[0]
        else:
            v = importObj(dirName, shape, dataToImport = ['v'])[0]
        
        v = v[:, idx, :]
        
        for tester in range(150):
    #        if not os.path.exists(saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/'):
    #            os.makedirs(saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/')
    #            
    #        fName = saveDirName + 'Tester_' + str(tester+1) + '/Blendshape/shape_' + str(shape) + '.obj'
            
            if not os.path.exists(saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/'):
                os.makedirs(saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/')
            fName = saveDirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(shape) + '.obj'
            
            exportObj(v[tester, :, :], fNameIn = 'mask.obj', fNameOut = fName)
            
            
dirName = '/home/nguyen/Documents/Data/facewarehouse/FaceWarehouse_Data_0/'
saveDirName = '/home/nguyen/Documents/Data/facewarehouse/Models/'

#geoV = importObj(dirName, shape = 0)[2]
#geoV = np.reshape(geoV, (150, 11510*3))
#textV = np.reshape(textV, (150, 11558*2))

#start = clock()

#elapsed = (clock() - start)

#mean = np.load(saveDirName + 'shape0meanS.npy')
#eigVec = np.load(saveDirName + 'shape0eigVecS.npy')
#fNameOut = 'test.obj'
#writeShape(mean, eigVec, a = np.r_[0, 10, np.zeros(78)])
        
#vt, f = importObj(dirName, shape = 0, dataToImport = ['vt', 'f'])
#plt.scatter(vt[:, 0], vt[:, 1], s = 1)

#numTesters = 150
#land = np.empty((150, 74, 2))
#pose = 0
#for i in range(numTesters):
#    fName = dirName + 'Tester_' + str(i + 1) + '/TrainingPose/pose_' + str(pose) + '.land'
#    with open(fName, 'r') as fd:
#        l = []
#        next(fd)
#        for line in fd:
#            l.append([float(coord) for coord in line.split(' ')])
#    land[i, :, :] = np.array(l)
#plt.scatter(l[:, 0], l[:, 1], s = 1)
#axes = plt.gca()
#axes.set_xlim([0, 1])
#axes.set_ylim([0, 1])

v, f = importObj('./masks/', shape = 0, dataToImport = ['v', 'f'])
f = f[1, :, :] - 1

vNew, fNew = subdivide(v[0, :, :], f)
#vNew2, fNew2 = subdivide(vNew, fNew)

#exportObj(facePt, fNameOut = 'subdivFace.obj')
#exportObj(edgePt, fNameOut = 'subdivEdge.obj')
#exportObj(newPt, fNameOut = 'subdivNew.obj')
#exportObj(vNew, f = fNew, fNameOut = 'subdivTest.obj')

"""
Some stuff for spherical harmonic illumination model that will be developed
"""
#k = np.array([np.sqrt(np.pi)/2, np.sqrt(np.pi/3), 2*np.pi*np.sqrt(5/(4*np.pi))/8])
#norm = np.array([np.sqrt((4*np.pi)), np.sqrt((4*np.pi)/3), np.sqrt((4*np.pi)/5)])
#Y = np.array([
#        1/np.sqrt(4*np.pi),                     # Y00
#        np.sqrt(3/(4*np.pi))*z,                 # Y10
#        np.sqrt(3/(4*np.pi))*x,                 # Y11e
#        np.sqrt(3/(4*np.pi))*y,                 # Y11o
#        1/2*np.sqrt(5/(4*np.pi))*(3*z^2 - 1),   # Y20
#        3*np.sqrt(5/(12*np.pi))*x*z,            # Y21e
#        3*np.sqrt(5/(12*np.pi))*y*z,            # Y21o
#        3/2*np.sqrt(5/(12*np.pi))*(x^2 - y^2),  # Y22e
#        3*np.sqrt(5/(12*np.pi))*x*y])           # Y22o