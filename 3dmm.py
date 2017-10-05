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

def importObj(dirName, shape = 0, dataToImport = ['v', 'vt', 'f']):
    """
    Return the geometric and texture vertices along with the quadrilaterials containing the geometric and texture indices for all 150 testers of FaceWarehouse for a certain shape/pose/expression. Input (1) a string for the directory name that contains the folders 'Tester_1' through 'Tester_150', (2) an int for the shape number, which is in the range [0, 46] (1 neutral + 46 expressions), and (3) a list containing strings to indicate what part of the .obj file to read ('v' = geometric vertices, 'vt' = texture vertices, 'f' = face quadrilaterals).
    """
    # Number of observations (people/testers) in the dataset
    numTesters = 150
    
    # Initialize array to store geometric vertices
    geoV = np.empty((numTesters, 11510, 3))
    
    # Initialize array to store UV coordinates that map to pixel locations in the source RGB image
    textV = np.empty((11558, 2))
    
    # Initialize arrays to store quadrilaterials containing the geometric (1) and texture (2) indices of the vertices
    quad = np.empty((2, 11400, 4), dtype = 'int')
    
    for i in range(numTesters):
        # Directory name that contains the folders for each Tester
        if not dirName.endswith('/'):
            dirName += '/'
        fName = dirName + 'Tester_' + str(i+1) + '/Blendshape/shape_' + str(shape) + '.obj'
#        fName = dirName + 'Tester_' + str(i+1) + '/TrainingPose/pose_' + str(shape) + '.obj'
        
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
        
        # Store the data for each shape
        if 'vt' in dataToImport and i == 0:
            textV[:, :] = np.array(vt)
        if 'f' in dataToImport and i == 0:
            quad[0, :, :] = np.array(f)[:, [0, 3, 6, 9]]
            quad[1, :, :] = np.array(f)[:, [1, 4, 7, 10]]
        if 'v' in dataToImport:
            geoV[i, :, :] = np.array(v)
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
#writeShape(mean, eigVec, a = np.ones(80))
#bpy.ops.import_scene.obj('test.obj')


        
#plt.scatter(textV[0, :, 0], textV[0, :, 1], s = 1)

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