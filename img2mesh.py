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

import openglRender
from mm import Bunch, onpick3, exportObj, generateFace, rotMat2angle, initialShapeCost2D, initialShapeGrad2D, estCamMat, splitCamMat, camWithShape, dR_dpsi, dR_dtheta, dR_dphi, calcNormals, shBasis, sh9, sph2cart
from time import clock
import glob, os, re, json
import numpy as np
from scipy.optimize import minimize, check_grad, least_squares, nnls
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import savefig

def calcZBuffer(vertexCoord):
    """
    Assumes that the nose will have smaller z values
    """
    # Transpose input if necessary so that dimensions are along the columns
    if vertexCoord.shape[0] == 3:
        vertexCoord = vertexCoord.T
    
    # Given an orthographically projected 3D mesh, convert the x and y vertex coordinates to integers to represent pixel coordinates
    vertex2pixel = vertexCoord[:, :2].astype(int)
    
    # Find the unique pixel coordinates from above, the first vertex they map to, and the count of vertices that map to each pixel coordinate
    pixelCoord, pixel2vertexInd, pixelCounts = np.unique(vertex2pixel, return_index = True, return_counts = True, axis = 0)
    
    # Initialize the z-buffer to have as many elements as there are unique pixel coordinates
    zBuffer = np.empty(pixel2vertexInd.size, dtype = int)
    
    # Loop through each unique pixel coordinate...
    for i in range(pixel2vertexInd.size):
        # If a given pixel coordinate only has 1 vertex, then the z-buffer will represent that vertex
        if pixelCounts[i] == 1:
            zBuffer[i] = pixel2vertexInd[i]
        
        # If a given pixel coordinate has more than 1 vertex...
        else:
            # Find the indices for the vertices that map to the pixel coordinate
            candidateVertexInds = np.where((vertex2pixel[:, 0] == pixelCoord[i, 0]) & (vertex2pixel[:, 1] == pixelCoord[i, 1]))[0]
            
            # Of the vertices, see which one has the smallest z coordinate and assign that vertex to the pixel coordinate in the z-buffer
            zBuffer[i] = candidateVertexInds[np.argmin(fitting[2, candidateVertexInds])]
    
    return zBuffer, pixelCoord

def textureCost(texCoef, x, mask, m, w = (1, 1)):
    """
    Energy formulation for fitting texture
    """
    # Photo-consistency
    Xcol = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1)).T
    
    r = (Xcol - x).flatten()
    
    Ecol = np.dot(r, r) / mask.size
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / m.texEval)
    
    return w[0] * Ecol + w[1] * Ereg

def textureGrad(texCoef, x, mask, m, w = (1, 1)):
    """
    Jacobian for texture energy
    """
    Xcol = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1)).T
    
    r = (Xcol - x).flatten()
    
    # Jacobian
    J = m.texEvec[:, mask, :].reshape((m.texMean[:, mask].size, m.texEval.size), order = 'F')
    
    return 2 * (w[0] * np.dot(J.T, r) / mask.size + w[1] * texCoef / m.texEval)

if __name__ == "__main__":
    
    os.chdir('/home/leon/f2f-fitting/obama/orig/')
    numFrames = 2882 #2260 #3744
    
    # Load 3DMM
    m = Bunch(np.load('../../models/bfm2017.npz'))
#    m.idEvec = m.idEvec[:, :, :80]
#    m.idEval = m.idEval[:80]
#    m.expEvec = m.expEvec[:, :, :76]
#    m.expEval = m.expEval[:76]
    
    targetLandmarkInds = np.array([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
    
#    plt.ioff()
    param = np.zeros((numFrames, m.idEval.size + m.expEval.size + 7))
    cam = 'orthographic'
    
    view = np.load('../viewInFrame.npz')
    
    wCol = 1000
    wLan = 10
    wReg = 1
    
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
#            texCoef = np.random.rand(m.texEval.size)
            param = np.r_[np.zeros(m.idEval.size + m.expEval.size + 6), 1]
        
        # Get the 3D xyz values of the 3DMM landmarks
        lm3D = generateFace(param, m, ind = sourceLandmarkInds).T
        
        # Estimate the camera projection matrix from the landmark correspondences
        P = estCamMat(lm, lm3D, 'orthographic')
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(P, 'orthographic')
        
        param = np.r_[idCoef, expCoef, angles, t, s]
        
        initFit = minimize(initialShapeCost2D, param, args = (lm, m, sourceLandmarkInds, (wLan, wReg)), jac = initialShapeGrad2D)
        param = initFit.x
        idCoef = param[:m.idEval.size]
        expCoef = param[m.idEval.size: m.idEval.size+m.expEval.size]
        
        # Project 3D model into 2D plane
        fitting = generateFace(np.r_[param[:-1], 0, param[-1]], m)
            
        # Plot the projected 3D model on top of the input RGB image
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(fitting[0, :], fitting[1, :], fitting[2, :])
        
#        plt.figure()
#        plt.imshow(img)
#        plt.scatter(fitting[0, :], fitting[1, :], s = 0.1, c = 'g')
#        plt.scatter(fitting[0, sourceLandmarkInds], fitting[1, sourceLandmarkInds], s = 3, c = 'b')
#        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'r')
        
        # Rendering of initial shape
        texture = m.texMean
        meshData = np.r_[fitting.T, texture.T].astype(np.float32)
        indexData = m.face.astype(np.uint16)
        openglRender.initializeContext(img.shape[1], img.shape[0], meshData, indexData)
        openglRender.render(indexData)
        rendering = openglRender.grabRendering(img.shape[1], img.shape[0])
        
        plt.figure()
        plt.imshow(rendering)
        
        zBuffer, pixelCoord = calcZBuffer(fitting)
        
#        mask = np.zeros(img.shape[:2], dtype = bool)
#        mask[pixelCoord[:, 1], pixelCoord[:, 0]] = True
#        plt.figure()
#        plt.imshow(mask)
        
        """
        """
        def initialTextureCost(texCoef, img, vertexCoord, m):
            vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
            
            openglRender.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T].astype(np.float32))
            openglRender.resetFramebufferObject()
            openglRender.render(m.face.astype(np.uint16))
            rendering = openglRender.grabRendering(img.shape[1], img.shape[0])
            
            mask = rendering.any(axis = -1)
            
            #
            Ecol = np.linalg.norm((rendering[mask] - img[mask]), axis = 1).sum() / mask.sum()
            r = (rendering[mask] - img[mask]).flatten()
            Ecol = np.dot(r, r) / mask.sum()
            
            # Statistical regularization
            Ereg = np.sum(texCoef ** 2 / m.texEval)
            
            return 100 * Ecol + Ereg
        
#        initTex = minimize(initialTextureCost, texCoef, args = (img, fitting, m), method = 'powell')
#        texCoef = initTex['x']
#        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
#        
#        openglRender.updateVertexBuffer(np.r_[fitting.T, texture.T].astype(np.float32))
#        openglRender.resetFramebufferObject()
#        openglRender.render(m.face.astype(np.uint16))
#        rendering2 = openglRender.grabRendering(img.shape[1], img.shape[0])
#        
#        plt.figure()
#        plt.imshow(rendering2)
#        break
        
        imgMasked = img[pixelCoord[:, 1], pixelCoord[:, 0]]
        initTex = minimize(textureCost, texCoef, args = (imgMasked, zBuffer, m, (wCol, wReg)), jac = textureGrad)
#        check_grad(textureCost, textureGrad, texCoef, imgMasked, zBuffer, m)
        
        texCoef = initTex['x']
        
        # Project 3D model into 2D plane
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        #exportObj(shape.T, c = texture.T, f = m.face, fNameOut = 'texTest')
        
        meshData = np.r_[fitting.T, texture.T].astype(np.float32)
        indexData = m.face.astype(np.uint16)
        openglRender.initializeContext(img.shape[1], img.shape[0], meshData, indexData)
        openglRender.render(indexData)
        rendering = openglRender.grabRendering(img.shape[1], img.shape[0])
        
        plt.figure()
        plt.imshow(rendering)
        
        def genTexture(vertexCoord, texParam, m):
            
            texCoef = texParam[:m.texEval.size]
            lightCoef = texParam[m.texEval.size:].reshape(9, 3)
            
            texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
            
            # Evaluate spherical harmonics at face shape normals
            vertexNorms = calcNormals(vertexCoord, m)
            B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
            
            norm = np.r_[np.pi, np.repeat(2*np.pi/3, 3), np.repeat(np.pi/4, 5)]
            
            B *= norm[..., np.newaxis]
            
            I = np.empty((3, m.numVertices))
            for c in range(3):
                I[c, :] = np.dot(lightCoef[:, c], B * texture[c, :])
            
            return I
        
        # Evaluate spherical harmonics at face shape normals
        vertexNorms = calcNormals(fitting, m)
        B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        
        norm = np.r_[np.pi, np.repeat(2*np.pi/3, 3), np.repeat(np.pi/4, 5)]
        
        B *= norm[..., np.newaxis]
        
        I = np.empty((3, 9, m.numVertices))
        l = np.empty((9, 3))
        for c in range(3):
            I[c, ...] = B * texture[c, :]
            l[:, c] = nnls(I[c, :, zBuffer], imgMasked[:, c])[0]
        
        texParam = np.r_[texCoef, l.flatten()]
        textureWithLighting = genTexture(fitting, texParam, m)
        
        meshData = np.r_[fitting.T, textureWithLighting.T].astype(np.float32)
        indexData = m.face.astype(np.uint16)
        openglRender.initializeContext(img.shape[1], img.shape[0], meshData, indexData)
        openglRender.render(indexData)
        rendering = openglRender.grabRendering(img.shape[1], img.shape[0])
        
        plt.figure()
        plt.imshow(rendering)
            
        def textureLightingCost(texParam, x, mask, B, m, w = (1, 1), option = 'tl', constCoef = None):
            """
            Energy formulation for fitting texture and spherical harmonic lighting coefficients
            """
            if option is 'tl':
                texCoef = texParam[:m.texEval.size]
                lightCoef = texParam[m.texEval.size:].reshape(9, 3)
            elif option is 't':
                texCoef = texParam
                lightCoef = constCoef.reshape(9, 3)
            elif option is 'l':
                texCoef = constCoef
                lightCoef = texParam.reshape(9, 3)
            
            # Photo-consistency
            texture = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1))
            
            I = np.empty((texture.shape[0], mask.size))
            for c in range(texture.shape[0]):
                I[c, :] = np.dot(lightCoef[:, c], B[:, mask] * texture[c, :])
            
            r = (I.T - x).flatten()
            
#            Ecol = np.dot(r, r) / mask.size
            Ecol = np.dot(r, r)
            
            # Statistical regularization
            Ereg = np.sum(texCoef ** 2 / m.texEval)
            
            if option is 'l':
                return w[0] * Ecol
            else:
                return w[0] * Ecol + w[1] * Ereg
        
        def textureLightingGrad(texParam, x, mask, B, m, w = (1, 1), option = 'tl', constCoef = None):
            """
            Jacobian for texture and spherical harmonic lighting coefficients
            """
            if option is 'tl':
                texCoef = texParam[:m.texEval.size]
                lightCoef = texParam[m.texEval.size:].reshape(9, 3)
            elif option is 't':
                texCoef = texParam
                lightCoef = constCoef.reshape(9, 3)
            elif option is 'l':
                texCoef = constCoef
                lightCoef = texParam.reshape(9, 3)
            
            # Photo-consistency
            texture = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1))
            
            J_texCoef = np.empty((3*mask.size, m.texEval.size))
            J_lightCoef = np.empty((27, mask.size))
            I = np.empty((3, mask.size))
            for c in range(3):
                J_texCoef[c*mask.size: (c+1)*mask.size, :] = np.tensordot(lightCoef[:, c], m.texEvec[np.newaxis, c, mask, :] * B[:, mask, np.newaxis], axes = 1)
                J_lightCoef[c*9: (c+1)*9, :] = B[:, mask] * texture[c, :]
                I[c, :] = np.dot(lightCoef[:, c], J_lightCoef[c*9: (c+1)*9, :])
            
            r = (I - x.T)
            
            if option is 'tl':
#                return 2 * w[0] * np.r_[r.flatten().dot(J_texCoef), J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] / mask.size + np.r_[2 * w[1] * texCoef / m.texEval, np.zeros(27)]
                return 2 * w[0] * np.r_[r.flatten().dot(J_texCoef), J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] + np.r_[2 * w[1] * texCoef / m.texEval, np.zeros(27)]
        
            # Texture only
            elif option is 't':
#                return 2 * (w[0] * r.flatten().dot(J_texCoef) / mask.size + w[1] * texCoef / m.texEval)
                return 2 * (w[0] * r.flatten().dot(J_texCoef) + w[1] * texCoef / m.texEval)
            
            # Light only
            elif option is 'l':
#                return 2 * w[0] * np.r_[J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] / mask.size
                return 2 * w[0] * np.r_[J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])]
        
        check_grad(textureLightingCost, textureLightingGrad, np.r_[texCoef, l.flatten()], imgMasked, zBuffer, B, m)
        check_grad(textureLightingCost, textureLightingGrad, texCoef, imgMasked, zBuffer, B, m, (1, 1), 't', l.flatten())
        check_grad(textureLightingCost, textureLightingGrad, l.flatten(), imgMasked, zBuffer, B, m, (1, 1), 'l', texCoef)
        
        initTexLight = minimize(textureLightingCost, np.r_[texCoef, l.flatten()], args = (imgMasked, zBuffer, B, m), jac = textureLightingGrad)
        
        texParam = initTexLight['x']
        texCoef = texParam[:m.texEval.size]
        lightCoef = texParam[m.texEval.size:].reshape(9, 3)
        
        texture = genTexture(fitting, texParam, m)
        break
        
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