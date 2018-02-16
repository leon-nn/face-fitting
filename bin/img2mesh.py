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
from mm import Bunch, onpick3, exportObj, generateFace, rotMat2angle, initialShapeCost2D, initialShapeGrad2D, estCamMat, splitCamMat, camWithShape, dR_dpsi, dR_dtheta, dR_dphi, calcNormals, sh9
from time import clock
import glob, os, re, json
import numpy as np
from scipy.optimize import minimize, check_grad, least_squares, nnls, lsq_linear
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from pylab import savefig

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
        img = io.imread(fNameImgOrig)
        img = img_as_float(img)
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
        openglRender.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = openglRender.grabRendering(img.shape[1], img.shape[0], return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        
#        imgReconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, indexData)
#        reconstruction = np.zeros(rendering.shape)
#        reconstruction[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
#        
#        plt.figure()
#        plt.imshow(np.fabs(reconstruction - rendering))
        
        # Get initial texture estimate
        
        numRandomFaces = 10000
        
        cost = np.zeros(20)
        for i in range(20):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTex = least_squares(textureResiduals, texCoef, jac = textureJacobian, args = (img, fitting, m, (wCol, wReg), randomFaces), loss = 'soft_l1')
            texCoef = initTex['x']
            cost[i] = initTex.cost
        
#        check_grad(initialTextureCost, initialTextureGrad, texCoef, img, fitting, m, (1000, 1))
        
#        initTex = least_squares(textureResiduals, texCoef, jac = textureJacobian, args = (img, fitting, m, (wCol, wReg)), loss = 'soft_l1')
#        texCoef = initTex['x']
        
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        
        openglRender.updateVertexBuffer(np.r_[fitting.T, texture.T].astype(np.float32), indexData)
        openglRender.resetFramebufferObject()
        openglRender.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = openglRender.grabRendering(img.shape[1], img.shape[0], return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        
        # Evaluate spherical harmonics at face shape normals
        vertexNorms = calcNormals(fitting, m)
        B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        
        imgMasked = img[pixelCoord[:, 0], pixelCoord[:, 1]]
        
        I = np.empty((3, pixelFaces.size, 9))
        l = np.empty((9, 3))
        for c in range(3):
            I[c, ...] = barycentricReconstruction(B * texture[c, :], pixelFaces, pixelBarycentricCoords, indexData)
#            l[:, c] = nnls(I[c, ...], imgMasked[:, c])[0]
            l[:, c] = lsq_linear(I[c, ...], imgMasked[:, c]).x
        
        texParam = np.r_[texCoef, l.flatten()]
        textureWithLighting = genTexture(fitting, texParam, m)
        
        openglRender.updateVertexBuffer(np.r_[fitting.T, textureWithLighting.T].astype(np.float32), indexData)
        openglRender.resetFramebufferObject()
        openglRender.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = openglRender.grabRendering(img.shape[1], img.shape[0], return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        break
        texParam2 = texParam.copy()
        
#        check_grad(textureLightingCost2, textureLightingGrad2, texParam, img, fitting, B, m, (1, 1))
        
        cost = np.zeros(10)
        for i in range(10):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTexLight = least_squares(textureLightingResiduals, texParam2, jac = textureLightingJacobian, args = (img, fitting, B, m, (1, 1), randomFaces), loss = 'soft_l1', max_nfev = 100)
            texParam2 = initTexLight['x']
            cost[i] = initTexLight.cost
            
        texParam2 = initTexLight['x']
        texCoef = texParam[:m.texEval.size]
        lightCoef = texParam[m.texEval.size:].reshape(9, 3)
        
        texture = genTexture(fitting, texParam2, m)
        
        openglRender.updateVertexBuffer(np.r_[fitting.T, texture.T].astype(np.float32), indexData)
        openglRender.resetFramebufferObject()
        openglRender.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = openglRender.grabRendering(img.shape[1], img.shape[0], return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        break
        
        '''
        Optimization
        '''

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