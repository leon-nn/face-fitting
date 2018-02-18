#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estCamMat, splitCamMat
import mm.optimize.image as opt
from mm.utils.mesh import calcNormals, generateFace, generateTexture, barycentricReconstruction
from mm.utils.transform import sh9

import os, json
import numpy as np
from scipy.optimize import minimize, check_grad, least_squares, nnls, lsq_linear
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from pylab import savefig

if __name__ == "__main__":
    
    # Change directory to the folder that holds the VRN data, OpenPose landmarks, and original images (frames) from the source video
    os.chdir('/home/leon/f2f-fitting/data/obama/')
    
    # Input the number of frames in the video
    numFrames = 2882 #2260 #3744
    
    # Load 3DMM
    m = MeshModel('../../models/bfm2017.npz')
    
    # Initialize shape parameters and 
    param = np.zeros((numFrames, m.idEval.size + m.expEval.size + 7))
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'
    
    # Set weights for the 3DMM RGB color fitting, landmark fitting, and regularization terms
    wCol = 1000
    wLan = 10
    wReg = 1
    
    for frame in np.arange(1, numFrames + 1):
        print(frame)
        fName = '{:0>5}'.format(frame)
        
        """
        Set filenames, read landmarks, load source video frames
        """
        # Frames from the source video
        fNameImgOrig = 'orig/' + fName + '.png'
        
        # OpenPose landmarks for each frame in the source video
        fNameLandmarks = 'landmark/' + fName + '.json'
        
        with open(fNameLandmarks, 'r') as fd:
            lm = json.load(fd)
        lm = np.array([l[0] for l in lm], dtype = int).squeeze()[:, :3]
        lmConf = lm[m.targetLMInd, -1]  # This is the confidence value of the landmarks
        lm = lm[m.targetLMInd, :2]
        
        # Load the source video frame and convert to 64-bit float
        img = io.imread(fNameImgOrig)
        img = img_as_float(img)
        
        # You can plot the landmarks over the frames if you want
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
        lm3D = generateFace(param, m, ind = m.sourceLMInd).T
        
        # Estimate the camera projection matrix from the landmark correspondences
        P = estCamMat(lm, lm3D, 'orthographic')
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(P, 'orthographic')
        
        param = np.r_[idCoef, expCoef, angles, t, s]
        
        initFit = minimize(opt.initialShapeCost, param, args = (lm, m, m.sourceLMInd, (wLan, wReg)), jac = opt.initialShapeGrad)
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
#        plt.scatter(fitting[0, m.sourceLMInd], fitting[1, m.sourceLMInd], s = 3, c = 'b')
#        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'r')
        
        # Rendering of initial shape
        texture = m.texMean
        meshData = np.r_[fitting.T, texture.T]
        indexData = m.face
        renderObj = Render(img.shape[1], img.shape[0], meshData, indexData)
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
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
            initTex = least_squares(opt.textureResiduals, texCoef, jac = opt.textureJacobian, args = (img, fitting, m, renderObj, (wCol, wReg), randomFaces), loss = 'soft_l1')
            texCoef = initTex['x']
            cost[i] = initTex.cost
        
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        
        renderObj.updateVertexBuffer(np.r_[fitting.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
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
        textureWithLighting = generateTexture(fitting, texParam, m)
        
        renderObj.updateVertexBuffer(np.r_[fitting.T, textureWithLighting.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        break
        texParam2 = texParam.copy()
        
#        check_grad(textureLightingCost2, textureLightingGrad2, texParam, img, fitting, B, m, (1, 1))
        
        cost = np.zeros(10)
        for i in range(10):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, fitting, B, m, (1, 1), randomFaces), loss = 'soft_l1', max_nfev = 100)
            texParam2 = initTexLight['x']
            cost[i] = initTexLight.cost
            
        texParam2 = initTexLight['x']
        texCoef = texParam[:m.texEval.size]
        lightCoef = texParam[m.texEval.size:].reshape(9, 3)
        
        texture = generateTexture(fitting, texParam2, m)
        
        renderObj.updateVertexBuffer(np.r_[fitting.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        break
        
        '''
        Optimization
        '''