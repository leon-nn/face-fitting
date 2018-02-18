#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.models import MeshModel
from mm.utils.opengl import Render
from mm.optimize.camera import estimateCamMat, splitCamMat
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
    os.chdir('/home/leon/f2f-shape/data/obama/')
    
    # Input the number of frames in the video
    numFrames = 2882 #2260 #3744
    
    # Load 3DMM
    m = MeshModel('../../models/bfm2017.npz')
    
    # Set an orthographic projection for the camera matrix
    cam = 'orthographic'
    
    # Set weights for the 3DMM RGB color shape, landmark shape, and regularization terms
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
        
        """
        Initial registration of similarity transform and shape coefficients
        """
        
        # Initialize 3DMM parameters for the first frame
        if frame == 1:
            idCoef = np.zeros(m.idEval.size)
            expCoef = np.zeros(m.expEval.size)
            texCoef = np.zeros(m.texEval.size)
            param = np.r_[np.zeros(m.idEval.size + m.expEval.size + 6), 1]
        
        # Get the vertex values of the 3DMM landmarks
        lm3D = generateFace(param, m, ind = m.sourceLMInd).T
        
        # Estimate the camera projection matrix from the landmark correspondences
        camMat = estimateCamMat(lm, lm3D, 'orthographic')
        
        # Factor the camera projection matrix into the intrinsic camera parameters and the rotation/translation similarity transform parameters
        s, angles, t = splitCamMat(camMat, 'orthographic')
        
        # Concatenate parameters for input into optimization routine. Note that the translation vector here is only (2,) for x and y (no z)
        param = np.r_[idCoef, expCoef, angles, t, s]
        
        # Initial optimization of shape parameters with similarity transform parameters
        initFit = minimize(opt.initialShapeCost, param, args = (lm, m, (wLan, wReg)), jac = opt.initialShapeGrad)
        param = initFit.x
        idCoef = param[:m.idEval.size]
        expCoef = param[m.idEval.size: m.idEval.size+m.expEval.size]
        
        # Generate 3DMM vertices from shape and similarity transform parameters
        shape = generateFace(np.r_[param[:-1], 0, param[-1]], m)
        
        # Plot the 3DMM in 3D
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(shape[0, :], shape[1, :], shape[2, :])
        
        # Plot the 3DMM landmarks with the OpenPose landmarks over the image
#        plt.figure()
#        plt.imshow(img)
#        plt.scatter(shape[0, m.sourceLMInd], shape[1, m.sourceLMInd], s = 3, c = 'b')
#        plt.scatter(lm[:, 0], lm[:, 1], s = 2, c = 'r')
        
        # Rendering of initial 3DMM shape with mean texture model
        texture = m.texMean
        meshData = np.r_[shape.T, texture.T]
        renderObj = Render(img.shape[1], img.shape[0], meshData, m.face)
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        # Plot rendering
        plt.figure()
        plt.imshow(rendering)
        
        # Using the barycentric parameters from the rendering, we can reconstruct the image with the 3DMM texture model by taking barycentric combinations of the 3DMM RGB values defined at the vertices
        imgReconstruction = barycentricReconstruction(texture, pixelFaces, pixelBarycentricCoords, m.face)
        
        # Put values from the reconstruction into a (height, width, 3) array for plotting
        reconstruction = np.zeros(rendering.shape)
        reconstruction[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
        
        # Plot the difference of the reconstruction with the rendering to see that they are very close-- the output values should be close to 0
        plt.figure()
        plt.imshow(np.fabs(reconstruction - rendering))
        
        """
        Get initial texture parameter guess
        """
        
        numRandomFaces = 10000
        
        cost = np.zeros(20)
        for i in range(20):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTex = least_squares(opt.textureResiduals, texCoef, jac = opt.textureJacobian, args = (img, shape, m, renderObj, (wCol, wReg), randomFaces), loss = 'soft_l1')
            texCoef = initTex['x']
            cost[i] = initTex.cost
        
        texture = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
        
        renderObj.updateVertexBuffer(np.r_[shape.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        
        """
        Get initial spherical harmonic lighting parameter guess
        """
        
        # Evaluate spherical harmonics at face shape normals
        vertexNorms = calcNormals(shape, m)
        B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        
        imgMasked = img[pixelCoord[:, 0], pixelCoord[:, 1]]
        
        I = np.empty((3, pixelFaces.size, 9))
        l = np.empty((9, 3))
        for c in range(3):
            I[c, ...] = barycentricReconstruction(B * texture[c, :], pixelFaces, pixelBarycentricCoords, m.face)
#            l[:, c] = nnls(I[c, ...], imgMasked[:, c])[0]
            l[:, c] = lsq_linear(I[c, ...], imgMasked[:, c]).x
        
        texParam = np.r_[texCoef, l.flatten()]
        textureWithLighting = generateTexture(shape, texParam, m)
        
        renderObj.updateVertexBuffer(np.r_[shape.T, textureWithLighting.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        break
        
        """
        Optimization over texture and lighting parameters
        """
        texParam2 = texParam.copy()
        
#        check_grad(textureLightingCost2, textureLightingGrad2, texParam, img, shape, B, m, (1, 1))
        
        cost = np.zeros(10)
        for i in range(10):
            randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
            initTexLight = least_squares(opt.textureLightingResiduals, texParam2, jac = opt.textureLightingJacobian, args = (img, shape, B, m, (1, 1), randomFaces), loss = 'soft_l1', max_nfev = 100)
            texParam2 = initTexLight['x']
            cost[i] = initTexLight.cost
            
        texParam2 = initTexLight['x']
        texCoef = texParam[:m.texEval.size]
        lightCoef = texParam[m.texEval.size:].reshape(9, 3)
        
        texture = generateTexture(shape, texParam2, m)
        
        renderObj.updateVertexBuffer(np.r_[shape.T, texture.T])
        renderObj.resetFramebufferObject()
        renderObj.render()
        rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)
        
        plt.figure()
        plt.imshow(rendering)
        
        '''
        Optimization over shape, texture, and lighting
        '''
        # Need to do
        break