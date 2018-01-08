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

from mm import Bunch, onpick3, exportObj, generateFace, rotMat2angle, initialRegistration, orthographicEst, perspectiveEst, estCamMat, splitCamMat, camWithShape, dR_dpsi, dR_dtheta, dR_dphi, calcNormals, shBasis
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