#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:43:00 2017

@author: nguyen
"""

if __name__ == "__main__":
    #bfm2fw = np.array([0, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 18, 21, 22, 23, 24, 29, 30, 32, 33, 34, 37, 38, 39])
    #fw2bfm = np.array([7, 59, 55, 62, 49, 39, 65, 34, 33, 31, 32, 52, 50, 45, 41, 40, 30, 29, 27, 28, 46, 48, 44, 37, 38])
    
#    targetLandmarkInds = np.array([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
#    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
#    
#    idCoef = np.zeros(m.idEval.shape)
#    idCoef[0] = 1
#    expCoef = np.zeros(m.expEval.shape)
#    expCoef[0] = 1
#    texCoef = np.zeros(m.texEval.shape)
#    texCoef[0] = 1
    
    ## Load 3D vertex indices of landmarks and find their vertices on the neutral face
    #landmarkInds3D = np.load('./data/landmarkInds3D.npy')
    #landmarkInds3D = np.r_[m.landmarkInd[bfm2fw], 16225, 16246, 16276, 22467, 22437, 22416]
    
    #fwlm = landmarkInds3D[np.r_[1, 3, 5, np.arange(7, 16)]]
    #bfmlm = landmarkInds3D[[25, 0, 30, 18, 17, 9, 8, 20, 4, 11, 2, 6]]
    
    #pose = 11
    #tester = 0
    
    # Gather 2D landmarks that correspond to manually chosen 3D landmarks
    #landmarkInds2D = np.array([0, 1, 4, 7, 10, 13, 14, 27, 29, 31, 33, 46, 49, 52, 55, 65])
    #landmarkInds2D = np.r_[fw2bfm, 1, 2, 3, 11, 12, 13]
    #landmarks = np.load('./data/landmarks2D.npy')[pose, tester, landmarkInds2D, :]
    #landmarkPixelInd = (landmarks * np.array([639, 479])).astype(int)
    #landmarkPixelInd[:, 1] = 479 - landmarkPixelInd[:, 1]
    
    ## Get target 3D coordinates of depth maps at the 16 landmark locations
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
    
    
    #reconstImg = np.zeros(img.shape)
    #reconstImg[:, :, 0].flat[np.ravel_multi_index(vertex2pixel[zBuffer, ::-1].T, img.shape[:2])] = texture[0, zBuffer]
    #reconstImg[:, :, 1].flat[np.ravel_multi_index(vertex2pixel[zBuffer, ::-1].T, img.shape[:2])] = texture[1, zBuffer]
    #reconstImg[:, :, 2].flat[np.ravel_multi_index(vertex2pixel[zBuffer, ::-1].T, img.shape[:2])] = texture[2, zBuffer]
    #plt.figure()
    #plt.imshow(reconstImg)
    
    # Plot the projected 3D model on top of the input RGB image
    #fName = dirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.png'
    #img = mpimg.imread(fName)
    #plt.figure()
    #plt.imshow(img)
    #plt.hold(True)
    #plt.scatter(fitting[0, :], fitting[1, :], s = 0.1, c = 'g')
    #plt.hold(True)
    #plt.scatter(fitting[0, landmarkInds3D], fitting[1, landmarkInds3D], s = 3, c = 'b')
    #plt.hold(True)
    #plt.scatter(landmarkPixelInd[:, 0], landmarkPixelInd[:, 1], s = 2, c = 'r')
    
    """
    
    """
    ## Get triangle faces
    #f = importObj('./masks2v2/', shape = 0, dataToImport = ['f'])[0][0, :, :] - 1
    #f = np.r_[f[:, [0, 1, 2]], f[:, [0, 2, 3]]]
    
    #X = m.idMean
    #X = R.dot(m.idMean + np.tensordot(m.idEvec, idCoef, axes = 1) + np.tensordot(m.expEvec, expCoef, axes = 1)) + t[:, np.newaxis]
    #normals = calcNormals(R, m, idCoef, expCoef)
    #shBases = shBasis(texture, normals)
    
    #exportObj(X.T, c = shb[:, :, 8].T, f = m.face, fNameOut = 'sh9')
    
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
    
    ## Confirm that 16 2D landmarks are in correspondence across poses and testers
    #pose = 0
    #tester = 0
    #landmarks = np.load('./data/landmarks2D.npy')[pose, tester, :, :]
    #landmarkPixelInd = (landmarks * np.array([639, 479])).astype(int)
    #landmarkPixelInd[:, 1] = 479 - landmarkPixelInd[:, 1]
    #fName = dirName + 'Tester_' + str(tester+1) + '/TrainingPose/pose_' + str(pose) + '.png'
    #img = mpimg.imread(fName)
    #
    ##bfm2fw = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 21, 22, 23, 24, 25, 27, 29, 30, 32, 33, 34, 37, 38, 39]
    ##fw2bfm = [7, 59, 55, 62, 49, 39, 65, 34, 15, 18, 33, 31, 32, 52, 50, 45, 41, 40, 30, 21, 24, 29, 27, 28, 46, 48, 44, 37, 38]
    #
    ##plt.scatter(x*640, (1-y)*480, s = 1)
    ##x = landmarkPixelInd[fw2bfm, 0]
    ##y = landmarkPixelInd[fw2bfm, 1]
    #x = landmarkPixelInd[:, 0]
    #y = landmarkPixelInd[:, 1]
    #
    #fig, ax = plt.subplots()
    #plt.imshow(img)
    ##plt.imshow(depth[pose, tester, :, :].astype(float))
    #plt.hold(True)
    #ax.scatter(x, y, s = 1, c = 'r', picker = True)
    #fig.canvas.mpl_connect('pick_event', onpick3)