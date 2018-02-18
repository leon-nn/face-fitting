import numpy as np

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

def gaussNewton(P, model, target, targetLandmarks, sourceLandmarkInds, NN, jacobi = True, calcId = True):
    """
    Energy function to be minimized for fitting.
    """
    # Shape eigenvector coefficients
    idCoef = P[: model.idEval.size]
    expCoef = P[model.idEval.size: model.idEval.size + model.expEval.size]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = P[model.idEval.size + model.expEval.size:][:3]
    R = rotMat2angle(angles)
    t = P[model.idEval.size + model.expEval.size:][3: 6]
    s = P[model.idEval.size + model.expEval.size:][6]
    
    # Transpose if necessary
    if targetLandmarks.shape[0] != 3:
        targetLandmarks = targetLandmarks.T
    
    # The eigenmodel, before rigid transformation and scaling
    model = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)
    
    # After rigid transformation and scaling
    source = s*np.dot(R, model) + t[:, np.newaxis]
    
    # Find the nearest neighbors of the target to the source vertices
#    start = clock()
    distance, ind = NN.kneighbors(source.T)
    targetNN = target[ind.squeeze(axis = 1), :].T
#    print('NN: %f' % (clock() - start))
    
    # Calculate resisduals
    rVert = targetNN - source
    rLand = targetLandmarks - source[:, sourceLandmarkInds]
    rAlpha = idCoef ** 2 / model.idEval
    rDelta = expCoef ** 2 / model.expEval
    
    # Calculate costs
    Ever = np.linalg.norm(rVert, axis = 0).sum() / model.numVertices
    Elan = np.linalg.norm(rLand, axis = 0).sum() / sourceLandmarkInds.size
    Ereg = np.sum(rAlpha) + np.sum(rDelta)
    
    if jacobi:
#        start = clock()
        
        drV_dalpha = -s*np.tensordot(R, model.idEvec, axes = 1)
        drV_ddelta = -s*np.tensordot(R, model.expEvec, axes = 1)
        drV_dpsi = -s*np.dot(dR_dpsi(angles), model)
        drV_dtheta = -s*np.dot(dR_dtheta(angles), model)
        drV_dphi = -s*np.dot(dR_dphi(angles), model)
        drV_dt = -np.tile(np.eye(3), [source.shape[1], 1])
        drV_ds = -np.dot(R, model)
        
        drR_dalpha = np.diag(2*idCoef / model.idEval)
        drR_ddelta = np.diag(2*expCoef / model.expEval)
        
        # Calculate Jacobian
        if calcId:
            
            r = np.r_[rVert.flatten('F'), rLand.flatten('F'), rAlpha, rDelta]
        
            J = np.r_[np.c_[drV_dalpha.reshape((source.size, idCoef.size), order = 'F'), drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')], np.c_[drV_dalpha[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, idCoef.size), order = 'F'), drV_ddelta[:, sourceLandmarkInds, :].reshape((targetLandmarks.size, expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')], np.c_[drR_dalpha, np.zeros((idCoef.size, expCoef.size + 7))], np.c_[np.zeros((expCoef.size, idCoef.size)), drR_ddelta, np.zeros((expCoef.size, 7))]]
            
            # Parameter update (Gauss-Newton)
            dP = -np.linalg.inv(np.dot(J.T, J)).dot(J.T).dot(r)
        
        else:
            
            r = np.r_[rVert.flatten('F'), rLand.flatten('F'), rDelta]
            
            J = np.r_[np.c_[drV_ddelta.reshape((source.size, expCoef.size), order = 'F'), drV_dpsi.flatten('F'), drV_dtheta.flatten('F'), drV_dphi.flatten('F'), drV_dt, drV_ds.flatten('F')], np.c_[drV_ddelta[:, sourceLandmarkInds, :].reshape((np.prod(targetLandmarks.shape), expCoef.size), order = 'F'), drV_dpsi[:, sourceLandmarkInds].flatten('F'), drV_dtheta[:, sourceLandmarkInds].flatten('F'), drV_dphi[:, sourceLandmarkInds].flatten('F'), drV_dt[:sourceLandmarkInds.size * 3, :], drV_ds[:, sourceLandmarkInds].flatten('F')], np.c_[drR_ddelta, np.zeros((expCoef.size, 7))]]
            
            # Parameter update (Gauss-Newton)
            dP = np.r_[np.zeros(model.idEval.size), -np.linalg.inv(np.dot(J.T, J)).dot(J.T).dot(r)]
        
#        print('GN: %f' % (clock() - start))
        
        return Ever + Elan + Ereg, dP
    
    return Ever + Elan + Ereg

def shBasis(alb, n):
    """
    SH basis functions                               lm
        1/np.sqrt(4*np.pi)                          Y00
        np.sqrt(3/(4*np.pi))*nz                     Y10
        np.sqrt(3/(4*np.pi))*nx                     Y11e
        np.sqrt(3/(4*np.pi))*ny                     Y11o
        1/2*np.sqrt(5/(4*np.pi))*(3*nz^2 - 1)       Y20
        3*np.sqrt(5/(12*np.pi))*nx*nz               Y21e
        3*np.sqrt(5/(12*np.pi))*ny*nz               Y21o
        3/2*np.sqrt(5/(12*np.pi))*(nx^2 - ny^2)     Y22e
        3*np.sqrt(5/(12*np.pi))*nx*ny               Y22o
    
    For a sphere, the Lambertian kernel has most of its energy in the first three bands of the spherical harmonic basis functions (above). This implies that Lambertian reflectance functions can be well-approximated by these low-order SH bases.
    """
    
    # Nine delta function locations (el, az) for point light sources to create positive lighting
    lsph = np.array([[0, 0], [68, -90], [74, 108], [80, 52], [85, -42], [85, -137], [85, 146], [85, -4], [51, 67]]) * np.pi / 180
#    lsph = np.array([[0, 0], [49, 17], [-68, 0], [73, -18], [77, 37], [-84, 47], [-84, -47], [82, -56], [-50, -84]]) * np.pi / 180
    
    # Transform to Cartesian coordinates
    lx, ly, lz = sph2cart(lsph[:, 0], lsph[:, 1])
    
    # Evaluate spherical harmonics at these point light source locations
    H = sh9(lx, ly, lz)
    
    # Evaluate spherical harmonics at face shape normals
    B = sh9(n[:, 0], n[:, 1], n[:, 2])
    
    I = np.empty((alb.shape[0], 9, alb.shape[1]))
    for c in range(alb.shape[0]):
        I[c, :, :] = np.dot(H.T, B * alb[c, :])
    
#    b = np.empty((alb.shape[0], alb.shape[1], 9))
#    b[:, :, 0] = np.pi * 1/np.sqrt(4*np.pi) * alb
#    b[:, :, 1] = 2*np.pi/3 * np.sqrt(3/(4*np.pi)) * n[:, 2] * alb
#    b[:, :, 2] = 2*np.pi/3 * np.sqrt(3/(4*np.pi)) * n[:, 0] * alb
#    b[:, :, 3] = 2*np.pi/3 * np.sqrt(3/(4*np.pi)) * n[:, 1] * alb
#    b[:, :, 4] = np.pi/4 * 1/2*np.sqrt(5/(4*np.pi)) * (3*np.square(n[:, 2]) - 1) * alb
#    b[:, :, 5] = np.pi/4 * 3*np.sqrt(5/(12*np.pi)) * n[:, 0] * n[:, 2] * alb
#    b[:, :, 6] = np.pi/4 * 3*np.sqrt(5/(12*np.pi)) * n[:, 1] * n[:, 2] * alb
#    b[:, :, 7] = np.pi/4 * 3/2*np.sqrt(5/(12*np.pi)) * (np.square(n[:, 0]) - np.square(n[:, 1])) * alb
#    b[:, :, 8] = np.pi/4 * 3*np.sqrt(5/(12*np.pi)) * n[:, 0] * n[:, 1] * alb
    
    return I

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
            zBuffer[i] = candidateVertexInds[np.argmin(vertexCoord[2, candidateVertexInds])]
    
    return zBuffer, pixelCoord

def textureCostV(texCoef, x, mask, model, w = (1, 1)):
    """
    Energy formulation for fitting texture
    """
    # Photo-consistency
    Xcol = (model.texMean[:, mask] + np.tensordot(model.texEvec[:, mask, :], texCoef, axes = 1)).T
    
    r = (Xcol - x).flatten()
    
    Ecol = np.dot(r, r) / mask.size
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / model.texEval)
    
    return w[0] * Ecol + w[1] * Ereg

def textureGradV(texCoef, x, mask, model, w = (1, 1)):
    """
    Jacobian for texture energy
    """
    Xcol = (model.texMean[:, mask] + np.tensordot(model.texEvec[:, mask, :], texCoef, axes = 1)).T
    
    r = (Xcol - x).flatten()
    
    # Jacobian
    J = model.texEvec[:, mask, :].reshape((model.texMean[:, mask].size, model.texEval.size), order = 'F')
    
    return 2 * (w[0] * np.dot(J.T, r) / mask.size + w[1] * texCoef / model.texEval)

def textureLightingCostV(texParam, x, mask, B, model, w = (1, 1), option = 'tl', constCoef = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    if option is 'tl':
        texCoef = texParam[:model.texEval.size]
        lightCoef = texParam[model.texEval.size:].reshape(9, 3)
    elif option is 't':
        texCoef = texParam
        lightCoef = constCoef.reshape(9, 3)
    elif option is 'l':
        texCoef = constCoef
        lightCoef = texParam.reshape(9, 3)
    
    # Photo-consistency
    texture = (model.texMean[:, mask] + np.tensordot(model.texEvec[:, mask, :], texCoef, axes = 1))
    
    I = np.empty((texture.shape[0], mask.size))
    for c in range(texture.shape[0]):
        I[c, :] = np.dot(lightCoef[:, c], B[:, mask]) * texture[c, :]
    
    r = (I.T - x).flatten()
    
    Ecol = np.dot(r, r) / mask.size
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / model.texEval)
    
    if option is 'l':
        return w[0] * Ecol
    else:
        return w[0] * Ecol + w[1] * Ereg

def textureLightingGradV(texParam, x, mask, B, model, w = (1, 1), option = 'tl', constCoef = None):
    """
    Jacobian for texture and spherical harmonic lighting coefficients
    """
    if option is 'tl':
        texCoef = texParam[:model.texEval.size]
        lightCoef = texParam[model.texEval.size:].reshape(9, 3)
    elif option is 't':
        texCoef = texParam
        lightCoef = constCoef.reshape(9, 3)
    elif option is 'l':
        texCoef = constCoef
        lightCoef = texParam.reshape(9, 3)
    
    # Photo-consistency
    texture = (model.texMean[:, mask] + np.tensordot(model.texEvec[:, mask, :], texCoef, axes = 1))
    
    J_texCoef = np.empty((3*mask.size, model.texEval.size))
    J_lightCoef = np.empty((27, mask.size))
    I = np.empty((3, mask.size))
    for c in range(3):
        J_texCoef[c*mask.size: (c+1)*mask.size, :] = np.dot(lightCoef[:, c], B[:, mask])[:, np.newaxis] * model.texEvec[c, mask, :]
        J_lightCoef[c*9: (c+1)*9, :] = texture[c, :] * B[:, mask]
        I[c, :] = np.dot(lightCoef[:, c], B[:, mask]) * texture[c, :]
    
    r = (I - x.T)
    
    if option is 'tl':
        return 2 * w[0] * np.r_[r.flatten().dot(J_texCoef), J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] / mask.size + np.r_[2 * w[1] * texCoef / model.texEval, np.zeros(27)]

    # Texture only
    elif option is 't':
        return 2 * (w[0] * r.flatten().dot(J_texCoef) / mask.size + w[1] * texCoef / model.texEval)
    
    # Light only
    elif option is 'l':
        return 2 * w[0] * np.r_[J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] / mask.size

if __name__ == "__main__":
    #bfm2fw = np.array([0, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 18, 21, 22, 23, 24, 29, 30, 32, 33, 34, 37, 38, 39])
    #fw2bfm = np.array([7, 59, 55, 62, 49, 39, 65, 34, 33, 31, 32, 52, 50, 45, 41, 40, 30, 29, 27, 28, 46, 48, 44, 37, 38])
    
#    targetLandmarkInds = np.array([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
#    sourceLandmarkInds = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])
#    
#    idCoef = np.zeros(model.idEval.shape)
#    idCoef[0] = 1
#    expCoef = np.zeros(model.expEval.shape)
#    expCoef[0] = 1
#    texCoef = np.zeros(model.texEval.shape)
#    texCoef[0] = 1
    
    ## Load 3D vertex indices of landmarks and find their vertices on the neutral face
    #landmarkInds3D = np.load('./data/landmarkInds3D.npy')
    #landmarkInds3D = np.r_[model.landmarkInd[bfm2fw], 16225, 16246, 16276, 22467, 22437, 22416]
    
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
    
    #X = model.idMean
    #X = R.dot(model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)) + t[:, np.newaxis]
    #normals = calcNormals(R, m, idCoef, expCoef)
    #shBases = shBasis(texture, normals)
    
    #exportObj(X.T, c = shb[:, :, 8].T, f = model.face, fNameOut = 'sh9')
    
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