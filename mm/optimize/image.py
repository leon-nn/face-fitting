#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:59:05 2018

@author: leon
"""

import numpy as np
from scipy.linalg import block_diag
from ..utils.mesh import generateFace, generateTexture, barycentricReconstruction
from ..utils.transform import rotMat2angle
from .derivative import dR_dpsi, dR_dtheta, dR_dphi

def initialShapeCost2D(P, target, m, sourceLandmarkInds, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Insert z translation
    P = np.r_[P[:-1], 0, P[-1]]
    
    # Landmark fitting cost
    source = generateFace(P, m, ind = sourceLandmarkInds)[:2, :]
    
    rlan = (source - target.T).flatten('F')
    Elan = np.dot(rlan, rlan) / sourceLandmarkInds.size
    
    # Regularization cost
    Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    return w[0] * Elan + w[1] * Ereg

def initialShapeGrad2D(P, target, m, sourceLandmarkInds, w = (1, 1)):
    # Shape eigenvector coefficients
    idCoef = P[: m.idEval.size]
    expCoef = P[m.idEval.size: m.idEval.size + m.expEval.size]
    
    # Rotation Euler angles, translation vector, scaling factor
    angles = P[m.idEval.size + m.expEval.size:][:3]
    R = rotMat2angle(angles)
    t = np.r_[P[m.idEval.size + m.expEval.size:][3: 5], 0]
    s = P[m.idEval.size + m.expEval.size:][5]
    
    # The eigenmodel, before rigid transformation and scaling
    model = m.idMean[:, sourceLandmarkInds] + np.tensordot(m.idEvec[:, sourceLandmarkInds, :], idCoef, axes = 1) + np.tensordot(m.expEvec[:, sourceLandmarkInds, :], expCoef, axes = 1)
    
    # After rigid transformation and scaling
    source = (s*np.dot(R, model) + t[:, np.newaxis])[:2, :]
    
    rlan = (source - target.T).flatten('F')
        
    drV_dalpha = s*np.tensordot(R, m.idEvec[:, sourceLandmarkInds, :], axes = 1)
    drV_ddelta = s*np.tensordot(R, m.expEvec[:, sourceLandmarkInds, :], axes = 1)
    drV_dpsi = s*np.dot(dR_dpsi(angles), model)
    drV_dtheta = s*np.dot(dR_dtheta(angles), model)
    drV_dphi = s*np.dot(dR_dphi(angles), model)
    drV_dt = np.tile(np.eye(2), [sourceLandmarkInds.size, 1])
    drV_ds = np.dot(R, model)
    
    Jlan = np.c_[drV_dalpha[:2, ...].reshape((source.size, idCoef.size), order = 'F'), drV_ddelta[:2, ...].reshape((source.size, expCoef.size), order = 'F'), drV_dpsi[:2, :].flatten('F'), drV_dtheta[:2, :].flatten('F'), drV_dphi[:2, :].flatten('F'), drV_dt, drV_ds[:2, :].flatten('F')]
    
    return 2 * (w[0] * np.dot(Jlan.T, rlan) / sourceLandmarkInds.size + w[1] * np.r_[idCoef / m.idEval, expCoef / m.expEval, np.zeros(6)])

def camWithShape(param, m, lm2d, lm3dInd, cam):
    """
    Minimize L2-norm of landmark fitting residuals and regularization terms for shape parameters
    """
    if cam == 'orthographic':
        P = param[:8]
        P = np.vstack((P.reshape((2, 4)), np.array([0, 0, 0, 1])))
        idCoef = param[8: 8 + m.idEval.size]
        expCoef = param[8 + m.idEval.size:]
    
    elif cam == 'perspective':
        P = param[:12]
        P = P.reshape((3, 4))
        idCoef = param[12: 12 + m.idEval.size]
        expCoef = param[12 + m.idEval.size:]
    
    # Convert to homogenous coordinates
    numLandmarks = lm3dInd.size
    
    lm3d = generateFace(np.r_[idCoef, expCoef, np.zeros(6), 1], m, ind = lm3dInd).T
    
    xlan = np.c_[lm2d, np.ones(numLandmarks)]
    Xlan = np.dot(np.c_[lm3d, np.ones(numLandmarks)], P.T)
    
    # Energy of landmark residuals
    rlan = (Xlan - xlan).flatten('F')
    Elan = np.dot(rlan, rlan)
    
    # Energy of shape regularization terms
    Ereg = np.sum(idCoef ** 2 / m.idEval) + np.sum(expCoef ** 2 / m.expEval)
    
    return Elan + Ereg

"""
Barycentric methods for optimizing texture and lighting
"""

def textureCost(texCoef, img, vertexCoord, m, renderObj, w = (1, 1)):
    indexData = m.face.astype(np.uint16)
    vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)[:2]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    # Color matching cost
    r = (rendering - img).flatten()
    Ecol = np.dot(r, r) / pixelCoord.shape[0]
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / m.texEval)
    
    return w[0] * Ecol + w[1] * Ereg

def textureGrad(texCoef, img, vertexCoord, m, renderObj, w = (1, 1)):
    indexData = m.face.astype(np.uint16)
    vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)
    numPixels = pixelFaces.size
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    pixelVertices = indexData[pixelFaces, :]
    
    r = (rendering - img).flatten('F')
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = barycentricReconstruction(m.texEvec[c].T, pixelFaces, pixelBarycentricCoords, indexData)
    
    w = (1, 1)
    
    return 2 * (w[0] * r.dot(J_texCoef) / numPixels + w[1] * texCoef / m.texEval)

def textureResiduals(texCoef, img, vertexCoord, m, renderObj, w = (1, 1), randomFaces = None):
    indexData = m.face.astype(np.uint16)
    vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)[:2]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    return np.r_[w[0] / numPixels * (rendering - img).flatten('F'), w[1] * texCoef ** 2 / m.texEval]

def textureJacobian(texCoef, img, vertexCoord, m, renderObj, w = (1, 1), randomFaces = None):
    indexData = m.face.astype(np.uint16)
    vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)[2:]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size
    
    pixelVertices = indexData[pixelFaces, :]
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = barycentricReconstruction(m.texEvec[c].T, pixelFaces, pixelBarycentricCoords, indexData)
    
    return np.r_[w[0] / numPixels * J_texCoef, w[1] * np.diag(texCoef / m.texEval)]

def textureLightingCost(texParam, img, vertexCoord, B, m, renderObj, w = (1, 1), option = 'tl', constCoef = None):
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
        
    indexData = m.face.astype(np.uint16)
    
    texture = generateTexture(vertexCoord, np.r_[texCoef, lightCoef.flatten()], m)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)[:2]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    # Color matching cost
    r = (rendering - img).flatten()
    Ecol = np.dot(r, r) / pixelCoord.shape[0]
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / m.texEval)
    
    if option is 'l':
        return w[0] * Ecol
    else:
        return w[0] * Ecol + w[1] * Ereg

def textureLightingGrad(texParam, img, vertexCoord, B, m, renderObj, w = (1, 1), option = 'tl', constCoef = None):
    if option is 'tl':
        texCoef = texParam[:m.texEval.size]
        lightCoef = texParam[m.texEval.size:].reshape(9, 3)
    elif option is 't':
        texCoef = texParam
        lightCoef = constCoef.reshape(9, 3)
    elif option is 'l':
        texCoef = constCoef
        lightCoef = texParam.reshape(9, 3)
        
    indexData = m.face.astype(np.uint16)
    vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, lightCoef.flatten()], m)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)
    numPixels = pixelFaces.size
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    pixelVertices = indexData[pixelFaces, :]
    
    r = rendering - img
    
    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, indexData)
    pixelSHBasis = barycentricReconstruction(B, pixelFaces, pixelBarycentricCoords, indexData)
    J_lightCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(m.texEvec[c].T, pixelFaces, pixelBarycentricCoords, indexData)
        pixelSHLighting = barycentricReconstruction(np.dot(lightCoef[:, c], B), pixelFaces, pixelBarycentricCoords, indexData)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]

    if option is 'tl':
        return 2 * w[0] * np.r_[r.flatten('F').dot(J_texCoef), r[:, 0].dot(J_lightCoef[0]), r[:, 1].dot(J_lightCoef[1]), r[:, 2].dot(J_lightCoef[2])] / numPixels + np.r_[2 * w[1] * texCoef / m.texEval, np.zeros(27)]

    # Texture only
    elif option is 't':
        return 2 * (w[0] * r.flatten('F').dot(J_texCoef) / numPixels + w[1] * texCoef / m.texEval)
    
    # Light only
    elif option is 'l':
        return 2 * w[0] * np.r_[r[:, 0].dot(J_lightCoef[0]), r[:, 1].dot(J_lightCoef[1]), r[:, 2].dot(J_lightCoef[2])] / numPixels
    
def textureLightingResiduals(texParam, img, vertexCoord, B, m, renderObj, w = (1, 1), randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    texCoef = texParam[:m.texEval.size]
    lightCoef = texParam[m.texEval.size:].reshape(9, 3)
        
    indexData = m.face.astype(np.uint16)
    
    texture = generateTexture(vertexCoord, np.r_[texCoef, lightCoef.flatten()], m)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    rendering, pixelCoord = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)[:2]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = pixelCoord.shape[0]
    
    rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    return np.r_[w[0] / numPixels * (rendering - img).flatten('F'), w[1] * texCoef ** 2 / m.texEval]

def textureLightingJacobian(texParam, img, vertexCoord, B, m, renderObj, w = (1, 1), randomFaces = None):
    texCoef = texParam[:m.texEval.size]
    lightCoef = texParam[m.texEval.size:].reshape(9, 3)
        
    indexData = m.face.astype(np.uint16)
    vertexColor = m.texMean + np.tensordot(m.texEvec, texCoef, axes = 1)
    texture = generateTexture(vertexCoord, np.r_[texCoef, lightCoef.flatten()], m)
    
    renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T].astype(np.float32), indexData)
    renderObj.resetFramebufferObject()
    renderObj.render()
    pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(img.shape[1], img.shape[0], return_info = True)[2:]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = pixelFaces[randomFaces]
        pixelBarycentricCoords = pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = pixelFaces.size
        
    pixelVertices = indexData[pixelFaces, :]
    
    pixelTexture = barycentricReconstruction(vertexColor, pixelFaces, pixelBarycentricCoords, indexData)
    pixelSHBasis = barycentricReconstruction(B, pixelFaces, pixelBarycentricCoords, indexData)
    J_lightCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(m.texEvec[c].T, pixelFaces, pixelBarycentricCoords, indexData)
        pixelSHLighting = barycentricReconstruction(np.dot(lightCoef[:, c], B), pixelFaces, pixelBarycentricCoords, indexData)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]
    
    texCoefSide = np.r_[w[0] / numPixels * J_texCoef, w[1] * np.diag(texCoef / m.texEval)]
    lightingCoefSide = np.r_[w[0] / numPixels * block_diag(*J_lightCoef), np.zeros((texCoef.size, lightCoef.size))]
    
    return np.c_[texCoefSide, lightingCoefSide]

"""
Vertex-based methods
"""
def textureCostV(texCoef, x, mask, m, w = (1, 1)):
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

def textureGradV(texCoef, x, mask, m, w = (1, 1)):
    """
    Jacobian for texture energy
    """
    Xcol = (m.texMean[:, mask] + np.tensordot(m.texEvec[:, mask, :], texCoef, axes = 1)).T
    
    r = (Xcol - x).flatten()
    
    # Jacobian
    J = m.texEvec[:, mask, :].reshape((m.texMean[:, mask].size, m.texEval.size), order = 'F')
    
    return 2 * (w[0] * np.dot(J.T, r) / mask.size + w[1] * texCoef / m.texEval)

def textureLightingCostV(texParam, x, mask, B, m, w = (1, 1), option = 'tl', constCoef = None):
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
        I[c, :] = np.dot(lightCoef[:, c], B[:, mask]) * texture[c, :]
    
    r = (I.T - x).flatten()
    
    Ecol = np.dot(r, r) / mask.size
    
    # Statistical regularization
    Ereg = np.sum(texCoef ** 2 / m.texEval)
    
    if option is 'l':
        return w[0] * Ecol
    else:
        return w[0] * Ecol + w[1] * Ereg

def textureLightingGradV(texParam, x, mask, B, m, w = (1, 1), option = 'tl', constCoef = None):
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
        J_texCoef[c*mask.size: (c+1)*mask.size, :] = np.dot(lightCoef[:, c], B[:, mask])[:, np.newaxis] * m.texEvec[c, mask, :]
        J_lightCoef[c*9: (c+1)*9, :] = texture[c, :] * B[:, mask]
        I[c, :] = np.dot(lightCoef[:, c], B[:, mask]) * texture[c, :]
    
    r = (I - x.T)
    
    if option is 'tl':
        return 2 * w[0] * np.r_[r.flatten().dot(J_texCoef), J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] / mask.size + np.r_[2 * w[1] * texCoef / m.texEval, np.zeros(27)]

    # Texture only
    elif option is 't':
        return 2 * (w[0] * r.flatten().dot(J_texCoef) / mask.size + w[1] * texCoef / m.texEval)
    
    # Light only
    elif option is 'l':
        return 2 * w[0] * np.r_[J_lightCoef[:9, :].dot(r[0, :]), J_lightCoef[9: 18, :].dot(r[1, :]), J_lightCoef[18: 27, :].dot(r[2, :])] / mask.size