#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from .transform import rotMat2angle, sh9
from sklearn.preprocessing import normalize

def generateFace(param, model, ind = None):
    """
    Generate vertices based off of eigenmodel and vector of parameters
    """
    # Shape eigenvector coefficients
    idCoef = param[: model.numId]
    expCoef = param[model.numId: model.numId + model.numExp]
    
    # Rotation Euler angles, translation vector, scaling factor
    R = rotMat2angle(param[model.numId + model.numExp:][:3])
    t = param[model.numId + model.numExp:][3: 6]
    s = param[model.numId + model.numExp:][6]
    
    # The eigenmodel, before rigid transformation and scaling
    if ind is None:
        model = model.idMean + np.tensordot(model.idEvec, idCoef, axes = 1) + np.tensordot(model.expEvec, expCoef, axes = 1)
    else:
        model = model.idMean[:, ind] + np.tensordot(model.idEvec[:, ind, :], idCoef, axes = 1) + np.tensordot(model.expEvec[:, ind, :], expCoef, axes = 1)
    
    # After rigid transformation and scaling
    return s*np.dot(R, model) + t[:, np.newaxis]

def generateTexture(vertexCoord, texParam, model):
            
    texCoef = texParam[:model.texEval.size]
    shCoef = texParam[model.texEval.size:].reshape(9, 3)
    
    texture = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    
    # Evaluate spherical harmonics at face shape normals
    vertexNorms = calcNormals(vertexCoord, model)
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
    
    I = np.empty((3, model.numVertices))
    for c in range(3):
        I[c, :] = np.dot(shCoef[:, c], sh) * texture[c, :]
    
    return I

def barycentricReconstruction(vertices, pixelFaces, pixelBarycentricCoords, indexData):
    pixelVertices = indexData[pixelFaces, :]
    
    if len(vertices.shape) is 1:
        vertices = vertices[np.newaxis, :]
    
    numChannels = vertices.shape[0]
        
    colorMat = vertices[:, pixelVertices.flat].reshape((numChannels, 3, pixelFaces.size), order = 'F')
    return np.einsum('ij,kji->ik', pixelBarycentricCoords, colorMat)

def calcNormals(vertices, model):
    """
    Calculate the per-vertex normal vectors for a model given shape coefficients
    """
    faceNorm = np.cross(vertices[:, model.face[:, 0]] - vertices[:, model.face[:, 1]], vertices[:, model.face[:, 0]] - vertices[:, model.face[:, 2]], axisa = 0, axisb = 0)
    
    vNorm = np.array([np.sum(faceNorm[faces, :], axis = 0) for faces in model.vertex2face])
    
    return normalize(vNorm)

def subdivide(v, f):
    """
    Use Catmull-Clark subdivision to subdivide a quad-mesh, increasing the number of faces by 4 times. Input the vertices and the face-vertex index mapping.
    """
    from collections import defaultdict
    from itertools import chain, compress

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
    for faceInd, edgesOnFace in enumerate(face2edge):
        for edge in edgesOnFace:
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
        print('Calculating new vertices for tester %d' % (tester + 1))
        # Face points: the mean of the vertices on a face
        facePt = np.array([np.mean(v[tester, vertexInd, :], axis = 0) for vertexInd in f])
        
        # Edge points
        edgePt = np.empty((len(edges), 3))
        for i, edge in enumerate(edges):
            # If an edge is only associated with one face, then it is on a border of the 3D model. The edge point is thus the midpoint of the vertices defining the edge.
            if len(edge2face[edge]) == 1:
                edgePt[i, :] = np.mean(v[tester, list(edge), :], axis = 0)
            
            # Else, the edge point is the mean of (1) the face points of the two faces adjacent to the edge and (2) the midpoint of the vertices defining the edge.
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
            
        vNew[tester, :, :] = np.r_[facePt, edgePt, newPt]
    
    # Form the new faces
    fNew = np.c_[f.flatten() + facePt.shape[0] + edgePt.shape[0], edgeInd + facePt.shape[0], np.repeat(np.arange(facePt.shape[0]), 4), edgeInd.reshape((edgeInd.shape[0]//4, 4))[:, [3, 0, 1, 2]].flatten() + facePt.shape[0]] + 1
    
    return vNew, fNew