#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mm.utils.opengl import Render
from mm.utils.io import importObj
from mm.utils.transform import rotMat2angle
from mm.models import MeshModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    
    frame = 0
    img = Image.open('../data/obama/orig/%05d.png' % (frame + 1))
    width, height = img.size
    img = np.array(img).astype(np.float32) / 255
    
    vertexCoords, indexData = importObj('../data/obama/shapes/%05d.obj' % (frame + 1), dataToImport = ['v', 'f'])
    indexData -= 1
    
    RTS = np.load('../data/obama/RTS.npy')
    # The Euler angles for the rotation matrix are the first 3 columns
    eulerAngles = RTS[frame, :3]
    # The translation vector is the next 3 columns
    T = RTS[frame, 3: 6]
    # The scale factor is the last column
    S = RTS[frame, 6]
    
    vertexCoords = S * np.dot(vertexCoords, rotMat2angle(eulerAngles).T) + T
    
    m = MeshModel('../models/bfm2017.npz')
    vertexColors = m.texMean.T
    
    meshData = np.r_[vertexCoords, vertexColors]
    
    r = Render(width, height, meshData, indexData, indexed = False, img = None)
    r.render()

    rendering = r.grabRendering()
    rendering, pixelCoord, pixelFaces, pixelBarycentricCoords = r.grabRendering(return_info = True)
    plt.figure()
    plt.imshow(rendering)
    
    for frame in range(1, 52, 10):
        img = Image.open('../data/obama/orig/%05d.png' % (frame + 1))
        img = img.tobytes()
        
        vertexCoords = importObj('../data/obama/shapes/%05d.obj' % (frame + 1), dataToImport = ['v'])
        eulerAngles = RTS[frame, :3]
        T = RTS[frame, 3: 6]
        S = RTS[frame, 6]
        
        vertexCoords = S * np.dot(vertexCoords, rotMat2angle(eulerAngles).T) + T
        
        meshData = np.r_[vertexCoords, vertexColors]
        
        r.updateVertexBuffer(meshData)
        r.resetFramebufferObject()
        r.render()
        
        rendering = r.grabRendering()
        plt.figure()
        plt.imshow(rendering)