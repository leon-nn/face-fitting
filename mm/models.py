#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os

class MeshModel:
    def __init__(self, modelFile, numIdEvecs = 80, numExpEvecs = 76, numTexEvecs = 80):
        
        model = os.path.splitext(os.path.basename(modelFile))[0]
        
        modelDict = np.load(modelFile)
        
        self.numId = numIdEvecs
        self.numExp = numExpEvecs
        self.numTex = numTexEvecs
        
        self.__dict__.update(modelDict)
        self.idEvec = self.idEvec[:, :, :self.numId]
        self.idEval = self.idEval[:self.numId]
        self.expEvec = self.expEvec[:, :, :self.numExp]
        self.expEval = self.expEval[:self.numExp]
        
        if model == 'bfm2017':
            self.texEvec = self.texEvec[:, :, :self.numTex]
            self.texEval = self.texEval[:self.numTex]
            
            # These are indices representing the OpenPose landmarks that we have a correspondence with for the BFM2017 model
            self.targetLMInd = np.array([0, 1, 2, 3, 8, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69])
            
            # These are vertex indices that correspond with the OpenPose landmark indices above
            self.sourceLMInd = np.array([16203, 16235, 16260, 16290, 27061, 22481, 22451, 22426, 22394, 8134, 8143, 8151, 8156, 6986, 7695, 8167, 8639, 9346, 2345, 4146, 5180, 6214, 4932, 4158, 10009, 11032, 12061, 13872, 12073, 11299, 5264, 6280, 7472, 8180, 8888, 10075, 11115, 9260, 8553, 8199, 7845, 7136, 7600, 8190, 8780, 8545, 8191, 7837, 4538, 11679])