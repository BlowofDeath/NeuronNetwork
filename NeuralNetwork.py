# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:26:38 2020

@author: Tomek
"""

import numpy as np
import math
import pickle

class NeuralNetwork:
    def __init__(self, X, eta=0.1, beta=1, network=[2,1], loadWeightsFrom=None, saveWeightsTo=None, log = False, epochs = 10000):
        self.X = X[:,:-network[-1]]
        self.D = X[:,np.shape(self.X)[1]:]
        self.samples = X
        self.eta = eta
        self.beta = beta
        self.network = network
        self.layers = []
        self.log = log
        self.epochs = epochs
        self.loadWeightsFrom = loadWeightsFrom
        self.saveWeightsTo = saveWeightsTo
        
        
        for i,layer in enumerate(network):
            self.layers.append([])
            for neuron_i in range(layer):
                if(i == 0):
                    w = np.random.rand(len(self.X[0])+1)

                else:
                    w = np.random.rand(network[i-1]+1)

                self.layers[i] = np.append(self.layers[i], Neuron(w))

        if(loadWeightsFrom != None):
            self.loadWeights()
        
    def loadWeights(self):
        weights = pickle.load( open( self.loadWeightsFrom, "rb" ) )
        # weights = [[[0.3,0.1,0.2], [0.6,0.4,0.5]],[[0.9,0.7,-0.8]]]
        for l,layer in enumerate(self.layers):
            for n,neuron in enumerate(layer):
                neuron.w = weights[l][n]
    def saveWeights(self):
        weights = []
        for l,layer in enumerate(self.layers):
            layerTemp = []
            for n,neuron in enumerate(layer):
                layerTemp.append(neuron.w)
            weights.append(layerTemp)
        pickle.dump( weights, open( self.saveWeightsTo, "wb" ) )
        
              
                    
    def activation(self, s):
        return (1/(1 + pow(math.e,-1 * self.beta * s)))
    def deltaS(self,wy, c):
        return c * (self.beta*wy*(1-wy))
        
    
    def printNeuron(self,neuron):
        if(self.log==True):
            print("x",neuron.we)
            print("w", neuron.w)
            print("s:",neuron.s)
            print("wy:",neuron.wy)
                
    def predict(self, x):
        prevWy = []
        resoults = []
        for i,layer in enumerate(self.layers):
            if(i == 0):
                for neuron in layer:
                    neuron.we = np.insert(x,0,1)
                    neuron.s = np.dot(neuron.we, neuron.w)
                    neuron.wy = self.activation(neuron.s)
                    prevWy.append(neuron.wy)
                    self.printNeuron(neuron)
        
            else:
                tempWy = prevWy
                prevWy = []
                for neuron in layer:

                    neuron.we = np.insert(tempWy,0,1)
                    neuron.s = np.dot(neuron.we, neuron.w)
                    neuron.wy = self.activation(neuron.s)
                    prevWy.append(neuron.wy)
                    self.printNeuron(neuron)
                    if(len(self.layers)-1==i):
                        resoults = np.append(resoults,neuron.wy)
        return resoults
    def backpropagation(self,d):
        preWe = []
        for num, layer in reversed(list(enumerate(self.layers))):
            for i_n,neuron in enumerate(layer):
                if num == len(self.layers) - 1:
                    self.printNeuron(neuron)
                    c = self.eta * (d[i_n] - neuron.wy) # eta * (d - wy)
                    deltaS = self.deltaS(neuron.wy, c)
                    neuron.deltaW = neuron.we * deltaS
                    
                    if(preWe==[]):
                        preWe.append(np.delete(neuron.w,0) * deltaS)
                        
                    else:
                        preWe[len(self.layers)-1-num] += np.delete(neuron.w,0) * deltaS
                    
                else: #trzeba zaktualizować by działało z wieloma warstwami
                    deltaS = self.deltaS(neuron.wy, preWe[len(self.layers)-2-num][i_n])
                    neuron.deltaW = neuron.we * deltaS
    def improveWeights(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.w += neuron.deltaW
    def learn(self):
        for e in range(self.epochs):
            for sample in self.samples:
                predict = self.predict(sample[:-self.network[-1]])
                self.backpropagation(sample[np.shape(self.X)[1]:])
                self.improveWeights()
                self.saveWeights()
                print("Sample: ", sample)
                print("Resoults: ", predict)
                print("--------------------------")
            np.random.shuffle(self.samples)
            
            
       
class Neuron:
    def __init__(self, w):
        self.we = np.asarray([1],dtype="float64"),
        self.w = w
        self.s = None
        self.wy = None
        self.deltaW = []

        
        
        
X = np.array([[1,0,1],[0,1,1],[1,1,0],[0,0,0]], dtype="float64")       
        
Network = NeuralNetwork(X, saveWeightsTo="weights")
Network.learn()


