# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:42:30 2020

@author: Tomek
"""
import numpy as np
import math
import pickle

class NeuronNetwork:
    def __init__(self, X, y, eta=0.2, beta=1, network=[2,1], loadWeightsFrom=None, saveWeightsTo=None):
        self.X = X
        self.y = y;
        self.w = np.random.rand(1, np.shape(X)[1] + 1)
        self.weights = [] #2d array with weights vectors for all layers and neurons
        self.eta = eta
        self.beta = beta
        self.resoults = []
        self.saveWeightsTo = saveWeightsTo
        
        if loadWeightsFrom == None:
            for num, layer in enumerate(network):
                array = []
                for neuron in range(layer):
                    if num == 0:
                        array.append(np.random.rand(1, np.shape(X)[1] + 1))
                    else:
                        array.append(np.random.rand(1, network[num-1] + 1))
                print(array)
                
                self.weights.append(array)
            if saveWeightsTo != None:
                self.saveWeights()
        else:
            self.weights = pickle.load( open( loadWeightsFrom, "rb" ) )
    def saveWeights(self):
        pickle.dump( self.weights, open( self.saveWeightsTo, "wb" ) )
                    
            
    def predict(self,x,w):
        x = np.append(x,1)
        s = float(np.dot(w, x))
        wy = (float)(1/(1 + pow(math.e,-1 * self.beta * s)))
        return [s,wy]
    def learn(self):
        for sample in self.X: #for all samples
            layer_array = []
            for num, layer in enumerate(self.weights):
                array = []
                for index, neuron_weights in enumerate(layer):
                    if num == 0:
                        neuron_resoult = self.predict(sample, neuron_weights)
                    else:
                        neuron_resoult = self.predict(layer_array[num-1][1], neuron_weights)
                    array.append(neuron_resoult)
                layer_array.append(array)
                
            for num in reversed(layer_array):
                print(num)
                
                
            #Tu powinna nastąpić korekcja wag dla danej próbki
            #print((layer_array))                       
                        
                    
        
        
        
        

#Neuron = NeuronNetwork([[1,2,3,4,5],[2,3,4,5,6]],[3,4])
Neuron = NeuronNetwork([[1,2,3,4,5]],[3,4], saveWeightsTo="weights", loadWeightsFrom="weights")
#print(Neuron.w)
        
#print(Neuron.predict([3,4,5,6,8]))     
Neuron.learn() 
#print((Neuron.resoults))



        