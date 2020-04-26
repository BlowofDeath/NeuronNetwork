# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:42:30 2020

@author: Tomek
"""
import numpy as np
import math
import pickle

class NeuronNetwork:
    def __init__(self, X, d = [[]], eta=0.1, beta=1, network=[2,1], loadWeightsFrom=None, saveWeightsTo=None):
        self.X = X
        self.d = d;
        self.w = np.random.rand(1, np.shape(X)[1] + 1)
        self.weights = [] #2d array with weights vectors for all layers and neurons
        self.eta = eta
        self.beta = beta
        self.saveWeightsTo = saveWeightsTo
        
        if loadWeightsFrom == None:
            for num, layer in enumerate(network):
                array = []
                for neuron in range(layer):
                    if num == 0:
                        
                        array.append(np.random.rand(np.shape(X)[1] + 1))
                    else:
                        array.append(np.random.rand(network[num-1] + 1))
                #print(array)
                
                self.weights.append(array)
            if saveWeightsTo != None:
                
                # print(self.weights)
                # a = np.array([0.3,0.1,0.2])
                # b = np.array([0.6,0.4,0.5])
                # c = np.array([0.9,0.7,-0.8])
                # self.weights = [[a,b],[c]]
                # print(self.weights)
                self.saveWeights()
                
        else:
            self.weights = pickle.load( open( loadWeightsFrom, "rb" ) )
            
    
    def saveWeights(self):
       
        pickle.dump( self.weights, open( self.saveWeightsTo, "wb" ) )
                    
            
    def predict(self,x,w):
        x = np.insert(x,0,1)
        s = float(np.dot(w, x))
        wy = (float)(1/(1 + pow(math.e,-1 * self.beta * s)))
        return [s,wy]
    def learn(self):
        for sample_num,sample in enumerate(self.X): #for all samples
            layer_array = []
            for num, layer in enumerate(self.weights):
                array = []
                for index, neuron_weights in enumerate(layer):
                    if num == 0:
                        neuron_resoult = self.predict(sample, neuron_weights)
                    else:
                        temp = np.array(layer_array[num-1])
                        temp = temp[:,1]
                        neuron_resoult = self.predict(temp, neuron_weights)
                    array.append(neuron_resoult)
                layer_array.append(array)
            
            
                
            for num, layer in reversed(list(enumerate(layer_array))):
                array = []
                for neuron_num, neuron in enumerate(layer):
                    if num == len(layer_array) - 1: #Ostatnia warstwa sieci
                        #corect
                        c = self.eta * (self.d[sample_num][neuron_num] - neuron[1]) # eta * (d - wy)
                        array.append([c])
                        print(neuron)
                    else: #reszta warstw sieci
                        print(neuron)
                        
                        
                print("a",array)
                print()
                    
                
                
            #Tu powinna nastąpić korekcja wag dla danej próbki
            #print((layer_array))                       
                        
                    
        
        
        
        

#Neuron = NeuronNetwork([[1,2,3,4,5],[2,3,4,5,6]],[3,4])
# Neuron = NeuronNetwork([[1,0]],[[3]], saveWeightsTo="weights_test")
Neuron = NeuronNetwork([[1,0]],[[1]], loadWeightsFrom="weights_test")
#print(Neuron.w)
        
#print(Neuron.predict([3,4,5,6,8]))     
Neuron.learn() 
#print((Neuron.resoults))



        