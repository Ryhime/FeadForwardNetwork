import numpy as np
import math
class Layer:
    def __init__(self,inputDim,outputDim,actFn):
        self.weights = np.matrix(np.random.rand(outputDim,inputDim))
        self.biases = np.matrix(np.random.rand(outputDim)).T
        self.activationFn = np.vectorize(actFn)
    
    def forwardPass(self,input,returnActLayer=False):
        # Lambda(Ax+b)
        out = self.activationFn(self.weights*input+self.biases)
        return out
class Network:
    def __init__(self,layerDims,learningRate=.0001):
        self.layers = []
        for i in range(1,len(layerDims)):
            self.layers.append(Layer(layerDims[i-1],layerDims[i],self.relu))
        self.learningRate = learningRate

    def relu(self,x):
        #func = lambda x: 1.0/(1+math.e**(-x))

        func = lambda x: x if x>0 else 0
        return np.vectorize(func)(x)
    
    def deRelu(self,x):
        #func = lambda x: (1.0/(1+math.e**(-x)))*(1-1.0/(1+math.e**(-x)))

        func = lambda x: 1 if x>0 else 0
        return np.vectorize(func)(x)
    
    def loss(self,predicted,actual):
        return .5*(sum(actual-predicted))**2
    
    def deLoss(self,predicted,actual):
        return predicted-actual

    def forwardPass(self,input,allOutputs=False):
        outs = [input]
        for i in range(len(self.layers)):
            out = np.matrix(self.layers[i].forwardPass(outs[-1]))
            outs.append(out)
        return outs if allOutputs else outs[-1]

    def computeLoss(self,inputs,actuals):
        totalLoss = 0
        for i in range(len(inputs)):
            pred = self.forwardPass(inputs[i])
            totalLoss+=self.loss(pred,actuals[i])
        return totalLoss
    
    def processBatch(self,input,actual):
        # Get change in loss of function
        prediction = list(reversed(self.forwardPass(input,True)))
        layersBack = list(reversed(self.layers))
        for i,layer in enumerate(layersBack):
            # Go through the weights
            # Loop over output neurons
            for row in range(layer.weights.shape[0]):
                outputNeuron = prediction[i][row]
                if (i==0):
                    deLoss = np.sum(self.deLoss(outputNeuron,actual[row]))
                else:
                    # Sum of weights of previous layer
                    deLoss = 0
                    targetWeights = layersBack[i-1].weights.T[row]
                    for k in range(len(targetWeights)):
                        deLoss+=np.sum(self.deLoss(prediction[0][k],actual[k]))*targetWeights[0,k]*self.deRelu(prediction[0][k])
                    

                deRelu = self.deRelu(outputNeuron)
                # Loop over input neurons
                for col in range(layer.weights.shape[1]):
                    currWeight = layer.weights.item((row,col))
                    inputNeuron = prediction[i+1][col]
                    deWeight = inputNeuron
                    layer.weights[row,col] = currWeight-self.learningRate*deLoss*deRelu*deWeight

                layer.biases[row,0] = layer.biases[row,0]-self.learningRate*deLoss*deRelu



    def epoch(self,inputs,actuals):
        for i in range(0,len(inputs)):
            self.processBatch(inputs[i],actuals[i])
        
if (__name__=="__main__"):
    # Want to predict function |x|
    net = Network([1,2,1],learningRate=.0001)
    xs = []
    ys = []
    for i in range(-100,100,1):
        k = i/100.0
        xs.append(np.matrix([k]).T)
        ys.append(np.matrix([abs(k)]).T)

    print(ys[0])

    xs = np.array(xs)
    ys = np.array(ys)


    for i in range(1000):
        net.epoch(xs,ys)
        print(net.computeLoss(xs,ys))
    print(net.layers[0].weights)
    print(net.layers[1].weights)
    print(net.forwardPass(np.matrix([.3]).T))
    print(net.forwardPass(np.matrix([-.4]).T))