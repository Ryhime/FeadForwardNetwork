import math
import numpy as np

class FeedForward():
    def __init__(self,W,B,activationFn="None",finalActivation="None"):
        if (len(B)!=len(W)):
            print("The number of weights and biases must be the same.") 
            return None
        self.W = W
        self.B = B
        self.numLayers = len(B)
        self.activation = activationFn
        self.finalActivation = finalActivation

        self.activationLookUp = {
            "relu": self.relu,
            "tanh": self.tanh,
            "sigmoid": self.sigmoid,
            "softmax": self.softmax
        }
        

    def matrixMultiply(self,A,B):
        A = np.array(A)
        B = np.array(B)
        if (A.shape[1]!=B.shape[0]):
            return False

        BT = B.T
        C = np.zeros((int(A.shape[0]),int(BT.shape[0])))
        for i in range(len(A)):
            for k in range(len(BT)):
                
                sm = 0
                for j in range(len(A[i])):
                    sm+=(A[i][j]*BT[k][j])
                C[i][k] = sm
        return C

    def relu(self,X):
        reluFn = lambda x: x if x>=0 else 0
        for i in range(len(X)):
            X[i] = np.array(list(map(reluFn,X[i])))

        return X
    
    def tanh(self,X):
        tanhFn = lambda x: math.tanh(x)
        for i in range(len(X)):
            X[i] = np.array(list(map(tanhFn,X[i])))

        return X
    
    def sigmoid(self,X):
        sigmoidFn = lambda x: 1/(1+math.exp(-x))
        for i in range(len(X)):
            X[i] = np.array(list(map(sigmoidFn,X[i])))
        return X


    def softmax(self,inputVector):
        softmaxFn = lambda x: math.exp(x)
        softMaxApplied = np.array(list(map(softmaxFn,inputVector)))
        sm = sum(softMaxApplied)
        softMaxApplied = np.array(list(map(lambda x: x/sm,softMaxApplied)))
        return softMaxApplied
    
    def _applyLayerFn(self,X,layerNum):
        # lambda(AX+b)
        W = self.W[layerNum]
        B = self.B[layerNum]
        AX = self.matrixMultiply(W,X)

        if (self.activation in self.activationLookUp):
            return self.activationLookUp[self.activation](AX+B)
        
        return AX+B
    
    def predict(self,X):
        for i in range(self.numLayers):
            X = self._applyLayerFn(X,i)

        if (self.finalActivation in self.activationLookUp):
            return self.activationLookUp[self.finalActivation](X)

        return X
    


    
# Weights Set to represent a absolute value function
ff = FeedForward([[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],[[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]]],[0,0],activationFn="relu")
print(ff.predict([[-1],[105],[-25]]))
