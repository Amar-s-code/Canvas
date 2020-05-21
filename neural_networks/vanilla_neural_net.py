import numpy as np
import mnist_loader


class NeuralNet:
    def __init__(self,sizes):
        self.sizes = sizes
        self.numberOfLayers = len(sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
    
    def train(self,minibatch_size):
        X,Y = mnist_loader.load_data("train")
        for batch_strt_idx in range(0,len(X),minibatch_size):
            minibatch_X = X[batch_strt_idx:batch_strt_idx+minibatch_size]
            minibatch_Y = Y[batch_strt_idx:batch_strt_idx+minibatch_size]
            cost_minibatch = []
            out_activs = []
            for x,y in zip(minibatch_X,minibatch_Y):
                #feedforward
                activs_nn = [x]
                zs = []
                for weight, bias in zip(self.weights,self.biases):
                    z = np.matmul(weight,activs_nn[-1]) + bias
                    a = sigmoid(z)
                    activs_nn.append(a)
                out_activs.append(activs_nn[-1])
            cost_minibatch.append(cost(minibatch_Y,out_activs))
            print(cost_minibatch)    
                    
                        
                

        


def cost(y,a):
    squared = np.square(y-a)
    cost_vect = np.apply_along_axis(np.mean,0,squared)
    return(cost_vect.mean())


    

def sigmoid(z):
    return (1.0/(1.0+np.exp(-z)))



def main():
    nn = NeuralNet([784,30,10])
    nn.train(5000)

if __name__ =='__main__':
    main()

