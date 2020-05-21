import numpy as np
import mnist_loader


class NeuralNet:
    def __init__(self,sizes):
        self.sizes = sizes
        self.numberOfLayers = len(sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
    
    def train(self,minibatch_size,lr):
        X,Y = mnist_loader.load_data("train")
        for batch_strt_idx in range(0,len(X),minibatch_size):
            minibatch_X = X[batch_strt_idx:batch_strt_idx+minibatch_size]
            minibatch_Y = Y[batch_strt_idx:batch_strt_idx+minibatch_size]
            cost_minibatch = []
            out_activs = []
            interm_grad_weights = [np.zeros(w.shape) for w in self.weights]
            interm_grad_biases = [np.zeros(b.shape) for b in self.biases]
            for x,y in zip(minibatch_X,minibatch_Y):
                #feedforward
                activs_nn = [x]
                zs = []
                for weight, bias in zip(self.weights,self.biases):
                    z = np.matmul(weight,activs_nn[-1]) + bias
                    zs.append(z)
                    a = sigmoid(z)
                    activs_nn.append(a)
                gw,gb = self.backpropagate(activs_nn,y,zs)
                interm_grad_weights = [i_gweight+gweight for i_gweight,gweight in zip(interm_grad_weights,gw)] 
                interm_grad_biases = [i_gbias+gbias for i_gbias,gbias in zip(interm_grad_biases,gb)] 
                out_activs.append(activs_nn[-1])
            grad_weights = [w/minibatch_size for w in interm_grad_weights]
            grad_biases = [b/minibatch_size for b in interm_grad_biases]
            self.weights = [w-(lr*grw) for w,grw in zip(self.weights,grad_weights)]
            self.biases = [b-(lr*grb) for b,grb in zip(self.biases,grad_biases)]
            cost_minibatch.append(cost(minibatch_Y,out_activs))
            print(cost_minibatch)
            
            
    def backpropagate(self,activations,y,zs):
        delta = []
        del_L = (activations[-1]-y)*(sigmoid(zs[-1])*(1-sigmoid(zs[-1])))
        delta.append(del_L)
        grad_w = []
        grad_b = []
        for w,z in zip(reversed(self.weights),reversed(zs[:-1])):
            del_l = (np.matmul(w.transpose(),delta[0])) * (sigmoid(z)*(1-sigmoid(z)))
            delta = [del_l]+delta
        for d,a in zip(delta,activations[:-1]):
            grad_b.append(d)
            grad_w.append(np.matmul(d,a.transpose()))
        return (grad_w,grad_b)



def cost(y,a):
    squared = np.square(y-a)
    cost_vect = np.apply_along_axis(np.mean,0,squared)
    return(cost_vect.mean())


    

def sigmoid(z):
    return (1.0/(1.0+np.exp(-z)))



def main():
    nn = NeuralNet([784,100,30,10])
    nn.train(50,0.5)

    nn1  = NeuralNet([784,30,10])
    nn1.train(50,0.5)

if __name__ =='__main__':
    main()

