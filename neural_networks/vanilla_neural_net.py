import numpy as np
import mnist_loader

"""
TO-DO: Implement the verifying the test data
"""
class NeuralNet:
    def __init__(self,layers):
        """
        A class to create a neural network object when provided the layer definitions as an array.
        The following are the parameters of the neural network objects:
            layers - a list of numbers, which represent the number of neurons in each layer(param)  
            numberOfLayers - a number, that stores the length of the 'layers' param
            weights - weights of the form Wij where 'i' is the number of neurons in the l+1^th layer and 
                        'j' is the number of neurons in the l^th layer, randomly generated numbers from
                        gaussian distribution with mean 0 and standard deviation 1
            biases  - biases of the form bj where 'j' is the number of nerons of the l^th layer,randomly 
                        generated numbers from gaussian distribution with mean 0 and standard deviation 1
                        note: it should be noted that the biases dont exist for the first layer as it 
                        would be the input for the first layer
        """
        self.layers = layers
        self.numberOfLayers = len(layers)
        self.weights = [np.random.randn(x,y) for x,y in zip(layers[1:],layers[:-1])]
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
    
    def train(self,minibatch_size,lr):
        """
        A method to train the neural network 
        parameters of the method:
            minibatch_size - an integer, the number of elements to batch in the training data
            lr - a float, used to represent the learning rate 
        
        Explanation:
            -the first for loop caters the starting index for each minibatch , we batch the X and y of our 
            data based on the minibatch size
            -some variables to notice  are the cost_minibatch, out_activs : out_activs list store the output 
            activations(final layer) for each training example. cost_minibatch stores the cost of each batch
            -the interm_grad_weights and interm_grad_biases are to store the gradient weights and gradient 
            biases for each training example, we initialize it with zero and add it at the end of backpropogation
            for each example
            -the second for loop just extracts the individual examples for the each minibatch           
            -activs_nn store the activations for each layer,zs store the weighted input for each layer
            -the third for loop just implements the forward propogation through the layers until we obtain the activations of the 
            output layer
            -the second for loop also calls the backpropogation function which returns the gradients of weights and biases
            after each training example which we then add the gradients to the interm_grad_weights and interm_grad_biases so that we 
            could later average the it for each minibatch
            -After each minibatch we just update the weights using the negative of gradients so that we move in the best possible 
            direction to reduce the cost
        """
        X,Y = mnist_loader.load_data("train")
        for batch_strt_idx in range(0,len(X),minibatch_size):
            minibatch_X = X[batch_strt_idx:batch_strt_idx+minibatch_size]
            minibatch_Y = Y[batch_strt_idx:batch_strt_idx+minibatch_size]
            cost_minibatch = []
            out_activs = []
            interm_grad_weights = [np.zeros(w.shape) for w in self.weights]
            interm_grad_biases = [np.zeros(b.shape) for b in self.biases]
            for x,y in zip(minibatch_X,minibatch_Y):
                activs_nn = [x]
                zs = []
                #feedforward
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
        """
        A function that backpropogates for each training example and calculates the errors for each layers and the
        gradients for the weights
        the parameters of the method are
        -activatiosn: list of the activations of the neural net for that training example
        -y:the actual output for the training example
        -zs: weighted inputs for each layer of the neural network for that training example

        Explanation:
        note: The errors are referred to as del throughout this method
        - The first step is to calculate the error of the output layer which is propogated backward through the network,
        as we have the variables necessary, we compute it and it to the delta list
        - This is actually called as backpropogation as the error of the output layer  L is multiplied with the weights of the 
        that connects L-1 layer with the L layer of the neural network  , the error of the last layer is already added and so the
        first for loop just calulates the error of the hidden layers
        -The second for loop uses the del values computed previously to compute the gradient change for the weights and biases and
        returns them for the training example that has been called for
        """
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
    """
    Implementation of the Mean Squared Error
    parameters are:
    -y:true output
    -a:obtained output(activation of last layer)
    """
    squared = np.square(y-a)
    cost_vect = np.apply_along_axis(np.mean,0,squared)
    return(cost_vect.mean())


    

def sigmoid(z):
    """
    Implementation of the sigmoid activation function
    parameters:
    -z:weighted input
    """
    return (1.0/(1.0+np.exp(-z)))



def main():
    nn = NeuralNet([784,100,30,10])
    nn.train(50,0.5)



if __name__ =='__main__':
    main()

