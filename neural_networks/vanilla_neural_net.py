import numpy as np

class NeuralNet:
    def __init__(self,sizes):
        self.sizes = sizes
        self.numberOfLayers = len(sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]


