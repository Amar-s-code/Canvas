import gzip
import numpy as np


def load_data(out="train"):
    dim_images = 28
    if out=="test":
        image_file = "D:\\Amar-s-code\\Datasets\\MNIST\\t10k-images-idx3-ubyte.gz"
        label_file = "D:\\Amar-s-code\\Datasets\\MNIST\\t10k-labels-idx1-ubyte.gz"
        number_images = 10000
    else:
        image_file = "D:\\Amar-s-code\\Datasets\\MNIST\\train-images-idx3-ubyte.gz"
        label_file = "D:\\Amar-s-code\\Datasets\\MNIST\\train-labels-idx1-ubyte.gz"
        number_images = 50000

    training_images_file = gzip.open(image_file,'r')
    training_images_file.read(16)
    buffer = training_images_file.read(dim_images*dim_images*number_images)
    data = np.frombuffer(buffer,dtype=np.uint8).astype(np.float32)
    data = data.reshape(number_images,dim_images*dim_images,1)

    training_labels = gzip.open(label_file,'r')
    training_labels.read(8)
    buffer = training_labels.read(number_images)
    labels = np.frombuffer(buffer,dtype=np.uint8).astype(np.int64)
    labels = np.eye(10)[labels]
    labels = labels[...,np.newaxis]

    return (data,labels)
    



