import numpy as np
import gzip
import urllib.request

def load_Mnist():
    url_base = "https://yann.lecun.com/exdb/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    def download_load(url,filename):
        filepath, _ = urllib.request.urlretrieve(url + filename, filename)  #filepath: path of file stored, _: additional info like headers used as a unwanted variable
        with gzip.open(filename,"rb", ) as f:
            if 'image' in filename:
                return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28) / 255.0   # reshape to 2D array -1 will calc the amount of rows automaticly, 28*28 is the size cuz pixels
            elif 'labels' in filename:
                return np.frombuffer(f.read(), np.uint8, offset=8) #f.read:reads the content of the file to memory as binary, frombuffer: converts to np array, unit8: 8 bit integers, offset: skip the first part of file which is header info
        
    data = {}

    for key, filename in files.item():
        data[key] = download_load(url_base,filename)
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

x_train, y_train, x_test, y_test = load_Mnist()

print('Training data shape:', x_train.shape)
print('Training labels shape:', y_train.shape)
print('Test data shape:', x_test.shape)
print('Test labels shape:', y_test.shape)

class layerDense:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs,self.weights) + self.bias
    
    # def backward(self,input):

class ReLu:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs) 

class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outpts = exp / np.sum(exp, axis=1, keepdims=True)

class Loss:
    def calc(self, output, y):
        loss = self.forward(output, y)
        return np.mean(loss)  #loss across the entire batch
    
class CatergorialCrossEntropyLoss(Loss): #subclass of loss
    def forward(self,y_pred,y_real):
        numb_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7) #clip to prevent log(0)
        confidence = y_pred_clipped[range(numb_samples), y_real] #how confidence it is on the correct answer
        neg_log_likelihood = -np.log(confidence) #finds loss cuz when log(1) -> neg close to 0 numb; when log(0.1) -> neg numb close to -one
        return neg_log_likelihood

layer1 = layerDense(784,128)
activation1 = ReLu()
layer2 = layerDense(128,10)
activation2 = Softmax()
lossFunction = CatergorialCrossEntropyLoss()

epochs = 10
learningRate = 0.1

 
asd