'''
神经网络所需函数
'''
import h5py 
import numpy as np 
from tqdm import tqdm

##读取数据
def load_files(fdir):
    train_dataset = h5py.File(fdir, 'r')
    train_set_x_orig = np.array(train_dataset['X_train'][:])
    train_set_y_orig = np.array(train_dataset['y_train'][:]).T
    test_set_x_orig = np.array(train_dataset['X_test'][:])
    test_set_y_orig = np.array(train_dataset['y_test'][:]).T
    train_dataset.close()

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[1]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[1]))

    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

##Relu
class relu:
    def forward(Z):
        A = np.maximum(0,Z)
        cache = Z
        
        return A, cache

    def backward(dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

##sigmoid
class sigmoid:
    def forward(Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        
        return A, cache

    def backward(dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1-s)
        
        return dZ

##初始化
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.098
        parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1)

    return parameters

##正向传播
class forward:
    def linear_forward(A, W, b):
        Z = np.dot(W, A) + b
        cache  = (A, W, b)
        
        return Z, cache

    def activation_forward(A_l, W, b, activation):
        Z, linear_cache = forward.linear_forward(A_l,W,b)

        if activation == "sigmoid":
            A, activation_cache = sigmoid.forward(Z)
        elif activation == "relu":
            A, activation_cache = relu.forward(Z)
        
        cache = (linear_cache, activation_cache)
        
        return A, cache

    def forward_function(X, parameters):
        A = X
        caches = []
        L = len(parameters)//2 
        for l in range(1, L):
            A_l = A
            A, cache = forward.activation_forward(A_l, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
            caches.append(cache)
        
        AL, cache = forward.activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)

        return AL, caches

    def cost_function(AL, Y):
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL + 1e-5),axis=1,keepdims=True)
        cost = np.squeeze(cost)   
        
        return cost

##反向传播
class backward:
    def linear_backward(dZ, cache):
        A_l, W, b = cache
        m = A_l.shape[1]
        dW = 1 / m * np.dot(dZ ,A_l.T)
        db = 1 / m * np.sum(dZ,axis = 1 ,keepdims=True)
        dA_l = np.dot(W.T,dZ) 
        
        return dA_l, dW, db

    def activation_backward(dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu.backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = sigmoid.backward(dA, activation_cache)

        dA_l, dW, db = backward.linear_backward(dZ, linear_cache)
            
        return dA_l, dW, db

    def backward_function(AL, Y, caches):
        grads = {} 
        L = len(caches)
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL + 1e-5) - np.divide(1 - Y, 1-AL + 1e-5))
        
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backward.activation_backward(dAL, current_cache, "sigmoid")
        
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_l_temp, dW_temp, db_temp = backward.activation_backward(grads["dA" + str(l+2)], current_cache,  "relu")
            grads["dA" + str(l+1)] = dA_l_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads

##梯度下降
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        
    return parameters

#前向预测
def predict(parameters, X):
    probas, caches = forward.forward_function(X, parameters)
    p = np.round(probas)

    return p

#NN模型
def model(X, Y, X_test, Y_test, layers_dims, learning_rate, num_iterations):
    np.random.seed(2)
    costs = []
    parameters = initialize_parameters(layers_dims)
    with tqdm(total=num_iterations) as t:
        for i in range(0, num_iterations):
            AL, caches = forward.forward_function(X, parameters)
            cost = np.squeeze(forward.cost_function(AL, Y))
            grads = backward.backward_function(AL, Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            costs.append(cost)
            t.set_description('Training %i' % i)
            t.set_postfix(cost=cost,learning_rate=learning_rate)
            t.update(1)

    Y_prediction_train = predict(parameters, X)
    Y_prediction_test = predict(parameters, X_test)
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")

    parameters["costs"] = costs
    return parameters, learning_rate 