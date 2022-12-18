from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from nn_functions import *

#pruning function
def prun(parameters, mask_w):
	for l in range(1,L):
		X = parameters['W'+str(l)]
		X = np.multiply(X, mask_w[l-1])
		parameters['W'+str(l)] = X
	return parameters

#Rewrite forward propagation function
def forward_f(X, parameters):
    A = X
    caches = []
    for l in range(1, L):
        A_l = A
        A, cache = forward.activation_forward(A_l, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
        caches.append(cache)
    
    AL, cache = forward.activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

#Retrain function
def retrain(parameters, X, Y, learning_rate, num_iterations):
	with tqdm(total=num_iterations) as t:
		for i in range(0, num_iterations):
			AL, caches = forward_f(X, parameters)
			cost = np.squeeze(forward.cost_function(AL, Y))
			grads = backward.backward_function(AL, Y, caches)
			parameters = update_parameters(parameters, grads, learning_rate)
			parameters = prun(parameters, mask_w)
			t.set_description('Retrain %i' % i)
			t.set_postfix(cost=cost)
			t.update(1)

	return parameters, cost

#Print accuracy
def accuracy(parameters):
	Y_prediction_train = predict(parameters, train_x)
	Y_prediction_test = predict(parameters, test_x)
	print("Training set accuracy："  , format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100) ,"%")
	print("Test set accuracy："  , format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100) ,"%")

##load dataset
datas = 'datasets/carvnocar.h5'
train_set_x, train_y, test_set_x, test_y = load_files(datas)
train_x = train_set_x / 255
test_x = test_set_x / 255

##load parameters
f_parameters = 'datasets/parameters.npy'
parameters = np.load(f_parameters, allow_pickle='TRUE').item()
L = len(parameters)//2 

print("Original parameters: ")
accuracy(parameters)

def array0(a):
	b=[]
	b.append(0)
	for i in a:
		for j in i:
			if j != 0:
				b.append(j)
	return b

prun_parameter = np.load(f_parameters, allow_pickle='TRUE').item()

h_threshold=1
delta=0.01
learning_rate=0.05
num_iterations=1000

n=h_threshold
degree=[]
costs=[]
for i in range(22):
	#Make mask
	mask_w=[]
	for l in range(1,L):
		b=array0(prun_parameter['W'+str(l)])
		threshold=np.percentile(b, 100-h_threshold)
		ms=prun_parameter['W'+str(l)]<threshold
		if i>1:
			ms=np.multiply(ms,mask_w_l[l-1])
		mask_w.append(ms)
		# print("threshold "+str(l)+": "+str(threshold))
	mask_w_l=mask_w
	#Pruning parameters
	prun_parameter=prun(prun_parameter, mask_w)#parameters pruning
	degree.append(n)
	print("\nParameter pruning degree: ", round(n,2),"%")
	print("\nPruning parameters: ")
	accuracy(prun_parameter)
	#Retrain parameters
	retrain_parameters, cost = retrain(prun_parameter, train_x, train_y, learning_rate, num_iterations)
	costs.append(cost)
	print("\nPruning and retrain parameters: ")
	accuracy(retrain_parameters)
	np.save('datasets/prun_parameter/prun_parameters'+str(n)+'.npy', retrain_parameters)
	n+=h_threshold
	# h_threshold+=delta*i

degree_costs={
	'degree':degree,
	'costs':costs
}
np.save('datasets/degree_costs.npy', degree_costs)