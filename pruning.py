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

#Print Pruning degree
def degree(parameters):
	n=0
	num=0
	L=len(parameters)//2
	for l in range(1, L):
		N=parameters['W'+str(l)]
		for i in N:
			for v in i:
				num+=1
				if v==0:
					n+=1
	degree=n/num*100
	print("Parameter pruning degree: ", round(degree,3),"%")
	return degree

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

def acc0(a):
	n=0
	for i in a:
		for j in i:
			if j == 0:
				n+=1
	return n

def array0(a):
	b=[]
	if acc0(a)>1:
		b.append(0)
	for i in a:
		for j in i:
			if j != 0:
				b.append(j)
	return b

prun_parameter = np.load(f_parameters, allow_pickle='TRUE').item()
#Hyper-parameters
h_threshold=5
delta=0.1
learning_rate=0.05
num_iterations=220

degrees=[]
costs=[]
for i in range(1, 40):
	#Make mask
	mask_w=[]
	for l in range(1,L):
		a=prun_parameter['W'+str(l)]
		b=array0(a)
		threshold=np.percentile(np.abs(b), h_threshold)
		ms=np.abs(a)>threshold
		if i>1:
			ms=np.multiply(ms,mask_w_l[l-1])# Multiply by the mask matrix of the previous step
		mask_w.append(ms)
		# print("threshold "+str(l)+": "+str(threshold))# print threshold
	mask_w_l=mask_w# Save the mask matrix of the previous step
	#Pruning parameters
	prun_parameter=prun(prun_parameter, mask_w)#parameters pruning
	print("\n "+str(i)+" Pruning parameters: ")
	n=degree(prun_parameter)
	degrees.append(n)
	accuracy(prun_parameter)
	#Retrain parameters
	retrain_parameters, cost = retrain(prun_parameter, train_x, train_y, learning_rate, num_iterations)
	costs.append(cost)
	print("\nPruning and retrain parameters: ")
	accuracy(retrain_parameters)
	np.save('datasets/prun_parameter_1/prun_parameters'+str(i)+'.npy', retrain_parameters)
	#Iterative pruning (Not used during the experiment)
    # h_threshold+=delta*i
	# h_threshold+=5

degree_costs={'degree':degrees,'costs':costs}
np.save('datasets/degree_costs.npy', degree_costs)