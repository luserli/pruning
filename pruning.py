import os
from PIL import Image
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
	costs = []
	with tqdm(total=num_iterations) as t:
		for i in range(0, num_iterations):
			AL, caches = forward_f(X, parameters)
			cost = np.squeeze(forward.cost_function(AL, Y))
			grads = backward.backward_function(AL, Y, caches)
			parameters = update_parameters(parameters, grads, learning_rate)
			parameters = prun(parameters, mask_w)
			costs.append(cost)
			t.set_description('Retrain %i' % i)
			t.set_postfix(cost=cost,learning_rate=learning_rate)
			t.update(1)

	return parameters, costs

#Return the retention of parameters after pruning
def parameter_retention(parameters):
	N=[]
	for n in range(1,L):
		nl=parameters['W'+str(l)]!=0
		N.append(nl)
	n=0
	num=0
	for i in N:
		for j in i:
			for v in j:
				num+=1
				if v == True:
					n+=1
	print("\nThe retention of parameters after pruning: ", round(n/num*100,2),"%")
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

prun_parameter = np.load(f_parameters, allow_pickle='TRUE').item()

#Make mask
mask_w=[]
threshold=np.percentile(prun_parameter['W'+str(l)], 100)
for l in range(1,L):
	ms=prun_parameter['W'+str(l)]<threshold
	mask_w.append(ms)

#Pruning parameters
prun_parameter=prun(prun_parameter, mask_w)#parameters pruning

print("\nPruning parameters: ")
accuracy(prun_parameter)

learning_rate=0.1
num_iterations=3000

retrain_parameters , costs= retrain(prun_parameter, train_x, train_y, learning_rate, num_iterations)
parameter_retention(retrain_parameters)

print("\nPruning and retrain parameters: ")
accuracy(retrain_parameters)

#save parameters
np.save('datasets/prun_parameters.npy', retrain_parameters)
np.save('datasets/prun_costs.npy', costs)

#图片验证
Imgs = []
sexs = []
Classes = []
i = 0

list = os.listdir('photos_demo')
list.sort(key=lambda x: int(x.replace("","").split('.')[0]))#按顺序排列

for fname in list:
    dir = 'photos_demo/' + fname
    Imgs.append(dir)
    image = Image.open(dir).resize((80, 45))
    image = np.array(image)
    my_image = np.array(image.reshape((1, train_x.shape[0])).T)#转为向量
    my_predicted_image = predict(retrain_parameters, my_image)#预测结果
    sex = np.squeeze(my_predicted_image)
    sexs.append(sex)
    if sex==1:
        classes = 'car'
    else:
        classes = 'noncar'
    Classes.append(classes)
    # print('The picture \"' + str(fname) + "\": y = " + str(sex) + ", your algorithm predicts a \"" + str(classes) + "\" picture.")
    i += 1

#显示判断结果
plt.figure(figsize=(13, 8))
numpx = int(len(Imgs) / 4) if len(Imgs) % 4 == 0 else int(len(Imgs) / 4 + 1)
for i in range(len(Imgs)):
    image = Image.open(Imgs[i])
    plt.subplot(numpx, 4, i+1)
    plt.imshow(image)
    plt.title('It\'s a \"' + str(Classes[i]) + '\"'+'('+ str(sexs[i]) +')'+'picture')
    plt.axis('off')

#cost figure
plt.figure()
plt.subplot(1, 1, 1)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate ="+str(learning_rate))
# f = plt.gcf()  #获取当前图像
# f.savefig(r'./photos/pruning_costs.png')
plt.show()