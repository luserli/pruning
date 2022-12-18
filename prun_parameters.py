import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from nn_functions import *

f1_parameters = 'datasets/parameters.npy'
prun_parameter = 'datasets/prun_parameter/prun_parameters20.npy'
f1_costs = 'datasets/prun_costs.npy'
parameters = np.load(f1_parameters, allow_pickle='TRUE').item()
prun_parameters = np.load(prun_parameter, allow_pickle='TRUE').item()
# print(parameters)
# print(prun_parameters)

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
	print("Parameter pruning degree: ", round(n/num*100,3),"%")

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

print('Ori parameter: ')
degree(parameters)
accuracy(parameters)
print('Pruned parameter: ')
degree(prun_parameters)
accuracy(prun_parameters)

fname='datasets/degree_costs.npy'
degree_costs = np.load(fname, allow_pickle='TRUE').item()

degree=degree_costs['degree']
costs=degree_costs['costs']
plt.plot(degree,costs)
plt.ylabel('cost')
plt.xlabel('pruning degree(%)')
# plt.show()

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
    my_predicted_image = predict(prun_parameters, my_image)#预测结果
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
plt.show()
