'''
调用训练好的参数对图片进行分类判断
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nn_functions import *

##读取数据
datas = 'datasets/carvnocar.h5'
train_set_x, train_y, test_set_x, test_y = load_files(datas)
train_x = train_set_x / 255
test_x = test_set_x / 255

##读取参数
fdir_parameters = 'datasets/parameters.npy'
fdir_data = 'datasets/data.npy'

parameters = np.load(fdir_parameters, allow_pickle='TRUE').item()
data = np.load(fdir_data, allow_pickle='TRUE').item()

print('模型训练时间: %s 秒' % data['times'])
#准确度
Y_prediction_train = predict(parameters, train_x)
Y_prediction_test = predict(parameters, test_x)
print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100) ,"%")
print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100) ,"%")

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
    my_predicted_image = predict(parameters, my_image)#预测结果
    sex = np.squeeze(my_predicted_image)
    sexs.append(sex)
    if sex==1:
        classes = 'car'
    else:
        classes = 'noncar'
    Classes.append(classes)
    #print('The picture \"' + str(fname) + "\": y = " + str(sex) + ", your algorithm predicts a \"" + str(classes) + "\" picture.")
    i += 1

#判断结果
plt.figure(figsize=(13, 10))
numpx = int(len(Imgs) / 4) if len(Imgs) % 4 == 0 else int(len(Imgs) / 4 + 1)
for i in range(len(Imgs)):
    image = Image.open(Imgs[i])
    plt.subplot(numpx, 4, i+1)
    plt.imshow(image)
    plt.title('It\'s a \"' + str(Classes[i]) + '\"'+'('+ str(sexs[i]) +')'+'picture')
    plt.axis('off')

#cost图
plt.figure()
plt.subplot(1, 1, 1)
costs = parameters['costs']
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(data["learning_rate"]))
# f = plt.gcf()  #获取当前图像
# f.savefig(r'./costs.png')
plt.show()