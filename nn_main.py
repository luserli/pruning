'''
使用训练好的参数, 前向传播进行预测
'''
import time
import matplotlib.pyplot as plt
from nn_functions import *

start = time.time()

#读取数据
dates = 'datasets/carvnocar.h5'
train_set_x, train_y, test_set_x, test_y = load_files(dates)
print("训练集数量：" + str(train_set_x.shape[1]))
print("测试集数量：" + str(test_set_x.shape[1]))
print("训练集维度：" + str(train_set_x.shape))
print("测试集维度：" + str(test_set_x.shape))

#指定层数和节点数
layer_dims = [train_set_x.shape[0], 5, 4, 3, 2, 1]
layer_dims = [train_set_x.shape[0], 5, 4, 3, 2, 1]
print('神经网络层数:' + str(len(layer_dims)))

#标准化
train_x = train_set_x / 255
test_x = test_set_x / 255

learning_rate = 0.035
num_iterations = 2950

parameters, learning_rate = model(train_x, train_y, test_x, test_y, layer_dims, learning_rate, num_iterations)

end = time.time()
print('运行时间: %s 秒' % np.round(end - start))

data = {}
data['learning_rate'] = learning_rate
data['times'] = np.round(end - start)
#保存参数
np.save('datasets/parameters.npy', parameters)
np.save('datasets/data.npy', data)

#绘制梯度下降图
plt.figure()
costs = parameters["costs"]
plt.subplot(1, 1, 1)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()