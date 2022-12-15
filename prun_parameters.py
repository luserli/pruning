import numpy as np
import matplotlib.pyplot as plt

f1_parameters = 'datasets/parameters.npy'
f2_parameters = 'datasets/prun_parameters.npy'
f1_costs = 'datasets/prun_costs.npy'
parameters = np.load(f1_parameters, allow_pickle='TRUE').item()
prun_parameters = np.load(f2_parameters, allow_pickle='TRUE').item()
prun_costs = np.load(f1_costs)
print(parameters)
print(prun_parameters)
plt.figure()
plt.subplot(1, 1, 1)
costs = prun_costs
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate ="+str(0.01))
# f = plt.gcf()  #获取当前图像
# f.savefig(r'./costs.png')
plt.show()