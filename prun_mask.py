#测试mask
import numpy as np
a = np.array([[0.98,0.12,0.88,0.02,0.97],[0.06,0.99,0.02,0.01,0.96]])
print("oriarray:")
print(a)
mask = a < np.percentile(a, 25)
a[mask] = 0
print(np.percentile(a, 25))
print("maskarray:")
print(a)