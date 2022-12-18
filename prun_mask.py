#make mask
import numpy as np
a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
print("oriarray:")
print(a)

mask = a > np.percentile(a, 20)
print(np.percentile(a,20))
a=np.multiply(a,mask)
print("pruning:")
print(a)
#直接继续进行20%的pruning，输出不变
mask = a > np.percentile(a, 20)
print(np.percentile(a,20))
a=np.multiply(a,mask)
print("pruning:")
print(a)

def array0(a):
	b=[]
	b.append(0)
	for i in a:
		for j in i:
			if j != 0:
				b.append(j)
	return b
#减去0后连续九次进行20%pruning
b=array0(a)
for l in range(9):
	mask = a > np.percentile(b, 20)
	a=np.multiply(a,mask)
	b=array0(a)

print("pruning:")
print(a)