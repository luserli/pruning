#make mask
import numpy as np
a = np.array([[1,-0.2,-2],[-2,3,0.1],[-0.1,0.2,3]])
print("oriarray:")
print(a)
threshold=np.percentile(np.abs(a), 10)
mask = np.abs(a) > threshold
print(threshold)
a=np.multiply(a,mask)
print("pruning:")
print(a)
#直接继续进行10%的pruning，输出不变
mask = np.abs(a) > threshold
print(threshold)
a=np.multiply(a,mask)
print("pruning:")
print(a)

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
#减去0后连续九次进行10%pruning
b=array0(a)
for l in range(9):
	mask = np.abs(a) > np.percentile(b, 10)
	a=np.multiply(a,mask)
	b=array0(a)

print("pruning:")
print(a)