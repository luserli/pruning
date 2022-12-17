#测试mask
import numpy as np
a = [[0,1,2,3,4],[5,6,7,8,9]]
print("oriarray:")
print(a)

mask = a > np.percentile(a, 10)
print(np.percentile(a,10))
a=np.multiply(a,mask)
print("masked:")
print(a)

mask = a > np.percentile(a, 10)
print(np.percentile(a,10))
a=np.multiply(a,mask)
print("masked:")
print(a)

def array0(a):
	b=[]
	b.append(0)
	for i in a:
		for j in i:
			if j != 0:
				b.append(j)
	return b

b=array0(a)
for l in range(9):
	mask = a > np.percentile(b, 20)
	a=np.multiply(a,mask)
	b=array0(a)

print("masked:")
print(a)