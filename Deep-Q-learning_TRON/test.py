import numpy as np


a = np.array([[1, 2], [3, 4]])
b = np.array([[7, 8], [9, 10]])
c = np.array([[7, 8], [9, 10]])
a=a.reshape(1,a.shape[0],a.shape[1])
b=b.reshape(1,b.shape[0],b.shape[1])
c=c.reshape(1,c.shape[0],c.shape[1])

print(np.concatenate((a, b),axis=0))
print(np.concatenate((a, b,c),axis=0).shape)