import numpy as np

from scipy.signal import convolve2d

A = np.array([[0,10,10,0], [20,30,30,20],[10,20,20,10],[0,5,5,0]])
w = np.array([[1,0],[0,2]])
K = w.shape
N = A.shape
print(N, K)

print(A)
print(w)

height = N[0] - K[0] + 1
width = N[1] - K[1] + 1 
s = np.zeros((height,width))


for i in range(0, height):
  for j in range(0, width):
    for ii in range(0, K[0]):
      for jj in range(0, K[1]):
        s[i,j] += A[i+ii, j+jj]*w[ii,jj]


# In order to make the "deep learning" convolution with scipy, we need to flip the filter 
# vertically and horizontal, also use mode='valid'
print(convolve2d(A, np.fliplr(np.flipud(w)), mode='valid'))

