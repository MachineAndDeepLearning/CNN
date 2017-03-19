import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load image and convert to black and white
img = mpimg.imread('../Data/lena.png')
plt.imshow(img)
plt.show()


bw = img.mean(axis=2)
plt.imshow(bw)
plt.show()

# create the gaussian filter and show it
W = np.zeros((20, 20))

for i in range(20):
	for j in range(20):
		dist = (i - 9.5)**2 + (j-9.5)**2
		W[i,j] = np.exp(-dist / 50)

plt.imshow(W, cmap='gray')
plt.show()

# convolve
out = convolve2d(bw, W)
plt.imshow(out)
plt.show()

print(out.shape)

# convolve whilst maintaing the dims

out2 = convolve2d(bw, W, mode='same')
plt.imshow(out2)
plt.show()

# try to run the convolution on the color channels
out3 = np.zeros(img.shape)
for i in range(3):
	out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')

plt.imshow(out3)
plt.show()