import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavfile import write

spf = wave.open('Data/helloworld.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("Numpy signal shape:", signal.shape)

plt.plot(signal)
plt.title("Hello world without echo")
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal ,delta)
print("noecho signal:", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.000001)
noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6
filt[12000] = 0.2
filt[15999] = 0.1

out = np.convolve(signal, filt)
out = out.astype(np.int16) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('out.wav', 16000, out)

plt.plot(out)
plt.show()