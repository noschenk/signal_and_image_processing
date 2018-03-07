################
#
# notes

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

def f(t, f0, f1, f2):
    res = 2 * np.sin(2* np.pi* f0* t) + np.cos(2* np.pi* f1* t) + 0.5* np.sin(2* np.pi* f2* t)
    return res

# all in Hz
# Hz measures frequency, the number of cycles per second.
f0 = 1
f1 = 3
f2 = 5
t = np.arange(0, 11, 0.001)

# the signal is a wave. t must have small intervals.
first = f(1, 3, 5, t)
plt.plot(t, first)

# check this https://en.wikipedia.org/wiki/Spectral_density#/media/File:Voice_waveform_and_spectrum.png

# sampling rate
fs = 1000  #Hz

#scipy.fftpack.fft()
# map function https://stackoverflow.com/questions/10973766/understanding-the-map-function




# mse.reduce(range, axis)
