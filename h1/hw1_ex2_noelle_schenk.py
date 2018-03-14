################
#
# hw1 ex2
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.fftpack as scfft

# ex2.1


def f(t, f0, f1, f2):
    """creates a 1D signal"""
    res = 2 * np.sin(2 * np.pi * f0 * t) + np.cos(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    return res

# Hz measures frequency, the number of cycles per second.
# the signal is a wave. t must have small intervals.

# sampling rate : 1000 Hz means 1000 samples per second.
# sampling from the function means only taking signal values at the given time points.
fs = np.arange(0, 10, 0.001)
samples = f(1, 3, 5, fs)
# sample.shape  # gives 10000 (equivalent of sampling during 10 seconds with a frequency of 1000 samples per second)

t_samples = scfft.fft(samples)  # the fourier transformed samples

# generate the two plots
plt.subplot(1, 2, 1)
plt.plot(fs, samples)
plt.title('Time domain')
plt.ylabel('signal f(t)')
plt.xlabel('time t [s]')

plt.subplot(1, 2, 2)
plt.plot(fs, t_samples.imag)
plt.title('Power spectrum')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')

# generate plot which shows the power spectral noise
plt.subplot(1, 3, 1)
plt.plot(fs, t_samples.real)
plt.title('Power spectrum real')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')
axes = plt.gca()
axes.set_ylim([-0.000000000002, 0.000000000002])
plt.subplot(1, 3, 2)
plt.plot(fs, t_samples.imag)
plt.title('Power spectrum imaginary')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')
axes = plt.gca()
axes.set_ylim([-0.000000000002, 0.000000000002])
plt.subplot(1, 3, 3)
plt.plot(fs, np.sqrt(np.power(t_samples.imag, 2) + np.power(t_samples.real, 2)))
plt.title('Power spectrum imag and real together')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')
axes = plt.gca()
axes.set_ylim([-0.000000000002, 0.000000000002])
plt.suptitle('Sampling frequency = 1000 Hz', fontsize=16)


# ex2.2
# downsampling frequency 50, 25, 10, 5, 2
current_freq = 2
sampling_frequency = np.arange(0, 10, 1/current_freq)

orig = f(1, 3, 5, np.arange(0, 10, 0.001))
samp = f(1, 3, 5, sampling_frequency)
t_samp = scfft.fft(samp)
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, 10, 0.001), orig)
plt.plot(sampling_frequency, samp, 'ro')
plt.title('Time domain')
plt.ylabel('signal f(t)')
plt.xlabel('time t [s]')
plt.subplot(1, 2, 2)
plt.plot(sampling_frequency, t_samp)
plt.title('Power spectrum')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')
axes = plt.gca()
axes.set_xlim([-1,10])
plt.suptitle('Sampling frequency = ' + str(current_freq) + ' Hz', fontsize=16)

# Nyquist theorem : take a sampling width that is : dx <= 1/2w
w = 1  # tod make more here


# ex2.3
# sc.signal.butter()
