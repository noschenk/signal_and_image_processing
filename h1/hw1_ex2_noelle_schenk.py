################
#
# hw1 ex2
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.fftpack as scfft
import scipy.signal as sig

# ex2.1


def f(t, f0=1, f1=3, f2=5):
    """creates a 1D signal"""
    res = 2 * np.sin(2 * np.pi * f0 * t) + np.cos(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    return res

# Hz measures frequency, the number of cycles per second.
# the signal is a wave. t must have small intervals.

# sampling rate : 1000 Hz means 1000 samples per second.
# sampling from the function means only taking signal values at the given time points.
sfr_original = 1000
fs = np.arange(0, 10, 1/sfr_original)
samples = f(fs, 1, 3, 5)
# sample.shape  # gives 10000 (equivalent of sampling during 10 seconds with a frequency of 1000 samples per second)

# t_samples = scfft.fft(samples)  # the fourier transformed samples
# generate the fourier transformed samples
t_samples = np.abs(scfft.fft(samples))**2
# make a 'x axis' : from 0 to maximum frequency (sampling frequency) with as many steps as
# I have samples
t_freq = np.linspace(0, sfr_original, 10*sfr_original, endpoint=False)

# generate the two plots
plt.subplot(1, 2, 1)
plt.plot(fs, samples)
plt.title('Time domain')
plt.ylabel('signal f(t)')
plt.xlabel('time t [s]')

plt.subplot(1, 2, 2)
plt.plot(t_freq, t_samples)
# plt.plot(t_freq, np.sqrt(np.power(t_samples.imag, 2) + np.power(t_samples.real, 2)))
plt.title('Power spectrum')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')
# axes = plt.gca()
# plt.xlim(0, 9)

# # generate plot which shows real and imaginary parts of the power spectral noise
# plt.subplot(1, 3, 1)
# plt.plot(fs, t_samples.real)
# plt.title('Power spectrum real')
# plt.ylabel('power spectral density F(w)')
# plt.xlabel('frequency w')
# axes = plt.gca()
# axes.set_ylim([-0.000000000002, 0.000000000002])
# plt.subplot(1, 3, 2)
# plt.plot(fs, t_samples.imag)
# plt.title('Power spectrum imaginary')
# plt.ylabel('power spectral density F(w)')
# plt.xlabel('frequency w')
# axes = plt.gca()
# axes.set_ylim([-0.000000000002, 0.000000000002])
# plt.subplot(1, 3, 3)
# plt.plot(fs, np.sqrt(np.power(t_samples.imag, 2) + np.power(t_samples.real, 2)))
# plt.title('Power spectrum imag and real together')
# plt.ylabel('power spectral density F(w)')
# plt.xlabel('frequency w')
# plt.suptitle('Sampling frequency = 1000 Hz', fontsize=16)


# ex2.2
# downsampling frequency 50, 25, 10, 5, 2
# sfr_original
# e.g. to sample 50 samples from my 1000, I need to take
# 1000 / 50 = every 20 samples
down_freq = 2
indices = np.arange(0, down_freq * 10, 1) * (10000 // (down_freq * 10))
indices.shape[0] == down_freq * 10
# power spectrum values
t_down_sampled = scfft.fft(samples[indices])
t_down_sampled = np.sqrt(np.power(t_down_sampled.imag, 2) + np.power(t_down_sampled.real, 2))
t_freqs = np.linspace(0, down_freq, 10*down_freq, endpoint=False)
# plot both spaces
plt.subplot(1, 2, 1)
plt.plot(fs, samples)
plt.plot(fs[indices], samples[indices], 'ro')
plt.plot(fs[indices], samples[indices])
plt.title('Time domain')
plt.ylabel('signal f(t)')
plt.xlabel('time t [s]')
plt.subplot(1, 2, 2)
plt.plot(t_freqs, t_down_sampled)
plt.title('Power spectrum')
plt.ylabel('power spectral density F(w)')
plt.xlabel('frequency w')
plt.suptitle('Sampling frequency = ' + str(down_freq) + ' Hz', fontsize=16)

# Nyquist theorem : take a sampling width that is : dx <= 1/2w
# w is the maximum frequency which is f2 which is 5.
# dx <= 1/10 <-> dx > 10 Hz
# all sampling frequencies lower (and inclusive) than 10 Hz lead to aliasing.


# ex2.3
# sc.signal.butter()
# generate Butterworth low-pass filter coefficients:
# select an appropriate cutoff frequency
# work on the 10 Hz example


b, a = sig.butter(4, 100, 'low', analog=True)
plt.plot(b, a)
w, h = sig.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))

# check out the sample functions and forum!