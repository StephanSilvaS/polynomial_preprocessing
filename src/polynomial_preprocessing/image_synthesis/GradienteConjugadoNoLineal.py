from astropy.io import fits
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation
from math import e, pi
import math
from array import *
import time
from scipy import optimize
from scipy.signal import convolve2d as conv

class GradienteConjugadoNoLineal:
	def __init__(self, visibilities, weights, n_iterations):
		self.visibilities = visibilities
		self.weights = weights
		self.n_iterations = n_iterations

	@staticmethod
	def gauss(ini, dim):
		array_x = np.linspace(-ini, ini, dim)
		array_x = np.reshape(array_x, (dim, 1))
		array_y = np.reshape(array_x, (1, dim))
		img = np.exp(-pi * (array_x ** 2 + array_y ** 2)) ** 2
		return (img)

	@staticmethod
	def norm(weights, x):
		return np.absolute(np.sqrt(np.sum(weights * np.absolute(x) ** 2)))

	@staticmethod
	def f_alpha(x: float, Vo, Im, Wo, s):

		Vm2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im + np.real(x) * s)))

		return np.sum(Wo * np.absolute(Vo - Vm2) ** 2)

	def conjugate_gradient(self):

		Im = np.zeros_like(self.visibilities, dtype=float)

		Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im)))

		grad = -1 * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.weights * (self.visibilities - Vm))))

		# print("Max grad:",np.max(grad))
		# title="grad"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(grad))
		# plt.colorbar(im)
		s = -grad

		grad_old = np.array(grad)

		for ite in range(0, self.n_iterations):
			diff = -grad
			diff_old = -grad_old

			beta = -np.conjugate(diff) * (diff - diff_old) / np.sum((diff_old * diff_old))

			beta[np.isinf(beta) == True] = 0
			beta[np.isnan(beta) == True] = 0

			if ite == 0:
				s = diff
			else:
				s = diff + beta * s

			a = optimize.brent(self.f_alpha, args=(self.visibilities, Im, self.weights, s))

			Im = Im + a * s

			Im.imag[Im.real < 0] = np.trunc(Im.imag)[Im.real < 0]

			Im.real[Im.real < 0] = np.trunc(Im.real)[Im.real < 0]

			grad_old = np.array(grad)

			Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im)))

			grad = -1 * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.weights * (self.visibilities - Vm))))

			grad[np.isinf(grad) == True] = 0
			grad[np.isnan(grad) == True] = 0

		return Im

