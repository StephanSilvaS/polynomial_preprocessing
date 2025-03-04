import numpy as np
from math import e, pi
from array import *
from scipy import optimize
from scipy.signal import convolve2d as conv

class ConjugateGradient:
	def __init__(self, Vo, Wo, n):
		self.Vo = Vo
		self.Wo = Wo
		self.n = n
		"""
		self.Im = np.zeros_like(Vo, dtype=float)
		self.Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im)))
		self.grad = -1 * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Wo * (Vo - self.Vm))))
		self.grad_old = np.array(self.grad)
		self.s = -self.grad
		"""

	@staticmethod
	def gauss(ini,dim):
		array_x = np.linspace(-ini,ini,dim)
		array_x = np.reshape(array_x,(dim,1))
		array_y = np.reshape(array_x,(1,dim))
		img = np.exp(-pi*(array_x**2 + array_y**2))**2
		return(img)

	@staticmethod
	def f_alpha(x:float,Vo,Im,Wo,s):
		
		Vm2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im + np.real(x)*s)))

		return np.sum(Wo*np.absolute(Vo - Vm2)**2)

					
	def CG(self):
		
		Im = np.zeros_like(self.Vo, dtype = float)
		
		Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im)))
		
		grad = -1*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo*(self.Vo - Vm))))

		
		#print("Max grad:",np.max(grad))
		#title="grad"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(grad))
		#plt.colorbar(im)
		s = -grad
		
		grad_old = np.array(grad)
		
		for ite in range(0, self.n):
			diff = -grad
			diff_old = -grad_old

			beta = -np.conjugate(diff)*(diff-diff_old)/np.sum((diff_old*diff_old))

			beta[np.isinf(beta) == True] = 0
			beta[np.isnan(beta) == True] = 0      

			if ite == 0:
				s = diff
			else:
				s = diff + beta*s


			a = optimize.brent(self.f_alpha,args=(self.Vo, Im, self.Wo, s))

			Im = Im + a*s
			
			Im.imag[Im.real < 0] = np.trunc(Im.imag)[Im.real < 0]
			
			Im.real[Im.real < 0] = np.trunc(Im.real)[Im.real < 0]

			grad_old = np.array(grad)
			
			Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im)))
			
			grad = -1*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo*(self.Vo - Vm))))
			
			
			grad[np.isinf(grad) == True] = 0
			grad[np.isnan(grad) == True] = 0

		return Im

	@staticmethod
	def norm(weights,x):
		return(np.absolute(np.sqrt(np.sum(weights*np.absolute(x)**2))))

	
	"""
	def gauss(self, ini, dim):
		array_x = np.linspace(-ini, ini, dim)
		array_x = np.reshape(array_x, (dim, 1))
		array_y = np.reshape(array_x, (1, dim))
		img = np.exp(-pi * (array_x**2 + array_y**2))**2
		return img

	def f_alpha(self, x: float, s):
		Vm2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im + np.real(x) * s)))
		return np.sum(self.Wo * np.absolute(self.Vo - Vm2)**2)

	def compute_gradient(self):
		for ite in range(self.n):
			diff = -self.grad
			diff_old = -self.grad_old

			if ite == 0:
				self.s = diff
			else:
				# Cálculo de beta sin el signo negativo extra y usando producto escalar real
				beta = np.sum(diff * (diff - diff_old)) / np.sum(diff_old * diff_old)
				self.s = diff + beta * self.s

			# Búsqueda del parámetro óptimo de escala a
			a = optimize.brent(self.f_alpha, args=(self.s,))
			
			# Actualizamos la imagen y forzamos su parte real
			self.Im = np.real(self.Im + a * self.s)

			self.grad_old = np.array(self.grad)
			self.Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im)))
			self.grad = -np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo * (self.Vo - self.Vm))))
			self.grad[np.isinf(self.grad)] = 0
			self.grad[np.isnan(self.grad)] = 0
			
			# (Opcional) Puedes evaluar la norma del gradiente para decidir si detener el ciclo
			# if np.linalg.norm(self.grad) < tol:
			#     break

		return self.Im


	def norm(self, weights, x):
		return np.absolute(np.sqrt(np.sum(weights * np.absolute(x)**2)))
	"""
	