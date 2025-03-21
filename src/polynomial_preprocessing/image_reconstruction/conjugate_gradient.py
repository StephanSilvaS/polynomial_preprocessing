import numpy as np
from array import *
from scipy import optimize

class ConjugateGradient:
	def __init__(self, Vo, Wo, n):
		self.Vo = Vo
		self.Wo = Wo
		self.n = n

	@staticmethod
	def f_alpha(x:float,Vo,Im,Wo,s):
		
		Vm2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im + np.real(x)*s)))

		return np.sum(Wo*np.absolute(Vo - Vm2)**2)

					
	def CG(self):
		
		Im = np.zeros_like(self.Vo, dtype = float)
		
		Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im)))
		
		grad = -1*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo*(self.Vo - Vm))))

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