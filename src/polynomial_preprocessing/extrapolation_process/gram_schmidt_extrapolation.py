from polynomial_preprocessing.preprocessing import preprocesamiento_datos_a_grillar
from astropy.io import fits
from numba import jit, prange
from astropy.coordinates import Angle
import cupy as cp
import cupynumeric as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import astropy.units as unit
import dask.array as da
from dask import delayed, compute
from dask.distributed import Client
from numba import jit, prange, complex128, float64, int32
from dask.distributed import Client, LocalCluster
from polynomial_preprocessing.image_reconstruction import conjugate_gradient

class GramSchmidtExtrapolation:
	def __init__(self, w, P, P_target, V, D, D_target, residual, final_data, err, s, sigma2, chunk_data, max_rep = 2):
		self.w = w
		self.P = P
		self.P_target = P_target
		self.V = V
		self.D = D
		self.D_target = D_target
		self.residual = residual
		self.final_data = final_data
		self.err = err
		self.s = s
		self.sigma2 = sigma2
		self.chunk_data = chunk_data
		self.max_rep = max_rep

	
	# weight,x,y: dim n
	# weight: real
	# x,y: complex
	# return: number
	# =sum_k{w_k*x_k*y_k}
	@staticmethod
	def dot(weights,x,y):
		return(np.sum((x*weights*np.conjugate(y))))

	@staticmethod
	def safe_divide(a, b):
		return np.where(b != 0, a / b, 0.0)


	# weight,x: dim n
	# weight: real
	# x: complex
	# return number
	# =sum_k{w_k*|x_k|^2}}
	@staticmethod
	def norm(weights,x):
		return(np.sqrt(np.sum(weights*np.absolute(x)**2)))


	@staticmethod
	def dot2x2(weights, matrix, pol, chunk_data):
		weights = np.array(weights)
		pol = np.array(pol)
		N1, N2, n = matrix.shape
		sub_size = int(N1 / chunk_data) + 1
		final_dot = np.zeros(shape=(N1, N2, 1), dtype=np.complex128)

		for chunk1 in range(0, sub_size):
			for chunk2 in range(0, sub_size):
				if chunk1 + chunk2 < sub_size:
					sub_m = np.array(matrix[chunk1*chunk_data:(chunk1+1)*chunk_data,
											chunk2*chunk_data:(chunk2+1)*chunk_data, :])
					N3, N4, n2 = sub_m.shape
					w = np.ones(shape=(N3, N4, n2), dtype=np.float64) * weights
					subsum = sub_m * w * np.conjugate(pol)
					subsum = np.sum(subsum, axis=2)
					subsum = np.reshape(subsum, (N3, N4, 1))
					final_dot[chunk1*chunk_data:(chunk1+1)*chunk_data,
							chunk2*chunk_data:(chunk2+1)*chunk_data, :] = subsum

		return final_dot

	@staticmethod
	def norm2x2(weights, matrix, chunk_data):
		weights = np.array(weights)
		N1, N2, n = matrix.shape
		sub_size = int(N1 / chunk_data) + 1
		final_norm = np.zeros(shape=(N1, N2, 1), dtype=np.complex128)

		for chunk1 in range(0, sub_size):
			for chunk2 in range(0, sub_size):
				if chunk1 + chunk2 < sub_size:
					sub_m = np.array(matrix[chunk1*chunk_data:(chunk1+1)*chunk_data,
											chunk2*chunk_data:(chunk2+1)*chunk_data, :])
					N3, N4, n2 = sub_m.shape
					w = np.ones(shape=(N3, N4, n2), dtype=np.float64) * weights
					subsum = w * np.absolute(sub_m)**2
					subsum = np.sum(subsum, axis=2)
					subsum = np.sqrt(subsum)
					subsum = np.reshape(subsum, (N3, N4, 1))
					final_norm[chunk1*chunk_data:(chunk1+1)*chunk_data,
							chunk2*chunk_data:(chunk2+1)*chunk_data, :] = subsum

		return final_norm

	def gram_schmidt_and_estimation_gpu(self):
			"""
			Realiza el proceso de ortogonalización de Gram-Schmidt y estimación usando GPU.

			Parámetros:
			- w: CuPy array 1D de pesos complejos.
			- P: CuPy array 3D de polinomios complejos.
			- P_target: CuPy array 3D de polinomios extrapolados.
			- V: CuPy array 3D de enteros, matriz de validación.
			- D: CuPy array 1D complejo, polinomio de referencia actual.
			- D_target: CuPy array 1D complejo, polinomio extrapolado de referencia.
			- residual: CuPy array 1D complejo, datos residuales.
			- final_data: CuPy array 1D complejo, resultado final.
			- err: CuPy array 1D flotante, errores estimados.
			- s: tamaño de la matriz de polinomios (entero).
			- sigma2: criterio de selección sigma al cuadrado.
			- max_rep: número de repeticiones para la ortogonalización de Gram-Schmidt.
			- chunk_data: tamaño de los bloques de datos.

			Retorna:
			- final_data, residual, err, P_target, P: Arrays finales con los resultados.
			"""
			# Asegurarse de que todas las variables estén en CuPy
			w = np.asarray(self.w)
			P = np.asarray(self.P)
			P_target = np.asarray(self.P_target)
			V = np.asarray(self.V)
			D = np.asarray(self.D)
			D_target = np.asarray(self.D_target)
			residual = np.asarray(self.residual)
			final_data = np.asarray(self.final_data)
			err = np.asarray(self.err)

			# GS procedure on polynomials + 1 repetion, iteration goes on contra diagonal for same total degree k
			for k in range(0, self.s): # total degree, correspond to last column
				#large3 = np.concatenate((large3,np.array([k])),axis=0)
				for j in range(0, k + 1): # position in contra diagonal (row degree)
					print(k-j, j)
					# total degree == (k-j)+j == k


					# after several experiments one repeat is sufficient
					for repeat in range(0, self.max_rep):
						
						if repeat > 0: # for first pass
							# normalizing P, P_Target
							no=self.norm(w,P[k-j,j,:])                     
							P[k-j, j, :] = self.safe_divide(P[k-j, j, :], no)
							P_target[k-j, j, :] = self.safe_divide(P_target[k-j, j, :], no)
					

						if k==0 and j==0: # for first pass
							# normalizing P, P_Target
							no=self.norm(w,P[k-j,j,:]) 
							P[k-j, j, :] = self.safe_divide(P[k-j, j, :], no)
							P_target[k-j, j, :] = self.safe_divide(P_target[k-j, j, :], no)
					
							
							D = np.array(P[k-j,j,:]) # store first polynomial
							D_target = np.array(P_target[k-j,j,:])
							V[k-j,j,:] = 0 # this pol was processed

						else: # for the remaining
							if repeat == 0:
								#print(k-j+l,j)
								if j==1 and k>0:
									no_data = self.norm2x2(w,P,self.chunk_data)
									no_data[V == 0] = 1
									P = self.safe_divide(P, no_data)
									P_target = self.safe_divide(P_target, no_data)

								# GS main operation
								dot_data = self.dot2x2(w,P*V,D,self.chunk_data) # dim ((n),(N1,N2,n),(n))  --> return (N1,N2,1)
								P = P - dot_data*D # dim (N1,N2,n) - (N1,N2,1)*(n)
								P_target = P_target - dot_data*D_target # dim (N1,N2,M) - (N1,N2,1)*(M)
								
								#normalization
								no=self.norm(w,P[k-j,j,:])
								P[k-j, j, :] = self.safe_divide(P[k-j, j, :], no)
								P_target[k-j, j, :] = self.safe_divide(P_target[k-j, j, :], no)
					

								if (j==0):
									# normalization
									no=self.norm(w,P[k-j,j,:])
									P[k-j, j, :] = self.safe_divide(P[k-j, j, :], no)
									P_target[k-j, j, :] = self.safe_divide(P_target[k-j, j, :], no)
					
								
							if repeat > 0:
								# Could be the same than step "GS main operation"
								for y in range(0,k+1):
									for x in range(0,y+1):
										if (y-x != k-j) or (j!=x):
											dot_data=self.dot(w,P[k-j,j,:],P[y-x,x,:])
											P[k-j,j,:] = P[k-j,j,:] - dot_data*P[y-x,x,:]
											P_target[k-j,j,:] = P_target[k-j,j,:] - dot_data*P_target[y-x,x,:]
										else:
											break 
									
									#normalization
									no=self.norm(w,P[k-j,j,:])
									P[k-j, j, :] = self.safe_divide(P[k-j, j, :], no)
									P_target[k-j, j, :] = self.safe_divide(P_target[k-j, j, :], no)
							
						P[np.isnan(P)] = 0.0
						P[np.isinf(P)] = 0.0
						P_target[np.isnan(P_target)] = 0.0
						P_target[np.isinf(P_target)] = 0.0
									
					# updating validation matrix and current polynomial
					V[k-j,j,:] = 0
					D = np.array(P[k-j,j,:])
					D_target = np.array(P_target[k-j,j,:])

					# calculating extrapolation using current polynomials
					M = self.dot(w,residual.flatten(),P[k-j,j,:])
					final_data = final_data + M*P_target[k-j,j,:]
					residual = residual - M*P[k-j,j,:]
				
					# error estimation
					err = err + np.absolute(P_target[k-j,j,:])**2
		
			final_data[err>self.sigma2]=0
			#plt.figure()
			#plt.plot(final_data.flatten(),color='b')
			#plt.figure()
			#plt.plot(data.flatten(),color='k')
			return final_data,err,residual, P_target, P