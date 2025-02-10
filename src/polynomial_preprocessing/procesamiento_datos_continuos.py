from preprocessing_orthogonal_polynomials.src.polynomial_preprocessing import preprocesamiento_datos_continuos
import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt

class ProcesamientoDatosContinuos:
	def __init__(self, fits_path, ms_path, num_polynomial, division_sigma, dx = None, image_size = None):
		self.fits_path = fits_path
		self.ms_path = ms_path
		self.num_polynomial = num_polynomial
		self.division_sigma = division_sigma
		self.dx = dx
		self.image_size = image_size

		if self.dx is None:
			pixel_size = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																						 ms_path=self.ms_path)
			_, _, _, _, pixels_size = pixel_size.fits_header_info()
			self.dx = pixels_size

		if self.image_size is None:
			fits_header = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																						  ms_path=self.ms_path)

			_, fits_dimensions, _, _, _ = fits_header.fits_header_info()

			self.image_size = fits_dimensions[1]


	def data_processing(self):
		interferometric_data = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		header, fits_dimensions, fits_data, du, dx = interferometric_data.fits_header_info()

		uvw_coords, visibilities, weights = interferometric_data.process_ms_file()

		M = 1  # Multiplicador de Pixeles
		N1 = self.image_size  # Numero de pixeles
		# N1 = 251
		S = self.num_polynomial  # Numero de polinomios
		sub_S = int(S)
		ini = 1  # Tamano inicial
		division = self.division_sigma
		dx = self.dx

		u_coords = np.array(uvw_coords[:, 0])  # Primera columna
		v_coords = np.array(uvw_coords[:, 1])  # Segunda columna
		w_coords = np.array(uvw_coords[:, 2])  # Tercera columna

		########################################## Cargar archivo de entrada Version MS
		# Eliminamos la dimension extra
		# u_ind, v_ind = np.nonzero(visibilities[0])
		gridded_visibilities_2d = visibilities[:, 0, 0]  # (1,251,251)->(251,251)
		gridded_weights_2d = weights[:, 0]  # (1,251,251)->(251,251)

		# Filtramos por los valores no nulos
		# nonzero_indices = np.nonzero(gridded_weights_2d)
		gv_sparse = gridded_visibilities_2d
		gw_sparse = gridded_weights_2d

		# Normalizacion de los datos

		gv_sparse = (gv_sparse / np.sqrt(np.sum(gv_sparse ** 2)))
		gw_sparse = (gw_sparse / np.sqrt(np.sum(gw_sparse ** 2)))

		u_data = u_coords
		v_data = v_coords

		du = 1 / (N1 * dx)

		umax = N1 * du / 2

		u_sparse = np.array(u_data) / umax
		v_sparse = np.array(v_data) / umax

		u_target = np.reshape(np.linspace(-ini, ini, N1), (1, N1)) * np.ones(shape=(N1, 1))
		v_target = np.reshape(np.linspace(-ini, ini, N1), (N1, 1)) * np.ones(shape=(1, N1))

		z_target = u_target + 1j * v_target
		z_sparse = u_sparse + 1j * v_sparse

		b = 1

		z_exp = np.exp(-z_target * np.conjugate(z_target) / (2 * b * b))

		max_memory = 1200000000
		max_data = float(int(max_memory / (S * S)))

		divide_data = int(np.size(gv_sparse[np.absolute(gv_sparse) != 0].flatten()) / max_data) + 1
		divide_target = int(N1 * N1 / max_data) + 1

		if divide_target > divide_data:
			divide_data = int(divide_target)

		if divide_data > int(divide_data):
			divide_data = int(divide_data) + 1

		chunk_data = int(((S * S) / divide_data) ** (1 / 2)) + 1
		if chunk_data == 0:
			chunk_data = 1

		# chunk_data = 1

		visibilities_model = np.zeros((N1, N1), dtype=np.complex128)

		print("New S:", S)
		print("Division:", division)

		visibilities_aux = np.zeros(N1 * N1, dtype=np.complex128)
		weights_aux = np.zeros(N1 * N1, dtype=float)

		start_time = time.time()

		visibilities_mini, err, residual, P_target, P = (self.recurrence2d(z_target.flatten(),
																		   z_sparse.flatten(),
																		   gw_sparse.flatten(),
																		   gv_sparse.flatten(),
																		   np.size(z_target.flatten()),
																		   S, division,
																		   chunk_data))

		visibilities_mini = np.reshape(visibilities_mini, (N1, N1))

		visibilities_model = np.array(visibilities_mini)

		plt.figure()
		plt.plot(visibilities_model.flatten(), color='g')


		sigma_weights = np.divide(1.0, gw_sparse, where=gw_sparse != 0, out=np.zeros_like(gw_sparse))  # 1.0/gw_sparse
		sigma = np.max(sigma_weights) / division
		weights_mini = np.array(1 / err)
		weights_mini[np.isnan(weights_mini)] = 0.0
		weights_mini[np.isinf(weights_mini)] = 0.0

		weights_mini = np.reshape(weights_mini, (N1, N1))

		weights_model = np.array(weights_mini)

		print("El tiempo de ejecución fue de: ", time.time() - start_time)

		image_model = (np.fft.fftshift
					   (np.fft.ifft2
						(np.fft.ifftshift
						 (visibilities_model * weights_model / np.sum(weights_model.flatten())))) * N1 ** 2)
		image_model = np.array(image_model.real)

		title = "Image model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(image_model)
		plt.colorbar(im)

		title = "Visibility model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.absolute(visibilities_model))
		plt.colorbar(im)

		title = "Weights model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(weights_model)
		plt.colorbar(im)

		plt.show()

		return image_model

	@staticmethod
	def dot2x2_gpu(weights, matrix, pol, chunk_data):
		"""
		Calcula el producto punto ponderado de una matriz y un polinomio en GPU.

		Parámetros:
		- weights: CuPy array de pesos complejos (1D).
		- matrix: CuPy array de polinomios complejos (3D).
		- pol: CuPy array de polinomio de referencia (1D).
		- chunk_data: Tamaño de bloque para procesamiento por partes.

		Retorna:
		- final_dot: Producto punto ponderado (3D CuPy array de forma (N1, N2, 1)).
		"""
		# Dimensiones de la matriz
		N1, N2, n = matrix.shape
		sub_size = (N1 // chunk_data) + 1
		final_dot = cp.zeros((N1, N2, 1), dtype=cp.complex128)

		for chunk1 in range(sub_size):
			for chunk2 in range(sub_size):
				if chunk1 + chunk2 < sub_size:
					# Tamaños de bloque, asegurando límites
					N3 = min(chunk_data, N1 - chunk1 * chunk_data)
					N4 = min(chunk_data, N2 - chunk2 * chunk_data)

					# Operación sobre el bloque de datos
					subsum = cp.zeros((N3, N4, 1), dtype=cp.complex128)

					# Operación de suma ponderada en la GPU para el bloque actual
					sub_matrix = matrix[chunk1 * chunk_data:(chunk1 + 1) * chunk_data,
								 chunk2 * chunk_data:(chunk2 + 1) * chunk_data, :]
					sub_weights = cp.broadcast_to(weights, sub_matrix.shape)
					sub_pol = cp.broadcast_to(cp.conjugate(pol), sub_matrix.shape)

					# Suma ponderada en la última dimensión
					subsum = cp.sum(sub_matrix * sub_weights * sub_pol, axis=2, keepdims=True)

					# Asignar el resultado al bloque en la matriz final
					final_dot[chunk1 * chunk_data:(chunk1 + 1) * chunk_data,
					chunk2 * chunk_data:(chunk2 + 1) * chunk_data, :] = subsum

		return final_dot

	@staticmethod
	def norm2x2_gpu(weights, matrix, chunk_data):
		"""
		Calcula la norma ponderada de una matriz en GPU.

		Parámetros:
		- weights: CuPy array de pesos complejos (1D).
		- matrix: CuPy array de polinomios complejos (3D).
		- chunk_data: Tamaño de bloque para procesamiento por partes.

		Retorna:
		- final_norm: Norma ponderada (3D CuPy array de forma (N1, N2, 1)).
		"""
		# Dimensiones de la matriz
		N1, N2, n = matrix.shape
		sub_size = (N1 // chunk_data) + 1
		final_norm = cp.zeros((N1, N2, 1), dtype=cp.complex128)

		for chunk1 in range(sub_size):
			for chunk2 in range(sub_size):
				if chunk1 + chunk2 < sub_size:
					# Tamaños de bloque, asegurando límites
					N3 = min(chunk_data, N1 - chunk1 * chunk_data)
					N4 = min(chunk_data, N2 - chunk2 * chunk_data)

					# Submatriz en el bloque actual
					sub_m = matrix[chunk1 * chunk_data:(chunk1 + 1) * chunk_data,
							chunk2 * chunk_data:(chunk2 + 1) * chunk_data, :]

					# Aplicar los pesos sobre la submatriz y calcular la norma ponderada
					sub_weights = cp.broadcast_to(weights, sub_m.shape)
					subsum = sub_weights * cp.abs(sub_m) ** 2
					subsum = cp.sum(subsum, axis=2)
					subsum = cp.sqrt(subsum)
					subsum = subsum.reshape((N3, N4, 1))

					# Asignar el resultado al bloque correspondiente en la matriz final
					final_norm[chunk1 * chunk_data:(chunk1 + 1) * chunk_data,
					chunk2 * chunk_data:(chunk2 + 1) * chunk_data, :] = subsum

		return final_norm

	@staticmethod
	def initialize_polynomials_cpu(z, z_target, w, s):
		P = np.zeros((s, s, len(z)), dtype=np.complex128)
		P_target = np.zeros((s, s, len(z_target)), dtype=np.complex128)

		for j in range(s):
			for k in range(s):
				P[k, j, :] = (z ** k) * np.conjugate(z) ** j
				P_target[k, j, :] = (z_target ** k) * np.conjugate(z_target) ** j

				# Normalización
				no = np.sqrt(np.sum(w * np.abs(P[k, j, :]) ** 2))
				if no != 0:
					P[k, j, :] /= no
					P_target[k, j, :] /= no

		return P, P_target


	def normalize_initial_polynomials_gpu(self, w, P, P_target, V, s, chunk_data):
		"""
		Normaliza los polinomios iniciales P y P_target usando CuPy para operaciones en GPU.

		Parámetros:
		- w: CuPy array 1D de pesos complejos.
		- P: CuPy array 3D de polinomios iniciales.
		- P_target: CuPy array 3D de polinomios objetivos.
		- V: CuPy array 3D de enteros para validación.
		- s: Dimensión de los polinomios.
		- chunk_data: Tamaño de los bloques para procesamiento.

		Retorna:
		- P: CuPy array normalizado.
		- P_target: CuPy array normalizado.
		"""
		# Asegurarse de que todos los datos estén en CuPy
		w = cp.asarray(w)
		P = cp.asarray(P)
		P_target = cp.asarray(P_target)

		# Calcular las normas para la normalización
		no_data = self.norm2x2_gpu(w, P, chunk_data)

		# Evitar divisiones por cero asignando 1 a los elementos de no_data que son cero
		no_data[no_data == 0] = 1

		# Normalizar P y P_target
		P = P / no_data
		P_target = P_target / no_data

		# Limpieza de valores NaN e Inf
		P = cp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
		P_target = cp.nan_to_num(P_target, nan=0.0, posinf=0.0, neginf=0.0)

		del w

		# Se libera la memoria utilizada por la GPU, para evitar un sobreconsumo de
		# esta.
		mempool = cp.get_default_memory_pool()
		mempool.free_all_blocks()

		return P, P_target


	def gram_schmidt_and_estimation_gpu(self, w, P, P_target, V, D, D_target, residual, final_data, err, s, sigma2, max_rep,
										chunk_data):
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
		w = cp.asarray(w)
		P = cp.asarray(P)
		P_target = cp.asarray(P_target)
		V = cp.asarray(V)
		D = cp.asarray(D)
		D_target = cp.asarray(D_target)
		residual = cp.asarray(residual)
		final_data = cp.asarray(final_data)
		err = cp.asarray(err)

		for k in range(s):  # Nivel de grado de los polinomios
			for j in range(k + 1):  # Grado de cada polinomio en la contradiagonal
				for repeat in range(max_rep):
					if repeat > 0 or (k == 0 and j == 0):
						# Normalización
						no = cp.sqrt(cp.sum(w * cp.abs(P[k - j, j, :]) ** 2))
						if no != 0:
							P[k - j, j, :] /= no
							P_target[k - j, j, :] /= no

						# Almacenar polinomios iniciales
						if k == 0 and j == 0:
							D = cp.array(P[k - j, j, :])
							D_target = cp.array(P_target[k - j, j, :])
							V[k - j, j, :] = 0

					# Evitar normalización innecesaria si el grado es superior a 1
					if j == 1 and k > 0 and repeat == 0:
						no_data = self.norm2x2_gpu(w, P, chunk_data)
						V_mask = cp.where(V == 0, 1, 0)  # Crear una máscara para V
						no_data *= V_mask  # Aplicar la máscara
						P /= cp.where(no_data != 0, no_data, 1)
						P_target /= cp.where(no_data != 0, no_data, 1)

					# Ortogonalización Gram-Schmidt
					if repeat == 0:
						dot_data = self.dot2x2_gpu(w, P * V, D, chunk_data)
						P -= dot_data * D
						P_target -= dot_data * D_target

				# Limpieza de valores NaN e Inf
				P = cp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
				P_target = cp.nan_to_num(P_target, nan=0.0, posinf=0.0, neginf=0.0)

				# Actualización de V y cálculo de extrapolación
				V[k - j, j, :] = 0
				D = cp.array(P[k - j, j, :])
				D_target = cp.array(P_target[k - j, j, :])
				M = cp.sum(w * residual.flatten() * cp.conjugate(P[k - j, j, :]))
				final_data += M * P_target[k - j, j, :]
				residual -= M * P[k - j, j, :]
				err += cp.abs(P_target[k - j, j, :]) ** 2

		del M
		del V
		del D
		del D_target
		del w

		# Se libera la memoria utilizada por la GPU, para evitar un sobreconsumo de
		# esta.
		mempool = cp.get_default_memory_pool()
		mempool.free_all_blocks()

		final_data[err > sigma2] = 0

		# Convertir las salidas de nuevo a NumPy para evitar errores fuera de esta función
		return cp.asnumpy(final_data), cp.asnumpy(residual), cp.asnumpy(err), cp.asnumpy(P_target), cp.asnumpy(P)

	def recurrence2d(self, z_target, z, weights, data, size, s, division_sigma, chunk_data):
		z = np.array(z)
		z_target = np.array(z_target)
		w = np.array(weights)
		residual = np.array(data)

		sigma_weights = np.divide(1.0, w, where=w != 0, out=np.zeros_like(w))
		sigma2 = np.max(sigma_weights) / division_sigma
		print("Sigma:", sigma2)

		final_data = np.zeros(shape=(size), dtype=np.complex128)
		# P = np.zeros(shape=(s, s, z.size), dtype=np.complex128)
		# P_target = np.zeros(shape=(s, s, size), dtype=np.complex128)
		V = np.ones(shape=(s, s, 1), dtype=int)
		D = np.zeros(z.size, dtype=np.complex128)
		D_target = np.zeros(size, dtype=np.complex128)
		err = np.zeros(shape=(size), dtype=float)

		# Inicialización de matrices polinómicas P y P_target
		P, P_target = self.initialize_polynomials_cpu(z, z_target, w, s)

		# Normalización inicial de P y P_target
		P, P_target = self.normalize_initial_polynomials_gpu(w, P, P_target, V, s, chunk_data)

		# Procedimiento Gram-Schmidt en los polinomios
		final_data, residual, err, P_target, P = self.gram_schmidt_and_estimation_gpu(w, P, P_target, V, D, D_target,
																				 residual, final_data, err, s, sigma2,
																				 max_rep=2, chunk_data=chunk_data)

		del w
		del D
		del D_target
		del z
		del z_target

		return final_data, err, residual, P_target, P


ejemplo1 = ProcesamientoDatosContinuos("/mnt/c/Users/steph/Desktop/proyecto_libreria/preprocessing_orthogonal_polynomials/src/polynomial_preprocessing/dirty_images_natural_251.fits",
    "/mnt/c/Users/steph/Desktop/proyecto_libreria/preprocessing_orthogonal_polynomials/src/polynomial_preprocessing/hd142_b9cont_self_tav.ms", 11, 0.014849768613424696, 0.0007310213536, 251)

imagen_final = ejemplo1.data_processing()