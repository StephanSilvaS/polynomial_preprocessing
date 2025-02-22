import numpy as np
import math
import time
import optuna
import torch
import piq
from polynomial_preprocessing import preprocesamiento_datos_a_grillar, procesamiento_datos_grillados


class OptimizacionParametrosGrillados:
	def __init__(self, fits_path, ms_path, poly_limits, division_limits, dx, image_size):
		self.fits_path = fits_path  # Ruta de archivo FITS
		self.ms_path = ms_path # Ruta de archivo MS
		self.poly_limits = poly_limits # [Lim. Inferior, Lim. Superior] -> Lista (Ej: [5, 20])
		self.division_limits = division_limits # [Lim. Inferior, Lim. Superior] -> Lista (Ej: [1e-3, 1e0])
		self.dx = dx # Tamaño del Pixel
		self.image_size = image_size # Cantidad de pixeles para la imagen

		if self.dx is None:
			pixel_size = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						 ms_path=self.ms_path,
																						image_size=self.image_size)
			_, _, _, _, pixels_size = pixel_size.fits_header_info()
			print("Pixel size of FITS: ", pixels_size)
			self.dx = pixels_size

		if self.image_size is None:
			fits_header = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						 ms_path=self.ms_path,
																						 image_size=self.image_size)

			_, fits_dimensions, _, _, _ = fits_header.fits_header_info()
			print("Image size of FITS: ", fits_dimensions[1])
			self.image_size = fits_dimensions[1]

	@staticmethod
	def create_mask(grid_shape, radius):
		"""
		Crea un arreglo de máscara basado en un filtro circular.

		Parameters:
		- grid_shape: tuple, las dimensiones de la grilla (rows, cols).
		- radius: float, el radio del círculo.

		Returns:
		- mask: numpy.ndarray, una matriz booleana donde True indica fuera del círculo y False dentro.
		"""
		# Crear coordenadas de la grilla
		rows, cols = grid_shape
		y, x = np.ogrid[:rows, :cols]

		# Calcular el centro de la grilla
		center_row, center_col = rows // 2, cols // 2

		# Calcular la distancia de cada punto al centro
		distance_from_center = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

		# Crear la máscara: True para fuera del círculo, False dentro
		mask = distance_from_center > radius
		return mask

	def mse(self, img_final, dim_grilla, radio):
		bool_arreglo = self.create_mask(dim_grilla, radio)
		# print(bool_arreglo)
		B = img_final * bool_arreglo
		mse = np.std(B) ** 2
		print(mse)
		return mse

	# Para minimizar se debe colocar un signo menos

	def psnr(self, img_fin):
		psnr_result = 20 * math.log10(np.max(np.max(img_fin)) / self.mse(img_fin, (251, 251), 47))
		return psnr_result  # comentary mse need to be taken outside the object

	@staticmethod
	def compute_brisque(image):
	
		"""
		Calcula el score BRISQUE para una imagen dada.

		Parameters:
		- image: numpy.ndarray, la imagen a evaluar.

		Returns:
		- brisque_score: float, el score BRISQUE de la imagen.
		"""
		# Convertir la imagen a un tensor de PyTorch
		image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

		# Calcular el score BRISQUE
		brisque_score = piq.brisque(image_tensor, data_range=255., reduction='none')

		return brisque_score.item()
	
	def grid_data(self):
		gridded_visibilities, gridded_weights, dx, grid_u, grid_v = (preprocesamiento_datos_a_grillar.
																		  PreprocesamientoDatosAGrillar(self.fits_path,
																										self.ms_path,
																										self.image_size).
																		  process_ms_file())
		return gridded_visibilities, gridded_weights, dx, grid_u, grid_v

	def optimize_parameters(self, trial):

		# Cargamos los archivos de entrada
		header, fits_dimensions, fits_data, du, dx = (preprocesamiento_datos_a_grillar.
																	PreprocesamientoDatosAGrillar(self.fits_path,
																								  self.ms_path,
																								  self.image_size).
																	fits_header_info())

		gridded_visibilities, gridded_weights, dx, grid_u, grid_v = self.grid_data()


		################# Parametros iniciales #############
		M = 1  # Multiplicador de Pixeles
		N1 = self.image_size  # Numero de pixeles
		N1 = N1 * M  # Numero de pixeles,  multiplicador #Version MS
		S = trial.suggest_int("S", self.poly_limits[0], self.poly_limits[1])  # Rango del número de polinomios
		sub_S = int(S)
		ini = 1  # Tamano inicial
		division = trial.suggest_float("division", self.division_limits[0], self.division_limits[1])
		dx = self.dx

		########################################## Cargar archivo de entrada Version MS
		# Eliminamos la dimension extra
		u_ind, v_ind = np.nonzero(gridded_visibilities[0])
		gridded_visibilities_2d = gridded_visibilities[0].flatten()  # (1,251,251)->(251,251)
		gridded_weights_2d = gridded_weights[0].flatten()  # (1,251,251)->(251,251)

		# Filtramos por los valores no nulos
		nonzero_indices = np.nonzero(gridded_weights_2d)
		gv_sparse = gridded_visibilities_2d[nonzero_indices]
		gw_sparse = gridded_weights_2d[nonzero_indices]

		# Normalizacion de los datos

		gv_sparse = (gv_sparse / np.sqrt(np.sum(gv_sparse ** 2)))
		gw_sparse = (gw_sparse / np.sqrt(np.sum(gw_sparse ** 2)))

		u_data = grid_u[u_ind]
		v_data = grid_v[v_ind]

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

		max_memory = 120000000
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
		#print(chunk_data)

		visibilities_model = np.zeros((N1, N1), dtype=np.complex128)

		print("New S:", S)
		print("Division:", division)

		visibilities_aux = np.zeros(N1 * N1, dtype=np.complex128)
		weights_aux = np.zeros(N1 * N1, dtype=float)

		start_time = time.time()

		# print(z_target.dtype)
		# print(z_sparse.dtype)
		# print(gw_sparse.dtype)
		# print(gv_sparse.dtype)
		# print(type(chunk_data))

		# Obtencion de los datos de la salida con G-S

		data_processing = procesamiento_datos_grillados.ProcesamientoDatosGrillados(self.fits_path, self.ms_path, S, division, self.dx, self.image_size)

		try:
			visibilities_mini, err, residual, P_target, P = (data_processing.recurrence2d
															 (z_target.flatten(),
															  z_sparse.flatten(),
															  gw_sparse.flatten(),
															  gv_sparse.flatten(),
															  np.size(z_target.flatten()),
															  S,
															  division,
															  chunk_data)
															 )

			visibilities_mini = np.reshape(visibilities_mini, (N1, N1))

			visibilities_model = np.array(visibilities_mini)

			weights_model = np.zeros((N1, N1), dtype=float)

			sigma_weights = np.divide(1.0, gw_sparse, where=gw_sparse != 0,
									  out=np.zeros_like(gw_sparse))  # 1.0/gw_sparse
			sigma = np.max(sigma_weights) / division
			weights_mini = np.array(1 / err)
			weights_mini[np.isnan(weights_mini)] = 0.0
			weights_mini[np.isinf(weights_mini)] = 0.0

			weights_mini = np.reshape(weights_mini, (N1, N1))

			weights_model = np.array(weights_mini)

			print("El tiempo de ejecución fue de: ", time.time() - start_time)

			####################################### GENERACION DE GRAFICOS DE SALIDA #####################################

			image_model = (np.fft.fftshift
						   (np.fft.ifft2
							(np.fft.ifftshift
							 (visibilities_model * weights_model / np.sum(weights_model.flatten())))) * N1 ** 2)
			image_model = np.array(image_model.real)

			# Procesamiento adicional para calcular métrica de evaluación (PSNR, MSE, etc.)

			# Normalizar imagen para las métricas
			synthesized_image = image_model - image_model.min()
			synthesized_image = (synthesized_image / synthesized_image.max()) * 255
			synthesized_image = synthesized_image.astype(np.uint8)

			# Calcular métricas
			brisque_score = self.compute_brisque(synthesized_image)

			# Minimizar ambas métricas (menores valores indican mejor calidad)
			return brisque_score
		
		except Exception as e:
			print(f"Error en el cálculo: {e}")
			return float("inf")
		
		"""
			psnr_result = self.psnr(np.real(image_model))
			return -psnr_result  # Negativo porque Optuna minimiza la métrica
		except Exception as e:
			print(f"Error en el cálculo: {e}")
			return float("inf")  # Penalizar valores inválidos
		"""
		
	def initialize_optimization(self, num_trials):
		# Configuración del estudio de Optuna
		study = optuna.create_study(direction="minimize")
		study.optimize(self.optimize_parameters, n_trials=num_trials)

		# Resultados
		print("Mejores parámetros:", study.best_params)
		print("Mejor valor (BRISQUE):", study.best_value)