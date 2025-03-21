import numpy as np
import cupy as cp
import math
import time
import optuna
import torch
import piq
from polynomial_preprocessing.extrapolation_process import procesamiento_datos_continuos
from polynomial_preprocessing.preprocessing import preprocesamiento_datos_continuos
from polynomial_preprocessing.image_reconstruction import conjugate_gradient
from optuna.visualization import plot_optimization_history
from astropy.coordinates import Angle
import astropy.units as unit
from plotly.io import show

class OptimizacionParametrosContinuos:
	def __init__(self, fits_path, ms_path, poly_limits, division_limits, pixel_size = None, image_size = None, n_iter_gc = 100, plots = False):
		self.fits_path = fits_path  # Ruta de archivo FITS
		self.ms_path = ms_path  # Ruta de archivo MS
		self.poly_limits = poly_limits  # [Lim. Inferior, Lim. Superior] -> Lista (Ej: [5, 20])
		self.division_limits = division_limits  # [Lim. Inferior, Lim. Superior] -> Lista (Ej: [1e-3, 1e0])
		self.pixel_size = pixel_size  # Tamaño del Pixel
		self.image_size = image_size  # Cantidad de pixeles para la imagen
		self.n_iter_gc = n_iter_gc # Numero de iteraciones de GC
		self.plots = plots # Flag para definir si se muestran graficos por pantalla

		if self.pixel_size is None:
			pixel_size = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																						 ms_path=self.ms_path)
			_, _, _, _, pixels_size = pixel_size.fits_header_info()
			
			# Se requiere transformar de grados a radianes el tam. de pixel.
			angulo = Angle(pixels_size, unit='deg')

			pixels_size_rad = angulo.radian * unit.rad

			print("Pixel size of FITS: ", pixels_size_rad)

			self.pixel_size = pixels_size_rad

		if self.image_size is None:
			fits_header = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																						 ms_path=self.ms_path)

			_, fits_dimensions, _, _, _ = fits_header.fits_header_info()
			print("Image size of FITS: ", fits_dimensions[1])
			self.image_size = fits_dimensions[1]

	@staticmethod
	def generate_filename(prefix, poly_limits, division_limits, pixel_size, num_pixels, object_name, extension):
		base_title = f"num_polynomial_{poly_limits[0]}_{poly_limits[1]}_division_sigma_{division_limits[0]}_{division_limits[1]}_pixel_size_{pixel_size}_image_size_{num_pixels}_{num_pixels}_{object_name}"
		return f"{prefix}{base_title}.{extension}"
	
	@staticmethod
	def comp_imagenes_model(imagen_verdad, imagen_algoritmo):
		imagen_verdad/=np.max(imagen_verdad)

		imagen_algoritmo/=np.max(imagen_algoritmo)

		imagen_residuo = imagen_verdad - imagen_algoritmo

		desviacion = np.std(imagen_residuo)
		
		return desviacion

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
		print(rows)
		print(cols)
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
		psnr_result = 20 * math.log10(np.max(np.max(img_fin)) / self.mse(img_fin, (self.pixel_size, self.pixel_size), 42))
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


	def optimize_parameters(self, trial):

		start_time = time.time()

		interferometric_data = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(self.fits_path, self.ms_path)

		# Cargamos los archivos de entrada
		header, fits_dimensions, fits_data, du, pixel_size = interferometric_data.fits_header_info()
		# print(header,"\n\n",fits_dimensions)}

		uvw_coords, visibilities, weights, dx = interferometric_data.process_ms_file()

		################# Parametros
		M = 1  # Multiplicador de Pixeles
		pixel_num  =  self.image_size  # Numero de pixeles

		num_polynomial = trial.suggest_int("num_polynomial", self.poly_limits[0], self.poly_limits[1])  # Rango del número de polinomios
		sub_S = int(num_polynomial)
		ini = 1  # Tamano inicial
		division = trial.suggest_float("division", self.division_limits[0], self.division_limits[1])
		pixel_size = self.pixel_size
		# pixel_size = pixel_size * 10

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

		du = 1 / (pixel_num  * pixel_size)

		umax = pixel_num  * du / 2

		u_sparse = np.array(u_data) / umax
		v_sparse = np.array(v_data) / umax

		u_target = np.reshape(np.linspace(-ini, ini, pixel_num ), (1, pixel_num )) * np.ones(shape=(pixel_num , 1))
		v_target = np.reshape(np.linspace(-ini, ini, pixel_num ), (pixel_num , 1)) * np.ones(shape=(1, pixel_num ))

		z_target = u_target + 1j * v_target
		z_sparse = u_sparse + 1j * v_sparse

		b = 1

		z_exp = np.exp(-z_target * np.conjugate(z_target) / (2 * b * b))

		max_memory = 1200000000
		max_data = float(int(max_memory / (num_polynomial * num_polynomial)))

		divide_data = int(np.size(gv_sparse[np.absolute(gv_sparse) != 0].flatten()) / max_data) + 1
		divide_target = int(pixel_num  * pixel_num  / max_data) + 1

		if divide_target > divide_data:
			divide_data = int(divide_target)

		if divide_data > int(divide_data):
			divide_data = int(divide_data) + 1

		chunk_data = int(((num_polynomial * num_polynomial) / divide_data) ** (1 / 2)) + 1
		if chunk_data == 0:
			chunk_data = 1

		# chunk_data = 1

		visibilities_model = np.zeros((pixel_num , pixel_num ), dtype=np.complex128)

		print("Max. polynomial degree:", num_polynomial)
		print("Division:", division)

		visibilities_aux = np.zeros(pixel_num  * pixel_num , dtype=np.complex128)
		weights_aux = np.zeros(pixel_num  * pixel_num , dtype=float)

		data_processing = procesamiento_datos_continuos.ProcesamientoDatosContinuos(self.fits_path, self.ms_path, num_polynomial, division, self.pixel_size, self.image_size, verbose = False)

		

		# Llamada a la función recurrence2d
		try:
			visibilities_mini, err, residual, P_target, P = (data_processing.recurrence2d
															 (z_target.flatten(),
															  z_sparse.flatten(),
															  gw_sparse.flatten(),
															  gv_sparse.flatten(),
															  np.size(z_target.flatten()),
															  num_polynomial,
															  division,
															  chunk_data)
															 )

			visibilities_mini = np.reshape(visibilities_mini, (pixel_num , pixel_num ))

			visibilities_model = np.array(visibilities_mini)

			weights_model = np.zeros((pixel_num , pixel_num ), dtype=float)

			sigma_weights = np.divide(1.0, gw_sparse, where=gw_sparse != 0,
									  out=np.zeros_like(gw_sparse))  # 1.0/gw_sparse
			sigma = np.max(sigma_weights) / division
			weights_mini = np.array(1 / err)
			weights_mini[np.isnan(weights_mini)] = 0.0
			weights_mini[np.isinf(weights_mini)] = 0.0

			weights_mini = np.reshape(weights_mini, (pixel_num , pixel_num ))

			weights_model = np.array(weights_mini)

			

			image_model = (np.fft.fftshift
						   (np.fft.ifft2
							(np.fft.ifftshift
							 (visibilities_model * weights_model / np.sum(weights_model.flatten())))) * pixel_num  ** 2)
			image_model = np.array(image_model.real)

			reconstructed_image = conjugate_gradient.ConjugateGradient(visibilities_model, weights_model, self.n_iter_gc)

			reconstructed_image_cg = reconstructed_image.CG()

			visibility_model = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(reconstructed_image_cg)))

			gc_image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model)))
			# Procesamiento adicional para calcular métrica de evaluación (PSNR, MSE, etc.)

			interferometric_data = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																						  ms_path=self.ms_path
																						  )
			_, _, data, _, _ = interferometric_data.fits_header_info()
			

			# Calcular métricas
			mse = self.comp_imagenes_model(data, np.real(gc_image_model))

			print("El tiempo de ejecución fue de: ", time.time() - start_time)

			cp.get_default_memory_pool().free_all_blocks()

			# Minimizar ambas métricas (menores valores indican mejor calidad)
			return mse
		
		except Exception as e:
			print(f"Error en el calculo: {e}")
			return float("inf")
		
		"""
			psnr_result = self.psnr(np.real(image_model))
			return -psnr_result  # Negativo porque Optuna minimiza la métrica
		except Exception as e:
			print(f"Error en el cálculo: {e}")
			return float("inf")  # Penalizar valores inválidos
		"""
			

	def initialize_optimization(self, num_trials):

		start_time = time.time()

		# Configuración del estudio de Optuna
		study = optuna.create_study(direction="minimize")
		study.optimize(self.optimize_parameters, n_trials=num_trials)

		# Resultados
		
		print("Mejores parametros:", study.best_params)
		print("Mejor valor (MSE):", study.best_value)


		#show(convergencia)

		interferometric_data = preprocesamiento_datos_continuos.PreprocesamientoDatosContinuos(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, _, _, _ = interferometric_data.fits_header_info()

		TITLE_1_OPTUNA = "continuum_optimimum_parameters_"
		TITLE_1_PLOT = "continuum_convergence_plot_"

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		# Generar nombres de archivos
		TITLE_OPTUNA_RESULT = self.generate_filename(TITLE_1_OPTUNA,
													self.poly_limits, 
													self.division_limits, 
													self.pixel_size,
													self.image_size, 
													object_name, 
													"txt")
		
		TITLE_PLOT_RESULT = self.generate_filename(TITLE_1_PLOT,
													self.poly_limits, 
													self.division_limits, 
													self.pixel_size,
													self.image_size, 
													object_name, 
													"png")
		
		convergencia = plot_optimization_history(study)

		convergencia.update_layout(
			title="Optimización de parámetros para extrapolación de imagen",
			xaxis_title="Intento",
			yaxis_title="MSE",
		)

		if self.plots == True:
			show(convergencia)

		# Cambiar ruta de guardado de graficos
		convergencia.write_image(f"/disk2/stephan/batch_pruebas/batch_optim_param/img_graf_convergencia/{TITLE_PLOT_RESULT}")
		
		tiempo_total_opti = time.time() - start_time

		print(f"El tiempo de ejecución de optimizacion fue de: {tiempo_total_opti:.2f} segundos ")

		# Guardar el tiempo de ejecución en un archivo de texto
		with open(TITLE_OPTUNA_RESULT , "w") as file:
			file.write(f"Mejores parametros: {study.best_params}\n\n Mejor valor (MSE): {study.best_value}\n\n Tiempo total de ejecucion: {tiempo_total_opti:.2f}")

		

	