import numpy as np
import cupy as cp
import math
import time
import optuna
import torch
import piq
import astropy.units as unit
import matplotlib.pyplot as plt
from polynomial_preprocessing.extrapolation_process import procesamiento_datos_grillados
from polynomial_preprocessing.preprocessing import preprocesamiento_datos_a_grillar
from polynomial_preprocessing.image_reconstruction import conjugate_gradient
from optuna.visualization import plot_optimization_history
from plotly.io import show
from astropy.coordinates import Angle
from astropy.io import fits
import gc


class OptimizacionParametrosGrillados:
	def __init__(self, fits_path, ms_path, poly_limits, division_limits, pixel_size = None, image_size = None, n_iter_gc = 100, verbose = True, plots = False, gpu_id = 0):
		self.fits_path = fits_path  # Ruta de archivo FITS
		self.ms_path = ms_path # Ruta de archivo MS
		self.poly_limits = poly_limits # [Lim. Inferior, Lim. Superior] -> Lista (Ej: [5, 20])
		self.division_limits = division_limits # [Lim. Inferior, Lim. Superior] -> Lista (Ej: [1e-3, 1e0])
		self.pixel_size = pixel_size # Tamaño del Pixel
		self.image_size = image_size # Cantidad de pixeles para la imagen
		self.n_iter_gc = n_iter_gc # Número de iteraciones de Grad. Conjugado
		self.verbose = verbose # Flag para guardar graficos e imagenes
		self.plots = plots # Flag booleano para plotear graficos por pantalla
		self.gpu_id = gpu_id # En caso de usar un cluster/servidor, se elige cual de todas las GPU se va a usar para procesar.

		if self.pixel_size is None:
			pixel_size = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						 ms_path=self.ms_path,
																						image_size=self.image_size)
			_, _, _, _, pixels_size = pixel_size.fits_header_info()
			print("Pixel size of FITS on degree: ", pixels_size)
			
			# Se requiere transformar de grados a radianes el tam. de pixel.
			angulo = Angle(pixels_size, unit='deg')

			pixels_size_rad = angulo.radian * unit.rad

			print("Pixel size of FITS on radians: ", pixels_size_rad)
			self.pixel_size = pixels_size_rad

		if self.image_size is None:
			fits_header = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						 ms_path=self.ms_path,
																						 image_size=self.image_size)

			_, fits_dimensions, _, _, _ = fits_header.fits_header_info()
			print("Image size of FITS: ", fits_dimensions[1])
			self.image_size = fits_dimensions[1]

		grid_visibilities, grid_weights, _, grid_u, grid_v = (preprocesamiento_datos_a_grillar.
																PreprocesamientoDatosAGrillar(self.fits_path,
																								self.ms_path,
																								pixel_size = self.pixel_size,																										
																								image_size = self.image_size,
																								plots = self.plots).
																		  process_ms_file())
		
		self.gridded_visibilities = grid_visibilities

		self.gridded_weights = grid_weights

		self.grid_u = grid_u

		self.grid_v = grid_v


	@staticmethod
	def generate_filename(prefix, poly_limits, division_limits, pixel_size, num_pixels, object_name, extension):
		base_title = f"num_polynomial_{poly_limits[0]}_{poly_limits[1]}_division_sigma_{division_limits[0]}_{division_limits[1]}_pixel_size_{pixel_size}_image_size_{num_pixels}_{num_pixels}_{object_name}"
		return f"{prefix}{base_title}.{extension}"
	
	# Función para generar nombres de archivos
	@staticmethod
	def generate_filename_per_trial(prefix, trial_num, num_polynomials, division, b, pixel_size, num_pixels, object_name, extension):
		base_title = f"trial_num_{trial_num}_num_polynomial_{num_polynomials}_division_sigma_{division}_b_{b}_pixel_size_{pixel_size}_image_size_{num_pixels}_{num_pixels}_{object_name}"
		return f"{prefix}{base_title}.{extension}"
	
	@staticmethod
	def comp_imagenes_model(imagen_verdad, imagen_algoritmo):
		imagen_verdad/=np.max(imagen_verdad)

		imagen_algoritmo/=np.max(imagen_algoritmo)

		imagen_residuo = imagen_verdad - imagen_algoritmo

		desviacion = np.std(np.abs(imagen_residuo))
		
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
	
	'''
	
	# img1, img2: dim(M,M)
	# img1,img2: real!!!! comentary: do not work for complex 
	# return mean quadratic diference
	@staticmethod
	def mse(img1, img2):
		N1, N2 = img1.shape
		err = np.sum((img1 - img2)**2)/(N1*N2)
		return err
	'''
	@staticmethod
	def calcular_rmse(visibilidades_observadas, visibilidades_modelo):
		"""
		Calcula el RMSE entre dos arreglos complejos de visibilidades.

		Parámetros:
		visibilidades_observadas: array complejo de visibilidades del FITS original
		visibilidades_modelo: array complejo de visibilidades extrapoladas

		Retorna:
		rmse: error cuadrático medio (float)
		"""
		diferencia = visibilidades_observadas - visibilidades_modelo
		error_cuadrado = np.abs(diferencia) ** 2
		N1, N2 = visibilidades_observadas.shape
		mse = np.sum(error_cuadrado) / (N1 * N2)
		rmse = np.sqrt(mse)
		return rmse

	@staticmethod
	def calcular_rmse_norm(visibilidades_observadas, visibilidades_modelo):
		"""
		Calcula el RMSE entre dos arreglos complejos de visibilidades, normalizados por su valor máximo absoluto.

		Parámetros:
		- visibilidades_observadas: array complejo de visibilidades del FITS original
		- visibilidades_modelo: array complejo de visibilidades extrapoladas

		Retorna:
		- rmse: error cuadrático medio (float)
		"""

		max_obs = np.max(np.abs(visibilidades_observadas))
		max_model = np.max(np.abs(visibilidades_modelo))

		if max_obs == 0 or max_model == 0:
			raise ValueError("Una de las visibilidades tiene valor máximo cero y no se puede normalizar.")

		vis_obs_norm = visibilidades_observadas / max_obs
		vis_model_norm = visibilidades_modelo / max_model

		diferencia = vis_obs_norm - vis_model_norm
		error_cuadrado = np.abs(diferencia) ** 2
		N1, N2 = visibilidades_observadas.shape
		mse = np.sum(error_cuadrado) / (N1 * N2)
		rmse = np.sqrt(mse)

		return rmse



	#def psnr(self, img_ini, img_fin):
		#return 20*math.log10(np.max(np.max(img_fin))/self.mse(img_ini, img_fin))

	# Para minimizar se debe colocar un signo menos

	def psnr(self, img_fin):
		psnr_result = 20 * math.log10(np.max(np.max(img_fin)) / self.mse(img_fin, (self.image_size, self.image_size), 47))
		return psnr_result  # comentary mse need to be taken outside the object

	
	
	@staticmethod
	def norm(weights,x):
		return(np.absolute(np.sqrt(np.sum(weights*np.absolute(x)**2))))

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
		gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v = (preprocesamiento_datos_a_grillar.
																		  PreprocesamientoDatosAGrillar(self.fits_path,
																										self.ms_path,																										
																										image_size = self.image_size,
																										pixel_size=self.pixel_size).
																		  process_ms_file())
		return gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v

	def optimize_parameters(self, trial):
		"""
		Optimización de parámetros individuales.
		Se asegura de liberar memoria correctamente y captura excepciones.
		"""

		try:
			cp.cuda.runtime.setDevice(self.gpu_id)  # Asegurar que trabajamos sobre la GPU correcta

			rmse = self._optimize_single_trial(trial)  # Usar una función auxiliar

			return rmse

		except Exception as e:
			print(f"[Error en trial {trial.number}] {e}")
			return float('inf')  # Penaliza el trial que falló

		finally:
			# Liberar memoria
			gc.collect()
			cp.get_default_memory_pool().free_all_blocks()

	def _optimize_single_trial(self, trial):
		"""
		Esta función sería tu actual "optimize_parameters" pero enfocado solo en 1 trial.
		Mueves todo el contenido grande que ahora tienes aquí adentro.
		"""

		start_time = time.time()
		
		# Cargamos los archivos de entrada
		header, _, fits_data, du, pixel_size = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(self.fits_path, 
																								   self.ms_path, 
																								   image_size = self.image_size, 
																								   pixel_size = self.pixel_size).fits_header_info()

		visibility_model_fits_first = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fits_data)))

		if self.plots == True:
				plt.figure()
				plt.plot(visibility_model_fits_first.flatten(), color='g')

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in header:
			object_name = header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")
		
		################# Parametros iniciales #############
		num_intento = str(trial.number)
		print("El numero de intento actual es el numero: ", num_intento)
		M = 1  # Multiplicador de Pixeles
		N1 = self.image_size  # Numero de pixeles
		N1 = N1 * M  # Numero de pixeles,  multiplicador #Version MS
		S = trial.suggest_int("S", self.poly_limits[0], self.poly_limits[1])  # Rango del número de polinomios
		sub_S = int(S)
		ini = 1  # Tamano inicial
		division = trial.suggest_float("division", self.division_limits[0], self.division_limits[1])
		pixel_size = self.pixel_size

		TITLE_1 = "gridded_visibility_model_natural_"
		TITLE_VISIBILITIES_RECONSTRUCTED = "reconstructed_visibility_model_natural_"
		TITLE_1_DIRTY_IMAGE = "dirty_image_gridded_model_natural_"
		TITLE_1_RECONSTRUCTED = "gridded_reconstructed_image_"
		TITLE_1_WEIGHTS = "gridded_weights_model_natural_"

		########################################## Cargar archivo de entrada Version MS
		# Eliminamos la dimension extra

		u_ind_w, v_ind_w = np.nonzero(self.gridded_weights[0]) # Se usan coordenadas no nulas de los pesos grillados.
		
		gridded_visibilities_2d = self.gridded_visibilities[0].flatten()  # (1,251,251)->(251,251)

		gridded_visibilities_2d_cuadricula = np.reshape(gridded_visibilities_2d, (N1, N1))

		gridded_weights_2d = self.gridded_weights[0].flatten()  # (1,251,251)->(251,251)

		gridded_weights_2d_cuadricula = np.reshape(gridded_weights_2d, (N1, N1))

		gc_gridded_image_data = conjugate_gradient.ConjugateGradient(self.gridded_visibilities[0], self.gridded_weights[0]/self.norm(self.gridded_weights[0].flatten(), self.gridded_visibilities[0].flatten()), self.n_iter_gc).CG()

		gridded_visibility_model_cg = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_gridded_image_data)))

		gridded_reconstructed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridded_visibility_model_cg)))

		rmse_sin_extrapolacion = self.calcular_rmse_norm(visibility_model_fits_first, gridded_visibility_model_cg)

		print("El RMSE de caso sin extrapolacion es: ", rmse_sin_extrapolacion)

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in header:
			object_name = header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		if self.plots == True:

			title=f"{object_name} griddeado + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(gridded_reconstructed_image))
			plt.colorbar(im)

			title=f"{object_name} cuadricula griddeado + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.log(np.absolute(gridded_weights_2d_cuadricula) + 0.00001))
			plt.colorbar(im)

			title = "Visibilidades fits sintetico"; fig_fits_file = plt.figure(title); plt.title(title); im_fits_file = plt.imshow(np.log(np.absolute(visibility_model_fits_first) + 0.00001))
			plt.colorbar(im_fits_file)

			title=f"Visibility model {object_name} griddeado + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.log(np.absolute(gridded_visibility_model_cg) + 0.00001)) 
			plt.colorbar(im)

			plt.show()


		# Filtramos por los valores no nulos
		nonzero_indices = np.nonzero(gridded_weights_2d)
		gv_sparse = gridded_visibilities_2d[nonzero_indices]
		gw_sparse = gridded_weights_2d[nonzero_indices]

		# Normalizacion de los datos

		# nose: SI NO FUNCIONA VOLVER ATRAS Y GUARDAR DIVISION SQRT(.....)  Y DESPUES DE EXTRAPOLAR Y ANTES DE GC MULTIPLIZAR VIS POR ESTE
		# YO CREO QUE ES UN BUG PORQUE UN NUMERO COMPLEJO AL CUADRADO ES COMPLEJO NO ES POSITIVO
		# gv_sparse = (gv_sparse / np.sqrt(np.sum(gv_sparse ** 2))) # ESTO ESTA RARO!!!!!!!
		#gw_sparse = (gw_sparse / np.sqrt(np.sum(gw_sparse ** 2))) # OJO: NORMALIZAR POR LA SUMA
		gw_sparse = (gw_sparse / np.sum(gw_sparse))

		u_data = self.grid_u[u_ind_w]
		v_data = self.grid_v[v_ind_w]

		############################################# Ploteo del Primary beam
		#if self.plots == True:
		#	plt.figure()
		#	plt.plot(gv_sparse, color='r')
		#	plt.title("Gridded visibilities distribution")



		du = 1 / (N1 * pixel_size)

		umax = N1 * du / 2

		u_sparse = np.array(u_data) / umax
		v_sparse = np.array(v_data) / umax

		u_target = np.reshape(np.linspace(-ini, ini, N1), (1, N1)) * np.ones(shape=(N1, 1))
		v_target = np.reshape(np.linspace(-ini, ini, N1), (N1, 1)) * np.ones(shape=(1, N1))

		z_target = u_target + 1j * v_target
		z_sparse = u_sparse + 1j * v_sparse

		b = trial.suggest_float("b", 0, 1)

		z_exp = np.exp(-z_target * np.conjugate(z_target) / (2 * b * b))

		max_memory = cp.cuda.Device(self.gpu_id).mem_info[1]
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

		

		# print(z_target.dtype)
		# print(z_sparse.dtype)
		# print(gw_sparse.dtype)
		# print(gv_sparse.dtype)
		# print(type(chunk_data))

		# Obtencion de los datos de la salida con G-S

		data_processing = procesamiento_datos_grillados.ProcesamientoDatosGrillados(self.fits_path, self.ms_path, S, division, self.pixel_size, self.image_size, verbose = False)

		try:
			visibilities_mini, err, residual, P_target, P = (data_processing.recurrence2d
															 (z_target.flatten(),
															  z_sparse.flatten(),
															  gw_sparse.flatten(),
															  gv_sparse.flatten(),
															  np.size(z_target.flatten()),
															  S,
															  division,
															  chunk_data,
															  b)
															 )


			#vis_rotadas = np.rot90(visibilities_mini, 3)
			#vis_flip = np.fliplr(vis_rotadas)

			#vis_gridded = self.gridded_weights[0]

			visibilities_mini = np.reshape(visibilities_mini, (N1, N1))

			# CAMBIAR DE AQUI PARA ABAJO

			vis_rotadas_extra = np.rot90(visibilities_mini, 1)
			vis_flip = np.fliplr(vis_rotadas_extra)


			# Crear una máscara booleana donde los pesos sean distintos de cero
			mascara_pesos_no_cero = gridded_weights_2d_cuadricula != 0


			# Reemplazar visibilidades originales en posiciones donde los pesos son distintos de cero
			vis_flip[mascara_pesos_no_cero] = gridded_visibilities_2d_cuadricula[mascara_pesos_no_cero]


			visibilities_model = np.array(vis_flip)

			if self.plots == True:
				plt.figure()
				plt.plot(visibilities_model.flatten(), color='g')

			weights_model = np.zeros((N1, N1), dtype=float)

			sigma_weights = np.divide(1.0, gw_sparse, where=gw_sparse != 0,
									  out=np.zeros_like(gw_sparse))  # 1.0/gw_sparse
			sigma = np.max(sigma_weights) / division
			weights_mini = np.array(1 / err)
			weights_mini[np.isnan(weights_mini)] = 0.0
			weights_mini[np.isinf(weights_mini)] = 0.0



			weights_mini = np.reshape(weights_mini, (N1, N1))

			# Reemplazar pesos originales en posiciones correspondientes (si es requerido)
			weights_mini[mascara_pesos_no_cero] = gridded_weights_2d_cuadricula[mascara_pesos_no_cero]

			weights_model = np.array(weights_mini)


			####################################### GENERACION DE GRAFICOS DE SALIDA #####################################

			image_model = (np.fft.fftshift
						   (np.fft.ifft2
							(np.fft.ifftshift
							 (visibilities_model * weights_model / np.sum(weights_model.flatten())))) * N1 ** 2)
			image_model = np.array(image_model.real)
			
			reconstructed_image = conjugate_gradient.ConjugateGradient(visibilities_model, weights_model/self.norm(weights_model.flatten(), visibilities_model.flatten()), self.n_iter_gc)

			reconstructed_image_cg = reconstructed_image.CG()

			visibility_model = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(reconstructed_image_cg)))

			gc_image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model)))

			#vis_rotadas = np.rot90(visibility_model, 3)

			# Procesamiento adicional para calcular métrica de evaluación (PSNR, MSE, etc.)

			interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
			_, _, fits_data, _, _ = interferometric_data.fits_header_info()

			visibility_model_fits = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fits_data)))


			

			if self.plots == True:
				title=f"Imagen FITS de {object_name}"; fig=plt.figure(title); plt.title(title); im=plt.imshow(fits_data)
				plt.colorbar(im)

				title=f"Imagen reconstruida de {object_name} + CG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(gc_image_model))
				plt.colorbar(im)


				plt.show()

			# Generar nombres de archivos


			TITLE_VISIBILITIES_RESULT = self.generate_filename_per_trial(TITLE_1, 
											num_intento,
											S, 
											division,
											b,
											pixel_size, 
											N1, 
											object_name, 
											"npz")

			TITLE_VISIBILITIES_RESULT_PNG = self.generate_filename_per_trial(TITLE_1, 
														num_intento,
														S, 
														division,
														b,
														pixel_size, 
														N1, 
														object_name, 
														"png")
			
			TITLE_WEIGHTS_RESULT_PNG = self.generate_filename_per_trial(TITLE_1_WEIGHTS, 
														num_intento,
														S, 
														division,
														b,
														pixel_size, 
														N1, 
														object_name, 
														"png")
			
			
			TITLE_DIRTY_IMAGE_FITS = self.generate_filename_per_trial(TITLE_1_DIRTY_IMAGE, 
													num_intento,
													S, 
													division,
													b,
													pixel_size, 
													N1, 
													object_name, 
													"fits")
			
			TITLE_RECONSTRUCTED_IMAGE_FITS = self.generate_filename_per_trial(TITLE_1_RECONSTRUCTED, 
													num_intento,
													S, 
													division,
													b,
													pixel_size, 
													N1, 
													object_name, 
													"fits")
			
			TITLE_RECONSTRUCTED_VISIBILITIES_PNG = self.generate_filename_per_trial(TITLE_VISIBILITIES_RECONSTRUCTED, 
													num_intento,
													S, 
													division,
													b,
													pixel_size, 
													N1, 
													object_name, 
													"PNG")
			

			if self.verbose == True:
				title = f"Visibility {object_name} model (dirty) (division sigma: " + f"{division:.2f}" + " num poly: " + str(S) + ")"; fig = plt.figure(title); plt.title(title); im_dirty = plt.imshow(np.log(np.absolute(visibilities_model) + 0.00001))
				plt.colorbar(im_dirty)
				plt.savefig(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_VISIBILITIES_RESULT_PNG}")

				title = "Visibilidades modelo reconstruido"; figure_vis_recons = plt.figure(title); plt.title(title); im_recons = plt.imshow(np.log(np.absolute(visibility_model) + 0.00001))
				plt.colorbar(im_recons)
				plt.savefig(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_RECONSTRUCTED_VISIBILITIES_PNG}")

				title = f"Weights {object_name} model (division sigma: " + f"{division:.2f}" + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(weights_model)
				plt.colorbar(im)
				plt.savefig(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_WEIGHTS_RESULT_PNG}")
				plt.close('all')

				# Guardar archivos
				np.savez(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_VISIBILITIES_RESULT}", visibilities_model)
				fits.writeto(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_DIRTY_IMAGE_FITS}", image_model, header, overwrite=True)
				fits.writeto(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_RECONSTRUCTED_IMAGE_FITS}", np.real(reconstructed_image_cg), header, overwrite=True)

			if self.plots == True:
				title = f"Visibility {object_name} model (dirty) (division sigma: " + f"{division:.2f}" + " num poly: " + str(S) + ")"; fig = plt.figure(title); plt.title(title); im_dirty = plt.imshow(np.log(np.absolute(visibilities_model) + 0.00001))
				plt.colorbar(im_dirty)

				title = f"Weights {object_name} model (division sigma: " + f"{division:.2f}" + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(weights_model)
				plt.colorbar(im)
				
				title = "Visibility model original"; fig_vis_original = plt.figure(title); plt.title(title); im_vis_original = plt.imshow(np.log(np.absolute(self.gridded_visibilities[0]) + 0.00001))
				plt.colorbar(im_vis_original)

				title = "Visibilidades modelo reconstruido"; figure = plt.figure(title); plt.title(title); im_recons = plt.imshow(np.log(np.absolute(visibility_model) + 0.00001))
				plt.colorbar(im_recons)

				title = "Visibilidades fits sintetico"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(visibility_model_fits) + 0.00001))
				plt.colorbar(im)

				plt.show()

				
			#psnr_result = self.psnr(data, np.real(gc_image_model))

			rmse = self.calcular_rmse_norm(visibility_model_fits, visibility_model)

			print("El RMSE de caso con extrapolacion es: ", rmse)

			print("El tiempo de ejecución fue de: ", time.time() - start_time)

			gc.collect()
			cp.get_default_memory_pool().free_all_blocks()
			

			# Minimizar ambas métricas (menores valores indican mejor calidad)
			return rmse
		
		except Exception as e:
			print(f"Error en el cálculo: {e}")
			gc.collect()
			cp.get_default_memory_pool().free_all_blocks()
			return float("inf")
		
		"""
			psnr_result = self.psnr(np.real(image_model))
			return -psnr_result  # Negativo porque Optuna minimiza la métrica
		except Exception as e:
			print(f"Error en el cálculo: {e}")
			return float("inf")  # Penalizar valores inválidos
		"""
		

	def initialize_optimization(self, num_trials):
		"""
		Inicializa el estudio de Optuna con samplers, pruners, y callbacks inteligentes.
		"""

		cp.cuda.runtime.setDevice(self.gpu_id)

		start_time = time.time()

		# ==== Definición de estrategia de búsqueda ====
		pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
		sampler = optuna.samplers.TPESampler(multivariate=True)

		study = optuna.create_study(
			direction="minimize",
			sampler=sampler,
			pruner=pruner
		)

		# ==== Callback para guardar progreso ====
		def save_study_callback(study, trial):
			try:
				study.trials_dataframe().to_csv(
					"/disk2/stephan/opti_kernel_5_grid/opti_hd142/study_trials_backup.csv",
					index=False
				)
			except Exception as e:
				print(f"Error guardando backup del estudio: {e}")

		# ==== Comenzar optimización ====
		study.optimize(self.optimize_parameters, n_trials=num_trials, callbacks=[save_study_callback])

		# ==== Mostrar y guardar resultados ====
		print("Mejores parámetros encontrados:", study.best_params)
		print("Mejor valor (RMSE):", study.best_value)

		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		visibility_model_fits = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fits_data)))

		gc_gridded_image_data = conjugate_gradient.ConjugateGradient(self.gridded_visibilities[0], self.gridded_weights[0]/self.norm(self.gridded_weights[0].flatten(), self.gridded_visibilities[0].flatten()), self.n_iter_gc).CG()

		gridded_visibility_model_cg = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_gridded_image_data)))

		gridded_reconstructed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridded_visibility_model_cg)))


		TITLE_1_OPTUNA = "gridded_optimimum_parameters_"
		TITLE_1_PLOT = "gridded_convergence_plot_"

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		TITLE_VISIBILITIES_FITS = f"{object_name}_fits_file_visibilities"

		if self.verbose == True:

			title = "Visibilidades fits sintetico"; fig_fits_file = plt.figure(title); plt.title(title); im_fits_file = plt.imshow(np.log(np.absolute(visibility_model_fits) + 0.00001))
			plt.colorbar(im_fits_file)
			plt.savefig(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_VISIBILITIES_FITS}.png")

			title = "Visibility model original"; fig_vis_original = plt.figure(title); plt.title(title); im_vis_original = plt.imshow(np.log(np.absolute(self.gridded_visibilities[0]) + 0.00001))
			plt.colorbar(im_vis_original)
			plt.savefig(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{f"visibilidades_orig_{object_name}.png"}")

			title=f"Imagen reconstruida de {object_name} griddeado orig + CG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.log(np.absolute(gridded_visibility_model_cg) + 0.00001)) 
			plt.colorbar(im)
			plt.savefig(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{f"vis_imagen_griddeada_orig_CG_{object_name}.png"}")

			fits.writeto(f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{f"imagen_griddeada_original_CG_{object_name}.fits"}", np.real(gridded_reconstructed_image), fits_header, overwrite=True)

		if self.plots == True:
			title = "Visibilidades fits sintetico"; fig_fits_file = plt.figure(title); plt.title(title); im_fits_file = plt.imshow(np.log(np.absolute(visibility_model_fits) + 0.00001))
			plt.colorbar(im_fits_file)


			title = "Visibility model original"; fig_vis_original = plt.figure(title); plt.title(title); im_vis_original = plt.imshow(np.log(np.absolute(self.gridded_visibilities[0]) + 0.00001))
			plt.colorbar(im_vis_original)

			plt.show()

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

		# Visualizar evolución
		fig_convergence = plot_optimization_history(study)
		fig_convergence.update_layout(
			title="Historial de Optimización",
			xaxis_title="Iteraciones",
			yaxis_title="RMSE"
		)

		# Mostrar en vivo (opcional)
		if self.plots:
			show(fig_convergence)

		if self.verbose == True:

			# Guardar gráfico final
			fig_convergence.write_image(
				f"/disk2/stephan/opti_kernel_5_grid/opti_hd142/{TITLE_PLOT_RESULT}"
			)

		tiempo_total_opti = time.time() - start_time
		
		print(f"El tiempo de ejecución de optimizacion fue de: {tiempo_total_opti:.2f} segundos ")

		# Guardar el tiempo de ejecución en un archivo de texto
		with open(TITLE_OPTUNA_RESULT , "w") as file:
			file.write(f"Mejores parametros: {study.best_params}\n\n Mejor valor (RMSE): {study.best_value}\n\n Tiempo total de ejecucion: {tiempo_total_opti:.6f}")

		print("Optimización completada exitosamente.")
