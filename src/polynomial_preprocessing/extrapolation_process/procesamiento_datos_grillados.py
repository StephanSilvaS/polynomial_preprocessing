from polynomial_preprocessing.preprocessing import preprocesamiento_datos_a_grillar
from astropy.io import fits
from numba import jit, prange
from astropy.coordinates import Angle
import cupy as cp
import numpy as np
import math
import multiprocessing as mp
from astropy.wcs import WCS
from matplotlib.colors import Normalize
import time
import matplotlib.pyplot as plt
import astropy.units as unit
import dask.array as da
from dask import delayed, compute
from dask.distributed import Client
from numba import jit, prange, complex128, float64, int32
from dask.distributed import Client, LocalCluster
from polynomial_preprocessing.image_reconstruction import conjugate_gradient
#from polynomial_preprocessing.extrapolation_process import gram_schmidt_extrapolation


class ProcesamientoDatosGrillados:
	def __init__(self, fits_path, ms_path, num_polynomial, division_sigma, b = 1, pixel_size=None, image_size=None, n_iter_gc = 10, verbose = True, plots = False, conv_gridding = False, gpu_id = 0):
		self.fits_path = fits_path
		self.ms_path = ms_path
		self.num_polynomial = num_polynomial
		self.division_sigma = division_sigma
		self.b = b
		self.pixel_size = pixel_size
		self.image_size = image_size
		self.n_iter_gc = n_iter_gc
		self.verbose = verbose
		self.plots = plots
		self.conv_gridding = conv_gridding
		self.gpu_id = gpu_id

		if self.pixel_size is None:
			pixel_size = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						ms_path=self.ms_path)
			_, _, _, _, pixels_size = pixel_size.fits_header_info()
			print("Pixel size of FITS on degree: ", pixels_size)
			
			# Se requiere transformar de grados a radianes el tam. de pixel.
			angulo = Angle(pixels_size, unit='deg')

			pixels_size_rad = angulo.radian * unit.rad

			print("Pixel size of FITS on radians: ", pixels_size_rad)
			self.pixel_size = pixels_size_rad


		if self.image_size is None:
			fits_header = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						 ms_path=self.ms_path)

			_, fits_dimensions, _, _, _ = fits_header.fits_header_info()
			print("Image size of FITS: ", fits_dimensions[1])
			self.image_size = fits_dimensions[1]

	def data_processing(self):
		gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v = self.grid_data()
		image_model, visibilities_model, weights_model, u_target, v_target, reconstructed_image, vis_reconstructed = self.gridded_data_processing(gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v)
		return image_model, visibilities_model, weights_model, u_target, v_target, reconstructed_image, vis_reconstructed

	@staticmethod
	def norm(weights, x):
		return(np.absolute(np.sqrt(np.sum(weights*np.absolute(x)**2))))

	def umbralizar_fits(self, umbral):
		"""
		Carga un archivo FITS, umbraliza sus valores (valores menores a `umbral` se hacen cero) y devuelve el arreglo modificado.

		Parámetros:
		path_fits : str
			Ruta al archivo FITS.
		umbral : float
			Valor mínimo de magnitud para conservar. Todo valor con |valor| < umbral se reemplaza por 0.
		extension : int
			Número de la extensión del HDU donde están los datos (por defecto: 0).

		Retorna:
		datos_umbralizados : ndarray
			Arreglo con los valores menores al umbral convertidos en cero.
		"""
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		# Aplica el umbral
		mascara = fits_data < umbral
		fits_data[mascara] = 0

		fits.writeto(f"/disk2/stephan/fits_umbralizados/{object_name}_modified.fits", fits_data, fits_header, overwrite=True)

		return fits_data

	def reemplazar_nan_fits(self):
		"""
		Carga un archivo FITS, reemplaza valores NaN por 0, y guarda el archivo modificado.

		Retorna:
		datos_limpiados : ndarray
			Arreglo con los valores NaN reemplazados por 0.
		"""

		# Cargar los datos desde el FITS
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(
			fits_path=self.fits_path,
			ms_path=self.ms_path
		)

		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		# Reemplazar NaNs por 0
		fits_data = np.nan_to_num(fits_data, nan=0.0)

		# Guardar nuevo archivo FITS
		output_path = f"/disk2/stephan/fits_nan_removidos/{object_name}_nan0.fits"
		fits.writeto(output_path, fits_data, fits_header, overwrite=True)
		print(f"Archivo guardado en: {output_path}")

		return fits_data

	def porcentaje_plano_lleno(self):
		gridded_visibilities, _, _, _, _ = self.grid_data()

		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		visibilidades = gridded_visibilities[0]

		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		title = "Visibility model original"; fig_vis_original = plt.figure(title); plt.title(title); im_vis_original = plt.imshow(np.log(np.absolute(gridded_visibilities[0]) + 0.00001))
		plt.colorbar(im_vis_original)
		plt.savefig(f"/disk2/stephan/fits_visibilidades/{f"visibilidades_orig_{object_name}.png"}")

		print("visibilidades: ", visibilidades)

		print("visibilidades.size: ", visibilidades.size)

		# Calcula la cantidad total de elementos
		total_elementos = visibilidades.size
		
		# Crea una máscara booleana para detectar elementos diferentes a 0.+0.j
		elementos_no_cero = visibilidades != 0.0 + 0.0j
		
		# Cuenta cuántos elementos son diferentes de cero
		cantidad_no_ceros = np.count_nonzero(elementos_no_cero)

		print("cantidad_no_ceros: ", cantidad_no_ceros)
		
		# Calcula el porcentaje
		porcentaje = (cantidad_no_ceros / total_elementos) * 100

		print(f"Porcentaje de elementos distintos de cero: {porcentaje:.4f}%")
		
		return porcentaje
	
	def porcentaje_no_ceros(self):
		
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		TITLE_VISIBILITIES_FITS = f"{object_name}_fits_file_visibilities"
		TITLE_IMAGE_FITS = f"{object_name}_fits_file_image"

		visibility_model_fits = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fits_data)))

		print("visibility_model_fits: ", visibility_model_fits)
		
		# Calcula la cantidad total de elementos
		total_elementos = visibility_model_fits.size
		
		# Crea una máscara booleana para detectar elementos diferentes a 0.+0.j
		elementos_no_cero = visibility_model_fits != 0.0 + 0.0j
		
		# Cuenta cuántos elementos son diferentes de cero
		cantidad_no_ceros = np.count_nonzero(elementos_no_cero)

		print("cantidad_no_ceros: ", cantidad_no_ceros)
		
		# Calcula el porcentaje
		porcentaje = (cantidad_no_ceros / total_elementos) * 100

		print(f"Porcentaje de elementos distintos de cero: {porcentaje:.4f}%")
		
		return porcentaje


	def convolutional_gridding(self):

		start_time = time.time()

		visibilidades_grilladas, pesos_grillados, _, _, _ = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path, ms_path=self.ms_path, pixel_size=self.pixel_size).process_ms_file()

		gridded_visibilities_2d = visibilidades_grilladas[0]  # (1,251,251)->(251,251)

		if self.plots == True:
			title = "Visibility model"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(gridded_visibilities_2d) + 0.00001))
			plt.colorbar(im)

			plt.show()

		gridded_weights_2d = pesos_grillados[0]  # (1,251,251)->(251,251)

		gc_image_1 = conjugate_gradient.ConjugateGradient(gridded_visibilities_2d, gridded_weights_2d/self.norm(gridded_weights_2d.flatten(), gridded_visibilities_2d.flatten()), self.n_iter_gc)

		gc_image_data = gc_image_1.CG()

		visibility_model = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_image_data)))

		gc_image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model)))

		#reconstructed_image = np.rot90(gc_image_model, 2)

		fits_header = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																						 ms_path=self.ms_path)

		header, _, _, _, _ = fits_header.fits_header_info()

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in header:
			object_name = header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		fits.writeto(f"/disk2/stephan/convolutional_gridding_kernel_5/convolutional_gridding_kernel_5_{object_name}.fits", np.real(gc_image_model), header, overwrite=True)

		if self.plots == True:
			title="Gridding de Convolucion + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(gc_image_model))
			plt.colorbar(im)

			title="Visibility model + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(visibility_model))
			plt.colorbar(im)

			plt.show()

		# Finalizar el contador de tiempo
		end_time = time.time()

		# Calcular el tiempo de ejecución
		execution_time = end_time - start_time

		print(f"Tiempo de ejecución (Gridding de Conv.): {execution_time:.2f} segundos")
		
		return gc_image_model, visibility_model

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
		B = img_final * bool_arreglo
		mse = np.std(B) ** 2
		print(mse)
		return mse

	def psnr(self, dim_grilla, radio):
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		_, _, fits_data, _, _ = interferometric_data.fits_header_info()
		psnr_result = 10 * math.log10(( np.max(fits_data) ** 2 ) / self.mse(fits_data, (dim_grilla, dim_grilla), radio))
		return psnr_result  # comentary mse need to be taken outside the object

	def grid_data(self):

		print("self.pixel_size: ", self.pixel_size)
		print("self.image_size: ", self.image_size)

		gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v = (preprocesamiento_datos_a_grillar.
																		  PreprocesamientoDatosAGrillar(self.fits_path,
																										self.ms_path,																										
																										image_size = self.image_size,
																										pixel_size = self.pixel_size,
																										plots = self.plots
																										).
																		  process_ms_file())
		
		return gridded_visibilities, gridded_weights, pixel_size,  grid_u, grid_v

	@staticmethod
	def norm(weights,x):
		return(np.absolute(np.sqrt(np.sum(weights*np.absolute(x)**2))))
	

	def graficar_visibilidades_sinteticos_grid_extra(self):
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		TITLE_VISIBILITIES_FITS = f"{object_name}_fits_file_visibilities"
		TITLE_IMAGE_FITS = f"{object_name}_fits_file_image"

		visibility_model_fits = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fits_data)))

		title = "Visibility fits original"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(visibility_model_fits) + 0.000001))
		plt.colorbar(im)
		plt.savefig(f"/disk2/stephan/imagenes_visibilidades_finales_kernel_5/{TITLE_VISIBILITIES_FITS}.png")

		# Supongo que ya tienes `fits_data` cargado como arreglo 2D
		title = f"Imagen FITS de {object_name}"

		vis_rotadas = np.rot90(fits_data, 2)

		fig, ax = plt.subplots(figsize=(8, 6))
		im = ax.imshow(vis_rotadas, cmap="plasma", origin='lower', 
					norm=Normalize(vmin=np.min(vis_rotadas), vmax=np.max(vis_rotadas)))
		ax.set_title(title)
		ax.set_xlabel("Pixel eje X")
		ax.set_ylabel("Pixel eje Y")
		plt.colorbar(im, ax=ax, label="Intensidad")
		plt.grid(True, color='white', linestyle='--', linewidth=0.5)
		plt.tight_layout()

		# Guardar figura
		plt.savefig(f"/disk2/stephan/imagenes_visibilidades_finales_kernel_5/{TITLE_IMAGE_FITS}.png", dpi=300, bbox_inches='tight')  # <- ESTA ES LA LÍNEA IMPORTANTE

		plt.show()


	def graficar_visibilidades_sinteticos_refer(self):
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, fits_data, _, _ = interferometric_data.fits_header_info()

		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		TITLE_VISIBILITIES_FITS = f"{object_name}_fits_file_visibilities"
		TITLE_IMAGE_FITS = f"{object_name}_fits_file_image_referencia"

		visibility_model_fits = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fits_data)))

		title = "Visibility fits original"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(visibility_model_fits) + 0.000001))
		plt.colorbar(im)
		plt.savefig(f"/disk2/stephan/imagenes_visibilidades_finales_kernel_5/{TITLE_VISIBILITIES_FITS}.png")

		# Supongo que ya tienes `fits_data` cargado como arreglo 2D
		title = f"Imagen FITS de {object_name}"

		#vis_rotadas = np.rot90(fits_data, 2)

		fig, ax = plt.subplots(figsize=(8, 6))
		im = ax.imshow(fits_data, cmap="plasma", origin='lower', 
					norm=Normalize(vmin=np.min(fits_data), vmax=np.max(fits_data)))
		ax.set_title(title)
		ax.set_xlabel("Pixel eje X")
		ax.set_ylabel("Pixel eje Y")
		plt.colorbar(im, ax=ax, label="Intensidad")
		plt.grid(True, color='white', linestyle='--', linewidth=0.5)
		plt.tight_layout()

		# Guardar figura
		plt.savefig(f"/disk2/stephan/imagenes_visibilidades_finales_kernel_5/{TITLE_IMAGE_FITS}.png", dpi=300, bbox_inches='tight')  # <- ESTA ES LA LÍNEA IMPORTANTE

		plt.show()

		


	def graficar_imagen_gc_original(self):
		gridded_visibilities, gridded_weights, _, _, _ = (preprocesamiento_datos_a_grillar.
																		  PreprocesamientoDatosAGrillar(self.fits_path,
																										self.ms_path,																										
																										image_size = self.image_size,
																										pixel_size = self.pixel_size,
																										plots = self.plots
																										).
																		  process_ms_file())
		
		interferometric_data = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path=self.fits_path,
																							   ms_path=self.ms_path)
		fits_header, _, _, _, _ = interferometric_data.fits_header_info()

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")
			
		gc_gridded_image_data = conjugate_gradient.ConjugateGradient(gridded_visibilities[0], gridded_weights[0]/self.norm(gridded_weights[0].flatten(), gridded_visibilities[0].flatten()), self.n_iter_gc).CG()

		gridded_visibility_model_cg = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_gridded_image_data)))

		gridded_reconstructed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridded_visibility_model_cg)))

		title = "Visibility fits original + CG"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(gridded_visibility_model_cg) + 0.000001))
		plt.colorbar(im)
		plt.savefig(f"/disk2/stephan/optim_sz114_con_param_b_v3/{f"vis_gridded_image_CG_{object_name}"}.png")

		#TERMINAR DE ARREGLAR FUNCION
		fits.writeto(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{f"imagen_orig_griddeada_con_CG"}", np.real(gridded_reconstructed_image), fits_header, overwrite=True)

	def gridded_data_processing(self, gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v):

		cp.cuda.runtime.setDevice(self.gpu_id)

		start_time = time.time()

		# Cargamos los archivos de entrada
		fits_header, _, _, _, _ = (preprocesamiento_datos_a_grillar.
																	PreprocesamientoDatosAGrillar(self.fits_path,
																								self.ms_path,																									
																								image_size = self.image_size,
																								pixel_size=self.pixel_size,
																								plots = self.plots
																								).
																	fits_header_info())


		################# Parametros iniciales #############
		M = 1  # Multiplicador de Pixeles
		pixel_num = self.image_size  # Numero de pixeles
		pixel_num = pixel_num * M  # Numero de pixeles,  multiplicador #Version MS
		num_polynomial = self.num_polynomial # Numero de polinomios
		sub_S = int(num_polynomial)
		ini = 1  # Tamano inicial
		division = self.division_sigma # division_sigma
		pixel_size = self.pixel_size

		# Constantes para archivos de salida
		TITLE_1 = "gridded_visibility_model_natural_"
		TITLE_1_DIRTY_IMAGE = "dirty_image_gridded_model_natural_"
		TITLE_1_WEIGHTS = "gridded_weights_model_natural_"
		TITLE_1_TIME = "execution_time_gridded_"
		TITLE_1_RECONSTRUCTED = "gridded_reconstructed_image_"

		########################################## Cargar archivo de entrada Version MS
		# Eliminamos la dimension extra
		# u_ind, v_ind = np.nonzero(gridded_visibilities[0])
		u_ind_w, v_ind_w = np.nonzero(gridded_weights[0]) # Se usan coordenadas no nulas de los pesos grillados.

		gridded_visibilities_2d = gridded_visibilities[0].flatten() 

		gridded_visibilities_2d_cuadricula = np.reshape(gridded_visibilities_2d, (pixel_num, pixel_num))
		gridded_weights_2d = gridded_weights[0].flatten()

		gridded_weights_2d_cuadricula = np.reshape(gridded_weights_2d, (pixel_num, pixel_num))

		# Filtramos por los valores no nulos

		nonzero_indices = np.nonzero(gridded_weights_2d)

		gv_sparse = gridded_visibilities_2d[nonzero_indices]
		gw_sparse = gridded_weights_2d[nonzero_indices]

		# Normalizacion de los datos

		#gv_sparse = (gv_sparse / np.sqrt(np.sum(gv_sparse ** 2)))
		gw_sparse = (gw_sparse / np.sum(gw_sparse))

		u_data = grid_u[u_ind_w]
		v_data = grid_v[v_ind_w]

		print("u_data : ", u_data.shape)

		print("v_data: ", v_data.shape)

		############################################# Ploteo del Primary beam
		if self.plots == True:
			plt.figure()
			plt.plot(gv_sparse, color='r')
			plt.title("Gridded visibilities distribution")

		du = 1 / (pixel_num * pixel_size)

		umax = pixel_num * du / 2 # EL DIVIDIDO EN 2 ESTÁ HARDCODEADO

		u_sparse = np.array(u_data) / umax
		v_sparse = np.array(v_data) / umax

		if self.plots == True:
			plt.figure()
			plt.xlim(-1, 1)
			plt.ylim(-1, 1)
			plt.scatter(u_sparse, v_sparse, s=1)
			plt.title("Gridded uv coverage")

		u_target = np.reshape(np.linspace(-ini, ini, pixel_num), (1, pixel_num)) * np.ones(shape=(pixel_num, 1))
		v_target = np.reshape(np.linspace(-ini, ini, pixel_num), (pixel_num, 1)) * np.ones(shape=(1, pixel_num))

		z_target = u_target + 1j * v_target
		z_sparse = u_sparse + 1j * v_sparse

		b = self.b

		z_exp = np.exp(-z_target * np.conjugate(z_target) / (2 * b * b))

		if self.plots == True:
			title = "Z exp"
			fig = plt.figure(title)
			plt.title(title)
			im = plt.imshow(np.abs(z_exp))  # Usar np.abs para evitar el warning
			plt.colorbar(im)
			plt.show()
			
		
		max_memory = cp.cuda.Device(self.gpu_id).mem_info[1]
		max_data = float(int(max_memory / (num_polynomial * num_polynomial)))

		divide_data = int(np.size(gv_sparse[np.absolute(gv_sparse) != 0].flatten()) / max_data) + 1
		divide_target = int(pixel_num * pixel_num / max_data) + 1

		if divide_target > divide_data:
			divide_data = int(divide_target)

		if divide_data > int(divide_data):
			divide_data = int(divide_data) + 1

		chunk_data = int(((num_polynomial * num_polynomial) / divide_data) ** (1 / 2)) + 1
		if chunk_data == 0:
			chunk_data = 1

		# chunk_data = 1
		print(chunk_data)

		visibilities_model = np.zeros((pixel_num, pixel_num), dtype=np.complex128)

		print("Max. polynomial degree:", num_polynomial)
		print("Division:", division)

		visibilities_aux = np.zeros(pixel_num * pixel_num, dtype=np.complex128)
		weights_aux = np.zeros(pixel_num * pixel_num, dtype=float)


		# print(z_target.dtype)
		# print(z_sparse.dtype)
		# print(gw_sparse.dtype)
		# print(gv_sparse.dtype)
		# print(type(chunk_data))

		# Obtencion de los datos de la salida con G-S

		visibilities_mini, err, residual, P_target, P = (self.recurrence2d
														 (z_target.flatten(),
														  z_sparse.flatten(),
														  gw_sparse.flatten(),
														  gv_sparse.flatten(),
														  np.size(z_target.flatten()),
														  num_polynomial,
														  division,
														  chunk_data,
														  b)
														 )

		#vis_rotadas = np.rot90(visibilities_mini, 3)
		#vis_flip = np.fliplr(vis_rotadas)

		#vis_gridded = self.gridded_weights[0]

		visibilities_mini = np.reshape(visibilities_mini, (pixel_num, pixel_num))

		# CAMBIAR DE AQUI PARA ABAJO

		vis_rotadas_extra = np.rot90(visibilities_mini, 1)
		vis_flip = np.fliplr(vis_rotadas_extra)


		# Crear una máscara booleana donde los pesos sean distintos de cero
		mascara_pesos_no_cero = gridded_weights_2d_cuadricula != 0


		# Reemplazar visibilidades originales en posiciones donde los pesos son distintos de cero
		vis_flip[mascara_pesos_no_cero] = gridded_visibilities_2d_cuadricula[mascara_pesos_no_cero]



		visibilities_model = np.array(vis_flip)


		#residual_mini = np.reshape(residual, (pixel_num, pixel_num))

		#residual_model = np.array(residual_mini)

		if self.plots == True:
			plt.figure()
			plt.plot(visibilities_model.flatten(), color='g')

		
		weights_model = np.zeros((pixel_num, pixel_num), dtype=float)

		sigma_weights = np.divide(1.0, gw_sparse, where=gw_sparse != 0, out=np.zeros_like(gw_sparse))  # 1.0/gw_sparse
		sigma = np.max(sigma_weights) / division
		weights_mini = np.array(1 / err)
		weights_mini[np.isnan(weights_mini)] = 0.0
		weights_mini[np.isinf(weights_mini)] = 0.0

		weights_mini = np.reshape(weights_mini, (pixel_num, pixel_num))

		# Reemplazar pesos originales en posiciones correspondientes (si es requerido)
		weights_mini[mascara_pesos_no_cero] = gridded_weights_2d_cuadricula[mascara_pesos_no_cero]

		weights_model = np.array(weights_mini)



		####################################### GENERACION DE GRAFICOS DE SALIDA #####################################

		image_model = (np.fft.fftshift
					   (np.fft.ifft2
						(np.fft.ifftshift
						 (visibilities_model * weights_model / np.sum(weights_model.flatten())))) * pixel_num ** 2)
		image_model = np.array(image_model.real)

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header:
			object_name = fits_header['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")


		if self.plots == True:
			title = f"Dirty Image de {object_name} (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(image_model)
			plt.colorbar(im)

			title = f"Visibility {object_name} model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(visibilities_model) + 0.00001))
			plt.colorbar(im)

			title = f"Weights {object_name} model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(weights_model)
			plt.colorbar(im)

			#title="Residual model (division sigma: "+str(division)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(residual_model)
			#plt.colorbar(im)

			plt.show()


		gc_image_data = conjugate_gradient.ConjugateGradient(visibilities_model, weights_model/self.norm(weights_model.flatten(), visibilities_model.flatten()), self.n_iter_gc).CG()

		visibility_model_cg = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_image_data)))

		reconstructed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model_cg)))

		if self.plots == True:

			title=f"Extrapolacion de {object_name} + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(reconstructed_image))
			plt.colorbar(im)

			title=f"Visibility model {object_name} + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.log(np.absolute(visibility_model_cg) + 0.00001))
			plt.colorbar(im)

		# Finalizar el contador de tiempo
		end_time = time.time()

		# Calcular el tiempo de ejecución
		execution_time = end_time - start_time

		print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

		if self.verbose == True:

			# Generar nombres de archivos
			TITLE_TIME_RESULT = self.generate_filename(TITLE_1_TIME, 
														num_polynomial, 
														division,
														pixel_size, 
														pixel_num, 
														object_name, 
														"txt")
			


			# Guardar el tiempo de ejecución en un archivo de texto
			with open(TITLE_TIME_RESULT , "w") as file:
				file.write(f"Tiempo de ejecucion: {execution_time:.2f} segundos\n")

			# Generar nombres de archivos
			TITLE_VISIBILITIES_RESULT = self.generate_filename(TITLE_1, 
													  num_polynomial, 
													  division,
													  pixel_size, 
													  pixel_num, 
													  object_name, 
													  "npz")
			
			TITLE_WEIGHTS_RESULT = self.generate_filename(TITLE_1_WEIGHTS, 
												 num_polynomial, 
												 division,
												 pixel_size, 
												 pixel_num, 
												 object_name, 
												 "npz")
			
			# Generar nombres de archivos
			TITLE_VISIBILITIES_RESULT_PNG = self.generate_filename(TITLE_1, 
													  num_polynomial, 
													  division,
													  pixel_size, 
													  pixel_num, 
													  object_name, 
													  "png")
			
			TITLE_WEIGHTS_RESULT_PNG = self.generate_filename(TITLE_1_WEIGHTS, 
												 num_polynomial, 
												 division,
												 pixel_size, 
												 pixel_num, 
												 object_name, 
												 "png")
			
			TITLE_DIRTY_IMAGE_FITS = self.generate_filename(TITLE_1_DIRTY_IMAGE, 
												   num_polynomial, 
												   division,
												   pixel_size, 
												   pixel_num, 
												   object_name, 
												   "fits")
			
			TITLE_RECONSTRUCTED_IMAGE_FITS = self.generate_filename(TITLE_1_RECONSTRUCTED, 
													num_polynomial, 
													division,
													pixel_size, 
													pixel_num, 
													object_name, 
													"fits")
			
			title = f"Visibility {object_name} model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(visibilities_model) + 0.00001))
			plt.colorbar(im)
			plt.savefig(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{TITLE_VISIBILITIES_RESULT_PNG}")

			title = f"Weights {object_name} model (division sigma: " + str(division) + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(weights_model)
			plt.colorbar(im)
			plt.savefig(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{TITLE_WEIGHTS_RESULT_PNG}")


			# Guardar archivos
			np.savez(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{TITLE_VISIBILITIES_RESULT}", visibilities_model)
			np.savez(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{TITLE_WEIGHTS_RESULT}", weights_model)
			fits.writeto(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{TITLE_DIRTY_IMAGE_FITS}", image_model, fits_header, overwrite=True)
			fits.writeto(f"/disk2/stephan/output_oficiales_kernel_5/HD163296/{TITLE_RECONSTRUCTED_IMAGE_FITS}", np.real(reconstructed_image), fits_header, overwrite=True)

		return image_model, visibilities_model, weights_model, u_target, v_target, np.real(reconstructed_image), np.absolute(visibility_model_cg)

	# Función para generar nombres de archivos
	@staticmethod
	def generate_filename(prefix, num_polynomials, division, pixel_size, num_pixels, object_name, extension):
		base_title = f"num_polynomial_{num_polynomials}_division_sigma_{division}_pixel_size_{pixel_size}_image_size_{num_pixels}_{num_pixels}_{object_name}"
		return f"{prefix}{base_title}.{extension}"

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
	@jit(parallel=True)
	def initialize_polynomials_cpu(z, z_target, w, s):
		P = np.zeros((s, s, len(z)), dtype=np.complex128)
		P_target = np.zeros((s, s, len(z_target)), dtype=np.complex128)

		for j in prange(s):
			for k in range(s):
				P[k, j, :] = (z ** (k)) * (np.conjugate(z) ** j)
				P_target[k, j, :] = (z_target ** (k)) * (np.conjugate(z_target) ** j)

				# Normalización
				no = np.sqrt(np.sum(w * np.abs(P[k, j, :]) ** 2))
				if no != 0:
					P[k, j, :] /= no
					P_target[k, j, :] /= no

		return P, P_target
	
	@staticmethod
	@jit(parallel=True)
	def norm2x2(weights, matrix, chunk_data):
		N1, N2, n = matrix.shape
		sub_size = (N1 // chunk_data) + 1
		final_norm = np.zeros((N1, N2, 1), dtype=np.complex128)

		for chunk1 in prange(sub_size):
			for chunk2 in prange(sub_size):
				if chunk1 + chunk2 < sub_size:
					# Extraer el subarray de matrix
					start1 = chunk1 * chunk_data
					end1 = min((chunk1 + 1) * chunk_data, N1)
					start2 = chunk2 * chunk_data
					end2 = min((chunk2 + 1) * chunk_data, N2)
					
					sub_m = matrix[start1:end1, start2:end2, :]
					N3, N4, n2 = sub_m.shape

					# Calcular subsum directamente sin broadcast
					subsum = np.zeros((N3, N4), dtype=np.float64)
					for i in prange(N3):
						for j in prange(N4):
							for k in range(n2):
								subsum[i, j] += weights[k] * np.abs(sub_m[i, j, k])**2
							subsum[i, j] = np.sqrt(subsum[i, j])

					# Asignar el resultado a la matriz final_norm
					final_norm[start1:end1, start2:end2, 0] = subsum

		return final_norm

	def normalize_initial_polynomials_cpu(self, z_target, z, w, P, P_target, V, s, chunk_data, b):
		P = P*np.exp(-z*np.conjugate(z)/(2*(b**2)))
		P_target = P_target*np.exp(-z_target*np.conjugate(z_target)/(2*(b**2)))
		no_data = self.norm2x2(w, P, chunk_data)

		# Dividimos por no_data solo en posiciones donde no_data no es cero
		for i in range(P.shape[0]):
			for j in range(P.shape[1]):
				if V[i, j, 0] != 0:  # Solo si el valor en V es distinto de cero
					if no_data[i, j] != 0:  # Evitar división por cero
						P[i, j, :] = P[i, j, :] / no_data[i, j]
						P_target[i, j, :] = P_target[i, j, :] / no_data[i, j]

		return P, P_target

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

	def gram_schmidt_and_estimation_gpu(self, w, P, P_target, V, D, D_target, residual, final_data, err, s, sigma2,
										max_rep,
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

						# Liberar memoria intermedia
						del dot_data
						cp.get_default_memory_pool().free_all_blocks()

					if repeat > 0:
						for y in range(0, k + 1):
							for x in range(0, y + 1):
								if (y - x != k - j) or (j != x):
									dot_data = self.dot(w, P[k - j, j, :], P[y - x, x, :])
									P[k - j, j, :] -= dot_data * P[y - x, x, :]
									P_target[k - j, j, :] -= dot_data * P_target[y - x, x, :]
								else:
									break
							# Normalización
							no = cp.sqrt(cp.sum(w * cp.abs(P[k - j, j, :]) ** 2))
							if no != 0:
								P[k - j, j, :] /= no
								P_target[k - j, j, :] /= no

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

		

		# Aplicar el criterio de selección sigma2
		final_data[err > sigma2] = 0

		# Liberar memoria
		del M, V, D, D_target, w
		cp.get_default_memory_pool().free_all_blocks()

		# Convertir las salidas de nuevo a NumPy para evitar errores fuera de esta función
		return cp.asnumpy(final_data), cp.asnumpy(residual), cp.asnumpy(err), cp.asnumpy(P_target), cp.asnumpy(P)
	
	@staticmethod
	def norm(weights,x):
		return(np.sqrt(np.sum(weights*np.absolute(x)**2)))
	
	@staticmethod
	def dot(weights,x,y):
		return(np.sum((x*weights*np.conjugate(y))))


	def generar_imagen_residuo(self, imagen_gridding, imagen_extrapolada):
		"""
		Calcula la imagen residuo entre una imagen obtenida por gridding convolucional y una imagen extrapolada.

		Parámetros:
		- imagen_gridding: np.ndarray, imagen obtenida mediante gridding (por ejemplo, iFFT de datos gridded)
		- imagen_extrapolada: np.ndarray, imagen extrapolada mediante modelo (por ejemplo, polinomios o HMC)

		Retorna:
		- residuo: np.ndarray, diferencia punto a punto (residuo)
		"""

		interferometric_data_grid = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path = imagen_gridding,
																							   ms_path=self.ms_path)
		fits_header_grid, _, imagen_grid, _, _ = interferometric_data_grid.fits_header_info()

		interferometric_data_extrapolada = preprocesamiento_datos_a_grillar.PreprocesamientoDatosAGrillar(fits_path = imagen_extrapolada,
																							   ms_path=self.ms_path)
		fits_header_extra, _, imagen_extra, _, _ = interferometric_data_extrapolada.fits_header_info()

		if imagen_grid.shape != imagen_extra.shape:
			raise ValueError("Las imágenes deben tener la misma forma para calcular el residuo.")

		residuo = imagen_grid - imagen_extra

		# Buscar el atributo OBJECT en el header
		if 'OBJECT' in fits_header_grid:
			object_name = fits_header_grid['OBJECT']
			print(f"El objeto en el archivo FITS es: {object_name}")
		else:
			object_name = "no_object_name"
			print("El atributo OBJECT no se encuentra en el header.")

		title=f"Imagen residual {object_name}"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(residuo))
		plt.colorbar(im)


		fits.writeto(f"/disk2/stephan/imagenes_residuo/imagen_residuo_{object_name}.fits", np.real(residuo), fits_header_extra, overwrite=True)

		return residuo

	@staticmethod
	def dot2x2(weights,matrix,pol,chunk_data):
		weights = cp.array(weights)
		pol = cp.array(pol)
		N1,N2,n = matrix.shape
		sub_size = int(N1/chunk_data) + 1
		final_dot = np.zeros(shape=(N1,N2,1),dtype=np.complex128)
		for chunk1 in range(0,sub_size):
			for chunk2 in range(0,sub_size):
				if chunk1 + chunk2 < sub_size:
					sub_m = cp.array(matrix[chunk1*chunk_data:(chunk1+1)*chunk_data, chunk2*chunk_data:(chunk2+1)*chunk_data,:])
					N3,N4,n2 = sub_m.shape
					w = cp.ones(shape=(N3,N4,n2),dtype=float)*weights
					subsum = sub_m*w*cp.conjugate(pol)
					subsum = cp.sum(subsum,axis=2)
					subsum = cp.reshape(subsum,(N3,N4,1))
					final_dot[chunk1*chunk_data:(chunk1+1)*chunk_data, chunk2*chunk_data:(chunk2+1)*chunk_data,:] = cp.asnumpy(subsum)

		return final_dot
	
	def recurrence2d(self, z_target, z, weights, data, size, s, division_sigma, chunk_data, b):
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

		print("Polinomios inicializados.")

		# Normalización inicial de P y P_target
		P, P_target = self.normalize_initial_polynomials_cpu(z_target, z, w, P, P_target, V, s, chunk_data, b)

		print("Polinomios normalizados.")

		# Procedimiento Gram-Schmidt en los polinomios

		#final_data, residual, err, P_target, P = gram_schmidt_extrapolation.GramSchmidtExtrapolation(w, P, P_target, V, D, D_target,
		#																		 residual, final_data, err, s, sigma2, chunk_data=chunk_data,
		#																		 max_rep=2).gram_schmidt_and_estimation_gpu()
		
		final_data, residual, err, P_target, P = self.gram_schmidt_and_estimation_gpu(w, P, P_target, V, D, D_target,
																					  residual, final_data, err, s,
																					  sigma2,
																					  max_rep=2, chunk_data=chunk_data)
		# final_data, residual, err = gram_schmidt_and_estimation(w, P, P_target, V, D, D_target, residual, final_data, err, s, sigma2, max_rep=2, chunk_data=chunk_data)
		
		print("Hice G-S.")

		del w
		del D
		del D_target
		del z
		del z_target

		# Se libera la memoria utilizada por la GPU, para evitar un sobreconsumo de
		# esta.
		mempool = cp.get_default_memory_pool()
		mempool.free_all_blocks()

		return final_data, err, residual, P_target, P

