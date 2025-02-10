from astropy.io import fits
from pyralysis.io import DaskMS
from pyralysis.transformers.weighting_schemes import Uniform, Robust
from pyralysis.transformers import Gridder, DirtyMapper, UVTaper
import dask
import dask.array as da
from matplotlib import pyplot as plt
import numpy as np

class PreprocesamientoDatosAGrillar:
	def __init__(self, fits_path, ms_path):
		self.fits_path = fits_path
		self.ms_path = ms_path

	# Funcion que saca informaciopn del header de un archivo fits
	# Entrada: path del archivo fits
	# Salida: header, dimensiones de la imagen, info de pixeles
	def fits_header_info(self):
		"""Carga un archivo FITS y extrae su información."""
		try:
			with fits.open(self.fits_path) as fits_image:
				header = fits_image[0].header
				data = fits_image[0].data.squeeze()
				dimensions = [header.get('NAXIS1', None), header.get('NAXIS2', None)]
				du = header.get('BPA', None)
				dx = header.get('CDELT1', None)
				fits_image.close()
				return header, dimensions, data, du, dx
		except FileNotFoundError:
			raise FileNotFoundError(f"Archivo FITS no encontrado: {self.fits_path}")

	# Funcion que convierte un archivo Measurement Set (MS) a un archivo NumPy
	# Entrada: path del archivo MS
	# Salida: array numpy con los datos del archivo MS, visibilidades y pesos grillados
	def process_ms_file(self):
		"""
		Lee un archivo Measurement Set (MS), extrae la columna DATA y aplica gridding si es necesario.

		:param file_path: Ruta al archivo MS.
		:param gridder_config: Configuración opcional para el gridder. Por defecto usa la configuración estándar.
		:return: np.ndarray con los datos gridded.
		"""
		# Cargar el archivo MS
		ms_data = DaskMS(input_name=self.ms_path)
		dataset = ms_data.read(filter_flag_column=False, calculate_psf=False)

		l = dataset.field.phase_direction_cosines[0]
		m = dataset.field.phase_direction_cosines[1]

		# PROCESO DE GRIDDING
		padding_factor = 1.0
		hermitian_symmetry = False

		_, fits_dimensions, _, _, _ = self.fits_header_info()

		imsize = fits_dimensions[1]
		dx = dataset.theo_resolution / 7

		pb = dataset.antenna.primary_beam
		pb.cellsize = dx
		chans = dataset.spws.dataset[0].CHAN_FREQ.data.squeeze(axis=0)

		# Ploteamos el dirty beam
		centers_l = (l / (-dx) + imsize // 2).astype(np.int32)
		centers_m = (m / dx + imsize // 2).astype(np.int32)

		p_beams = da.array(
			[
				pb.beam(
					chans, (imsize, imsize), antenna=np.array([0]), imcenter=(centers_l[i], centers_m[i])
				) for i in range(centers_l.size)
			]
		)

		p_beams_together = da.sum(p_beams, axis=(0, 1, 2))
		plt.imshow(p_beams_together)
		plt.colorbar()
		plt.show()

		# Gridding

		gridder = Gridder(
			imsize=imsize,
			cellsize=dx,
			hermitian_symmetry=hermitian_symmetry,
			padding_factor=padding_factor
		)

		robust_param = 2.0
		robust = Robust(input_data=dataset, robust_parameter=robust_param, gridder=gridder)

		robust.apply()

		ckernel = None

		dirty_mapper = DirtyMapper(
			input_data=dataset,
			imsize=imsize,
			cellsize=dx,
			stokes=["I"],
			hermitian_symmetry=hermitian_symmetry,
			padding_factor=padding_factor,
			ckernel_object=ckernel
		)

		dirty_images_robust = dirty_mapper.transform()
		dirty_image, dirty_beam = dask.compute(
			*[dirty_images_robust[0].data[0], dirty_images_robust[1].data[0]]
		)

		plt.imshow(dirty_image, origin="lower", vmin=np.min(dirty_image), vmax=np.max(dirty_image))
		plt.colorbar()

		plt.show()
		gridded_visibilities, gridded_weights = da.compute(
			*[dirty_mapper.uvgridded_visibilities, dirty_mapper.uvgridded_weights]
		)

		# UV continuas
		for i, ms in enumerate(dataset.ms_list):
			uvw = ms.visibilities.uvw.data

		# UV grilladas
		m, n = dirty_image.shape
		du = 1 / (n * dx)
		dv = 1 / (m * dx)
		u = np.fft.fftshift(np.fft.fftfreq(n)) * n * du
		v = np.fft.fftshift(np.fft.fftfreq(m)) * m * dv

		return gridded_visibilities, gridded_weights, dx, uvw.compute(), u, v