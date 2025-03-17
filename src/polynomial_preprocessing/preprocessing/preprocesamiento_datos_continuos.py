from astropy.io import fits
from pyralysis.io import DaskMS

class PreprocesamientoDatosContinuos:
	def __init__(self, fits_path, ms_path):
		self.fits_path = fits_path
		self.ms_path = ms_path

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

	def process_ms_file(self):
		"""Carga un archivo MS y extrae las visibilidades y pesos."""

		ms_data = DaskMS(input_name=self.ms_path)
		dataset = ms_data.read(filter_flag_column=False, calculate_psf=False)

		dx = dataset.theo_resolution / 7

		print("Resolución teórica de MS: ", dx)

		# UV continuas
		for i, ms in enumerate(dataset.ms_list):
			uvw = ms.visibilities.uvw.data
			visibilities_data = ms.visibilities.data
			weights_data = ms.visibilities.weight.data

		return uvw.compute(), visibilities_data.compute().values, weights_data.compute(), dx
