from polynomial_preprocessing.extrapolation_process import procesamiento_datos_continuos, procesamiento_datos_grillados
from polynomial_preprocessing.preprocessing import preprocesamiento_datos_a_grillar
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos, optimizacion_parametros_grillados
from polynomial_preprocessing.image_reconstruction import conjugate_gradient
import numpy as np
from scipy.interpolate import griddata
from astropy.io import fits
from matplotlib import pyplot as plt
import astropy.units as unit

ejemplo_dc_hd142 = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
	fits_path = "/disk2/stephan/datasets/HD142/hd142_p251_cell_0.03.fits",
    ms_path = "/disk2/stephan/datasets/HD142/hd142_b9cont_self_tav.ms", 
	num_polynomial = 20, 
    division_sigma =  10**(-1),
    pixel_size= 0.0007310213536,
    n_iter_gc = 251,
    verbose = True,
    plots = True
)

ejemplo_dc_hd142.data_processing()