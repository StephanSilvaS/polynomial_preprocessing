from polynomial_preprocessing import procesamiento_datos_continuos, procesamiento_datos_grillados
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos, optimizacion_parametros_grillados
from polynomial_preprocessing.image_synthesis import conjugate_gradient
import numpy as np
from scipy.interpolate import griddata
from astropy.io import fits
from matplotlib import pyplot as plt

ejemplo1 = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
	"/home/stephan/polynomial_preprocessing/datasets/GWLup/GWLup_p0.01_n513.fits",
    "/home/stephan/polynomial_preprocessing/datasets/GWLup/GWLup_continuum.ms", 
	11, 
    10**(-4),
    0.9997e-05,
    65,
    verbose = False
)

dirty_image, vis, weights, _, _ = ejemplo1.data_processing()