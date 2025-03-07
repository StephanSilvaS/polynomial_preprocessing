from polynomial_preprocessing import procesamiento_datos_continuos
from polynomial_preprocessing import procesamiento_datos_grillados
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos
from polynomial_preprocessing.optimization import optimizacion_parametros_grillados
"""


ejemplo_dc = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
    "/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms", 
    11, 
    0.014849768613424696, 
    0.0007310213536, 
    251)

visibilidades_extrapoladas, pesos = ejemplo_dc.data_processing()



ejemplo_dg = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
    "/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms", 
    11, 
    0.014849768613424696, 
    0.0007310213536, 
    251)

visibilidades_grilladas, pesos = ejemplo_dg.data_processing()

"""
"""

ejemplo_opti_dg = optimizacion_parametros_grillados.OptimizacionParametrosGrillados(
    "/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
	"/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms",
	[5, 21],
	[1e-3, 1e0],
	0.0007310213536,
	251)

"""
ejemplo_opti_dc = optimizacion_parametros_continuos.OptimizacionParametrosContinuos(
	"/home/stephan/polynomial_preprocessing/datasets/GWLup/GWLup_p0.01_n513.fits",
    "/home/stephan/polynomial_preprocessing/datasets/GWLup/GWLup_continuum.ms", 
	[10, 20], 
    [10**(-4), 10**(-1)],
    -1.7777777e-05,
    251)

ejemplo_opti_dc.initialize_optimization(30)



"""
ejemplo_opti_dc = optimizacion_parametros_continuos.OptimizacionParametrosContinuos(
    "/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
	"/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms",
	[15, 25],
	[1e-3, 1e0],
	0.0007310213536,
	251)

ejemplo_opti_dc.initialize_optimization(2)
"""
