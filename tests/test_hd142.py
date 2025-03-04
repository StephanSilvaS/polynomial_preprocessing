from polynomial_preprocessing import procesamiento_datos_continuos
from polynomial_preprocessing import procesamiento_datos_grillados
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos
from polynomial_preprocessing.optimization import optimizacion_parametros_grillados
import cupy as cp

print("Memoria de GPU: ", cp.cuda.Device(0).mem_info[1])

ejemplo_dc = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/dirty_images_natural_251.fits",
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/hd142_b9cont_self_tav.ms", 
    19, 
    0.0750780409680797,
    0.0007310213536)

dirty_image, pesos, visibilidades, _, _ = ejemplo_dc.data_processing()



"""
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
	[10, 21],
	[1e-3, 1e0],
	0.0007310213536,
	251)

ejemplo_opti_dg.initialize_optimization(3)
"""

"""
ejemplo_opti_dc = optimizacion_parametros_continuos.OptimizacionParametrosContinuos(
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/dirty_images_natural_251.fits",
	"/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/hd142_b9cont_self_tav.ms",
	[15, 30],
	[1e-8, 1e0],
	0.0007310213536,
	251)

ejemplo_opti_dc.initialize_optimization(100)
"""

