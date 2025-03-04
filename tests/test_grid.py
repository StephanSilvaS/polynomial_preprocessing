from polynomial_preprocessing import procesamiento_datos_continuos, procesamiento_datos_grillados
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos, optimizacion_parametros_grillados

ejemplo_dg = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
    "/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms", 
    8, 
    0.023107219110480887, 
    -2.5e-07, 
    129)

visibilidades_grilladas_doar25, pesos_grillados_doar25 = ejemplo_dg.data_processing()