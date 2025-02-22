from polynomial_preprocessing import procesamiento_datos_continuos, procesamiento_datos_grillados
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos, optimizacion_parametros_grillados

ejemplo_dg_2 = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
    "/home/stephan/polynomial_preprocessing/datasets/DoAr25/imagen_recortada.fits",
    "/home/stephan/polynomial_preprocessing/datasets/DoAr25/DoAr25_continuum.ms", 
    10, 
    0.023107219110480887, 
    -4e-08, 
    2082)

visibilidades_grilladas_doar25, pesos_grillados_doar25 = ejemplo_dg_2.data_processing()