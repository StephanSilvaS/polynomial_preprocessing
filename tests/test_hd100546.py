from polynomial_preprocessing import procesamiento_datos_continuos, procesamiento_continuos_gpu, procesamiento_datos_grillados

ejemplo1 = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
	"/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD100546/hd100546_selfcal_cont_13_p513_cell_0005.fits",
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD100546/hd100546_selfcal_cont_13.ms", 
	19, 
    0.0001,
    -0.388888888889e-08)

dirty_image, vis, weights, _, _ = ejemplo1.data_processing()