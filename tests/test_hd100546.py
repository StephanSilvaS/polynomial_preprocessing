from polynomial_preprocessing.extrapolation_process import procesamiento_datos_continuos, procesamiento_datos_grillados

hd100546_gridded = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
	fits_path="/disk2/stephan/datasets/HD100546/hd100546_p251_cell_0.008.fits",
    ms_path="/disk2/stephan/datasets/HD100546/hd100546_selfcal_cont_13.ms", 
	num_polynomial=80, 
    division_sigma=0.664854673262534,
    verbose=True)

dirty_image_hd100546, vis_hd100546, weights_hd100546, _, _, reconstructed_image_hd100546, visibility_recons_ = hd100546_gridded.data_processing()

"""

ejemplo_2 = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
    "/home/stephan/polynomial_preprocessing/datasets/HD100546/hd100546_selfcal_cont_13_p513_cell_0005.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD100546/hd100546_selfcal_cont_13.ms", 
    11, 
    0.0001,
    -0.3888e-04)

dirty_image, pesos, visibilidades, _, _ = ejemplo_2.data_processing()"
"""