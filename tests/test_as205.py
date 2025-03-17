from polynomial_preprocessing import procesamiento_datos_grillados


ejemplo_dg_as205 = procesamiento_datos_grillados.ProcesamientoDatosGrillados(
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/AS205/AS205_p313_cell_0.006.fits",
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/AS205/AS205_continuum_model.ms", 
    num_polynomial=20, 
    division_sigma=0.023107219110480887, 
    verbose=True)

dirty_image_as205, vis_gridded_as205, weights_gridded_as205, _, _ = ejemplo_dg_as205.data_processing()