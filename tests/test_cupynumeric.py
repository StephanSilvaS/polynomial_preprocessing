from polynomial_preprocessing import procesamiento_datos_continuos, procesamiento_datos_grillados, preprocesamiento_datos_a_grillar, procesamiento_datos_grillados_cupynumeric
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos, optimizacion_parametros_grillados
from polynomial_preprocessing.image_reconstruction import conjugate_gradient
import numpy as np
from scipy.interpolate import griddata
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.io import fits

ejemplo_dg_as205 = procesamiento_datos_grillados_cupynumeric.ProcesamientoDatosGrilladosCupyNumeric(
	fits_path = "/disk2/stephan/TesisAlgoritmoParalelo/datasets/AS205/AS205_p313_cell_0.006.fits",
    ms_path = "/disk2/stephan/TesisAlgoritmoParalelo/datasets/AS205/AS205_continuum_model.ms", 
	num_polynomial = 40, 
    division_sigma = 0.046207724495770736,
    verbose = True,
    plots = False
)

gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v = ejemplo_dg_as205.grid_data()

dirty_image, vis, weights, _, _ = ejemplo_dg_as205.gridded_data_processing(gridded_visibilities, gridded_weights, pixel_size, grid_u, grid_v)

"""
def norm(weights,x):
    return(np.absolute(np.sqrt(np.sum(weights*np.absolute(x)**2))))

gc_image_2 = conjugate_gradient.ConjugateGradient(vis, weights/norm(weights.flatten(), vis.flatten()), 10)

gc_image_data = gc_image_2.CG()

visibility_model = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_image_data)))

gc_image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model)))


print("MAX FINAL:",np.max(gc_image_model))

print("MIN FINAL:",np.min(visibility_model))


title="Extrapolacion hd142 + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(gc_image_model))
plt.colorbar(im)

title="Visibility model + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(visibility_model))
plt.colorbar(im)


plt.show()

"""
