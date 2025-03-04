from polynomial_preprocessing import procesamiento_datos_continuos, procesamiento_continuos_gpu
from polynomial_preprocessing.optimization import optimizacion_parametros_continuos
import matplotlib.pyplot as plt


"""
ejemplo1 = procesamiento_continuos_gpu.ProcesamientoDatosContinuosGPU(
	"/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/dirty_images_natural_251.fits",
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/hd142_b9cont_self_tav.ms", 
	19, 
    0.0750780409680797,
    0.0007310213536, 
    251)

dirty_image, vis, weights, _, _ = ejemplo1.data_processing()

namefile_vis = "Desarrollos/version_GPU/hd142_b9cont_self_tav.npz"
namefile_weights = "Desarrollos/version_GPU/hd142_b9cont_self_tav.npz"

title = "Visibility model (division sigma: " + str() + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.absolute(vis))
plt.colorbar(im)

title = "Weights model (division sigma: " + str() + ")"; fig = plt.figure(title); plt.title(title); im = plt.imshow(weights_model)
plt.colorbar(im)
"""

import cupy as cp
print("Número de GPUs disponibles:", cp.cuda.runtime.getDeviceCount())

ejemplo1 = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
	"/disk2/stephan/TesisAlgoritmoParalelo/datasets/GWLup/GWLup_p0.01_n513.fits",
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/GWLup/GWLup_continuum.ms", 
	25, 
    10**(-1),
    -1.7e-05)

dirty_image, vis, weights, _, _ = ejemplo1.data_processing()


"""
ejemplo_opti_dc = optimizacion_parametros_continuos.OptimizacionParametrosContinuos(
	"/disk2/stephan/TesisAlgoritmoParalelo/datasets/GWLup/GWLup_p0.01_n513.fits",
    "/disk2/stephan/TesisAlgoritmoParalelo/datasets/GWLup/GWLup_continuum.ms", 
	[10, 45], 
    [10**(-4), 10**(-1)],
    -1.7e-05,
    251)

ejemplo_opti_dc.initialize_optimization(5)

"""