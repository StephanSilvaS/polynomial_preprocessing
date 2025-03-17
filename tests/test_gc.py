from polynomial_preprocessing.extrapolation_process import procesamiento_datos_continuos
from polynomial_preprocessing.image_reconstruction import gradiente_conjugado_no_lineal, conjugate_gradient
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

def norm(weights,x):
    return(np.absolute(np.sqrt(np.sum(weights*np.absolute(x)**2))))

"""
ejemplo1 = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
	"/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms", 
	11, 
    0.014849768613424696, 
    0.0007310213536, 
    251)
"""

ejemplo1 = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
	"/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms", 
	19, 
    0.750780409680797,
    0.0007310213536, 
    251, 
    verbose=False)

image_model, weights_model, visibilities_model, _, _ = ejemplo1.data_processing()

print(image_model.shape)

N = 30

gc_image_1 = conjugate_gradient.ConjugateGradient(visibilities_model, 
                                                  weights_model/norm(weights_model.flatten(), visibilities_model.flatten()), 
                                                  N)

gc_image_data = gc_image_1.CG()


print("####### TASK DONE 1 #######")

#gc_image_data = gc_image.compute_gradient()

print(gc_image_data)

print("####### TASK DONE 2 #######")

visibility_model = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_image_data)))

gc_image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model)))


print("MAX FINAL:",np.max(gc_image_model))

print("MIN FINAL:",np.min(visibility_model))

print(gc_image_model.shape)

fits_image = fits.open("/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits")
header = fits_image[0].header
fits.writeto("/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251_TESTING.fits", gc_image_model.real, header,overwrite=True)


title="Image model + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(gc_image_model))
plt.colorbar(im)

title="Visibility model + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(visibility_model))
plt.colorbar(im)


plt.show()


print(gc_image_model.shape)

print("####### TASK DONE 3 #######")
