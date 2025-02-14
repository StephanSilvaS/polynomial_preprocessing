from polynomial_preprocessing import procesamiento_datos_continuos
from polynomial_preprocessing.image_synthesis import gradiente_conjugado_no_lineal
import numpy as np
from matplotlib import pyplot as plt


ejemplo1 = procesamiento_datos_continuos.ProcesamientoDatosContinuos(
	"/home/stephan/polynomial_preprocessing/datasets/HD142/dirty_images_natural_251.fits",
    "/home/stephan/polynomial_preprocessing/datasets/HD142/hd142_b9cont_self_tav.ms", 
	11, 
	0.014849768613424696, 
	0.0007310213536, 
	251)

image_model, weights_model = ejemplo1.data_processing()

gc_image = gradiente_conjugado_no_lineal.GradienteConjugadoNoLineal(image_model, weights_model, 251)

print("####### TASK DONE 1 #######")

gc_image_data = gc_image.conjugate_gradient()

print("####### TASK DONE 2 #######")

visibility_model = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gc_image_data)))

gc_image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibility_model)))

print(gc_image_model)

title="Image model + NCG"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(gc_image_model))
plt.colorbar(im)

plt.show()

print(gc_image_model.shape)

print("####### TASK DONE 3 #######")
