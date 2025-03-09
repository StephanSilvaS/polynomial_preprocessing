from matplotlib import pyplot as plt
import numpy as np

# Nombre de archivo a cargar
namefile_fulldata = "/disk2/stephan/gridded_visibility_model_natural_num_polynomial_19_division_sigma_0.0001_pixel_size_-3.88888888889e-09_image_size_513_513_HD100546.npz"

# Cargar archivo de entrada
full_data = np.load(namefile_fulldata)

visibilities = full_data["arr_0"]

print (full_data)

title = "Visibility model"; fig = plt.figure(title); plt.title(title); im = plt.imshow(np.log(np.absolute(visibilities) + 0.00001))
plt.colorbar(im)

plt.show()

