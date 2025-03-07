from pyralysis.estimators import NearestNeighbor
from pyralysis.optimization.fi import Chi2
from pyralysis.optimization import ObjectiveFunction
from pyralysis.optimization.linesearch import Brent
from pyralysis.optimization.optimizer import HagerZhang
from pyralysis.io import DaskMS, FITS
from pyralysis.reconstruction import Image
from pyralysis.reconstruction.mask import Mask
import matplotlib.pyplot as plt
import astropy.units as u
import psutil
import dask.array as da
import numpy as np

n_workers = psutil.cpu_count(logical=False) - 1

# Load dataset and a FITS image (replace with your files)
ms_file = "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/hd142_b9cont_self_tav.ms"
fits_file = "/disk2/stephan/TesisAlgoritmoParalelo/datasets/HD142/hd142_b9cont_self_tav_p513_cell_0.01.fits"

#Cargo archivo MS
x = DaskMS(input_name = ms_file)
dataset = x.read(filter_flag_column=False, calculate_psf=False)

# Cargo archivo FITS
fits_io = FITS(input_name = fits_file)
image = fits_io.read()

# Calculo tamano de pixel
cellsize_ms = dataset.theo_resolution / 7

image = Image(data=image.data, cellsize = cellsize_ms, name="I")


chunks = (image.data.shape[0] // n_workers, image.data.shape[1] // n_workers)

print("image.data.shape:", image.data.shape)

image.data = image.data.chunk(chunks)



freq = [dataset.spws.min_nu.value]
pb = dataset.antenna.primary_beam
pointings = dataset.field.phase_direction_cosines[0:2]
pointings_cartesian = (pointings / image.cellsize).value
beams = da.array(
    [
        pb.beam(
            frequency=freq,
            imsize=image.data.shape,
            cellsize=image.cellsize,
            antenna=np.array([0]),  # Token antenna, for default mask
            x_0=pointings[0][i].value,
            y_0=pointings[1][i].value,
            imcenter=(image.center_pixel[0], image.center_pixel[1])
        ) for i in range(pointings.shape[-1])
    ]
)
beam = da.sum(beams, axis=(0, 1, 2))
normalized_beam = beam / da.max(beam)

dims = ["x", "y"]
inverse_beam = (1 / normalized_beam).compute()  # The same as squeeze

th = np.percentile(inverse_beam, 20)
print("threshold: ", th)

x_pix = np.arange(0, image.data.shape[0])
y_pix = np.arange(0, image.data.shape[0])
Z = np.zeros((image.data.shape[0], image.data.shape[1]))
# Creating 2-D grid of features
X, Y = np.meshgrid(x_pix, y_pix)
Z[inverse_beam <= th] = 1

plt.imshow(inverse_beam)
plt.colorbar()
plt.contour(X, Y, Z)
plt.title("Mask size")

plt.show()


# Create a mask (e.g., based on a threshold from the primary beam)

mask = Mask(dataset=dataset, imsize=image.data.shape, threshold=th, cellsize=image.cellsize)

# Choose a ModelVisibilities estimator – for example, NearestNeighbor is used here.
mv = NearestNeighbor(
    input_data=dataset,
    image=image,
    hermitian_symmetry=False,
    padding_factor=1.0
)
mv.transform()

# Define the objective function using a Chi² term (additional Fi terms like L1Norm or Entropy can be added)
fi_list = [Chi2(model_visibility=mv, normalize=True)]
of = ObjectiveFunction(fi_list=fi_list, image=image, persist_gradient=True)

# Set up line search (Brent) and the optimizer (Hager–Zhang)
ls = Brent(objective_function=of)
niter = 10
optim = HagerZhang(
    image=image,
    objective_function=of,
    linesearch=ls,
    mask=mask,
    max_iter=niter,
    projection=None  # Optionally, a projection function can be provided.
)

# Run the optimization
result = optim.optimize(verbose=True, partial_image=False)
result_image = result.data.compute()

# Visualize the reconstructed image
plt.imshow(result_image, origin="lower")
plt.title("Reconstructed Image")
plt.colorbar()
plt.show()