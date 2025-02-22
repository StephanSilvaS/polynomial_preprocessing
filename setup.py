from setuptools import setup, find_packages

setup(
    name="polynomial_preprocessing",  # Nombre del paquete
    version="1.0.1",  # Versión inicial
    author="Stephan Silva Sanguinetti",
    author_email="stephan.silva@usach.cl",
    description="Pre-procesamiento de alto rendimiento basado en polinomios discretos ortogonales para síntesis de imágenes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/StephanSilvaS/polynomial_preprocessing.git",  # URL del repositorio GitHub
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Encuentra automáticamente todos los paquetes en el directorio
    install_requires=[
        "astropy==6.0.0",
        "dask==2024.8.2",
        "dafits==1.1.0",
        "xarray==2024.01.1",
        "zarr==2.18.2",
        "dask-ms==0.2.21",
        "distributed==2024.8.2",
        "more-itertools==10.2.0",
        "multimethod==1.11.2",
        "numba==0.59.1",
        "numpy<2.0.0",
        "python-casacore==3.6.1",
        "radio-beam==0.3.7",
        "scipy==1.13.1",
        "spectral-cube==0.6.5",
        "cupy==13.2.0",
		"optuna",
        "line_profiler"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux"
    ],
    python_requires=">=3.12"
)
