from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Diffusion Time Step Function using AB2',
    ext_modules = cythonize("ab2_diffusion_time_step.pyx"),
)

setup(
    name='Allen Cahn Time Step Function using AB2',
    ext_modules=cythonize("ab2_allen_cahn_timestep.pyx"),
)