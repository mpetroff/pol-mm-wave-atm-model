# Polarized millimeter-wave atmospheric emission model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3526651.svg)](https://doi.org/10.5281/zenodo.3526651)

This code serves as a supplement to the paper titled _Two-year Cosmology Large Angular Scale Surveyor (CLASS) Observations: A First Detection of Atmospheric Circular Polarization at Q Band_. It simulates millimeter-wave atmospheric emission, including polarization. This emission is primarily from molecular oxygen, with emission by water vapor and the dry continuum also considered.

The `src` directory contains the simulation source code, with the `simulation.py` script containing the simulation logic. The `constants.py` file defines constants used in the simulation, and the `wofz.py` and `wofz_cython.pyx` files provide a Numba-compatible interface to SciPy's `wofz` function.

The `input-data` directory contains tabular data copied from HITRAN and various publications. It also contains an atmosphere profile generated using the `notebooks/calc-nrlmsise00-profile-cerro-toco.ipynb` notebook.

The `notebooks` directory contains examples for running the code to produce the plots and results shown in the aforementioned paper.

Finally, the `result-data` directory contains plots and data files generated by the previously mentioned notebooks.

Testing was performed using [Python](https://www.python.org/) 3.7.4, [Numba](https://numba.pydata.org/) 0.45.1, [NumPy](https://numpy.org/) 1.17.2, [SciPy](https://scipy.org/scipylib/) 1.3.1, [Cython](https://cython.org/) 0.29.13, [Matplotlib](https://matplotlib.org/) 3.1.1, and [MSISE00](https://github.com/space-physics/msise00) 1.6.1.


## License

This work is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `COPYING` file for details.


## Credits

This code was written by Matthew A. Petroff ([ORCID:0000-0002-4436-4215](https://orcid.org/0000-0002-4436-4215)).

If using this work in the course of academic research, please cite Petroff et al. (2019), _Two-year Cosmology Large Angular Scale Surveyor (CLASS) Observations: A First Detection of Atmospheric Circular Polarization at Q Band_.
