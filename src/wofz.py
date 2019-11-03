import ctypes
import numba
from numba.extending import get_cython_function_address
import numpy as np

import pyximport

pyximport.install()
import wofz_cython

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("wofz_cython", "wofz")
functype = ctypes.CFUNCTYPE(None, _dble, _dble, _ptr_dble, _ptr_dble)
wofz_fn = functype(addr)


@numba.njit
def wofz(z):
    w_real = np.empty(1, dtype=np.float64)
    w_imag = np.empty(1, dtype=np.float64)
    wofz_fn(np.real(z), np.imag(z), w_real.ctypes, w_imag.ctypes)
    return np.complex(w_real[0] + 1j * w_imag[0])
