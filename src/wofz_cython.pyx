# Based on https://github.com/numba/numba/issues/3086#issuecomment-472046807

cimport scipy.special.cython_special
cimport numpy as np

cdef api wofz(double z_real, double z_imag, double *w_real, double *w_imag):
  cdef double complex z
  z.real = z_real
  z.imag = z_imag
  
  cdef double complex w = scipy.special.cython_special.wofz(z)
  
  w_real[0] = w.real
  w_imag[0] = w.imag
