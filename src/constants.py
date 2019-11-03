from pathlib import Path
import numpy as np
import astropy.constants
import astropy.units as u

BASE_PATH = (Path(__file__).parent / "..").absolute().resolve()

K_B = astropy.constants.k_B.to("J/K").value
C = astropy.constants.c.to("m/s").value
H = astropy.constants.h.to("J s").value
MU_B = astropy.constants.muB.to("J/T").value
R = astropy.constants.R.to("J/(K mol)").value
MOL = astropy.constants.N_A.value

TORR2MBAR = 760 / 1013.25

T_CMB = 2.72548  # (K), from Fixsen (2009)

O2_VOL_FRAC = 0.2095  # O2 volume fraction, from Machta & Hughes (1970)
O2_MOL_MASS = (
    31.998
)  # (g mol^-1), from weighted average of O2 isotopologues in https://hitran.org/media/molparam.txt

# Load data tables
ZEEMAN_COEFF = np.loadtxt(BASE_PATH / "input-data/zeeman_coeff.txt")
O2_PARAMS = np.loadtxt(BASE_PATH / "input-data/o2_params.txt")
O2_HITRAN = np.loadtxt(BASE_PATH / "input-data/o2_hitran.txt")
H2O_PARAMS = np.loadtxt(BASE_PATH / "input-data/h2o_params.txt")
H2O_HITRAN = np.loadtxt(BASE_PATH / "input-data/h2o_hitran.txt")
