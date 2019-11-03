"""
Polarized millimeter-wave atmospheric emission model
Copyright (c) 2019 Matthew Petroff

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import numba
import scipy.linalg

from constants import *

from wofz import wofz  # Cython module needed to use SciPy function with Numba


@numba.njit
def delta_DeltaM(N, B, M, DeltaM):
    """
    Calculates frequency shift of Zeeman components in GHz.

    Based on Larsson et al. (2019).

    Parameters
    ----------
    N : int
        Total rotational angular momentum quantum number, with sign matching change in rotational quantum number
    B : float
        Magnetic field strength (nT)
    M : int
        Magnetic quantum number
    DeltaM : int
        Change in magnetic quantum number

    Returns
    -------
    float
        Zeeman frequency shift (GHz)
    """
    assert np.abs(N) % 2 == 1
    assert np.abs(M) <= np.abs(N)
    assert np.abs(DeltaM) <= 1
    g_1 = ZEEMAN_COEFF[np.abs(N) - 1, 2]
    g_2 = ZEEMAN_COEFF[np.abs(N) - 1, 2 + np.sign(N)]
    return -MU_B / H * B * (g_1 * M + g_2 * (M + DeltaM)) * 1e-9 * 1e-9


@numba.njit("complex128[:,:](int64, float64)")
def rho_DeltaM(DeltaM, theta):
    """
    Returns transition matricies for Zeeman transitions.

    Parameters
    ----------
    DeltaM : int
        Change in magnetic quantum number
    theta : float
        Angle between line of sight and magnetic field direction (rad)

    Returns
    -------
    np.array
        Transition matrix
    """
    if DeltaM == -1:
        return np.array(
            [[1, 1j * np.cos(theta)], [-1j * np.cos(theta), np.cos(theta) ** 2]],
            dtype=np.complex128,
        )
    elif DeltaM == 1:
        return np.array(
            [[1, -1j * np.cos(theta)], [1j * np.cos(theta), np.cos(theta) ** 2]],
            dtype=np.complex128,
        )
    assert DeltaM == 0
    return np.array([[0j, 0j], [0j, np.sin(theta) ** 2]], dtype=np.complex128)


@numba.njit
def P_trans(N, M, DeltaJ, DeltaM):
    """
    Calculates transition probability.

    Parameters
    ----------
    N : int
        Total rotational angular momentum quantum number
    M : int
        Magnetic quantum number
    DeltaJ : int
        Change in rotational quantum number
    DeltaM : int
        Change in magnetic quantum number

    Returns
    -------
    float
        Transition probability
    """
    # fmt: off
    if DeltaJ == 1: # Liebe 1981
        if DeltaM == 1:
            return 3 * (N + M + 1) * (N + M + 2) / (4 * (N + 1) * (2*N + 1) * (2*N + 3))
        elif DeltaM == 0:
            return 3 * ((N + 1)**2 - M**2) / ((N + 1) * (2 * N + 1) * (2 * N + 3))
        elif DeltaM == -1:
            return 3 * (N - M + 1) * (N - M + 2) / (4 * (N + 1) * (2*N + 1) * (2*N + 3))
    elif DeltaJ == -1:
        if DeltaM == 1:
            return 3 * (N + 1) * (N - M) * (N - M - 1) / (4 * N * (2 * N + 1) * (2 * N**2 + N - 1))
        elif DeltaM == 0:
            return 3 * (N + 1) * (N**2 - M**2) / (N * (2 * N + 1) * (2 * N**2 + N - 1))
        elif DeltaM == -1:
            return 3 * (N + 1) * (N + M) * (N + M - 1) / (4 * N * (2 * N + 1) * (2 * N**2 + N - 1))
    # fmt: on


@numba.njit
def F(T, nu, nu_k, P, i, p_frac_h2o):
    """
    Calculates line profile intensity.

    Parameters
    ----------
    T : float
        Temperature of given atmosphere layer (K)
    nu : float
        Frequency to evaluate propagation at (GHz)
    nu_k : float
        Frequency of emission line (GHz)
    P : float
        Pressure (mbar)
    i : int
        Line number index.
    p_frac_h2o : float
        Pressure fraction of water vapor [0, 1]

    Returns
    -------
    float
        Line profile intensity (GHz)
    """
    p_frac_air = 1 - p_frac_h2o

    # Makarov et al. (2011)
    y_l = (O2_PARAMS[i, 1] + O2_PARAMS[i, 2] * (300 / T - 1)) * (
        300 / T
    ) ** 0.8  # bar^-1
    g_l = (O2_PARAMS[i, 3] + O2_PARAMS[i, 4] * (300 / T - 1)) * (
        300 / T
    ) ** 1.6  # bar^-2
    deltanu_l = (O2_PARAMS[i, 5] + O2_PARAMS[i, 6] * (300 / T - 1)) * (
        300 / T
    ) ** 1.6  # GHz bar^-2

    N = int(np.abs(O2_PARAMS[i, 0]))  # Line number (unitless)
    # M.A. Koshelev et al. (2016), Eq. (3) & Table 3
    gamma_air = 1.132150 + 1.348131 / (
        1 + 0.18844 * N - 0.007720 * N ** 2 + 1.660877e-5 * N ** 4
    )  # MHz / Torr
    # M.A. Koshelev et al. (2015), Eq. (1) & Table 2
    gamma_h2o = 1.140129 + 1.528118 / (
        1 + 0.10118 * N - 4.78115e-3 * N ** 2 + 7.40233e-6 * N ** 4
    )  # MHz / Torr
    # M.A. Koshelev et al. (2016), Eq. (1) & Table 1
    dnc = gamma_air / 1000 * TORR2MBAR * (P * p_frac_air) * (296 / T) ** 0.75412  # GHz
    # Reuse previous temperature dependence due to lack of better value
    dnc += gamma_h2o / 1000 * TORR2MBAR * (P * p_frac_h2o) * (296 / T) ** 0.75412  # GHz

    # Varghese & Hanson (1984), https://doi.org/10.1364/AO.23.002376
    # Herbert (1974), https://doi.org/10.1016/0022-4073(74)90021-1
    mol_mass = O2_MOL_MASS / MOL / 1e3  # kg
    doppler_half_width = (
        np.sqrt(2 * K_B * T / mol_mass) * nu_k / C
    )  # (1 / e) Doppler half width (GHz)

    # Combine Larsson (2014) method that adds line mixing with Melsheimer (2005) method that approximates VVW
    nu_prime = (
        nu - nu_k - deltanu_l * (P / 1000) ** 2
    ) / doppler_half_width  # Unitless
    a = dnc / doppler_half_width  # Unitless
    z1 = nu_prime + 1j * a  # Unitless
    nu_prime = (
        nu + nu_k + deltanu_l * (P / 1000) ** 2
    ) / doppler_half_width  # Unitless
    z2 = nu_prime + 1j * a  # Unitless
    return (
        (nu / nu_k) ** 2
        / (doppler_half_width * np.sqrt(np.pi))
        * (
            (1 + g_l * (P / 1000) ** 2 - 1j * y_l * (P / 1000)) * wofz(z1)
            + (1 + g_l * (P / 1000) ** 2 + 1j * y_l * (P / 1000)) * wofz(z2)
        )
    )  # GHz


@numba.jitclass(
    [
        ("B", numba.int32),
        ("atm_profile", numba.float64[:, ::1]),
        ("min_altitude", numba.float64),
        ("max_altitude", numba.float64),
        ("relative_humidity", numba.float64),
        ("layer_z", numba.float64),
    ]
)
class AtmSim(object):
    def __init__(
        self,
        B,
        atm_profile,
        min_altitude,
        max_altitude=100000,
        relative_humidity=0.1,
        layer_z=0.2,
    ):
        """
        Initializes atmospheric simulator.

        Parameters
        ----------
        B : int32
            Magnetic field strength (nT)
        atm_profile : float64[:, ::1]
            Atmosphere profile [Altitude (m), Temperature (K), Pressure (mbar), Temperature stddev (K), Pressure stddev (mbar)]
        min_altitude : float64
            Minimum simulation altitude (m)
        max_altitude : float64
            Maximum simulation altitude (m)
        relative_humidity : float64
            Relative humidity fraction [0, 1]
        layer_z : float64
            Height of atmosphere layers to simulate (km)
        """
        self.B = B
        self.atm_profile = atm_profile
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.relative_humidity = relative_humidity
        self.layer_z = layer_z

    def calc_propagation_matrix(self, nu, P, T, theta):
        """
        Calculates propagation matrix.

        Parameters
        ----------
        nu : float
            Frequency to evaluate propagation at (GHz)
        P : float
            Pressure (mbar)
        T : float
            Temperature of given atmosphere layer (K)
        theta : float
            Angle between line of sight and magnetic field direction (radian)

        Returns
        -------
        np.array
            Propagation matrix (Np km^-1)
        """
        # Pressure fraction of water vapor
        # Liebe (1993)
        p_frac_h2o = (
            self.relative_humidity
            * 2.408e11
            * (300 / T) ** 5
            * np.exp(-22.644 * (300 / T))
            / P
        )
        p_frac_air = 1 - p_frac_h2o

        # Initialize attenuation matrix to zero
        total = np.zeros((2, 2), dtype=np.complex128)

        # Non-resonant continuum
        # Tretyakov (2016); eq. 14 & Table 5
        c_w = 7.82e-6  # dB km^-1 GHz^-2 kPa^-2
        x_w = 7.5  # Unitless
        c_air = 2.36e-7  # dB km^-1 GHz^-2 kPa^-2
        x_air = 3  # Unitless
        c_dry = 3.18e-11  # dB km^-1 GHz^-2 kPa^-2
        x_dry = 3.35  # Unitless
        alpha_con = (
            (
                c_w * (300 / T) ** x_w * ((P / 10) * p_frac_h2o) ** 2
                + c_air
                * (300 / T) ** x_air
                * ((P / 10) * p_frac_air)
                * ((P / 10) * p_frac_h2o)
                + c_dry * (300 / T) ** x_dry * ((P / 10) * p_frac_air) ** 2
            )
            * nu ** 2
            / (10 * np.log10(np.e))
        )  # Np / km
        total[0, 0] += alpha_con / 2  # Power attenuation to field attenuation
        total[1, 1] += alpha_con / 2

        # Water line intensities
        # HITRAN (2019-03-18)
        h2o_freq = H2O_HITRAN.T[0] * C * 1e2 * 1e-9  # GHz
        h2o_ints = H2O_HITRAN.T[1] * C * 1e2 * 1e-6 * 1e-4  # MHz / (molecule m^-2)
        h2o_ints *= MOL  # MHz m^2 mol^-1
        h2o_ints *= (P * 1e2) * p_frac_h2o / (R * T)  # GHz km^-1
        E_low = H2O_HITRAN.T[2] * C * 1e-2 * H  # J
        h2o_ints *= (296 / T) ** 2.5 * np.exp(
            E_low / (K_B * 296) * (1 - 296 / T)
        )  # GHz km^-1

        # Water lines
        # Rosenkranz (1998), Liebe (1993)
        v_cutoff = 750  # GHz
        for i in range(H2O_PARAMS.shape[0]):
            v_l = h2o_freq[i]  # GHz
            w_f = H2O_PARAMS[i, 1]  # GHz / kPa
            x_f = H2O_PARAMS[i, 2]
            w_s = H2O_PARAMS[i, 3]  # GHz / kPa
            x_s = H2O_PARAMS[i, 4]
            gamma_h2o = (  # Rosenkranz (1998)
                w_s * (P / 10) * p_frac_h2o * (300 / T) ** x_s
                + w_f * (P / 10) * p_frac_air * (300 / T) ** x_f
            )  # GHz
            fpp_h2o = (
                nu ** 2
                * gamma_h2o
                / (np.pi * v_l ** 2)
                * (
                    1 / ((nu - v_l) ** 2 + gamma_h2o ** 2)
                    + 1 / ((nu + v_l) ** 2 + gamma_h2o ** 2)
                    - 2 / (v_cutoff ** 2 + gamma_h2o ** 2)
                )
            )  # GHz^-1
            alpha_h2o = h2o_ints[i] * fpp_h2o  # Np km^-1
            total[0, 0] += alpha_h2o / 2
            total[1, 1] += alpha_h2o / 2

        # Oxygen line intensities
        # HITRAN (2019-03-18)
        o2_freq = O2_HITRAN.T[0] * C * 1e2 * 1e-9  # GHz
        o2_ints = O2_HITRAN.T[1] * C * 1e2 * 1e-6 * 1e-4  # MHz / (molecule m^-2)
        o2_ints *= MOL  # MHz m^2 mol^-1
        o2_ints *= (P * 1e2) * p_frac_air / (R * T) * O2_VOL_FRAC  # GHz km^-1
        E_low = O2_HITRAN.T[2] * C * 1e-2 * H  # J
        o2_ints *= (296 / T) ** 2.5 * np.exp(
            E_low / (K_B * 296) * (1 - 296 / T)
        )  # GHz km^-1

        # Oxygen lines
        for i in range(O2_PARAMS.shape[0]):
            N = int(O2_PARAMS[i][0])
            nu_t = o2_freq[i]  # GHz
            tmp0 = np.zeros((2, 2), dtype=np.complex128)
            DeltaJ = np.sign(N)
            for DeltaM in [-1, 0, 1]:
                tmp1 = 0
                Ms = range(-np.abs(N), np.abs(N) + 1)
                for M in Ms:
                    nu_k = nu_t + delta_DeltaM(N, self.B, M, DeltaM)  # GHz
                    line_profile = F(T, nu, nu_k, P, i, p_frac_h2o)  # GHz
                    tmp1 += (
                        P_trans(np.abs(N), M, DeltaJ, DeltaM) * line_profile
                    )  # Np km^-1
                tmp0 += rho_DeltaM(DeltaM, theta) * tmp1
            total += tmp0 * o2_ints[i] / 2
        return total

    def calc_propagation_matrix_avg(self, freq, theta, zenith_angle):
        """
        Calculates full line-of-sight propagation matrix.

        Parameters
        ----------
        freq : float
            Frequency to evaluate propagation at (GHz)
        theta : float
            Angle between line of sight and magnetic field direction (rad)
        zenith_angle : float
            Zenith angle (rad), [0, pi/2)

        Returns
        -------
        np.array
            Propagation matrix
        """
        delta_z = self.layer_z / np.cos(zenith_angle)  # km

        # Start off with CMB brightness temperature, since we're looking toward space
        x0 = H * freq * 1e9 / K_B / T_CMB
        cmb_rj = x0 ** 2 * np.exp(x0) / np.expm1(x0) ** 2 * T_CMB
        total = np.identity(2, np.complex128) * cmb_rj

        altitude = self.max_altitude
        while altitude >= self.min_altitude:
            Temp = np.interp(altitude, self.atm_profile.T[0], self.atm_profile.T[1])
            Pres = np.interp(altitude, self.atm_profile.T[0], self.atm_profile.T[2])

            G_i = self.calc_propagation_matrix(freq, Pres, Temp, theta) * delta_z

            with numba.objmode(e_G="complex128[::1,:]", e_G_ct="complex128[::1,:]"):
                e_G = scipy.linalg.expm(-G_i)
                e_G_ct = scipy.linalg.expm(-G_i.conj().T)
            total = e_G @ total @ e_G_ct + Temp * (
                np.identity(2, np.complex128) - e_G @ e_G_ct
            )

            altitude -= self.layer_z * 1e3
        return total

    def calc_propagation_matrix_avg_band(
        self, freq_start, freq_stop, steps, theta, zenith_angle
    ):
        """
        Calculates band-averaged Stokes parameters.

        Parameters
        ----------
        freq_start : float
            Frequency of low end of band (GHz)
        freq_stop : float
            Frequency of high end of band (GHz)
        steps : int
            Number of frequency points to evaluate propagation matrix at
        theta : float
            Angle between line of sight and magnetic field direction (rad)
        zenith_angle : float
            Zenith angle (rad), [0, pi/2)

        Returns
        -------
        float[4]
            [Stokes I (K), Stokes Q (K), Stokes U (K), Stokes V (K)]
        """
        out = np.zeros((steps, 4))
        for i, freq in enumerate(np.linspace(freq_start, freq_stop, steps)):
            mat = self.calc_propagation_matrix_avg(freq, theta, zenith_angle)
            out[i] = to_stokes(mat)
        # As of v0.45, Numba doesn't support axis argument for np.mean
        I = np.mean(out[:, 0])
        Q = np.mean(out[:, 1])
        U = np.mean(out[:, 2])
        V = np.mean(out[:, 3])
        return np.array([I, Q, U, V])


# From Principles of Optics, Born & Wolf (1959)
# Page 550 in 1st ed., page 554 in 3rd & 4th ed., and page 630 in 7th ed.
# Modified with extra factor of 0.5 for brightness temperature


@numba.njit
def to_I(mat):
    return 0.5 * np.real(mat[0, 0] + mat[1, 1])


@numba.njit
def to_Q(mat):
    return 0.5 * np.real(mat[0, 0] - mat[1, 1])


@numba.njit
def to_U(mat):
    return 0.5 * np.real(mat[0, 1] + mat[1, 0])


@numba.njit
def to_V(mat):
    return 0.5 * np.real(1j * (mat[1, 0] - mat[0, 1]))


# From Ellipsometry and Polarized Light, Azzam & Bashara (1977)
# Page 62
# Modified with extra factor of 0.5 for brightness temperature

STOKES_MATRIX = np.array(
    [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -1j, 1j, 0]], dtype=np.complex128
)


@numba.njit
def to_stokes(mat):
    return 0.5 * np.real(STOKES_MATRIX @ mat.reshape((4, 1))).T
