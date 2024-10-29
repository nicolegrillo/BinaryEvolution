import numpy as np
import scipy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar
from scipy.interpolate import interp1d

from analysis import get_match_pads, get_m_1, get_m_2, get_r_isco, get_f_isco

import pickle
from typing import Tuple

import dynesty
from dynesty import plotting as dyplot

from tqdm.auto import trange
import corner
from scipy.interpolate import griddata
from scipy.stats import scoreatpercentile

import environments_handy_functions

# Reload libraries (in case you change something)
import importlib
importlib.reload(environments_handy_functions)

from environments_handy_functions import (
    df_dt,  
    find_grid, 
    time_to_coal_cumul,  
    phase_f_cumul, 
    f_1yr, 
    h_0, 
    mycalculate_SNR, 
    amplitude, 
    Psi,
    myVacuumBinary, myAccretionDisk, myDarkMatter, myCombination)


G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
YR = 365.25 * 24 * 3600  # s

# Set detector
detector = "LISA"

# Set PSDs, choose observation time and SNR threshold (will set distance in signal system below)
from analysis import S_n_LISA as S_n, f_range_LISA as f_range_n  

T_OBS = 1 * YR #seconds
SNR_THRESH = 100.0
TITLE = "LISA"

def myget_signal_system() -> Tuple[myVacuumBinary, myCombination, Tuple[float, float]]:
    """
    Creates an accretion disk with SNR and duration as set above for given detector.
    
    """
    m1 = 1e5 * MSUN # kg
    m2 = 10 * MSUN # kg
    
    rho6 = 1.17e17 * MSUN / PC**3
    r6 =  PC / 1e6
    gammas = 7/3
    epsv = 0.58
    
    r_s = 2 * G * m1/ C**2 # Schwartzschild radius of m1
    r0 = 3 * r_s
    Mach = 100 
    sigma0 = 1.5e10 / Mach**2
    alpha = -1/2
    
    m_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    _VB = myVacuumBinary(
    m1=m1,
    m2=m2,
    dist=100e6 * PC, 
    chirp_mass=m_chirp)
    
    q = _VB.q
    logL = np.log10(1 / q**(1/2))
    
    TT_C = 0.0 # time of coalescence
    F_C = _VB.frequency(6 * G * m1 / C**2)
    
    # m1, m2, dist, mach, sigma0, alpha, r0, chirp_mass, q, gammas, rho6, r6, epsv
    
    _COMBO = myCombination(m1=m1, 
                        m2=m2, 
                        dist=100e6 * PC, 
                        mach=Mach, 
                        sigma0=sigma0, 
                        alpha=alpha,
                        r0=r0,
                        q=q, 
                        gammas=gammas, 
                        rho6=rho6, 
                        r6=r6, 
                        epsv=epsv, 
                        chirp_mass=m_chirp)

    # Frequency range and grids
    F_RANGE_D = find_grid(_COMBO, T_OBS)
    FS = np.linspace(max(F_RANGE_D[0], f_range_n[0]), min(F_RANGE_D[1], f_range_n[1]), 50_000)

    # Get dL such that SNR is as set above
    _fn = lambda dL: mycalculate_SNR(myCombination(m1=m1, 
                                                  m2=m2, 
                                                  dist=dL, 
                                                  mach=Mach, 
                                                  sigma0=sigma0, 
                                                  alpha=alpha,
                                                  r0=r0,
                                                  q=q, 
                                                  gammas=gammas, 
                                                  rho6=rho6, 
                                                  r6=r6, 
                                                  epsv=epsv, 
                                                  chirp_mass=m_chirp), FS[::-1], S_n)
    
    res = root_scalar(lambda dL: (_fn(dL) - SNR_THRESH), bracket=(0.1e6 * PC, 100000e6 * PC))
    assert res.converged
    DL = res.root
    
    # Redefine DM and VB with "new" distance
    
    _COMBO_new = myCombination(m1=m1, 
                        m2=m2, 
                        dist=DL, 
                        mach=Mach, 
                        sigma0=sigma0,
                        alpha = alpha,
                        r0=r0,
                        q=q, 
                        gammas=gammas, 
                        rho6=rho6, 
                        r6=r6, 
                        epsv=epsv, 
                        chirp_mass=m_chirp)
    
    _VB_new = myVacuumBinary(
    m1=m1,
    m2=m2,
    dist=DL, 
    chirp_mass=m_chirp)

    return _VB_new, _COMBO_new, F_RANGE_D

_VB, _COMBO, F_RANGE_D = myget_signal_system()

FS = np.linspace(F_RANGE_D[-1], F_RANGE_D[0], 3000)  # coarser grid

PAD_LOW, PAD_HIGH = get_match_pads(FS[::-1])  # padding for likelihood calculation

def get_frequency_noise(psd, fs):
    
    delta_f = fs[0] - fs[1]
    sigma = np.sqrt(psd(fs)/(4 * delta_f))
    not_zero = (sigma != 0)
    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_co = np.random.normal(0, sigma_red)

    noise_red = (1/np.sqrt(2)) * (noise_re + 1j * noise_co)

    noise = np.zeros(len(sigma), dtype=complex)
    noise[not_zero] = noise_red

    return noise

def waveform(params_h, fs, S_n):
    flen = len(fs)
    delta_f = fs[0] - fs[1]
    
    wf_h = amplitude(fs, params_h) * np.exp(1j * Psi(fs, params_h, TTC=0.0, PHI_C=0.0))
    noise = get_frequency_noise(S_n, fs)
    
    wf_h_noise = wf_h + noise
    
    return wf_h, noise, wf_h_noise

# Define the signal: 

signal = waveform(_COMBO, FS, S_n)[0] # 0: noiseless, 1: noise-only, 2: noisy

def calculate_match_unnormd_fft(
    params_h, params_d, fs, pad_low, pad_high, S_n=S_n
):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value
    and t_c using the fast Fourier transform.
    """
    df = fs[0] - fs[1] # I have set out a reversed grid wr to the pydd version
    wf_h = amplitude(fs, params_h) * np.exp(1j * Psi(fs, params_h, TTC=0.0, PHI_C=0.0)) # h is the model/template
    wf_d = signal # d is the signal including noise
    Sns = S_n(fs)

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand = 4 * wf_h.conj() * wf_d / Sns * df 
    integrand_padded = np.concatenate((pad_low, integrand, pad_high))
    
    return np.abs(len(integrand_padded) * np.fft.ifft(integrand_padded)).max()

def loglikelihood_fft(
    params_h, params_d, fs, pad_low, pad_high, S_n=S_n
):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary, Phi_c and t_c (i.e., all
    extrinsic parameters).
    """
    # Waveform magnitude
    ip_hh = mycalculate_SNR(params_h, fs, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd_fft(params_h, params_d, fs, pad_low, pad_high, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh

# symmetric priors around true value

true_value = np.array([_COMBO.Binary_init.chirp_mass, _COMBO.DarkMatter_init.rho6, _COMBO.DarkMatter_init.gammas, _COMBO.Binary_init.q, _COMBO.Accretion_init.alpha, _COMBO.Accretion_init.sigma0])

# set smaller prior range

def ptform(u: np.ndarray) -> np.ndarray:
    """
    Maps [0, 1] to deviations away (in log space) from true values. () brackets have flat prior choices.
    """
    assert u.shape == (4,)
    
    m_chirp = np.array([2 * (0.5) * (u[0]-0.5)])
    rho6 = np.array([np.log10(true_value[1]) - 0.001 + 2 * (0.001) * u[1]])
    gamma = np.array([2 * (0.1) * (u[2]-0.5)])
    q = np.array([np.log10(true_value[3]) - 0.05 + 2 * (0.05) * u[3]])
    #alpha = np.array([2 * (0.6) * (u[4]-0.5)])
    #sigma0 = np.array([np.log10(true_value[5]) - 0.5 + 2 * (0.5) * u[5]])
    
    return np.array([m_chirp, rho6, gamma, q]).reshape(4,) # , alpha, sigma0


def unpack(x: np.ndarray) -> myDarkMatter:
    """
    Convenience function to unpack parameters into a dark dress.
    """
    dMc = x[0]
    drho6 = x[1]
    dgamma = x[2]
    dq = x[3]
    #dalpha = x[4]
    #dsigma0 = x[5]
    
    Mc = _COMBO.Binary_init.chirp_mass + dMc * MSUN
    rho6 = 10**(drho6)
    gammas = _COMBO.DarkMatter_init.gammas + dgamma
    q = 10**(dq)
    #alpha = _COMBO.Accretion_init.alpha + dalpha
    #sigma0 = 10**(dsigma0)
    
    m_1 = get_m_1(Mc, q)
    m_2 = get_m_2(Mc, q)
    
    DL = _COMBO.Binary_init.dist
    
    return myDarkMatter(m1=m_1,
                        m2=m_2,
                        dist=DL,
                        q=q,
                        gammas=gammas,
                        rho6=rho6,
                        r6=_COMBO.DarkMatter_init.r6,
                        epsv=_COMBO.DarkMatter_init.epsv,
                        chirp_mass=Mc)

def get_ll_fft(x: np.ndarray) -> np.ndarray:
    """
    Likelihood function
    """
    ad_h = unpack(x)
    return loglikelihood_fft(ad_h, _COMBO, FS, PAD_LOW, PAD_HIGH, S_n)

mtrue = 0
gamma_true = 0
#alpha_true = 0

logrho6_true = np.log10(_COMBO.DarkMatter_init.rho6)
logq_true = np.log10(_COMBO.Binary_init.q)
#logsigma0_true = np.log10(_COMBO.Accretion_init.sigma0)

# Initialize the nested sampler
# Use 500 - 2000 live points. You need a lot, otherwise you may miss the high-likelihood region!
sampler = dynesty.NestedSampler(get_ll_fft, ptform, 4, nlive=1200, bound='single')

# Run the nested sampling
sampler.run_nested(dlogz=1.0)

# Extract the results
results = sampler.results

import pickle

with open('sampling_results_dmVScombo.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to 'sampling_results_dmVScombo.pkl'")
