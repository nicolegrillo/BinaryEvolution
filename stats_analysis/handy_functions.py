import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy as sp
from scipy.integrate import quad_vec, simpson
from scipy.special import hyp2f1, betainc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LogNorm
from scipy.integrate import cumulative_trapezoid, cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, solve_ivp, simps
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import environments

# -------------------------- Define some constants: -----------------------------------

G = 6.67408e-11 # kg^-1 m^3 / s^2
c = 299792458.0 # m / s
pc = 3.08567758149137e16 # m
m_sun = 1.98855e30 # kg
year = 31557600.0  # s

# -------------------------- General functions: ---------------------------------------

def hypgeom_scipy(theta, y):
    '''Compute the hypergeometric function using scipy.'''
    return hyp2f1(1, theta, 1 + theta, -y**(-5 / (3*theta)))

def hyp2f1_derivative(hyp2f1, f):
    '''Finds derivative of the gauss_hypergeom function through differentiation.'''
    delta_function = np.concatenate(([np.min(hyp2f1)], hyp2f1[1:] - hyp2f1[:-1]))
    delta_fs = np.concatenate(([np.max(f)], f[1:] - f[:-1]))
    return delta_function/delta_fs
    
def find_cf(m1, gammas, M_tot, r_s, rho_s, epsv, logL):

    return 5 * c**5 / (8 * m1**2) * np.pi**(2 * (gammas - 4) / 3) * G**(-(2 + gammas) / 3) * M_tot**((1 - gammas)/3) * r_s**(gammas) * epsv * rho_s * logL

def to_integrate_phase(f, df_dt):
    function = 2 * np.pi * df_dt**(-1) * f 
    return interp1d(f, function, kind='cubic')

def phase_f(f, df_dt):
    '''Finds the binary phase as a function of frequency.'''
    to_integrate_f = to_integrate_phase(f, df_dt)
    
    # Differential equation for the phase
    def phase_ode(f, y):
        return to_integrate_f(f)
    
    # Initial condition
    y0 = [0]
    
    # Solving the IVP
    result = solve_ivp(phase_ode, [f[0], f[-1]], y0, t_eval=f, rtol=1e-14, atol=1e-14)
    
    # Extracting the solution
    phase_f = result.y[0]
    
    return phase_f

# ---------------------- Conversion functions: -------------------------------------

def get_r_s(m_1, rho_s, gamma_s):
    return ((3 - gamma_s) * 0.2 ** (3 - gamma_s) * m_1 / (2 * np.pi * rho_s)) ** (1 / 3)
    
def get_rho_s(rho_6, m_1, gamma_s):
    a = 0.2
    r_6 = 1e-6 * pc
    m_tilde = ((3 - gamma_s) * a ** (3 - gamma_s)) * m_1 / (2 * np.pi)
    return (rho_6 * r_6 ** gamma_s / (m_tilde ** (gamma_s / 3))) ** (1 / (1 - gamma_s / 3))

def get_rho_6(rho_s, m_1, gamma_s):
    a = 0.2
    r_s = ((3 - gamma_s) * a ** (3 - gamma_s) * m_1 / (2 * np.pi * rho_s)) ** (1 / 3)
    r_6 = 1e-6 * pc
    return rho_s * (r_6 / r_s) ** -gamma_s
    
# -------------------------- Grid functions: -------------------------------------------
    
def find_grid(accretion):

    f_l = f_1yr(accretion)

    r_isco = 6 * G * accretion.Binary_init.m1 / c**2
    f_h = accretion.Binary_init.frequency(r_isco)
    
    return (f_l, f_h)
    
# ----------------------- Phase functions: --------------------------------------------


def df_dt_accretion(f, accretion):
    '''Finds the binary frequency time derivative.'''
    r = accretion.Binary_init.radius(f)
    df_dr = accretion.Binary_init.df_dr(r)
    dot_r = accretion.dot_r_acc(r) + accretion.Binary_init.dot_r_gw(r)
    return df_dr * dot_r
    
def df_dt_gw(f, accretion):
    '''Finds the binary frequency time derivative.'''
    r = accretion.Binary_init.radius(f)
    df_dr = accretion.Binary_init.df_dr(r)
    dot_r = accretion.Binary_init.dot_r_gw(r)
    #print(df_dr)
    return df_dr * dot_r
    
def ddot_phase_acc(f, accretion):
    '''Finds the binary second derivative phase as a function of frequency, using the equations for the accretion disk.'''
    return 2 * np.pi * df_dt_accretion(f, accretion)
    
def h_0(f, accretion):
    
    '''Finds the strain as a function of frequency, and \ddot{\Phi}'''
    
    return 1/2 * 4 * np.pi**(2/3) * G**(5/3) * accretion.Binary_init.chirp_mass**(5/3) * f**(2/3) / c**4 * (2 * np.pi / ddot_phase_acc(f, accretion))**(1/2) #/ (vacuum.dist)
    
def amplitude(f, accretion):
    '''Amplitude averaged over inclination angle.'''
    return np.sqrt(4 / 5) * h_0(f, accretion) / accretion.Binary_init.dist
    
    
def time_to_coal(f_grid, accretion):

    def to_integrate(f):
        function = df_dt_accretion(f, accretion)**(-1)
        return interp1d(f, function) # , kind='cubic', fill_value="extrapolate"
        
    to_integrate_function = to_integrate(f_grid)

    # Differential equation for the time to coalescence
    def time_to_coal_int(f, y):
        return to_integrate_function(f)

    # Initial condition
    y0 = [0]

    # Solving the IVP
    result = solve_ivp(time_to_coal_int, [f_grid[0], f_grid[-1]], y0, t_eval=f_grid, rtol=1e-14, atol=1e-14)

    # Extracting the solution
    t_coal_f = result.y[0]
    
    return t_coal_f
    
def f_1yr(accretion):
    
    '''Finds the coalescence time integrating over the frequency domain.'''
    
    r_isco = 6 * accretion.Binary_init.m1 * G / c**2
    f_isco = accretion.Binary_init.frequency(r_isco)
    f_grid = np.linspace(f_isco, f_isco * 0.001, int(1e6))
    
    t_coal_f = -time_to_coal(f_grid, accretion)
        
    interpolation = interp1d(t_coal_f, f_grid)
    
    return interpolation(year)
    
def mycalculate_SNR(accretion, fs, S_n):
    integrand = amplitude(fs, accretion) ** 2 / S_n(fs)
    return np.sqrt(4 * np.trapz(integrand, fs))
    

def PhiT(f, accretion):
    return 2 * np.pi * f * time_to_coal(f, accretion) - phase_f(f, df_dt_accretion(f, accretion))
    
def Psi(f, accretion, TTC, PHI_C):
    return 2 * np.pi * f * TTC - PHI_C - np.pi / 4 - PhiT(f, accretion)

