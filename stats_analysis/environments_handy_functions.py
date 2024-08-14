import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy as sp
from typing import Callable, NamedTuple, Tuple, Type, Union
from scipy.integrate import quad_vec, simpson
from scipy.special import hyp2f1, betainc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LogNorm
from scipy.integrate import cumulative_trapezoid, cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, solve_ivp, simps, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# -------------------------- Define some constants: -----------------------------------

G = 6.67408e-11 # kg^-1 m^3 / s^2
c = 299792458.0 # m / s
pc = 3.08567758149137e16 # m
m_sun = 1.98855e30 # kg
year = 31557600.0  # s

# -------------------------- Define some conversion functions: -----------------------------------

def find_cf(m1, gammas, M_tot, r_s, rho_s, epsv, logL):

    return 5 * c**5 / (8 * m1**2) * np.pi**(2 * (gammas - 4) / 3) * G**(-(2 + gammas) / 3) * M_tot**((1 - gammas)/3) * r_s**(gammas) * epsv * rho_s * logL
    
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
    
def hypgeom_scipy(theta, y):
    '''Compute the hypergeometric function using scipy.'''
    return hyp2f1(1, theta, 1 + theta, -y**(-5 / (3*theta)))

def hyp2f1_derivative(hypr2f1, f):
    '''Finds derivative of the gauss_hypergeom function through differentiation.'''
    delta_function = np.concatenate(([np.min(hypr2f1)], hypr2f1[1:] - hypr2f1[:-1]))
    delta_fs = np.concatenate(([np.max(f)], f[1:] - f[:-1]))
    return delta_function/delta_fs

# -------------------------- Environments: -----------------------------------

class myVacuumBinary:
    
    """
    VacuumBinary

    Overview:
    Initializes an environment for a vacuum binary system, evaluates the derivative of the radius, and aids in finding the phase to coalescence of the binary system.

    Attributes:
    ----------
    m1 : float
        Mass of the first body in kilograms.
    
    m2 : float
        Mass of the second body in kilograms.
    
    mu : float
        Reduced mass of the binary system.
    
    q : float
        Mass ratio of the binary system (m2/m1).
    
    M_tot : float
        Total mass of the binary system.
    
    dist : float
        Distance to the binary system.
    
    chirp_mass : float
        Chirp mass of the binary system, relevant for gravitational wave calculations.

    Methods:
    -------
    frequency(r: float) -> float
        Finds the binary frequency assuming circular orbits, at fixed radius r.

    radius(f: float) -> float
        Finds the binary radius assuming circular orbits, at fixed frequency f.

    df_dr(r: float) -> float
        Finds the binary frequency radial derivative assuming circular orbits.

    vacuum_phase(r: float) -> float
        Analytical vacuum phase solution, integral from frequency to innermost stable circular orbit (ISCO) frequency.

    dvacuum_phase_df(r: float) -> float
        Analytical derivative of the vacuum phase solution with respect to frequency.

    dot_r_gw(r: float) -> float
        Finds the derivative of radial separation in the vacuum case.

    Examples:
    --------
    >>> binary = VacuumBinary(1.4 * 1.989e30, 1.4 * 1.989e30, 1e20)
    >>> r = 1e7  # example radius
    >>> freq = binary.frequency(r)
    >>> phase = binary.vacuum_phase(r)
    >>> dr_dt = binary.dot_r_gw(r)
    """
    
    def __init__(self, m1, m2, dist, chirp_mass):
        
        """
        Initializes the VacuumBinary class.

        Parameters:
        ----------
        m1 : float
            Mass of the first body in kilograms.
        
        m2 : float
            Mass of the second body in kilograms.
        
        dist : float
            Distance to the binary system.
        """
        
        self.m1 = m1
        self.m2 = m2
        self.mu = self.m1 * self.m2 / (self.m1 + self.m2)
        self.q = self.m2/self.m1
        self.M_tot = self.m1 + self.m2
        self.dist = dist
        self.chirp_mass = chirp_mass
    
    def frequency(self, r):
        return np.sqrt(G * self.M_tot / r**3) / np.pi
    
    def radius(self, f):
        return (G * self.M_tot / (f**2 * np.pi**2))**(1/3)
    
    def df_dr(self, r):
        return 1 / np.pi * (-3/2) * np.sqrt(G * self.M_tot / r**5)
    
    def vacuum_phase(self, r):
        freqs = self.frequency(r)
        r_isco = 6 * G * self.m1 / c**2
        f_isco = self.frequency(r_isco)
        return 1/16 * (c**3 / (np.pi * G * self.chirp_mass))**(5/3) * (-freqs**(-5/3) + f_isco**(-5/3))
    
    def dvacuum_phase_df(self, r):
        freqs = self.frequency(r)
        return 1/16 * (c**3 / (np.pi * G * self.chirp_mass))**(5/3) * (-5/3) * (-freqs**(-5/3-1))
    
    def dot_r_gw(self, r):
        return - 64 * self.M_tot * G**3 * self.m1 * self.m2 / (5 * c**5 * r**3)


class myAccretionDisk:
    
    """
    AccretionDisk

    Overview:
    Initializes an environment for a binary system embedded in an accretion disk, evaluates the surface density and density profile of the disk, and calculates torques and radial separation derivatives within the disk.

    Attributes:
    ----------
    Binary_init : VacuumBinary
        An instance of the VacuumBinary class representing the binary system.

    mach : float
        Mach number characterizing the disk.

    sigma0 : float
        Surface density normalization constant.

    alpha : float
        Power-law index for the surface density profile.

    r0 : float
        Reference radius for the surface density normalization.

    Methods:
    -------
    sigma_acc(r: float) -> float
        Finds the surface density profile of an accretion disk.

    rho_disk(r: float) -> float
        Finds the density profile of the disk using \rho ≈ Σ(r) / (2h), where h = r / Mach.

    gas_torque(r: float) -> float
        Finds the gas torque on the secondary component of the binary.

    dot_r_acc(r: float) -> float
        Finds the derivative of radial separation within an accretion disk.

    Examples:
    --------
    >>> disk = AccretionDisk(1.4 * 1.989e30, 1.4 * 1.989e30, 1e20, 10, 1e3, -1, 1e9)
    >>> r = 1e7  # example radius
    >>> sigma = disk.sigma_acc(r)
    >>> rho = disk.rho_disk(r)
    >>> torque = disk.gas_torque(r)
    >>> dr_dt = disk.dot_r_acc(r)
    """
    
    def __init__(self, m1, m2, dist, mach, sigma0, alpha, r0, chirp_mass):
        
        """
        Initializes the AccretionDisk class.

        Parameters:
        ----------
        m1 : float
            Mass of the first body in kilograms.
        
        m2 : float
            Mass of the second body in kilograms.
        
        dist : float
            Distance to the binary system.
        
        mach : float
            Mach number characterizing the disk.
        
        sigma0 : float
            Surface density normalization constant.
        
        alpha : float
            Power-law index for the surface density profile.
        
        r0 : float
            Reference radius for the surface density normalization.
        """
        
        self.Binary_init = myVacuumBinary(m1 = m1, m2 = m2, dist = dist, chirp_mass = chirp_mass)
        self.mach = mach
        self.sigma0 = sigma0
        self.alpha = alpha
        self.r0 = r0
        self.chirp_mass = chirp_mass
        
    def sigma_acc(self, r):
        return self.sigma0 * (r / self.r0)**(self.alpha)

    def rho_disk(self, r):
        h = r / self.mach
        return self.sigma_acc(r) / (2 * h)
    
    def gas_torque(self, r):
        omega_2 = G * self.Binary_init.M_tot / r**3
        return - self.sigma_acc(r) * r**4 * omega_2 * self.Binary_init.q**2 * self.mach**2
    
    def dot_r_acc(self, r):
    
        r_dot_acc = 2 * self.gas_torque(r) * r**(1/2) / (self.Binary_init.mu * (G * self.Binary_init.M_tot)**(1/2))
        
        #r_dot_acc = self.gas_torque(r) * r**(1/2) / (2 * G**(1/2) * self.Binary_init.m2 * self.Binary_init.M_tot**(1/2))
        
        return r_dot_acc


class myDarkMatter:
    
    """
    DarkMatter

    Overview:
    Initializes an environment for a binary system embedded in a dark matter spike, evaluates the dark matter density profile, and calculates the radial separation derivatives in a dark matter environment.

    Attributes:
    ----------
    Binary_init : VacuumBinary
        An instance of the VacuumBinary class representing the binary system.

    logL : float
        The Coulomb logarithm.

    gammas : float
        Slope of the dark matter density profile.

    rho6 : float
        Density normalization constant at radius r6.

    r6 : float
        Reference radius for the density normalization.

    epsv : float
        Velocity dispersion parameter.

    Methods:
    -------
    rho_dm(r: float) -> float
        Finds the density profile of the dark matter using the specified formulation.

    dot_r_dm_s(r: float) -> float
        Finds the derivative of radial separation within a static dark matter environment.

    dot_r_dm_eff(r: float) -> float
        Finds the derivative of radial separation considering an effective dark matter environment from previous numerical fits on the HaloFeedback algorithm.

    Examples:
    --------
    >>> dm = DarkMatter(1.4 * 1.989e30, 1.4 * 1.989e30, 1e20, 0.1, 1e-2, 1e3, 1e9, 0.1)
    >>> r = 1e7  # example radius
    >>> rho = dm.rho_dm(r)
    >>> dr_dt_s = dm.dot_r_dm_s(r)
    >>> dr_dt_eff = dm.dot_r_dm_eff(r)
    """
    
    def __init__(self, m1, m2, dist, q, gammas, rho6, r6, epsv, chirp_mass):
        
        """
        Initializes the DarkMatter class.

        Parameters:
        ----------
        m1 : float
            Mass of the first body in kilograms.
        
        m2 : float
            Mass of the second body in kilograms.
        
        dist : float
            Distance to the binary system.
        
        q : float
            Mass ratio parameter.

        gammas : float
            Slope of the dark matter density profile.
        
        rho6 : float
            Density normalization constant at radius r6.
        
        r6 : float
            Reference radius for the density normalization.
        
        epsv : float
            Velocity dispersion parameter.
        """
    
        self.Binary_init = myVacuumBinary(m1 = m1, m2 = m2, dist = dist, chirp_mass = chirp_mass)
        self.logL = 1 / np.sqrt(q)
        self.gammas = gammas
        self.rho6 = rho6
        self.r6 = r6
        self.epsv = epsv
        self.chirp_mass = chirp_mass
    
    def rho_dm(self, r):
        return self.rho6 * (self.r6 / r)**(self.gammas)
    
    def dot_r_dm_s(self, r):
        r_dot_dm = - 8 * np.pi * G**(1/2) * self.Binary_init.mu * self.logL * self.rho_dm(r) * r**(5/2) * self.epsv / self.Binary_init.M_tot**(3/2)
        return r_dot_dm
    
    def dot_r_dm_eff(self, r):

        freqs = self.Binary_init.frequency(r)
        df_dr_s = self.Binary_init.df_dr(r)
        
        # Vacuum
        
        phase_vacuum = self.Binary_init.vacuum_phase(r)
        dphase_vacuum_df = self.Binary_init.dvacuum_phase_df(r)
        
        # Constants
        
        rho_s = get_rho_s(self.rho6, self.Binary_init.m1, self.gammas)
        r_s = get_r_s(self.Binary_init.m1, rho_s, self.gammas)
        
        gamma_e = 5/2
        cf = find_cf(self.Binary_init.m1, self.gammas, self.Binary_init.M_tot, r_s, rho_s, self.epsv, self.logL)
        f_eq = cf**(3 / (11 - 2 * self.gammas))
        theta = 5 / (2 * gamma_e)
        lambda_ = (11 - 2 * (self.gammas + gamma_e)) / 3
        
        alpha_1 = 1.4412
        alpha_2 = 0.4511
        beta = 0.8163
        xi = - 0.4971
        gamma_r = 1.4396
        f_b = beta * (self.Binary_init.m1 / (1000 * m_sun))**(-alpha_1) * (self.Binary_init.m2 / m_sun)**(alpha_2) * (1 + xi * np.log(self.gammas/gamma_r))
        
        f_t = f_b
        eta = (5 + 2*gamma_e) / (2 * (8 - self.gammas)) * (f_eq / f_b)**((11 - 2 * self.gammas) / 3)
        
        # Variables
    
        y = freqs / f_t
        dy_df = 1 / f_t
        
        # Hypergeometric function
        
        gauss_hypergeom = hyp2f1(1, theta, 1 + theta, - y**(-5/(3 * theta)))
        dgauss_hypergeom_df = hyp2f1_derivative(gauss_hypergeom, freqs)
        
        # Find phase derivative in frequency
        
        exp_y = y**(-lambda_)
        d_phase_df_s = dphase_vacuum_df - (dphase_vacuum_df * eta * exp_y + phase_vacuum * eta * (-lambda_) * exp_y/y * dy_df) + eta * (-lambda_ * exp_y/y * phase_vacuum * dy_df * gauss_hypergeom + exp_y * dphase_vacuum_df * gauss_hypergeom + exp_y * phase_vacuum * dgauss_hypergeom_df)
        
        df_dt_s = 2 * np.pi * freqs / d_phase_df_s
        
        #print(df_dt)
        
        return df_dt_s * df_dr_s**(-1)
        
        
class myCombination:
    
    def __init__(self, m1, m2, dist, mach, sigma0, alpha, r0, chirp_mass, q, gammas, rho6, r6, epsv):
        
        self.Binary_init = myVacuumBinary(m1 = m1, m2 = m2, dist = dist, chirp_mass = chirp_mass)
        self.Accretion_init = myAccretionDisk(m1 = m1, m2 = m2, dist = dist, mach = mach, sigma0 = sigma0, alpha = alpha, r0 = r0, chirp_mass = chirp_mass)
        self.DarkMatter_init = myDarkMatter(m1 = m1, m2 = m2, dist = dist, q = q, gammas = gammas, rho6 = rho6, r6 = r6, epsv = epsv, chirp_mass = chirp_mass)
    
    def dot_r_combo(self, r):
    
        r_dot_acc = self.Accretion_init.dot_r_acc(r)
        r_dot_dm = self.DarkMatter_init.dot_r_dm_eff(r)
        
        return r_dot_acc + r_dot_dm



# -------------------------- Grid functions: -------------------------------------------
    
def find_grid(params):
    
    r_isco = 6 * G * params.Binary_init.m1 / c**2
    f_h = params.Binary_init.frequency(r_isco)

    if isinstance(params, myAccretionDisk):
        accretion = params
        f_l = f_1yr(accretion)
        
    if isinstance(params, myDarkMatter):
        darkmatter = params
        f_l = f_1yr(darkmatter)
        
    if isinstance(params, myCombination):
        combo = params
        f_l = f_1yr(combo)
        
        
    return (f_l, f_h)

# -------------------------- Base functions: -------------------------------------------
    
def df_dt(f, params):
    '''Finds the binary frequency time derivative.'''
    r = params.Binary_init.radius(f)
    df_dr = params.Binary_init.df_dr(r)
    
    if isinstance(params, myAccretionDisk):
        accretion = params
        dot_r = accretion.dot_r_acc(r) + accretion.Binary_init.dot_r_gw(r)
    
    if isinstance(params, myDarkMatter):
        darkmatter = params
        dot_r = darkmatter.dot_r_dm_eff(r)
    
    if isinstance(params, myCombination):
        combo = params
        dot_r = combo.DarkMatter_init.dot_r_dm_eff(r) + combo.Accretion_init.dot_r_acc(r)
    
    return df_dr * dot_r
    
    
def ddot_phase(f, params):
    '''Finds the binary second derivative phase as a function of frequency, using the equations for the accretion disk.'''
    return 2 * np.pi * df_dt(f, params)

def h_0(f, params):
    '''Finds the strain as a function of frequency, and \ddot{\Phi}'''
    return 1/2 * 4 * np.pi**(2/3) * G**(5/3) * params.Binary_init.chirp_mass**(5/3) * f**(2/3) / c**4 * (2 * np.pi / ddot_phase(f, params))**(1/2) #/ (vacuum.dist)
    
# -------------------------- Time to coalescence: -------------------------------------------

def time_to_coal(f_grid, params):
    '''Time to coalescence integrated using ODE rule.'''
    def to_integrate(f):
        function = df_dt(f, params)**(-1)
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

def time_to_coal_cumul(f_grid, params):
    '''Time to coalescence integrated using simple cumulative rule (faster!).'''
    to_integrate = df_dt(f_grid, params)**(-1)
    time = cumulative_trapezoid(to_integrate, f_grid, initial=0)
    return time

# -------------------------- Phase to coalescence: -------------------------------------------

def phase_f_cumul(f_grid, params):
    '''Phase to coalescence integrated using simple cumulative rule (faster!).'''
    to_integrate = 2 * np.pi * df_dt(f_grid, params)**(-1) * f_grid
    phase = cumulative_trapezoid(to_integrate, f_grid, initial=0)
    return phase

def f_1yr(params):
    '''Finds the coalescence time integrating over the frequency domain.'''
    r_isco = 6 * params.Binary_init.m1 * G / c**2
    f_isco = params.Binary_init.frequency(r_isco)
    f_grid = np.linspace(f_isco, f_isco * 0.001, int(1e6))

    t_coal_f = -time_to_coal(f_grid, params)
    #print(t_coal_f)
        
    interpolation = interp1d(t_coal_f, f_grid)
    return interpolation(year)
    
# -------------------------- Extrinsic functions: -------------------------------------------
    
def mycalculate_SNR(params, fs, S_n):
    integrand = amplitude(fs, params) ** 2 / S_n(fs)
    return np.sqrt(4 * -np.trapz(integrand, fs)) # - is for reversed f-grid
     

def PhiT(f, params):
    return 2 * np.pi * f * time_to_coal(f, params) - phase_f(f, df_dt(f, params))
    
def PhiT_cumul(f, params):
    return 2 * np.pi * f * time_to_coal_cumul(f, params) - phase_f_cumul(f, params)
    
def Psi(f, params, TTC, PHI_C):
    return 2 * np.pi * f * TTC - PHI_C - np.pi / 4 - PhiT_cumul(f, params) # changed PhiT with PhiT_cumul

def amplitude(f, params):
    '''Amplitude averaged over inclination angle.'''
    return np.sqrt(4 / 5) * h_0(f, params) / params.Binary_init.dist
