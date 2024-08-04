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


# Define some constants: 

G = 6.67408e-11 # kg^-1 m^3 / s^2
c = 299792458.0 # m / s
pc = 3.08567758149137e16 # m
m_sun = 1.98855e30 # kg
year = 365.25 * 24 * 3600  # s


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


class DarkMatter: 
    
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
    
    def __init__(self, m1, m2, dist, q, gammas, rho6, r6, epsv):
        
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
    
        self.Binary_init = VacuumBinary(m1 = m1, m2 = m2, dist = dist)
        self.logL = 1 / np.sqrt(q)
        self.gammas = gammas
        self.rho6 = rho6
        self.r6 = r6
        self.epsv = epsv
    
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
        
        df_dt = 2 * np.pi * freqs / d_phase_df_s
        
        return df_dt * df_dr_s**(-1)


        
