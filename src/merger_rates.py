### ---> General
import numpy as np
from scipy import interpolate

### ---> Cosmology
from astropy import constants as const
from astropy.cosmology import Planck15
from astropy import units as u


from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmo = cosmology.setCosmology('planck15')



class halo:
    def __init__(self, M_halo, R_ns_over_R_NFW):
        self.M_halo = M_halo
        self.R_ns_over_R_NFW = R_ns_over_R_NFW

        self.C_lst = self.concentration()

        ### First we compute the normalized masses of the haloes then obtain rho_NFW_0/rho_crit
        ### NFW_mass_integral stands for 4 pi Int dx x^2 NFW(x)
        NFW_mass_integral = self.NFW_tilde_integral_1(self.C_lst) - self.NFW_tilde_integral_1(np.zeros(len(self.C_lst)))
        rho_NFW_0_over_rho_crit = 200.*self.C_lst**3/3./NFW_mass_integral


        ### Now we can obtain rho_NFW_0 for all the halos
        rho_crit = Planck15.critical_density(0).to(u.Msun/u.kpc**3) # M_sun/kpc^3
        self.rho_NFW_0_lst = rho_NFW_0_over_rho_crit*rho_crit # M_sun/kpc^3


        ### Similarly, we obtain the R_NFW_0 and R_vir for all the halos
        R_NFW_0_cube_rho_crit = self.M_halo/NFW_mass_integral/4./np.pi/rho_NFW_0_over_rho_crit
        self.R_NFW_0_lst = (R_NFW_0_cube_rho_crit/rho_crit)**(1./3.) # kpc
        self.R_vir_lst = self.R_NFW_0_lst*self.C_lst # kpc


        ### Number of neutron stars in each halo is computed as
        SFHR_params = [10**(-1.777), 10**(11.514), -1.412, 0.316, 3.508]
        self.M_star_lst = self.stellar_mass(SFHR_params)
        self.N_ns_lst = self.N_ns(SFHR_params)["N_ns"]
        self.R_ns_0_lst = R_ns_over_R_NFW*self.R_NFW_0_lst

        ### We evaluate rho_ns_0 by the normalization condition 4 pi rho_ns_0 R_ns_0^3 Int dx x^2 NS(x) = N_ns M_ns
        ### M_ns is assumed to be 1 M_sun
        ns_mass_integral = self.NS_tilde_integral(self.C_lst*self.R_NFW_0_lst/self.R_ns_0_lst) - self.NS_tilde_integral(0.)
        denominator = 4.*np.pi*self.R_ns_0_lst**3*ns_mass_integral
        numerator = self.N_ns_lst*u.M_sun
        self.rho_ns_0_lst = numerator/denominator


        ### We also prepare the NFW and ns profiles for later integrals
        self.x_lst = np.geomspace(1e-5, self.C_lst, 20000).T # This seems to be a reasonable resolution
        self.NFW_profiles = self.NFW_tilde(self.x_lst)
        self.ns_profiles = self.NS_tilde(self.x_lst, np.array([self.R_NFW_0_lst, self.R_ns_0_lst]))

        ### And for plotting purposes
        self.x_lst_plotting = np.geomspace(1e-15, self.C_lst, 10000).T
        self.NFW_profiles_plotting = self.NFW_tilde(self.x_lst_plotting)
        self.ns_profiles_plotting = self.NS_tilde(self.x_lst_plotting, np.array([self.R_NFW_0_lst, self.R_ns_0_lst]))


        ### The virial velocities and velocity dispersion of each halo
        C_max = 2.1626
        self.v_vir_lst = np.sqrt(const.G*self.M_halo/self.R_vir_lst).to(u.km/u.s) # km/s
        self.v_dm_lst = self.v_vir_lst*np.sqrt(self.C_lst/self.g(self.C_lst))\
                        /np.sqrt(C_max/self.g(C_max))/np.sqrt(2.) # km/s


    def NFW_tilde(self, x):

            return 1./x/(1. + x)**2

    def NFW_tilde_integral_1(self, x):
        ### Int x^2 NFW(x) dx

        return 1./(1. + x) + np.log(1. + x)

    def NFW_tilde_integral_2(self, x):
        ### Int x^2 NFW(x)^2 dx

        return -1./3./(1. + x)**3

    def NS_tilde_integral(self, x):
        ### Int x^2 NS(x) dx

        return (-2. -2.*x - x**2)*np.exp(-x)
        #return np.exp(-x)*(-6. - 6.*x - 3.*x**2 - x**3)


    def NS_tilde(self, x, params):

        params_length = len(params[0,:])

        R_NFW_0 = params[0,:].reshape(params_length, 1)
        R_ns_0 = params[1,:].reshape(params_length, 1)

        return np.exp(-x*R_NFW_0/R_ns_0)

    def N_ns(self, SHMR_params):

        salpeter_prefac = 0.17 # If int phi(m)mdm = 1, then salpeter_prefac = 0.17
        salpeter_power = -2.35


        NS_M_min = 8. # M_sun
        NS_M_max = 20. # M_sun

        M_stellar = self.stellar_mass(SHMR_params)
        #M_NGC_4993 = 2.95*1e10 M_sun

        integral_1 = salpeter_prefac*(1./(salpeter_power + 1.))*\
                            (NS_M_max**(salpeter_power + 1.) - NS_M_min**(salpeter_power + 1.))


        #t_0 = 3. #Gyr
        #tau = 0.1

        #t_life = 0.02 # Gyr
        #t_lst = np.linspace(0.1 - t_life, 10 - t_life, 100)
        #psi = self.SFH(t_lst, [t_0, tau])

        #integral_2 = np.trapz(psi, t_lst)

        N_ns = M_stellar*integral_1#*integral_2
        return {"N_ns":N_ns, "M_stellar":M_stellar}
        #, "SFH":psi


    def SFH(self, t_lst, SFH_params):
        t_0, tau = SFH_params

        tmp_1 = 1./t_lst/np.sqrt(2.*np.pi*tau**2)
        tmp_2 = np.exp(-(np.log(t_lst) - np.log(t_0))**2/2./tau**2)

        return tmp_1*tmp_2


    def g(self, C):
        return np.log(1. + C) - C/(1. + C)



    def stellar_mass(self, SHMR_params):
        ### --- Fitting function from appendix A of 1401.7329
        ### --- M_halo in M_sun units

        epsilon, M_1, alpha, gamma, delta = SHMR_params
        f = lambda x: -np.log10(10**(alpha*x) + 1.) + delta*(np.log10(1. + np.exp(x))**gamma)/(1. + np.exp(10**(-x)))

        M_star = epsilon*M_1*10**(f(np.log10(self.M_halo.value/M_1)) - f(0.))
        #print("M_star = ", M_star)


        return M_star

    def concentration(self):
        ### --- Fitting function from appendix C of 1601.02624
        ### --- M_halo in M_sun/h units

        h = 0.7
        reference_mass = 1e10*h*u.Msun # in M_sun/h
        xi = 1./(self.M_halo/reference_mass)

        c_0 = 3.395
        beta = 0.307
        gamma_1 = 0.628
        gamma_2 = 0.317
        nu_0 = 4.135 - 0.564 - 0.210 + 0.0557 - 0.00348

        delta_sc = 1.686
        sigma_M = 22.26*xi**0.292/(1. + 1.53*xi**0.275 + 3.36*xi**0.198)
        nu = delta_sc/sigma_M

        concentration = c_0*(nu/nu_0)**(-gamma_1)*(1. + (nu/nu_0)**(1./beta))**(-beta*(gamma_2 - gamma_1))

        return concentration

    def halo_properties(self):

        halo_property_dictionary = {"Concentration":self.C_lst,
                                    "rho_NFW_0":self.rho_NFW_0_lst,
                                    "R_NFW_0":self.R_NFW_0_lst,
                                    "R_virial":self.R_vir_lst, \
                                    "N_ns":self.N_ns_lst,
                                    "rho_ns_0":self.rho_ns_0_lst,
                                    "x_lst":self.x_lst,
                                    "NFW_profiles":self.NFW_profiles,
                                    "ns_profiles":self.ns_profiles,
                                    "x_lst_plotting":self.x_lst_plotting,
                                    "NFW_profiles_plotting":self.NFW_profiles_plotting,
                                    "ns_profiles_plotting":self.ns_profiles_plotting,
                                    "M_star":self.M_star_lst}

        return halo_property_dictionary


class rates:
    def __init__(self, halo, binary_mass_params):
        self.halo = halo
        self.m_1 = binary_mass_params["m_1"]
        self.m_2 = binary_mass_params["m_2"]
        self.M = binary_mass_params["M"]
        self.eta = binary_mass_params["eta"]


    def cross_section(self):

        ### Now we turn to velocities

        v_vir_lst = self.halo.v_vir_lst
        v_dm_lst = self.halo.v_dm_lst


        ### Here is our Maxwell-Boltzmann distribution for each halo
        v_PBH_lst = np.linspace(1e-12, np.ones(len(v_vir_lst)), 10000).T ### Normalized to v_vir

        exponent_1_numerator = v_PBH_lst**2*v_vir_lst.reshape(len(v_vir_lst), 1)**2
        exponent_1_denominator = v_dm_lst.reshape(len(v_dm_lst), 1)**2
        exponent_1 = -exponent_1_numerator/exponent_1_denominator

        exponent_2_numerator = np.ones(np.shape(v_PBH_lst))*v_vir_lst.reshape(len(v_vir_lst), 1)**2
        exponent_2_denominator = v_dm_lst.reshape(len(v_dm_lst), 1)**2
        exponent_2 = -exponent_2_numerator/exponent_2_denominator

        ### The MB dist
        P_v_PBH = np.exp(exponent_1) - np.exp(exponent_2)

        ### We normalize the distributions
        normalization_integral = v_vir_lst**3*np.trapz(P_v_PBH*v_PBH_lst**2, v_PBH_lst, axis = 1)
        F_0_lst = 1./4./np.pi/normalization_integral

        ### The normalized distributions
        P_v_PBH = F_0_lst.reshape(len(F_0_lst), 1)*P_v_PBH

        ### The expectation value of the velocity term
        #4.*np.pi*
        # In units of (km/s)^-11/7
        avg_vel_PBH_factor = v_vir_lst**(10./7.)*np.trapz(P_v_PBH*v_PBH_lst**(3./7.), v_PBH_lst, axis = 1)
        # In units of (kpc/yr)^-11/7
        self.avg_vel_PBH_factor = avg_vel_PBH_factor.to(u.kpc**(-11./7.)/u.year**(-11./7.))


        ### The velocity independent part of <sigma v>
        sigma_prefac = np.pi*(85.*np.pi/3.)**(2./7.)*2.**(4./7.)*const.G**2*self.M**2*self.eta**(2./7.)*const.c**(-10./7.)
        sigma_prefac = sigma_prefac.to(u.kpc**(32/7)/u.year**(18/7))

        ### <sigma v> in units of kpc^3/yr
        self.sigma_v_avg = sigma_prefac*avg_vel_PBH_factor

        ### sigma for a reference velocity of 200 km/s, in units of kpc^2
        reference_vel = (200.*u.km/u.s).to(u.kpc/u.year)
        sigma_reference_200 = sigma_prefac*reference_vel**(-18./7.)

        velocity_dictionary = {"v_vir_lst":v_vir_lst,
                               "v_dm_lst":v_dm_lst,
                               "sigma_v_avg":self.sigma_v_avg,
                               "sigma_reference_200":sigma_reference_200}

        return velocity_dictionary

    def rates_per_halo(self):

        cross_section_properties = self.cross_section()

        ### PBH-PBH merger rate per halo

        ### NFW-NFW integral
        density_integral = np.trapz(self.halo.x_lst**2*self.halo.NFW_profiles**2, self.halo.x_lst, axis = 1)
        ### PBH-PBH rate per halo in units of 1/yr
        PBHPBH_Rate_per_halo = self.sigma_v_avg*self.halo.rho_NFW_0_lst**2*self.halo.R_NFW_0_lst**3*0.5*4.*np.pi*density_integral/self.m_1/self.m_2
        self.PBHPBH_Rate_per_halo = PBHPBH_Rate_per_halo.to(1./u.year)

        ### Analytical expression
        ### The C-dependent part
        tmp_C_term = (1. - 1./(1. + self.halo.C_lst)**3)/self.halo.g(self.halo.C_lst)**2

        ### Other Halo dependent parts
        tmp_halo_term = self.halo.M_halo**2*self.avg_vel_PBH_factor/self.halo.R_NFW_0_lst**3

        ### The numerical prefactor
        anal_prefac = (85.*np.pi/3.)**(2./7.)*const.G**2*const.c**(-10./7.)/6.
        #(85.*np.pi/12./np.sqrt(2.))**(2./7.)*9.*2.**(3./7.)*const.G**2*const.c**(-10./7.)

        ### PBH-PBH rate per halo
        PBHPBH_Rate_per_halo_anal = anal_prefac*tmp_C_term*tmp_halo_term
        ### In units of 1/yr
        self.PBHPBH_Rate_per_halo_anal = PBHPBH_Rate_per_halo_anal.to(1./u.year)

        ### PBH-NS merger rate per halo

        ### NFW-NS integral
        density_integral = np.trapz(self.halo.x_lst**2*self.halo.NFW_profiles*self.halo.ns_profiles, self.halo.x_lst, axis = 1)
        ### PBH-PBH rate per halo in units of 1/yr
        PBHNS_Rate_per_halo = self.sigma_v_avg*self.halo.rho_NFW_0_lst*self.halo.rho_ns_0_lst*\
                                            self.halo.R_NFW_0_lst**3*4.*np.pi*density_integral/self.m_1/self.m_2
        self.PBHNS_Rate_per_halo = PBHNS_Rate_per_halo.to(1./u.year)


        rate_per_halo_dictionary = {"PBHPBH_Rate_per_halo":self.PBHPBH_Rate_per_halo,
                                    "PBHPBH_Rate_per_halo_anal":self.PBHPBH_Rate_per_halo_anal,
                                    "PBHNS_Rate_per_halo":self.PBHNS_Rate_per_halo}

        return rate_per_halo_dictionary


    def integrated_rates(self, M_c_lst):

        rates_per_halo = self.rates_per_halo()

        PBHPBH_rate_lst = np.zeros(len(M_c_lst))
        PBHNS_rate_lst = np.zeros(len(M_c_lst))

        ### https://bdiemer.bitbucket.io/colossus/lss_mass_function.html
        halo_mass_function = mass_function.massFunction(self.halo.M_halo.value, 0., mdef = '200c', model = 'tinker08', q_out = 'dndlnM')

        for indx, val in enumerate(M_c_lst):
            PBHPBH_rate_y = ((1./0.7**3)*self.PBHPBH_Rate_per_halo.value*1e9*halo_mass_function)[self.halo.M_halo.value > val]
            PBHPBH_rate_x = (np.log(self.halo.M_halo.value))[self.halo.M_halo.value > val]

            PBHNS_rate_y = ((1./0.7**3)*self.PBHNS_Rate_per_halo.value*1e9*halo_mass_function)[self.halo.M_halo.value > val]
            PBHNS_rate_x = (np.log(self.halo.M_halo.value))[self.halo.M_halo.value > val]

            PBHPBH_rate_lst[indx] = np.trapz(PBHPBH_rate_y, PBHPBH_rate_x)
            PBHNS_rate_lst[indx] = np.trapz(PBHNS_rate_y, PBHNS_rate_x)

        return {"halo_mass_function":halo_mass_function, "PBHPBH_rate_lst":PBHPBH_rate_lst, "PBHNS_rate_lst":PBHNS_rate_lst}




class spike:

    def __init__(self, halo, gamma):
        self.halo = halo


        a = 8.12
        b = 4.24

        alpha_gamma_interp = np.array([0.00733, 0.120, 0.140, 0.142, 0.135, 0.122, 0.103, 0.0818, 0.017])
        gamma_interp = np.array([0.05, 0.2, 0.4,0.6, 0.8, 1.0, 1.2, 1.4, 2.])
        alpha_gamma_interpolation_function = interpolate.interp1d(gamma_interp, alpha_gamma_interp)

        if gamma <= min(gamma_interp) or gamma > max(gamma_interp):
            print("ERROR gamma = %5.3f is not supported." % gamma)
        else:
            self.alpha_gamma = alpha_gamma_interpolation_function(gamma)

        reference_vel = 200*u.km/u.s
        self.M_SMBH = (10**a*(self.halo.v_dm_lst/reference_vel)**b)*u.Msun

        self.R_schw = 2.*const.G*self.M_SMBH/const.c**2
        self.R_spike = self.alpha_gamma*self.halo.R_NFW_0_lst*(self.M_SMBH/self.halo.rho_NFW_0_lst/self.halo.R_NFW_0_lst**3)**(1./(3. - gamma))

        self.gamma_spike = (9. - 2.*gamma)/(4. - gamma)
        self.rho_spike_0 = self.halo.rho_NFW_0_lst*(self.R_spike/self.halo.R_NFW_0_lst)**(-gamma)



    def spike_properties(self):

        spike_dictionary = {"M_SMBH":self.M_SMBH,
                            "R_spike":self.R_spike,
                            "R_schw":self.R_schw,
                            "rho_spike_0":self.rho_spike_0,
                            "alpha_gamma":self.alpha_gamma}

        return spike_dictionary

    def spike_profiles(self):
        '''
            Returns the spike profiles without the rho_spike_0 prefactor and for all halos
        '''
        ratio = (self.R_schw/self.R_spike).to(u.m**0)
        length = len(ratio)

        ### For integration
        x_lst = np.geomspace(4.*ratio, np.ones(length), 100000).T
        spike_profiles = (1. - 4.*ratio.reshape(length, 1)/x_lst)**3*(1./x_lst)**self.gamma_spike

        ### For plotting
        x_lst_plotting = np.geomspace(4.*ratio, 1e7*np.ones(length), 10000).T
        spike_profiles_plotting = (1. - 4.*ratio.reshape(length, 1)/x_lst_plotting)**3*(1./x_lst_plotting)**self.gamma_spike

        return {"x_list":x_lst,
                "spike_profiles":spike_profiles,
                "x_list_plotting":x_lst_plotting,
                "spike_profiles_plotting":spike_profiles_plotting}
