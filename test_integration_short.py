#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: научный рабочий
"""

from scipy.integrate import quad, dblquad
import numpy as np
import math

def voigt_asym(x, fwhm, asymmetry, Gausian_share):
    """ returns pseudo-voigt profile composed of Gaussian and Lorentzian,
         which would be normalized by unit area if symmetric
         The funtion as defined in Analyst: 10.1039/C8AN00710A"""
    x_distorted = x*(1 - np.exp(-(x)**2/(2*(2*fwhm)**2))*asymmetry*x/fwhm)
    Lor_asym = fwhm / (x_distorted**2+fwhm**2/4) / (2*np.pi)
    Gauss_asym = (4*np.log(2)/np.pi)**0.5/fwhm * np.exp(-(x_distorted**2*4*np.log(2))/fwhm**2)
    voigt_asym = (1-Gausian_share)*Lor_asym + Gausian_share*Gauss_asym
    return voigt_asym



def voigt_asym_math(x, fwhm, asymmetry, Gausian_share):
    """ same as voigt_asym, but without numpy,
    accepts only scalar values, not numpy arrays.
    This function is used in integration."""
    x_distorted = x*(1 - math.exp(-(x)**2/(2*(2*fwhm)**2))*asymmetry*x/fwhm)
    Lor_asym = fwhm / (x_distorted**2+fwhm**2/4) / (2*math.pi)
    Gauss_asym = (4*math.log(2)/math.pi)**0.5/fwhm * math.exp(-(x_distorted**2*4*math.log(2))/fwhm**2)
    voigt_asym = (1-Gausian_share)*Lor_asym + Gausian_share*Gauss_asym
    return voigt_asym



class MultiPeak ():
    """ Abstract spectral feature, with no x-axis defined
     Order of parameters in array:
     0:    x0 (default 0)
     1:    fwhm (defauld 1)
     2:    asymmetry (default 0)
     3:    Gaussian_share (default 0, i.e. Lorentzian peak)
     4:    voigt_amplitude (~area, not height)

     Asymmetric peaks calculated on x-asis (a grid of wavenumbers).
    It is possible to set a peak height,
        Changing fwhm keeps area same, while changes height.
        Changing height changes area while keeps fwhm.
    """

    def __init__(self, wn=np.linspace(0, 1, 129), number_of_peaks=1) :
        self.specs_array = np.zeros((number_of_peaks, 5))
        self.specs_array[:, 1] = 1 # set default fwhm to 1. Otherwise we can get division by 0
        self.specs_array[:, 0] = (wn[-1]-wn[0])/2
        self.wn = wn
        self.number_of_peaks = number_of_peaks
        self.linear_baseline_scalarpart = np.zeros_like(wn)
        self.linear_baseline_slopepart = np.zeros_like(wn)
        # self.baseline = np.zeros_like(wn)
        self.d2baseline = np.zeros_like(wn)
        # self.linear_baseline_offset = 0
        # self.linear_baseline_slope = 0
        
    @property
    def position(self):
        return self.specs_array[:, 0]
    @position.setter
    def position (self, position) :
        self.specs_array[:, 0] = position

    @property
    def fwhm(self):
        return self.specs_array[:, 1]
    @fwhm.setter
    def fwhm (self, fwhm) :
        self.specs_array[:, 1] = fwhm
    
    @property
    def asymmetry(self):
        return self.specs_array[:, 2]
    @asymmetry.setter
    def asymmetry (self, asymmetry) :
        self.specs_array[:, 2] = asymmetry

    @property
    def Gaussian_share(self):
        return self.specs_array[:, 3]
    @Gaussian_share.setter
    def Gaussian_share (self, Gaussian_share) :
        self.specs_array[:, 3] = Gaussian_share

    @property
    def voigt_amplitude(self):
        return self.specs_array[:, 4]
    @voigt_amplitude.setter
    def voigt_amplitude (self, voigt_amplitude) :
        self.specs_array[:, 4] = voigt_amplitude[:]

    @property
    def peak_area (self) :
        peak_area = (1 - self.Gaussian_share) * self.voigt_amplitude * (1 + 0.69*self.asymmetry**2 + 1.35 * self.asymmetry**4) + self.Gaussian_share * self.voigt_amplitude * (1 + 0.67*self.asymmetry**2 + 3.43*self.asymmetry**4)
        return peak_area

    @property
    def peak_height (self):
        #@Test&Debug # # print('calling getter')
        return self.specs_array[:, 3] * self.specs_array[:, 4]*(4*np.log(2)/np.pi)**0.5 / self.specs_array[:, 1] + (1-self.specs_array[:, 3]) * self.specs_array[:, 4]*2/(np.pi*self.specs_array[:, 1])
    @peak_height.setter # works only with array, not with a single element!
    def peak_height (self, peak_height):
        #@Test&Debug # # print('calling setter')
        self.specs_array[:, 4] = peak_height[:] / (
            self.specs_array[:, 3] * (4*np.log(2)/np.pi)**0.5 / self.specs_array[:, 1] + (1-self.specs_array[:, 3]) * 2/(np.pi*self.specs_array[:, 1])
            )

    @property
    def fwhm_asym (self) :
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.fwhm * (1 + 0.4*self.asymmetry**2 + 1.35*self.asymmetry**4)
        return fwhm_asym

    @property
    def curve(self):
        """ Asymmetric pseudo-Voigt funtion as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = np.zeros((len(self.wn), self.number_of_peaks))
        for i in range(self.number_of_peaks):
            curve[:,i] = self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
        curve = np.sum(curve, axis=1)
        return curve

    @property
    def multicurve (self) :
        """ array of curves
        """
        multicurve = np.zeros((len(self.wn), self.number_of_peaks))
        for i in range(self.number_of_peaks):
            multicurve[:,i] = self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
        return multicurve



    def curve_for_integration(self, x):
        """ x here is a scalar value
        """
        curve_at_point_x = 0
        for i in range(self.number_of_peaks):
            curve_at_point_x += self.voigt_amplitude[i] * voigt_asym_math(x-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
        return curve_at_point_x

    def integrate_alpha0_double(self, lam=1e6):
        """ double integration test, 2023-01-23"""
        integration_limits = [min(self.wn), max(self.wn)]
        integration_limits[0] -= 0.5 * abs(min(self.wn) - max(self.wn))
        integration_limits[1] += 0.5 * abs(min(self.wn) - max(self.wn))
        def double_curve_mod(x, y):
            lam_sc = 2**0.5 * lam**0.25
            return self.curve_for_integration(x) * self.curve_for_integration(y) * ( math.exp(-abs(x-y)/lam_sc) *
                                              (math.cos(abs(x-y)/lam_sc) + math.sin(abs(x-y)/lam_sc)) / (2*lam_sc) )
        doubleintegral = dblquad(double_curve_mod, integration_limits[0], integration_limits[1], integration_limits[0], integration_limits[1])
        print('lam = {:.0e}, doubleintegral = {:.4e}, precision ~ {:.4e}'.format(lam, doubleintegral[0], doubleintegral[1]) )
        return doubleintegral[0]
    
    def integrate_F_square(self):
        """ double integration test, 2023-01-23"""
        integration_limits = [min(self.wn), max(self.wn)]
        integration_limits[0] -= 0.5 * abs(min(self.wn) - max(self.wn))
        integration_limits[1] += 0.5 * abs(min(self.wn) - max(self.wn))
        def double_curve_mod(y):
            return self.curve_for_integration(y)**2
        singleintegral = quad(double_curve_mod, integration_limits[0], integration_limits[1])
        print('singleintegral = {:.4e}, precision ~ {:.4e}'.format(singleintegral[0], singleintegral[1]) )
        return singleintegral[0]
    





if __name__ == '__main__':

    Lorentz_positions = (384, 720)
    Lorentz_FWHMs = (32, 64)
    amplitudes0 = (128*2, 128*8)
    wavenumber = np.linspace(0, 1024, num=1025)
    
    # now let's make it in class:
    dermultipeak = MultiPeak(wavenumber, 2)
    dermultipeak.fwhm = Lorentz_FWHMs
    dermultipeak.position = Lorentz_positions
    dermultipeak.voigt_amplitude = amplitudes0
    doubleintegral_from_class3 = dermultipeak.integrate_alpha0_double(1e3)
    doubleintegral_from_class4 = dermultipeak.integrate_alpha0_double(1e4)
    doubleintegral_from_class5 = dermultipeak.integrate_alpha0_double(1e5)
    doubleintegral_from_class6 = dermultipeak.integrate_alpha0_double(1e6)
    doubleintegral_from_class7 = dermultipeak.integrate_alpha0_double(1e7)
    integrate_F_square_all = dermultipeak.integrate_F_square()

    print('\n now differences: \n')    
    print('{:.4e}'.format(integrate_F_square_all - doubleintegral_from_class3))
    print('{:.4e}'.format(integrate_F_square_all - doubleintegral_from_class4))
    print('{:.4e}'.format(integrate_F_square_all - doubleintegral_from_class5))
    print('{:.4e}'.format(integrate_F_square_all - doubleintegral_from_class6))
    print('{:.4e}'.format(integrate_F_square_all - doubleintegral_from_class7))

