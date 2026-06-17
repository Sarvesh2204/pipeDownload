# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 01:47:13 2025

@author: Alexis Brandeker, alexis@astro.su.se
"""
import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def next_period(BJD, BJD0, P0):
    """Computes the next periodic phenomena (as transit, eclipse)
    after BJD, as a function of initial BJD0 and period P0.
    """
    return BJD + P0 - np.mod(BJD-BJD0, P0)


def find_orbits(BJD):
    """Method to find orbits. Returns indices of location of orbit starts,
    defined to be first observation after gap.
    """
    P0 = 99.05/60/24 # CHEOPS orbit duration in days
    n_orb = int((BJD[-1]-BJD[0])/P0)
    d = np.diff(BJD)
    s = np.sort(d)
    ind = []

    def find_inds():
        """ Find indices of first measurement of each orbit, as
        defined by gaps (if existent)
        """
        dmin = np.nanmedian(s[:-n_orb])
        dmax = np.nanmedian(s[-n_orb:])
        lev = dmin + 0.2*(dmax-dmin)
        inds = np.where(d > lev)[0] + 1
        jumpsBJD = BJD[inds]
        chi2 = np.zeros_like(jumpsBJD)

        for m, BJD0 in enumerate(jumpsBJD):
            guess = np.zeros(n_orb)
            BJD_first = next_period(BJD[0], BJD0, P0)
            guess = np.linspace(BJD_first, BJD_first+P0*(n_orb-1), n_orb)
            for g in guess:
                chi2[m] += np.nanmin((jumpsBJD-g)**2)
        
        BJD0 = jumpsBJD[chi2 == np.nanmin(chi2)]
        BJD_first = next_period(BJD[0], BJD0, P0)
        if BJD_first+P0*n_orb < BJD[-1]:
            guess = np.linspace(BJD_first, BJD_first+P0*n_orb, n_orb+1)
        else:
            guess = np.linspace(BJD_first, BJD_first+P0*(n_orb-1), n_orb)
        sel = closest_match(guess, jumpsBJD)
        ret_ind = []
        for n in range(len(sel)):
            if np.abs(BJD[inds[sel[n]]]-guess[n]) > 0.6*P0:
                min_dBJD = np.abs(BJD-guess[n])
                ret_ind.append(np.where(min_dBJD == np.nanmin(min_dBJD))[0][0])
            else:
                ret_ind.append(inds[sel[n]])
        return ret_ind
    
    def closest_match(guess, exact):
        """Pick the entries in "exact" that best match
        the entries in "guess", and return the selection
        """
        sel = []
        for g in guess:
            dist = np.abs(exact-g)
            sel.append(np.where(dist==np.nanmin(dist))[0][0])
        return sel

    if np.nanmax(d) > 3*np.nanmedian(d):
        return find_inds()
    else:
        orb_BJD = BJD[0]
        for n in range(n_orb):
            orb_BJD += P0
            min_dBJD = np.abs(BJD-orb_BJD)
            ind.append(np.where(min_dBJD == np.nanmin(min_dBJD))[0][0])
    return ind


def deroll(roll, bjd, flux, sel, tdens=20):
    """Returns de-trending array.
    roll - array of roll values (in degrees)
    bjd - time array in units of days
    flux - the flux array to be spline de-rolled
    sel - binary selection array (array of bools). True means datum is used.
    tdens - the distance between spline knot points, in degrees
    """
    ind = find_orbits(bjd)
    if ind[-1] < len(flux):
        ind.append(len(flux))
    ind0 = 0
    nflux = np.ones_like(flux)
    for n,ind1 in enumerate(ind):
        nflux[ind0:ind1] = flux[ind0:ind1] / np.median(flux[ind0:ind1][sel[ind0:ind1]])
        ind0 = ind1    
    srt_ind = np.argsort(roll[sel])
    dat_len = len(srt_ind)
    rolls = np.zeros(3*dat_len)
    nf = np.zeros(3*dat_len)
    rolls[:dat_len] = roll[sel][srt_ind] - 360
    rolls[dat_len:(2*dat_len)] = roll[sel][srt_ind]
    rolls[(2*dat_len):] = roll[sel][srt_ind] + 360
    nf[:dat_len] = nflux[sel][srt_ind]
    nf[dat_len:(2*dat_len)] = nflux[sel][srt_ind]
    nf[(2*dat_len):] = nflux[sel][srt_ind]    
    zi = np.where(np.abs(rolls) == np.min(np.abs(rolls)))[0]
    ti = np.arange(zi-tdens, zi+dat_len+tdens, tdens)
    t = rolls[ti]

    ind = np.ones(len(rolls), dtype=bool)
    for n in range(3):
        spl = LSQUnivariateSpline(rolls[ind], nf[ind], t)
        q = nf/spl(np.mod(rolls, 360))
        s = np.std(q)
        ind = np.abs(q/np.median(q)-1) < 3*s
    
    return spl(np.mod(roll, 360))

    
        