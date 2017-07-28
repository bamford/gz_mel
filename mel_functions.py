from __future__ import print_function, division

import numpy as np
import pandas as pd

def rates_function(rates, rates_evol, t, t0, BD0=False):
    # this used to take pandas DataFrames,
    # but now requires numpy arrays for efficiency
    if rates_evol is None:
        # shortcut if no evolution
        return rates
    rates = rates + rates_evol * (t - t0)
    # ensure rates are valid and sensible
    rates = np.clip(rates, 0, 1)
    # * For a closed model the columns of the rates matrix must sum to zero
    #   so there is no creation/destruction of galaxies, this is enforced here.
    # * Creation/destruction of galaxies basically means they move in to or
    #   out of the sample given the selection criteria.
    # * Another possibility is merging, where one galaxies is 'destroyed' in
    #   each merger.
    # * If one wishes to create/destroy galaxies then that could perhaps be
    #   achieved by allowing rates to not sum to zero in each column.  However,
    #   the total numbers of galaxies moving between states are proportional to
    #   the number currently in the initial state, which probably doesn't make
    #   much sense for creation.  Alternatively, one could add an additional
    #   state(s) representing the pool(s) from which galaxies are
    #   created/destroyed (but this would be evolved itself).  This is done
    #   here. Another possibility might be to add additional terms in the
    #   differential equation to describe the creation/destruction of galaxies.
    # * Another issue is multiple selection criteria, e.g. mass bins.  These
    #   are essentially different states, and we can model transitions between
    #   them in the same manner: in a small time a galaxy in a given state will
    #   have a particular (potentially time dependent) probability of moving
    #   from one mass bin to the next
    #   (proportional to [SFR + merger accretion rate] / mass_bin_width).
    #   This is done here.
    # * In principle, the rates related to the destruction of galaxies in
    #   particular states due to merging and the mass growth of the merger
    #   remnant are related.  In practise this would be pretty complex to
    #   manage unless we make some strict assumptions.  For example, we could
    #   assume that only galaxies in the same state merge (hence ~doubling
    #   their mass) and thus require the destruction rate to be proportional to
    #   the mass bin transition rate (where the factor depends on the width of
    #   the mass bin).  If we get too complicated, then it would make more
    #   sense to simulate a large population of individual galaxies rather than
    #   consider a small set of states.
    di = np.diag_indices(len(rates))
    rates[di] = 0
    from_sum = rates.sum(0)
    rates[di] = -from_sum
    if BD0:
        rates[0, 0] = 0  # no changes to BD0
    return rates


def calc_F(N, Nerr=None):
    F = pd.DataFrame(index=N.index)
    Ferr = pd.DataFrame(index=N.index)
    F['$f_{R|D}$'] = N['$N_{RD}$'] / (N['$N_{RD}$'] + N['$N_{BD}$'])
    F['$f_{D|R}$'] = N['$N_{RD}$'] / (N['$N_{RD}$'] + N['$N_{RE}$'])
    if Nerr is not None:
        NDsum = N['$N_{RD}$'] + N['$N_{BD}$']
        dfRDdNRD = 1 / NDsum - N['$N_{RD}$'] / NDsum**2
        dfRDdNBD = - N['$N_{RD}$'] / NDsum**2
        Ferr['$f_{R|D}$'] = np.sqrt(dfRDdNRD**2 * Nerr['$N_{RD}$']**2 +
                                              dfRDdNBD**2 * Nerr['$N_{BD}$']**2)
        NRsum = N['$N_{RD}$'] + N['$N_{RE}$']
        dfRDdNRD = 1 / NRsum - N['$N_{RD}$'] / NRsum**2
        dfRDdNRE = - N['$N_{RD}$'] / NRsum**2
        Ferr['$f_{D|R}$'] = np.sqrt(dfRDdNRD**2 * Nerr['$N_{RD}$']**2 +
                                              dfRDdNRE**2 * Nerr['$N_{RE}$']**2)
        return F, Ferr
    else:
        return F


def calc_F_m(N, Nerr=None):
    F = pd.DataFrame(index=N.index)
    if Nerr is not None:
        Ferr = pd.DataFrame(index=N.index)
    for i in range(1, 5):
        m = str(i)
        F['$f_{R|D,' + m + '}$'] = N['$N_{RD' + m + '}$'] / (N['$N_{RD' + m + '}$'] + N['$N_{BD' + m + '}$'])
        F['$f_{D|R,' + m + '}$'] = N['$N_{RD' + m + '}$'] / (N['$N_{RD' + m + '}$'] + N['$N_{RE' + m + '}$'])
        if Nerr is not None:
            NDsum = N['$N_{RD' + m + '}$'] + N['$N_{BD' + m + '}$']
            dfRDdNRD = 1 / NDsum - N['$N_{RD' + m + '}$'] / NDsum**2
            dfRDdNBD = - N['$N_{RD' + m + '}$'] / NDsum**2
            Ferr['$f_{R|D,' + m + '}$'] = np.sqrt(dfRDdNRD**2 * Nerr['$N_{RD' + m + '}$']**2 +
                                                  dfRDdNBD**2 * Nerr['$N_{BD' + m + '}$']**2)
            NRsum = N['$N_{RD' + m + '}$'] + N['$N_{RE' + m + '}$']
            dfRDdNRD = 1 / NRsum - N['$N_{RD' + m + '}$'] / NRsum**2
            dfRDdNRE = - N['$N_{RD' + m + '}$'] / NRsum**2
            Ferr['$f_{D|R,' + m + '}$'] = np.sqrt(dfRDdNRD**2 * Nerr['$N_{RD' + m + '}$']**2 +
                                                  dfRDdNRE**2 * Nerr['$N_{RE' + m + '}$']**2)
    if Nerr is not None:
        return F, Ferr
    else:
        return F


def plot_N(N, Nerr, Nfit=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for j, col in enumerate(N):
        ax.errorbar(N.index, N[col], Nerr[col], label=col, color='C{}'.format(j))
        if Nfit is not None:
            ax.plot(N.index, Nfit[col], '--', color='C{}'.format(j))
        ax.set(xlabel='$z$', ylabel='$N(z)/N(z={:.1f})$'.format(z0),
              ylim=(0, 1), title='$\log(M/M_\odot) \sim {}$'.format(mdata[i-1]))
    ax.legend(loc=1, ncol=3)
    plt.tight_layout()


def plot_N_m(N, Nerr, Nfit=None):
    fig, axarr = plt.subplots(2, 2, figsize=(12, 10))
    axarr = axarr.flat
    for i in range(4):
        Nm = N.filter(regex=('.*{}.*'.format(i+1)))
        Nerrm = Nerr.filter(regex=('.*{}.*'.format(i+1)))
        if Nfit is not None:
            Nfitm = Nfit.filter(regex=('.*{}.*'.format(i+1)))
        for j, col in enumerate(Nm):
            axarr[i].errorbar(Nm.index, Nm[col], Nerrm[col], label=col, color='C{}'.format(j))
            if Nfit is not None:
                axarr[i].plot(Nfitm.index, Nfitm[col], '--', color='C{}'.format(j))
        axarr[i].set(xlabel='$z$', ylabel='$N(z)/N(z={:.1f})$'.format(z0),
                  ylim=(0, 0.35), title='$\log(M/M_\odot) \sim {}$'.format(mdata[i]))
        handles, labels = axarr[i].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axarr[i].legend(handles, labels, loc=1, ncol=3)
    plt.tight_layout()


def plot_F(F, Ferr, Ffit=None):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr = axarr.flat
    Fm = F.filter(regex=('.*R.D.*'))
    Ferrm = Ferr.filter(regex=('.*R.D.*'))
    if Ffit is not None:
        Ffitm = Ffit.filter(regex=('.*R.D.*'))
    for j, col in enumerate(Fm):
        axarr[0].errorbar(Fm.index, Fm[col], Ferrm[col], label=col, color='C{}'.format(j))
        if Ffit is not None:
            axarr[0].plot(Ffitm.index, Ffitm[col], '--', color='C{}'.format(j))
    axarr[0].set(xlabel='$z$', ylabel='$f_{R|D}(z)$',
              ylim=(0, 0.4))
    axarr[0].legend(loc=1, ncol=4)
    Fm = F.filter(regex=('.*D.R.*'))
    Ferrm = Ferr.filter(regex=('.*D.R.*'))
    if Ffit is not None:
        Ffitm = Ffit.filter(regex=('.*D.R.*'))
    for j, col in enumerate(Fm):
        axarr[1].errorbar(Fm.index, Fm[col], Ferrm[col], label=col, color='C{}'.format(j))
        if Ffit is not None:
            axarr[1].plot(Ffitm.index, Ffitm[col], '--', color='C{}'.format(j))
    axarr[1].set(xlabel='$z$', ylabel='$f_{D|R}(z)$',
              ylim=(0, 0.4))
    axarr[1].legend(loc=1, ncol=4)
    plt.tight_layout()


def plot_F_m(F, Ferr, Ffit=None):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr = axarr.flat
    Fm = F.filter(regex=('.*R.D,[1234].*'))
    Ferrm = Ferr.filter(regex=('.*R.D,[1234].*'))
    if Ffit is not None:
        Ffitm = Ffit.filter(regex=('.*R.D,[1234].*'))
    for j, col in enumerate(Fm):
        axarr[0].errorbar(Fm.index, Fm[col], Ferrm[col], label=col, color='C{}'.format(j))
        if Ffit is not None:
            axarr[0].plot(Ffitm.index, Ffitm[col], '--', color='C{}'.format(j))
    axarr[0].set(xlabel='$z$', ylabel='$f_{R|D}(z)$',
              ylim=(0, 0.4))
    handles, labels = axarr[0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axarr[0].legend(handles, labels, loc=1, ncol=4)
    Fm = F.filter(regex=('.*D.R,[1234].*'))
    Ferrm = Ferr.filter(regex=('.*D.R,[1234].*'))
    if Ffit is not None:
        Ffitm = Ffit.filter(regex=('.*D.R,[1234].*'))
    for j, col in enumerate(Fm):
        axarr[1].errorbar(Fm.index, Fm[col], Ferrm[col], label=col, color='C{}'.format(j))
        if Ffit is not None:
            axarr[1].plot(Ffitm.index, Ffitm[col], '--', color='C{}'.format(j))
    axarr[1].set(xlabel='$z$', ylabel='$f_{D|R}(z)$',
              ylim=(0, 0.4))
    handles, labels = axarr[1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axarr[1].legend(handles, labels, loc=1, ncol=4)
    plt.tight_layout()


def plot_NF(N, F):
    fig, (axN, axF) = plt.subplots(1, 2, figsize=(12, 6))
    N.plot(ax=axN)
    axN.set(xlabel='$z$', ylabel='$N(z)/N(z={:.1f})$'.format(zstart), ylim=(0, 1));
    F.plot(ax=axF)
    axF.set(xlabel='$z$', ylabel='$f(z)$', ylim=(0, 0.4));
    plt.tight_layout()


class logProbabilities:
    def __init__(self, fit_N0, fit_rates, fit_rates_evol, Nmask,
                 Ndata, Nerr, tdata, ratefn, N0_init, rates_init, rates_evol_init,
                 t0, N_labels, rate_labels, rate_evol_labels):
        # a boolean matrix describing which rates and evolutions to fit
        self.fit_N0 = fit_N0.copy()
        self.nFitN0 = self.fit_N0.values.sum()
        self.fit_rates = fit_rates.copy()
        self.nStates = len(fit_rates)
        self.nFitRates = self.fit_rates.values.sum()
        if rates_evol_init is not None:
            self.fit_rates_evol = fit_rates_evol.copy()
            self.nFitRatesEvol = self.fit_rates_evol.values.sum()
        else:
            self.fit_rates_evol = None
            self.nFitRatesEvol = 0
        self.pars = []
        self.pars.extend(N_labels.values[self.fit_N0.values])
        self.pars.extend(rate_labels.values[self.fit_rates.values])
        if rates_evol_init is not None:
            self.pars.extend(rate_evol_labels.values[self.fit_rates_evol.values])
        self.pars = pd.DataFrame(self.pars, columns=['parameter'])
        self.nPar = self.nFitN0 + self.nFitRates + self.nFitRatesEvol
        # the measured number density, Ndata,
        # in each state at time, tdata,
        # with its error, Nerr
        self.Ndata = Ndata.copy()
        self.Nerr = Nerr.copy()
        self.tdata = tdata.copy()
        # the current rate matrix is determined using ratefn
        # from N0, the number densities at t0; rates, the rates at t0; and rates_evol
        self.ratefn = ratefn
        self.N0 = N0_init.copy()
        self.rates = rates_init.copy()
        if rates_evol_init is not None:
            self.rates_evol = rates_evol_init.copy()
        else:
            self.rates_evol = None
        self.t0 = t0
        # Which states are actually measured
        self.Nmask = Nmask

    def split_pars(self, p):
        start, end = (0, self.nFitN0)
        p_N0 = p[start:end]
        start, end = (end, end + self.nFitRates)
        p_rates = p[start:end]
        start, end = (end, end + self.nFitRatesEvol)
        p_rates_evol = p[start:end]
        return p_N0, p_rates, p_rates_evol

    def evaluate(self, t, p, strip=True):
        # update the model quantities given the current parameters
        p_N0, p_rates, p_rates_evol = self.split_pars(p)
        self.N0.values[self.fit_N0.values] = p_N0
        self.rates.values[self.fit_rates.values] = p_rates
        if self.rates_evol is not None:
            self.rates_evol.values[self.fit_rates_evol.values] = p_rates_evol
        # compute N at each tdata, given current parameters
        N0 = self.N0.values
        rates = self.rates.values
        if self.rates_evol is None:
            rates_evol = None
        else:
            rates_evol = self.rates_evol.values
        N = odeint(model, N0, t,
                   args=(self.ratefn, rates, rates_evol, self.t0))
        if strip:
            # strip off unmeasured quantitites
            N = N[:, self.Nmask]
            # convert to normalised relative densities at each redshift
            N = (N.T / N.sum(1)).T
        return N

    def prior(self, inpars):
        inshape = inpars.shape
        inpars = inpars.reshape((-1, self.nPar))
        lnP = np.zeros(inpars.shape[0])
        for i, p in enumerate(inpars):
            p_N0, p_rates, p_rates_evol = self.split_pars(p)
            lnP[i] = stats.uniform(0, 1).logpdf(p_N0).sum()
            lnP[i] += stats.uniform(0, 1).logpdf(p_rates).sum()
            lnP[i] += norm_logpdf(p_rates_evol, 0, 0.1).sum()
        return lnP.reshape(inshape[:-1])

    def likelihood(self, inpars):
        inshape = inpars.shape
        inpars = inpars.reshape((-1, self.nPar))
        lnL = np.zeros(inpars.shape[0])
        for i, p in enumerate(inpars):
            N = self.evaluate(self.tdata, p)
            # calculate Likelihood of computed N given Ndata and Nerr
            lnL[i] = norm_logpdf(N, self.Ndata.values, self.Nerr.values).sum()
        return lnL.reshape(inshape[:-1])

    def posterior(self, inpars):
        lnL = self.likelihood(inpars)
        lnP = self.prior(inpars)
        return lnL + lnP

    def neg_posterior(self, inpars):
        return -self.posterior(inpars)

    def init_pars(self, shape):
        ip = np.zeros(shape + (self.nPar,))
        start, end = (0, self.nFitN0)
        ip[..., start:end] = stats.uniform.rvs(0, 1, size=shape + (self.nFitN0,))
        start, end = (end, end + self.nFitRates)
        ip[..., start:end] = stats.uniform.rvs(0, 1, size=shape + (self.nFitRates,))
        start, end = (end, end + self.nFitRatesEvol)
        ip[..., start:end] = stats.norm.rvs(0, 0.1, size=shape + (self.nFitRatesEvol,))
        return ip
