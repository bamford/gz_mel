from __future__ import print_function, division
import warnings
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from palettable.colorbrewer.qualitative import Set3_12 as palette
import corner
from emcee import autocorr
import nestle
from attrdict import AttrDict

from convenience import (flatten_without_burn, lnprobability_without_burn,
                          summary, log_evidence, autocor_checks,
                          check_acc_frac, Tee)


# better-looking plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 10


# Suppress warnings
warnings.filterwarnings('ignore', 'tight_layout')
warnings.filterwarnings('ignore', 'converting a masked element to nan')


def plot_prior(ax, func, range, label, args=[]):
    x = np.linspace(range[0], range[1], 1000)
    lnprior = func(x, *args)
    ax.plot(x, lnprior)
    ax.set_xlim(range[0], range[1])
    ax.set_xlabel(label)
    ax.set_ylabel('ln(prior)')
    ax.hlines(lnprior.max(), range[0], range[1], linestyle='dotted')
    ax.hlines(lnprior.max() - 2, range[0], range[1], linestyle='dashed')


def plot_chain(sampler, par, nburn=None, itemp=0, outfile=None):
    nwalkers, nsamp = sampler.chain.shape[1:3]
    if nburn is None:
        nburn = nsamp // 10
    nrow = int(np.ceil(len(par) / 2.0))
    if len(par) == 1:
        ncol = 1
        plt.figure(figsize=(10, 5))
    else:
        ncol = 2
    samplestep = max(nsamp // 100, 1)
    for i, p in enumerate(par):
        plt.subplot(nrow, ncol, i + 1)
        for w in range(0, nwalkers, 2):
            # only show half of walkers and thin to 100-200 samples
            plt.plot(np.arange(len(sampler.chain[itemp, 0, :, 0]),
                               step=samplestep),
                     sampler.chain[itemp, w, ::samplestep, i], 'r-',
                     alpha=1.0 / nwalkers)
        plt.ylabel(p)
        plt.xlabel('sample number')
        aymin, aymax = plt.ylim()
        plt.vlines(nburn, aymin, aymax, linestyle=':')
        plt.ylim(aymin, aymax)
    plt.suptitle('chain itemp={}'.format(itemp))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if outfile is not None:
        plt.savefig(outfile)


def plot_hist(sampler, par, nburn=None, weights=None, outfile=None):
    if nburn is None:
        nburn = sampler.chain.shape[2] // 10
    nrow = int(np.ceil(len(par) / 2.0))
    if len(par) == 1:
        ncol = 1
        plt.figure(figsize=(10, 5))
    else:
        ncol = 2
    for i, p in enumerate(par):
        plt.subplot(nrow, ncol, i + 1)
        plt.hist(flatten_without_burn(sampler, nburn)[:, i], weights=weights,
                 bins=100, histtype='stepfilled', alpha=0.75)
        plt.xlabel(p)
    plt.suptitle('histograms')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if outfile is not None:
        plt.savefig(outfile)


def plot_func(sampler, model, xmin, xmax, xdata, ydata, yerror,
              nburn=None, outfile=None, model_pars=[]):
    if nburn is None:
        nburn = sampler.chain.shape[2] // 10
    xchain = np.arange(xmin, xmax, 1.0)
    ychain = [model(xchain, p, *model_pars)
              for p in flatten_without_burn(sampler, nburn)]
    plt.errorbar(xdata, ydata, yerror, None, 'o')
    # plot a subset of the samples
    for i, y in enumerate(ychain[::100]):
        plt.plot(xchain, y, '-', alpha=0.1, c='green')
    plt.xlim(xmin, xmax)
    plt.suptitle('function samples')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if outfile is not None:
        plt.savefig(outfile)


def plot_triangle(samples, par, model=None, xdata=None, ydata=None, yerror=None,
                  xlabel='', ylabel='', weights=None, truths=None,
                  itemp=0, outfile=None, model_pars=[]):
    if len(par) == 1:
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 2, 1)
        ax.hist(samples[:, 0], bins=100, histtype='stepfilled', alpha=0.75)
        ax.set_xlabel(ylabel)
        ax.set_ylabel('frequency')
        ax = plt.subplot(1, 2, 2)
    else:
        corner.corner(samples, labels=par, truths=truths)
        # n = 2 if (len(par) % 2 == 0) else 3
        n = 3
        ax = plt.subplot(n, n, n)
    if model is None:
        ax.axis('off')
    else:
        # plot a subset of the samples
        xchain = np.linspace(xdata[0], xdata[-1], 100)
        idx = np.random.choice(len(samples), size=20, replace=False)
        ychain = [model(xchain, p, *model_pars) for p in samples[idx]]
        for y in ychain:
            ax.plot(xchain, y, 'r-', alpha=min(10.0 / len(ychain), 0.5))
        # plot the "average" solution of the samples
        ymean = model(xchain, samples.mean(0), *model_pars)
        ax.plot(xchain, ymean, 'b-', lw=3, alpha=0.75)
        if weights is not None:
            # plot the "maximum probability" solution of the samples
            best = weights.argmax()
            ybest = model(xchain, samples[best], *model_pars)
            ax.plot(xchain, ybest, 'g--', lw=3, alpha=0.75)
        if len(ydata.shape) > 1:
            for c in ydata:
                ax.errorbar(xdata.value, ydata[c].values, yerror[c].values, None, 'ok')
        else:
            ax.errorbar(xdata, ydata, yerror, None, 'ok')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_label_coords(0.5, -0.22)
        ax.yaxis.set_label_coords(-0.22, 0.5)
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        [l.set_rotation(45) for l in ax.get_yticklabels()]
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if len(par) == 1:
        plt.tight_layout()
    plt.suptitle('triangle')
    if outfile is not None:
        plt.savefig(outfile)


def check_betas(sampler, nburn=500, outfile=None):
    logls = sampler.lnlikelihood[:, :, nburn:]
    logls = ma.masked_array(logls, mask=logls == -np.inf)
    mean_logls = logls.mean(axis=-1).mean(axis=-1)
    fig, [ax1, ax2] = plt.subplots(2)
    ax1.semilogx(sampler.betas, mean_logls, "-o")
    ax1.set_xlabel('beta')
    ax1.set_ylabel('<log(L)>')
    integral = np.zeros_like(sampler.betas)
    for i, b in enumerate(sampler.betas[1:]):
        j = i+1
        integral[j] = -np.trapz(mean_logls[:j], sampler.betas[:j])
    ax2.semilogx(sampler.betas[1:], integral[1:], "-o")
    ax2.set_xlabel('beta')
    ax2.set_ylabel('integral(<log(L)>) using > beta')
    plt.suptitle('check_betas')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if outfile is not None:
        plt.savefig(outfile)
    # return the betas index below which 50%, 90% of the integral is found
    integral /= integral[-1]
    integral.sort()
    idx50 = np.searchsorted(integral, 0.5)
    idx90 = np.searchsorted(integral, 0.9)
    return idx50, idx90


def plot_autocorr(sampler, nburn, itemp=0, outfile=None):
    nwalkers = sampler.chain.shape[1]

    samples_before = sampler.chain[itemp, :, :nburn]
    samples_after = sampler.chain[itemp, :, nburn:]

    a_before = [autocorr.function(samples_before[i]) for i in range(nwalkers)]
    a_int_before = max(np.max([autocorr.integrated_time(samples_before[i])
                               for i in range(nwalkers)], 0))

    fig, [ax1, ax2] = plt.subplots(2)
    for a in a_before:
        ax1.plot(a[:200], "k", alpha=0.1)
    ax1.axhline(0, color="k")
    ax1.set_xlim(0, 200)
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"Autocorrelation during burn-in")
    ax1.text(0.9, 0.9, '{}'.format(a_int_before), horizontalalignment='right',
             verticalalignment='top', transform=ax1.transAxes)

    a_after = [autocorr.function(samples_after[i]) for i in range(nwalkers)]
    a_int_after = max(np.max([autocorr.integrated_time(samples_after[i])
                              for i in range(nwalkers)], 0))

    for a in a_after:
        ax2.plot(a[:200], "k", alpha=0.1)
    ax2.axhline(0, color="k")
    ax2.set_xlim(0, 200)
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(r"Autocorrelation after burn-in")
    ax2.text(0.9, 0.9, '{}'.format(a_int_after), horizontalalignment='right',
             verticalalignment='top', transform=ax2.transAxes)

    plt.suptitle('autocorrelation')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if outfile is not None:
        plt.savefig(outfile)


def plot_convergence(sampler, itemp=0, outfile=None):
    niter = sampler.chain.shape[2]
    iterno = np.arange(1, niter + 1)
    mean_over_walkers = sampler.chain[itemp].mean(axis=0)
    mean_vs_iteration = (np.cumsum(mean_over_walkers, axis=0).T / iterno).T
    stdev_over_walkers = sampler.chain[itemp].std(axis=0)
    stdev_vs_iteration = (np.cumsum(stdev_over_walkers, axis=0).T / iterno).T
    mean_track = ((mean_vs_iteration - mean_vs_iteration[-1]) /
                  stdev_vs_iteration[-1])
    stdev_track = (stdev_vs_iteration / stdev_vs_iteration[-1]) - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    # only plot every tenth iteration
    ax.set_color_cycle(palette.mpl_colors)
    ax.plot(iterno[::10], mean_track[::10], linestyle='-')
    ax.set_color_cycle(palette.mpl_colors)
    ax.plot(iterno[::10], stdev_track[::10], linestyle='--')
    ax.set_xlabel('sample number')
    ax.set_ylabel("Convergence criteria")
    plt.suptitle('convergence')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if outfile is not None:
        plt.savefig(outfile)


def print_emcee(sampler, par, model=None, x=None, y=None, yerror=None, nburn=0,
                xlabel='', ylabel='', truths=None,
                outfilename=None, **kwargs):
    stats = {}
    ntemp = sampler.chain.shape[0]
    if outfilename is None:
        outlog = 'print_emcee.log'
    else:
        outlog = outfilename + '.log'
    with Tee(outlog) as outfile:
        samples = flatten_without_burn(sampler, nburn)
        lnprob = lnprobability_without_burn(sampler, nburn)
        stats['mean'], stats['sigma'] = summary(samples,
                                                par, truths=truths,
                                                outfile=outfile)
        if ntemp > 3:
            stats['logz'], stats['logzerr'] = log_evidence(sampler, nburn,
                                                           outfile=outfile)
        try:
            stats['a_exp'], stats['a_int'] = autocor_checks(sampler, nburn,
                                                            outfile=outfile)
        except autocorr.AutocorrError:
            stats['a_exp'], stats['a_int'] = (None, None)
        stats['acc_frac'] = check_acc_frac(sampler, outfile=outfile)
        print(file=outfile)
        if outfilename is not None:
            pdf = PdfPages(outfilename + '.pdf')
            def page():
                pdf.savefig()
                plt.close()
        else:
            pdf = None
            def page(title=None):
                pass
        plot_triangle(samples, par, model, x, y, yerror,
                      xlabel, ylabel, weights=lnprob,
                      truths=truths)
        page()
        if ntemp > 3:
            itemp50, itemp90 = check_betas(sampler, nburn)
            page()
        else:
            itemp50, itemp90 = (0, 0)
        stats.update(itemp50=itemp50, itemp90=itemp90)
        for itemp in np.unique((0, itemp50, itemp90)):
            plt.figure()
            plot_chain(sampler, par, nburn, itemp=itemp)
            page()
            try:
                plot_autocorr(sampler, nburn, itemp=itemp)
            except:
                pass
            else:
                page()
            plot_convergence(sampler, itemp=itemp)
            page()
        if pdf is not None:
            pdf.close()
    return stats


def print_nestle(res, par, model, x, y, yerror, outfile=None):
    print(res.summary)
    print(nestle.mean_and_cov(res.samples, res.weights))
    plot_triangle(res.samples, par, model, x, y, yerror, weights=res.weights)
    if outfile is not None:
        plt.savefig(outfile)
