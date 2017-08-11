from __future__ import print_function, division
import sys
import time
import warnings
import numpy as np
from numpy import ma
from emcee import PTSampler, autocorr
import nestle
from multiprocess import Pool


# Suppress warnings
warnings.filterwarnings('ignore', 'converting a masked element to nan')


def flatten_chain(c):
    return c.reshape((np.product(c.shape[:-1]), c.shape[-1]))


def flatten_without_burn(sampler, nburn, itemp=0):
    c = sampler.chain
    if c.ndim == 4:
        c = c[itemp]
    c = c[:, nburn:]
    return flatten_chain(c)


def weight_without_burn(sampler, nburn, itemp=0):
    c = sampler.lnprobability
    if c.ndim == 3:
        c = c[itemp]
    c = c[:, nburn:]
    w = np.exp(c.reshape(np.product(c.shape)))
    return w / w.sum()


def lnprobability_without_burn(sampler, nburn, itemp=0):
    c = sampler.lnprobability
    if c.ndim == 3:
        c = c[itemp]
    c = c[:, nburn:]
    w = c.reshape(np.product(c.shape))
    return w


def lnlikelihood_without_burn(sampler, nburn, itemp=0):
    c = sampler.lnlikelihood
    if c.ndim == 3:
        c = c[itemp]
    c = c[:, nburn:]
    w = np.exp(c.reshape(np.product(c.shape)))
    return w / w.sum()


def expand_ranges(ranges, fraction):
    ranges = ranges.copy()
    range_widths = ranges[:, 1] - ranges[:, 0]
    ranges[:, 0] -= fraction * range_widths
    ranges[:, 1] += fraction * range_widths
    return ranges


def autocor_checks(sampler, nburn, itemp=0, outfile=None):
    print('Chains contain {} samples'.format(sampler.chain.shape[-2]),
          file=outfile)
    print('Specified burn-in is {} samples'.format(nburn), file=outfile)
    a_exp = sampler.acor[0]
    a_int = np.max([autocorr.integrated_time(sampler.chain[itemp, i, nburn:])
                    for i in range(sampler.chain.shape[1])], 0)
    a_exp = max(a_exp)
    a_int = max(a_int)
    print('A reasonable burn-in should be around '
          '{:d} steps'.format(int(10 * a_exp)), file=outfile)
    print('After burn-in, each chain produces one independent '
          'sample per {:d} steps'.format(int(a_int)), file=outfile)
    return a_exp, a_int


def check_acc_frac(sampler, outfile=None):
    acc_frac = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction: {0:.3f}".format(acc_frac), file=outfile)
    return acc_frac


def round_sig(x, sig=1):
    try:
        d = sig - int(np.floor(np.log10(x))) - 1
    except ValueError:
        d = 3
    d = max(0, d)
    return np.round(x, d), d


def print_parameters(pval, perr, par, truths=None, outfile=None):
    for i, p in enumerate(par):
        err, dp = round_sig(perr[i], 1)
        val = round(pval[i], dp)
        post = 'e}' if np.abs(np.log10(np.abs(val))) > 3 else 'f}'
        dps = str(dp) + post
        outstr = ('{:16s} = {:8.' + dps +
                  ' +- {:<8.' + dps).format(p, val, err)
        if truths is not None:
            dpt = str(dp + 1) + post
            outstr += ('   ({:8.' + dpt + ')').format(truths[i])
        print(outstr, file=outfile)


def summary(samples, par, truths=None, outfile=None):
    mean = samples.mean(0)
    sigma = samples.std(0)
    outstr = print_parameters(mean, sigma, par, truths, outfile)
    return mean, sigma


def log_evidence(sampler, nburn, outfile=None):
    logls = sampler.lnlikelihood[:, :, nburn:]
    logls = ma.masked_array(logls, mask=logls == -np.inf)
    mean_logls = logls.mean(axis=-1).mean(axis=-1)
    logZ = -np.trapz(mean_logls, sampler.betas)
    logZ2 = -np.trapz(mean_logls[::2], sampler.betas[::2])
    logZerr = abs(logZ2 - logZ)
    print('estimated evidence = {} +- {}'.format(logZ, logZerr), file=outfile)
    return logZ, logZerr


def check_init_pars(logl, logp, p0, outfile=None):
    ok = True
    p0 = p0.reshape((1, -1, p0.shape[-1]))
    prior0bad = (logp(p0) == -np.inf).sum() / np.product(p0.shape)
    if prior0bad > 0:
        ok = False
        print('Warning: {:.2f}% of initial parameters '
              'have zero prior'.format(prior0bad * 100), file=outfile)
    lnfunc0bad = (logl(p0) == -np.inf).sum() / np.product(p0.shape)
    if lnfunc0bad > 0:
        ok = False
        print('Warning: {:.2f}% of initial parameters '
              'have zero likelihood'.format(lnfunc0bad * 100), file=outfile)
    return ok


def run_emcee(logl, logp, p0func,
              ntemps=0, nwalkers=50, nsamples=2500, thin=1,
              minlogbeta=None, nupdates=10,
              threads=1, outfilename=None, saveall=True,
              **kwargs):
    if minlogbeta is None:
        if ntemps == 0:
            # use cunning ladder
            betas = np.concatenate((np.linspace(0, -0.9375, 16),
                                    np.linspace(-1, -1.875, 8),
                                    np.linspace(-2, -3.75, 8),
                                    np.linspace(-4, -7.5, 8),
                                    np.linspace(-8, -15, 8),
                                    np.linspace(-16, -30, 8),
                                    np.linspace(-32, -56, 4)))
            betas = 10**(np.sort(betas)[::-1])
        else:
            betas = None  # use emcee default
    else:
        betas = np.logspace(0, minlogbeta, ntemps)
    if betas is not None:
        ntemps = len(betas)
    pos = p0func((ntemps, nwalkers))
    if not check_init_pars(logl, logp, pos):
        return None
    ndim = pos.shape[-1]
    if threads > 1:
        pool = Pool(threads)
    else:
        pool = None
    sampler = PTSampler(ntemps, nwalkers, ndim, logl, logp,
                        betas=betas, pool=pool)
    if nupdates > 0:
        start = time.clock()
        print('Steps:', end='')
        sys.stdout.flush()
        nsteps = nsamples // nupdates
        for i in range(nupdates):
            pos, lnprob, rstate = sampler.run_mcmc(pos, nsteps, thin=thin)
            print(' {}'.format((i + 1) * nsteps), end='')
            sys.stdout.flush()
    nsteps = nsamples - sampler.chain.shape[-2]
    if nsteps > 0:
        pos, lnprob, rstate = sampler.run_mcmc(pos, nsteps, thin=thin)
    if nupdates > 0:
        print('\nTime taken = {:.2f} secs'.format(time.clock() - start))
    if outfilename is None:
        outfilename = 'emcee_sampler.npz'
    try:
        acor = sampler.acor
    except autocorr.AutocorrError:
        acor = None
    if saveall:
        np.savez_compressed(
            outfilename,
            acceptance_fraction=sampler.acceptance_fraction,
            acor=acor, beta=sampler.betas, chain=sampler.chain,
            lnlikelihood=sampler.lnlikelihood,
            lnprobability=sampler.lnprobability,
            tswap_acceptance_fraction=sampler.tswap_acceptance_fraction)
    else:
        # only save lowest temperature and thin samples by factor of ten
        np.savez_compressed(
            outfilename,
            acceptance_fraction=sampler.acceptance_fraction,
            acor=acor, beta=sampler.betas,
            chain=sampler.chain[0, :, ::10],
            lnlikelihood=sampler.lnlikelihood[0, :, ::10],
            lnprobability=sampler.lnprobability[0, :, ::10],
            tswap_acceptance_fraction=sampler.tswap_acceptance_fraction)
    return sampler


def run_nestle(logl, logp, ranges, npoints=1000, method='single'):
    def logprob(p):
        return logl(p) + logp(p)

    def prior_transform(p):
        return (ranges[:, 1] - ranges[:, 0]) * p + ranges[:, 0]

    return nestle.sample(logprob, prior_transform, len(ranges),
                         method=method, npoints=npoints)


class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, "w")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def write(self, message):
        sys.stdout.write(message)
        self.file.write(message)


def save_emcee(filename, sample, data={}, metadata={}):
    # This is a new (untested) approach to saving the emcee sampler
    # and associated data, rather than simply np.savez.
    # Using dill is another option.
    # This would be best called from run_fit.run, where it has access to
    # the required metadata, rather than from convenience.run_emcee.
    h5file = h5py.File(filename, 'w')

    arrays = dict(acceptance_fraction=sampler.acceptance_fraction,
                  acor=sampler.acor, beta=sampler.betas, chain=sampler.chain,
                  lnlikelihood=sampler.lnlikelihood,
                  lnprobability=sampler.lnprobability,
                  tswap_acceptance_fraction=sampler.tswap_acceptance_fraction)
    arrays.update(data)

    for key, item in arrays.items():
        h5file.create_dataset(key, data=item, compression="gzip", shuffle=True)

    for key, item in metadata.items():
        h5file.attrs[key] = item
