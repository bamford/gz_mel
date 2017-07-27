from __future__ import print_function, division
import numpy as np
from scipy import special


_norm_pdf_C = np.sqrt(2 * np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def norm_pdf(x, loc=0, scale=1):
    y = (x - loc) / scale
    z = np.exp(-y**2 / 2.0) / _norm_pdf_C
    return z / scale


def norm_logpdf(x, loc=0, scale=1):
    y = (x - loc) / scale
    z = -y**2 / 2.0 - _norm_pdf_logC
    return z - np.log(scale)


def gamma_logpdf(x, a, loc=0, scale=1):
    x = (x - loc) / scale
    z = special.xlogy(a - 1.0, x) - x - special.gammaln(a)
    z -= np.log(scale)
    z = np.where(x < 0, -np.inf, z)
    if z.ndim == 0:
        return z[()]
    return z


def beta_logpdf(x, a, b, loc=0, scale=1):
    x = (x - loc) / scale
    z = special.xlog1py(b - 1.0, -x) + special.xlogy(a - 1.0, x)
    z -= special.betaln(a, b)
    z -= np.log(scale)
    z = np.where((x < 0) | (x > 1), -np.inf, z)
    if z.ndim == 0:
        return z[()]
    return z
