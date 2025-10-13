import numpy as np
import scipy.optimize

def upperRoot(x, moment_2, moment_3, pfa):
    term = (-(moment_3 - 2 * moment_2 ** 2) / moment_3 + x) / (moment_3 / (2 * moment_2))
    return np.abs((1 - scipy.special.gammainc((4 * moment_2 ** 3) / moment_3 ** 2, term)) - pfa)

def lowerRoot(x, moment_2, moment_3, pfa):
    term = (-(moment_3 - 2 * moment_2 ** 2) / moment_3 + x) / (moment_3 / (2 * moment_2))
    return np.abs(scipy.special.gammainc((4 * moment_2 ** 3) / moment_3 ** 2, term) - pfa)

def compute_sk_thresholds(M, N=1, d=1, pfa=0.0013499):
    """
    Computes SK thresholds using Pearson Type III approximation.
    Based on Nita et al. 2016 and adapted from ETSmit's implementation.
    """
    Nd = N * d * 1.0

    moment_1 = 1
    moment_2 = (2 * M**2 * Nd * (1 + Nd)) / ((M - 1) * (6 + 5 * M * Nd + M**2 * Nd**2))
    moment_3 = (8 * M**3 * Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4 + Nd)))) / (
        (M - 1)**2 * (2 + M * Nd) * (3 + M * Nd) * (4 + M * Nd) * (5 + M * Nd)
    )
    moment_4 = (12 * M**4 * Nd * (1 + Nd) * (
        24 + Nd * (48 + 84 * Nd + M * (-32 + Nd * (-245 - 93 * Nd + M * (
            125 + Nd * (68 + M + (3 + M) * Nd)
        ))))
    )) / (
        (M - 1)**3 * (2 + M * Nd) * (3 + M * Nd) * (4 + M * Nd) * (5 + M * Nd) *
        (6 + M * Nd) * (7 + M * Nd)
    )

    std_sk = np.sqrt(moment_2)
    delta = moment_1 - ((2 * moment_2**2) / moment_3)
    beta = 4 * (moment_2**3) / (moment_3**2)
    alpha = moment_3 / (2 * moment_2)
    error_4 = np.abs((100 * 3 * beta * (2 + beta) * alpha**4) / (moment_4 - 1))

 # Root finding
    x0 = 1.0
    upper = scipy.optimize.newton(upperRoot, x0, args=(moment_2, moment_3, pfa))
    lower = scipy.optimize.newton(lowerRoot, x0, args=(moment_2, moment_3, pfa))

    return lower, upper, std_sk
 
