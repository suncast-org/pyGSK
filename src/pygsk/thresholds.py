"""
Threshold computation for SK detection using Pearson Type III approximation.

This module implements the numerical estimation of upper and lower SK thresholds
based on higher-order moments of the SK distribution. It uses root-finding techniques
to invert the incomplete gamma function and match a target false alarm probability (PFA).

The implementation follows the analytical framework from Nita et al. (2016), adapted
for reproducible SK validation and detection benchmarking.
"""

import numpy as np
import scipy.optimize

def upperRoot(x, moment_2, moment_3, pfa):
    """
    Objective function for computing the upper SK threshold via root finding.

    This function inverts the Pearson Type III cumulative distribution function
    to match the desired upper-tail false alarm probability.

    Parameters:
        x (float): Candidate SK threshold value.
        moment_2 (float): Second central moment of the SK distribution.
        moment_3 (float): Third central moment of the SK distribution.
        pfa (float): One-sided false alarm probability.

    Returns:
        float: Absolute difference between target and achieved upper-tail probability.
    """
    term = (-(moment_3 - 2 * moment_2 ** 2) / moment_3 + x) / (moment_3 / (2 * moment_2))
    return np.abs((1 - scipy.special.gammainc((4 * moment_2 ** 3) / moment_3 ** 2, term)) - pfa)

def lowerRoot(x, moment_2, moment_3, pfa):
    """
    Objective function for computing the lower SK threshold via root finding.

    This function inverts the Pearson Type III cumulative distribution function
    to match the desired lower-tail false alarm probability.

    Parameters:
        x (float): Candidate SK threshold value.
        moment_2 (float): Second central moment of the SK distribution.
        moment_3 (float): Third central moment of the SK distribution.
        pfa (float): One-sided false alarm probability.

    Returns:
        float: Absolute difference between target and achieved lower-tail probability.
    """
    term = (-(moment_3 - 2 * moment_2 ** 2) / moment_3 + x) / (moment_3 / (2 * moment_2))
    return np.abs(scipy.special.gammainc((4 * moment_2 ** 3) / moment_3 ** 2, term) - pfa)

def compute_sk_thresholds(M, N=1, d=1, pfa=0.0013499):
    """
    Compute SK detection thresholds using Pearson Type III approximation.
    Based on Nita et al. 2016 and adapted from ETSmit's implementation.

    This function estimates the second, third, and fourth moments of the SK distribution
    and uses them to derive thresholds that bound the central (1 - 2*pfa) region.
    It numerically solves for the SK values that yield the desired false alarm probability.

    Parameters:
        M (int): Number of frequency channels.
        N (int, optional): Number of accumulations per channel. Default is 1.
        d (float, optional): Scaling factor. Default is 1.
        pfa (float, optional): One-sided false alarm probability. Default is 0.0013499.

    Returns:
        tuple:
            lower (float): Lower SK detection threshold.
            upper (float): Upper SK detection threshold.
            std_sk (float): Theoretical standard deviation of SK under null hypothesis.
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
 
