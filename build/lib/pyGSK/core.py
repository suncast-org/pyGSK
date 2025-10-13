import numpy as np
import scipy.special
import scipy.optimize
from pyGSK.plot import plot_sk_dual_histogram
from pyGSK.thresholds import compute_sk_thresholds

def get_sk(s1, s2, M, N=1, d=1):
    """
    Compute the Generalized Spectral Kurtosis (SK) estimator.

    Parameters:
        s1 (ndarray): Accumulated power samples
        s2 (ndarray): Accumulated squared power samples
        d (float): Shape factor of the power Gamma distribution describing the raw power
        N (int): On-board accumulation length
        M (int): Accumulation length

    Returns:
        ndarray: SK values
    """
    sk = ((M * N * d + 1) / (M - 1)) * ((M * s2) / (s1**2) - 1)
    return sk


def renorm_sk(s1, s2, M):
    """
    Empirically renormalize SK values using mode-based shape factor.

    Based on:
    Nita, G.M. and Hellbourg, G. (2020).  
    A Cross-Correlation Based Spectral Kurtosis RFI Detector.  
    URSI GASS, IEEE. https://doi.org/10.23919/URSIGASS49373.2020.9232173

    Parameters:
        s1 (ndarray): Accumulated power samples
        s2 (ndarray): Accumulated squared power samples
        M (int): Accumulation length

    Returns:
        tuple: (d_empirical, sk_renormalized)
    """
    sk_raw = get_sk(s1, s2, M)
    mode = np.median(sk_raw.flatten())  # robust mode estimate
    d_empirical = (M - mode + 1) / (mode * M)
    sk_renorm = get_sk(s1, s2, M, d=d_empirical)
    return d_empirical, sk_renorm

    
def run_sk_test(M=128, N=64, d=1.0, ns=10000, alpha=0.0013499, seed=42, plot=False, save_path=None):
    print()
    print(f"Running SK test with M = {M}, N = {N}, d ={d}, ns = {ns}")
    
    rng = np.random.default_rng(seed)

    # Step 1: Simulate ns × M samples, each as sum of N exponential(1) → Gamma(N, 1.0)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))

    # SK computation
    s1 = np.sum(samples, axis=1)
    s2 = np.sum(samples**2, axis=1)

    # Step 2: Compute SK
    sk = get_sk(s1, s2, M, N, d)

    # Step 3: Compute thresholds
    lower, upper, std_sk = compute_sk_thresholds(M, N, d, alpha)

    # Step 4: Count false alarms
    below = np.sum(sk < lower)
    above = np.sum(sk > upper)
    total = sk.size
    empirical_pfa = (below + above) / total
    expected_total_pfa = 2 * alpha

    # Step 5: Print diagnostics
    mean_sk = np.mean(sk)
    empirical_std_sk = np.std(sk)
    print(f"SK mean = {mean_sk:.4f}")
    print(f"Empirical std = {empirical_std_sk:.4f}, Expected std = {std_sk:.4f}")
    print(f"Thresholds: lower = {lower:.4f}, upper = {upper:.4f}")
    print(f"False alarms: below = {below}, above = {above}, total = {total}")
    print(f"Empirical PFA = {empirical_pfa:.5f}, expected = {expected_total_pfa:.5f}")

    # Step 6: Assert match
    assert abs(empirical_pfa - expected_total_pfa) < 0.001, (
        f"Empirical PFA = {empirical_pfa:.5f}, expected ~{expected_total_pfa:.5f}"
    )

    if plot:
        from pyGSK.plot import plot_sk_histogram
        plot_sk_histogram(sk, lower, upper, M, N, d, ns, alpha, below, above, total, save_path)

def run_renorm_sk_test(M=128, N=64, ns=10000, seed=42, plot=False, save_path=None, assumed_N=1.0, log_count=False, log_bins=False, log_x=False):
    from pyGSK.thresholds import compute_sk_thresholds
    print()
    print(f"Running renormalized SK test with M = {M}, N = {N}, ns = {ns} under the wrong assumption that N=1")

    rng = np.random.default_rng(seed)

    # Simulate ns × M samples, each as sum of N exponential(1) → Gamma(N, 1.0)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))

    s1 = np.sum(samples, axis=1)
    s2 = np.sum(samples**2, axis=1)

    # Renormalize SK
    d_empirical, sk_renorm = renorm_sk(s1, s2, M)

    mean_sk = np.mean(sk_renorm)
    std_sk = np.std(sk_renorm)
    print(f"Empirical d = {d_empirical:.6f}")
    print(f"Renormalized SK mean = {mean_sk:.4f}")
    print(f"Renormalized SK std = {std_sk:.4f}")

    # Validate recovery of N (since true d = 1.0)
    assert abs(d_empirical - N) < 0.5, (
        f"Recovered N = {d_empirical:.2f}, expected ≈ {N}"
    )

    # Thresholds based on empirical d
    alpha = 0.0013499
    lower, upper, expected_std = compute_sk_thresholds(M, N, d_empirical, alpha)

    below = np.sum(sk_renorm < lower)
    above = np.sum(sk_renorm > upper)
    total = sk_renorm.size
    empirical_pfa = (below + above) / total
    expected_total_pfa = 2 * alpha

    print(f"Thresholds: lower = {lower:.4f}, upper = {upper:.4f}")
    print(f"False alarms: below = {below}, above = {above}, total = {total}")
    print(f"Empirical PFA = {empirical_pfa:.5f}, expected = {expected_total_pfa:.5f}")

    # Validate false alarm rate
    assert abs(empirical_pfa - expected_total_pfa) < 0.001, (
        f"Empirical PFA = {empirical_pfa:.5f}, expected ~{expected_total_pfa:.5f}"
    )

    if plot:
        from pyGSK.plot import plot_sk_dual_histogram
        from pyGSK.core import get_sk
        from pyGSK.thresholds import compute_sk_thresholds

        # Raw SK under assumed N
        sk_raw = get_sk(s1, s2, M, d=1.0)
        lower_raw, upper_raw, _ = compute_sk_thresholds(M, N=assumed_N, d=1.0, pfa=alpha)
        below_raw = np.sum(sk_raw < lower_raw)
        above_raw = np.sum(sk_raw > upper_raw)

        # Renormalized SK already computed
        plot_sk_dual_histogram(
            sk_raw=sk_raw, lower_raw=lower_raw, upper_raw=upper_raw, d_raw=1.0,
            sk_renorm=sk_renorm, lower_renorm=lower, upper_renorm=upper, d_empirical=d_empirical,
            M=M, N=N, ns=ns, alpha=alpha,
            below_raw=below_raw, above_raw=above_raw,
            below_renorm=below, above_renorm=above,
            assumed_N=assumed_N, log_count=log_count,log_x=log_x, log_bins=log_bins,
            save_path=save_path
        )