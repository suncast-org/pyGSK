# Theoretical Background

The **Spectral Kurtosis (SK)** estimator is a powerful statistical tool for detecting non-Gaussian features in time-frequency data, such as radio-frequency interference (RFI) or transient signals.  
This section summarizes the theoretical framework behind the **Generalized Spectral Kurtosis (GSK)** formulation, following the derivations presented in *Nita & Gary (2010, PASP 122, 595–607)*.

---

## 1. Definition

For a sequence of complex spectral power measurements \( P_i \) (with \( i = 1, \ldots, M \)), the **Spectral Kurtosis estimator** is defined as

\[
\widehat{S}_K = \frac{M+1}{M-1}
\left( \frac{M \sum_i P_i^2}{\left( \sum_i P_i \right)^2} - 1 \right).
\]

This dimensionless quantity has an **expected value of 1** for purely Gaussian signals and deviates from unity when non-Gaussian components (such as RFI or bursts) are present.

---

## 2. Generalized Formulation

The **Generalized Spectral Kurtosis (GSK)** estimator extends the above definition to the case where power estimates are averaged over \( N \) independent spectra, each derived from \( M \) accumulations:

\[
\widehat{S}_K(M, N, d) = \frac{M N d + 1}{M N d - 1}
\left(
\frac{M N \, S_2}{S_1^2} - 1
\right),
\]

where:

- \( S_1 = \sum_{j=1}^N P_j \) — the sum of averaged powers  
- \( S_2 = \sum_{j=1}^N P_j^2 \) — the sum of squared averaged powers  
- \( d \) — the **shape parameter** of the underlying Gamma distribution (for complex data, \( d = 1 \))

The generalized form reduces to the classical SK when \( N = 1 \) and \( d = 1 \).

---

## 3. Statistical Properties

Under Gaussian statistics, the power estimates \( P_j \) follow a **Gamma distribution** with shape parameter \( M d \) and scale parameter \( \theta \).  
In this case:

\[
\mathbb{E}[\widehat{S}_K] = 1,
\qquad
\mathrm{Var}[\widehat{S}_K] = \frac{4}{M N (M N d + 1)}.
\]

These relationships show that increasing \( M \) or \( N \) reduces the estimator variance, improving the reliability of SK-based detection.

---

## 4. Thresholds and Detection

The SK statistic follows an approximately scaled **Gamma distribution**, allowing the determination of **probability-of-false-alarm (PFA)** thresholds for detecting deviations from Gaussianity.

Given a target \( \mathrm{PFA} \) (e.g., \( 10^{-3} \)), one can define **two-sided thresholds** \( [S_{K,\mathrm{low}}, S_{K,\mathrm{high}}] \) such that:

\[
P(S_K < S_{K,\mathrm{low}}) = P(S_K > S_{K,\mathrm{high}}) = \frac{\mathrm{PFA}}{2}.
\]

Values of \( S_K \) lying outside this interval indicate statistically significant departures from Gaussian noise.

The **pyGSK** package computes these thresholds using direct integration of the Gamma distribution and provides one- or two-sided options.

---

## 5. Renormalized SK Estimator

Finite-sample bias can lead to slight deviations of the SK mean from unity.  
To correct for this, the **renormalized estimator** is defined as:

\[
\widehat{S}_K^\mathrm{(R)} =
\frac{\widehat{S}_K}{\mathbb{E}[\widehat{S}_K]},
\]

ensuring \( \mathbb{E}[\widehat{S}_K^\mathrm{(R)}] = 1 \) for Gaussian inputs.  
This form is implemented in the CLI command [`renorm-sk-test`](cli_guide.md).

---

## 6. Practical Interpretation

- \( \widehat{S}_K \approx 1 \): data consistent with Gaussian noise  
- \( \widehat{S}_K > S_{K,\mathrm{high}} \): impulsive or non-stationary power excess (e.g., RFI burst)  
- \( \widehat{S}_K < S_{K,\mathrm{low}} \): power deficit or coherence (e.g., sinusoidal contamination)

Thus, SK provides an **adaptive, non-parametric detection statistic** independent of absolute power levels.

---

## 7. References

- Nita, G. M., & Gary, D. E. (2010). *The generalized spectral kurtosis estimator.* 
  MNRAS Letters, 406(1), L60–L64. https://doi.org/10.1111/j.1745-3933.2010.00882.x

---

## Next Steps

- Learn to **use the CLI tools** in [cli_guide.md](cli_guide.md)  
- For implementation details, see [dev_guide.md](dev_guide.md)

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
