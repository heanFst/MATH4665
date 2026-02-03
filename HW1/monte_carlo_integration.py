"""
Monte Carlo Integration with Control Variates for I = ∫_0^1 exp(-x^2) dx

This script demonstrates:
1. Basic Monte Carlo integration
2. Control variates variance reduction technique
3. Analysis of standard error and convergence
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    The function we want to integrate: f(x) = exp(-x^2)
    Integral of exp(-x^2) from 0 to 1 has no closed form but is approximately 0.746824
    """
    return np.exp(-x**2)

def g(x):
    """
    Control variate function: g(x) = 1/(1+x)
    This is chosen because:
    1. It has similar shape to exp(-x^2) on [0,1] (decreasing, convex)
    2. Its integral can be computed analytically: ∫_0^1 1/(1+x) dx = ln(2)
    3. It is positively correlated with f(x)
    """
    return 1.0 / (1.0 + x)

def monte_carlo_integration(f_func, n_samples, seed=42):
    """
    Basic Monte Carlo integration.

    Parameters:
    -----------
    f_func : callable
        Function to integrate
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    estimate : float
        Monte Carlo estimate of the integral
    std_error : float
        Standard error of the estimator
    samples : ndarray
        The function values at sample points
    """
    np.random.seed(seed)
    U = np.random.rand(n_samples)
    Y = f_func(U)

    estimate = np.mean(Y)
    variance = np.var(Y, ddof=1)
    std_error = np.sqrt(variance / n_samples)

    return estimate, std_error, Y

def control_variates(f_func, g_func, mu_g, n_samples, seed=42):
    """
    Monte Carlo integration with control variates.

    Parameters:
    -----------
    f_func : callable
        Function to integrate
    g_func : callable
        Control variate function
    mu_g : float
        Known true mean of g(X)
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    estimate : float
        Control variates estimate of the integral
    std_error : float
        Standard error of the estimator
    c_optimal : float
        Optimal coefficient
    variance_reduction : float
        Factor of variance reduction
    """
    np.random.seed(seed)
    U = np.random.rand(n_samples)
    Y = f_func(U)
    C = g_func(U)

    # Compute optimal coefficient
    cov_matrix = np.cov(Y, C, ddof=1)
    c_optimal = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Apply control variates correction
    Y_cv = Y - c_optimal * (C - mu_g)

    estimate = np.mean(Y_cv)
    variance_cv = np.var(Y_cv, ddof=1)
    std_error = np.sqrt(variance_cv / n_samples)

    # Compute variance reduction factor
    variance_basic = np.var(Y, ddof=1)
    variance_reduction = variance_basic / variance_cv

    return estimate, std_error, c_optimal, variance_reduction

def convergence_analysis():
    """Analyze convergence rate for different sample sizes."""
    print("=" * 60)
    print("CONVERGENCE ANALYSIS: Standard Error vs Sample Size")
    print("=" * 60)

    # Reference value using numerical integration (for error comparison)
    from scipy.integrate import quad
    true_value, _ = quad(f, 0, 1)

    n_values = [10**i for i in range(1, 6)]  # 10, 100, 1000, 10000, 100000

    print(f"\nTrue value (numerical): {true_value:.10f}\n")
    print(f"{'N':>10} | {'MC Estimate':>12} | {'Std Error':>12} | {'Actual Error':>12}")
    print("-" * 60)

    for n in n_values:
        estimate, std_error, _ = monte_carlo_integration(f, n, seed=42)
        actual_error = abs(estimate - true_value)
        print(f"{n:>10} | {estimate:>12.8f} | {std_error:>12.8f} | {actual_error:>12.8f}")

    print("\nObservation: As N increases by factor of 10, standard error")
    print("decreases by factor of ~√10 ≈ 3.16, confirming O(1/√N) convergence.")

def main():
    """Main simulation comparing basic Monte Carlo and control variates."""
    np.random.seed(42)
    N = 100000

    print("=" * 60)
    print("MONTE CARLO INTEGRATION: I = int_0^1 exp(-x^2) dx")
    print("=" * 60)
    print(f"Number of samples: N = {N}\n")

    # Part (a): Basic Monte Carlo
    print("-" * 60)
    print("PART (a): BASIC MONTE CARLO METHOD")
    print("-" * 60)

    mc_estimate, mc_std_error, Y = monte_carlo_integration(f, N)

    print(f"Estimate:     I_hat = {mc_estimate:.10f}")
    print(f"Std Error:    SE = {mc_std_error:.10f}")
    print(f"95% CI:       [{mc_estimate - 1.96*mc_std_error:.8f}, {mc_estimate + 1.96*mc_std_error:.8f}]")

    # Theoretical variance of exp(-U^2) where U ~ Uniform(0,1)
    # E[Y^2] - (E[Y])^2 where Y = exp(-U^2)
    # This can be derived but we use empirical estimate
    print(f"\nVariance of estimator: Var(I_hat) = {np.var(Y, ddof=1)/N:.10f}")
    print(f"Variance of Y:          Var(Y)  = {np.var(Y, ddof=1):.10f}")

    # Part (b): Control Variates
    print("\n" + "-" * 60)
    print("PART (b): CONTROL VARIATES METHOD")
    print("-" * 60)

    # Known integral of control variate: ∫_0^1 1/(1+x) dx = ln(2)
    mu_g = np.log(2)

    cv_estimate, cv_std_error, c_optimal, vr_factor = control_variates(f, g, mu_g, N)

    print(f"Control variate:       g(x) = 1/(1+x)")
    print(f"Known integral:        int_0^1 g(x)dx = ln(2) = {mu_g:.10f}")
    print(f"Optimal coefficient:   c* = {c_optimal:.10f}")
    print(f"\nEstimate:     I_cv = {cv_estimate:.10f}")
    print(f"Std Error:    SE_cv = {cv_std_error:.10f}")
    print(f"95% CI:       [{cv_estimate - 1.96*cv_std_error:.8f}, {cv_estimate + 1.96*cv_std_error:.8f}]")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Variance Reduction Factor: {vr_factor:.2f}x")
    print(f"Std Error Reduction:       {mc_std_error / cv_std_error:.2f}x")
    print(f"\nThe control variates estimator achieves the same precision")
    print(f"with approximately {N/vr_factor:.0f} samples instead of {N}.")

    # Run convergence analysis
    print("\n")
    convergence_analysis()

    # Optional: Create visualization
    create_visualization()

def create_visualization():
    """Create convergence plot."""
    try:
        from scipy.integrate import quad
        true_value, _ = quad(f, 0, 1)
    except ImportError:
        # If scipy not available, use known approximation
        true_value = 0.746824132812427

    np.random.seed(42)
    N_values = np.logspace(1, 6, 20).astype(int)

    mc_errors = []
    cv_errors = []

    for N in N_values:
        mc_est, _, _ = monte_carlo_integration(f, int(N))
        cv_est, _, _, _ = control_variates(f, g, np.log(2), int(N))

        mc_errors.append(abs(mc_est - true_value))
        cv_errors.append(abs(cv_est - true_value))

    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, mc_errors, 'bo-', label='Basic Monte Carlo', markersize=6)
    plt.loglog(N_values, cv_errors, 'rs-', label='Control Variates', markersize=6)

    # Add reference lines showing 1/sqrt(N) slope
    ref_line = mc_errors[0] * np.sqrt(N_values[0]) / np.sqrt(N_values)
    plt.loglog(N_values, ref_line, 'k--', alpha=0.5, label=r'$O(1/\sqrt{N})$')

    plt.xlabel('Number of Samples ($N$)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Monte Carlo Convergence: $\\int_0^1 e^{-x^2}dx$', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('mc_convergence.png', dpi=300)
    print("\nConvergence plot saved as 'mc_convergence.png'")

if __name__ == "__main__":
    main()
