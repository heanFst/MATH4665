import numpy as np

def f(x):
    """
    The function we want to integrate: f(x) = exp(x)
    Integral of exp(x) from 0 to 1 is e - 1 approx 1.71828
    """
    return np.exp(x)

def g(x):
    """
    The control variate function: g(x) = 1 + x
    This function is correlated with exp(x) and its integral is known.
    Integral of (1+x) from 0 to 1 is 1 + 0.5 = 1.5
    """
    return 1 + x

def main():
    np.random.seed(42)
    N = 10000  # Number of samples
    
    # 1. Generate random samples from Uniform(0, 1)
    U = np.random.rand(N)
    
    # 2. Evaluate f(U) and g(U)
    Y = f(U)
    C = g(U)
    
    # True expected value of control variate g(U)
    # E[g(U)] = integral_0^1 (1+x) dx = [x + x^2/2]_0^1 = 1.5
    mu_c = 1.5
    
    # --- Standard Monte Carlo Estimator ---
    # Estimate = mean(Y)
    mc_estimate = np.mean(Y)
    mc_variance = np.var(Y, ddof=1) / N  # Variance of the estimator
    
    print(f"Standard Monte Carlo Estimate: {mc_estimate:.6f}")
    print(f"Standard MC Variance: {mc_variance:.10f}")
    
    # --- Control Variates Estimator ---
    # We want to estimate E[f(U)].
    # We know E[g(U)] = mu_c.
    # New estimator: Y_cv = Y - c * (C - mu_c)
    # Optimal c* = Cov(Y, C) / Var(C)
    
    cov_matrix = np.cov(Y, C)
    cov_yc = cov_matrix[0, 1]
    var_c = cov_matrix[1, 1]
    
    c_star = cov_yc / var_c
    
    Y_cv = Y - c_star * (C - mu_c)
    
    cv_estimate = np.mean(Y_cv)
    cv_variance = np.var(Y_cv, ddof=1) / N
    
    print("-" * 30)
    print(f"Optimal c*: {c_star:.6f}")
    print(f"Control Variates Estimate: {cv_estimate:.6f}")
    print(f"Control Variates Variance: {cv_variance:.10f}")
    
    # --- Comparison ---
    print("-" * 30)
    variance_reduction_factor = mc_variance / cv_variance
    print(f"Variance Reduction Factor: {variance_reduction_factor:.2f}x")
    print(f"True Value: {np.exp(1) - 1:.6f}")

if __name__ == "__main__":
    main()
