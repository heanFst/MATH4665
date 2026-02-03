import numpy as np
import matplotlib.pyplot as plt

def visualize_ex_integration(N=100):
    x_fine = np.linspace(0, 1, 1000)
    y_fine = np.exp(x_fine)
    
    # Random samples
    np.random.seed(42)
    U = np.random.rand(N)
    Y = np.exp(U)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    ax.plot(x_fine, y_fine, 'r-', lw=2, label='$f(x) = e^x$')
    ax.fill_between(x_fine, y_fine, alpha=0.2, color='red', label='True Integral $I$')
    
    # Plot MC samples as "strips" or points
    ax.scatter(U, Y, color='blue', s=20, alpha=0.6, label='MC Samples $f(U_i)$')
    for ui, yi in zip(U, Y):
        ax.vlines(ui, 0, yi, color='blue', alpha=0.1, lw=1)
        
    ax.set_title(f"Monte Carlo Integration of $e^x$ on [0, 1] (N={N})")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('mc_ex_integration.png', dpi=300)
    plt.close()

def visualize_error_distribution(N=1000, trials=1000):
    estimates = []
    for _ in range(trials):
        U = np.random.rand(N)
        estimates.append(np.mean(np.exp(U)))
        
    true_val = np.exp(1) - 1
    
    plt.figure(figsize=(10, 6))
    plt.hist(estimates, bins=30, density=True, alpha=0.7, color='green', label='Estimator Distribution')
    
    # Plot Gaussian fit based on CLT
    mu = true_val
    sigma = np.sqrt(np.var(np.exp(np.random.rand(10000))) / N)
    x = np.linspace(min(estimates), max(estimates), 100)
    plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ), 
             'r--', lw=2, label='CLT Prediction $\\mathcal{N}(I, \\sigma^2/N)$')
    
    plt.axvline(true_val, color='black', linestyle='-', lw=2, label=f'True Value $\\approx {true_val:.4f}$')
    plt.title(f"Distribution of MC Estimator $\\hat{{I}}_N$ (N={N}, trials={trials})")
    plt.xlabel("Estimated Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mc_clt_distribution.png', dpi=300)
    plt.close()

def visualize_variance_reduction(N=1000, trials=1000):
    np.random.seed(42)
    mc_estimates = []
    cv_estimates = []
    
    true_val = np.exp(1) - 1
    mu_c = 1.5 # Integral of 1+x on [0,1]
    
    for _ in range(trials):
        U = np.random.rand(N)
        Y = np.exp(U)
        C = 1 + U
        
        # Standard MC
        mc_estimates.append(np.mean(Y))
        
        # Control Variates
        cov_matrix = np.cov(Y, C)
        c_star = cov_matrix[0, 1] / cov_matrix[1, 1]
        Y_cv = Y - c_star * (C - mu_c)
        cv_estimates.append(np.mean(Y_cv))
        
    plt.figure(figsize=(10, 6))
    plt.hist(mc_estimates, bins=40, density=True, alpha=0.5, color='blue', label='Standard MC')
    plt.hist(cv_estimates, bins=40, density=True, alpha=0.5, color='orange', label='Control Variates')
    
    plt.axvline(true_val, color='black', linestyle='-', lw=2, label=f'True Value')
    plt.title(f"Variance Reduction Comparison (N={N}, trials={trials})\n$f(x)=e^x$, $g(x)=1+x$")
    plt.xlabel("Estimated Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mc_variance_reduction.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    visualize_ex_integration()
    visualize_error_distribution()
    visualize_variance_reduction()
    print("Saved: mc_ex_integration.png, mc_clt_distribution.png, mc_variance_reduction.png")
