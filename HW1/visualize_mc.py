import numpy as np
import matplotlib.pyplot as plt

def visualize_pi_estimation(N=2000):
    # Generate random points in [-1, 1] x [-1, 1]
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    
    # Check if points are inside the circle
    inside = x**2 + y**2 <= 1
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x[inside], y[inside], color='blue', s=1, label='Inside Circle')
    plt.scatter(x[~inside], y[~inside], color='red', s=1, label='Outside Circle')
    
    # Draw circle boundary
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), color='black', linewidth=2)
    
    plt.title(f"Monte Carlo $\\pi$ Estimation (N={N})\nEstimated $\\pi \\approx$ {4 * np.sum(inside) / N:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('mc_pi_darts.png', dpi=300)
    plt.close()

def visualize_convergence():
    N_values = np.logspace(1, 5, 50, dtype=int)
    errors = []
    
    true_pi = np.pi
    
    for N in N_values:
        x = np.random.uniform(-1, 1, N)
        y = np.random.uniform(-1, 1, N)
        est_pi = 4 * np.sum(x**2 + y**2 <= 1) / N
        errors.append(abs(est_pi - true_pi))
    
    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, errors, 'o-', label='Empirical Error')
    plt.loglog(N_values, 1/np.sqrt(N_values), '--', label='$1/\\sqrt{N}$ Theoretical Rate', color='gray')
    
    plt.title("Monte Carlo Convergence Rate")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Absolute Error $|\\hat{\\pi} - \\pi|$")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig('mc_convergence_rate.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    print("Generating Monte Carlo visualizations...")
    visualize_pi_estimation()
    visualize_convergence()
    print("Images saved: mc_pi_darts.png, mc_convergence_rate.png")
