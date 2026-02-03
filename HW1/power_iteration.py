"""
Power Iteration Method for Finding the Dominant Eigenpair

This script implements the power iteration algorithm to find the largest
eigenvalue (in magnitude) and its corresponding eigenvector of a matrix.
"""

import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, max_iter=1000, tol=1e-10):
    """
    Power iteration algorithm to find the dominant eigenpair.

    Parameters:
    -----------
    A : numpy.ndarray
        A square matrix
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence

    Returns:
    --------
    lambda_val : float
        The dominant eigenvalue
    v : numpy.ndarray
        The corresponding eigenvector (normalized)
    errors : list
        History of errors during iteration
    """
    n = A.shape[0]

    # Initialize with random vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    # Find the true eigenvector using numpy for error calculation
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues))
    true_lambda = eigenvalues[idx]
    true_v = eigenvectors[:, idx].real
    # Ensure consistent sign
    if true_v[0] < 0:
        true_v = -true_v

    errors = []

    for k in range(max_iter):
        # Store previous vector for convergence check
        v_prev = v.copy()

        # Power iteration step
        Av = A @ v
        v = Av / np.linalg.norm(Av)

        # Rayleigh quotient for eigenvalue estimate
        lambda_val = (v @ A @ v) / (v @ v)

        # Calculate error (infinity norm of difference from true eigenvector)
        # Handle sign ambiguity
        if np.sign(v[0]) != np.sign(true_v[0]):
            v_current = -v
        else:
            v_current = v

        error = np.linalg.norm(v_current - true_v, np.inf)
        errors.append(error)

        # Check for convergence
        if np.linalg.norm(v - v_prev) < tol:
            break

    return lambda_val, v, errors


def main():
    """Test power iteration on a random symmetric matrix."""
    np.random.seed(42)

    # Create a random symmetric matrix of size 100x100
    n = 100
    B = np.random.rand(n, n)
    A = (B + B.T) / 2  # Make it symmetric

    # Run power iteration
    lambda_val, v, errors = power_iteration(A)

    # Verify with numpy
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues))
    true_lambda = eigenvalues[idx].real

    print(f"Computed dominant eigenvalue: {lambda_val:.10f}")
    print(f"True dominant eigenvalue:     {true_lambda:.10f}")
    print(f"Absolute error:               {abs(lambda_val - true_lambda):.2e}")
    print(f"Number of iterations:         {len(errors)}")

    # Plot error convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(errors, 'b-', linewidth=2, label=rf'$||v^{{(k)}} - v||_\infty$')
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'Error ($\infty$-norm)', fontsize=12)
    plt.title(r'Power Iteration Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Add convergence rate information
    # Estimate rate from last 50% of iterations
    mid_point = len(errors) // 2
    rate_estimate = (errors[-1] / errors[mid_point]) ** (1 / (len(errors) - mid_point))
    plt.text(0.7, 0.1, f'Est. rate: {rate_estimate:.4f}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('power_iteration_convergence.png', dpi=300)
    plt.savefig('power_iteration_convergence.pdf')
    print("\nPlot saved as 'power_iteration_convergence.png' and '.pdf'")
    plt.show()


if __name__ == "__main__":
    main()
