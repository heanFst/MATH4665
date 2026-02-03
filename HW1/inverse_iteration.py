"""
Inverse Iteration Method for Finding Eigenpairs Near a Shift

This script implements the inverse iteration algorithm to find the eigenvalue
closest to a given shift sigma and its corresponding eigenvector.
"""

import numpy as np
import matplotlib.pyplot as plt

def inverse_iteration(A, sigma=1.1, max_iter=1000, tol=1e-10):
    """
    Inverse iteration algorithm to find eigenpair near shift sigma.

    Parameters:
    -----------
    A : numpy.ndarray
        A square matrix
    sigma : float
        The shift value
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence

    Returns:
    --------
    lambda_val : float
        The eigenvalue closest to sigma
    v : numpy.ndarray
        The corresponding eigenvector (normalized)
    errors : list
        History of errors during iteration
    """
    n = A.shape[0]

    # Form (A - sigma*I) and factorize once
    M = A - sigma * np.eye(n)

    # Initialize with random vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    # Find the true eigenvector for the eigenvalue closest to sigma
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmin(np.abs(eigenvalues - sigma))
    true_lambda = eigenvalues[idx].real
    true_v = eigenvectors[:, idx].real
    # Ensure consistent sign
    if true_v[0] < 0:
        true_v = -true_v

    errors = []

    for k in range(max_iter):
        # Store previous vector for convergence check
        v_prev = v.copy()

        # Inverse iteration step: solve (A - sigma*I) * w = v
        w = np.linalg.solve(M, v)
        v = w / np.linalg.norm(w)

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
    """Test inverse iteration on a random symmetric matrix."""
    np.random.seed(42)

    # Create a random symmetric matrix of size 100x100
    n = 100
    B = np.random.rand(n, n)
    A = (B + B.T) / 2  # Make it symmetric

    # Set shift value
    sigma = 1.1

    # Run inverse iteration
    lambda_val, v, errors = inverse_iteration(A, sigma)

    # Verify with numpy
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmin(np.abs(eigenvalues - sigma))
    true_lambda = eigenvalues[idx].real

    print(f"Shift value (sigma):          {sigma:.2f}")
    print(f"Computed eigenvalue:          {lambda_val:.10f}")
    print(f"True closest eigenvalue:      {true_lambda:.10f}")
    print(f"Absolute error:               {abs(lambda_val - true_lambda):.2e}")
    print(f"Distance from shift:          {abs(lambda_val - sigma):.2e}")
    print(f"Number of iterations:         {len(errors)}")

    # Plot error convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(errors, 'r-', linewidth=2, label=rf'$||v^{{(k)}} - v||_\infty$')
    plt.xlabel('Iteration $k$', fontsize=12)
    plt.ylabel(r'Error ($\infty$-norm)', fontsize=12)
    plt.title(rf'Inverse Iteration Convergence ($\sigma = {sigma}$)', fontsize=14)
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
    plt.savefig('inverse_iteration_convergence.png', dpi=300)
    plt.savefig('inverse_iteration_convergence.pdf')
    print("\nPlot saved as 'inverse_iteration_convergence.png' and '.pdf'")
    plt.show()


if __name__ == "__main__":
    main()
