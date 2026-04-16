import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

# Generate data
np.random.seed(42)
X = np.sort(np.random.rand(40, 1) * 2 - 1, axis=0) # X in [-1, 1]
y = np.sin(2 * np.pi * X).ravel()
y_noise = y + np.random.normal(0, 0.3, size=y.shape)

X_test = np.linspace(-1, 1, 200)[:, np.newaxis]
y_true = np.sin(2 * np.pi * X_test).ravel()

# Plot setup
plt.figure(figsize=(10, 6))
plt.scatter(X, y_noise, color='black', s=20, label='Noisy Data', zorder=3)
plt.plot(X_test, y_true, color='gray', linestyle='--', linewidth=2, label='True $f_0(x) = \sin(2\pi x)$', zorder=2)

# Lambdas to test
lambdas = [1e-1, 1e-4, 1e-8]
colors = ['blue', 'orange', 'red']
labels = [r'$\lambda = 10^{-1}$ (Underfit)', r'$\lambda = 10^{-4}$ (Good Fit)', r'$\lambda = 10^{-8}$ (Overfit)']

for alpha, color, label in zip(lambdas, colors, labels):
    kr = KernelRidge(kernel='rbf', gamma=10, alpha=alpha)
    kr.fit(X, y_noise)
    y_pred = kr.predict(X_test)
    plt.plot(X_test, y_pred, color=color, linewidth=2, label=label, alpha=0.8)

plt.title('Kernel Ridge Regression for Different $\lambda$ Values', fontsize=14)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save figure
plt.savefig('krr_plot.pdf', format='pdf')
