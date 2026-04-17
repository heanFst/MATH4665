import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge


def main() -> None:
    np.random.seed(42)

    x_train = np.sort(np.random.rand(40, 1), axis=0)
    y_true_train = np.sin(2 * np.pi * x_train).ravel()
    y_train = y_true_train + np.random.normal(0, 0.20, size=y_true_train.shape)

    x_test = np.linspace(0.0, 1.0, 400)[:, np.newaxis]
    y_true = np.sin(2 * np.pi * x_test).ravel()

    lambdas = [1e-1, 1e-3, 1e-6]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    plt.figure(figsize=(9, 5.4))
    plt.scatter(x_train, y_train, color="black", s=18, label="Noisy observations", zorder=3)
    plt.plot(
        x_test,
        y_true,
        color="0.45",
        linestyle="--",
        linewidth=2.0,
        label=r"True function $f_0(x)=\sin(2\pi x)$",
        zorder=2,
    )

    for lam, color in zip(lambdas, colors):
        model = KernelRidge(kernel="rbf", gamma=30.0, alpha=lam)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        plt.plot(
            x_test,
            y_pred,
            color=color,
            linewidth=2.0,
            label=rf"$\lambda={lam:.0e}$",
        )

    plt.xlim(0.0, 1.0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Kernel ridge regression under different regularization levels")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("krr_plot.pdf", format="pdf")


if __name__ == "__main__":
    main()
