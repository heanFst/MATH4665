I will update `Sol1.tex` to include a self-contained report in the "Question 1" section.

The report will be structured as follows:
1.  **Introduction**: Briefly introduce Power Iteration as the foundational method for finding dominant eigenvalues.
2.  **Inverse Iteration with Shift**: Explain how shifting and inverting the matrix transforms the problem of finding an eigenvalue closest to a scalar $\sigma$ (in this case, $\pi$) into finding the dominant eigenvalue of a transformed matrix.
3.  **Algorithmic Procedure**: Provide a step-by-step, mathematically rigorous algorithm ready for code implementation.
    *   **Step 1 (Setup)**: Initialize vector $x^{(0)}$ and shift $\sigma = \pi$.
    *   **Step 2 (Linear Solve)**: Instead of inverting $(A - \sigma I)$, solve the system $(A - \sigma I)y^{(k)} = x^{(k-1)}$.
    *   **Step 3 (Normalization)**: Normalize the vector to maintain numerical stability.
    *   **Step 4 (Rayleigh Quotient)**: Estimate the eigenvalue $\lambda^{(k)} = (x^{(k)})^T A x^{(k)}$.
4.  **Convergence**: Briefly explain that convergence depends on the separation of the closest eigenvalue from the others relative to $\sigma$.

**Planned Edit:**
-   Modify `d:\文档\MATH4665\Sol1.tex` to insert the LaTeX content into the `Question 1` section.
