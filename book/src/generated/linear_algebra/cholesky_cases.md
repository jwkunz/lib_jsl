| matrix | Cholesky | LU |
|---|---|---|
| $\begin{pmatrix}4 & 2\\ 2 & 3\end{pmatrix}$, positive definite | succeeds | succeeds |
| $\begin{pmatrix}1 & 2\\ 2 & 1\end{pmatrix}$, indefinite (eigenvalues 3, −1) | fails: the matrix is not positive definite (pivot 1) | succeeds |
| $\begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix}$, semidefinite (singular) | fails: the matrix is not positive definite (pivot 1) | fails: the matrix is singular (zero pivot at step 1) |
| Hilbert matrix of order 12, positive definite with $\kappa \approx 10^{16}$ | succeeds | succeeds |
