import numpy as np
rs = np.random.default_rng(42)

# ----- Setup -----
A = rs.standard_normal((5, 5))
B = rs.integers(1, 10, size=(5, 5))
v = rs.standard_normal((5,))
X = rs.standard_normal((8, 3))
y = rs.standard_normal((8,))
M = rs.standard_normal((6, 6))

# 1) Matrix operations ---------------------------------------------------------
# a) Compute dot product of A and B using np.dot and @
# TODO: dot1, dot2
dot1 = np.dot(A, B.astype(np.float32))
dot2 = A @ B.astype(np.float32)

# b) Compute matrix-vector product of A and v
# TODO: Av
Av = A @ v

# c) Compute L2 norm of v and angle (in radians) between v and another random vector w
w = rs.standard_normal((5,))
# TODO: v_norm, w_norm, angle
v_norm = np.linalg.norm(v, order=2)
# w_norm = np.sqrt(np.sum(w**2))
w_norm = np.linalg.norm(w, order=2)
angle = np.arccos(np.dot(v, w) / (v_norm * w_norm))

# 2) Decompositions ------------------------------------------------------------
# a) SVD of A: U, S, Vt = np.linalg.svd(A)
# TODO: U, S, Vt
U, S, Vt = ...

# b) Eigenvalues and eigenvectors of M
# TODO: eigvals, eigvecs
eigvals, eigvecs = ...

# 3) Solving Linear Systems ----------------------------------------------------
# a) Solve Ax = v for x (assume A is invertible)
# TODO: x_solve
x_solve = ...

# b) Compute inverse and pseudo-inverse of A
# TODO: A_inv, A_pinv
A_inv = ...
A_pinv = ...

# 4) Orthogonalization ---------------------------------------------------------
# a) QR decomposition of X
# TODO: Q, R
Q, R = ...

# b) Use QR to solve least-squares: X @ theta ≈ y
# TODO: theta_qr
theta_qr = ...

# 5) Mini-project: Linear Regression -------------------------------------------
# Implement Linear Regression using the Normal Equation:
# θ = (X.T @ X)^-1 @ X.T @ y
# TODO: theta_normal
theta_normal = ...

# Validate vs. np.linalg.lstsq
# TODO: theta_lstsq
theta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)

# Check that both solutions are close
assert np.allclose(theta_normal, theta_lstsq, atol=1e-6)
print("All tasks ran. Shapes:", dict(dot1=dot1.shape, U=U.shape, eigvals=eigvals.shape, Q=Q.shape, theta_normal=theta_normal.shape))
