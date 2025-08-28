import numpy as np

print("hello world...")

# ufuncs are c implementations for faster performance


rs = np.random.default_rng(0)

A = rs.integers(1, 10, size=(4, 4))
B = rs.standard_normal((4, 4))
# v = rs.integers(1, 5, size=(4, )) # same as 1 d array, size=(4), trailing comma dropped
# X = rs.standard_normal((3, 1))
# Y = rs.standard_normal((1, 4))
v = np.arange(1, 5, 1)  # 1d array, shape is (4) or (4,)
X = np.arange(1, 4, 1).reshape((3, 1))
Y = np.arange(2, 10, 2).reshape((1, 4))

W = rs.standard_normal((6, 50))  # 6 "word" embeddings, dim=50

# 1) Vectorized operations -----------------------------------------------------
# a) Elementwise: double A, sqrt of |B|, and affine transform 3*B - 2
A2 = A + A
Bsqrt = np.sqrt(np.abs(B))
BAff = B * 3 - 2

# b) Scalar vs elementwise: scale row-wise and col-wise using broadcasting
# Scale rows by v (shape (4,)) and columns by v[:,None]
# shape (4,) is 1d array in tuple representation, numpy can take an integer or tuple format for 1d array
rowscaled = A * v[None, :]
colscaled = A * v[:, None]

# 2) Universal functions (ufuncs) ---------------------------------------------
# a) Use np.add, np.multiply to compute A + B and A * B without operators
addAB_withops = A + B
mulAB_withops = A * B
addAB = np.add(A, B)
mulAB = np.multiply(A, B)


# b) Stable log(1+exp(z)) using ufuncs only (no loops, no conditionals)
logsum = np.log(1 + np.exp(B))
logsum_stable = np.log1p(np.exp(B))


# 3) Broadcasting rules --------------------------------------------------------
# a) Practice (3,1) + (1,4) -> (3,4)
xplusy = X + Y
yplusx = Y + X
X = X.ravel()
Y = Y.ravel()
xy_sum_withops = X[:, None] + Y
xy_sum = np.add.outer(X, Y)
yx_sum_withops = Y[:, None] + X
yx_sum = np.add.outer(Y, X)

# b) Given v shape (4,), make it (1,4) and (4,1) without copy using view
v_row = v[None, :]
v_col = v[:, None]

# c) Use v_col and v_row to build an outer-sum matrix of shape (4,4)
outer_sum = v_col + v_row

# 4) Advanced ops --------------------------------------------------------------
# a) np.where: replace negative B with 0, keep positive as-is
Bpos = np.where(B < 0, 0, B)


# b) np.clip: clip B to [-1, 1]
Bclip = np.clip(B, -1, 1)


# c) np.tile / np.repeat: 
# - Tile v to a (4,4) matrix by repeating across rows
# - Repeat each element of v 3 times -> shape (12,)
Vtile = np.tile(v, (4, 1))  # repeat v 4 times along rows, 1 time along cols, because v is (4,)
Vrep = np.repeat(v, 3)

# 5) Pairwise distances (broadcasting; no loops) -------------------------------
# For M in R^{nxd}, compute squared Euclidean distance matrix D in R^{n x n}
# Hint: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
M = rs.standard_normal((8, 5))
# TODO: D (shape (8,8)), all diagonal ~ 0 (numerical eps OK)
# Use only dot/broadcasting; no explicit loops.
row_norms = np.sum(M * M, axis=1)
D = row_norms[:, None] + row_norms[None, :] - 2 * M @ M.T
print(row_norms)

# arr = np.array([[1,2,3],[10, 20, 30]])
# print(arr[:,1])
# print(arr[1,:])
# print(arr[1: None])
# print(arr[None: 1])

# 6) Mini-project: cosine similarity (no loops) --------------------------------
# Given W in R^{m x d}, return S = cosine_similarity(W) in R^{m x m}
# S_ij = (Wi·Wj) / (||Wi|| * ||Wj||)
def cosine_similarity(X: np.ndarray) -> np.ndarray:
    # TODO:
    # 1) Compute dot matrix G = X @ X.T
    # 2) Row norms n (shape (m,)), then nnT via outer product
    # 3) Divide with broadcasting; guard division by zero with np.maximum
    G = X @ X.T
    # row_norms = np.sqrt(np.sum(X, axis=1)) 
    row_norms = np.linalg.norm(X, axis=1)
    product = row_norms[:, None] * row_norms[None, :]
    return G / np.maximum(product, 1e-8)
    #n = linalg.norm()

# c code of what is happening ^^^
#    double arr[2][3] = { {1, 2, 3}, {4, 5, 6} };
#    // Compute dot products
#    double G00 = arr[0][0]*arr[0][0] + arr[0][1]*arr[0][1] + arr[0][2]*arr[0][2];
#    double G01 = arr[0][0]*arr[1][0] + arr[0][1]*arr[1][1] + arr[0][2]*arr[1][2];
#    double G10 = arr[1][0]*arr[0][0] + arr[1][1]*arr[0][1] + arr[1][2]*arr[0][2];
#    double G11 = arr[1][0]*arr[1][0] + arr[1][1]*arr[1][1] + arr[1][2]*arr[1][2];

#    // Compute norms
#    double norm0 = sqrt(G00);
#    double norm1 = sqrt(G11);

#    // Compute cosine similarities
#    double S00 = G00 / (norm0 * norm0);
#    double S01 = G01 / (norm0 * norm1);
#    double S10 = G10 / (norm1 * norm0);
#    double S11 = G11 / (norm1 * norm1);

 #   printf("%f %f\n", S00, S01);
 #   printf("%f %f\n", S10, S11);



S = cosine_similarity(W)
# Basic checks
assert S.shape == (W.shape[0], W.shape[0])
assert np.allclose(np.diag(S), 1, atol=1e-6)  # self-similarity ~ 1
print("All tasks ran. Shapes:",
      dict(A2=A2.shape, Bsqrt=Bsqrt.shape, xy_sum=xy_sum.shape, D=D.shape, S=S.shape))