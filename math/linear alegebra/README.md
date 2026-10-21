# 00.1 Linear Algebra

## PLAN

### Day 1 — Vectors, Matrices & Computational Geometry

#### Core

1. Scalars
2. Vectors
3. Matrices
4. Tensors
5. Matrix shapes
6. Matrix indexing
7. Matrix addition
8. Scalar multiplication
9. Elementwise multiplication
10. Matrix multiplication
11. Matrix-vector multiplication
12. Broadcasting
13. Vectorization
14. Dense matrices
15. Sparse matrices

#### Mathematics

- Vector representation
- Matrix representation
- Matrix-vector multiplication
- Matrix-matrix multiplication
- Elementwise operation
- Shape algebra

#### Sub Material

- Vector spaces
- Coordinate systems

#### Algorithm Mechanism

- Matrix-vector multiplication
- Matrix-matrix multiplication

#### Implementation

- NumPy
- Vectorization
- Broadcasting

#### Diagnostics

- Matrix shape
- Sparse density

#### Failure Modes

- Shape mismatch
- Dense-memory explosion

#### From Scratch

- Matrix multiplication

#### Experiment

- Dense vs sparse memory

#### Case

**Linear regression geometry**

Mulai dari:

$$
y = X\beta
$$

Lalu pahami hubungan:

```text
feature vector → matrix X → parameter β → prediction y
```

Tujuan hari ini adalah membuat representasi matematis data terasa natural sebelum masuk ke struktur ruang vektor.

---

### Day 2 — Dot Product, Norms, Distance & Similarity

#### Core

16. Dot product
17. Inner product
18. Outer product
19. Norms
20. L1 norm
21. L2 norm
22. Infinity norm
23. Frobenius norm
24. Euclidean distance
25. Manhattan distance
26. Cosine similarity
27. Orthogonality
28. Orthonormality
29. Projection
30. Orthogonal projection matrix
31. Projection vs rejection

#### Mathematics

- `||x||₁`
- `||x||₂`
- `||x||∞`
- Euclidean distance
- Manhattan distance
- Cosine similarity
- Inner product
- Projection
- Orthogonal projection matrix

#### Sub Material

- Orthonormal basis
- Gram-Schmidt

#### Interpretation

- Vector direction
- Vector magnitude
- Projection

#### Comparison

- Euclidean vs Manhattan vs cosine
- Projection vs rejection

#### From Scratch

- Dot product
- Norms
- Distances
- Projection

#### Experiment

**Distance concentration**

Bandingkan Euclidean, Manhattan, dan cosine pada:

- dimensi rendah
- dimensi menengah
- dimensi tinggi

Hubungkan langsung ke:

#### Case — KNN distance

```text
vector representation → distance metric → nearest neighbor → prediction
```

Jadi KNN tidak dipelajari sebagai algoritma terpisah, tetapi sebagai konsekuensi langsung dari geometri vektor.

---

### Day 3 — Span, Basis, Independence & Subspaces

#### Core

32. Linear combination
33. Span
34. Basis
35. Dimension
36. Vector spaces
37. Subspaces
38. Coordinate systems
39. Change of basis
40. Linear independence
41. Column space
42. Row space
43. Null space
44. Left null space

#### Mathematics

- Linear combination

$$
x = \sum_i c_i v_i
$$

- Span
- Basis
- Dimension
- Linear independence

#### Sub Material

- Vector spaces
- Subspaces
- Coordinate systems
- Change of basis

#### Interpretation

- Vector direction
- Rank as dimensionality

#### Algorithm Mechanism

- Gaussian elimination

#### Diagnostics

- Matrix rank

#### Failure Modes

- Rank deficiency

#### From Scratch

- Gaussian elimination

#### Experiment

**Multicollinearity**

Buat feature:

$$
x_3 = x_1 + x_2
$$

Kemudian lihat:

```text
linear dependence → rank deficiency → redundant information → unstable solution
```

#### Case — Linear regression geometry

Hubungkan:

```text
columns of X → feature space → column space → representable predictions
```

hingga memahami mengapa regresi tidak hanya tentang mencari angka β, tetapi mencari representasi y di dalam **column space X**.

---

### Day 4 — Rank, Null Space, Inverse & Pseudoinverse

#### Core

45. Rank
46. Nullity
47. Rank-nullity theorem
48. Matrix inverse
49. Pseudoinverse
50. Determinant
51. Trace

#### Mathematics

- Rank-nullity

$$
\text{rank}(A)+\text{nullity}(A)=n
$$

- Matrix inverse

$$
A^{-1}A=I
$$

- Pseudoinverse

$$
A^+
$$

#### Sub Material

- Matrix rank
- Numerical rank

#### Assumptions

- Full-rank design matrix
- Invertibility

#### Comparison

- Inverse vs pseudoinverse

#### Diagnostics

- Rank
- Singular value spectrum

#### Failure Modes

- Singular matrix
- Rank deficiency

#### From Scratch

- Gaussian elimination
- Least-squares solver

#### Experiment

**Condition-number instability**

Bangun matrix yang mendekati singular lalu observasi:

```text
small perturbation → huge solution change
```

#### Case — Least-squares projection

Hubungkan:

$$
A^+b
$$

dengan kondisi matrix dan keberadaan solusi.

Fokus utama hari ini:

```text
inverse → ketika bisa dipakai
```

vs

```text
pseudoinverse → ketika sistem tidak square / singular / overdetermined
```

---

### Day 5 — Matrix Structure, Quadratic Forms & Positive Definiteness

#### Core

52. Symmetric matrices
53. Orthogonal matrices
54. Diagonal matrices
55. Triangular matrices
56. Sparse matrices
57. Dense matrices
58. Positive definite matrices
59. Positive semidefinite matrices
60. Quadratic forms

#### Mathematics

- Quadratic form

$$
x^TAx
$$

- Symmetry

$$
A=A^T
$$

- Positive definiteness

$$
x^TAx>0
$$

#### Sub Material

- Gram matrices
- Kernel matrices

#### Statistics

- Covariance matrices
- Gram matrices
- Correlation matrices

#### Interpretation

- Quadratic form as geometry
- Positive eigenvalue structure

#### Optimization

- Quadratic minimization

#### Algorithm Mechanism

- Cholesky decomposition

#### Assumptions

- Positive definiteness
- Symmetry

#### Comparison

- LU vs QR vs Cholesky

#### Diagnostics

- Eigenvalue spectrum
- Condition number

#### Failure Modes

- Covariance inversion instability

#### Experiment

**Covariance matrix geometry**

Generate correlated variables lalu lihat:

```text
covariance matrix → ellipse → principal direction → eigenvectors
```

#### Case — Mahalanobis financial anomaly

Hubungkan:

$$
d_M(x,\mu)=
\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

dengan:

```text
covariance → scale → correlation → geometry → anomaly distance
```

---

### Day 6 — Orthogonalization, QR & Least Squares

#### Core

61. Gram-Schmidt
62. Orthonormal basis
63. QR decomposition
64. Least-squares geometry
65. Normal equations
66. Least-squares estimator
67. Residual orthogonality

#### Mathematics

- Projection

$$
\hat y = X\hat\beta
$$

- Normal equations

$$
X^TX\hat\beta=X^Ty
$$

- Least-squares estimator

$$
\hat\beta=(X^TX)^{-1}X^Ty
$$

- Residual orthogonality

$$
X^Tr=0
$$

#### Algorithm Mechanism

- QR decomposition
- Iterative least squares

#### Optimization

- Least-squares optimization

$$
\min_\beta ||X\beta-y||_2^2
$$

#### Assumptions

- Full-rank design matrix

#### Diagnostics

- Residual norm
- Projection error

#### Comparison

- OLS vs QR

#### From Scratch

- Gram-Schmidt
- QR
- Least-squares solver

#### Experiment

**OLS vs QR**

Bandingkan:

```text
normal equation → inverse-related instability
```

dengan:

```text
QR → orthogonal basis → numerically safer computation
```

#### Case — Linear regression geometry

Full chain:

```text
data → X → column space → projection → QR → β̂ → residual
```

Hari ini menjadi jembatan utama antara **linear algebra dan machine learning**.

---

### Day 7 — Eigenvalues, Eigenvectors & Spectral Geometry

#### Core

68. Eigenvalues
69. Eigenvectors
70. Eigenvalue equation
71. Eigendecomposition
72. Spectral decomposition
73. Spectral theorem
74. Rayleigh quotient
75. Power iteration
76. Inverse iteration

#### Mathematics

$$
Av=\lambda v
$$

- Eigendecomposition

$$
A=Q\Lambda Q^{-1}
$$

- Rayleigh quotient

$$
R(x)=\frac{x^TAx}{x^Tx}
$$

#### Sub Material

- Spectral decomposition
- Singular values
- Eigenvalue spectrum
- Variance directions

#### Algorithm Mechanism

- Power iteration
- Inverse iteration

#### Optimization

- Rayleigh quotient
- Eigenvalue optimization
- PCA variance maximization

#### Diagnostics

- Eigenvalue spectrum

#### Interpretation

- Eigenvector direction
- Eigenvalue magnitude
- Variance directions

#### Experiment

**PCA via eigendecomposition**

Lakukan:

```text
covariance matrix → eigenvectors → eigenvalues → principal directions
```

#### Case — PCA geometry

Hubungkan langsung:

```text
data cloud → covariance → eigenvectors → principal axes → explained variance
```

---

### Day 8 — SVD, Low Rank & PCA

#### Core

77. Singular Value Decomposition
78. Singular values
79. Low-rank approximation
80. Numerical rank
81. Randomized SVD
82. Exact vs randomized SVD

#### Mathematics

$$
A=U\Sigma V^T
$$

- SVD
- Pseudoinverse via SVD

$$
A^+=V\Sigma^+U^T
$$

- Low-rank approximation

$$
A_k=U_k\Sigma_kV_k^T
$$

#### Sub Material

- Singular values
- Numerical rank

#### Algorithm Mechanism

- Randomized SVD
- Low-rank approximation

#### Optimization

- PCA reconstruction-error minimization

#### Hyperparameters

- Rank
- Number of retained components
- Randomized SVD oversampling

#### Diagnostics

- Singular value spectrum
- Reconstruction error

#### Comparison

- Eigendecomposition vs SVD
- Exact vs randomized SVD

#### From Scratch

- PCA
- PCA via SVD

#### Experiment

**Singular-value decay**

Bandingkan:

```text
full rank matrix → singular spectrum → effective rank → compression
```

#### Experiment

**PCA via eigendecomposition vs PCA via SVD**

Verifikasi bahwa kedua pendekatan menghasilkan struktur principal components yang ekuivalen secara matematis, dengan perbedaan implementasi dan numeriknya.

#### Case — PCA geometry

```text
covariance → SVD/eigen → principal components → dimensionality reduction → reconstruction
```

---

### Day 9 — Numerical Linear Algebra, Stability & High-Dimensional Geometry

#### Core

83. Condition number
84. Numerical stability
85. Floating-point precision
86. Sparse linear algebra
87. Iterative least squares
88. Memory mapping
89. Batch matrix operations
90. High-dimensional geometry
91. Distance concentration

#### Mathematics

- Condition number

$$
\kappa(A)=||A||\,||A^{-1}||
$$

- Frobenius norm

$$
||A||_F
$$

- Mahalanobis distance

#### Sub Material

- Graph matrices
- Laplacian matrices

#### Statistics

- Multivariate geometry
- Elliptical distributions
- Mahalanobis geometry
- Multicollinearity
- Condition number

#### Implementation

- SciPy
- BLAS
- LAPACK
- CSR
- CSC
- COO
- Memory mapping
- Batch matrix operations

#### Assumptions

- Numerical stability
- Suitable metric geometry
- Meaningful feature scale

#### Hyperparameters

- Solver tolerance
- Maximum iterations
- Floating-point precision

#### Diagnostics

- Condition number
- Numerical precision
- Sparse density

#### Failure Modes

- Ill-conditioning
- Numerical overflow
- Numerical underflow
- Catastrophic cancellation
- Dense-memory explosion
- Distance concentration
- Covariance inversion instability

#### Experiment

**Condition-number instability**

```text
κ(A) ↑ → sensitivity ↑ → numerical error ↑
```

#### Experiment

**Distance concentration**

```text
dimension ↑ → distance differences shrink → nearest-neighbor geometry degrades
```

#### Experiment

**Dense vs sparse memory**

Bandingkan dense matrix dan CSR/CSC/COO pada matrix dengan sparsity tinggi.

#### Case — KNN distance

Hubungkan kembali:

```text
high-dimensional representation → metric geometry → distance concentration → KNN limitations
```

#### Case — Mahalanobis financial anomaly

Hubungkan:

```text
covariance estimation → conditioning → inverse stability → Mahalanobis distance
```

---

### Day 10 — Integration: Linear Algebra → Statistics → Optimization → ML

#### Core

92. Matrix calculus
93. Covariance matrix
94. Correlation matrix
95. Gram matrices
96. Kernel matrices
97. Graph matrices
98. Laplacian matrices
99. Quadratic minimization
100. Ridge-regularized linear systems
101. PCA variance maximization
102. PCA reconstruction-error minimization
103. Constrained projection
104. Matrix rank
105. Numerical rank
106. Singular values
107. Eigenvalue spectrum
108. Reconstruction error
109. Projection error

#### Mathematics

- Mean vector
- Covariance matrix
- Correlation matrix
- Variance as covariance diagonal
- Quadratic form
- Matrix calculus
- Normal equations
- Least-squares estimator
- Residual orthogonality
- Mahalanobis distance

#### Optimization

- Quadratic minimization
- Least-squares optimization
- Rayleigh quotient
- Eigenvalue optimization
- PCA variance maximization
- PCA reconstruction-error minimization
- Ridge-regularized linear systems
- Constrained projection

#### Hyperparameters

- Rank
- Number of retained components
- Solver tolerance
- Maximum iterations
- Regularization strength
- Randomized SVD oversampling
- Floating-point precision

#### Diagnostics

- Matrix shape
- Rank
- Singular value spectrum
- Eigenvalue spectrum
- Condition number
- Residual norm
- Reconstruction error
- Projection error
- Numerical precision
- Sparse density

#### Comparison

- Inverse vs pseudoinverse
- LU vs QR vs Cholesky
- Eigendecomposition vs SVD
- Dense vs sparse
- Exact vs randomized SVD
- Euclidean vs Manhattan vs cosine
- Projection vs rejection

#### Implementation

- NumPy
- SciPy
- BLAS
- LAPACK
- CSR
- CSC
- COO
- Vectorization
- Broadcasting
- Memory mapping
- Batch matrix operations

#### From Scratch

- Dot product
- Matrix multiplication
- Norms
- Distances
- Projection
- Gram-Schmidt
- Gaussian elimination
- LU
- QR
- Cholesky
- PCA
- Covariance matrix
- Least-squares solver

#### Experiments

**Multicollinearity**

$$
X^TX
$$

amati bagaimana korelasi antar-feature memengaruhi conditioning.

**Condition-number instability**

$$
\kappa(X^TX)
$$

bandingkan dengan QR.

**OLS vs QR**

```text
same problem → different numerical route
```

**PCA via eigendecomposition**

```text
covariance → eigenvectors/eigenvalues
```

**PCA via SVD**

```text
data matrix → singular vectors/values
```

**Singular-value decay**

```text
σ₁ ≥ σ₂ ≥ ... ≥ σ_n
```

**Dense vs sparse memory**

```text
representation → memory → computational cost
```

**Distance concentration**

```text
dimension → geometry → ML behavior
```

#### Case 1 — Linear regression geometry

$$
\hat y=X\hat\beta
$$

```text
feature space → column space → projection → residual
```

---

#### Case 2 — PCA geometry

$$
\Sigma v=\lambda v
$$

```text
covariance → eigenvector → variance direction → principal component
```

---

#### Case 3 — KNN distance

$$
d(x_i,x_j)
$$

```text
vector representation → metric → nearest neighbor → concentration problem
```

---

#### Case 4 — SVM hyperplane

$$
w^Tx+b=0
$$

```text
inner product → orthogonality → projection → hyperplane → margin
```

---

#### Case 5 — Mahalanobis financial anomaly

$$
d_M(x,\mu)
=
\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

```text
covariance → quadratic form → inverse → scale-aware distance → anomaly
```

---

#### Case 6 — Least-squares projection

$$
\hat y=P_Xy
$$

dengan

$$
P_X=X(X^TX)^{-1}X^T
$$

```text
column space → orthogonal projection → least squares → residual
```

---

## FINAL INTEGRATION

Pada akhir 10 hari, seluruh materi dipahami sebagai satu sistem:

```text
SCALAR
  ↓
VECTOR
  ↓
MATRIX
  ↓
VECTOR SPACE
  ↓
SUBSPACE
  ↓
BASIS
  ↓
LINEAR COMBINATION
  ↓
ORTHOGONALITY
  ↓
PROJECTION
  ↓
RANK / NULL SPACE
  ↓
MATRIX DECOMPOSITION
  ├── LU
  ├── QR
  ├── Cholesky
  ├── Eigendecomposition
  └── SVD
        ↓
COVARIANCE
        ↓
EIGENVECTORS / SINGULAR VECTORS
        ↓
PCA
        ↓
LOW-RANK REPRESENTATION
        ↓
LEAST SQUARES
        ↓
OPTIMIZATION
        ↓
NUMERICAL STABILITY
        ↓
HIGH-DIMENSIONAL GEOMETRY
        ↓
MACHINE LEARNING
```

Dan hubungan **math → algorithm → experiment → case** terlihat sebagai:

```text
Inner Product
    ↓
Dot Product
    ↓
Projection
    ↓
Least Squares
    ↓
Linear Regression
```

```text
Covariance Matrix
    ↓
Symmetric / PSD Matrix
    ↓
Eigenvalues + Eigenvectors
    ↓
Spectral Decomposition
    ↓
PCA
```

```text
Matrix
    ↓
Singular Values
    ↓
SVD
    ↓
Low-Rank Approximation
    ↓
Dimensionality Reduction
```

```text
Covariance
    ↓
Quadratic Form
    ↓
Mahalanobis Distance
    ↓
Anomaly Detection
```

```text
Vector Distance
    ↓
Euclidean / Manhattan / Cosine
    ↓
High-Dimensional Geometry
    ↓
Distance Concentration
    ↓
KNN Behavior
```

```text
Linear Independence
    ↓
Rank
    ↓
Rank Deficiency
    ↓
Multicollinearity
    ↓
Condition Number
    ↓
Numerical Instability
```

```text
Orthogonality
    ↓
Gram-Schmidt
    ↓
QR
    ↓
Stable Least Squares
    ↓
OLS vs QR Experiment
```

## SUB MATERIAL

- Vector spaces
- Subspaces
- Coordinate systems
- Change of basis
- Gram-Schmidt
- Orthonormal basis
- Spectral decomposition
- Frobenius norm
- Matrix rank
- Numerical rank
- Singular values
- Covariance matrices
- Gram matrices
- Kernel matrices
- Graph matrices
- Laplacian matrices

## MATHEMATICS

- `||x||₁`
- `||x||₂`
- `||x||∞`
- Euclidean distance
- Manhattan distance
- Cosine similarity
- Inner product
- Projection
- Orthogonal projection matrix
- Quadratic form
- Rank-nullity
- Eigenvalue equation
- Eigendecomposition
- SVD
- Pseudoinverse
- Normal equations
- Least-squares estimator
- Residual orthogonality
- Mahalanobis distance

## STATISTICS

- Mean vector
- Covariance matrix
- Correlation matrix
- Variance as covariance diagonal
- Eigenvalue spectrum
- Variance directions
- Multivariate geometry
- Elliptical distributions
- Mahalanobis geometry
- Multicollinearity
- Condition number

## ALGORITHM MECHANISM

- Matrix-vector multiplication
- Matrix-matrix multiplication
- Gaussian elimination
- LU decomposition
- QR decomposition
- Cholesky decomposition
- Power iteration
- Inverse iteration
- Iterative least squares
- Randomized SVD
- Sparse linear algebra
- Low-rank approximation

## OPTIMIZATION

- Quadratic minimization
- Least-squares optimization
- Rayleigh quotient
- Eigenvalue optimization
- PCA variance maximization
- PCA reconstruction-error minimization
- Ridge-regularized linear systems
- Constrained projection

## ASSUMPTIONS

- Full-rank design matrix
- Invertibility
- Positive definiteness
- Symmetry
- Numerical stability
- Suitable metric geometry
- Meaningful feature scale

## HYPERPARAMETERS

- Rank
- Number of retained components
- Solver tolerance
- Maximum iterations
- Regularization strength
- Randomized SVD oversampling
- Floating-point precision

## DIAGNOSTICS

- Matrix shape
- Rank
- Singular value spectrum
- Eigenvalue spectrum
- Condition number
- Residual norm
- Reconstruction error
- Projection error
- Numerical precision
- Sparse density

## FAILURE MODES

- Singular matrix
- Rank deficiency
- Ill-conditioning
- Multicollinearity
- Numerical overflow
- Numerical underflow
- Catastrophic cancellation
- Dense-memory explosion
- Shape mismatch
- Distance concentration
- Covariance inversion instability

## INTERPRETATION

- Vector direction
- Vector magnitude
- Projection
- Eigenvector direction
- Eigenvalue magnitude
- Singular value importance
- Rank as dimensionality
- Residual as orthogonal error
- Condition number as sensitivity

## COMPARISON

- Inverse vs pseudoinverse
- LU vs QR vs Cholesky
- Eigendecomposition vs SVD
- Dense vs sparse
- Exact vs randomized SVD
- Euclidean vs Manhattan vs cosine
- Projection vs rejection

## IMPLEMENTATION

- NumPy
- SciPy
- BLAS
- LAPACK
- CSR
- CSC
- COO
- Vectorization
- Broadcasting
- Memory mapping
- Batch matrix operations

## FROM SCRATCH

- Dot product
- Matrix multiplication
- Norms
- Distances
- Projection
- Gram-Schmidt
- Gaussian elimination
- LU
- QR
- Cholesky
- PCA
- Covariance matrix
- Least-squares solver

## EXPERIMENTS

- Distance concentration
- Multicollinearity
- Condition-number instability
- OLS vs QR
- PCA via eigendecomposition
- PCA via SVD
- Singular-value decay
- Dense vs sparse memory

## CASE

1. Linear regression geometry
2. PCA geometry
3. KNN distance
4. SVM hyperplane
5. Mahalanobis financial anomaly
6. Least-squares projection