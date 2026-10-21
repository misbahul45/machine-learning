# 00.1 — Advanced Linear Algebra for Deep Learning & LLMs

> **Track:** Mathematics → Numerical Algorithms → Deep Learning Systems → Transformer/LLM Foundations  
> **Level:** Advanced undergraduate / graduate bridge; assumes fluency with elementary linear algebra  
> **Objective:** Understand not only what matrix operations do, but why their algorithms work, when their numerical results can be trusted, how their computational cost scales, and how they appear inside neural networks and large language models.

---

## Table of Contents

- [0. Position of This Module in the Learning Path](#0-position-of-this-module-in-the-learning-path)
- [1. Study Protocol](#1-study-protocol-derive--implement--verify--stress-test--connect)
- [2. Mathematical and Computational Conventions](#2-mathematical-and-computational-conventions)
- [Part I — Operators, Structure, and Geometry](#part-i--operators-structure-and-geometry)
  - [3. Linear Operators and Invariants](#3-linear-operators-and-invariants)
  - [4. Matrix Norms, Operator Gain, and Perturbation Geometry](#4-matrix-norms-operator-gain-and-perturbation-geometry)
  - [5. Spectral Structure and Matrix Classes](#5-spectral-structure-and-matrix-classes)
- [Part II — Factorizations and Stable Solvers](#part-ii--factorizations-and-stable-solvers)
  - [6. Factorizations as Algorithmic Strategies](#6-factorizations-as-algorithmic-strategies)
  - [7. Least Squares, Projections, and Regularized Inverse Problems](#7-least-squares-projections-and-regularized-inverse-problems)
- [Part III — Eigenvalues, SVD, Low Rank, and Randomization](#part-iii--eigenvalues-svd-low-rank-and-randomization)
  - [8. Eigenvalues, Singular Values, and Spectral Diagnostics](#8-eigenvalues-singular-values-and-spectral-diagnostics)
  - [9. Low-Rank Approximation and PCA](#9-low-rank-approximation-and-pca)
  - [10. Matrix Perturbation, Numerical Rank, and Backward Stability](#10-matrix-perturbation-numerical-rank-and-backward-stability)
- [Part IV — Matrix Calculus and Learning Dynamics](#part-iv--matrix-calculus-and-learning-dynamics)
  - [11. Matrix Differential Calculus](#11-matrix-differential-calculus)
  - [12. Jacobian Products, Conditioning, and Deep Signal Propagation](#12-jacobian-products-conditioning-and-deep-signal-propagation)
- [Part V — Direct Bridge to Transformers and LLMs](#part-v--direct-bridge-to-transformers-and-llms)
  - [13. Linear Algebra Inside a Transformer Block](#13-linear-algebra-inside-a-transformer-block)
  - [14. Low-Rank Adaptation, Matrix Compression, and Parameter Efficiency](#14-low-rank-adaptation-matrix-compression-and-parameter-efficiency)
- [Part VI — Sparse, Structured, and Large-Scale Linear Algebra](#part-vi--sparse-structured-and-large-scale-linear-algebra)
  - [15. Sparse Operators and Structured Matrices](#15-sparse-operators-and-structured-matrices)
  - [16. High-Dimensional Geometry, Kernels, and Similarity](#16-high-dimensional-geometry-kernels-and-similarity)
- [Part VII — Hardware-Aware Linear Algebra](#part-vii--hardware-aware-linear-algebra)
  - [17. Complexity, Memory Traffic, and Kernel Behavior](#17-complexity-memory-traffic-and-kernel-behavior)
- [Part VIII — Implementation Labs From Scratch](#part-viii--implementation-labs-from-scratch)
  - [18. Required Algorithms and Acceptance Criteria](#18-required-algorithms-and-acceptance-criteria)
  - [19. Experiment Design](#19-experiment-design-claims-must-be-falsifiable)
- [Part IX — Case Studies](#part-ix--case-studies)
- [Part X — A Dependency-Driven Advanced Roadmap](#part-x--a-dependency-driven-advanced-roadmap)
- [Part XI — Knowledge Checks and Mastery Gates](#part-xi--knowledge-checks-and-mastery-gates)
- [Part XII — Project Structure and Reproducibility](#part-xii--project-structure-and-reproducibility)
- [Part XIII — Master Concept Map](#part-xiii--master-concept-map)
- [Part XIV — Final Mastery Standard](#part-xiv--final-mastery-standard)

---

## 0. Position of This Module in the Learning Path

Linear algebra is not a vocabulary checkpoint. It is the language for representing learnable functions, analyzing optimization geometry, understanding information compression, and controlling numerical error. For an LLM, the same mathematics appears in embedding tables, projection matrices, attention scores, MLP blocks, normalization, output logits, gradient propagation, low-rank adaptation, quantization, and distributed training.

This README replaces a topic-by-topic introductory sequence with a **dependency-driven advanced curriculum**. Basic scalar/vector/matrix operations remain prerequisites, but they are not treated as the main learning outcome.

### What this module must enable

By completion, you should be able to:

1. Derive the important identities used by linear regression, PCA, least squares, neural-network backpropagation, and attention.
2. Select numerical algorithms based on matrix structure, conditioning, memory cost, and accuracy requirements—not habit.
3. Implement core algorithms from scratch, validate them against trusted libraries, and explain discrepancies.
4. Diagnose rank deficiency, ill-conditioning, cancellation, overflow/underflow, gradient instability, and low-precision error.
5. Estimate time and memory complexity for dense, sparse, batched, and low-rank operations.
6. Explain why SVD, QR, eigensystems, Krylov methods, matrix calculus, and random projections matter to practical ML.
7. Connect mathematical abstractions to the actual tensor shapes and data flow in a Transformer.
8. Produce reproducible experiments that support or reject a mathematical claim.

### Prerequisite mastery

Do not spend an entire study day re-learning these items. Verify that you can use them without looking them up:

- Matrix and tensor shapes; transposition; broadcasting; batched matrix multiplication.
- Linear combinations, span, basis, rank, null space, orthogonality, and projection.
- Dot products, common norms, distance metrics, and basic complex-number notation.
- Derivatives, partial derivatives, chain rule, gradients, and basic probability/statistics.
- Python, NumPy array semantics, plotting, unit testing, and Big-O notation.

If a prerequisite is weak, repair it on demand while studying the advanced topic that depends on it.

---

## 1. Study Protocol: Derive → Implement → Verify → Stress-Test → Connect

For every major concept, complete the same five passes.

### Pass A — Derive

- State the problem and its assumptions.
- Define dimensions and domains explicitly.
- Derive the result from definitions or previously established identities.
- Distinguish an exact mathematical statement from an approximation or numerical heuristic.
- Identify conditions under which the result fails or becomes non-unique.

### Pass B — Implement

- Write a small, transparent NumPy reference implementation.
- Implement the central algorithm with explicit loops where the purpose is to expose the mechanism.
- Keep the scratch implementation separate from the optimized implementation.
- Do not use a library decomposition inside an implementation that is supposed to teach that decomposition.

### Pass C — Verify

- Compare with an independent reference such as NumPy/SciPy/PyTorch.
- Test dimensions, degenerate inputs, rank-deficient inputs, and different numerical scales.
- Use residuals and invariant checks rather than checking only one expected output.
- Use tolerances appropriate to dtype and conditioning; do not require bitwise equality from floating-point algorithms.

### Pass D — Stress-test

- Vary matrix size, rank, condition number, sparsity, and dtype.
- Measure wall time and peak memory separately from mathematical error.
- Repeat experiments with fixed seeds and report environment details.
- Plot error against a meaningful control variable, such as condition number or retained rank.

### Pass E — Connect

Explain where the concept is used in a real learning system. At minimum, connect every major topic to one or more of:

- Parameterized linear maps and MLP layers.
- Backpropagation and gradient propagation.
- Attention and embedding geometry.
- Optimization and conditioning.
- Low-rank approximation, compression, or quantization.
- Numerical kernels and accelerator execution.

A topic is not mastered merely because its formula can be recalled.

---

## 2. Mathematical and Computational Conventions

Unless explicitly stated otherwise:

- Vectors are columns: \(x\in\mathbb{R}^{n}\).
- Matrices act on vectors from the left: \(y=Ax\), with \(A\in\mathbb{R}^{m\times n}\).
- The transpose is \(A^T\); the conjugate transpose for complex matrices is \(A^*\).
- \(I\) denotes an identity matrix of the required dimension.
- \(\|A\|_2\) is the spectral/operator norm; \(\|A\|_F\) is the Frobenius norm.
- \(\sigma_i(A)\) denotes the singular values in descending order.
- \(\lambda_i(A)\) denotes eigenvalues, with the ordering and field stated when relevant.
- \(\operatorname{rank}(A)\) is exact rank in mathematics; **numerical rank** is tolerance-dependent.
- \(\epsilon_{\text{mach}}\) denotes machine precision, while \(u\) often denotes unit roundoff. State which convention is used in code.
- \(\kappa_2(A)=\|A\|_2\|A^{-1}\|_2\) for nonsingular square \(A\); for full-column-rank rectangular matrices, the 2-norm condition number is \(\sigma_{\max}(A)/\sigma_{\min}(A)\).
- Complexity estimates distinguish arithmetic operations, memory traffic, and storage.

For complex-valued material, replace transposes by conjugate transposes where required. Deep-learning examples will usually use real-valued tensors, but the complex case is useful for understanding spectral theory and signal-processing connections.

---

# PART I — OPERATORS, STRUCTURE, AND GEOMETRY

## 3. Linear Operators and Invariants

### 3.1 From arrays to linear maps

Treat a matrix as a representation of a linear operator between vector spaces, not merely a rectangular collection of numbers.

For \(T:V\to W\), linearity means

\[
T(\alpha x+\beta y)=\alpha T(x)+\beta T(y).
\]

For a finite-dimensional representation \(T(x)=Ax\), investigate:

- **Kernel/null space:** \(\ker(A)=\{x:Ax=0\}\).
- **Image/column space:** \(\operatorname{im}(A)=\{Ax:x\in\mathbb{R}^n\}\).
- **Rank-nullity:** \(\dim\ker(A)+\operatorname{rank}(A)=n\).
- **Injectivity:** equivalent to zero null space for finite-dimensional linear maps.
- **Surjectivity:** equivalent to rank equal to the output dimension.
- **Invertibility:** bijectivity, requiring a square operator in the ordinary finite-dimensional setting.

### 3.2 Invariants under a change of coordinates

Given an invertible basis transformation \(S\), a linear operator changes representation by similarity:

\[
A_{\text{new}}=S^{-1}AS.
\]

The entries change, but properties such as characteristic polynomial, determinant, trace, eigenvalues, and rank remain invariant. For an orthogonal coordinate change \(Q\), \(Q^{-1}=Q^T\), which preserves Euclidean lengths and angles.

**Advanced exercise:** Construct two visibly different matrices representing the same operator under a basis change. Verify the invariant quantities numerically and explain why singular values are generally not invariant under arbitrary similarity transformations.

### 3.3 Adjoint and dual viewpoints

For the Euclidean inner product, the adjoint of a real matrix is \(A^T\), satisfying

\[
\langle Ax,y\rangle=\langle x,A^Ty\rangle.
\]

This identity is central to reverse-mode differentiation: the backward pass propagates cotangents through the adjoint of the forward linear map.

For \(y=Ax\), a perturbation \(\delta x\) induces \(\delta y=A\delta x\). Under a scalar loss with upstream gradient \(g_y=\nabla_y L\),

\[
\nabla_x L=A^Tg_y.
\]

The transpose in backpropagation is not an arbitrary coding convention; it follows from the adjoint relationship and the chain rule.

**From-scratch task:** Implement a linear layer forward pass and its vector-Jacobian product. Validate gradients with finite differences for tiny random inputs.

---

## 4. Matrix Norms, Operator Gain, and Perturbation Geometry

### 4.1 Norms as measurement rules

For a vector \(x\),

\[
\|x\|_p=\left(\sum_i |x_i|^p\right)^{1/p},\quad p\geq1,
\]

with the usual maximum-norm definition for \(p=\infty\). For matrices, distinguish entrywise norms from induced/operator norms. In particular,

\[
\|A\|_2=\max_{\|x\|_2=1}\|Ax\|_2=\sigma_{\max}(A),
\]

and

\[
\|A\|_F=\sqrt{\sum_{ij}|a_{ij}|^2}
=\sqrt{\operatorname{tr}(A^*A)}.
\]

Do not assume \(\|A\|_F=\|A\|_2\). The Frobenius norm aggregates all singular directions, while the spectral norm measures the largest gain along one direction.

### 4.2 Submultiplicativity and error propagation

For compatible matrices and a consistent operator norm,

\[
\|AB\|\leq\|A\|\,\|B\|.
\]

Therefore, a chain of linear maps can amplify perturbations according to the product of operator norms. This gives a first diagnostic for exploding or vanishing signals through a deep composition:

\[
\|A_L\cdots A_2A_1x\|\leq
\left(\prod_{\ell=1}^{L}\|A_\ell\|_2\right)\|x\|_2.
\]

This is an upper bound, not a complete prediction: alignment between singular directions and nonlinear Jacobians matters.

### 4.3 Condition numbers

For the linear system \(Ax=b\), conditioning describes sensitivity of the mathematical solution to perturbations in the input. For nonsingular \(A\),

\[
\kappa_2(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.
\]

A large condition number means that some perturbations may be strongly amplified. It does **not** by itself prove that a particular computed solution is inaccurate; backward stability, data perturbation, and the direction of the error also matter.

A common first-order perspective is

\[
\frac{\|\delta x\|}{\|x\|}
\lesssim \kappa(A)\frac{\|\delta b\|}{\|b\|},
\]

subject to the assumptions and perturbation model for the system under consideration.

### 4.4 Condition number is not one universal scalar for every problem

Different tasks have different condition measures:

- Solving \(Ax=b\): sensitivity of the solution to perturbations in \(A\) and/or \(b\).
- Eigenvalues: eigenvalue-specific conditioning; nonnormal matrices may be especially sensitive.
- Least squares: conditioning of the design matrix and the chosen algorithm.
- Matrix products: error depends on accumulation order, precision, and cancellation.
- Neural networks: local Jacobian gain, depth, normalization, residual pathways, and data distribution jointly affect signal propagation.

**Experiment — controlled ill-conditioning**

1. Generate \(A=U\operatorname{diag}(1,\ldots,10^{-k})V^T\) from orthogonal \(U,V\).
2. Set \(b=Ax_*\) for a known vector \(x_*\).
3. Perturb \(b\) by controlled magnitudes and solve.
4. Compare solution error with residual error and report \(\kappa_2(A)\).
5. Repeat with float64 and float32; optionally use float16/bfloat16 only where the platform supports meaningful behavior.
6. Explain why a small residual does not guarantee a small forward error for an ill-conditioned system.

---

## 5. Spectral Structure and Matrix Classes

Do not memorize matrix classes as disconnected vocabulary. For each class, know its defining property, what theorem becomes available, which algorithm exploits it, and which assumptions are required.

### 5.1 Symmetric and Hermitian matrices

For real symmetric \(A=A^T\), the spectral theorem gives an orthogonal eigendecomposition:

\[
A=Q\Lambda Q^T,\qquad Q^TQ=I.
\]

Eigenvalues are real, and eigenvectors associated with distinct eigenvalues are orthogonal. For complex Hermitian matrices, use \(A=A^*\) and a unitary eigenvector matrix.

### 5.2 Positive definite and positive semidefinite matrices

A real symmetric matrix is positive definite if

\[
x^TAx>0\quad\forall x\ne0,
\]

and positive semidefinite if \(x^TAx\geq0\). Equivalent characterizations under the appropriate finite-dimensional assumptions include positive eigenvalues/nonnegative eigenvalues. Positive definiteness enables Cholesky factorization and supports strict convexity in quadratic optimization.

A Gram matrix \(G=X^TX\) is positive semidefinite because

\[
x^TGx=\|Xx\|_2^2\geq0.
\]

It may be singular if the columns of \(X\) are linearly dependent.

### 5.3 Normal, orthogonal, and unitary matrices

A matrix is normal if \(A^*A=AA^*\). Orthogonal matrices satisfy \(Q^TQ=I\); unitary matrices satisfy \(Q^*Q=I\). Such transformations preserve the Euclidean norm. Orthogonal/unitary transformations are important in numerically stable algorithms because they do not magnify the 2-norm of a vector.

### 5.4 Nonnormal matrices

A matrix can have well-behaved eigenvalues yet exhibit sensitive transient amplification when it is nonnormal. Eigenvalues alone may not describe short-term behavior of \(A^kx\) or the sensitivity of eigenvalues to perturbations.

Study:

- Eigenvector conditioning.
- Schur decomposition \(A=QTQ^*\).
- Pseudospectra as a tool for understanding perturbation sensitivity.
- Why a spectral radius less than one is not the same as every finite-step norm being small.

This material deepens understanding of recurrent systems, iterative algorithms, and products of learned transformations.

### 5.5 Quadratic forms and ellipsoids

For symmetric positive definite \(A\), the level set

\[
x^TAx=c
\]

defines an ellipsoid. Eigenvectors define principal axes; eigenvalues determine scaling along those axes. Covariance matrices, Mahalanobis distance, Gaussian distributions, and second-order optimization all use this geometry.

**Advanced exercise:** Generate a covariance matrix with controlled eigenvalue decay, visualize its ellipsoid in two dimensions, and show how whitening transforms correlated coordinates into approximately isotropic coordinates.

---

# PART II — FACTORIZATIONS AND STABLE SOLVERS

## 6. Factorizations as Algorithmic Strategies

A factorization is not merely an alternative formula. It exposes structure so a problem can be solved with improved stability, cost, or interpretability.

### 6.1 LU with pivoting

For a suitable square matrix, partial pivoting produces

\[
PA=LU,
\]

where \(P\) is a permutation matrix, \(L\) is lower triangular, and \(U\) is upper triangular. Solve \(Ax=b\) through

\[
Ly=Pb,\qquad Ux=y.
\]

Avoid forming \(A^{-1}\) to solve a linear system. Factor once and solve for each right-hand side. Study pivot growth, stability, and the difference between algebraic nonsingularity and a numerically reliable solve.

**Implementation:** Gaussian elimination with partial pivoting, forward substitution, backward substitution, and residual/backward-error reporting.

### 6.2 QR factorization

For \(A\in\mathbb{R}^{m\times n}\), with \(m\geq n\), a thin QR factorization is

\[
A=QR,\qquad Q^TQ=I,
\]

where \(R\) is upper triangular. QR is a central tool for least squares and a numerically safer alternative to solving normal equations in many situations.

Compare:

- Classical Gram–Schmidt: simple but can lose orthogonality.
- Modified Gram–Schmidt: generally improves numerical behavior.
- Householder reflectors: standard robust choice for dense QR.
- Givens rotations: useful for structured, sparse, or incremental settings.

Do not treat these methods as equally stable. Measure \(\|I-Q^TQ\|\), residuals, runtime, and memory as conditioning worsens.

### 6.3 Cholesky factorization

For symmetric positive definite \(A\),

\[
A=LL^T.
\]

It is usually cheaper than general LU and exploits symmetry. It fails or requires modification if the matrix is not positive definite; failure is useful diagnostic evidence, not merely an implementation annoyance.

Applications include covariance systems, Gaussian models, quadratic optimization, and some second-order learning algorithms.

### 6.4 Eigendecomposition and Schur decomposition

For a diagonalizable matrix, one may write \(A=V\Lambda V^{-1}\). Do not assume \(V\) is orthogonal unless the matrix has the required structure, such as being real symmetric. For general matrices, the Schur form

\[
A=QTQ^*
\]

is a more broadly applicable numerically stable representation, with \(T\) triangular (or real quasi-triangular in the real Schur form).

### 6.5 Singular value decomposition

For \(A\in\mathbb{R}^{m\times n}\),

\[
A=U\Sigma V^T,
\]

where columns of \(U\) and \(V\) are orthonormal singular vectors and diagonal entries of \(\Sigma\) are nonnegative singular values ordered nonincreasingly.

SVD provides a coherent foundation for:

- Rank and numerical rank.
- The 2-norm and Frobenius norm.
- The Moore–Penrose pseudoinverse.
- Least squares and minimum-norm solutions.
- Low-rank approximation and PCA.
- Conditioning and numerical diagnostics.

### 6.6 Pseudoinverse and minimum-norm solutions

From SVD, define \(\Sigma^+\) by reciprocating the nonzero singular values under the selected exact or numerical rank convention:

\[
A^+=V\Sigma^+U^T.
\]

The vector \(x=A^+b\) gives a least-squares solution with minimum Euclidean norm among least-squares minimizers. In finite precision, the threshold used to treat small singular values as zero materially affects the answer. Document the threshold; do not call every small singular value mathematically zero.

### 6.7 Algorithm selection matrix

| Problem structure | Candidate method | Why it is appropriate | Critical caveat |
|---|---|---|---|
| General square solve | LU with pivoting | Reuses a factorization across right-hand sides | Pivot growth and conditioning still matter |
| Full-column-rank least squares | Householder QR | Avoids explicitly forming \(A^TA\) | Costs more than normal equations in some cases |
| Symmetric positive-definite solve | Cholesky | Exploits symmetry and definiteness | Invalid for indefinite matrices without modification |
| Rank-deficient or minimum-norm least squares | SVD / rank-revealing QR | Exposes numerical rank | More expensive for large dense problems |
| Leading singular components, large matrix | Randomized SVD / Lanczos-type method | Targets a subspace instead of full factorization | Approximation quality and spectral gap matter |
| Large sparse system | Preconditioned iterative solver | Can avoid dense storage and factorization | Convergence depends on spectrum and preconditioner |

### 6.8 Never invert by reflex

For numerical computation, prefer a solve or structured factorization over explicitly calculating \(A^{-1}\) when the task is to compute \(A^{-1}b\). The inverse may be dense even when \(A\) is sparse, incurs extra work, and can obscure the intended algorithm. The inverse remains a useful mathematical object and can be needed for particular derivations; the rule is about computational practice, not about banning the concept.

**Implementation milestone:** Build a solver dispatcher that selects an algorithm based on matrix properties, reports assumptions, computes diagnostics, and refuses to claim a stable answer when validation fails.

---

## 7. Least Squares, Projections, and Regularized Inverse Problems

### 7.1 Least squares as projection

For an overdetermined system \(Ax\approx b\), solve

\[
\hat{x}=\arg\min_x\|Ax-b\|_2^2.
\]

The residual \(r=b-A\hat{x}\) is orthogonal to the column space under the exact least-squares optimum:

\[
A^Tr=0.
\]

This leads to the normal equations

\[
A^TA\hat{x}=A^Tb,
\]

but explicitly forming \(A^TA\) squares the 2-norm condition number for full-column-rank \(A\):

\[
\kappa_2(A^TA)=\kappa_2(A)^2.
\]

This is one central reason QR or SVD is often preferred for numerically difficult least-squares problems.

### 7.2 Projection matrix

When \(A\) has full column rank,

\[
P_A=A(A^TA)^{-1}A^T
\]

is the orthogonal projection onto its column space. It satisfies \(P_A^T=P_A\), \(P_A^2=P_A\), and \(P_AA=A\). For rank-deficient settings, define the projection with a suitable pseudoinverse or an orthonormal basis instead of blindly applying the full-rank expression.

### 7.3 Ridge regularization

Ridge regression solves

\[
\hat{x}_\lambda=\arg\min_x\left(\|Ax-b\|_2^2+\lambda\|x\|_2^2\right),\qquad\lambda>0.
\]

Its normal-equation form is

\[
(A^TA+\lambda I)\hat{x}_\lambda=A^Tb.
\]

Regularization modifies the spectrum and can reduce sensitivity, but the value of \(\lambda\) changes bias and variance and should be selected for the objective—not used to hide a broken data pipeline.

### 7.4 Generalized Tikhonov regularization

A more general inverse problem is

\[
\hat{x}=\arg\min_x\left(\|Ax-b\|_2^2+\lambda\|Lx\|_2^2\right),
\]

where \(L\) encodes which components should be penalized. The choice \(L=I\) is ridge regularization; other choices can encode smoothness or structural priors.

### 7.5 Constrained least squares

Study problems with linear constraints such as \(Cx=d\) and non-negativity constraints. For equality constraints, derive the Karush–Kuhn–Tucker (KKT) system and identify when its block matrix is nonsingular. Distinguish constraints that define feasible solutions from penalties that merely encourage behavior.

### 7.6 Iterative methods for large systems

For scale, learn the mechanism and limitations of:

- Conjugate gradient for symmetric positive-definite systems.
- MINRES for suitable symmetric indefinite systems.
- GMRES for general nonsymmetric systems.
- LSQR/LSMR for least-squares systems.
- Preconditioning and its effect on the spectrum of the transformed problem.

For a Krylov method, the candidate solution is built from a subspace generated by repeated applications of a matrix or operator. Understand the recurrence, stopping criterion, residual behavior, and why preconditioning can be more important than a small constant-factor optimization.

**Experiments:** Compare normal equations, QR, SVD, and LSQR across matrix size, conditioning, rank deficiency, and sparsity. Report residual, forward error when ground truth is known, iterations, runtime, and peak memory.

---

# PART III — EIGENVALUES, SVD, LOW RANK, AND RANDOMIZATION

## 8. Eigenvalues, Singular Values, and Spectral Diagnostics

### 8.1 Eigenvalues versus singular values

Eigenvalues describe directions that are scaled without changing direction under a square linear map, subject to the existence of eigenvectors over the chosen field. Singular values describe maximal gains between orthogonal input and output directions and are available for every rectangular matrix.

For a general matrix, eigenvalues need not equal singular values. For a symmetric positive semidefinite matrix, singular values equal its nonnegative eigenvalues. Keep this distinction explicit in PCA, conditioning, and model analysis.

### 8.2 Rayleigh quotient

For symmetric \(A\),

\[
R(x)=\frac{x^TAx}{x^Tx},\qquad x\ne0.
\]

The quotient lies between the smallest and largest eigenvalues; extremal values are attained by corresponding eigenvectors. This connects eigenvalue estimation to constrained optimization on the unit sphere.

### 8.3 Power iteration

Power iteration repeatedly applies \(A\) and normalizes:

\[
x_{k+1}=\frac{Ax_k}{\|Ax_k\|_2}.
\]

It can converge to a dominant eigenvector under appropriate assumptions, including a suitable spectral gap and nonzero projection of the initial vector onto the dominant eigenspace. Analyze convergence rate, sign/phase ambiguity, and failure under repeated dominant eigenvalues or unfavorable initializations.

### 8.4 Lanczos and Arnoldi methods

For large matrices, Krylov-subspace methods estimate selected eigenvalues without computing a full eigendecomposition. Lanczos exploits symmetry/Hermitian structure; Arnoldi is more general. Learn:

- Krylov subspace \(\mathcal{K}_k(A,v)=\operatorname{span}\{v,Av,\ldots,A^{k-1}v\}\).
- Orthogonalization and loss of orthogonality.
- Ritz values and residual-based stopping.
- Restarting and memory growth.
- Why matrix-vector products can be preferable to materializing the entire operator.

### 8.5 Singular spectrum and effective rank

The decay pattern \(\sigma_1\geq\sigma_2\geq\dots\) reveals how well a matrix may be approximated by a low-rank representation. Exact rank, tolerance-based numerical rank, stable rank, and entropy-based effective rank are different summaries; do not substitute them without declaring the definition.

One useful stable-rank definition is

\[
\operatorname{srank}(A)=\frac{\|A\|_F^2}{\|A\|_2^2},
\]

which is less brittle than exact rank under small perturbations but is not identical to the number of singular values above a threshold.

---

## 9. Low-Rank Approximation and PCA

### 9.1 Eckart–Young principle

Given the SVD \(A=U\Sigma V^T\), retain the top \(k\) singular components:

\[
A_k=U_k\Sigma_kV_k^T.
\]

This is a best rank-\(k\) approximation to \(A\) under both the Frobenius norm and spectral norm. The errors are

\[
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2,
\qquad
\|A-A_k\|_2=\sigma_{k+1}.
\]

These results turn singular-value decay into quantitative compression analysis.

### 9.2 PCA as an optimization problem

For centered data \(X\in\mathbb{R}^{n\times d}\), each row representing an observation, PCA seeks orthogonal directions that capture variance. Define the sample covariance under the chosen denominator convention, commonly

\[
C=\frac{1}{n-1}X^TX.
\]

The first principal direction solves

\[
\max_{\|v\|_2=1}v^TCv,
\]

and is a top eigenvector of \(C\). Subsequent components are constrained to be orthogonal to earlier components.

Equivalent computations can use the SVD of the centered data matrix. Do not automatically form \(X^TX\) for ill-conditioned or high-dimensional problems: it changes numerical conditioning and can be unnecessarily large.

### 9.3 Whitening

Whitening rescales principal coordinates by their variance. With a stabilized eigenvalue/singular-value spectrum, a whitening transform is conceptually of the form

\[
Z=XQ\Lambda^{-1/2},
\]

where small eigenvalues may need regularization or truncation. Exact whitening can magnify low-variance noise; discuss the role of a floor, shrinkage, or rank truncation.

### 9.4 Randomized SVD

For large matrices, randomized methods estimate the dominant column/row subspaces with random sketches. A typical high-level scheme is:

1. Draw a random test matrix \(\Omega\).
2. Form a sample of the range, \(Y=A\Omega\).
3. Orthonormalize to get a basis \(Q\).
4. Optionally apply power iterations with careful re-orthogonalization.
5. Factor the smaller matrix \(Q^TA\).
6. Recover the approximate singular vectors and values.

Study oversampling, power iteration count, spectral gap, numerical stability, and pass count over the original data. A randomized method is an approximation with measurable error, not automatically a free replacement for exact SVD.

### 9.5 Sketching and Johnson–Lindenstrauss intuition

Random projections can preserve pairwise distances approximately when embedding dimension is sufficiently large relative to the logarithm of the number of points and the target distortion. Learn the assumptions and why guarantees are probabilistic. Connect sketching to approximate nearest neighbors, least squares, and reducing intermediate compute.

**Project — spectral compression report:** On several matrices, compare exact SVD, randomized SVD, and a simple low-rank factorization. Report singular-spectrum capture, reconstruction error, latency, peak memory, and the rank at which compression becomes worthwhile.

---

## 10. Matrix Perturbation, Numerical Rank, and Backward Stability

### 10.1 Exact mathematics versus floating-point computation

Real-number algebra assumes infinite precision. Computer arithmetic rounds results; elementary operations are therefore approximations. For normal floating-point operations, a standard model is often written

\[
\operatorname{fl}(a\circ b)=(a\circ b)(1+\delta),\qquad |\delta|\lesssim u,
\]

when the operation is in a normal range and the model applies. Underflow, overflow, subnormals, cancellation, and hardware-specific behavior need separate attention.

### 10.2 Forward error, backward error, and conditioning

- **Forward error:** distance between computed and exact answer.
- **Backward error:** size of the perturbation to the input for which the computed answer is exact.
- **Conditioning:** sensitivity of the problem itself.
- **Stability:** behavior of the algorithm under finite-precision computation.

A backward-stable algorithm applied to a well-conditioned problem usually produces a trustworthy answer. A stable algorithm applied to a severely ill-conditioned problem can still have large forward error. Conversely, a small residual can coexist with a large solution error.

### 10.3 Cancellation and accumulation order

Floating-point addition is not associative:

\[
(a+b)+c\ne a+(b+c)
\]

in finite precision in general. Summing values with different magnitudes can erase small terms. Dot products and reductions in parallel kernels may differ slightly because operation ordering changes.

Study pairwise summation, compensated summation, accumulation precision, and the numerical implications of fused multiply-add operations.

### 10.4 Rank thresholds

Numerical rank is a decision based on a threshold, often linked to matrix scale and machine precision, for example a threshold proportional to

\[
\max(m,n)\,\epsilon_{\text{mach}}\,\sigma_{\max}(A),
\]

though practical defaults and application-specific noise levels may differ. Report the threshold and test sensitivity to it. Avoid interpreting a numerical-rank estimate as an exact physical truth.

### 10.5 Perturbation of singular values and subspaces

Study how singular values and singular vectors change under \(A+E\), including the roles of \(\|E\|\), singular-value gaps, and near-degenerate subspaces. A stable subspace may exist even when individual vectors within a repeated or tightly clustered eigenspace are not uniquely identifiable.

### 10.6 Diagnostics checklist

For a solver or factorization, report as appropriate:

- Relative residual: \(\|Ax-b\|/\|b\|\), with zero-denominator handling.
- Backward-error estimate.
- Condition number or an estimate of it.
- Orthogonality error: \(\|Q^TQ-I\|\).
- Factorization error: \(\|A-QR\|/\|A\|\), or the corresponding factorization residual.
- Reconstruction error and retained spectral energy.
- Rank threshold, dtype, and numerical-library configuration.

**Experiment:** Create two matrices that differ only by a tiny perturbation. Compare exact rank in construction, numerical rank, pseudoinverse solution, and downstream least-squares predictions as the perturbation shrinks.

---

# PART IV — MATRIX CALCULUS AND LEARNING DYNAMICS

## 11. Matrix Differential Calculus

Matrix calculus is the bridge between linear algebra and training neural networks. Use explicit shape annotations throughout. Many gradient mistakes are dimension errors concealed by notation.

### 11.1 Differential-first viewpoint

For a scalar function \(f(X)\), define the differential by

\[
df=\langle \nabla_X f,dX\rangle_F
=\operatorname{tr}((\nabla_X f)^T dX)
\]

for the real-valued convention used here. The differential is a linear map in the perturbation \(dX\); matching coefficients yields the gradient.

This approach is often safer than trying to memorize matrix derivative rules.

### 11.2 Essential identities

For compatible real matrices, learn to derive these identities rather than only memorize them:

\[
d\,\operatorname{tr}(X)=\operatorname{tr}(dX)
\]

\[
d\,\operatorname{tr}(AX)=\operatorname{tr}(A\,dX)
\]

\[
\nabla_X\,\operatorname{tr}(A^TX)=A
\]

\[
\nabla_x\,(a^Tx)=a
\]

\[
\nabla_x\,(x^TAx)=(A+A^T)x
\]

and, if \(A\) is symmetric,

\[
\nabla_x\,(x^TAx)=2Ax.
\]

For a scalar loss \(L=\frac12\|Ax-b\|_2^2\),

\[
\nabla_x L=A^T(Ax-b).
\]

Each identity requires compatible dimensions; symmetric-matrix simplifications require symmetry.

### 11.3 Matrix-valued mappings and Jacobians

For \(Y=WX\), the full Jacobian can be enormous. Deep-learning systems generally avoid materializing it and instead compute Jacobian-vector products (JVPs) or vector-Jacobian products (VJPs).

Understand:

- Forward-mode differentiation computes directional derivatives/JVPs.
- Reverse-mode differentiation computes VJPs and is efficient for scalar losses with many parameters.
- Reverse mode propagates gradients through adjoint linear maps.
- Automatic differentiation is different from symbolic differentiation and finite differences.

### 11.4 Gradient of a linear layer

Let

\[
Y=XW^T+\mathbf{1}b^T,
\]

where \(X\in\mathbb{R}^{B\times d_{in}}\), \(W\in\mathbb{R}^{d_{out}\times d_{in}}\), \(b\in\mathbb{R}^{d_{out}}\), and \(Y\in\mathbb{R}^{B\times d_{out}}\). Given upstream gradient \(G=\nabla_Y L\), derive and validate

\[
\nabla_X L=GW,
\qquad
\nabla_W L=G^TX,
\qquad
\nabla_b L=\sum_{i=1}^{B}G_{i,:}.
\]

If the forward convention or tensor layout changes, the formulas and shapes must be adjusted accordingly.

### 11.5 Gradients of common matrix objectives

Derive from differentials:

- Frobenius reconstruction loss \(\frac12\|X-UV^T\|_F^2\).
- Quadratic form \(x^TAx\).
- Ridge objective \(\|Ax-b\|^2+\lambda\|x\|^2\).
- Trace objectives involving \(X^TAX\).
- Softmax-cross-entropy with logits produced by a matrix projection.

When deriving each gradient, explicitly label which variables are trainable and which are constants.

### 11.6 Gradient verification

For scalar \(f\), compare analytical gradients with central finite differences:

\[
\frac{\partial f}{\partial x_i}\approx
\frac{f(x+h e_i)-f(x-h e_i)}{2h}.
\]

Use small problems and float64 for gradient checks. Select several step sizes: a step that is too large introduces truncation error; one that is too small amplifies roundoff. Finite differences are a debugging tool, not the default training algorithm.

**From-scratch milestone:** Implement a two-layer MLP forward pass, loss, backward pass, gradient checker, and one optimizer update without autograd. Then compare gradients and parameter updates against PyTorch on a tiny deterministic batch.

---

## 12. Jacobian Products, Conditioning, and Deep Signal Propagation

### 12.1 Composition of Jacobians

For a deep composition

\[
h_L=f_L(f_{L-1}(\cdots f_1(x))),
\]

the Jacobian is a product of local Jacobians. In a direction \(v\), forward perturbations propagate as repeated JVPs; gradients propagate through transposed/adjoint Jacobians.

The products can shrink or amplify different directions. The relevant property is not only the average gradient norm but also the singular-value structure of the Jacobian product.

### 12.2 Vanishing and exploding gradients

A simplified linear-depth model makes the mechanism visible:

\[
h_L=W_LW_{L-1}\cdots W_1x.
\]

The singular values of the product control the amplification of specific directions. Repeated contractions can erase signal; repeated expansion can cause instability. Nonlinear activation derivatives alter the effective Jacobians, and the weight matrices need not share singular vectors, so multiplying each layer's largest singular value is only a bound.

### 12.3 Dynamical isometry intuition

Study the distribution of singular values of the input-output Jacobian and why keeping them near one can improve signal propagation in some architectures and initialization regimes. Understand that this is a design objective/analysis tool, not a universal guarantee that a model will train well.

### 12.4 Residual pathways and normalization

A residual block often resembles

\[
h_{\ell+1}=h_\ell+F_\ell(h_\ell).
\]

Its Jacobian is \(I+J_{F_\ell}\). The identity pathway changes gradient propagation relative to a plain composition. Layer normalization and related mechanisms introduce their own Jacobian structure; their effects depend on the implementation and architecture.

**Experiment:** Compare products of random matrices under multiple variance scalings. Track singular-value distributions, norm growth, condition estimates, and gradient norms as depth increases. Add residual connections and compare instead of merely stating that residuals “solve” vanishing gradients.

---

# PART V — DIRECT BRIDGE TO TRANSFORMERS AND LLMS

## 13. Linear Algebra Inside a Transformer Block

This module does not attempt to teach the entire Transformer architecture. It identifies the linear-algebra operations whose behavior must be understood before building one from scratch.

### 13.1 Embedding tables

An embedding table is a parameter matrix

\[
E\in\mathbb{R}^{V\times d},
\]

where \(V\) is vocabulary size and \(d\) is embedding dimension. Token IDs perform row selection; this is not the same computation pattern as multiplying a one-hot vector by \(E\), even though the results can be mathematically equivalent.

Understand:

- Parameter and activation shapes.
- Lookup gradients and how only selected rows receive direct lookup gradients in a batch.
- Tied embeddings and output projection weights.
- Memory cost \(Vd\) and the trade-off between width and vocabulary size.
- Why embedding geometry depends on training and does not automatically make Euclidean distance semantically meaningful.

### 13.2 Projection matrices and learned subspaces

The attention projections are commonly represented as

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
\]

with exact transposes determined by tensor-layout convention. Each projection maps representations into a learned feature space. Analyze the rank and singular spectrum of these matrices, how output dimensions constrain representable directions, and how parameter sharing changes memory and computation.

### 13.3 Scaled dot-product attention

For one head, in a common row-major convention,

\[
S=\frac{QK^T}{\sqrt{d_k}},\qquad
P=\operatorname{softmax}(S+M),\qquad
O=PV,
\]

where \(M\) is a mask that excludes disallowed positions. Different codebases transpose tensor axes differently; annotate dimensions instead of copying formulas blindly.

For sequence length \(T\), batch size \(B\), number of heads \(H\), and head dimension \(d_k\), the score tensor commonly has shape \(B\times H\times T\times T\). The quadratic \(T^2\) term is a central memory/compute cost of dense attention.

Analyze:

- Dot-product scale and the role of \(1/\sqrt{d_k}\).
- Softmax sensitivity to logit magnitude.
- Attention-score rank bounds induced by \(QK^T\).
- Effects of masks on legal interactions.
- Numerical stability of softmax using max subtraction.
- Compute and storage cost of materializing the score/probability matrices.

### 13.4 Why scale attention scores?

Under a simplified model in which query/key coordinates are independent, zero-mean, unit-variance random variables, the dot product's variance typically grows with \(d_k\). Dividing by \(\sqrt{d_k}\) keeps the score scale from growing linearly in variance under those assumptions. The assumptions are idealized; training distributions and correlations can differ.

### 13.5 Softmax as a nonlinear transformation of logits

For scores \(s\),

\[
\operatorname{softmax}(s)_i=\frac{e^{s_i}}{\sum_j e^{s_j}}.
\]

Numerically stable evaluation uses

\[
\operatorname{softmax}(s)_i=
\frac{e^{s_i-m}}{\sum_j e^{s_j-m}},
\qquad m=\max_j s_j.
\]

This shift leaves the mathematical result unchanged but helps avoid overflow. Study softmax's Jacobian:

\[
J=\operatorname{diag}(p)-pp^T.
\]

The Jacobian reveals coupling between probabilities and helps explain saturation and gradient behavior.

### 13.6 Attention backward pass

Implement the forward and backward passes for a tiny attention head. Trace gradients through \(O=PV\), softmax, and \(S=QK^T/\sqrt{d_k}\). Verify gradients using finite differences or autograd. The main target is not merely to produce the right tensor shape, but to derive how each upstream gradient contributes to \(Q\), \(K\), and \(V\).

### 13.7 Flash-style attention and IO awareness

The naive algorithm materializes \(S\) and \(P\), producing quadratic intermediate storage in sequence length. Tiled/IO-aware algorithms reorganize the computation to reduce high-bandwidth-memory traffic and avoid storing the full attention matrix, while preserving an equivalent or numerically close result under the algorithm's computation model.

At the linear-algebra level, understand the stable online softmax recurrence and blockwise accumulation. At the systems level, distinguish FLOPs from memory traffic and account for the hardware's compute-to-memory balance.

### 13.8 MLP/feed-forward block

A two-projection feed-forward block can be expressed as

\[
H=\phi(XW_1+b_1),\qquad Y=HW_2+b_2.
\]

Track activation shapes, parameter count, gradient paths, and matrix-multiplication costs. Explain how the intermediate width changes the block's representational capacity and compute requirements.

### 13.9 Logits and output projection

Given the final hidden state \(h\) and output matrix \(W_{out}\), vocabulary logits are a linear projection such as

\[
z=hW_{out}^T.
\]

The projection's output dimension is vocabulary size. Analyze its parameter count, memory bandwidth, loss gradient, tied-weight option, and why output projection can be a significant compute cost.

**Transformer lab:** For a tiny configuration, record every tensor shape from token IDs to logits. Implement the core linear algebra in a clear reference version, check forward equivalence with a framework implementation, and compare backward gradients. Include a causal mask test that proves future tokens cannot influence earlier-position outputs in the tested block.

---

## 14. Low-Rank Adaptation, Matrix Compression, and Parameter Efficiency

### 14.1 Low-rank parameter updates

A low-rank update expresses a weight modification as

\[
\Delta W=BA,
\]

where \(A\) and \(B\) have a bottleneck dimension \(r\) much smaller than the original matrix dimensions. The exact scaling and initialization depend on the adaptation method. The key linear-algebra idea is that the update is constrained to a low-dimensional factorized form.

Study:

- Parameter-count reduction.
- Rank as an adaptation-capacity control.
- Initialization and gradient flow through both factors.
- Comparison with a dense update under equal training budgets.
- Merge-time computation and numerical differences.
- Why low-rank structure may work well empirically without proving that all useful updates are low rank.

### 14.2 Matrix compression is not always end-to-end acceleration

A lower parameter count can reduce storage but may introduce extra matrix multiplications, poor kernel utilization, or data movement. Evaluate:

- Parameter memory.
- Activation/intermediate memory.
- FLOPs and arithmetic intensity.
- Wall-clock latency.
- Quality degradation under a defined test set.
- Compatibility with optimized kernels and batching.

### 14.3 Quantization as approximate linear algebra

Quantization maps continuous values to a finite representation. A simplified affine quantizer is

\[
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right)+z\right),
\]

with scale \(s\), zero point \(z\), and an integer range set by the chosen format. Dequantization approximately reconstructs \(x\) from \(q\).

Quantized matrix multiplication introduces structured rounding error into dot products. Analyze activation and weight scales, outliers, accumulation precision, per-tensor versus per-channel scaling, calibration data, and the difference between quantizing weights and quantizing activations.

Quantization is not just “use fewer bits.” Its success depends on the distribution of values, the quantization scheme, kernel implementation, and the sensitivity of downstream computations.

### 14.4 Numerical sensitivity of a model block

For a linear layer, a perturbation in weights or activations yields first-order terms

\[
\delta Y\approx \delta X W^T+X\delta W^T,
\]

ignoring higher-order products. This gives a starting point for evaluating how quantization error propagates through a network. Across nonlinear blocks, sensitivity depends on the local Jacobian and data distribution.

**Experiment:** Compare a dense update, a rank-limited factorization, and a quantized matrix multiply on the same controlled task. Report reconstruction/function error, memory, latency, and variance across input scales. Do not infer real-world LLM quality from a toy matrix error alone.

---

# PART VI — SPARSE, STRUCTURED, AND LARGE-SCALE LINEAR ALGEBRA

## 15. Sparse Operators and Structured Matrices

### 15.1 Sparse representations

For a matrix with many zeros, dense storage wastes memory and arithmetic. Know the data layout trade-offs:

- COO: convenient for constructing from coordinate triplets and combining duplicate entries.
- CSR: efficient row operations and matrix-vector products in common cases.
- CSC: efficient column access and some factorization workflows.
- Block-sparse layouts: useful when nonzeros occur in dense blocks.

Sparse structures have indexing overhead, and sparse kernels can be memory-bound. A matrix being sparse mathematically does not guarantee that a sparse implementation is faster for its size and pattern.

### 15.2 Graph Laplacians

For an undirected graph with adjacency matrix \(A\) and degree matrix \(D\), the combinatorial Laplacian is

\[
L=D-A.
\]

It is positive semidefinite, and its quadratic form encodes smoothness across edges:

\[
x^TLx=\frac12\sum_{i,j}A_{ij}(x_i-x_j)^2
\]

under the standard symmetric adjacency convention. Laplacian eigenvectors relate to graph structure, spectral clustering, and graph-based learning.

### 15.3 Kronecker products and structured operators

The Kronecker product can represent structured multidimensional linear maps and separable computations. Learn its shape and product identities, and analyze when a structured representation avoids materializing a much larger dense matrix.

Useful connections include multidimensional signals, tensor operations, separable transformations, and some structured parameterizations.

### 15.4 Toeplitz, circulant, and convolution operators

Convolution can be written as a structured linear operation. The fully materialized matrix representation is often wasteful; implementations exploit local connectivity, weight sharing, and sometimes Fourier structure. Understand the equivalence without assuming that explicitly constructing the matrix is computationally sensible.

### 15.5 Matrix-free computation

Many algorithms do not need all entries of a matrix; they need only products such as \(Av\) or \(A^Tv\). Define operators through matrix-vector products to support iterative methods, large scientific problems, and model components with implicit structure.

**Implementation milestone:** Build a common linear-operator interface exposing shape, dtype, forward multiplication, transpose multiplication, and optional diagnostics. Test it against explicit matrices on small cases.

---

## 16. High-Dimensional Geometry, Kernels, and Similarity

### 16.1 Distance concentration

In high dimensions, distances between random points can become relatively concentrated under common distributions. This can weaken the contrast between “nearest” and “farthest” points and affect nearest-neighbor methods. The result depends on the distribution, scaling, metric, and data manifold; it is not a universal claim that all distances become identical.

Measure concentration empirically instead of relying on a slogan. Plot the distribution of pairwise distances, their coefficient of variation, and nearest-neighbor contrast as dimension increases.

### 16.2 Cosine similarity and normalization

For nonzero vectors,

\[
\operatorname{cos}(x,y)=\frac{x^Ty}{\|x\|_2\|y\|_2}.
\]

Cosine similarity removes magnitude information. It is useful when direction is the intended signal but can be inappropriate when magnitude carries meaning. Normalization changes geometry; it does not make embeddings intrinsically meaningful.

### 16.3 Mahalanobis distance

For mean \(\mu\) and positive-definite covariance \(\Sigma\),

\[
d_M(x,\mu)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}.
\]

This scales deviations by covariance structure. In practice, covariance estimation and inverse stability are major issues, especially when dimensionality is high relative to sample count. Prefer solves or structured/shrinkage estimators over blindly forming an inverse.

### 16.4 Gram and kernel matrices

A Gram matrix stores pairwise inner products. A valid positive-semidefinite kernel yields a positive-semidefinite Gram matrix under its conditions. Kernel methods can operate in an implicit feature space, but matrix storage may scale quadratically in sample count. Analyze positive semidefiniteness, rank, conditioning, and approximation strategies.

### 16.5 Geometry of a learned embedding space

Investigate whether similarity metrics correlate with a task-specific target. Measure neighborhoods, anisotropy, dominant directions, hubness, and sensitivity to centering/normalization. Avoid claims such as “cosine is always better” without evidence from the target representation and evaluation task.

---

# PART VII — HARDWARE-AWARE LINEAR ALGEBRA

## 17. Complexity, Memory Traffic, and Kernel Behavior

### 17.1 FLOPs are not the whole cost

Dense matrix multiplication of \(m\times k\) and \(k\times n\) matrices requires \(O(mkn)\) arithmetic and stores/reads data whose cost depends on layout and reuse. On modern accelerators, the bottleneck may be compute, memory bandwidth, launch overhead, or synchronization. Estimate arithmetic intensity rather than comparing FLOP counts alone.

### 17.2 Data layout and batching

Understand:

- Contiguous versus strided memory.
- Row-major and column-major conventions.
- Transposed views versus physical copies.
- Leading dimensions and batch dimensions.
- Why a mathematically identical operation may invoke different kernels after a layout change.
- The difference between broadcasting as a logical view and an actual materialized expansion.

### 17.3 BLAS and LAPACK as algorithmic boundaries

Use mature numerical libraries in production-facing experiments, but retain a transparent from-scratch implementation for learning and verification. Learn what level of operation each layer supplies: BLAS for core dense linear algebra kernels; LAPACK for factorization, eigenvalue, and related numerical routines; higher-level frameworks for tensor/autograd composition.

Do not attribute all behavior to the Python code alone. Record library versions, dtype, thread settings, accelerator, and relevant environment details.

### 17.4 Mixed precision

Common formats differ in exponent range, significand precision, and hardware support. Understand the distinction between storage dtype and accumulation dtype. Mixed-precision matrix multiplication may use lower-precision operands and a wider accumulation type. Loss scaling, stable normalization, and careful reductions can be important for training.

Do not assume float16 and bfloat16 have interchangeable numerical behavior: they trade significand precision and exponent range differently.

### 17.5 Benchmarking correctly

- Warm up kernels and separate compilation/startup from steady-state timing.
- Synchronize accelerators when timing asynchronous execution.
- Run multiple trials and report median plus dispersion.
- Control thread count and avoid benchmarking under changing system load when possible.
- Verify the answer before comparing speed.
- Report shapes, dtype, batch size, hardware, software versions, and whether memory allocation is included.

**Experiment — algorithmic crossover:** Compare dense, sparse, and low-rank methods across a grid of dimensions and sparsity/rank levels. Identify the empirical crossover points on your own hardware rather than claiming one method is always faster.

---

# PART VIII — IMPLEMENTATION LABS FROM SCRATCH

## 18. Required Algorithms and Acceptance Criteria

Implement a learning version and validate it against an independent reference. Scratch implementations should prioritize clarity and correctness over speed.

| ID | Implementation | Required validation |
|---|---|---|
| LA-01 | Matrix multiplication | Shape, reference agreement, scaling with dimensions |
| LA-02 | Stable dot product and norms | Edge cases, overflow/cancellation discussion |
| LA-03 | Gaussian elimination with partial pivoting | Solve residual and singular/near-singular tests |
| LA-04 | Forward/back substitution | Triangular solve checks and multiple RHS support |
| LA-05 | LU factorization with pivoting | \(PA\approx LU\) and pivot correctness |
| LA-06 | Modified Gram–Schmidt | Orthogonality and reconstruction errors |
| LA-07 | Householder QR | \(A\approx QR\), orthogonality, rank-deficient cases |
| LA-08 | Cholesky | \(A\approx LL^T\), positive-definite and failure cases |
| LA-09 | Power iteration | Dominant eigenpair residual and convergence experiment |
| LA-10 | Basic Lanczos mechanism | Krylov basis checks, Ritz residuals, limitations documented |
| LA-11 | SVD-based pseudoinverse | Moore–Penrose identities within tolerances |
| LA-12 | Least-squares solver | Compare QR/SVD and report residual plus forward error |
| LA-13 | PCA via SVD | Centering, explained variance, reconstruction error |
| LA-14 | Randomized low-rank approximation | Error/runtime/memory versus exact baseline |
| LA-15 | JVP/VJP for a linear layer | Finite-difference/autograd agreement |
| LA-16 | Two-layer MLP backward pass | Gradient checks and loss decrease on a tiny task |
| LA-17 | Stable softmax and cross-entropy | Large-logit and extreme-scale cases |
| LA-18 | Attention forward/backward | Shape checks, mask test, gradient verification |
| LA-19 | Sparse matrix-vector multiplication | Dense-reference agreement and memory accounting |
| LA-20 | Quantized matrix multiplication prototype | Error/latency/memory trade-off report |

### General acceptance criteria

Each implementation must include:

1. A mathematical description and assumptions.
2. Function signatures, input shapes, output shapes, dtype behavior, and error handling.
3. Tests for ordinary, degenerate, and adversarial inputs.
4. An independent numerical reference.
5. A residual/error measure with a justified tolerance.
6. Complexity analysis for time and storage.
7. A reproducible experiment and a short interpretation of results.
8. A written statement of what the implementation does **not** guarantee.

Avoid writing fake “from scratch” implementations that simply call `np.linalg.svd`, `np.linalg.qr`, `torch.linalg.solve`, or equivalent routines inside the algorithm being studied. Using libraries for validation is encouraged.

---

## 19. Experiment Design: Claims Must Be Falsifiable

Every experiment should start with a question and a prediction. For each run, record seed, shapes, rank, condition level, dtype, algorithm, elapsed time, memory, error metric, and software/hardware context.

### Experiment A — Normal equations versus QR/SVD

**Question:** How does conditioning affect solution quality across algorithms?  
**Control:** Known ground-truth solution, identical input matrix, same target dtype where feasible.  
**Measurements:** Forward error, residual, condition number, runtime, and memory.  
**Expected analysis:** Normal equations can degrade faster because the Gram matrix squares the condition number; exact severity depends on the matrix and precision.

### Experiment B — Gram–Schmidt variants

**Question:** How does orthogonality degrade as columns become nearly dependent?  
**Measurements:** \(\|I-Q^TQ\|\), reconstruction error, runtime.  
**Compare:** Classical Gram–Schmidt, modified Gram–Schmidt, Householder QR.  
**Conclusion requirement:** Explain which method behaved better for tested conditions and what that experiment cannot prove universally.

### Experiment C — SVD rank threshold

**Question:** How sensitive are downstream solutions to the numerical-rank threshold?  
**Control:** Singular-value spectrum with a gradual tail.  
**Measurements:** Effective rank under each threshold, pseudoinverse norm, solution norm, residual.  
**Conclusion requirement:** Distinguish model-selection/regularization choices from floating-point limitations.

### Experiment D — PCA by covariance eigendecomposition versus data SVD

**Question:** When are the computed principal subspaces equivalent, and when does the implementation route matter?  
**Measurements:** Principal angles between subspaces, explained variance, reconstruction error, runtime, memory.  
**Important:** Individual principal vectors may differ by sign; within repeated/near-repeated eigenvalues, different bases may describe the same subspace.

### Experiment E — Randomized SVD

**Question:** How do oversampling, power iterations, and spectrum decay affect approximation quality?  
**Measurements:** Spectral/Frobenius reconstruction error, wall time, memory, passes over data.  
**Conclusion requirement:** Compare at equal compute or wall-time budgets as well as at equal rank.

### Experiment F — Gradient correctness

**Question:** Does the manual backward pass agree with the analytical/numerical reference?  
**Measurements:** Relative gradient error by parameter tensor and selected coordinate.  
**Control:** Tiny float64 tensors, deterministic seed, non-saturated inputs where possible.  
**Conclusion requirement:** Diagnose shape/transposition issues separately from finite-difference step-size issues.

### Experiment G — Attention numerical stability

**Question:** How do naive softmax and max-shifted softmax behave as score magnitude grows?  
**Measurements:** NaN/Inf count, probability sum error, output difference.  
**Extension:** Compare materialized and blockwise attention outputs within a documented tolerance.

### Experiment H — Mixed-precision sensitivity

**Question:** Which operations are sensitive to reduced precision in the tested workload?  
**Measurements:** Relative output error, gradient error, memory, throughput.  
**Controls:** Same data and shapes, explicit accumulation type, repeatable environment.  
**Conclusion requirement:** Separate numerical error from performance changes due to hardware kernel selection.

### Experiment I — Sparse/dense crossover

**Question:** At which shapes and sparsity patterns does sparse computation win on the selected hardware?  
**Measurements:** Storage bytes, runtime, throughput, preprocessing cost.  
**Conclusion requirement:** Explain indexing overhead and pattern dependence.

### Experiment J — Low-rank approximations for learned matrices

**Question:** Does retained rank predict approximation error or downstream behavior for the selected matrix?  
**Measurements:** Singular spectrum, reconstruction error, activation-output error, latency and storage.  
**Caveat:** Matrix reconstruction quality alone does not establish language-model quality.

---

# PART IX — CASE STUDIES

## 20. Case Study 1 — Linear Regression as Geometry and Numerical Analysis

Given \(X\in\mathbb{R}^{n\times d}\) and target \(y\in\mathbb{R}^{n}\), solve

\[
\hat\beta=\arg\min_\beta\|X\beta-y\|_2^2.
\]

The complete analysis should cover:

- Column space and representable predictions.
- Orthogonal projection and residual orthogonality.
- Rank deficiency and non-identifiability of coefficients.
- Multicollinearity and singular-value decay.
- Normal equations versus QR/SVD.
- Ridge regularization and the bias/variance trade-off.
- Residual diagnostics and data scaling.

**Deliverable:** A report showing how duplicated or nearly duplicated features affect rank, condition number, coefficient variance, fitted values, and solver behavior.

## 21. Case Study 2 — PCA and Representation Compression

Given a centered data matrix \(X\), analyze its SVD and principal subspace. Show how retained rank affects reconstruction error and explained variance. Compare covariance-eigendecomposition to direct SVD and explain the effect of the data shape \(n\ll d\) versus \(n\gg d\).

**Deliverable:** A reproducible PCA implementation with explicit centering, rank selection, reconstruction metrics, principal-subspace comparison, and discussion of scaling/whitening.

## 22. Case Study 3 — KNN and High-Dimensional Metrics

Represent examples as vectors and compare Euclidean, Manhattan, cosine, and appropriately estimated Mahalanobis distances. Investigate normalization, irrelevant features, feature-scale changes, and distance concentration.

**Deliverable:** A metric comparison that explains which geometric property each metric emphasizes, when it is sensitive to scale, and why the best choice depends on the data/task.

## 23. Case Study 4 — SVM and Hyperplane Geometry

A linear decision boundary has form

\[
w^Tx+b=0.
\]

The normal vector \(w\) is orthogonal to the hyperplane, and the distance from a point to the hyperplane is proportional to \(|w^Tx+b|/\|w\|_2\). Connect margins to vector norms and optimization constraints. Then describe how kernel matrices replace explicit feature coordinates in kernelized methods.

**Deliverable:** Derive the geometric margin for a simple linear classifier and test how feature scaling changes the solution when regularization is present.

## 24. Case Study 5 — Mahalanobis Anomaly Detection

Use

\[
d_M(x,\mu)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}
\]

only after evaluating whether the covariance estimate is reliable. Analyze sample-size-to-dimension ratio, singular covariance, shrinkage, scaling, and solve-based computation. Compare Euclidean and Mahalanobis rankings on a controlled correlated dataset.

**Deliverable:** An experiment demonstrating when covariance-aware distance helps and when unstable covariance estimates make it unreliable.

## 25. Case Study 6 — Backpropagation Through Matrix Operations

Build a small MLP using only manually implemented dense linear layers, activation, loss, and gradient operations. Derive every gradient shape and compare against autograd. Then add a residual connection and study Jacobian/gradient behavior.

**Deliverable:** A gradient-verification report and training trace, including at least one intentionally introduced bug that is detected by tests.

## 26. Case Study 7 — One Attention Head From Scratch

Implement embedding lookup, Q/K/V projections, scaled dot-product scores, causal masking, stable softmax, weighted values, and a projection back to the model dimension.

**Deliverable:**

- Annotated shape trace from token indices to output vectors.
- Forward reference comparison.
- Backward gradient comparison.
- A causal leakage test.
- Numerical stress tests with extreme logits and longer sequence length.
- Memory analysis of the \(T\times T\) attention intermediate.

## 27. Case Study 8 — Low-Rank Update and Quantized Inference

Take a sample matrix and evaluate a low-rank approximation, a factorized update, and a quantized representation. Separate storage savings from latency. Measure output deviation over representative input distributions, not just the weights themselves.

**Deliverable:** A Pareto-style comparison across quality proxy, latency, and memory, with limitations stated clearly.

---

# PART X — A DEPENDENCY-DRIVEN ADVANCED ROADMAP

This is a suggested sequence, not a requirement to finish one stage per calendar day. Advance when the exit criteria are satisfied. A shallow ten-day sprint is less valuable than a reliable mathematical and experimental foundation.

## Stage A — Operator thinking and numerical geometry

**Study:** Linear operators, adjoints, induced norms, matrix classes, spectral structure, condition numbers, perturbation analysis.  
**Build:** Matrix/norm utilities and conditioning experiments.  
**Exit criteria:** Explain forward error, backward error, conditioning, and why a low residual can coexist with a poor solution.

## Stage B — Stable direct methods

**Study:** Pivoted LU, QR variants, Cholesky, SVD, pseudoinverse, rank-revealing methods.  
**Build:** Transparent direct solvers and diagnostics.  
**Exit criteria:** Select a solver for a given matrix structure and justify the choice with assumptions, error measurements, and complexity.

## Stage C — Approximation and scalable solvers

**Study:** Krylov spaces, iterative methods, preconditioning, randomized SVD, low-rank approximation, sketching, sparse/matrix-free operators.  
**Build:** LSQR or CG implementation, plus a randomized low-rank pipeline.  
**Exit criteria:** Describe convergence dependencies and compare methods with reproducible compute/memory measurements.

## Stage D — Differential calculus and learning dynamics

**Study:** Differentials, matrix gradients, JVPs/VJPs, Jacobian products, conditioning of deep compositions, residual paths.  
**Build:** Manual MLP backward pass and gradient checker.  
**Exit criteria:** Derive and verify weight/input/bias gradients with no unexplained transpose.

## Stage E — Transformer linear algebra

**Study:** Embedding tables, projections, attention score geometry, stable softmax, MLP projections, vocabulary logits, low-rank updates and quantization error.  
**Build:** One attention head and a minimal Transformer block in a reference implementation.  
**Exit criteria:** Trace tensor dimensions and gradients, detect causal-mask errors, and explain the dominant compute/memory terms.

## Stage F — Systems-aware computation

**Study:** BLAS/LAPACK, memory layout, batching, mixed precision, arithmetic intensity, IO-aware algorithms, sparse and low-rank crossover points.  
**Build:** Benchmark suite with environment recording and result tables.  
**Exit criteria:** Explain performance results without confusing FLOPs, latency, memory bandwidth, and allocation overhead.

---

# PART XI — KNOWLEDGE CHECKS AND MASTERY GATES

## 28. Derivation Gate

You should be able to derive from definitions or explain assumptions for:

1. Rank-nullity theorem and the relation between null space and non-identifiability.
2. The SVD pseudoinverse and minimum-norm least-squares solution.
3. Orthogonal projection and residual orthogonality.
4. The relationship between the condition numbers of \(A\) and \(A^TA\) for full-column-rank \(A\).
5. The Eckart–Young low-rank approximation error formulas.
6. PCA as variance maximization and reconstruction minimization.
7. The gradient of \(\frac12\|Ax-b\|_2^2\).
8. The matrix-gradient formulas for a batched dense linear layer.
9. The adjoint relationship underlying reverse-mode differentiation.
10. The dimensions and major arithmetic cost of scaled dot-product attention.
11. The stable max-shifted softmax identity.
12. The role and limitations of a numerical-rank threshold.

## 29. Algorithm Selection Gate

For each scenario below, choose an approach and justify alternatives:

- A dense well-conditioned square solve with many right-hand sides.
- An ill-conditioned tall least-squares system.
- A rank-deficient least-squares problem requiring a minimum-norm answer.
- A large sparse SPD system.
- Only the leading few singular directions of a large matrix.
- A covariance matrix with sample count much smaller than feature dimension.
- A long-sequence attention computation constrained by accelerator memory.
- A model weight matrix that must be stored using fewer bits.

A correct answer must mention the assumptions and trade-offs, not only the algorithm name.

## 30. Debugging Gate

Given a failing numerical result, investigate systematically:

- Are dimensions and transposes correct?
- Are assumptions (symmetry, positive definiteness, rank) satisfied?
- Is the matrix ill-conditioned?
- Is the residual small, and is the backward error acceptable?
- Is there overflow, underflow, or catastrophic cancellation?
- Does reduced precision explain the discrepancy?
- Is the threshold for numerical rank reasonable and documented?
- Does a reference implementation use the same convention and scaling?
- Is nondeterminism from parallel reductions or kernel algorithms plausible?
- Is the performance bottleneck compute, memory, launch overhead, or synchronization?

## 31. Research-Reading Gate

When reading a paper or technical report involving a matrix algorithm, extract:

1. The mathematical problem and objective.
2. Required assumptions and data regime.
3. Exact versus approximate result.
4. Time, storage, and communication complexity.
5. Numerical stability and error guarantee.
6. Baseline and fair comparison conditions.
7. Hyperparameters and sensitivity analysis.
8. Failure cases and limitations.
9. Whether the evidence supports the authors' claims.
10. Which part you can reproduce with a small controlled experiment.

---

# PART XII — PROJECT STRUCTURE AND REPRODUCIBILITY

## 32. Suggested Repository Layout

```text
linear-algebra-advanced/
├── README.md
├── pyproject.toml
├── src/
│   └── lax/
│       ├── norms.py
│       ├── operators.py
│       ├── elimination.py
│       ├── lu.py
│       ├── qr.py
│       ├── cholesky.py
│       ├── least_squares.py
│       ├── eigen.py
│       ├── svd.py
│       ├── randomized.py
│       ├── iterative.py
│       ├── matrix_calculus.py
│       ├── attention.py
│       └── quantization.py
├── tests/
│   ├── test_shapes.py
│   ├── test_factorizations.py
│   ├── test_solvers.py
│   ├── test_numerics.py
│   ├── test_gradients.py
│   └── test_attention.py
├── experiments/
│   ├── conditioning.py
│   ├── qr_stability.py
│   ├── rank_thresholds.py
│   ├── randomized_svd.py
│   ├── mixed_precision.py
│   └── attention_memory.py
├── reports/
│   ├── experiment_template.md
│   └── results/
└── notebooks/
    └── visualizations.ipynb
```

The structure is a suggestion, not a requirement. Keep mathematical implementations, tests, experiments, and explanatory reports separate enough that results can be reproduced.

## 33. Experiment Report Template

Use this template for each substantive experiment.

```markdown
# Experiment: <name>

## Question
What measurable claim is being investigated?

## Hypothesis
What result is expected, and under which assumptions?

## Mathematical Setup
Define matrices, distributions, dimensions, rank, condition number, and objective.

## Algorithms
List methods compared and explain why each belongs in the comparison.

## Environment
- Python:
- NumPy/SciPy/PyTorch versions:
- CPU/GPU:
- BLAS/LAPACK backend if known:
- Dtype:
- Thread count:
- Random seed:

## Protocol
Specify input generation, repetitions, warm-up, synchronization, and timing boundaries.

## Metrics
Define each error, speed, memory, and quality metric precisely.

## Results
Include tables/plots and report variability where useful.

## Interpretation
Explain the observed result from the mathematical assumptions and algorithm behavior.

## Failure Analysis
State where the method degraded, failed, or contradicted the hypothesis.

## Limitations
State what this experiment does not establish.

## Reproduction
Provide the exact command and configuration required to rerun it.
```

## 34. Quality Rules for Mathematical Code

- Make shape contracts explicit in docstrings or type-level conventions.
- Check invalid dimensions and unsupported matrix structures.
- Do not silently replace a singular solve with an arbitrary answer.
- Expose solver tolerance and rank threshold where relevant.
- Do not compare runtimes before verifying correctness.
- Use deterministic seeds for experiments but test more than one seed when randomness is part of the claim.
- Avoid hidden global state and implicit dtype changes.
- Keep optimized implementations separate from the pedagogical version.
- Include tests for zero-sized dimensions only when the API intends to support them.
- Never report a benchmark without the matrix shape, dtype, and compute environment.

---

# PART XIII — MASTER CONCEPT MAP

## 35. The Full Dependency Graph

```text
Linear operators and vector spaces
        │
        ├── Kernel / image / rank-nullity
        │        └── Identifiability and rank deficiency
        │
        ├── Inner products / adjoints / norms
        │        ├── Orthogonality and projection
        │        ├── Operator gain and conditioning
        │        └── Backpropagation as adjoint propagation
        │
        ├── Matrix structure
        │        ├── Symmetric / Hermitian / PSD / SPD
        │        ├── Orthogonal / unitary / normal / nonnormal
        │        └── Quadratic forms and covariance geometry
        │
        ├── Factorizations
        │        ├── LU with pivoting ──> general direct solves
        │        ├── QR ────────────────> stable least squares
        │        ├── Cholesky ──────────> SPD systems
        │        ├── Schur/eigen ───────> spectral analysis
        │        └── SVD ───────────────> rank, pseudoinverse, low rank, PCA
        │
        ├── Numerical analysis
        │        ├── Floating-point error
        │        ├── Forward/backward error
        │        ├── Conditioning and perturbation
        │        └── Stable versus unstable algorithms
        │
        ├── Large-scale methods
        │        ├── Krylov methods and preconditioning
        │        ├── Sparse and matrix-free operators
        │        ├── Randomized SVD and sketching
        │        └── Low-rank approximation
        │
        ├── Matrix calculus
        │        ├── Differentials and gradients
        │        ├── JVP / VJP / adjoints
        │        └── Jacobian products and signal propagation
        │
        └── Deep learning / LLM
                 ├── Embedding lookup and parameter matrices
                 ├── Dense projections and MLP layers
                 ├── Scaled dot-product attention and softmax
                 ├── Backpropagation and gradient verification
                 ├── LoRA / low-rank updates
                 ├── Quantization and mixed precision
                 └── IO-aware kernels and system-level trade-offs
```

## 36. Concept-to-LLM Mapping

| Linear-algebra concept | Concrete LLM connection | Question to investigate |
|---|---|---|
| Matrix multiplication | Q/K/V projections, MLPs, vocabulary logits | Is the operation compute-bound or bandwidth-bound? |
| Inner product | Attention scores, similarity | How does dimension and normalization affect score distribution? |
| Adjoint/transpose | Backpropagation through linear maps | Can every gradient shape be derived from the forward map? |
| Singular values | Weight sensitivity, low-rank structure | Is the spectrum concentrated or broad? |
| Numerical rank | Effective dimensionality | How does the rank threshold affect interpretation? |
| QR/SVD | Stable solvers and decomposition | Which structure justifies the chosen method? |
| Condition number | Optimization and numerical sensitivity | Does small perturbation produce a large response? |
| Softmax Jacobian | Attention and token probabilities | How do extreme logits affect gradients and stability? |
| Low-rank factorization | Parameter-efficient adaptation | What rank captures the useful update under a fixed budget? |
| Quantization error | Low-bit inference/training | Which values and operations are most sensitive? |
| Sparse/matrix-free methods | Structured computation and large systems | Can the operator be applied without materializing it? |
| Memory traffic / arithmetic intensity | Efficient attention and accelerator kernels | Is data movement dominating arithmetic? |

---

# PART XIV — FINAL MASTERY STANDARD

## 37. Definition of Done

This module is complete only when the following evidence exists—not merely when all headings have been read.

### Mathematical fluency

- You can derive the central identities and explain their assumptions.
- You can reason about rank, null spaces, singular values, spectral structure, and conditioning without conflating them.
- You can separate exact properties from numerical approximations.
- You can derive gradients of core linear operations with explicit shapes.

### Algorithmic fluency

- You have implemented and tested the key direct factorizations and solver mechanisms.
- You know why an algorithm is appropriate for a matrix class and when it should be rejected.
- You have compared direct, iterative, randomized, dense, sparse, and low-rank methods where relevant.

### Numerical fluency

- You can detect ill-conditioning, cancellation, precision loss, unstable formulas, and unreliable rank decisions.
- You report residuals and other diagnostics rather than trusting a successful return value.
- You distinguish stability of the algorithm from conditioning of the mathematical problem.

### Deep-learning fluency

- You can implement a linear layer and its backward pass without autograd, then verify it.
- You can explain singular-value/Jacobian effects on signal and gradient propagation.
- You can implement a small attention head, validate its mask, and explain the \(T^2\) storage issue.
- You can reason about rank, quantization, and memory/compute trade-offs without overclaiming what a toy experiment proves.

### Research and engineering fluency

- Every major experimental claim is reproducible and falsifiable.
- Runtime, memory, numerical error, and task-level quality are measured separately.
- Results include failure cases and limitations.
- You can read a method section in a technical paper and identify its mathematical problem, assumptions, guarantees, costs, and evidence.

---

## 38. Final Principle

The goal is not to memorize more linear-algebra terminology. The goal is to build a mental model in which **structure determines algorithm choice, conditioning determines sensitivity, numerical analysis determines trustworthiness, and computational constraints determine what is feasible**.

For deep learning and LLMs, this chain matters repeatedly:

```text
mathematical formulation
        ↓
operator structure and assumptions
        ↓
algorithm and approximation choice
        ↓
numerical behavior and error analysis
        ↓
compute / memory / communication cost
        ↓
training or inference behavior
        ↓
controlled experiment and evidence
```

Do not conclude that a model component is understood because it runs. Derive the operation, verify the implementation, stress-test its assumptions, measure its costs, and explain what the evidence actually supports.
