# 00.2 — Advanced Calculus for Deep Learning, Transformers & LLM Research

> **Track:** Mathematical Foundations → Differential Calculus → Multivariable Calculus → Automatic Differentiation → Optimization Dynamics → Deep Learning/LLM Derivatives → Advanced Research Calculus  
> **Level:** Advanced undergraduate to graduate bridge; assumes algebra, functions, trigonometry, and elementary derivative familiarity.  
> **Primary objective:** Understand calculus deeply enough to derive, implement, verify, and diagnose the derivatives and continuous mathematical ideas used in neural networks, Transformers, optimization, probabilistic learning, and advanced AI research.

> **Scope rule:** This is a calculus roadmap, not a linear-algebra roadmap. Vectors, Jacobians, Hessians, and matrix notation appear only as tools for representing multivariable derivatives. The emphasis remains on limits, rates of change, differentiation, integration, composition, approximation, sensitivity, and optimization.

---

## Table of Contents

- [0. Mission, Outcomes, and Prerequisites](#0-mission-outcomes-and-prerequisites)
- [1. How to Study Every Topic](#1-how-to-study-every-topic)
- [2. Notation and Numerical Conventions](#2-notation-and-numerical-conventions)
- [Part I — Foundations of Continuous Change](#part-i--foundations-of-continuous-change)
  - [3. Day 1 — Functions, Limits, Continuity, and Real-Analysis Foundations](#3-day-1--functions-limits-continuity-and-real-analysis-foundations)
  - [4. Day 2 — Derivatives from First Principles](#4-day-2--derivatives-from-first-principles)
  - [5. Day 3 — Differentiation Rules and Implicit Dependence](#5-day-3--differentiation-rules-and-implicit-dependence)
  - [6. Day 4 — Mean Value Theorems, Taylor Expansion, and Error](#6-day-4--mean-value-theorems-taylor-expansion-and-error)
  - [7. Day 5 — Integration, Accumulation, and the Fundamental Theorem](#7-day-5--integration-accumulation-and-the-fundamental-theorem)
- [Part II — Multivariable Differential Calculus](#part-ii--multivariable-differential-calculus)
  - [8. Day 6 — Multivariable Limits, Continuity, and Differentiability](#8-day-6--multivariable-limits-continuity-and-differentiability)
  - [9. Day 7 — Partial Derivatives, Directional Derivatives, and Gradients](#9-day-7--partial-derivatives-directional-derivatives-and-gradients)
  - [10. Day 8 — Total Derivatives and the Multivariable Chain Rule](#10-day-8--total-derivatives-and-the-multivariable-chain-rule)
  - [11. Day 9 — Jacobians as Derivatives and JVP/VJP](#11-day-9--jacobians-as-derivatives-and-jvpvjp)
  - [12. Day 10 — Higher-Order Derivatives, Hessians, and Curvature](#12-day-10--higher-order-derivatives-hessians-and-curvature)
  - [13. Day 11 — Vector Calculus and Continuous Fields](#13-day-11--vector-calculus-and-continuous-fields)
- [Part III — Integration Beyond the Basics](#part-iii--integration-beyond-the-basics)
  - [14. Day 12 — Integration Techniques, Improper Integrals, and Convergence](#14-day-12--integration-techniques-improper-integrals-and-convergence)
  - [15. Day 13 — Multiple Integrals, Change of Variables, and Probability](#15-day-13--multiple-integrals-change-of-variables-and-probability)
  - [16. Day 14 — Differentiation Under the Integral Sign](#16-day-14--differentiation-under-the-integral-sign)
- [Part IV — Computational and Automatic Differentiation](#part-iv--computational-and-automatic-differentiation)
  - [17. Day 15 — Numerical Differentiation and Gradient Verification](#17-day-15--numerical-differentiation-and-gradient-verification)
  - [18. Day 16 — Computational Graphs and Automatic Differentiation](#18-day-16--computational-graphs-and-automatic-differentiation)
  - [19. Day 17 — Reverse-Mode Differentiation and Backpropagation](#19-day-17--reverse-mode-differentiation-and-backpropagation)
- [Part V — Calculus Inside Deep Learning and LLMs](#part-v--calculus-inside-deep-learning-and-llms)
  - [20. Day 18 — Loss Functions, Activations, and Gradient Pathologies](#20-day-18--loss-functions-activations-and-gradient-pathologies)
  - [21. Day 19 — Softmax, Cross-Entropy, and Attention Derivatives](#21-day-19--softmax-cross-entropy-and-attention-derivatives)
  - [22. Day 20 — Normalization, Residual Paths, and Deep Signal Flow](#22-day-20--normalization-residual-paths-and-deep-signal-flow)
  - [23. Day 21 — Gradient-Based Optimization as Calculus](#23-day-21--gradient-based-optimization-as-calculus)
- [Part VI — Advanced Optimization and Learning Dynamics](#part-vi--advanced-optimization-and-learning-dynamics)
  - [24. Day 22 — Constrained Optimization, Lagrange Multipliers, and KKT](#24-day-22--constrained-optimization-lagrange-multipliers-and-kkt)
  - [25. Day 23 — Implicit Differentiation, Hypergradients, and Bilevel Learning](#25-day-23--implicit-differentiation-hypergradients-and-bilevel-learning)
  - [26. Day 24 — Gradient Flow, Differential Equations, and Stability](#26-day-24--gradient-flow-differential-equations-and-stability)
  - [27. Day 25 — Probability Gradients and Stochastic Computation](#27-day-25--probability-gradients-and-stochastic-computation)
  - [28. Day 26 — Sensitivity, Lipschitzness, and Robustness](#28-day-26--sensitivity-lipschitzness-and-robustness)
- [Part VII — Advanced and Research-Oriented Calculus](#part-vii--advanced-and-research-oriented-calculus)
  - [29. Day 27 — Calculus of Variations and Functional Derivatives](#29-day-27--calculus-of-variations-and-functional-derivatives)
  - [30. Day 28 — Differential Equations, Adjoint Sensitivity, and Neural ODEs](#30-day-28--differential-equations-adjoint-sensitivity-and-neural-odes)
  - [31. Day 29 — Higher-Order Automatic Differentiation and Meta-Learning](#31-day-29--higher-order-automatic-differentiation-and-meta-learning)
  - [32. Day 30 — Research Integration: Build and Analyze a Tiny Transformer](#32-day-30--research-integration-build-and-analyze-a-tiny-transformer)
- [Part VIII — Implementation Labs from Scratch](#part-viii--implementation-labs-from-scratch)
- [Part IX — Experiments and Diagnostics](#part-ix--experiments-and-diagnostics)
- [Part X — Master Dependency Map](#part-x--master-dependency-map)
- [Part XI — Mastery Gates](#part-xi--mastery-gates)
- [Part XII — Project Structure and Reproducibility](#part-xii--project-structure-and-reproducibility)
- [Part XIII — Final Definition of Done](#part-xiii--final-definition-of-done)

---

# 0. Mission, Outcomes, and Prerequisites

Calculus is the mathematics of local change, accumulation, approximation, and the behavior of functions under perturbation. In deep learning, it explains how a loss changes when parameters or activations change, how gradients pass through a composition of operations, how optimization updates behave, and how continuous probability models are differentiated.

This module develops calculus in the following order:

1. Define continuous behavior using limits and continuity.
2. Build differentiation from the difference quotient.
3. Understand derivatives as local linear approximations, not merely slope formulas.
4. Extend derivatives to several variables and compositions.
5. Derive higher-order behavior and curvature.
6. Connect differentiation to integration, probability, and accumulation.
7. Implement numerical and automatic differentiation, then verify their correctness.
8. Derive the gradients used by neural networks and Transformers.
9. Study optimization dynamics, implicit gradients, and continuous-time learning models.

## Learning outcomes

By completing this roadmap, you should be able to:

- Explain the difference between continuity, differentiability, smoothness, and analyticity.
- Derive derivative rules from definitions and understand where each rule applies.
- Interpret derivatives as rates of change, local linear models, and sensitivity measures.
- Derive gradients, total derivatives, Jacobian-vector products, vector-Jacobian products, and Hessian-vector products.
- Use Taylor expansions to estimate approximation error and understand local optimization behavior.
- Explain the Fundamental Theorem of Calculus and the connection between integration and differentiation.
- Implement finite differences and automatic differentiation, including a minimal reverse-mode engine.
- Derive gradients for common loss functions, activations, softmax, cross-entropy, and attention.
- Diagnose vanishing/exploding gradients, saturation, poor conditioning, and numerical derivative failure.
- Understand constrained optimization, implicit differentiation, gradient flow, and basic Neural ODE sensitivity.
- Distinguish exact mathematical derivatives, numerical approximations, and derivatives computed by an automatic differentiation system.
- Produce experiments that test mathematical claims instead of merely demonstrating that code runs.

## Prerequisite checklist

Before starting, confirm practical familiarity with:

- Algebraic manipulation, equations, inequalities, and logarithms.
- Functions, inverse functions, exponentials, and trigonometric functions.
- Limits and elementary derivatives at least at an introductory level.
- Python, NumPy, plotting, unit tests, and basic floating-point types.
- Elementary probability: random variables, expectations, and distributions.
- Basic vector/matrix notation; no advanced linear algebra is required for this module.

If a prerequisite is missing, repair it in a short focused review and continue. Do not let a prerequisite gap turn the entire roadmap into an unstructured detour.

---

# 1. How to Study Every Topic

Use the same five-pass protocol for each major concept.

## Pass A — Define and derive

1. State the function, domain, codomain, and assumptions.
2. Start from the definition where practical.
3. Derive the result without skipping the chain rule or algebraic steps.
4. Separate a theorem from an intuition, approximation, or heuristic.
5. Identify whether the claim is local or global and whether the assumptions are necessary.

## Pass B — Visualize

- Plot the function and its derivative when possible.
- Compare a function to its tangent-line or tangent-plane approximation.
- Use contour plots for two-variable functions.
- Visualize gradient directions, level sets, and curvature.
- Plot numerical error as a function of step size or dtype.

## Pass C — Implement from scratch

- Implement the operation transparently before calling an autodiff framework.
- For derivative code, make the variable of differentiation explicit.
- Keep analytic, finite-difference, and automatic derivatives separate.
- For integrals, compare quadrature estimates against known exact results when available.

## Pass D — Verify and stress-test

- Compare analytic derivatives with central finite differences for small test cases.
- Use gradient checking carefully: finite differences themselves are approximate and may fail under poor scaling or nonsmooth operations.
- Test boundary values, saturated activations, extreme logits, and different scales.
- Report absolute error, relative error, and the relevant scale of the function or gradient.

## Pass E — Connect to learning systems

For each concept, answer:

- Where does this derivative appear in a neural network?
- What assumptions make the formula valid?
- How can the derivative become too small, too large, undefined, or numerically inaccurate?
- What evidence would show the implementation is correct?
- Which computation or learning behavior does this concept help explain?

## Study time suggestion

A “day” is a learning unit, not a promise that every topic can be mastered in one calendar day. For deep mastery, allocate time to derivation, implementation, experimentation, and review. Revisit any day whose mastery gate is not satisfied before advancing.

---

# 2. Notation and Numerical Conventions

- Scalar input: `x`; scalar-valued function: `f(x)`.
- Multivariable input: `x = (x₁, …, xₙ)`; scalar objective: `f(x)`.
- First derivative: `f'(x)` or `df/dx`.
- Partial derivative: `∂f/∂xᵢ`.
- Gradient: `∇f(x)`; its orientation depends on convention, so state it when deriving code.
- Total differential: `df`; small perturbation: `dx` or `δx`.
- Jacobian: `J_f(x)` or `Df(x)`.
- Hessian: `H_f(x)` or `∇²f(x)` for a scalar-valued function.
- Directional derivative in direction `v`: `D_v f(x)`.
- Integral: `∫ f(x) dx`; definite integral: `∫ₐᵇ f(x) dx`.
- Learning objective: `L(θ)`; model parameters: `θ`; gradient: `∇θ L`.
- Numerical differentiation step: `h`; machine precision and rounding depend on dtype.

**Numerical rule:** a derivative produced by software is not automatically exact. Automatic differentiation applies chain rules to the operations represented in its computational graph, subject to the numerical behavior of those operations. Finite differences introduce truncation and rounding error. Symbolic expressions may still be numerically unstable when evaluated.

---

# PART I — FOUNDATIONS OF CONTINUOUS CHANGE

## 3. Day 1 — Functions, Limits, Continuity, and Real-Analysis Foundations

### Core topics

1. Function domains, codomains, ranges, and composition.
2. One-sided limits and two-sided limits.
3. Infinite limits and limits at infinity.
4. Algebra of limits and indeterminate forms.
5. Continuity at a point and on an interval.
6. Removable, jump, and infinite discontinuities.
7. Epsilon-delta definition of a limit.
8. Sequential characterization of limits.
9. Intermediate Value Theorem.
10. Compactness intuition and why closed/bounded domains matter for extrema.
11. Continuity of compositions.
12. Continuity versus differentiability.

### Mathematics

The limit definition is

\[
\lim_{x\to a}f(x)=L
\]

when for every \(\varepsilon>0\), there exists \(\delta>0\) such that

\[
0<|x-a|<\delta \implies |f(x)-L|<\varepsilon.
\]

Continuity at \(a\) requires that \(f(a)\) exists, the limit exists, and

\[
\lim_{x\to a}f(x)=f(a).
\]

### Understand, do not only memorize

- A limit describes behavior near an input, not necessarily the function value at that input.
- Continuity is weaker than differentiability.
- A function may be continuous but nondifferentiable, such as \(f(x)=|x|\) at zero.
- A function can have a derivative at a point even when its derivative is not continuous in a neighborhood.
- In multiple variables, checking a few paths is enough to disprove a limit when they disagree, but agreeing along several paths does not prove the limit exists.

### Derivation tasks

1. Prove a simple polynomial limit using epsilon-delta reasoning.
2. Use the definition to show that \(f(x)=x^2\) is continuous at an arbitrary real number.
3. Explain why \(|x|\) has no two-sided derivative at zero.
4. Construct a piecewise function that is continuous at a point but has different one-sided derivatives there.
5. Examine a two-variable limit along different paths and distinguish evidence from proof.

### Implementation

- Plot discontinuous, continuous, and continuous-but-nondifferentiable functions.
- Write a simple numerical limit estimator and demonstrate why it cannot prove a limit.
- Sweep sample distances toward the target point and plot the observed function values.

### Diagnostics and failure modes

- Confusing the limit with the function value.
- Treating a numerical plot as proof.
- Assuming continuity implies differentiability.
- Assuming a multivariable limit exists because several tested paths agree.

### Case — Soft threshold and activation behavior

Analyze \(f(x)=\max(0,x)\). Locate its continuity and differentiability properties. Explain why the function can be used in optimization even though its classical derivative is undefined at one point, and state the conventional subgradient choice used in implementations.

### Mastery gate

You can state the formal limit and continuity definitions, apply them to examples, and distinguish an informal numerical observation from a mathematical proof.

---

## 4. Day 2 — Derivatives from First Principles

### Core topics

1. Difference quotients.
2. Derivative as instantaneous rate of change.
3. Derivative as tangent slope.
4. Derivative as best local linear approximation.
5. Left and right derivatives.
6. Differentiability implies continuity.
7. Nondifferentiable points: corners, cusps, vertical tangents, discontinuities.
8. Derivatives of constants, powers, and elementary functions.
9. Local sensitivity and units of a derivative.

### Mathematics

The derivative is

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},
\]

when the limit exists. The first-order approximation is

\[
f(x+h)=f(x)+f'(x)h+o(h),\qquad h\to0.
\]

The remainder term \(o(h)\) means that its ratio to \(|h|\) tends to zero. It is stronger and more precise than merely saying “the error is small.”

### Derivation tasks

1. Derive the derivative of \(x^2\), \(x^3\), and \(1/x\) from the difference quotient on their valid domains.
2. Derive \(d(\sin x)/dx\) using the trigonometric limit \(\lim_{h\to0}\sin h/h=1\).
3. Show that differentiability implies continuity.
4. Derive the tangent-line approximation to a function at \(x=a\).
5. Explain the unit of \(df/dx\) if \(f\) measures loss and \(x\) measures a parameter.

### Implementation

- Implement forward difference and central difference estimates.
- Compare the estimates against analytic derivatives for several functions.
- Sweep step size \(h\) across many orders of magnitude.

### Experiment — Step-size trade-off

Investigate why reducing \(h\) initially improves a finite-difference estimate but eventually makes rounding error significant. Plot error versus \(h\) on logarithmic axes and explain truncation error versus floating-point cancellation.

### Failure modes

- Treating a derivative as simply a line drawn on a plot.
- Forgetting domain restrictions.
- Assuming every function has a derivative everywhere.
- Believing a smaller finite-difference step is always more accurate.

### Case — Loss sensitivity

For a scalar loss \(L(\theta)\), use the derivative to estimate how a small parameter perturbation affects the loss:

\[
L(\theta+\Delta\theta)\approx L(\theta)+L'(\theta)\Delta\theta.
\]

State when this approximation is useful and when a large perturbation invalidates it.

### Mastery gate

You can derive a derivative from first principles and explain the meaning and limitations of the local linear approximation.

---

## 5. Day 3 — Differentiation Rules and Implicit Dependence

### Core topics

1. Sum, difference, product, and quotient rules.
2. Chain rule for nested scalar functions.
3. Exponential and logarithmic derivatives.
4. Trigonometric and inverse-trigonometric derivatives.
5. Logarithmic differentiation.
6. Implicit differentiation.
7. Inverse-function derivative theorem.
8. Parametric differentiation.
9. Related rates.
10. Domain and differentiability assumptions behind each rule.

### Mathematics

Product rule:

\[
\frac{d}{dx}[f(x)g(x)]=f'(x)g(x)+f(x)g'(x).
\]

Chain rule:

\[
\frac{d}{dx}f(g(x))=f'(g(x))g'(x).
\]

Implicit differentiation handles a relation \(F(x,y)=0\). Where the required derivative exists and \(\partial F/\partial y\ne0\),

\[
\frac{dy}{dx}=-\frac{F_x}{F_y}.
\]

### Derivation and practice

- Derive the product rule from the difference quotient.
- Derive the chain rule from the local behavior of nested functions.
- Differentiate \(\log(1+e^x)\) and simplify the result.
- Differentiate \(x^x\) for \(x>0\).
- Use implicit differentiation on a circle and on a nonlinear relation.
- Explain why differentiating \(f(g(x))\) requires evaluating the outer derivative at \(g(x)\).

### Implementation

Create a worksheet of at least 25 derivatives covering all rules. Verify a subset using symbolic differentiation and numerical finite differences. The goal is to discover structural mistakes, not to trust one tool as an oracle.

### Case — Log-likelihood and log-sum-exp

Study derivatives of logarithms of positive functions. Use them to prepare for stable log-likelihood calculations and the log-sum-exp expression:

\[
\operatorname{LSE}(z)=\log\sum_i e^{z_i}.
\]

Derive its derivative in terms of a normalized exponential distribution, then explain why subtracting the maximum logit improves numerical evaluation without changing the mathematical value.

### Failure modes

- Dropping an inner derivative in the chain rule.
- Applying logarithmic differentiation outside its domain.
- Treating an implicit relation as a globally single-valued function without checking conditions.
- Simplifying exponentials in a way that causes overflow during numerical evaluation.

### Mastery gate

You can apply the rules while explaining which dependency each term captures and which assumptions keep the derivation valid.

---

## 6. Day 4 — Mean Value Theorems, Taylor Expansion, and Error

### Core topics

1. Rolle's theorem.
2. Mean Value Theorem (MVT).
3. Cauchy Mean Value Theorem as an advanced extension.
4. Taylor polynomials.
5. Taylor's theorem with remainder.
6. Maclaurin expansions.
7. Local versus global approximation.
8. Lipschitz continuity and bounded derivatives.
9. Smoothness classes \(C^0,C^1,C^2,\ldots\).
10. Analyticity versus infinite differentiability.
11. Truncation error and approximation order.

### Mathematics

The Mean Value Theorem states that if \(f\) is continuous on \([a,b]\) and differentiable on \((a,b)\), there exists \(c\in(a,b)\) such that

\[
f'(c)=\frac{f(b)-f(a)}{b-a}.
\]

A Taylor expansion around \(a\) is

\[
f(a+h)=f(a)+f'(a)h+\frac{f''(a)}{2!}h^2+\cdots+\frac{f^{(n)}(a)}{n!}h^n+R_n(h).
\]

If the appropriate derivative bound exists, the remainder can be bounded. For example, under standard Taylor-theorem assumptions,

\[
|R_n(h)|\leq \frac{M|h|^{n+1}}{(n+1)!}
\]

when the absolute value of the \((n+1)\)-st derivative is bounded by \(M\) over the relevant interval.

### Derivation tasks

1. Use MVT to show that a differentiable function with zero derivative throughout an interval is constant.
2. Derive second-order Taylor approximation for a quadratic and a smooth nonlinear function.
3. Compare the Taylor approximations of \(e^x\), \(\sin x\), and \(\log(1+x)\) in their valid regions.
4. Investigate a smooth function whose Taylor series does not represent it globally, distinguishing smoothness from analyticity.
5. Derive how a bounded first derivative produces a Lipschitz bound on an interval.

### Implementation

- Implement Taylor polynomial evaluation.
- Measure absolute and relative approximation error as the expansion order changes.
- Plot approximation error against distance from the expansion point.

### Experiment — When local models fail

Compare first-, second-, and higher-order approximations across increasingly large parameter steps. Identify the region where a local approximation is reliable and explain why a derivative at one point cannot guarantee global behavior.

### Case — A gradient step as a Taylor approximation

For a differentiable loss,

\[
L(\theta-\eta g)\approx L(\theta)-\eta\nabla L(\theta)^Tg,
\]

where \(g\) is a proposed update direction. For a gradient step, \(g=\nabla L(\theta)\), the first-order term suggests a decrease when the step is sufficiently small, but higher-order terms determine whether a finite step actually decreases the loss.

### Failure modes

- Using a Taylor series outside its useful region.
- Ignoring the remainder term.
- Confusing a local derivative with a global Lipschitz constant.
- Claiming a gradient step must decrease loss for any learning rate.

### Mastery gate

You can derive a Taylor approximation, estimate its error, and connect local curvature to the behavior of finite optimization steps.

---

## 7. Day 5 — Integration, Accumulation, and the Fundamental Theorem

### Core topics

1. Antiderivatives and indefinite integrals.
2. Riemann sums and definite integrals.
3. Signed area versus geometric area.
4. Integrability and continuity as a sufficient condition for Riemann integrability.
5. Fundamental Theorem of Calculus (FTC), Parts I and II.
6. Net accumulation and cumulative functions.
7. Average value of a function.
8. Substitution as reverse chain rule.
9. Integration by parts as reverse product rule.
10. Elementary numerical quadrature.

### Mathematics

For continuous \(f\), define

\[
F(x)=\int_a^x f(t)\,dt.
\]

The first part of the Fundamental Theorem gives

\[
F'(x)=f(x).
\]

If \(F'=f\) on \([a,b]\), then

\[
\int_a^b f(x)\,dx=F(b)-F(a).
\]

### Derivation tasks

- Derive the area approximation using Riemann sums.
- Explain why the definite integral can be negative.
- Use FTC to differentiate an accumulation function.
- Derive substitution from the chain rule.
- Derive integration by parts from the product rule.

### Implementation

Implement left/right Riemann sums, midpoint rule, and trapezoidal rule. If ready, implement Simpson's rule and compare the rates of convergence on sufficiently smooth functions.

### Experiment — Quadrature accuracy

For functions with known integrals, vary the number of intervals and plot absolute error against interval width. Compare smooth integrands with integrands that have a discontinuity or singularity.

### Case — Expected loss as an integral

For a continuous random variable with density \(p(x)\), the expected loss can be represented as

\[
\mathbb{E}[L(X)]=\int L(x)p(x)\,dx,
\]

when the expectation exists. Interpret the integral as weighted accumulation, not merely area under an arbitrary curve.

### Failure modes

- Confusing an antiderivative with the definite integral.
- Forgetting that integrability and convergence require conditions.
- Using a quadrature rule without examining discontinuities or tails.
- Treating a finite numerical integral as exact.

### Mastery gate

You can explain the two parts of FTC, implement basic numerical integration, and use an integral to express accumulation or expectation.

---

# PART II — MULTIVARIABLE DIFFERENTIAL CALCULUS

## 8. Day 6 — Multivariable Limits, Continuity, and Differentiability

### Core topics

1. Functions of several variables.
2. Domains, level sets, and contour plots.
3. Multivariable limits and path dependence.
4. Continuity in multiple dimensions.
5. Partial differentiability versus total differentiability.
6. Linear approximation and the error condition for differentiability.
7. Directional behavior near singular points.
8. Differentiability as a stronger condition than existence of partial derivatives.

### Mathematics

A function \(f:\mathbb{R}^n\to\mathbb{R}\) is differentiable at \(a\) if a linear map \(L\) exists such that

\[
f(a+h)=f(a)+L(h)+o(\|h\|)\quad\text{as }h\to0.
\]

The central point is that one linear approximation must work for every direction of approach, not merely along coordinate axes.

### Derivation tasks

- Test limits along multiple paths and identify a counterexample when paths disagree.
- Construct or study a function whose partial derivatives exist at a point but which is not continuous there.
- Compare directional derivatives with total differentiability.
- Derive a tangent-plane approximation for a smooth two-variable function.

### Implementation

- Generate contour and surface plots for representative functions.
- Visualize tangent planes and approximation errors in neighborhoods around a point.
- Numerically sample errors along many directions; state why sampling is evidence rather than proof.

### Diagnostics

- A few successful paths do not establish a multivariable limit.
- Existence of all partial derivatives does not by itself imply differentiability.
- A directional derivative in every direction does not automatically imply total differentiability without suitable conditions.

### Case — Parameterized loss surface

Visualize a scalar loss in two parameter dimensions. Compare the actual loss near a point with its first-order affine approximation and determine how the error changes as the neighborhood radius shrinks.

### Mastery gate

You can distinguish partial derivatives, directional derivatives, and a total derivative, and you understand why multidimensional differentiability requires a uniform local approximation.

---

## 9. Day 7 — Partial Derivatives, Directional Derivatives, and Gradients

### Core topics

1. Partial derivatives and notation.
2. Gradient as the collection of first partial derivatives.
3. Directional derivative.
4. Gradient as the direction of steepest increase under the Euclidean metric.
5. Level sets and normal directions.
6. Tangent planes and linearization.
7. Coordinate scaling and the meaning of “steepest.”
8. Gradient units and parameter scaling.
9. Partial derivative versus causal effect.

### Mathematics

For differentiable scalar-valued \(f\),

\[
\nabla f(x)=\left(\frac{\partial f}{\partial x_1},\ldots,\frac{\partial f}{\partial x_n}\right).
\]

For a unit direction \(v\),

\[
D_vf(x)=\nabla f(x)^Tv.
\]

By the Cauchy–Schwarz inequality, among unit directions in the Euclidean metric the directional derivative is maximized by the gradient direction when the gradient is nonzero. This interpretation depends on the metric and coordinate scale used to define a unit direction.

### Derivation tasks

1. Derive the gradient of a multivariable polynomial.
2. Derive the tangent plane to \(z=f(x,y)\).
3. Derive the directional derivative using a one-dimensional path \(x(t)=x+tv\).
4. Prove the steepest-ascent statement under the Euclidean norm.
5. Explain how rescaling a parameter changes gradient coordinates and can change optimization behavior.

### Implementation

- Implement the gradient of a scalar function using coordinate-wise partial derivatives.
- Estimate directional derivatives using directional finite differences.
- Plot gradient arrows over a contour map.

### Experiment — Coordinate scaling

Optimize the same function in original and rescaled coordinates. Compare gradient magnitudes, step trajectories, and convergence. Explain why a gradient is not a scale-free recommendation independent of parameterization.

### Case — Gradient descent direction

For an objective \(L(\theta)\), derive the first-order predicted change for an update \(\Delta\theta\). Verify numerically whether the negative-gradient direction locally decreases the loss for sufficiently small steps.

### Failure modes

- Treating the gradient as a direction without accounting for scale.
- Confusing a partial derivative with an entire system's response.
- Assuming the gradient always points directly to a global minimum.
- Using gradient magnitude alone as a complete measure of training progress.

### Mastery gate

You can derive and visualize gradients, directional derivatives, and tangent-plane approximations and explain the metric assumptions behind steepest descent.

---

## 10. Day 8 — Total Derivatives and the Multivariable Chain Rule

### Core topics

1. Composite multivariable functions.
2. Total derivative and total differential.
3. Chain rule along a parameterized path.
4. Nested dependencies and dependency graphs.
5. Direct versus indirect effects.
6. Derivatives of functions composed across several stages.
7. Sensitivity through a sequence of transformations.
8. Derivative of an implicitly defined quantity (preview of implicit differentiation).

### Mathematics

If \(y=f(u)\) and \(u=g(x)\), then in scalar form

\[
\frac{dy}{dx}=\frac{df}{du}\frac{du}{dx}.
\]

For a path \(x(t)\),

\[
\frac{d}{dt}f(x(t))=\nabla f(x(t))^T\frac{dx}{dt}.
\]

For nested vector-valued operations, the total derivative composes the corresponding local linear maps. Shape and orientation conventions must be stated when writing Jacobian products.

### Derivation tasks

- Derive the chain rule for a two-stage scalar composition from the derivative definition.
- Derive \(dL/dt\) for a scalar loss that depends on several time-varying intermediate variables.
- Trace a derivative through a three-operation computational graph.
- Separate the direct derivative of a variable from the indirect derivative carried through dependent variables.

### Implementation

Represent a small computational graph in Python, store each operation's local derivative, and multiply local sensitivities in the correct order. Use simple scalar functions first, then add vector-valued intermediates.

### Experiment — Deep composition

Compose functions with derivatives smaller than one, larger than one, or varying across inputs. Measure how the total derivative changes with depth and compare it with the product of local derivatives.

### Case — A three-layer network

For

\[
h_1=\phi_1(W_1x+b_1),\quad h_2=\phi_2(W_2h_1+b_2),\quad L=\ell(h_2),
\]

derive the dependency structure of \(dL/dW_1\). The focus is the chain of dependencies, not a full linear-algebra treatment of the weight matrices.

### Failure modes

- Forgetting intermediate dependencies.
- Reordering scalar factors or Jacobian products incorrectly.
- Mixing up derivatives with respect to inputs, activations, and parameters.
- Confusing an operation's local derivative with the derivative of the whole composition.

### Mastery gate

You can derive derivatives through nested functions without relying on an autodiff framework and explain every factor in the result.

---

## 11. Day 9 — Jacobians as Derivatives and JVP/VJP

> **Purpose:** Learn Jacobians as a notation for multivariable differentiation. This is not a matrix-algebra unit.

### Core topics

1. Derivative of a vector-valued function.
2. Jacobian as the local linear map.
3. Shape conventions for Jacobians.
4. Jacobian-vector product (JVP).
5. Vector-Jacobian product (VJP).
6. Directional sensitivity without materializing a full Jacobian.
7. Forward-mode versus reverse-mode automatic differentiation.
8. Adjoint relationship as the mathematical basis for backpropagation.

### Mathematics

For \(f:\mathbb{R}^n\to\mathbb{R}^m\), the Jacobian contains the partial derivatives of output components with respect to input components. Under the common row-output convention,

\[
J_f(x)_{ij}=\frac{\partial f_i}{\partial x_j}.
\]

The first-order behavior is

\[
f(x+h)\approx f(x)+J_f(x)h.
\]

A JVP computes \(J_f(x)v\) without necessarily materializing \(J_f\). A VJP computes \(u^TJ_f(x)\), or its equivalent under the chosen gradient orientation.

### Study tasks

- Derive the Jacobian for a small nonlinear function with two inputs and two outputs.
- Verify the local approximation as the input perturbation shrinks.
- Calculate JVP and VJP directly from a small explicit Jacobian.
- Explain why a scalar-output function often favors reverse mode when there are many parameters but few outputs.
- Explain why forward mode can be attractive when there are few input directions of interest.

### Implementation

- Implement an explicit Jacobian for a small function.
- Implement a JVP using finite differences and compare it to the analytic product.
- Implement a VJP by applying chain rules to a tiny graph.
- Compare explicit-Jacobian memory with directional product memory.

### Diagnostics

- Jacobian entry errors due to transposed shape conventions.
- Building a full Jacobian when only a product is needed.
- Assuming JVP/VJP avoids all cost; the cost depends on the graph and chosen differentiation mode.
- Treating a VJP as an ordinary forward pass.

### Case — Input sensitivity of a model

For a scalar score \(s(x)\), investigate the directional sensitivity \(\nabla_x s(x)^Tv\). Explain why one direction provides only local information and does not establish global robustness.

### Mastery gate

You can define the Jacobian correctly, compute JVP/VJP products, and select a differentiation mode based on input/output dimensions and computational graph structure.

---

## 12. Day 10 — Higher-Order Derivatives, Hessians, and Curvature

### Core topics

1. Second derivatives and mixed partial derivatives.
2. Conditions for equality of mixed partials.
3. Hessian as second-order local behavior.
4. Multivariable Taylor expansion.
5. Curvature and local quadratic approximation.
6. Convexity tests from second derivatives.
7. Hessian-vector products (HVPs).
8. Third-order derivatives and directional higher-order derivatives.
9. Finite-difference Hessian checks.
10. Why explicitly materializing higher-order derivative objects can be expensive.

### Mathematics

For a twice-differentiable scalar function, the Hessian is the collection of second partial derivatives. The local second-order model is

\[
f(x+h)\approx f(x)+\nabla f(x)^Th+\frac12 h^T\nabla^2 f(x)h.
\]

Under suitable continuity assumptions on second partial derivatives, mixed partial derivatives agree. A positive semidefinite Hessian at a point is a local second-order condition, not by itself a universal guarantee of a global minimum for an arbitrary nonconvex function.

### Derivation tasks

- Compute the gradient and Hessian of quadratic, polynomial, and nonlinear functions.
- Derive the second-order Taylor formula along a path.
- Classify stationary points using eigenvalue/sign information only after stating the required assumptions.
- Construct a saddle point and distinguish it from a local minimum.
- Derive the directional second derivative \(v^T H v\).

### Implementation

- Compute Hessians analytically for small functions.
- Approximate Hessians using finite differences of gradients.
- Implement a Hessian-vector product without storing the complete Hessian, using directional differentiation or automatic differentiation when available.

### Experiment — Curvature versus gradient magnitude

Compare points with similar gradient norms but different curvature. Measure how a fixed learning rate behaves and explain why gradient size alone is insufficient for choosing a safe step.

### Case — Second-order local model of a loss

Compare actual changes in a two-parameter loss with its first- and second-order Taylor models. Identify the neighborhood where each model is useful.

### Failure modes

- Assuming the Hessian is positive definite just because a point is stationary.
- Treating an indefinite Hessian as a numerical bug.
- Ignoring the assumptions behind mixed-partial equality.
- Constructing a dense Hessian for a model when an HVP is the actual requirement.

### Mastery gate

You can use second-order derivatives to reason about local curvature, stationary-point classification, and the limitations of first-order optimization.

---

## 13. Day 11 — Vector Calculus and Continuous Fields

### Core topics

1. Scalar fields and vector fields.
2. Gradient field.
3. Divergence and curl.
4. Conservative fields and potential functions.
5. Line integrals.
6. Fundamental theorem for line integrals.
7. Green's theorem.
8. Surface integrals and flux.
9. Divergence theorem.
10. Stokes' theorem.
11. Path independence and simply connected domains.
12. Connections to physical systems and continuous probability flows.

### Mathematics

For a vector field \(F=(F_1,\ldots,F_n)\), divergence measures local net expansion/contraction, while curl describes local rotational behavior in three dimensions. A conservative field can be represented as the gradient of a potential under suitable domain conditions.

For a sufficiently smooth potential \(\phi\) and a path from \(a\) to \(b\),

\[
\int_C \nabla\phi\cdot dr=\phi(b)-\phi(a).
\]

Green's, divergence, and Stokes' theorems relate integrals over regions or surfaces to integrals over their boundaries under appropriate smoothness and geometry conditions.

### Study tasks

- Compute gradient, divergence, and curl for simple scalar/vector fields.
- Identify conservative fields and find a potential when one exists.
- Evaluate a line integral directly and using a potential function.
- Draw the geometry behind Green's theorem and the divergence theorem.
- Explain why domain topology and boundary orientation matter.

### Implementation

- Plot two-dimensional vector fields.
- Numerically approximate a line integral along a parameterized curve.
- Compare the direct integral with the potential-difference result for a conservative field.

### AI connection

These topics are not central to ordinary dense-layer backpropagation, but they provide mathematical tools for continuous normalizing flows, physical modeling, vector-field learning, energy-based models, and continuous-time neural systems. Keep this section conceptually solid without allowing it to displace the derivative skills required for LLM engineering.

### Mastery gate

You can interpret field derivatives geometrically, evaluate basic line integrals, and explain the conditions behind the major integral theorems.

---

# PART III — INTEGRATION BEYOND THE BASICS

## 14. Day 12 — Integration Techniques, Improper Integrals, and Convergence

### Core topics

1. Substitution and integration by parts.
2. Partial fractions and trigonometric substitutions.
3. Improper integrals over infinite intervals.
4. Integrals with singular endpoints.
5. Absolute versus conditional convergence for relevant improper integrals.
6. Comparison tests for convergence.
7. Numerical integration near singularities.
8. Integration error and adaptive quadrature.

### Tasks

- Derive integration by parts and use it on logarithmic and exponential integrands.
- Analyze when an improper integral converges.
- Compare truncating an infinite-domain integral with a change of variables that maps an infinite interval to a finite interval.
- Implement trapezoidal and adaptive-style interval refinement for selected integrands.

### Experiment

Compare quadrature on a smooth bounded integrand, a slowly decaying tail, and an endpoint singularity. Report whether error comes from discretization, domain truncation, or the function's behavior.

### Case — Normalization constants

A probability density must integrate to one. Examine a candidate density, determine whether its integral is finite, and identify the normalization constant only when it exists.

### Failure modes

- Assuming every improper integral converges.
- Ignoring tail behavior.
- Comparing quadrature methods without controlling the target accuracy and evaluation budget.

### Mastery gate

You can judge convergence for common improper integrals and choose an integration strategy consistent with the integrand's behavior.

---

## 15. Day 13 — Multiple Integrals, Change of Variables, and Probability

### Core topics

1. Double and triple integrals.
2. Iterated integrals and Fubini/Tonelli intuition.
3. Integration over non-rectangular domains.
4. Change of variables in multiple dimensions.
5. Jacobian determinant as a volume-scaling factor.
6. Polar, cylindrical, and spherical coordinates.
7. Joint probability densities and marginalization.
8. Conditional densities and integration domains.
9. Expectation and variance using integrals.
10. Continuous entropy as an integral, with appropriate qualifications.

### Mathematics

For a differentiable invertible change of variables \(x=g(u)\), the integration measure transforms through the absolute determinant of the derivative map:

\[
\int_{\Omega_x} f(x)\,dx
=\int_{\Omega_u} f(g(u))\,|\det Dg(u)|\,du,
\]

under the appropriate change-of-variables assumptions. The determinant factor accounts for local volume scaling; it is not an optional correction.

### Derivation tasks

- Evaluate a double integral over a simple region in two different integration orders.
- Derive the polar-coordinate volume factor \(r\).
- Show how a joint density is marginalized into a one-variable density.
- Derive expectation from a density and verify a known distribution.
- Explain why the change-of-variables formula requires conditions on the transformation and domain.

### Implementation

- Numerically approximate a two-dimensional integral.
- Estimate an expectation by quadrature and Monte Carlo, then compare both error types.
- Validate a change of variables on an integral with a known closed-form answer.

### Case — Latent-variable models

Understand that integrating over latent variables can define a marginal likelihood, but the integral may be intractable at model scale. Explain the distinction between the exact mathematical quantity and a computational approximation.

### Mastery gate

You can use multiple integration to reason about joint distributions and understand the calculus behind density transformations and marginalization.

---

## 16. Day 14 — Differentiation Under the Integral Sign

### Core topics

1. Parameter-dependent integrals.
2. Leibniz integral rule.
3. Differentiating an integral with fixed limits.
4. Differentiating when limits depend on parameters.
5. Conditions for interchanging derivative and integral.
6. Dominated convergence intuition.
7. Score-function gradient identity.
8. Gradient of an expected objective.

### Mathematics

For suitable regularity conditions, fixed integration limits, and a differentiable integrand,

\[
\frac{d}{d\theta}\int_a^b f(x,\theta)\,dx
=\int_a^b \frac{\partial f(x,\theta)}{\partial\theta}\,dx.
\]

With parameter-dependent limits, the full Leibniz rule includes boundary terms. Interchanging derivative and integration is not automatically valid; integrability and domination/regularity conditions are important.

### Derivation tasks

- Derive the Leibniz rule for a simple integral with a moving boundary.
- Differentiate an expectation expressed as an integral when the distribution and when the integrand depend on a parameter.
- Derive the score-function identity under explicit support and differentiability assumptions.
- Identify a case where the support depends on the parameter and explain why naive differentiation can fail.

### Implementation

Compare the derivative of an integral estimated by finite differences against an integral of the analytic derivative for a well-behaved example. Separately examine a deliberately problematic example and explain which assumption is missing.

### Case — Expected risk

For

\[
J(\theta)=\mathbb{E}_{X\sim p}[\ell(\theta,X)],
\]

identify conditions under which

\[
\nabla_\theta J(\theta)=\mathbb{E}_{X\sim p}[\nabla_\theta\ell(\theta,X)].
\]

Explain why this relationship supports stochastic gradient estimation and what changes when the sampling distribution also depends on \(\theta\).

### Failure modes

- Swapping derivative and integral without conditions.
- Omitting moving-boundary terms.
- Confusing a gradient of the integrand with a gradient of the distribution.
- Assuming unbiasedness implies low variance.

### Mastery gate

You can state why differentiation under the integral sign is useful and name the regularity conditions that justify it in a given setting.

---

# PART IV — COMPUTATIONAL AND AUTOMATIC DIFFERENTIATION

## 17. Day 15 — Numerical Differentiation and Gradient Verification

### Core topics

1. Forward finite differences.
2. Backward finite differences.
3. Central differences.
4. Truncation error orders.
5. Floating-point rounding error.
6. Complex-step differentiation and its assumptions/limitations.
7. Finite-difference directional derivatives.
8. Gradient checking for multivariable functions.
9. Scale-aware absolute and relative tolerances.
10. Nonsmooth operations and unreliable checks near kinks.

### Mathematics

Forward difference:

\[
f'(x)\approx\frac{f(x+h)-f(x)}{h}.
\]

Central difference:

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
\]

For sufficiently smooth functions, central difference has a higher-order truncation error than forward difference, but both are affected by rounding and cancellation. The best step size depends on scale, smoothness, precision, and the function evaluation's numerical behavior.

### Implementation

Build a derivative-checking module that accepts a function, input, analytic/autodiff gradient, and finite-difference scheme. Report per-coordinate absolute and relative discrepancies.

### Experiment

- Compare forward and central difference error against step size.
- Test a well-scaled polynomial, a saturated sigmoid, a function with cancellation, and a piecewise function near a kink.
- Compare float32 and float64.

### Diagnostics

- A passing gradient check on one input is not proof for all inputs.
- A failed check can indicate a wrong derivative, poor step size, nondeterminism, a nonsmooth point, or precision problems.
- Relative error becomes misleading when both expected and computed values are near zero; report absolute error too.

### Case — Verify a training loss

Before training a neural network, gradient-check a small parameter subset or tiny model. Keep the test problem small enough that finite differences are affordable.

### Mastery gate

You can distinguish a derivative bug from a numerical gradient-checking failure using controlled evidence.

---

## 18. Day 16 — Computational Graphs and Automatic Differentiation

### Core topics

1. Expression graphs and primitive operations.
2. Local derivatives per operation.
3. Forward-mode automatic differentiation with dual numbers.
4. Reverse-mode automatic differentiation.
5. Tangent values and cotangent/adjoint values.
6. Derivative accumulation at branching nodes.
7. Topological order and graph traversal.
8. Tape-based differentiation.
9. Static versus dynamic computational graphs.
10. Differentiability rules for primitive operations.

### Mathematics

Automatic differentiation repeatedly applies exact chain-rule identities to the sequence of primitive operations in a computation. “Exact” here means exact differentiation of the represented operations in real arithmetic; actual floating-point evaluation remains approximate.

For a primitive operation \(z=\phi(u)\), the engine records or knows its local derivative. Forward mode propagates input tangents; reverse mode propagates output sensitivities backward through the graph.

### From scratch

Build a tiny scalar autodiff engine supporting:

- Addition, subtraction, multiplication, division.
- Negation, exponentials, logarithms, powers.
- Sine, cosine, and tanh.
- A stable log-sum-exp primitive or a clearly decomposed equivalent.
- A graph with shared subexpressions and correct gradient accumulation.
- A topological backward pass.

Do not begin with a large neural network. Validate each primitive in isolation first.

### Verification

1. Compare against analytic derivatives.
2. Compare against finite differences.
3. Test an expression where one variable affects multiple graph paths.
4. Test repeated use of the same intermediate node.
5. Test input scaling and numerical extremes.

### Case — A tiny reverse-mode engine

For a composite scalar expression, draw the graph, calculate the local derivative at every node, and show exactly how the final gradient accumulates. Then make the engine produce that result.

### Failure modes

- Overwriting rather than accumulating gradients at shared nodes.
- Evaluating nodes in an invalid order.
- Forgetting to reset gradients between backward passes.
- Incorrect local derivatives for domain-restricted operations.
- Assuming automatic differentiation fixes unstable forward computations.

### Mastery gate

You can build a correct small reverse-mode autodiff system and explain the graph's forward values and backward sensitivities.

---

## 19. Day 17 — Reverse-Mode Differentiation and Backpropagation

### Core topics

1. Backpropagation as repeated chain-rule application.
2. Computational graph decomposition.
3. Local gradients for affine maps and elementwise activations.
4. Reverse accumulation through branching paths.
5. Loss gradients versus parameter gradients.
6. Batch-reduction semantics: sum versus mean.
7. Gradient shapes and broadcasting reduction.
8. Memory/computation trade-offs in storing activations.
9. Checkpointing and recomputation conceptually.
10. Backpropagation through time as repeated composition.

### Derivation template

For a chain

\[
x\to h_1\to h_2\to\cdots\to L,
\]

calculate the gradient by multiplying or composing each local derivative in reverse dependency order. At each node, explain what the incoming sensitivity means and which local derivative transforms it.

### From-scratch labs

- Scalar computational graph.
- Two-layer multilayer perceptron with a scalar loss.
- Vectorized mini-batch mean-squared error.
- Binary logistic regression.
- A small recurrent computation unrolled for a few steps.

Each implementation must include a test that compares gradients to an independent reference.

### Experiment — Reduction semantics

Train or evaluate the same tiny model with summed versus averaged losses. Show how the gradient scale changes with batch size and explain why the learning rate may need corresponding adjustment.

### Diagnostics

- Correct local derivatives but wrong parameter accumulation.
- Batch dimension accidentally summed or averaged twice.
- Broadcasting forward correctly but failing to reduce gradient contributions over broadcast dimensions.
- In-place mutation destroying values needed by the backward pass.
- Excessive memory use when storing every intermediate activation.

### Case — MLP backward pass

Trace one scalar loss back through output loss, output activation, hidden activation, and first layer. Explain every intermediate gradient instead of presenting only a final formula.

### Mastery gate

You can derive and implement backpropagation for a small MLP without autograd, then match an established framework within justified numerical tolerances.

---

# PART V — CALCULUS INSIDE DEEP LEARNING AND LLMs

## 20. Day 18 — Loss Functions, Activations, and Gradient Pathologies

### Core topics

1. Mean squared error and its derivative.
2. Binary cross-entropy and logistic loss.
3. Negative log-likelihood.
4. Sigmoid, tanh, ReLU, GELU, and smooth approximations.
5. Saturation and derivative magnitude.
6. Chain products across depth.
7. Vanishing and exploding gradient mechanisms.
8. Initialization as a way to influence signal/gradient scale.
9. Residual connections and gradient paths.
10. Subgradients for piecewise-smooth functions.
11. Loss scaling and gradient clipping as interventions, not mathematical cures.

### Mathematics and derivation tasks

Derive gradients from definitions for the common loss functions. For sigmoid,

\[
\sigma(x)=\frac{1}{1+e^{-x}},\qquad
\sigma'(x)=\sigma(x)(1-\sigma(x)).
\]

Show why the derivative becomes small for large-magnitude inputs. Derive the derivative of tanh and compare its saturation behavior. Analyze the ReLU derivative away from its kink and state the implementation convention at the kink.

### Experiment — Saturation and depth

- Plot activation and derivative functions over wide input ranges.
- Compose repeated scalar nonlinear maps and monitor derivative products.
- Compare gradient norms under several input scales and initializations.
- Explain why this controlled scalar experiment illustrates a mechanism but does not fully predict deep-network behavior.

### Failure modes

- Claiming all gradients vanish in every deep network.
- Confusing a nonsmooth point with an operation that cannot be differentiated almost everywhere.
- Treating gradient clipping as a guarantee of stable learning.
- Ignoring the difference between loss scale and model sensitivity.

### Case — Binary classifier from first principles

Derive the binary cross-entropy gradient through the sigmoid output, implement it, and compare it to a numerically stable combined logit-loss implementation. Test extreme logits.

### Mastery gate

You can derive common loss/activation gradients and explain saturation, chain-product effects, and numerical stability without conflating them.

---

## 21. Day 19 — Softmax, Cross-Entropy, and Attention Derivatives

### Core topics

1. Softmax as normalized exponentials.
2. Softmax Jacobian.
3. Log-sum-exp and numerical stabilization.
4. Softmax-cross-entropy gradient simplification.
5. Temperature scaling and derivative effects.
6. Masked softmax and invalid positions.
7. Scaled dot-product attention as a composition of differentiable operations.
8. Gradient paths through queries, keys, values, and output weights.
9. Attention saturation and score-scale sensitivity.
10. Gradient behavior for long contexts and large logits.

### Mathematics

For logits \(z\),

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

The softmax derivative is

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
\]

For a target distribution \(y\) and cross-entropy \(L=-\sum_i y_i\log p_i\), when \(y\) is a fixed target distribution and \(p=\operatorname{softmax}(z)\),

\[
\frac{\partial L}{\partial z_i}=p_i-y_i.
\]

This simplified result follows from the composition of log-softmax/cross-entropy and softmax; it does not mean the derivative of softmax alone is simply \(p-y\).

### Attention derivative roadmap

For scaled dot-product attention,

\[
S=\frac{QK^T}{\sqrt{d_k}}+M,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
\]

study the chain in this order:

1. Derivative of the loss with respect to \(O\).
2. Gradient from \(O=PV\) to \(P\) and \(V\).
3. Gradient through row-wise softmax from \(P\) to \(S\).
4. Gradient from \(S\) to \(Q\) and \(K\).
5. Gradient through input projections to the underlying model parameters.
6. Effect of the scaling factor, temperature, and mask.

The purpose is calculus through a composed operation, not a separate linear-algebra unit.

### From-scratch tasks

- Implement stable softmax.
- Implement softmax-cross-entropy and its analytic gradient.
- Implement a single attention head for a tiny input sequence.
- Implement backward derivatives for softmax and attention, then compare against PyTorch autograd and finite differences on a tiny example.

### Experiments

- Compare unshifted softmax with max-subtracted stable softmax for extreme logits.
- Vary logit scale and plot entropy and gradient magnitudes.
- Compare temperature values and explain how they change distribution sharpness and derivative scale.
- Validate masked positions receive no probability mass and no unintended gradient contribution.

### Failure modes

- Overflow in direct exponentiation.
- Incorrect softmax axis.
- Incorrect handling of causal/padding masks.
- Confusing the softmax Jacobian with the combined softmax-cross-entropy gradient.
- Assuming low entropy always implies a useful or harmful gradient independent of context.

### Mastery gate

You can derive and implement stable softmax-cross-entropy gradients and trace a gradient through a tiny attention head.

---

## 22. Day 20 — Normalization, Residual Paths, and Deep Signal Flow

### Core topics

1. Layer normalization as a differentiable composition.
2. RMS normalization and scale normalization.
3. Mean and variance derivatives.
4. Epsilon terms and stability.
5. Residual connections as direct gradient paths.
6. Activation scale and gradient scale.
7. Pre-normalization versus post-normalization as computational compositions.
8. Gradient flow through repeated Transformer blocks.
9. Why initialization and normalization interact.
10. Empirical gradient diagnostics across depth.

### Study tasks

- Derive the derivative through centering and variance normalization for a small vector.
- Derive the role of the epsilon term and explain why it changes the function being computed.
- Compare a residual block \(y=x+F(x)\) with a plain block \(y=F(x)\) by deriving their local derivatives.
- Explain why the identity contribution in a residual derivative can improve gradient propagation but is not a universal guarantee of stable training.

### Implementation

- Implement LayerNorm forward and backward for a tiny batch/feature shape.
- Implement RMSNorm forward and backward.
- Compare outputs and gradients against framework implementations.
- Test nearly constant inputs and extreme scales.

### Experiment — Gradient norms across depth

Construct a stack of tiny residual/non-residual blocks. Log activation and gradient norms by layer. Vary initialization, normalization location, and depth. State limitations: the result depends on architecture, data, nonlinearities, and initialization.

### Diagnostics

- Wrong axes for normalization statistics.
- Incorrectly reducing gradient contributions across features.
- Omitting the derivative through the mean or variance.
- Treating the epsilon term as irrelevant in low-variance cases.
- Inferring universal architectural superiority from a tiny experiment.

### Mastery gate

You can derive and check normalization gradients and explain how residual paths alter the local derivative composition.

---

## 23. Day 21 — Gradient-Based Optimization as Calculus

### Core topics

1. Objective functions and stationary points.
2. Gradient descent derived from local approximation.
3. Learning rate and step-size effects.
4. Convex versus nonconvex functions.
5. Local minima, maxima, and saddle points.
6. First- and second-order optimality conditions.
7. Momentum as a dynamical update rule.
8. Adaptive optimizers as gradient transformations.
9. Gradient clipping.
10. Line search and sufficient decrease concepts.
11. Stochastic gradients and gradient noise.
12. Objective decrease versus generalization.

### Mathematics

A gradient step is

\[
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t).
\]

Using a second-order local model,

\[
L(\theta+\Delta)\approx L(\theta)+\nabla L(\theta)^T\Delta+
\frac12\Delta^T H(\theta)\Delta.
\]

For a gradient step \(\Delta=-\eta g\), with \(g=\nabla L(\theta)\),

\[
L(\theta-\eta g)\approx L(\theta)-\eta\|g\|^2+
\frac{\eta^2}{2}g^THg.
\]

This local approximation explains why a very large step can increase the loss even when the gradient direction is locally descending.

### Derivation tasks

- Derive gradient descent from the first-order local model.
- Use the Hessian to analyze step-size behavior on a quadratic objective.
- Distinguish stationary points from minima.
- Compare deterministic and stochastic gradient estimates.
- Explain why lower training loss does not establish better generalization.

### Implementation

- Optimize a quadratic with known curvature.
- Optimize a nonconvex two-dimensional function and plot trajectories.
- Compare gradient descent, momentum, and an adaptive optimizer on controlled functions.
- Log loss, gradient norm, update norm, and parameter norm separately.

### Failure modes

- Assuming every stationary point is a minimum.
- Using one learning rate for all parameterizations without testing.
- Equating loss descent with generalization.
- Assuming adaptive optimization removes conditioning problems.

### Mastery gate

You can derive the gradient update, use curvature to reason about step sizes, and interpret optimization diagnostics without overclaiming what the loss curve proves.

---

# PART VI — ADVANCED OPTIMIZATION AND LEARNING DYNAMICS

## 24. Day 22 — Constrained Optimization, Lagrange Multipliers, and KKT

### Core topics

1. Equality-constrained optimization.
2. Lagrangian function.
3. Lagrange multipliers as sensitivity to constraints.
4. Geometric tangency conditions.
5. Inequality constraints.
6. Karush–Kuhn–Tucker (KKT) conditions.
7. Feasibility, stationarity, complementary slackness, and dual feasibility.
8. Constraint qualifications.
9. Convex problems versus general nonconvex problems.
10. Penalty and barrier methods as related approaches.

### Mathematics

For minimizing \(f(x)\) subject to \(g_i(x)=0\), the Lagrangian can be written

\[
\mathcal{L}(x,\lambda)=f(x)+\sum_i\lambda_i g_i(x).
\]

At a regular constrained optimum, first-order stationarity requires the objective gradient to be balanced by constraint-gradient contributions. For inequality constraints, KKT adds primal feasibility, dual feasibility, and complementary slackness, under suitable constraint qualifications.

### Study tasks

- Derive the constrained optimum for a simple two-variable problem geometrically and using the Lagrangian.
- Interpret a multiplier as a local rate of change of the optimal value with respect to a constraint bound, under suitable regularity assumptions.
- Work through KKT conditions for a one-dimensional inequality-constrained problem.
- Explain why KKT conditions are not automatically sufficient for global optimality in a general nonconvex problem.

### AI connection

Constraints arise in norm-constrained optimization, constrained decoding formulations, resource allocation, fairness objectives, and policy optimization. Do not assume that every practical regularizer is equivalent to a hard constraint.

### Mastery gate

You can construct a Lagrangian, derive stationarity conditions, and distinguish necessary from sufficient conditions.

---

## 25. Day 23 — Implicit Differentiation, Hypergradients, and Bilevel Learning

### Core topics

1. Implicitly defined functions.
2. Implicit Function Theorem intuition.
3. Differentiating a root or fixed-point condition.
4. Hyperparameters as differentiable inputs.
5. Bilevel optimization.
6. Hypergradients through an optimization process.
7. Exact versus unrolled differentiation.
8. Implicit differentiation of optimality conditions.
9. Numerical conditioning of implicit gradients.
10. Applications to meta-learning and hyperparameter optimization.

### Mathematics

Suppose \(z^*(\theta)\) satisfies a stationarity equation

\[
F(z^*(\theta),\theta)=0.
\]

When the relevant derivative with respect to \(z\) is nonsingular and the regularity assumptions hold, implicit differentiation gives

\[
\frac{dz^*}{d\theta}=-\left(\frac{\partial F}{\partial z}\right)^{-1}
\frac{\partial F}{\partial\theta},
\]

with orientation adjusted to the dimensions and derivative convention. The formula is a mathematical relation; computing it efficiently often requires solving a linear system rather than explicitly forming an inverse.

### Study tasks

- Derive the derivative of a simple implicitly defined root.
- Derive the stationarity equation for an inner optimization problem.
- Show how an outer objective depends on parameters through the solution of the inner problem.
- Compare unrolling a finite number of optimization steps with differentiating the idealized solution condition.
- Identify how poor conditioning can amplify hypergradient errors.

### Implementation

Build a small bilevel example in which an inner scalar optimization depends on a hyperparameter. Compare finite-difference hypergradients, unrolled differentiation, and implicit differentiation where assumptions are satisfied.

### Research connection

This provides mathematical foundations for differentiable optimization, meta-learning, learned optimizers, hyperparameter optimization, and certain self-improving training loops. It does **not** imply that an arbitrary self-modifying LLM can safely optimize itself; the mathematical objective, data dependence, compute budget, and evaluation procedure still determine what is being optimized.

### Mastery gate

You can derive an implicit gradient, state its assumptions, and explain when it is preferable to unrolling.

---

## 26. Day 24 — Gradient Flow, Differential Equations, and Stability

### Core topics

1. Ordinary differential equations (ODEs) as continuous dynamical systems.
2. Initial-value problems.
3. Existence and uniqueness intuition.
4. Gradient flow.
5. Discrete gradient descent as a numerical time-stepping method.
6. Euler and higher-order ODE solvers at a conceptual level.
7. Stability versus convergence.
8. Step-size effects in continuous/discrete dynamics.
9. Energy decrease along gradient flow.
10. Continuous-time interpretation limits for stochastic and adaptive optimizers.

### Mathematics

Gradient flow is

\[
\frac{d\theta(t)}{dt}=-\nabla L(\theta(t)).
\]

Along a sufficiently differentiable trajectory,

\[
\frac{d}{dt}L(\theta(t))
=\nabla L(\theta(t))^T\frac{d\theta}{dt}
=-\|\nabla L(\theta(t))\|^2\leq0.
\]

This result describes the idealized continuous gradient-flow system under the stated assumptions. A discrete optimizer may not inherit the same monotonic-decrease property at arbitrary step sizes.

### Study tasks

- Derive the decrease of the objective along gradient flow.
- Derive explicit Euler as a discretization of gradient flow.
- Analyze a one-dimensional quadratic ODE and its Euler discretization.
- Compare stable and unstable step sizes for the discrete system.
- Explain why a continuous-time result may not transfer directly to finite-step training.

### AI connection

Gradient flow provides a lens for studying optimization trajectories, implicit bias, continuous-time approximations, and neural differential equations. It is an analytical model, not a literal description of every training run.

### Mastery gate

You can connect an ODE to a learning update, derive the continuous loss-change identity, and explain how discretization changes stability.

---

## 27. Day 25 — Probability Gradients and Stochastic Computation

### Core topics

1. Differentiating expectations.
2. Score-function estimator.
3. Log-derivative trick.
4. Reparameterization trick.
5. Pathwise gradients.
6. Gradient variance and control variates.
7. Monte Carlo estimation error.
8. Continuous latent-variable objectives.
9. Connection to variational inference.
10. Distinguishing bias from variance and computational cost.

### Mathematics

For a parameterized distribution \(p_\theta(x)\), under suitable conditions,

\[
\nabla_\theta\mathbb{E}_{x\sim p_\theta}[f(x)]
=\mathbb{E}_{x\sim p_\theta}
\left[f(x)\nabla_\theta\log p_\theta(x)
+\nabla_\theta f(x)\right],
\]

where the explicit derivative term is included when \(f\) itself depends on \(\theta\). The exact assumptions and support behavior matter.

For a reparameterization \(x=g_\theta(\epsilon)\) with noise \(\epsilon\) from a parameter-independent base distribution, a pathwise estimator may be written

\[
\nabla_\theta\mathbb{E}_{\epsilon}[f(g_\theta(\epsilon))]
=\mathbb{E}_{\epsilon}[\nabla_x f(g_\theta(\epsilon))\,\nabla_\theta g_\theta(\epsilon)],
\]

under the relevant differentiability/interchange assumptions.

### Study tasks

- Derive the log-derivative identity.
- Derive a reparameterization gradient for a simple location-scale distribution.
- Compare the estimators on a controlled expectation where the exact answer is known.
- Measure empirical variance across repeated Monte Carlo runs.
- Explain why a lower-variance estimator may be more useful even if both estimators are unbiased.

### Case — Expected model objective

Model a differentiable stochastic computation, identify the source of randomness, and choose an estimator based on the structure of the sampling process. Explain when reparameterization is not straightforward, especially for discrete samples.

### Mastery gate

You can derive and compare score-function and pathwise gradients and understand the assumptions beneath each estimator.

---

## 28. Day 26 — Sensitivity, Lipschitzness, and Robustness

### Core topics

1. Local sensitivity and perturbation analysis.
2. Lipschitz continuity.
3. Gradient bounds and Lipschitz functions on domains.
4. Smoothness through gradient Lipschitzness.
5. Sensitivity propagation through compositions.
6. Input-gradient interpretation and its limitations.
7. Adversarial perturbations as a local optimization problem.
8. Gradient saturation and misleading gradient signals.
9. Finite perturbations versus infinitesimal derivatives.
10. Robustness claims and valid experimental evidence.

### Mathematics

A function is Lipschitz on a domain when there exists \(K\) such that

\[
|f(x)-f(y)|\leq K\|x-y\|
\]

for every pair in the domain. A bounded derivative implies a Lipschitz bound on an interval; in multivariable settings, a suitable bound on the derivative over a convex domain gives an analogous result.

Gradient Lipschitzness is a common smoothness assumption:

\[
\|\nabla f(x)-\nabla f(y)\|\leq L\|x-y\|.
\]

Do not confuse a global bound, a local derivative measurement, and an empirical estimate on sampled points.

### Study tasks

- Prove a Lipschitz bound from a bounded derivative in one dimension.
- Estimate local input sensitivity by directional derivatives and finite perturbations.
- Construct a function with a small local gradient but a large change farther away.
- Analyze why gradient-based adversarial methods can be useful but do not alone define a complete threat model.

### Experiment

Vary perturbation magnitude and direction. Compare first-order predictions with actual function changes. Report both local derivative estimates and finite-distance behavior.

### Mastery gate

You can distinguish local gradients, local Lipschitz estimates, global guarantees, and empirical robustness results.

---

# PART VII — ADVANCED AND RESEARCH-ORIENTED CALCULUS

## 29. Day 27 — Calculus of Variations and Functional Derivatives

### Core topics

1. Functionals: functions whose inputs are functions.
2. Variations and perturbation functions.
3. First variation.
4. Euler–Lagrange equation.
5. Boundary conditions.
6. Integral objectives over trajectories or fields.
7. Regularization as a functional penalty.
8. Connections to optimal control and continuous learning systems.

### Mathematics

For a functional

\[
J[y]=\int_a^b F(x,y(x),y'(x))\,dx,
\]

a stationary function under suitable boundary and smoothness conditions satisfies the Euler–Lagrange equation

\[
\frac{\partial F}{\partial y}-\frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right)=0.
\]

This is the function-space counterpart of setting a derivative to zero, but the object being varied is a function or trajectory rather than a finite vector of parameters.

### Study tasks

- Derive the first variation for a simple functional.
- Derive the Euler–Lagrange equation using a perturbation \(y+\varepsilon\eta\).
- Solve a simple variational problem with fixed endpoints.
- Explain the relationship between boundary conditions and admissible perturbations.

### AI connection

Calculus of variations is useful background for optimal control, trajectory optimization, continuous models, and some theoretical perspectives on regularization. It is an advanced extension rather than a prerequisite for implementing a conventional Transformer.

### Mastery gate

You can derive an Euler–Lagrange equation for a basic functional and distinguish a functional derivative from an ordinary finite-dimensional gradient.

---

## 30. Day 28 — Differential Equations, Adjoint Sensitivity, and Neural ODEs

### Core topics

1. ODE solution maps and parameter dependence.
2. Sensitivity equations.
3. Differentiating through numerical solver steps.
4. Continuous adjoint sensitivity at a conceptual and derivational level.
5. Solver tolerance and gradient accuracy.
6. Stiffness and numerical stability concepts.
7. Direct differentiation through a discretized trajectory versus continuous adjoint formulations.
8. Memory/runtime trade-offs.
9. Failure modes in solver-based learning.

### Mathematics

Consider a parameterized ODE

\[
\frac{dh(t)}{dt}=f(h(t),t,\theta),\qquad h(0)=h_0(\theta).
\]

The final output depends on \(\theta\) through the entire trajectory, not only through the final evaluation of \(f\). Sensitivity can be studied by differentiating the ODE, or by an adjoint equation for an appropriately defined objective.

### Study tasks

- Solve a simple ODE analytically and numerically.
- Derive the sensitivity equation for a scalar parameterized ODE.
- Compare finite-difference sensitivity with gradients obtained by differentiating the solver computation.
- Explain how solver tolerance affects both the forward trajectory and the resulting gradient estimate.

### Experiment

Vary solver method and tolerance on a known ODE. Measure forward solution error and gradient error separately. Demonstrate that a visually accurate trajectory does not automatically imply an equally accurate derivative.

### Mastery gate

You can explain the dependency of a learned ODE solution on parameters and identify the trade-offs between discretized and continuous sensitivity approaches.

---

## 31. Day 29 — Higher-Order Automatic Differentiation and Meta-Learning

### Core topics

1. Differentiating gradients.
2. Hessian-vector products.
3. Higher-order computational graphs.
4. Forward-over-reverse and reverse-over-forward concepts.
5. Hypergradients through optimizer updates.
6. Differentiable inner loops.
7. Truncated unrolling and approximation bias.
8. Memory cost of higher-order gradients.
9. Numerical verification of second-order results.
10. Limits of differentiating through discrete choices or nondifferentiable operations.

### Study tasks

- Derive the derivative of one gradient-descent update with respect to its initialization and learning rate.
- Compute an HVP for a small function.
- Compare explicitly forming a Hessian with calculating Hessian-vector products.
- Implement a two-step differentiable inner optimization loop.
- Check a hypergradient using finite differences of the complete outer objective.

### Research bridge — Self-improving systems

Many adaptive or meta-learning methods can be expressed as optimization over an inner learning process. Calculus tells you how an outer objective changes when inner parameters or hyperparameters change. It does **not** establish that the chosen outer objective is aligned with genuine improvement, that data are trustworthy, or that iterative self-modification is safe. Those are separate evaluation, systems, and safety problems.

### Mastery gate

You can compute or approximate a hypergradient for a small inner loop and explain its assumptions, cost, and potential sources of bias.

---

## 32. Day 30 — Research Integration: Build and Analyze a Tiny Transformer

### Objective

Create a tiny, inspectable sequence model and demonstrate that you understand the calculus of its training process. This is a mathematical verification project, not a scale benchmark.

### Minimum components

1. Token-like integer inputs and an embedding lookup.
2. One small attention head.
3. Stable softmax and attention mask.
4. A small feed-forward block with an activation.
5. Optional normalization and residual connections.
6. Output logits and a cross-entropy objective.
7. Forward pass, backward pass, and parameter update.
8. Gradient checker for selected parameters.
9. Training and evaluation curves.
10. A report on derivative behavior and numerical limitations.

### Required derivations

- Cross-entropy gradient with respect to logits.
- Gradient through the output projection.
- Gradient through the activation.
- Gradient through one attention head: output multiplication, softmax, and score computation.
- Gradient through residual addition and normalization if included.
- Gradient of the final objective with respect to one early parameter.

### Required verification

- Compare manual gradients against a trusted autodiff implementation.
- Run finite-difference checks on a tiny model with deterministic inputs.
- Test extreme logits and verify numerical stability.
- Confirm padding/causal masks behave as intended.
- Check that a parameter update has the predicted first-order effect for a sufficiently small step in a controlled test.

### Required experiment

Select one question and test it systematically, for example:

- How does logit scale affect attention entropy and gradient magnitude?
- How does normalization affect gradients across depth in this small model?
- How does the finite-difference step size affect the gradient check?
- How does sequence length affect attention compute and intermediate memory?

### Required report

```text
# Experiment: [Title]

## Question
What specific claim is being investigated?

## Mathematical prediction
What does the derivation suggest, and under which assumptions?

## Setup
Architecture, dimensions, dtype, seeds, optimizer, and environment.

## Method
Exact steps to reproduce the experiment.

## Measurements
Loss, gradient norms, absolute/relative gradient errors, runtime, and memory where relevant.

## Results
Plots/tables with explicit axes and units.

## Interpretation
Which predictions were supported, contradicted, or left unresolved?

## Limitations
What does this experiment not establish?

## Reproduction command
A command that reruns the experiment.
```

### Final mastery gate

You can trace, derive, implement, and verify the main gradients in a tiny attention-based model. You can explain where its gradients can become inaccurate or unstable and make claims proportional to the evidence.

---

# PART VIII — IMPLEMENTATION LABS FROM SCRATCH

## 33. Required Algorithm and Derivation Inventory

Complete these in increasing order of dependency. “From scratch” means implementing the pedagogical mechanism yourself; using libraries as independent references is encouraged.

### A. Single-variable calculus

- [ ] Numerical limit estimator with a clear warning that it is not a proof.
- [ ] Forward finite difference.
- [ ] Backward finite difference.
- [ ] Central finite difference.
- [ ] Taylor polynomial evaluator.
- [ ] Taylor approximation error plots.
- [ ] Left/right derivative checker.
- [ ] Riemann sum integration.
- [ ] Trapezoidal quadrature.
- [ ] Simpson's rule for suitable smooth functions.

### B. Multivariable calculus

- [ ] Partial derivative calculator for toy functions.
- [ ] Directional finite difference.
- [ ] Analytic gradient for a scalar function.
- [ ] Explicit Jacobian for a small vector-valued function.
- [ ] JVP from an explicit Jacobian.
- [ ] VJP from an explicit Jacobian.
- [ ] Hessian for a small scalar objective.
- [ ] Finite-difference Hessian checker.
- [ ] Hessian-vector product.

### C. Automatic differentiation

- [ ] Scalar reverse-mode autodiff engine.
- [ ] Graph topological sort.
- [ ] Correct accumulation for shared graph nodes.
- [ ] Backward rules for arithmetic and common activations.
- [ ] Gradient checker for the autodiff engine.
- [ ] Two-layer MLP forward pass.
- [ ] Two-layer MLP manual backward pass.
- [ ] Mini-batch loss with correct reduction semantics.

### D. Neural-network and Transformer calculus

- [ ] MSE gradient.
- [ ] Logistic loss gradient.
- [ ] Stable log-sum-exp.
- [ ] Stable softmax.
- [ ] Softmax Jacobian-vector product.
- [ ] Softmax-cross-entropy gradient.
- [ ] Attention forward pass.
- [ ] Attention backward pass for a tiny tensor shape.
- [ ] LayerNorm forward/backward or a validated analytical derivative.
- [ ] Tiny Transformer gradient-checking experiment.

### E. Advanced research calculus

- [ ] Finite-difference hypergradient.
- [ ] Unrolled inner optimization gradient.
- [ ] Implicit derivative for a small equation.
- [ ] Score-function gradient experiment.
- [ ] Reparameterization gradient experiment.
- [ ] Sensitivity equation for a scalar ODE.
- [ ] Euler–Lagrange derivation for a basic functional.

## 34. Required tests

Each implementation should test:

- A regular random input.
- A small-magnitude input.
- A large-magnitude input, if mathematically in-domain.
- A boundary or nearly degenerate case.
- A case where the derivative is zero or near zero.
- A case where the derivative is large.
- An independent reference calculation.

Use tolerances that account for dtype, derivative magnitude, conditioning, and the expected error of the numerical method. Do not set an arbitrary tolerance simply to make tests pass.

---

# PART IX — EXPERIMENTS AND DIAGNOSTICS

## 35. Experiment catalog

| ID | Experiment | Controlled variable | Measure | Core lesson |
|---|---|---|---|---|
| C01 | Finite-difference step sweep | Step size \(h\) | Absolute/relative derivative error | Truncation versus rounding error |
| C02 | Taylor approximation range | Distance from expansion point | Remainder error | Local approximation has a useful region |
| C03 | Partial derivative counterexample | Direction/path | Function behavior | Partial derivatives do not guarantee total differentiability |
| C04 | Coordinate rescaling | Parameter scale | Trajectory/convergence | Gradient behavior depends on parameterization |
| C05 | Derivative through depth | Number of composed functions | Product magnitude | Local sensitivities accumulate through composition |
| C06 | Curvature versus learning rate | Step size and Hessian scale | Loss change | Finite updates depend on second-order behavior |
| C07 | Numerical quadrature | Interval count | Integral error | Smoothness affects integration convergence |
| C08 | Improper integral truncation | Domain cutoff | Approximation error | Tail behavior matters |
| C09 | AD versus finite differences | Dtype and step size | Gradient discrepancy | Every derivative method has assumptions and numerical limits |
| C10 | Sigmoid/tanh saturation | Input magnitude | Derivative magnitude | Activations can have small local derivatives |
| C11 | Softmax logit scale | Logit temperature/scale | Entropy and gradient norm | Score scale influences distribution sharpness |
| C12 | Attention mask behavior | Mask pattern | Probabilities and gradients | Invalid positions must be excluded correctly |
| C13 | Normalization gradients | Feature variance | Gradient error | Epsilon and reduction axes matter |
| C14 | Residual versus plain stack | Depth and initialization | Layerwise gradient norm | Gradient flow depends on path structure |
| C15 | Sum versus mean loss | Batch size | Gradient scale | Reduction semantics change gradient magnitude |
| C16 | Hypergradient method comparison | Inner steps | Hypergradient error/cost | Unrolling and implicit differentiation trade off cost and approximation |
| C17 | Score-function variance | Sample count/baseline | Empirical variance | Unbiased estimates can still be noisy |
| C18 | Solver tolerance | ODE tolerance | Trajectory and gradient error | Forward accuracy and derivative accuracy are distinct |

## 36. Standard experiment protocol

Every experiment must include:

1. **Question:** One narrow, falsifiable question.
2. **Hypothesis:** A prediction derived from a formula or theorem.
3. **Assumptions:** Smoothness, domain, dtype, boundedness, or independence assumptions as applicable.
4. **Independent variable:** The factor being changed.
5. **Controlled variables:** Factors held fixed.
6. **Measurements:** Metrics that reflect the mathematical claim.
7. **Baseline:** An independent reference or analytically known result.
8. **Repetitions:** More than one seed when randomness affects the claim.
9. **Limitations:** What the result cannot prove.
10. **Reproduction:** Command, versions, seed, and configuration.

## 37. Diagnostic checklist

When a derivative or optimization experiment behaves unexpectedly, inspect:

- Is the function defined at the input and differentiable there?
- Is the derivative formula valid on this domain?
- Are all intermediate dependencies included in the chain rule?
- Are batch and reduction axes correct?
- Does broadcasting require gradient reduction in the backward pass?
- Are extreme exponentials or logarithms causing numerical issues?
- Is the finite-difference step appropriate for the input/output scale?
- Is the computation deterministic during gradient checking?
- Is the loss summed or averaged?
- Is the gradient clipped, scaled, detached, or otherwise transformed?
- Does a solver tolerance affect the forward result, backward result, or both?
- Is the observed effect local, global, or merely empirical on the tested inputs?

---

# PART X — MASTER DEPENDENCY MAP

## 38. From Continuity to LLM Training

```text
Functions and domains
    ↓
Limits and continuity
    ↓
Derivative from first principles
    ↓
Differentiation rules and chain rule
    ↓
Mean Value Theorem + Taylor approximation + error
    ↓
Integration + Fundamental Theorem of Calculus
    ↓
Multivariable limits and total differentiability
    ↓
Partial derivatives + directional derivatives + gradients
    ↓
Multivariable chain rule
    ↓
Jacobian as local derivative
    ├── JVP / VJP
    └── forward-mode / reverse-mode AD
    ↓
Higher-order derivatives + Hessian + curvature
    ↓
Numerical differentiation + gradient checking
    ↓
Computational graphs + reverse-mode autodiff
    ↓
Backpropagation through neural networks
    ↓
Loss functions + activation derivatives
    ↓
Softmax + cross-entropy + attention derivatives
    ↓
Normalization + residual gradient paths
    ↓
Gradient descent + curvature + stochastic gradients
    ↓
Constrained optimization + implicit differentiation
    ↓
Gradient flow + probability gradients + sensitivity
    ↓
Functional derivatives + Neural ODEs + higher-order AD
    ↓
Tiny Transformer: derivation → implementation → verification
```

## 39. Calculus concept-to-LLM mapping

| Calculus concept | Where it appears | Main question |
|---|---|---|
| Chain rule | Every composed neural-network operation | Which intermediate dependency carries the gradient? |
| Local linear approximation | Small parameter updates and sensitivity | How accurate is the first-order prediction at this step size? |
| Taylor expansion | Optimization analysis | How do curvature and higher-order terms alter a finite update? |
| Gradient | Training objective and input sensitivity | Which local direction increases/decreases the objective? |
| Jacobian products | Vector-valued layers and attention | Can the required directional derivative be computed without a full Jacobian? |
| Hessian/HVP | Curvature-aware optimization and analysis | How does the gradient change with a perturbation? |
| Finite differences | Gradient validation | Is an analytic/autodiff gradient consistent with independent estimates? |
| Reverse-mode AD | Training models with many parameters and scalar loss | How do output sensitivities accumulate to parameter gradients? |
| Softmax derivative | Attention and token probabilities | How do changes in logits affect normalized probabilities? |
| Cross-entropy derivative | Next-token prediction | How does the prediction error change with logits? |
| Normalization derivatives | LayerNorm/RMSNorm | How do mean, variance, and scale operations affect the gradient? |
| Implicit differentiation | Hyperparameters, meta-learning, differentiable solvers | How does an optimal solution change as the problem changes? |
| Gradient flow | Continuous-time optimization theory | What behavior belongs to the idealized continuous system? |
| Differentiation under an integral | Expected objectives and probabilistic learning | When can a derivative move inside an expectation/integral? |
| Reparameterization | Stochastic latent-variable models | How can gradients pass through a sampling operation? |
| Functional derivatives | Continuous trajectories and optimal control | What does changing an entire function do to the objective? |
| ODE sensitivity | Neural ODEs and continuous-depth models | How does parameter variation alter a full trajectory? |

---

# PART XI — MASTERY GATES

## 40. Gate A — Differential foundations

Pass only when you can:

- State and use limit, continuity, and derivative definitions.
- Derive common derivative rules rather than only recall formulas.
- Explain the relation between the derivative and local linear approximation.
- Apply the Mean Value Theorem and Taylor's theorem with assumptions.
- Estimate approximation error in controlled examples.

## 41. Gate B — Multivariable derivatives

Pass only when you can:

- Distinguish partial, directional, and total derivatives.
- Derive gradients and tangent-plane approximations.
- Apply the multivariable chain rule to a computational graph.
- Define a Jacobian with consistent dimensions.
- Compute JVPs, VJPs, gradients, and Hessian-vector products.
- Explain the limits of local derivative information.

## 42. Gate C — Integration and probability

Pass only when you can:

- Explain FTC and implement elementary quadrature.
- Analyze common improper integrals.
- Use multiple integration to describe joint and marginal distributions.
- Apply change of variables with the appropriate determinant factor.
- Explain assumptions behind differentiation under an integral sign.

## 43. Gate D — Automatic differentiation

Pass only when you can:

- Implement finite differences and interpret their error.
- Build a tiny reverse-mode autodiff engine.
- Handle graph branching and shared nodes correctly.
- Implement backpropagation for a small MLP without autograd.
- Verify results against an independent reference.

## 44. Gate E — Transformer calculus

Pass only when you can:

- Derive MSE, logistic loss, and softmax-cross-entropy gradients.
- Explain activation saturation and gradient propagation through depth.
- Derive the backward path through a tiny attention head.
- Implement stable softmax and correctly handle attention masks.
- Validate normalization gradients if normalization is included.
- Make numerical claims based on diagnostics instead of loss alone.

## 45. Gate F — Advanced research calculus

Pass only when you can:

- Derive KKT conditions for a small constrained problem.
- Derive an implicit gradient and state its assumptions.
- Explain gradient flow and how its discrete approximation can behave differently.
- Compare score-function and pathwise gradient estimators.
- Derive a simple sensitivity equation for a parameterized ODE.
- Explain at least one calculus-of-variations problem.

---

# PART XII — PROJECT STRUCTURE AND REPRODUCIBILITY

## 46. Suggested repository layout

```text
00.2-calculus/
├── README.md
├── notes/
│   ├── 01-limits-continuity.md
│   ├── 02-derivatives.md
│   ├── 03-taylor-error.md
│   ├── 04-integration.md
│   ├── 05-multivariable-calculus.md
│   ├── 06-chain-rule-jacobians.md
│   ├── 07-higher-order-derivatives.md
│   ├── 08-automatic-differentiation.md
│   ├── 09-backpropagation.md
│   ├── 10-transformer-gradients.md
│   ├── 11-optimization-dynamics.md
│   └── 12-advanced-calculus.md
├── scratch/
│   ├── finite_differences.py
│   ├── quadrature.py
│   ├── multivariable_derivatives.py
│   ├── autodiff_scalar.py
│   ├── mlp_backprop.py
│   ├── softmax_cross_entropy.py
│   ├── attention_backward.py
│   └── tiny_transformer.py
├── experiments/
│   ├── configs/
│   ├── results/
│   ├── plots/
│   └── reports/
└── tests/
    ├── test_finite_differences.py
    ├── test_autodiff.py
    ├── test_gradients.py
    └── test_attention.py
```

## 47. Rules for trustworthy implementation

- Keep the mathematical derivation next to the implementation.
- Specify shapes and reduction semantics at every stage of a gradient calculation.
- Separate educational code from production-optimized code.
- Use framework autograd as a reference, not as a substitute for understanding.
- Never claim exactness for a finite-difference estimate.
- Record dtype, random seed, framework version, hardware, and configuration for performance results.
- Do not benchmark incorrect code.
- Include tests for extreme but valid inputs.
- Make nondifferentiable points and subgradient conventions explicit.
- Report forward numerical error and gradient error separately.
- Retain failed experiments and state what they taught you.
- Avoid unsupported claims about convergence, stability, or robustness.

## 48. Standard note template

```markdown
# Topic: [Name]

## 1. Problem and motivation
What is the mathematical problem, and why does it matter?

## 2. Definitions and assumptions
State domains, smoothness, dimensions, and assumptions.

## 3. Derivation
Show the derivation step by step.

## 4. Geometric/physical interpretation
Explain what changes, accumulates, or is approximated.

## 5. Algorithmic interpretation
Describe how the result is computed.

## 6. Implementation from scratch
Write a transparent implementation.

## 7. Independent verification
Compare with an analytic or trusted reference.

## 8. Stress tests
Vary step size, scale, dtype, depth, or conditioning.

## 9. Connection to deep learning/LLMs
Identify the exact operation and dependency path.

## 10. Failure modes
State invalid assumptions and numerical pitfalls.

## 11. Mastery questions
Write questions that require derivation, not memorization.

## 12. Evidence
Link tests, plots, configuration, and experiment report.
```

## 49. Standard experiment template

```markdown
# Experiment: [Title]

## Question
What specific claim is tested?

## Mathematical prediction
What should happen and under which assumptions?

## Setup
Function/model, input scale, dtype, seed, libraries, and hardware.

## Independent variable
What is changed?

## Controlled variables
What is held fixed?

## Method
Exact reproducible steps.

## Metrics
Absolute error, relative error, gradient norm, loss, runtime, memory, etc.

## Results
Tables and plots with labeled axes and units.

## Interpretation
Which predictions are supported or contradicted?

## Limitations
What does the experiment not establish?

## Reproduction command
Exact command and configuration.
```

---

# PART XIII — FINAL DEFINITION OF DONE

## 50. What mastery means

This module is complete when you can demonstrate the following, with derivations and tests rather than self-assessment alone.

### Mathematical mastery

- You understand derivatives as limits and local linear approximations.
- You distinguish continuity, differentiability, smoothness, and analyticity.
- You can derive and apply multivariable chain rules.
- You can reason about gradients, Jacobians, higher derivatives, and curvature.
- You understand integration as accumulation and know when differentiation/integration may be interchanged.
- You can state the assumptions behind optimization and sensitivity results.

### Computational mastery

- You can implement numerical differentiation and integration.
- You can build a small reverse-mode automatic differentiation system.
- You can implement and verify manual backpropagation.
- You can compute and check gradients through softmax, cross-entropy, attention, and normalization on tiny examples.
- You can distinguish a mathematical error from a finite-precision issue or a faulty test.

### Advanced mastery

- You understand KKT and implicit differentiation at a derivation level.
- You can explain gradient flow, ODE sensitivity, and differentiable optimization loops.
- You know why score-function and pathwise gradients differ.
- You understand what functional derivatives add beyond ordinary parameter gradients.
- You can read an advanced paper and identify its objective, differentiability assumptions, gradient estimator, approximation, numerical risks, and empirical evidence.

### Research discipline

- Every experiment has a falsifiable question and a reproducible setup.
- Every claim distinguishes a theorem, a numerical result, and an interpretation.
- Gradient norms, loss values, forward error, and derivative error are treated as distinct measurements.
- Failed experiments and limitations are recorded.
- No conclusion is stronger than the evidence.

---

# FINAL INTEGRATION — CALCULUS AS A SYSTEM

```text
LIMITS
  ↓
CONTINUITY
  ↓
DERIVATIVE FROM FIRST PRINCIPLES
  ↓
LOCAL LINEAR APPROXIMATION
  ↓
CHAIN RULE
  ↓
TAYLOR EXPANSION + ERROR
  ↓
INTEGRATION + FUNDAMENTAL THEOREM
  ↓
PARTIAL / DIRECTIONAL / TOTAL DERIVATIVES
  ↓
GRADIENT + JACOBIAN
  ↓
JVP / VJP + HIGHER-ORDER DERIVATIVES
  ↓
NUMERICAL DIFFERENTIATION + GRADIENT CHECKING
  ↓
AUTOMATIC DIFFERENTIATION
  ↓
BACKPROPAGATION
  ↓
LOSS / ACTIVATION / SOFTMAX DERIVATIVES
  ↓
ATTENTION + NORMALIZATION GRADIENTS
  ↓
OPTIMIZATION + CURVATURE + STOCHASTIC GRADIENTS
  ↓
CONSTRAINED / IMPLICIT / META-OPTIMIZATION
  ↓
GRADIENT FLOW + PROBABILITY GRADIENTS
  ↓
FUNCTIONAL DERIVATIVES + ODE SENSITIVITY
  ↓
TINY TRANSFORMER: DERIVE → IMPLEMENT → VERIFY → EXPERIMENT
```

The central discipline is:

**Define the function → state the assumptions → derive the derivative → understand the local approximation → implement the computation → verify the gradient → stress-test numerical behavior → connect the result to model learning.**

Do not conclude that you understand a neural-network operation because a framework computes its gradient. Derive what changes, trace every dependency, verify the implementation independently, and identify the conditions under which the result can be trusted.

**End of `00.2 — Advanced Calculus for Deep Learning, Transformers & LLM Research`.**
