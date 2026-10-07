# CLASSICAL MACHINE LEARNING — MASTER DEEP ROADMAP BEFORE DEEP LEARNING

> Repository-style, granular key-point roadmap.
> Concept → Intuition → Mathematics → Statistics → Algorithm Mechanism → Optimization → Assumptions → Hyperparameters → Implementation → Diagnostics → Failure Modes → Evaluation → Interpretation → Comparison → Experiments → Real-world application.

---

# STAGE 00 — MATHEMATICAL FOUNDATIONS



## 00.2 Calculus

### PLAN

1. Functions
2. Limits
3. Continuity
4. Derivative
5. Partial derivative
6. Gradient
7. Directional derivative
8. Jacobian
9. Hessian
10. Chain rule
11. Product rule
12. Quotient rule
13. Exponential derivatives
14. Logarithmic derivatives
15. Taylor approximation
16. Stationary points
17. Local minima
18. Global minima
19. Saddle points
20. Convexity
21. Strict convexity
22. Strong convexity
23. Smoothness
24. Lipschitz continuity
25. Matrix calculus
26. Differentiable optimization
27. Non-differentiable optimization
28. Subgradients

### SUB MATERIAL

* Geometric derivative
* Gradient direction
* Gradient magnitude
* Level sets
* Curvature
* Hessian spectrum
* Taylor expansion
* Convex sets
* Convex functions
* Saddle geometry

### MATHEMATICS

* `f'(x)`
* Partial derivatives
* Gradient
* Directional derivative
* Jacobian
* Hessian
* First-order Taylor
* Second-order Taylor
* Chain rule
* Quadratic-form gradient
* Logistic derivative
* Softmax derivative
* Cross-entropy gradient
* Hinge-loss subgradient

### OPTIMIZATION

* First-order conditions
* Second-order conditions
* Gradient descent
* Newton's method
* Quasi-Newton
* Coordinate descent
* Subgradient descent
* Proximal optimization

### ASSUMPTIONS

* Differentiability
* Twice differentiability
* Convexity
* Smoothness
* Lipschitz gradients
* Constraint feasibility

### DIAGNOSTICS

* Gradient magnitude
* Hessian eigenvalues
* Curvature
* Loss trajectory
* Gradient variance
* Parameter change

### FAILURE MODES

* Vanishing gradients
* Exploding gradients
* Oscillation
* Divergence
* Saddle-point stagnation
* Poor scaling
* Ill-conditioned curvature

### COMPARISON

* Derivative vs partial derivative
* Gradient vs Jacobian
* Gradient vs Hessian
* First-order vs second-order
* Smooth vs non-smooth
* Convex vs non-convex

### FROM SCRATCH

* Numerical derivative
* Gradient checker
* Gradient descent
* Newton's method
* Coordinate descent
* Subgradient optimizer
* Proximal optimizer

### CASE

1. Linear regression optimization
2. Logistic regression optimization
3. Ridge optimization
4. SVM optimization
5. Chain-rule readiness for backpropagation

---

## 00.3 Probability

### PLAN

1. Sample space
2. Events
3. Probability axioms
4. Union
5. Intersection
6. Complement
7. Conditional probability
8. Independence
9. Conditional independence
10. Random variables
11. Discrete variables
12. Continuous variables
13. PMF
14. PDF
15. CDF
16. Quantiles
17. Joint distribution
18. Marginal distribution
19. Conditional distribution
20. Expectation
21. Variance
22. Covariance
23. Correlation
24. Moments
25. Bayes theorem
26. Law of total probability
27. Likelihood
28. Maximum likelihood
29. MAP
30. Gaussian
31. Bernoulli
32. Binomial
33. Multinomial
34. Poisson
35. Exponential
36. Uniform
37. Beta
38. Gamma
39. Student-t
40. Mixture distributions
41. Entropy
42. Cross-entropy
43. KL divergence
44. Jensen-Shannon divergence
45. Mutual information
46. Law of Large Numbers
47. Central Limit Theorem
48. Exponential family

### STATISTICS

* Prior
* Likelihood
* Posterior
* Posterior predictive
* Conjugate prior
* Bayesian updating
* Sampling distribution
* Calibration
* Coverage

### ALGORITHM MECHANISM

* Bayes inference
* MLE
* MAP
* Naive Bayes
* Gaussian likelihood
* Bernoulli likelihood
* Multinomial likelihood
* Mixture likelihood

### CASE

1. Gaussian MLE
2. Naive Bayes posterior
3. Logistic likelihood
4. Cross-entropy as negative log-likelihood
5. GMM likelihood

---

## 00.4 Statistics

### PLAN

1. Population
2. Sample
3. Sampling
4. Estimator
5. Point estimation
6. Interval estimation
7. Bias
8. Variance
9. Standard error
10. Consistency
11. Efficiency
12. Sufficiency
13. Confidence interval
14. Hypothesis testing
15. Null hypothesis
16. Alternative hypothesis
17. p-value
18. Statistical significance
19. Effect size
20. Power
21. Correlation
22. Covariance
23. Multicollinearity
24. Residual analysis
25. Sampling bias
26. Selection bias
27. Measurement bias
28. Confounding
29. Missingness mechanisms
30. Distribution shift
31. Statistical dependence
32. Estimator uncertainty

### SUB MATERIAL

* Frequentist inference
* Bayesian inference
* Fisher information
* Cramer-Rao bound
* Likelihood ratio
* Wald test
* Score test
* Bootstrap
* Jackknife
* Permutation test
* Multiple testing
* Bonferroni
* FDR

### CASE

1. Confidence interval
2. Feature significance
3. Bootstrap model metric
4. Permutation test
5. Power analysis

---

## 00.5 Optimization

### PLAN

1. Objective function
2. Loss function
3. Cost function
4. Empirical risk
5. Expected risk
6. Optimization landscape
7. Gradient descent
8. Batch GD
9. SGD
10. Mini-batch SGD
11. Learning rate
12. Convergence
13. Convex optimization
14. Non-convex optimization
15. Closed-form optimization
16. Coordinate descent
17. Proximal gradient
18. Projected gradient
19. Newton
20. BFGS
21. L-BFGS
22. Momentum
23. Nesterov
24. Lagrangian
25. KKT conditions
26. Line search
27. Trust region
28. Preconditioning

### HYPERPARAMETERS

* Learning rate
* Batch size
* Momentum
* Tolerance
* Maximum iterations
* Weight decay
* Gradient clipping
* Early stopping

### DIAGNOSTICS

* Objective curve
* Gradient norm
* Parameter norm
* Convergence speed
* Gradient variance
* Hessian spectrum
* KKT residual
* Duality gap

### FAILURE MODES

* Divergence
* Slow convergence
* Oscillation
* Saddle point
* Bad initialization
* Numerical instability
* Ill-conditioned objective

### COMPARISON

* Closed-form vs iterative
* GD vs SGD
* First-order vs second-order
* Momentum vs Nesterov
* BFGS vs L-BFGS
* Line search vs trust region

### CASE

1. OLS vs gradient descent
2. Logistic regression
3. Lasso coordinate descent
4. SVM constrained optimization

---

# STAGE 01 — DATA FOUNDATIONS

## 01.1 Dataset Structure

### PLAN

1. Observation
2. Entity
3. Feature
4. Predictor
5. Target
6. Label
7. Feature matrix `X`
8. Target vector `y`
9. Numerical feature
10. Categorical feature
11. Binary feature
12. Ordinal feature
13. Continuous feature
14. Discrete feature
15. Identifier
16. Timestamp
17. Group identifier
18. Entity key
19. Primary key
20. Foreign key
21. Static feature
22. Dynamic feature
23. Historical feature

### SUB MATERIAL

* Tabular data
* Time series
* Cross-sectional data
* Panel data
* Grouped observations
* Hierarchical data
* Sparse data
* Wide data
* Long data

---

## 01.2 Data Quality

### PLAN

1. Missing values
2. Duplicate rows
3. Duplicate entities
4. Invalid values
5. Impossible values
6. Inconsistent categories
7. Wrong types
8. Unit inconsistency
9. Noise
10. Measurement error
11. Corruption
12. Sparse observations
13. Schema drift
14. Distribution anomalies
15. Timestamp inconsistencies

### DIAGNOSTICS

* Schema validation
* Range validation
* Cardinality
* Missingness
* Duplicate rate
* Distribution checks
* Unit checks
* Data-type checks

---

## 01.3 Data Leakage

### PLAN

1. Target leakage
2. Label leakage
3. Feature leakage
4. Train-test contamination
5. Preprocessing leakage
6. Aggregation leakage
7. Temporal leakage
8. Group leakage
9. Future-information leakage
10. Cross-validation leakage
11. Post-hoc leakage
12. Human-process leakage

### CASE

1. Scaling before split
2. Target mean encoding before split
3. Future transaction aggregate
4. Patient leakage across folds
5. Duplicate entity across train/test

---

# STAGE 02 — EXPLORATORY DATA ANALYSIS

## 02.1 Univariate Analysis

### SUB MATERIAL

* Mean
* Median
* Mode
* Variance
* Standard deviation
* Quantiles
* Percentiles
* IQR
* Skewness
* Kurtosis
* Frequency
* Cardinality
* Histogram
* Box plot
* Density plot
* ECDF

### CASE

1. Distribution diagnosis
2. Outlier diagnosis
3. Skew diagnosis
4. Rare-category diagnosis

---

## 02.2 Bivariate Analysis

### SUB MATERIAL

* Feature-feature relationship
* Feature-target relationship
* Correlation
* Covariance
* Conditional distribution
* Group comparison
* Scatter plot
* Cross-tabulation
* Group statistics
* Pearson correlation
* Spearman correlation

---

## 02.3 Multivariate Analysis

### SUB MATERIAL

* Correlation matrix
* Covariance matrix
* Pair plot
* Multicollinearity
* Redundancy
* Interaction
* Confounding
* VIF
* PCA inspection

---

## 02.4 Target Analysis

### SUB MATERIAL

* Target distribution
* Target skew
* Class balance
* Rare classes
* Target outliers
* Target leakage
* Regression target
* Binary target
* Multiclass target

---

# STAGE 03 — DATA PREPROCESSING

## 03.1 Missing Data

### PLAN

1. Detect missingness
2. Missingness patterns
3. MCAR
4. MAR
5. MNAR
6. Drop rows
7. Drop columns
8. Mean imputation
9. Median imputation
10. Mode imputation
11. Constant imputation
12. Group-based imputation
13. KNN imputation
14. Iterative imputation
15. Model-based imputation
16. Missing indicator
17. Forward fill
18. Backward fill
19. Interpolation
20. Multiple imputation

### FAILURE MODES

* Variance distortion
* Correlation distortion
* Selection bias
* Leakage
* Noise amplification
* Unseen-category interaction

---

## 03.2 Outliers & Noise

### PLAN

1. Statistical outlier
2. Contextual outlier
3. Global outlier
4. Measurement error
5. IQR
6. Z-score
7. Modified Z-score
8. MAD
9. Winsorization
10. Clipping
11. Transformation
12. Robust scaling
13. Remove vs retain
14. Influence analysis
15. Cook's distance
16. Leverage

---

## 03.3 Categorical Encoding

### PLAN

1. One-hot
2. Ordinal encoding
3. Label encoding
4. Frequency encoding
5. Count encoding
6. Binary encoding
7. Hash encoding
8. Target encoding
9. Leave-one-out encoding
10. Smoothing
11. Rare-category grouping
12. Unknown-category handling
13. High-cardinality encoding

### FAILURE MODES

* Feature explosion
* False ordinal structure
* Target leakage
* Unseen-category failure
* Sparse matrix explosion

---

## 03.4 Numerical Cleaning

### PLAN

1. Type coercion
2. Range validation
3. Unit normalization
4. Sentinel handling
5. Sign validation
6. Rounding
7. Precision
8. Duplicate detection
9. Schema validation
10. Semantic validation

---

# STAGE 04 — FEATURE ENGINEERING

## 04.1 Numerical Features

### SUB MATERIAL

* Ratios
* Differences
* Products
* Interactions
* Polynomial features
* Log transform
* Log1p
* Power transform
* Root transform
* Reciprocal
* Binning
* Quantile transform
* Clipping

---

## 04.2 Temporal Features

### SUB MATERIAL

* Year
* Month
* Week
* Day
* Hour
* Day of week
* Weekend
* Quarter
* Season
* Lag
* Rolling mean
* Rolling std
* Expanding mean
* Trend
* Recency
* Frequency
* Time since event
* Cyclical encoding

---

## 04.3 Aggregation Features

### SUB MATERIAL

* Group mean
* Group median
* Group sum
* Group count
* Group min/max
* Group variance
* Group standard deviation
* Group rank
* Group percentile
* Historical cumulative features
* Historical rolling features
* Smoothed target rate
* Entity frequency
* Entity lifetime value

### STATISTICS

* Within-group variation
* Between-group variation
* Group sparsity
* Partial pooling
* Aggregation bias
* Ecological fallacy
* Target-aggregation leakage

---

## 04.4 Categorical Interactions

### SUB MATERIAL

* Category combinations
* Crossed features
* Group ratios
* Count share
* Conditional means
* Hierarchical categories
* Feature crosses

---

## 04.5 Feature Selection

### PLAN

1. Filter methods
2. Variance threshold
3. Correlation filtering
4. Mutual information
5. Chi-square
6. ANOVA
7. Wrapper methods
8. RFE
9. Sequential selection
10. Embedded selection
11. L1 selection
12. Tree-based selection
13. SelectKBest

### FAILURE MODES

* Removing interacting features
* Correlated-feature instability
* CV overfitting
* Selection bias
* Information leakage

---

# STAGE 05 — SCALING & TRANSFORMATION

## 05.1 Scaling

### SUB MATERIAL

* Min-Max
* Standardization
* Robust scaling
* Max-absolute
* L2 normalization

## 05.2 Transformation

### SUB MATERIAL

* Log
* Log1p
* Box-Cox
* Yeo-Johnson
* Quantile transform
* Square root
* Reciprocal

## 05.3 MODEL SENSITIVITY

### SUB MATERIAL

* KNN sensitivity
* SVM sensitivity
* K-Means sensitivity
* Logistic sensitivity
* Ridge sensitivity
* Lasso sensitivity
* Kernel sensitivity
* Tree scale independence

### EXPERIMENTS

1. KNN before/after scaling
2. SVM before/after scaling
3. Ridge before/after scaling
4. PCA before/after scaling
5. Tree with vs without scaling

---

# STAGE 06 — DATA SPLITTING & VALIDATION

## 06.1 Splitting

### PLAN

1. Train set
2. Validation set
3. Test set
4. Holdout
5. Random split
6. Stratified split
7. Group split
8. Time split
9. Rolling split
10. Nested split

## 06.2 Cross Validation

### SUB MATERIAL

* K-Fold
* Stratified K-Fold
* Group K-Fold
* Repeated K-Fold
* Time-series CV
* Nested CV
* Leave-one-out
* Bootstrap validation

## 06.3 Experimental Validity

### SUB MATERIAL

* Random seed
* Reproducibility
* Evaluation variance
* Validation bias
* Test contamination
* Fold independence
* Temporal ordering

---

# STAGE 07 — BASELINE MODELS

### PLAN

1. Mean predictor
2. Median predictor
3. Majority class
4. Random baseline
5. Linear baseline
6. Logistic baseline
7. Shallow tree baseline
8. Baseline metric
9. Baseline error
10. Complexity justification

### CASE

1. House price baseline
2. Churn baseline
3. Fraud baseline
4. Air-quality baseline

---

# STAGE 08 — LINEAR REGRESSION

### PLAN

1. Regression problem
2. Continuous target
3. Feature matrix
4. Parameter vector
5. Intercept
6. Coefficient
7. Prediction
8. Residual
9. Least squares
10. Ordinary least squares
11. Normal equation
12. Matrix formulation
13. Geometric interpretation
14. Projection interpretation
15. Gradient descent
16. Closed-form solution
17. MSE
18. SSE
19. Residual analysis
20. Coefficient interpretation

### MATHEMATICS

* Hypothesis function
* OLS objective
* Normal equation
* Gradient
* Hessian
* Projection
* Residual orthogonality

### ASSUMPTIONS

* Linearity
* Independence
* Homoscedasticity
* No perfect multicollinearity
* Error assumptions

### DIAGNOSTICS

* Residual plot
* Residual distribution
* Heteroscedasticity
* Multicollinearity
* High leverage
* Influence

### CASE

1. House price
2. Student score
3. Delivery time
4. Restaurant rating

---

# STAGE 09 — POLYNOMIAL REGRESSION

### PLAN

1. Nonlinear relationship
2. Polynomial expansion
3. Degree
4. Interaction terms
5. Model complexity
6. Bias
7. Variance
8. Underfitting
9. Overfitting
10. Validation curve
11. Regularization
12. Feature explosion

### CASE

1. Study hours vs grade
2. Price vs demand
3. Distance vs delivery time

---

# STAGE 10 — REGULARIZATION

## 10.1 Ridge Regression

### PLAN

1. Overfitting
2. L2 penalty
3. Objective
4. Coefficient shrinkage
5. Bias-variance
6. Multicollinearity
7. Alpha
8. Closed-form Ridge
9. Standardization
10. Cross-validation

### MATHEMATICS

* Penalized MSE
* L2 norm
* Ridge estimator
* Shrinkage path

### CASE

1. Correlated house-price features
2. Air-quality variables
3. Wine features

---

## 10.2 Lasso Regression

### PLAN

1. L1 penalty
2. Sparse coefficients
3. Feature selection
4. Coefficient zeroing
5. Subgradient
6. Coordinate descent
7. Alpha
8. Selection instability

### CASE

1. High-dimensional regression
2. Churn feature selection
3. Pollutant selection

---

## 10.3 Elastic Net

### PLAN

1. L1 + L2
2. Alpha
3. L1 ratio
4. Sparse solution
5. Correlated features
6. Stability
7. Ridge comparison
8. Lasso comparison

---

# STAGE 11 — KNN

## 11.1 KNN Regression

### PLAN

1. Instance-based learning
2. Distance
3. Neighborhood
4. K
5. Uniform weighting
6. Distance weighting
7. Mean target
8. Local approximation
9. Scaling
10. Curse of dimensionality

## 11.2 KNN Classification

### PLAN

1. Majority voting
2. Weighted voting
3. Decision boundary
4. Distance metric
5. Scaling
6. Class imbalance
7. K selection

### COMPARISON

* Euclidean
* Manhattan
* Cosine
* Weighted vs uniform

---

# STAGE 12 — LOGISTIC REGRESSION

### PLAN

1. Binary classification
2. Linear score
3. Sigmoid
4. Probability
5. Logit
6. Odds
7. Odds ratio
8. Threshold
9. Bernoulli likelihood
10. Log likelihood
11. Binary cross entropy
12. Gradient
13. Regularization
14. Decision boundary
15. Probability calibration

### MATHEMATICS

* Sigmoid
* Logit
* BCE
* Likelihood
* Gradient
* Hessian
* Odds ratio

### DIAGNOSTICS

* Confusion matrix
* Calibration
* Residual/deviance
* Class-wise errors

---

# STAGE 13 — SOFTMAX REGRESSION

### PLAN

1. Multiclass classification
2. Class scores
3. Softmax
4. Probability normalization
5. One-hot labels
6. Multiclass cross entropy
7. Gradient
8. Decision boundary
9. One-vs-rest
10. One-vs-one

---

# STAGE 14 — NAIVE BAYES

## 14.1 Gaussian Naive Bayes

### PLAN

1. Bayes theorem
2. Prior
3. Likelihood
4. Posterior
5. Conditional independence
6. Gaussian assumption
7. Mean
8. Variance
9. Log probability
10. Class prediction

## 14.2 Multinomial Naive Bayes

### PLAN

1. Text classification
2. Vocabulary
3. Bag of Words
4. Word counts
5. Class prior
6. Word likelihood
7. Laplace smoothing
8. Log probability
9. Document likelihood

## 14.3 Bernoulli Naive Bayes

### PLAN

1. Binary features
2. Word presence
3. Conditional probability
4. Smoothing
5. Multinomial comparison

---

# STAGE 15 — DECISION TREES

### PLAN

1. Recursive partitioning
2. Root node
3. Internal node
4. Leaf node
5. Split
6. Threshold
7. Gini impurity
8. Entropy
9. Information gain
10. Gain ratio
11. CART
12. Recursive splitting
13. Stopping rules
14. Max depth
15. Min samples split
16. Min samples leaf
17. Pruning
18. Overfitting
19. Feature importance
20. Decision path

### COMPARISON

* Gini vs entropy
* Deep vs shallow
* Pruned vs unpruned
* Classification tree vs regression tree

---

# STAGE 16 — SUPPORT VECTOR MACHINES

## 16.1 Linear SVM

### PLAN

1. Separating hyperplane
2. Margin
3. Support vector
4. Maximum-margin principle
5. Soft margin
6. Slack variables
7. Hinge loss
8. C parameter
9. Regularization
10. Dual formulation
11. Kernel introduction

## 16.2 Kernel SVM

### PLAN

1. Nonlinear boundary
2. Feature mapping
3. Kernel trick
4. Polynomial kernel
5. RBF kernel
6. Gamma
7. C
8. Kernel matrix
9. Support vectors
10. Overfitting

---

# STAGE 17 — ENSEMBLE LEARNING

## 17.1 Bagging

### PLAN

1. Bootstrap
2. Sampling with replacement
3. Base learner
4. Model diversity
5. Aggregation
6. Voting
7. Averaging
8. Variance reduction

## 17.2 Random Forest

### PLAN

1. Bagged trees
2. Bootstrap samples
3. Random feature subsets
4. Multiple trees
5. Voting
6. Averaging
7. OOB samples
8. OOB error
9. Feature importance
10. Correlated-tree reduction

## 17.3 AdaBoost

### PLAN

1. Weak learner
2. Sample weights
3. Misclassification emphasis
4. Sequential learning
5. Learner weight
6. Weighted voting
7. Exponential loss

## 17.4 Gradient Boosting

### PLAN

1. Weak learner
2. Initial prediction
3. Residual
4. Negative gradient
5. Sequential fitting
6. Additive model
7. Learning rate
8. Number of estimators
9. Tree depth
10. Loss function
11. Shrinkage
12. Overfitting

## 17.5 XGBoost

### PLAN

1. Regularized objective
2. Gradient
3. Hessian
4. Split gain
5. Tree complexity penalty
6. Shrinkage
7. Row subsampling
8. Column subsampling
9. Tree pruning
10. Missing-value handling

## 17.6 LightGBM

### PLAN

1. Histogram learning
2. Leaf-wise growth
3. Feature bundling
4. Gradient-based sampling
5. Leaf complexity
6. Overfitting control

## 17.7 Voting

### PLAN

1. Hard voting
2. Soft voting
3. Probability aggregation
4. Model diversity

## 17.8 Stacking

### PLAN

1. Base learners
2. Meta learner
3. Out-of-fold predictions
4. Meta-features
5. Leakage prevention
6. Blending

---

# STAGE 18 — UNSUPERVISED LEARNING

## 18.1 K-Means

### PLAN

1. Unlabeled data
2. Clustering objective
3. Centroid
4. Initialization
5. Distance
6. Assignment
7. Update
8. Convergence
9. Inertia
10. Elbow
11. Silhouette
12. Cluster interpretation
13. K selection
14. Initialization sensitivity

## 18.2 Hierarchical Clustering

### PLAN

1. Distance matrix
2. Agglomerative clustering
3. Merge closest clusters
4. Single linkage
5. Complete linkage
6. Average linkage
7. Ward linkage
8. Dendrogram
9. Cluster cutting

## 18.3 DBSCAN

### PLAN

1. Density
2. Epsilon
3. Min samples
4. Core point
5. Border point
6. Noise point
7. Cluster expansion
8. Arbitrary-shape clusters
9. Outlier detection

## 18.4 Gaussian Mixture Model

### PLAN

1. Mixture components
2. Gaussian component
3. Mean
4. Covariance
5. Mixture weight
6. Soft assignment
7. Likelihood
8. Expectation-Maximization
9. E-step
10. M-step
11. Convergence
12. AIC
13. BIC
14. Uncertainty

### COMPARISON

* K-Means vs GMM
* Hard vs soft clustering
* Centroid vs density
* Density vs probabilistic clustering

---

# STAGE 19 — ANOMALY DETECTION

### PLAN

1. Point anomaly
2. Contextual anomaly
3. Collective anomaly
4. Statistical detection
5. Z-score
6. IQR
7. Distance detection
8. KNN distance
9. Density detection
10. DBSCAN noise
11. GMM probability
12. Isolation Forest
13. One-Class SVM
14. Local Outlier Factor
15. Threshold selection
16. False anomaly
17. Missed anomaly

### CASE

1. Fraud
2. Network intrusion
3. Sensor anomaly
4. Credit-risk anomaly
5. Manufacturing anomaly

---

# STAGE 20 — DIMENSIONALITY REDUCTION

## 20.1 PCA

### PLAN

1. High-dimensional data
2. Feature redundancy
3. Centering
4. Scaling
5. Covariance matrix
6. Eigenvalues
7. Eigenvectors
8. Principal components
9. Explained variance
10. Projection
11. Reconstruction
12. Reconstruction error
13. SVD implementation
14. Component selection

### COMPARISON

* PCA covariance method
* PCA SVD method

## 20.2 LDA

### PLAN

1. Supervised projection
2. Class means
3. Overall mean
4. Within-class scatter
5. Between-class scatter
6. Projection
7. Class separation
8. Generalized eigenproblem

## 20.3 t-SNE

### PLAN

1. High-dimensional similarities
2. Low-dimensional similarities
3. Neighborhoods
4. Perplexity
5. KL divergence
6. Optimization
7. Local structure
8. Visualization limitations

## 20.4 UMAP

### PLAN

1. Manifold assumption
2. Neighbor graph
3. Local structure
4. Global structure
5. n_neighbors
6. min_dist
7. Embedding
8. Visualization

### COMPARISON

* PCA vs LDA
* PCA vs t-SNE
* PCA vs UMAP
* t-SNE vs UMAP
* Visualization vs modeling

---

# STAGE 21 — MODEL EVALUATION

## 21.1 Regression Metrics

### SUB MATERIAL

* Residual
* Absolute error
* Squared error
* MAE
* MSE
* RMSE
* R²
* Adjusted R²
* MAPE
* SMAPE
* Median absolute error
* Quantile loss

## 21.2 Classification Metrics

### SUB MATERIAL

* TP
* TN
* FP
* FN
* Accuracy
* Precision
* Recall
* Specificity
* F1
* Balanced accuracy
* MCC

## 21.3 Threshold Analysis

### PLAN

1. Probability prediction
2. Threshold
3. TPR
4. FPR
5. Precision
6. Recall
7. Threshold sweep
8. Cost-sensitive threshold
9. Risk-sensitive threshold

## 21.4 ROC / PR

### SUB MATERIAL

* ROC
* AUC
* Precision-recall
* PR-AUC
* Imbalanced-class interpretation

## 21.5 Probability Calibration

### SUB MATERIAL

* Calibration
* Reliability diagram
* Brier score
* Platt scaling
* Isotonic regression
* Calibration curve

---

# STAGE 22 — BIAS / VARIANCE / GENERALIZATION

### PLAN

1. Training error
2. Validation error
3. Test error
4. Bias
5. Variance
6. Irreducible error
7. Underfitting
8. Overfitting
9. Model capacity
10. Complexity
11. Learning curve
12. Validation curve
13. Regularization
14. Dataset size
15. Generalization

### CASE

1. Polynomial degree
2. Tree depth
3. Ridge alpha
4. KNN K

---

# STAGE 23 — HYPERPARAMETER OPTIMIZATION

### PLAN

1. Parameter vs hyperparameter
2. Search space
3. Grid Search
4. Random Search
5. Bayesian optimization
6. Successive halving
7. Hyperband
8. Cross-validation tuning
9. Nested CV
10. Early stopping
11. Search budget
12. Validation overfitting

### CASE

1. Ridge alpha
2. KNN K
3. Tree depth
4. Random Forest parameters
5. XGBoost parameters

---

# STAGE 24 — IMBALANCED LEARNING

### PLAN

1. Class imbalance
2. Minority class
3. Majority class
4. Accuracy failure
5. Stratified sampling
6. Class weights
7. Oversampling
8. Undersampling
9. SMOTE
10. Threshold adjustment
11. Cost-sensitive learning
12. Precision-recall tradeoff

### CASE

1. Fraud
2. Disease detection
3. Credit default
4. Churn

---

# STAGE 25 — MODEL INTERPRETABILITY

## 25.1 Global

### SUB MATERIAL

* Linear coefficients
* Odds ratios
* Tree importance
* Permutation importance
* Partial dependence

## 25.2 Local

### SUB MATERIAL

* ICE
* SHAP
* LIME
* Local surrogate models

## 25.3 Interpretation Limits

### SUB MATERIAL

* Correlation vs causation
* Importance vs causality
* Explanation instability
* Feature dependence
* Surrogate limitations
* Interaction effects

---

# STAGE 26 — MODEL DIAGNOSTICS

### PLAN

1. Poor training score
2. Poor validation score
3. Train-validation gap
4. Data leakage
5. Wrong target
6. Weak features
7. Bad preprocessing
8. Class imbalance
9. High bias
10. High variance
11. Label noise
12. Dataset insufficiency
13. Wrong metric
14. Wrong threshold
15. Distribution shift
16. Feature drift
17. Concept drift

### CASE

1. Low train / low test
2. High train / low test
3. Good accuracy / poor recall
4. Excellent CV / bad deployment

---

# STAGE 27 — DISTRIBUTION SHIFT

### PLAN

1. Train distribution
2. Test distribution
3. Covariate shift
4. Label shift
5. Concept drift
6. Population shift
7. Temporal shift
8. Feature drift
9. Label drift
10. Monitoring
11. Drift detection
12. Retraining triggers

---

# STAGE 28 — RECOMMENDER SYSTEMS

## 28.1 Collaborative Filtering

### SUB MATERIAL

* User
* Item
* Rating
* Interaction
* Explicit feedback
* Implicit feedback
* User-item matrix
* Sparsity

## 28.2 Neighbor Methods

### SUB MATERIAL

* User-based KNN
* Item-based KNN
* Cosine similarity
* Pearson similarity
* Rating prediction
* Top-N recommendation

## 28.3 Matrix Factorization

### PLAN

1. Latent factors
2. User embeddings
3. Item embeddings
4. Matrix factorization
5. Reconstruction error
6. Regularization
7. Alternating optimization
8. Cold-start problem

## 28.4 Recommender Evaluation

### SUB MATERIAL

* Precision@K
* Recall@K
* MAP
* NDCG
* Hit Rate
* Coverage
* Diversity
* Cold start

---

# STAGE 29 — REAL-WORLD ML PIPELINE

### PLAN

1. Problem formulation
2. Business objective
3. Scientific objective
4. Target definition
5. Data collection
6. Data validation
7. EDA
8. Leakage audit
9. Data splitting
10. Baseline
11. Preprocessing
12. Feature engineering
13. Model selection
14. Cross-validation
15. Hyperparameter tuning
16. Evaluation
17. Error analysis
18. Interpretation
19. Final model
20. Deployment
21. Monitoring
22. Retraining

---

# STAGE 30 — REAL-WORLD CASE STUDIES

## 30.1 Air Quality Prediction

### PLAN

1. Problem definition
2. Target definition
3. Sensor features
4. Missing values
5. Outliers
6. Correlation
7. Multicollinearity
8. Time features
9. Baseline
10. Linear Regression
11. Ridge
12. Decision Tree
13. Random Forest
14. Gradient Boosting
15. MAE
16. RMSE
17. R²
18. Residual analysis
19. Feature importance
20. Error analysis

## 30.2 Breast Cancer Classification

### PLAN

1. Binary target
2. Class distribution
3. Feature distribution
4. Scaling
5. Logistic Regression
6. Gaussian NB
7. SVM
8. Decision Tree
9. Random Forest
10. Precision
11. Recall
12. F1
13. ROC-AUC
14. PR-AUC
15. False-negative analysis
16. Threshold tuning
17. Calibration
18. Interpretation

## 30.3 Churn Prediction

### PLAN

1. Churn definition
2. Customer behavior
3. Tenure
4. Monthly charge
5. Contract type
6. Support activity
7. Categorical encoding
8. Numerical scaling
9. Feature engineering
10. Logistic Regression
11. Decision Tree
12. Random Forest
13. Gradient Boosting
14. Precision
15. Recall
16. F1
17. ROC-AUC
18. Threshold
19. High-risk segmentation
20. Retention action

## 30.4 Credit Risk

### PLAN

1. Risk classes
2. Class distribution
3. Income
4. Debt ratio
5. Loan amount
6. Payment delay
7. Previous defaults
8. Financial outliers
9. Encoding
10. Scaling
11. Logistic Regression
12. Naive Bayes
13. Decision Tree
14. Random Forest
15. SVM
16. Macro F1
17. Weighted F1
18. Class-wise recall
19. Confusion matrix
20. Risk-factor interpretation

## 30.5 Wine Quality

### PLAN

1. Quality target
2. Regression framing
3. Classification framing
4. Chemical features
5. Correlation
6. Multicollinearity
7. Outliers
8. Linear Regression
9. Ridge
10. Lasso
11. Decision Tree
12. Random Forest
13. Gradient Boosting
14. MAE
15. RMSE
16. R²
17. Classification metrics
18. Feature interpretation
19. Model comparison

---

# STAGE 31 — MODEL SELECTION

### PLAN

1. Problem type
2. Dataset size
3. Feature type
4. Dimensionality
5. Sparsity
6. Noise
7. Nonlinearity
8. Missing values
9. Outliers
10. Interpretability
11. Latency
12. Memory
13. Training cost
14. Prediction cost
15. Stability
16. Calibration
17. Monitoring complexity

---

# STAGE 32 — ALGORITHM COMPARISON

### REQUIRED COMPARISONS

```text
Linear Regression
vs
Ridge
vs
Lasso
vs
Elastic Net
```

```text
Linear Regression
vs
Polynomial Regression
vs
Tree Regression
vs
Random Forest
vs
Gradient Boosting
```

```text
Logistic Regression
vs
Naive Bayes
vs
KNN
vs
Decision Tree
vs
SVM
```

```text
Decision Tree
vs
Random Forest
vs
Gradient Boosting
vs
XGBoost
vs
LightGBM
```

```text
K-Means
vs
Hierarchical
vs
DBSCAN
vs
GMM
```

```text
PCA
vs
LDA
vs
t-SNE
vs
UMAP
```

```text
Isolation Forest
vs
One-Class SVM
vs
LOF
```

---

# STAGE 33 — EXPERIMENTAL THINKING

### PLAN

1. Hypothesis
2. Research question
3. Controlled experiment
4. Baseline
5. Ablation
6. Feature ablation
7. Model ablation
8. Hyperparameter experiment
9. Multiple runs
10. Random seeds
11. Confidence intervals
12. Statistical uncertainty
13. Error analysis
14. Reproducibility
15. Experiment tracking
16. Result interpretation

---

# STAGE 34 — FROM-SCRATCH IMPLEMENTATION

### CORE ALGORITHMS

1. Linear Regression
2. Gradient Descent
3. Ridge
4. Lasso
5. Logistic Regression
6. Softmax Regression
7. KNN
8. Naive Bayes
9. Decision Tree
10. K-Means
11. PCA
12. SVM concept
13. Gradient Boosting concept
14. Random Forest concept

### IMPLEMENTATION LAYERS

1. Pure Python
2. NumPy
3. Vectorized NumPy
4. scikit-learn comparison
5. Unit testing
6. Numerical verification
7. Performance comparison

---

# STAGE 35 — ML SYSTEM THINKING

### PLAN

1. Dataset pipeline
2. Training pipeline
3. Validation pipeline
4. Feature pipeline
5. Feature store concept
6. Model artifact
7. Batch inference
8. Online inference
9. Model serving
10. Latency
11. Throughput
12. Monitoring
13. Data drift
14. Model drift
15. Concept drift
16. Retraining
17. Model versioning
18. Reproducibility
19. Train-serving parity

---

# STAGE 36 — STATISTICAL LEARNING THEORY

### PLAN

1. Hypothesis space
2. Model class
3. Model capacity
4. Empirical risk
5. Expected risk
6. Empirical Risk Minimization
7. Structural Risk Minimization
8. Generalization
9. Generalization error
10. VC dimension
11. Uniform convergence
12. Bias-variance decomposition
13. Regularization
14. Occam's razor
15. Curse of dimensionality
16. No Free Lunch theorem

---

# STAGE 37 — ADVANCED CLASSICAL MACHINE LEARNING

## 37.1 Generalized Linear Models

### SUB MATERIAL

* GLM
* Link function
* Exponential family
* Logistic GLM
* Poisson GLM
* Binomial GLM

## 37.2 Robust & Quantile Regression

### SUB MATERIAL

* Huber regression
* Quantile regression
* Robust loss
* Median regression
* Outlier-resistant estimation

## 37.3 Bayesian Regression

### SUB MATERIAL

* Bayesian linear regression
* Prior
* Posterior
* Posterior predictive
* Conjugacy
* Uncertainty estimation

## 37.4 Gaussian Processes

### SUB MATERIAL

* Kernel
* Covariance function
* Prior function
* Posterior function
* Predictive uncertainty
* Gaussian Process Regression

## 37.5 Discriminant Methods

### SUB MATERIAL

* LDA
* QDA
* Class covariance
* Discriminant function

## 37.6 Kernel & Manifold Methods

### SUB MATERIAL

* Kernel PCA
* Spectral clustering
* Isomap
* Locally Linear Embedding
* Manifold learning

## 37.7 Structured / Sequence Models

### SUB MATERIAL

* Hidden Markov Models
* Conditional Random Fields
* Sequence likelihood
* State transitions

## 37.8 Survival & Ranking

### SUB MATERIAL

* Survival analysis
* Hazard
* Censoring
* Kaplan-Meier
* Cox regression
* Ranking
* Pairwise ranking
* Learning-to-rank foundations

---

# STAGE 38 — MASTER CAPSTONE

## PROJECT 1 — FULL TABULAR REGRESSION

```text
Problem
→ Data Understanding
→ EDA
→ Leakage Audit
→ Split
→ Preprocessing
→ Feature Engineering
→ Baseline
→ Linear Regression
→ Ridge
→ Lasso
→ Elastic Net
→ Tree
→ Random Forest
→ Gradient Boosting
→ Cross Validation
→ Hyperparameter Tuning
→ Evaluation
→ Error Analysis
→ Interpretation
→ Final Model
```

## PROJECT 2 — FULL BINARY CLASSIFICATION

```text
Problem
→ Data Understanding
→ EDA
→ Class Distribution
→ Leakage Audit
→ Split
→ Preprocessing
→ Feature Engineering
→ Baseline
→ Logistic Regression
→ KNN
→ Naive Bayes
→ Decision Tree
→ SVM
→ Random Forest
→ Gradient Boosting
→ ROC-AUC
→ PR-AUC
→ Calibration
→ Threshold Tuning
→ Error Analysis
→ Interpretation
```

## PROJECT 3 — FULL UNSUPERVISED PROJECT

```text
Problem
→ EDA
→ Representation
→ Scaling
→ K-Means
→ Hierarchical
→ DBSCAN
→ GMM
→ Cluster Validation
→ PCA
→ Visualization
→ Interpretation
```

## PROJECT 4 — END-TO-END ML SYSTEM

```text
Data
→ Validation
→ Feature Pipeline
→ Training
→ Cross Validation
→ Model Selection
→ Model Artifact
→ Inference
→ Monitoring
→ Drift Detection
→ Retraining
```

---

# STAGE 39 — CLASSICAL ML MASTER CHECKLIST

## MATHEMATICS

```text
[ ] Linear algebra
[ ] Calculus
[ ] Probability
[ ] Statistics
[ ] Optimization
```

## DATA

```text
[ ] Dataset structure
[ ] Data quality
[ ] EDA
[ ] Missing data
[ ] Outliers
[ ] Leakage
[ ] Encoding
[ ] Scaling
[ ] Feature engineering
```

## SUPERVISED LEARNING

```text
[ ] Linear Regression
[ ] Polynomial Regression
[ ] Ridge
[ ] Lasso
[ ] Elastic Net
[ ] KNN Regression
[ ] Logistic Regression
[ ] Softmax Regression
[ ] KNN Classification
[ ] Naive Bayes
[ ] Decision Trees
[ ] SVM
```

## ENSEMBLES

```text
[ ] Bagging
[ ] Random Forest
[ ] AdaBoost
[ ] Gradient Boosting
[ ] XGBoost
[ ] LightGBM
[ ] Voting
[ ] Stacking
```

## UNSUPERVISED

```text
[ ] K-Means
[ ] Hierarchical Clustering
[ ] DBSCAN
[ ] GMM
[ ] Anomaly Detection
[ ] PCA
[ ] LDA
[ ] t-SNE
[ ] UMAP
```

## EVALUATION

```text
[ ] Regression metrics
[ ] Classification metrics
[ ] ROC-AUC
[ ] PR-AUC
[ ] Calibration
[ ] Threshold tuning
[ ] Bias-variance
[ ] Generalization
```

## ENGINEERING

```text
[ ] Cross-validation
[ ] Hyperparameter tuning
[ ] Experiment design
[ ] Error analysis
[ ] Interpretability
[ ] Distribution shift
[ ] Monitoring
[ ] Reproducibility
```

## IMPLEMENTATION

```text
[ ] From scratch
[ ] NumPy
[ ] scikit-learn
[ ] Vectorization
[ ] Numerical stability
[ ] Unit testing
[ ] Benchmarking
```

---

# STAGE 40 — DEEP LEARNING READINESS

## REQUIRED FOUNDATIONS

### MATHEMATICS

```text
[ ] Vectors
[ ] Matrices
[ ] Matrix multiplication
[ ] Norms
[ ] Eigenvalues
[ ] SVD
[ ] Derivatives
[ ] Gradients
[ ] Jacobians
[ ] Hessians
[ ] Chain rule
[ ] Probability
[ ] Expectation
[ ] Variance
[ ] Conditional probability
[ ] Optimization
```

### MACHINE LEARNING

```text
[ ] Linear models
[ ] Logistic models
[ ] Softmax
[ ] Loss functions
[ ] Cross entropy
[ ] Gradient descent
[ ] Regularization
[ ] Bias-variance
[ ] Generalization
[ ] Train/validation/test
[ ] Cross-validation
[ ] Hyperparameters
[ ] Feature representation
[ ] Model capacity
[ ] Evaluation
[ ] Error analysis
```

### CONCEPTUAL BRIDGE

```text
Linear Regression
→ Linear Layer

Logistic Regression
→ Sigmoid Output

Softmax Regression
→ Softmax Output Layer

Gradient Descent
→ Neural Network Optimization

Cross Entropy
→ Classification Loss

L2 Regularization
→ Weight Decay

Feature Engineering
→ Representation Learning

Chain Rule
→ Backpropagation

Model Capacity
→ Network Capacity

Bias-Variance
→ Deep Model Generalization

Classical Optimization
→ Neural Network Optimizers
```

---

# STAGE 41 — TRANSITION INTO DEEP LEARNING

## 41.1 Perceptron

### PLAN

1. Linear classifier
2. Weights
3. Bias
4. Activation
5. Decision boundary
6. Perceptron loss
7. Weight update
8. Linear separability
9. Limitations

## 41.2 Multilayer Perceptron

### PLAN

1. Layer
2. Neuron
3. Weight
4. Bias
5. Activation
6. Forward pass
7. Loss
8. Backpropagation
9. Gradient computation
10. Parameter update
11. Regularization
12. Generalization

## 41.3 Optimization Bridge

### PLAN

1. Batch gradient descent
2. Mini-batch SGD
3. Momentum
4. Adam
5. Learning-rate scheduling
6. Initialization
7. Gradient flow
8. Gradient clipping

## 41.4 Representation Learning

### PLAN

1. Manual feature engineering
2. Learned features
3. Distributed representations
4. Hierarchical representations
5. Latent spaces

## 41.5 Deep Learning Entry

### PLAN

1. MLP
2. CNN
3. RNN
4. Attention
5. Transformer
6. Representation learning
7. Large-scale optimization
8. Deep model regularization
9. Modern evaluation
10. Deep learning systems
