# CLASSICAL MACHINE LEARNING — 60 DAY ALGORITHM-FIRST MASTER ROADMAP

Pendekatan diubah total menjadi:

```text
ALGORITHM / PROBLEM
        ↓
WHY DOES IT EXIST?
        ↓
MECHANISM
        ↓
MATH NEEDED BY THAT ALGORITHM
        ↓
STATISTICS / PROBABILITY
        ↓
OPTIMIZATION
        ↓
FROM SCRATCH
        ↓
SCIKIT-LEARN
        ↓
DIAGNOSTICS
        ↓
FAILURE MODES
        ↓
EVALUATION
        ↓
INTERPRETATION
        ↓
COMPARISON
        ↓
REAL EXPERIMENT
```

Jadi **bukan**:

```text
Linear Algebra 2 minggu
→ Calculus 2 minggu
→ Probability
→ Statistics
→ baru ML
```

Tetapi:

```text
Linear Regression
→ "kenapa perlu matrix?"
→ belajar matrix yang dibutuhkan

Logistic Regression
→ "kenapa sigmoid?"
→ belajar derivative + likelihood yang dibutuhkan

SVM
→ "kenapa margin?"
→ belajar geometry + constrained optimization yang dibutuhkan

GMM
→ "kenapa EM?"
→ belajar likelihood + latent variables yang dibutuhkan
```

Dengan pola ini, matematika selalu punya **konteks algoritmik**.

---

# TARGET 60 HARI

Pada hari ke-60 targetnya bukan:

> “pernah mempelajari semua algoritma.”

Targetnya:

```text
[✓] Understand
[✓] Explain
[✓] Derive core mathematics
[✓] Implement simplified version
[✓] Use sklearn
[✓] Tune hyperparameters
[✓] Diagnose failures
[✓] Compare models
[✓] Design experiments
[✓] Interpret results
[✓] Build end-to-end ML pipeline
[✓] Understand statistical assumptions
[✓] Understand generalization
[✓] Understand deployment/monitoring
[✓] Be ready for Deep Learning
```

Target realistis: **5–7 jam/hari**.

Pola harian:

```text
90 min  → algorithm
90 min  → math/statistics
90 min  → implementation
60 min  → experiment
30 min  → diagnostics / notes
```

---

# MASTER MAP 60 HARI

```text
D01–D05   Regression Core
D06–D10   Classification Core
D11–D15   Trees + SVM
D16–D20   Ensemble Learning
D21–D25   Unsupervised Learning
D26–D30   Dimensionality + Anomaly + Recommender
D31–D35   Evaluation + Statistical Learning
D36–D40   Data Engineering for ML + Feature Engineering
D41–D45   Advanced Classical ML + Temporal ML
D46–D50   Modern Learning Paradigms + Causal + Decision
D51–D55   ML Systems + Theory + Advanced Integration
D56–D60   Master Capstone + Deep Learning Bridge
```

---

# PHASE 01 — REGRESSION CORE
# DAY 01–05

---

## DAY 01 — LINEAR REGRESSION

### ALGORITHM

```text
Linear Regression
Ordinary Least Squares
```

### MATH YANG KENA

```text
Vectors
Matrices
Dot product
Matrix multiplication
Transpose
Norm
Derivative
Partial derivative
Gradient
Quadratic function
Convexity
```

### STATISTICS

```text
Mean
Variance
Covariance
Residual
Expectation
Noise
```

### MECHANISM

```text
X
↓
Xβ + b
↓
ŷ
↓
residual
↓
squared error
↓
sum
↓
minimize
```

### IMPLEMENTATION

```text
Pure Python
NumPy
Closed-form OLS
Gradient descent OLS
sklearn LinearRegression
```

### DONE

Harus bisa menjelaskan:

```text
Kenapa least squares?
Kenapa residual dikuadratkan?
Kenapa solution menggunakan XᵀX?
Kenapa gradient menuju minimum?
```

---

## DAY 02 — OLS DEEP DIVE

### ALGORITHM

```text
OLS optimization
Normal Equation
Gradient Descent
```

### MATH

```text
Matrix calculus
Quadratic forms
Hessian
Positive semidefinite matrix
Orthogonal projection
Vector spaces
```

### STATISTICS

```text
Estimator
Bias
Variance
Standard error
Sampling distribution
```

### ASSUMPTIONS

```text
Linearity
Independence
Homoscedasticity
No perfect multicollinearity
```

### DIAGNOSTIC

```text
Residual plot
QQ plot
Leverage
Influence
Cook's distance
VIF
```

### DONE

Implement:

```text
OLS closed-form
OLS gradient descent
Residual diagnostics
VIF
Cook's distance
```

---

## DAY 03 — POLYNOMIAL REGRESSION

### ALGORITHM

```text
Polynomial Regression
Interaction Features
```

### MATH

```text
Polynomial functions
Taylor approximation
Feature mapping
Optimization
Bias-variance
```

### STATISTICS

```text
Underfitting
Overfitting
Bias
Variance
Generalization
```

### EXPERIMENT

```text
degree = 1
degree = 2
degree = 3
degree = 5
degree = 10
degree = 20
```

Lihat:

```text
training error
validation error
test error
```

---

## DAY 04 — RIDGE

### ALGORITHM

```text
Ridge Regression
```

### MATH

```text
L2 norm
Quadratic penalty
Matrix inverse
Eigenvalues
Hessian
Shrinkage
```

### STATISTICS

```text
Bias-variance tradeoff
Multicollinearity
Estimator stability
```

### CORE OBJECTIVE

```text
MSE + λ ||β||²
```

### EXPERIMENT

```text
OLS
vs
Ridge α=0.01
α=0.1
α=1
α=10
α=100
```

### DONE

Paham:

```text
kenapa coefficient mengecil
kenapa ridge stabil
kenapa scaling penting
```

---

## DAY 05 — LASSO + ELASTIC NET

### ALGORITHM

```text
Lasso
Elastic Net
```

### MATH

```text
L1 norm
L2 norm
Subgradient
Convex optimization
Coordinate descent
```

### STATISTICS

```text
Sparse estimation
Feature selection
Correlated predictors
Selection instability
```

### CORE

```text
Lasso
MSE + λ||β||₁

Elastic Net
MSE + λ[(1-r)||β||² + r||β||₁]
```

### IMPLEMENT

```text
Lasso coordinate descent
Elastic Net
sklearn comparison
```

### BLOCK 1 DONE

```text
Linear
Polynomial
Ridge
Lasso
Elastic Net
```

---

# PHASE 02 — CLASSIFICATION CORE
# DAY 06–10

---

## DAY 06 — LOGISTIC REGRESSION

### ALGORITHM

```text
Logistic Regression
```

### MATH

```text
Sigmoid
Derivative
Logarithm
Chain rule
Gradient
Hessian
Convex optimization
```

### PROBABILITY

```text
Bernoulli
Conditional probability
Likelihood
Log-likelihood
MLE
```

### CORE

```text
z = Xβ
p = sigmoid(z)
```

### LOSS

```text
Binary Cross Entropy
=
Negative Log Likelihood
```

### IMPLEMENT

```text
Pure NumPy
Gradient descent
sklearn
```

---

## DAY 07 — LOGISTIC INTERPRETATION

### ALGORITHM

```text
Logistic Regression
regularized logistic
```

### MATH

```text
Logit
Odds
Log-odds
Derivative of BCE
```

### STATISTICS

```text
Odds ratio
Confidence interval
Wald test
Likelihood ratio
Deviance
```

### DIAGNOSTICS

```text
Confusion matrix
Probability distribution
Calibration
Residual/deviance
```

---

## DAY 08 — SOFTMAX REGRESSION

### ALGORITHM

```text
Multinomial Logistic Regression
Softmax Regression
One-vs-Rest
One-vs-One
```

### MATH

```text
Exponential
Softmax
Jacobian
Cross-entropy
Chain rule
```

### PROBABILITY

```text
Categorical distribution
Multinomial
Maximum likelihood
```

### CORE

```text
softmax(z_i)
=
exp(z_i) / Σ exp(z_j)
```

### NUMERICAL STABILITY

```text
log-sum-exp trick
```

---

## DAY 09 — KNN

### ALGORITHM

```text
KNN Classification
KNN Regression
```

### MATH

```text
Euclidean distance
Manhattan distance
Minkowski distance
Cosine similarity
Norms
Nearest-neighbor geometry
```

### STATISTICS

```text
Local averaging
Local probability
Curse of dimensionality
Bias-variance
```

### EXPERIMENT

```text
K=1
K=3
K=5
K=10
K=50
```

Bandingkan:

```text
uniform
vs
distance weighted
```

---

## DAY 10 — NAIVE BAYES

### ALGORITHM

```text
Gaussian Naive Bayes
Multinomial Naive Bayes
Bernoulli Naive Bayes
```

### MATH

```text
Conditional probability
Bayes theorem
Expectation
Variance
Logarithm
```

### PROBABILITY

```text
Gaussian
Bernoulli
Multinomial
Conditional independence
Prior
Likelihood
Posterior
```

### IMPLEMENT

```text
Gaussian NB from scratch
Multinomial NB from scratch
Laplace smoothing
Log-probability
```

### BLOCK 2 DONE

```text
Logistic
Softmax
KNN
Naive Bayes
```

---

# PHASE 03 — TREES + SVM
# DAY 11–15

---

## DAY 11 — DECISION TREE

### ALGORITHM

```text
Decision Tree Classification
Decision Tree Regression
CART
```

### MATH

```text
Entropy
Gini impurity
Variance reduction
Information gain
Optimization by split search
```

### INFORMATION THEORY

```text
Entropy
Conditional entropy
Information gain
```

### MECHANISM

```text
Feature
↓
candidate threshold
↓
split
↓
impurity reduction
↓
best split
↓
recursive partition
```

---

## DAY 12 — TREE REGULARIZATION

### ALGORITHM

```text
Pruning
Depth control
Leaf control
```

### MATH

```text
Recursive partition
Piecewise constant functions
Greedy optimization
Complexity
```

### HYPERPARAMETERS

```text
max_depth
min_samples_split
min_samples_leaf
max_features
ccp_alpha
```

### DIAGNOSTICS

```text
train score
validation score
depth
leaf count
```

---

## DAY 13 — LINEAR SVM

### ALGORITHM

```text
Linear SVM
Maximum Margin Classifier
Soft Margin SVM
```

### MATH

```text
Vector geometry
Dot product
Norm
Hyperplane
Distance to hyperplane
Margin
Quadratic optimization
Lagrangian
```

### STATISTICS

```text
Hinge loss
Regularization
Margin-based generalization
```

### CORE

```text
maximize margin
subject to classification constraints
```

---

## DAY 14 — KERNEL SVM

### ALGORITHM

```text
Kernel SVM
Polynomial kernel
RBF kernel
```

### MATH

```text
Inner product
Feature mapping
Kernel function
Distance
Exponentials
High-dimensional geometry
```

### OPTIMIZATION

```text
Dual optimization
Support vectors
C
Gamma
```

### EXPERIMENT

```text
linear
poly
rbf
```

---

## DAY 15 — SVM DEEP DIVE + COMPARISON

### COMPARE

```text
Logistic
vs
KNN
vs
Tree
vs
SVM
```

### ANALYZE

```text
Boundary geometry
Scaling
Noise
Outliers
High-dimensionality
Probability calibration
Inference cost
```

### BLOCK 3 DONE

---

# PHASE 04 — ENSEMBLE LEARNING
# DAY 16–20

---

## DAY 16 — BAGGING

### ALGORITHM

```text
Bagging
Bootstrap Aggregation
```

### MATH

```text
Probability
Sampling with replacement
Variance
Expectation
Law of large numbers
```

### CORE IDEA

```text
many unstable models
↓
average
↓
variance reduction
```

---

## DAY 17 — RANDOM FOREST

### ALGORITHM

```text
Random Forest
```

### MATH

```text
Bootstrap
Random sampling
Correlation
Variance reduction
Expectation
Probability
```

### MECHANISM

```text
bootstrap rows
+
random feature subsets
+
many trees
=
forest
```

### EXTRA

```text
OOB prediction
OOB error
Permutation importance
```

---

## DAY 18 — ADABOOST

### ALGORITHM

```text
AdaBoost
```

### MATH

```text
Weighted loss
Exponentials
Optimization
Weighted voting
```

### STATISTICS

```text
Weak learners
Misclassification weighting
```

### CORE

```text
wrong examples
↓
higher weight
↓
next learner
```

---

## DAY 19 — GRADIENT BOOSTING

### ALGORITHM

```text
Gradient Boosting
```

### MATH

```text
Gradient
Functional gradient
Taylor approximation
Loss derivative
Additive models
```

### CORE

```text
current model
↓
negative gradient
↓
fit weak learner
↓
add learner
↓
repeat
```

---

## DAY 20 — XGBOOST + LIGHTGBM

### ALGORITHM

```text
XGBoost
LightGBM
Voting
Stacking
```

### MATH

```text
Gradient
Hessian
Second-order Taylor
Regularization
Split optimization
```

### XGBOOST

```text
objective
gradient
hessian
split gain
tree penalty
shrinkage
```

### LIGHTGBM

```text
histogram
leaf-wise growth
feature bundling
gradient-based sampling
```

### BLOCK 4 DONE

```text
Bagging
Random Forest
AdaBoost
Gradient Boosting
XGBoost
LightGBM
Voting
Stacking
```

---

# PHASE 05 — UNSUPERVISED LEARNING
# DAY 21–25

---

## DAY 21 — K-MEANS

### ALGORITHM

```text
K-Means
```

### MATH

```text
Euclidean distance
Mean
Squared distance
Centroid
Optimization
```

### MECHANISM

```text
initialize centroid
↓
assign
↓
recalculate centroid
↓
repeat
```

### METRICS

```text
Inertia
Elbow
Silhouette
```

---

## DAY 22 — HIERARCHICAL CLUSTERING

### ALGORITHM

```text
Agglomerative Clustering
```

### MATH

```text
Distance matrix
Linkage
Graph/tree structure
```

### LINKAGE

```text
Single
Complete
Average
Ward
```

### OUTPUT

```text
Dendrogram
```

---

## DAY 23 — DBSCAN

### ALGORITHM

```text
DBSCAN
```

### MATH

```text
Distance
Density
Neighborhood
Threshold geometry
```

### CONCEPT

```text
Core point
Border point
Noise
Epsilon
Min samples
```

### APPLICATION

```text
clustering
+
outlier detection
```

---

## DAY 24 — GMM + EM

### ALGORITHM

```text
Gaussian Mixture Model
Expectation Maximization
```

### MATH

```text
Multivariate Gaussian
Covariance matrix
Determinant
Matrix inverse
Likelihood
Log-likelihood
Latent variables
Expectation
Optimization
```

### EM

```text
E-step
→ posterior responsibility

M-step
→ update parameters
```

---

## DAY 25 — UNSUPERVISED COMPARISON

### COMPARE

```text
K-Means
vs
Hierarchical
vs
DBSCAN
vs
GMM
```

### QUESTIONS

```text
hard vs soft cluster?
spherical vs arbitrary?
density vs probabilistic?
noise handling?
```

### BLOCK 5 DONE

---

# PHASE 06 — DIMENSIONALITY + ANOMALY + RECOMMENDER
# DAY 26–30

---

## DAY 26 — PCA

### ALGORITHM

```text
PCA
```

### MATH

```text
Mean centering
Covariance matrix
Eigenvalues
Eigenvectors
Orthogonal projection
Variance maximization
SVD
```

### CORE

```text
data
↓
center
↓
covariance
↓
eigenvectors
↓
principal components
```

### METRICS

```text
Explained variance
Reconstruction error
```

---

## DAY 27 — LDA + MANIFOLD

### ALGORITHM

```text
Linear Discriminant Analysis
Kernel PCA
```

### MATH

```text
Within-class scatter
Between-class scatter
Generalized eigenvalue problem
Kernel
Projection
```

### ADD

```text
Isomap
Locally Linear Embedding
Spectral clustering
```

---

## DAY 28 — t-SNE + UMAP

### ALGORITHM

```text
t-SNE
UMAP
```

### MATH

```text
Probability distributions
Pairwise similarity
KL divergence
Optimization
Manifold assumption
Graph
Neighborhood
```

### CRITICAL

Understand:

```text
visualization
≠
predictive feature extraction
```

---

## DAY 29 — ANOMALY DETECTION

### ALGORITHM

```text
Isolation Forest
One-Class SVM
LOF
Distance-based detection
Statistical detection
```

### MATH

```text
Z-score
IQR
MAD
Distance
Density
Kernel
Probability
```

### CONCEPTS

```text
Point anomaly
Contextual anomaly
Collective anomaly
False anomaly
Missed anomaly
Threshold
```

---

## DAY 30 — RECOMMENDER SYSTEMS

### ALGORITHM

```text
User-based KNN
Item-based KNN
Matrix Factorization
Collaborative Filtering
```

### MATH

```text
Cosine similarity
Pearson correlation
Matrix factorization
Low-rank approximation
Dot product
Regularization
Alternating optimization
```

### EVALUATION

```text
Precision@K
Recall@K
MAP
NDCG
Hit Rate
Coverage
Diversity
```

### BLOCK 6 DONE

---

# PHASE 07 — EVALUATION + STATISTICAL LEARNING
# DAY 31–35

---

## DAY 31 — REGRESSION EVALUATION

### ALGORITHMS / METRICS

```text
MAE
MSE
RMSE
R²
Adjusted R²
MAPE
SMAPE
Median Absolute Error
Quantile Loss
```

### MATH

```text
Absolute value
Square
Expectation
Variance
Quantiles
```

### CRITICAL

Pahami:

```text
metric
≠
objective
≠
business utility
```

---

## DAY 32 — CLASSIFICATION EVALUATION

### METRICS

```text
Accuracy
Precision
Recall
Specificity
F1
Balanced Accuracy
MCC
```

### MATH

```text
Confusion matrix
Conditional probability
Rates
Harmonic mean
```

### IMBALANCED DATA

```text
accuracy failure
minority class
class weights
threshold
```

---

## DAY 33 — ROC / PR / CALIBRATION

### TOPICS

```text
ROC
AUC
PR Curve
PR-AUC
Threshold sweep
Brier Score
Reliability Diagram
Calibration
Platt Scaling
Isotonic Regression
```

### MATH

```text
Probability
Conditional probability
Ranking
Integration
Expected squared loss
```

---

## DAY 34 — BIAS VARIANCE GENERALIZATION

### THEORY

```text
Training error
Validation error
Test error
Bias
Variance
Irreducible error
Overfitting
Underfitting
Capacity
Generalization
```

### MATH

```text
Expected squared error
Variance decomposition
Expectation
Probability
```

### EXPERIMENT

```text
Polynomial degree
Tree depth
K in KNN
Ridge alpha
```

---

## DAY 35 — HYPERPARAMETER OPTIMIZATION

### ALGORITHMS

```text
Grid Search
Random Search
Bayesian Optimization
Successive Halving
Hyperband
Nested CV
```

### MATH

```text
Optimization
Probability
Sampling
Expected improvement
Search space
```

### STATISTICS

```text
Validation bias
Multiple testing
Selection bias
```

### BLOCK 7 DONE

---

# PHASE 08 — DATA + FEATURE ENGINEERING
# DAY 36–40

Di sini **bukan algoritma model yang menjadi pusat**, tetapi semua hal yang menentukan apakah model menerima data yang benar.

---

## DAY 36 — DATA QUALITY

### TOPICS

```text
Missing values
Duplicates
Invalid values
Impossible values
Schema validation
Type mismatch
Unit inconsistency
Timestamp inconsistency
Measurement error
Noise
```

### MATH / STATS

```text
Mean
Median
Variance
Distribution
Quantile
IQR
Missingness
```

### MISSINGNESS

```text
MCAR
MAR
MNAR
```

---

## DAY 37 — OUTLIERS + IMPUTATION

### ALGORITHMS / METHODS

```text
Mean imputation
Median imputation
Mode
Constant
KNN Imputation
Iterative Imputation
Multiple Imputation
Forward fill
Backward fill
Interpolation
```

### OUTLIER

```text
Z-score
Modified Z-score
MAD
IQR
Winsorization
Clipping
Robust scaling
```

### STATISTICS

```text
Bias
Variance
Selection bias
Measurement error
Influence
Leverage
```

---

## DAY 38 — CATEGORICAL + NUMERICAL PREPROCESSING

### METHODS

```text
One-hot
Ordinal
Frequency
Count
Binary
Hash
Target encoding
Leave-one-out encoding
Smoothing
Rare-category grouping
Unknown handling
```

### MATH

```text
Probability
Frequency
Conditional mean
Smoothing
Target distribution
```

### CRITICAL

Pahami:

```text
encoding
→ feature representation
→ potential leakage
```

---

## DAY 39 — FEATURE ENGINEERING

### NUMERICAL

```text
Ratios
Difference
Product
Interaction
Polynomial
Log
Log1p
Power transform
Root
Reciprocal
Binning
Quantile transform
Clipping
```

### TEMPORAL

```text
Lag
Rolling mean
Rolling std
Expanding mean
Recency
Frequency
Time since event
Cyclical encoding
```

### AGGREGATION

```text
Group mean
Median
Count
Sum
Rank
Percentile
Cumulative features
Historical features
```

### MATH

```text
Transformation
Moments
Aggregation
Correlation
Conditional statistics
```

---

## DAY 40 — FEATURE SELECTION + LEAKAGE

### METHODS

```text
Variance threshold
Correlation filter
Mutual information
Chi-square
ANOVA
RFE
Sequential selection
L1 selection
Tree importance
SelectKBest
```

### LEAKAGE

```text
Target leakage
Feature leakage
Aggregation leakage
Temporal leakage
Group leakage
Preprocessing leakage
Cross-validation leakage
Future-information leakage
Post-hoc leakage
```

### DONE

Harus mampu melakukan:

```text
Leakage Audit
```

sebelum model training.

---

# PHASE 09 — TEMPORAL ML + ADVANCED CLASSICAL ML
# DAY 41–45

---

## DAY 41 — TIME SERIES FOUNDATIONS

### ALGORITHM

```text
Naive Forecast
Moving Average
Exponential Smoothing
```

### MATH

```text
Time indexing
Lag
Difference
Rolling statistics
Weighted average
```

### STATISTICS

```text
Trend
Seasonality
Stationarity
Autocorrelation
Partial autocorrelation
```

### VALIDATION

```text
Random split ❌
Temporal split ✓
Walk-forward ✓
Rolling-origin ✓
```

---

## DAY 42 — ARIMA FAMILY

### ALGORITHM

```text
AR
MA
ARMA
ARIMA
SARIMA
```

### MATH

```text
Linear difference equations
Autocorrelation
Differencing
Likelihood
Optimization
```

### CONCEPTS

```text
forecast horizon
stationarity
seasonality
residual diagnostics
```

---

## DAY 43 — ML FOR TIME SERIES

### ALGORITHM

```text
Lag-feature regression
Random Forest forecasting
Gradient Boosting forecasting
Direct forecasting
Recursive forecasting
Multi-horizon forecasting
```

### MATH

```text
Lag transformation
Rolling statistics
Regression
Conditional expectation
```

### CRITICAL

```text
Temporal leakage
```

---

## DAY 44 — GLM + ROBUST REGRESSION

### ALGORITHM

```text
Generalized Linear Models
Poisson Regression
Binomial GLM
Huber Regression
Quantile Regression
```

### MATH

```text
Exponential family
Link function
Log-likelihood
Robust loss
Absolute loss
Quantiles
```

### STATISTICS

```text
Count data
Median regression
Heavy tails
Outlier resistance
```

---

## DAY 45 — BAYESIAN REGRESSION + GAUSSIAN PROCESS

### ALGORITHM

```text
Bayesian Linear Regression
Gaussian Process Regression
```

### MATH

```text
Bayes theorem
Prior
Likelihood
Posterior
Multivariate Gaussian
Covariance matrix
Kernel
Matrix inverse
Matrix determinant
Conditional Gaussian
```

### OUTPUT

Bukan hanya:

```text
ŷ = value
```

tetapi:

```text
prediction
+
uncertainty
```

---

# PHASE 10 — MODERN LEARNING + CAUSAL + DECISION
# DAY 46–50

---

## DAY 46 — SEMI-SUPERVISED + WEAK SUPERVISION

### ALGORITHMS

```text
Self-training
Pseudo-labeling
Label Propagation
Label Spreading
Weak supervision
```

### MATH

```text
Probability
Confidence
Graph propagation
Similarity
Optimization
```

### CONCEPT

```text
small labeled data
+
large unlabeled data
```

---

## DAY 47 — SELF-SUPERVISED + ACTIVE LEARNING

### ALGORITHMS

```text
Contrastive Learning concept
Metric Learning concept
Active Learning
Uncertainty Sampling
Query by Committee
```

### MATH

```text
Similarity
Distance
Dot product
Cosine
Probability
Entropy
KL divergence
Mutual information
```

### CONCEPT

```text
data
→ representation
→ uncertainty
→ query human
→ obtain label
→ learn
```

---

## DAY 48 — ONLINE / CONTINUAL LEARNING

### ALGORITHM

```text
Online Learning
Incremental Learning
Continual Learning
```

### MATH

```text
Incremental statistics
Online optimization
Gradient updates
Probability
```

### CONCEPTS

```text
Streaming data
Concept drift
Catastrophic forgetting
Replay
Experience buffer
Model update
```

---

## DAY 49 — CAUSAL ML

### ALGORITHM / FRAMEWORK

```text
Potential Outcomes
DAG
Propensity Score
Matching
Inverse Probability Weighting
Doubly Robust Estimation
Causal Forest concept
```

### MATH

```text
Conditional probability
Expectation
Counterfactual
Regression
Weighting
Optimization
```

### STATISTICS

```text
Confounding
Treatment
Outcome
ATE
CATE
Intervention
Backdoor criterion
```

### CRITICAL

```text
prediction
≠
causation
```

---

## DAY 50 — DECISION INTELLIGENCE

### FRAMEWORK

```text
Prediction
↓
Probability
↓
Uncertainty
↓
Cost
↓
Risk
↓
Utility
↓
Decision
↓
Action
```

### MATH

```text
Expected value
Expected utility
Conditional expectation
Decision threshold
Cost-sensitive optimization
Risk minimization
```

### EXAMPLE

```text
P(default)=0.7
```

belum otomatis:

```text
reject
```

Karena:

```text
cost(FP)
cost(FN)
action cost
risk tolerance
```

---

# PHASE 11 — ADVANCED SEQUENCE + RANKING + THEORY
# DAY 51–55

---

## DAY 51 — HMM + CRF

### ALGORITHM

```text
Hidden Markov Model
Conditional Random Field
```

### MATH

```text
Probability
Conditional probability
Markov property
Matrix multiplication
Dynamic programming
Log-likelihood
```

### HMM

```text
Hidden state
↓
Transition
↓
Emission
↓
Observed sequence
```

### ALGORITHMS

```text
Forward
Backward
Viterbi
Baum-Welch
```

---

## DAY 52 — SURVIVAL + RANKING

### ALGORITHM

```text
Kaplan-Meier
Cox Regression
Pairwise Ranking
Learning-to-Rank foundations
```

### MATH

```text
Probability
Survival function
Hazard
Likelihood
Partial likelihood
Ranking loss
```

### CONCEPTS

```text
Censoring
Hazard ratio
Time-to-event
Pairwise preference
```

---

## DAY 53 — STATISTICAL LEARNING THEORY

### THEORY

```text
Hypothesis space
Model class
Capacity
ERM
SRM
Generalization error
VC dimension
Uniform convergence
Occam's razor
Curse of dimensionality
No Free Lunch
```

### MATH

```text
Expectation
Probability
Optimization
Combinatorics
Bounds
```

### CORE QUESTION

```text
Why can a model perform well on unseen data?
```

---

## DAY 54 — EXPERIMENTAL THINKING

### EXPERIMENT DESIGN

```text
Hypothesis
Research question
Controlled experiment
Baseline
Ablation
Feature ablation
Model ablation
Multiple seeds
Confidence interval
Bootstrap
Permutation test
Power analysis
Effect size
```

### STATISTICS

```text
Sampling distribution
Bootstrap
Hypothesis testing
p-value
Power
Multiple testing
Bonferroni
FDR
```

---

## DAY 55 — INTERPRETABILITY + ERROR ANALYSIS

### GLOBAL

```text
Linear coefficients
Odds ratio
Tree importance
Permutation importance
PDP
```

### LOCAL

```text
ICE
SHAP
LIME
Local surrogate
```

### LIMITS

```text
Correlation ≠ causation
Importance ≠ causation
Feature dependence
Explanation instability
Surrogate limitations
```

### ERROR ANALYSIS

```text
False positive clusters
False negative clusters
Segment error
Temporal error
Entity error
Distribution error
```

---

# PHASE 12 — ML SYSTEM + CAPSTONE
# DAY 56–60

---

# DAY 56 — END-TO-END DATA PIPELINE

Bangun satu repository utama.

```text
raw data
↓
schema validation
↓
data cleaning
↓
EDA
↓
leakage audit
↓
split
↓
preprocessing
↓
feature engineering
```

### IMPLEMENT

```text
NumPy
pandas
scikit-learn Pipeline
ColumnTransformer
```

### HARUS ADA

```text
train
validation
test
```

dan bukan:

```text
train/test saja
```

ketika tuning dilakukan.

---

# DAY 57 — FULL SUPERVISED BENCHMARK

Pilih satu regression dataset + satu classification dataset.

### REGRESSION

```text
Mean baseline
Linear
Ridge
Lasso
Elastic Net
Tree
Random Forest
Gradient Boosting
XGBoost
```

### CLASSIFICATION

```text
Majority baseline
Logistic
KNN
Naive Bayes
Tree
SVM
Random Forest
Gradient Boosting
XGBoost
```

### RECORD

```text
training score
CV score
test score
latency
memory
model size
calibration
stability
```

---

# DAY 58 — FULL UNSUPERVISED + ANOMALY + REPRESENTATION

### PIPELINE

```text
EDA
↓
Scaling
↓
PCA
↓
K-Means
↓
Hierarchical
↓
DBSCAN
↓
GMM
↓
Anomaly Detection
↓
Visualization
```

### COMPARE

```text
cluster quality
stability
interpretability
noise sensitivity
```

---

# DAY 59 — FULL ML SYSTEM

Bangun:

```text
DATA
 ↓
VALIDATION
 ↓
FEATURE PIPELINE
 ↓
TRAINING
 ↓
CROSS VALIDATION
 ↓
MODEL SELECTION
 ↓
MODEL ARTIFACT
 ↓
INFERENCE
 ↓
PREDICTION LOG
 ↓
MONITORING
 ↓
DRIFT DETECTION
 ↓
FEEDBACK
 ↓
RETRAINING
```

### SYSTEM CONCEPTS

```text
Model artifact
Model version
Dataset version
Feature version
Experiment tracking
Data lineage
Model lineage
Prediction logging
Train-serving parity
Reproducibility
Rollback
Shadow deployment
Canary deployment
```

### DRIFT

```text
Covariate shift
Label shift
Concept drift
Feature drift
Population shift
Temporal drift
```

---

# DAY 60 — FINAL MASTER CAPSTONE + DEEP LEARNING READINESS

Hari terakhir bukan belajar algoritma baru.

Semua knowledge harus dipakai.

---

# CAPSTONE MASTER

## PROBLEM

Ambil satu real-world problem.

Contoh:

```text
Air Quality Prediction
```

atau:

```text
Customer Churn
```

atau:

```text
Fraud Detection
```

---

## PIPELINE

```text
Problem Definition
        ↓
Business Objective
        ↓
Scientific Objective
        ↓
Target Definition
        ↓
Data Understanding
        ↓
Data Generating Process
        ↓
EDA
        ↓
Data Quality
        ↓
Missingness
        ↓
Outlier
        ↓
Leakage Audit
        ↓
Train/Val/Test Split
        ↓
Feature Engineering
        ↓
Baseline
        ↓
Classical Models
        ↓
Cross Validation
        ↓
Hyperparameter Optimization
        ↓
Calibration
        ↓
Threshold Optimization
        ↓
Error Analysis
        ↓
Interpretability
        ↓
Uncertainty
        ↓
Decision Layer
        ↓
Final Model
        ↓
Inference
        ↓
Monitoring
        ↓
Drift
        ↓
Feedback
        ↓
Retraining Strategy
```

---

# FINAL MODEL COMPARISON

Untuk regression:

```text
Baseline
vs
Linear
vs
Ridge
vs
Lasso
vs
Elastic Net
vs
Polynomial
vs
Random Forest
vs
Gradient Boosting
vs
XGBoost
```

Untuk classification:

```text
Baseline
vs
Logistic
vs
KNN
vs
Naive Bayes
vs
Decision Tree
vs
SVM
vs
Random Forest
vs
Gradient Boosting
vs
XGBoost
```

Untuk clustering:

```text
K-Means
vs
Hierarchical
vs
DBSCAN
vs
GMM
```

Untuk anomaly:

```text
Isolation Forest
vs
One-Class SVM
vs
LOF
```

Untuk dimensionality:

```text
PCA
vs
LDA
vs
t-SNE
vs
UMAP
```

---

# MASTER MATHEMATICS MAP

Dengan pendekatan baru ini, **tidak perlu mempelajari seluruh matematika sebagai blok terpisah**.

Matematika akan muncul ketika algoritma membutuhkannya.

---

## LINEAR ALGEBRA

### Muncul saat:

```text
Linear Regression
Ridge
Lasso
Logistic
SVM
PCA
GMM
Matrix Factorization
Gaussian Process
HMM
Neural Networks
```

### Yang wajib:

```text
Vector
Matrix
Transpose
Dot Product
Norm
Distance
Rank
Inverse
Pseudo-inverse
Projection
Orthogonality
Eigenvalue
Eigenvector
Positive Definite
Covariance Matrix
SVD
Low-rank approximation
```

---

# CALCULUS

### Muncul saat:

```text
Linear Regression
Logistic Regression
Ridge
Lasso
Gradient Boosting
SVM
Gaussian Process
Optimization
Deep Learning
```

### Yang wajib:

```text
Derivative
Partial Derivative
Gradient
Jacobian
Hessian
Chain Rule
Taylor Expansion
Stationary Point
Convexity
Smoothness
Lipschitz continuity
Subgradient
```

---

# PROBABILITY

### Muncul saat:

```text
Logistic
Naive Bayes
GMM
Random Forest
AdaBoost
Calibration
Gaussian Process
HMM
Causal ML
Uncertainty
```

### Yang wajib:

```text
Probability
Conditional Probability
Independence
Conditional Independence
Random Variable
PMF
PDF
CDF
Expectation
Variance
Covariance
Bayes
Likelihood
Posterior
Gaussian
Bernoulli
Binomial
Multinomial
Poisson
Exponential
Beta
Gamma
Student-t
Mixture
```

---

# STATISTICS

### Muncul saat:

```text
OLS
Model evaluation
Cross-validation
Experimentation
Feature selection
Calibration
Uncertainty
Causal ML
```

### Yang wajib:

```text
Estimator
Bias
Variance
Standard Error
Consistency
Efficiency
Confidence Interval
Hypothesis Test
p-value
Effect Size
Power
Bootstrap
Permutation Test
Multiple Testing
FDR
```

---

# INFORMATION THEORY

### Muncul saat:

```text
Decision Tree
Naive Bayes
Classification
Representation Learning
t-SNE
Active Learning
```

### Yang wajib:

```text
Entropy
Cross Entropy
KL Divergence
Jensen-Shannon
Mutual Information
```

---

# OPTIMIZATION

### Muncul saat:

```text
Linear Regression
Logistic
Ridge
Lasso
SVM
K-Means
GMM
Gradient Boosting
Hyperparameter Search
Deep Learning
```

### Yang wajib:

```text
Objective
Loss
Empirical Risk
Gradient Descent
SGD
Mini-batch
Coordinate Descent
Proximal Gradient
Newton
BFGS
L-BFGS
Lagrangian
KKT
Line Search
Trust Region
Preconditioning
```

---

# STATISTICAL LEARNING THEORY

### Muncul setelah model-model sudah dipahami:

```text
Hypothesis Space
Capacity
ERM
SRM
Generalization
VC Dimension
Bias-Variance
Regularization
Curse of Dimensionality
No Free Lunch
```

---

# SETIAP ALGORITMA HARUS MEMILIKI TEMPLATE YANG SAMA

Ini bagian paling penting dari seluruh roadmap.

Untuk **setiap algoritma**, jangan berhenti ketika model sudah berjalan.

Gunakan:

```text
01. What problem does it solve?
02. Why was it invented?
03. What assumption does it make?
04. What is the intuition?
05. What is the geometry?
06. What probability/statistics does it use?
07. What mathematics does it require?
08. What is its objective?
09. How is the objective optimized?
10. What are the hyperparameters?
11. What happens when hyperparameters change?
12. What is the implementation?
13. Can I implement it from scratch?
14. Can I implement it with sklearn?
15. How do I validate it?
16. How do I diagnose it?
17. How does it fail?
18. How do I interpret it?
19. What does it compare well against?
20. When should I not use it?
21. How does it behave under distribution shift?
22. What experiment proves my conclusion?
```

---

# FROM-SCRATCH MINIMUM SET

Tidak semua algoritma perlu implementation sekompleks production library.

Yang **wajib benar-benar dibuat dari scratch**:

```text
Linear Regression
Gradient Descent
Ridge
Lasso
Logistic Regression
Softmax Regression
KNN
Gaussian Naive Bayes
Decision Tree
K-Means
PCA
Gradient Boosting concept
```

Yang cukup conceptual implementation:

```text
SVM
Random Forest
XGBoost
LightGBM
GMM/EM
LOF
Isolation Forest
HMM
Gaussian Process
Matrix Factorization
```

---

# 60-DAY MASTER CHECKLIST

## REGRESSION

```text
[ ] Mean baseline
[ ] Linear Regression
[ ] Polynomial Regression
[ ] Ridge
[ ] Lasso
[ ] Elastic Net
[ ] KNN Regression
[ ] Tree Regression
[ ] Random Forest Regression
[ ] Gradient Boosting
[ ] XGBoost
[ ] Quantile Regression
[ ] Robust Regression
[ ] Bayesian Regression
[ ] Gaussian Process
```

## CLASSIFICATION

```text
[ ] Majority baseline
[ ] Logistic Regression
[ ] Softmax
[ ] KNN
[ ] Naive Bayes
[ ] Decision Tree
[ ] Linear SVM
[ ] Kernel SVM
[ ] Random Forest
[ ] AdaBoost
[ ] Gradient Boosting
[ ] XGBoost
[ ] LightGBM
```

## UNSUPERVISED

```text
[ ] K-Means
[ ] Hierarchical
[ ] DBSCAN
[ ] GMM
[ ] Spectral Clustering
```

## DIMENSIONALITY

```text
[ ] PCA
[ ] LDA
[ ] Kernel PCA
[ ] t-SNE
[ ] UMAP
[ ] Isomap
[ ] LLE
```

## ANOMALY

```text
[ ] Z-score
[ ] IQR
[ ] MAD
[ ] KNN distance
[ ] DBSCAN
[ ] GMM probability
[ ] Isolation Forest
[ ] One-Class SVM
[ ] LOF
```

## RECOMMENDER

```text
[ ] User KNN
[ ] Item KNN
[ ] Collaborative Filtering
[ ] Matrix Factorization
[ ] Precision@K
[ ] Recall@K
[ ] MAP
[ ] NDCG
```

## TEMPORAL

```text
[ ] Naive Forecast
[ ] Moving Average
[ ] Exponential Smoothing
[ ] AR
[ ] MA
[ ] ARMA
[ ] ARIMA
[ ] SARIMA
[ ] ML Forecasting
[ ] Walk-forward Validation
[ ] Rolling-origin
```

## ADVANCED

```text
[ ] GLM
[ ] Bayesian Regression
[ ] Gaussian Process
[ ] HMM
[ ] CRF
[ ] Survival Analysis
[ ] Cox Regression
[ ] Ranking
```

## MODERN LEARNING

```text
[ ] Semi-supervised
[ ] Self-training
[ ] Pseudo-labeling
[ ] Weak supervision
[ ] Self-supervised learning
[ ] Contrastive learning concept
[ ] Active learning
[ ] Online learning
[ ] Continual learning
[ ] Concept drift
```

## CAUSAL

```text
[ ] DAG
[ ] Confounding
[ ] Potential outcomes
[ ] Counterfactual
[ ] ATE
[ ] CATE
[ ] Propensity score
[ ] Matching
[ ] IPW
[ ] Doubly robust
[ ] Causal Forest concept
```

## EVALUATION

```text
[ ] MAE
[ ] MSE
[ ] RMSE
[ ] R²
[ ] Accuracy
[ ] Precision
[ ] Recall
[ ] Specificity
[ ] F1
[ ] MCC
[ ] ROC-AUC
[ ] PR-AUC
[ ] Calibration
[ ] Brier
[ ] Threshold tuning
```

## DATA

```text
[ ] Missingness
[ ] MCAR
[ ] MAR
[ ] MNAR
[ ] Outliers
[ ] Encoding
[ ] Scaling
[ ] Transformation
[ ] Feature engineering
[ ] Feature selection
[ ] Leakage
```

## STATISTICS

```text
[ ] Estimation
[ ] Confidence interval
[ ] Hypothesis testing
[ ] Bootstrap
[ ] Permutation
[ ] Effect size
[ ] Power
[ ] Multiple testing
[ ] FDR
```

## ML SYSTEM

```text
[ ] Data validation
[ ] Training pipeline
[ ] Feature pipeline
[ ] Model artifact
[ ] Model versioning
[ ] Dataset versioning
[ ] Experiment tracking
[ ] Inference
[ ] Monitoring
[ ] Data drift
[ ] Model drift
[ ] Concept drift
[ ] Retraining
[ ] Train-serving parity
```

---

# DEFINITION OF “DONE” PADA HARI KE-60

Jangan menganggap:

```text
sklearn.fit()
```

sebagai selesai.

Untuk algoritma utama, status **DONE** harus seperti:

```text
I can explain it
        ↓
I understand why it works
        ↓
I know the mathematics
        ↓
I understand the statistical assumptions
        ↓
I can derive the core equation
        ↓
I can implement a simplified version
        ↓
I can use sklearn
        ↓
I can tune it
        ↓
I can diagnose it
        ↓
I know its failure modes
        ↓
I can compare it
        ↓
I can interpret it
        ↓
I can evaluate it correctly
        ↓
I can use it on real data
```

---

# FINAL ARCHITECTURE OF YOUR LEARNING

Setelah diubah ke pendekatan algorithm-first, keseluruhan roadmap sebenarnya menjadi:

```text
                    CLASSICAL ML
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   SUPERVISED       UNSUPERVISED      TEMPORAL
        │                │                │
   Regression        Clustering        Forecasting
   Classification   GMM/PCA/etc       Sequential ML
        │                │                │
        └────────────────┼────────────────┘
                         │
                    EVALUATION
                         │
               STATISTICS / UNCERTAINTY
                         │
                  DECISION LAYER
                         │
                 EXPERIMENTATION
                         │
                   ML SYSTEMS
                         │
              ONLINE / CONTINUAL ML
                         │
                    CAUSAL ML
                         │
               REPRESENTATION LEARNING
                         │
                    DEEP LEARNING
```

Dan hubungan matematikanya:

```text
Linear Models
→ Linear Algebra
→ Calculus
→ Optimization

Logistic
→ Probability
→ Likelihood
→ Calculus

Naive Bayes
→ Probability
→ Bayes

Trees
→ Information Theory

KNN
→ Geometry
→ Norms
→ Distance

SVM
→ Geometry
→ Convex Optimization
→ KKT

Random Forest
→ Probability
→ Bootstrap
→ Variance Reduction

Boosting
→ Calculus
→ Gradient
→ Taylor

GMM
→ Probability
→ Multivariate Gaussian
→ Likelihood
→ EM

PCA
→ Linear Algebra
→ Eigen
→ SVD

Matrix Factorization
→ Linear Algebra
→ Optimization

Gaussian Process
→ Probability
→ Linear Algebra
→ Bayesian inference

HMM
→ Probability
→ Dynamic Programming

Time Series
→ Statistics
→ Autocorrelation
→ Sequential dependence

Causal ML
→ Probability
→ Statistics
→ Counterfactual reasoning

Self-Supervised
→ Similarity
→ Information Theory
→ Optimization

Continual Learning
→ Online Optimization
→ Distribution Shift

Decision Intelligence
→ Probability
→ Expected Utility
→ Risk
```

---

# SETELAH 60 HARI

Pada titik ini, **jangan mengulang seluruh matematika dari awal**.

Masuk ke:

```text
DAY 61+
        ↓
PERCEPTRON
        ↓
MLP
        ↓
BACKPROPAGATION
        ↓
DEEP OPTIMIZATION
        ↓
CNN
        ↓
RNN
        ↓
ATTENTION
        ↓
TRANSFORMER
        ↓
REPRESENTATION LEARNING
        ↓
FOUNDATION MODELS
        ↓
RL
        ↓
AGENTIC ML
        ↓
MEMORY
        ↓
TOOL USE
        ↓
PLANNING
        ↓
FEEDBACK LOOP
        ↓
SELF-EVALUATION
        ↓
SELF-IMPROVEMENT
```

