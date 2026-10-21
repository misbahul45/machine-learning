# 00.2 Statistics
# Advanced Statistical Foundations for Machine Learning, AI Research & Data-Driven Decisions

> **Objective:** Develop a rigorous understanding of statistics as a mathematical framework for learning from uncertain, incomplete, noisy, dependent, and potentially biased data.
>
> **Level:** Advanced Undergraduate → Graduate → Research-Oriented Statistics
>
> **Prerequisite:** Linear Algebra, Calculus, Programming, and Basic Mathematical Reasoning.
>
> **Primary Stack:** Python, NumPy, SciPy, pandas, statsmodels, scikit-learn, PyMC, ArviZ.
>
> **Learning Philosophy:** Probability → Statistical Theory → Inference → Modeling → Diagnostics → Experiments → Causal Reasoning → Decision-Making.

---

# 00. OVERVIEW

Statistik bukan sekadar kumpulan formula untuk menghitung mean, median, variance, correlation, confidence interval, atau p-value.

Statistik adalah disiplin matematis untuk memahami bagaimana informasi tentang suatu populasi dapat dipelajari dari data yang dihasilkan melalui proses yang tidak sepenuhnya diketahui.

Tujuan utama roadmap ini adalah memahami:

1. Bagaimana data dihasilkan oleh suatu stochastic process.
2. Bagaimana random variables dan probability distributions merepresentasikan ketidakpastian.
3. Bagaimana sampling menghasilkan variasi statistik.
4. Bagaimana suatu estimator dapat dievaluasi secara matematis.
5. Bagaimana parameter populasi diestimasi dari observasi terbatas.
6. Bagaimana mengukur ketidakpastian sebuah estimasi.
7. Bagaimana menguji hipotesis tanpa salah menafsirkan evidence.
8. Bagaimana membangun model statistik dengan asumsi yang eksplisit.
9. Bagaimana menghadapi missing data, measurement error, selection bias, dan distribution shift.
10. Bagaimana membedakan association, prediction, dan causation.
11. Bagaimana merancang eksperimen agar menghasilkan evidence yang valid.
12. Bagaimana melakukan Bayesian inference dan computational statistics.
13. Bagaimana membangun statistical models untuk data longitudinal, survival, time series, dan high-dimensional settings.
14. Bagaimana melakukan uncertainty quantification terhadap prediksi machine learning.
15. Bagaimana menerapkan statistik secara bertanggung jawab untuk healthcare, finance, AI evaluation, personalization, dan scientific research.

## 00.1 Relationship to Linear Algebra

Linear Algebra menyediakan representasi dan operasi matematis.

Statistics menggunakan representasi tersebut untuk mempelajari struktur probabilistik dan membuat inferensi dari data.

```text
LINEAR ALGEBRA
    |
    v
Vectors / Matrices / Subspaces
    |
    v
Covariance / Inner Products / Matrix Decompositions
    |
    v
PROBABILITY THEORY
    |
    v
Random Variables / Distributions / Expectations
    |
    v
SAMPLING THEORY
    |
    v
Sampling Distributions / LLN / CLT
    |
    v
STATISTICAL INFERENCE
    |
    v
Estimation / Testing / Uncertainty Quantification
    |
    v
STATISTICAL MODELING
    |
    v
Regression / GLM / Hierarchical Models / Bayesian Models
    |
    v
CAUSAL INFERENCE & EXPERIMENTAL DESIGN
    |
    v
STATISTICAL LEARNING
    |
    v
Prediction / Calibration / Generalization / Model Evaluation
    |
    v
SCIENTIFIC DECISION-MAKING
```

Linear algebra tetap digunakan dalam covariance estimation, likelihood optimization, regression, Bayesian computation, multivariate analysis, dan machine learning.

Namun, fokus roadmap ini adalah **makna statistik dari operasi tersebut**, asumsi yang mendasarinya, serta kondisi ketika kesimpulannya valid atau gagal.

## 00.2 The Three Fundamental Questions of Statistics

Setiap permasalahan statistik sebaiknya dibedakan menjadi tiga pertanyaan.

### A. Description

Apa yang terlihat dalam data?

Contoh:

- Bagaimana distribusi nilai suatu variabel?
- Apakah dua variabel bergerak bersama?
- Apakah terdapat kelompok atau pola?
- Seberapa besar variasi observasi?

### B. Inference

Apa yang dapat disimpulkan mengenai proses atau populasi yang menghasilkan data?

Contoh:

- Berapa estimasi parameter populasi?
- Seberapa besar ketidakpastian estimasi tersebut?
- Apakah suatu efek didukung oleh data?
- Seberapa sensitif kesimpulan terhadap asumsi model?

### C. Prediction and Decision

Apa yang dapat diprediksi, dan keputusan apa yang dapat dibuat berdasarkan informasi tersebut?

Contoh:

- Seberapa akurat prediksi untuk observasi baru?
- Apakah model terkalibrasi?
- Apakah suatu intervensi menghasilkan perubahan?
- Apakah evidence cukup untuk mengambil keputusan?

Ketiga pertanyaan tersebut saling berkaitan, tetapi tidak identik.

Model yang baik untuk prediction belum tentu menghasilkan estimasi causal effect yang valid. Hubungan statistik yang kuat juga belum tentu menunjukkan hubungan sebab-akibat.

---

# 01. LEARNING PROTOCOL

Setiap modul harus dipelajari menggunakan enam lapisan berikut.

```text
LAYER 1 — MATHEMATICAL DEFINITION
    |
    v
LAYER 2 — PROBABILISTIC / STATISTICAL ASSUMPTIONS
    |
    v
LAYER 3 — ESTIMATION OR INFERENCE MECHANISM
    |
    v
LAYER 4 — IMPLEMENTATION FROM SCRATCH
    |
    v
LAYER 5 — SIMULATION, DIAGNOSTICS & FAILURE ANALYSIS
    |
    v
LAYER 6 — REAL-WORLD CASE STUDY
```

Jangan menganggap sebuah topik selesai hanya karena dapat menggunakan library.

Untuk setiap metode, jawab pertanyaan berikut:

1. Parameter atau quantity apa yang sebenarnya ingin dipelajari?
2. Apa yang dianggap tetap dan apa yang dianggap random?
3. Apa data-generating process yang diasumsikan?
4. Apa identifiability conditions yang diperlukan?
5. Apa estimator atau inferential procedure yang digunakan?
6. Apa jaminan matematis yang dimiliki metode tersebut?
7. Apa yang terjadi ketika asumsi dilanggar?
8. Bagaimana cara mendeteksi pelanggaran tersebut?
9. Apakah hasil dapat direplikasi melalui simulation?
10. Dalam kondisi apa metode alternatif lebih tepat?

## 01.1 Statistical Knowledge Levels

```text
LEVEL 1 — COMPUTATIONAL
Can calculate a statistic.

LEVEL 2 — INTERPRETATIVE
Can interpret its meaning and uncertainty.

LEVEL 3 — MATHEMATICAL
Can derive the estimator or inferential procedure.

LEVEL 4 — DIAGNOSTIC
Can identify assumptions, bias, and failure modes.

LEVEL 5 — COMPARATIVE
Can justify why one method is preferred over another.

LEVEL 6 — RESEARCH
Can extend, evaluate, or develop methods for unfamiliar settings.
```

Target akhirnya bukan hanya mampu menjalankan metode statistik, tetapi dapat menentukan apakah hasil metode tersebut layak dipercaya.

---

# 02. ADVANCED STATISTICS ROADMAP

# DAY 1 — Probability Theory, Random Variables & Stochastic Foundations

## Objective

Memahami probability sebagai bahasa matematis untuk merepresentasikan ketidakpastian dan menghubungkannya dengan statistical inference.

Karena Linear Algebra sudah dibangun secara terpisah, fokus di sini bukan operasi matriks, melainkan probabilistic structure.

## Core

1. Probability spaces
2. Sample spaces
3. Events and event algebras
4. Probability measures
5. Conditional probability
6. Conditional independence
7. Bayes' theorem
8. Random variables
9. Probability mass functions
10. Probability density functions
11. Cumulative distribution functions
12. Joint distributions
13. Marginal distributions
14. Conditional distributions
15. Independence and dependence
16. Expectation operators
17. Conditional expectation
18. Variance and covariance
19. Correlation
20. Moment-generating functions
21. Characteristic functions
22. Transformations of random variables
23. Random vectors
24. Probability inequalities
25. Stochastic processes

## Mathematics

Conditional probability:

\[
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
\]

Bayes' theorem:

\[
P(A \mid B)
=
\frac{P(B \mid A)P(A)}{P(B)}
\]

Expectation:

\[
\mathbb{E}[X]
\]

Variance:

\[
\operatorname{Var}(X)
=
\mathbb{E}\left[(X-\mathbb{E}[X])^2\right]
\]

Covariance:

\[
\operatorname{Cov}(X,Y)
=
\mathbb{E}\left[
(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])
\right]
\]

Law of total expectation:

\[
\mathbb{E}[Y]
=
\mathbb{E}\left[\mathbb{E}[Y\mid X]\right]
\]

Law of total variance:

\[
\operatorname{Var}(Y)
=
\mathbb{E}\left[\operatorname{Var}(Y\mid X)\right]
+
\operatorname{Var}\left(\mathbb{E}[Y\mid X]\right)
\]

## Advanced Probability Concepts

### A. Dependence Structures

Pelajari perbedaan:

- Independence
- Conditional independence
- Uncorrelatedness
- Exchangeability
- Positive dependence
- Negative dependence
- Tail dependence

Korelasi nol tidak selalu berarti independence.

Independence dan exchangeability juga tidak boleh diperlakukan sebagai asumsi yang identik.

### B. Distribution Families

#### Discrete

- Bernoulli
- Binomial
- Geometric
- Negative binomial
- Poisson
- Categorical
- Multinomial
- Hypergeometric

#### Continuous

- Uniform
- Normal
- Exponential
- Gamma
- Beta
- Chi-square
- Student's t
- F-distribution
- Weibull
- Log-normal
- Logistic
- Pareto
- Generalized extreme-value distributions

#### Multivariate

- Multivariate normal
- Multivariate t
- Dirichlet
- Multinomial
- Wishart
- Gaussian copula
- Student-t copula

Jangan sekadar menghafal distribution family.

Pahami data-generating process, parameterisasi, support, moments, tail behavior, dan kondisi ketika distribusi tersebut masuk akal.

## Algorithm Mechanism

- Inverse transform sampling
- Rejection sampling
- Importance sampling
- Conditional simulation
- Monte Carlo expectation estimation
- Sampling from joint distributions
- Sampling from multivariate distributions

## From Scratch

Implementasikan:

1. Random variable simulation.
2. Empirical PMF dan PDF approximation.
3. Conditional probability estimator.
4. Bayes updating untuk model diskret.
5. Monte Carlo integration.
6. Empirical expectation dan variance.
7. Sampling dari distribusi dengan inverse transform.
8. Rejection sampling untuk distribusi sederhana.

## Experiment

### Experiment 1 — Law of Large Numbers

Bandingkan estimasi expectation ketika sample size meningkat.

\[
\hat{\mu}_n
=
\frac{1}{n}\sum_{i=1}^{n}X_i
\]

Amati konvergensi empirical mean terhadap population expectation.

### Experiment 2 — Dependence vs Correlation

Bangun variabel yang tidak berkorelasi tetapi tidak independent.

Tujuannya adalah memahami batas informasi yang diberikan correlation.

### Experiment 3 — Tail Behavior

Bandingkan normal, Student's t, dan Pareto dalam:

- Extreme observations
- Outlier frequency
- Tail probability
- Sample mean variability
- Sensitivity to rare events

## Failure Modes

- Confusing probability with observed frequency
- Assuming independence without justification
- Treating correlation as causation
- Ignoring tail behavior
- Confusing marginal and conditional distributions
- Misinterpreting probability density as point probability
- Assuming finite variance without verification
- Using Gaussian assumptions for strongly heavy-tailed data

## Case — Medical Risk Probability

```text
Population
    |
    v
Disease Prevalence
    |
    v
Test Sensitivity and Specificity
    |
    v
Observed Positive Result
    |
    v
Bayesian Updating
    |
    v
Posterior Probability of Disease
```

Tujuan:

Memahami mengapa probabilitas suatu kondisi setelah hasil tes positif tidak identik dengan sensitivity atau specificity alat tes.

---

# DAY 2 — Sampling Theory, Convergence & Asymptotic Statistics

## Objective

Memahami mengapa statistik yang dihitung dari sampel berbeda-beda, bagaimana sampling uncertainty muncul, dan kapan estimasi dapat mendekati suatu nilai populasi.

## Core

26. Population and sample
27. Sampling frame
28. Probability sampling
29. Simple random sampling
30. Stratified sampling
31. Cluster sampling
32. Systematic sampling
33. Sampling with replacement
34. Sampling without replacement
35. Sampling distributions
36. Standard errors
37. Sampling variability
38. Bias from sampling design
39. Weak law of large numbers
40. Strong law of large numbers
41. Central limit theorem
42. Multivariate central limit theorem
43. Convergence in probability
44. Almost-sure convergence
45. Convergence in distribution
46. Convergence in \(L^p\)
47. Slutsky's theorem
48. Continuous mapping theorem
49. Delta method
50. Asymptotic normality
51. Concentration inequalities
52. Finite-sample bounds

## Mathematics

Standard error of the sample mean for independent observations with finite variance:

\[
\operatorname{SE}(\bar X)
=
\frac{\sigma}{\sqrt n}
\]

Central Limit Theorem, under appropriate assumptions:

\[
\frac{\sqrt n(\bar X-\mu)}{\sigma}
\overset{d}{\longrightarrow}
N(0,1)
\]

The asymptotic approximation is not the same as an exact finite-sample distribution.

Delta method:

Jika

\[
\sqrt n(T_n-\theta)
\overset{d}{\longrightarrow}
N(0,V)
\]

dan \(g\) differentiable di \(\theta\), maka, under regularity conditions,

\[
\sqrt n\left[g(T_n)-g(\theta)\right]
\overset{d}{\longrightarrow}
N\left(0,[g'(\theta)]^2V\right)
\]

Untuk estimator vektor, variance asymptotic diperoleh menggunakan Jacobian.

## Sub Material

- Glivenko-Cantelli theorem
- Berry-Esseen bounds
- Hoeffding inequality
- Bernstein inequality
- Chernoff bounds
- Empirical process foundations
- Bootstrap consistency
- Effective sample size
- Finite-population correction
- Survey sampling weights
- Cluster-robust sampling uncertainty
- Missingness induced selection

## Interpretation

Pahami perbedaan:

```text
Population Variability
        |
        v
Sampling Variability
        |
        v
Estimator Uncertainty
        |
        v
Confidence Interval / Inferential Uncertainty
```

Population variance menjelaskan variasi antarunit dalam populasi.

Sampling variance menjelaskan variasi suatu estimator akibat proses sampling.

Keduanya bukan quantity yang sama.

## Algorithm Mechanism

- Simulation-based sampling distributions
- Monte Carlo estimation of standard errors
- Bootstrap approximation
- Concentration-bound calculation
- Delta-method approximation
- Survey-weighted estimation

## Diagnostics

- Sample representativeness
- Effective sample size
- Sampling weights
- Cluster dependence
- Asymptotic approximation quality
- Monte Carlo error
- Tail sensitivity
- Stability across sample sizes

## From Scratch

1. Simulate repeated samples from a known population.
2. Estimate sampling distribution of the mean.
3. Compare empirical standard error with theoretical standard error.
4. Demonstrate the Central Limit Theorem.
5. Estimate Monte Carlo error.
6. Implement a simple concentration-bound calculation.
7. Compare simple random and clustered sampling.
8. Measure the effect of unequal sampling probabilities.

## Experiments

### Experiment 1 — CLT Under Different Distributions

Bandingkan:

- Normal
- Bernoulli
- Exponential
- Student's t
- Heavy-tailed distributions

Periksa bagaimana bentuk sampling distribution berubah ketika \(n\) meningkat.

Jangan menyimpulkan bahwa convergence berjalan dengan kecepatan sama untuk semua distribusi.

### Experiment 2 — Sample Size vs Uncertainty

Evaluasi:

\[
\operatorname{SE}(\bar X)\propto n^{-1/2}
\]

Periksa apakah penambahan sampel memperkecil ketidakpastian sesuai pendekatan teoritis.

### Experiment 3 — IID vs Clustered Data

Bandingkan standard error ketika observasi independen dengan kondisi ketika observasi memiliki intraclass correlation.

## Failure Modes

- Treating observations as IID when they are clustered
- Using CLT without checking relevant assumptions
- Assuming a large sample fixes selection bias
- Confusing population variance with estimator variance
- Ignoring unequal sampling probabilities
- Reporting standard errors for a misspecified sampling design
- Applying asymptotic approximations to very rare events without diagnostics

## Case — Generalization Uncertainty in AI Evaluation

```text
Evaluation Dataset
    |
    v
Sampling Mechanism
    |
    v
Observed Model Metric
    |
    v
Sampling Distribution
    |
    v
Confidence Interval
    |
    v
Uncertainty of Model Comparison
```

Tujuan:

Memahami mengapa skor model yang berbeda tipis belum tentu memberikan evidence kuat bahwa salah satu model benar-benar lebih baik pada populasi target.

---

# DAY 3 — Statistical Estimation, Decision Theory & Efficiency

## Objective

Memahami estimator sebagai fungsi dari data, bukan sekadar formula untuk menebak parameter.

Fokus utama adalah kualitas estimator, optimality, risk, identifiability, dan statistical decision theory.

## Core

53. Parameters and estimands
54. Statistics as functions of observations
55. Estimators
56. Estimates
57. Sampling distributions of estimators
58. Bias
59. Variance
60. Mean squared error
61. Consistency
62. Asymptotic bias
63. Efficiency
64. Relative efficiency
65. Sufficiency
66. Minimal sufficiency
67. Ancillary statistics
68. Completeness
69. Rao-Blackwell theorem
70. Lehmann-Scheffé theorem
71. Fisher information inequality
72. Cramér-Rao lower bound
73. Identifiability
74. Statistical decision theory
75. Loss functions
76. Risk functions
77. Minimax estimation
78. Admissibility
79. Shrinkage estimation
80. Empirical Bayes

## Mathematics

Estimator:

\[
\hat\theta=T(X_1,\ldots,X_n)
\]

Bias:

\[
\operatorname{Bias}(\hat\theta)
=
\mathbb E[\hat\theta]-\theta
\]

Mean squared error:

\[
\operatorname{MSE}(\hat\theta)
=
\mathbb E[(\hat\theta-\theta)^2]
\]

Bias-variance decomposition:

\[
\operatorname{MSE}(\hat\theta)
=
\operatorname{Var}(\hat\theta)
+
\operatorname{Bias}(\hat\theta)^2
\]

Statistical risk:

\[
R(\theta,\delta)
=
\mathbb E_\theta[L(\theta,\delta(X))]
\]

Di sini \(\delta\) adalah decision rule dan \(L\) adalah loss function.

## Advanced Concepts

### A. Estimand vs Estimator

- **Estimand:** Quantity yang ingin diketahui.
- **Estimator:** Prosedur matematis untuk mengestimasinya.
- **Estimate:** Nilai yang dihasilkan oleh estimator dari sampel tertentu.

Contoh:

Estimand:

\[
\theta=\mathbb E[Y]
\]

Estimator:

\[
\hat\theta=\frac{1}{n}\sum_{i=1}^{n}Y_i
\]

Estimate: nilai numerik yang diperoleh dari sampel aktual.

Pembedaan ini penting dalam causal inference dan experimental design.

### B. Estimator Optimality

Pahami bahwa tidak ada estimator yang optimal untuk semua kondisi dan semua loss functions.

Perbandingan estimator membutuhkan:

- Target parameter
- Model atau asumsi probabilistik
- Loss function
- Sample size
- Bias
- Variance
- Robustness terhadap model misspecification

### C. Shrinkage

Pelajari mengapa estimator yang sengaja digeser menuju nilai pusat terkadang dapat memiliki MSE lebih rendah daripada estimator yang tidak bias.

Contoh:

- James-Stein estimator
- Ridge-type shrinkage
- Empirical Bayes estimation
- Hierarchical partial pooling

## Algorithm Mechanism

- Method of moments
- Maximum likelihood estimation
- Minimum distance estimation
- Generalized method of moments
- Rao-Blackwellization
- Shrinkage estimation
- Decision-theoretic optimization

## Diagnostics

- Bias
- Variance
- MSE
- Coverage
- Relative efficiency
- Sensitivity to outliers
- Identifiability
- Parameter stability
- Estimator performance under misspecification

## From Scratch

1. Implement method-of-moments estimation.
2. Compare biased and unbiased estimators.
3. Estimate empirical MSE through simulation.
4. Compare estimators under several loss functions.
5. Demonstrate shrinkage estimation.
6. Compare pooled, unpooled, and partially pooled estimates.
7. Simulate the effect of model misspecification.

## Experiment

### Bias-Variance-Risk Trade-off

Bangun eksperimen ketika:

```text
Estimator A:
Low Bias + High Variance

Estimator B:
Moderate Bias + Lower Variance

Estimator C:
Potentially Lower Variance Through Shrinkage
```

Bandingkan MSE pada berbagai sample sizes dan data-generating processes.

## Failure Modes

- Treating unbiasedness as a guarantee of optimality
- Ignoring variance when comparing estimators
- Choosing an estimator without identifying the target quantity
- Optimizing MSE when the operational loss is asymmetric
- Ignoring identifiability
- Extrapolating estimator guarantees beyond their assumptions
- Confusing parameter identification with numerical optimization

## Case — Hospital-Level Performance Estimation

Misalkan dua fasilitas kesehatan memiliki jumlah observasi yang sangat berbeda.

Estimasi langsung dari fasilitas dengan sampel kecil dapat jauh lebih tidak stabil.

Bandingkan:

```text
Raw Facility Estimate
        |
        v
Variance Assessment
        |
        v
Partial Pooling
        |
        v
Hierarchical Estimate
        |
        v
More Stable Comparison
```

Tujuan:

Memahami mengapa partial pooling dapat membantu membedakan variasi nyata dengan noise akibat sampel kecil.

---

# DAY 4 — Likelihood Theory, MLE & Advanced Parameter Estimation

## Objective

Memahami likelihood sebagai pusat dari banyak metode inferensi statistik modern.

## Core

81. Statistical models
82. Likelihood functions
83. Log-likelihood
84. Maximum likelihood estimation
85. Score functions
86. Observed Fisher information
87. Expected Fisher information
88. Hessian of log-likelihood
89. Likelihood geometry
90. Identifiability and likelihood
91. Exponential families
92. Sufficient statistics
93. Method of moments
94. Profile likelihood
95. Constrained maximum likelihood
96. Penalized likelihood
97. Likelihood ratio
98. Wald inference
99. Score inference
100. Likelihood-based confidence intervals
101. EM algorithm
102. Missing-data likelihood
103. Boundary parameters
104. Nonregular likelihood problems

## Mathematics

Likelihood:

\[
L(\theta;x)
=
p(x\mid\theta)
\]

Log-likelihood untuk observasi independen:

\[
\ell(\theta)
=
\sum_{i=1}^{n}\log p(x_i\mid\theta)
\]

Maximum likelihood estimator:

\[
\hat\theta_{\mathrm{MLE}}
=
\arg\max_\theta \ell(\theta)
\]

Score function:

\[
U(\theta)
=
\frac{\partial \ell(\theta)}{\partial\theta}
\]

Fisher information, dalam kondisi regularitas yang sesuai:

\[
I_n(\theta)
=
\mathbb E_\theta
\left[
U(\theta)U(\theta)^T
\right]
\]

Untuk model regular yang memenuhi syarat, asymptotic covariance dari MLE berkaitan dengan invers Fisher information.

Namun, hasil asimtotik tersebut tidak otomatis berlaku pada semua model, terutama ketika parameter berada di boundary atau identifiability bermasalah.

## Advanced Concepts

### A. Probability vs Likelihood

Probability mempertimbangkan kemungkinan outcome ketika parameter diketahui.

Likelihood mempertimbangkan seberapa didukung nilai parameter yang berbeda oleh data yang diamati.

Keduanya berhubungan, tetapi mempunyai interpretasi matematis berbeda.

### B. Exponential Family

Pelajari model yang dapat ditulis dalam bentuk:

\[
p(x\mid\theta)
=
h(x)\exp\left(
\eta(\theta)^T T(x)-A(\theta)
\right)
\]

Pahami hubungan antara:

- Natural parameters
- Sufficient statistics
- Partition functions
- Expectation parameters
- Fisher information
- Generalized linear models

### C. Likelihood-Based Inference

Untuk menguji hipotesis yang sesuai dengan model regular, likelihood-ratio statistic dapat ditulis sebagai:

\[
\Lambda
=
2\left[
\ell(\hat\theta)
-
\ell(\hat\theta_0)
\right]
\]

Dalam kondisi regular, statistik tersebut memiliki pendekatan distribusi asimtotik tertentu.

Jangan mengasumsikan pendekatan chi-square berlaku secara universal.

## Algorithm Mechanism

- Gradient-based MLE
- Newton-Raphson
- Fisher scoring
- Iteratively reweighted least squares
- Profile likelihood
- EM algorithm
- Constrained optimization
- Penalized likelihood
- Likelihood-ratio testing

## From Scratch

1. Implement likelihood untuk Bernoulli dan Binomial.
2. Derive dan implement MLE untuk beberapa distribution families.
3. Implement Newton-Raphson untuk optimasi parameter.
4. Compare Newton-Raphson dan gradient ascent.
5. Implement confidence intervals berbasis likelihood.
6. Implement profile likelihood untuk model sederhana.
7. Build a simple EM algorithm untuk Gaussian mixture.
8. Compare closed-form estimator dengan numerical estimator.

## Experiment

### Likelihood Geometry

Buat likelihood surface untuk parameter yang berbeda.

Amati:

- Flat likelihood
- Sharp likelihood
- Multiple local maxima
- Parameter correlation
- Weak identifiability
- Sensitivity to sample size

Hubungkan curvature likelihood dengan estimasi ketidakpastian parameter.

## Diagnostics

- Convergence of optimizer
- Gradient norm
- Hessian conditioning
- Likelihood profile
- Identifiability
- Boundary estimates
- Multiple local optima
- Parameter sensitivity
- Standard errors
- Model goodness-of-fit

## Failure Modes

- Local optimum mistaken for global optimum
- Non-identifiable parameters
- Infinite or undefined MLE
- Complete separation in logistic regression
- Incorrect likelihood under dependent observations
- Invalid asymptotic approximations
- Numerical instability in Hessian inversion
- Treating optimizer convergence as proof that the model is correct

## Case — Rare-Event Risk Estimation

Untuk kejadian yang sangat jarang, estimasi probabilitas dapat sensitif terhadap jumlah observasi dan model yang digunakan.

```text
Rare Events
    |
    v
Likelihood Structure
    |
    v
Parameter Identifiability
    |
    v
MLE / Penalized MLE / Bayesian Model
    |
    v
Uncertainty Assessment
    |
    v
Risk Interpretation
```

Tujuan:

Membedakan estimasi numerik dari evidence statistik yang cukup untuk mendukung suatu kesimpulan.

---

# DAY 5 — Frequentist Inference, Hypothesis Testing & Multiplicity

## Objective

Memahami bagaimana confidence intervals dan hypothesis tests dibangun dari sampling distributions, serta bagaimana mencegah kesalahan interpretasi dan false discoveries.

## Core

105. Null and alternative hypotheses
106. Test statistics
107. Null distributions
108. Type I error
109. Type II error
110. Statistical power
111. Significance levels
112. P-values
113. Confidence intervals
114. Confidence regions
115. Likelihood-ratio tests
116. Wald tests
117. Score tests
118. Exact tests
119. Asymptotic tests
120. One-sided and two-sided tests
121. Equivalence testing
122. Non-inferiority testing
123. Multiple hypothesis testing
124. Family-wise error rate
125. False discovery rate
126. Sequential testing
127. Optional stopping
128. Selective inference
129. Post-selection inference
130. Statistical versus practical significance

## Mathematics

Type I error:

\[
\alpha
=
P(\text{Reject }H_0\mid H_0\text{ is true})
\]

Type II error untuk alternatif tertentu:

\[
\beta
=
P(\text{Fail to reject }H_0\mid H_1\text{ is true})
\]

Statistical power:

\[
\text{Power}=1-\beta
\]

P-value secara konseptual:

\[
p
=
P(\text{Test statistic at least as extreme as observed}
\mid H_0)
\]

Confidence interval dengan confidence level \(1-\alpha\) adalah prosedur yang, di bawah asumsi yang sesuai, mempunyai coverage jangka panjang sebesar \(1-\alpha\).

Confidence level bukan probabilitas bahwa parameter fixed berada dalam interval setelah interval tersebut dihitung, dalam interpretasi frequentist standar.

## Advanced Concepts

### A. Confidence Interval vs Prediction Interval

Confidence interval berfokus pada quantity seperti parameter atau conditional mean.

Prediction interval mempertimbangkan ketidakpastian outcome baru, termasuk variasi outcome yang tidak dijelaskan model.

Keduanya menjawab pertanyaan berbeda.

### B. Multiple Testing

Ketika banyak hipotesis diuji secara bersamaan, peluang mendapatkan false positive dapat meningkat.

Pelajari:

- Bonferroni
- Holm procedure
- Benjamini-Hochberg
- False discovery rate
- Family-wise error control
- Empirical Bayes approaches to multiple testing
- Selective inference

### C. Sequential Analysis

Pahami perbedaan antara:

- Fixed-horizon testing
- Group-sequential designs
- Alpha-spending procedures
- Sequential probability ratio tests
- Always-valid inference
- Adaptive experimentation

P-value standar tidak otomatis tetap valid ketika proses pengumpulan data dihentikan berdasarkan hasil sementara.

## Algorithm Mechanism

- Test statistic construction
- Critical-value testing
- P-value calculation
- Bootstrap confidence intervals
- Multiple-testing correction
- Power analysis
- Sequential monitoring
- Equivalence testing

## From Scratch

1. Implement a two-sample permutation test.
2. Implement a bootstrap confidence interval.
3. Simulate Type I error under the null hypothesis.
4. Estimate power under multiple effect sizes.
5. Implement Bonferroni correction.
6. Implement Benjamini-Hochberg procedure.
7. Demonstrate the effect of repeated testing.
8. Compare fixed-horizon dan sequential decision procedures.

## Experiment

### P-Value Misinterpretation

Simulasikan ribuan eksperimen ketika null hypothesis benar.

Amati:

```text
One Pre-Specified Test
        |
        v
Controlled Type I Error

Many Uncorrected Tests
        |
        v
Accumulation of False Discoveries
```

Tujuannya adalah memahami perbedaan antara statistical evidence, multiplicity, dan reproducibility.

## Diagnostics

- Effect size
- Confidence interval width
- Statistical power
- Type I error
- Multiple-testing burden
- Dependence between tests
- Pre-specified analysis plan
- Stopping rule
- Practical importance
- Robustness across analysis choices

## Failure Modes

- Interpreting p-values as posterior probabilities
- Treating non-significance as proof of no effect
- Confusing confidence intervals with prediction intervals
- Ignoring multiplicity
- Using optional stopping with invalid inference
- Reporting only significant results
- Confusing statistical significance with practical importance
- Using a test without checking its assumptions

## Case — A/B Testing for an Educational Platform

Misalkan produk personalization diuji terhadap baseline.

Outcome:

- Learning completion
- Retention
- Time to mastery
- Assessment score
- User engagement

Analisis harus membedakan:

1. Apakah perubahan terdeteksi?
2. Seberapa besar estimasi efek?
3. Seberapa tidak pasti estimasi tersebut?
4. Apakah efek cukup penting untuk keputusan produk?
5. Apakah hasil konsisten pada subgroup yang telah direncanakan?
6. Apakah multiplicity atau peeking memengaruhi validitas hasil?

---

# DAY 6 — Bootstrap, Permutation, Nonparametric & Robust Statistics

## Objective

Memahami metode statistik ketika distribusi parametrik klasik tidak sesuai, sulit dipertahankan, atau tidak memberikan inferensi yang memadai.

## Core

131. Nonparametric inference
132. Distribution-free procedures
133. Bootstrap
134. Parametric bootstrap
135. Nonparametric bootstrap
136. Cluster bootstrap
137. Block bootstrap
138. Wild bootstrap
139. Jackknife
140. Permutation tests
141. Randomization inference
142. Rank-based methods
143. Wilcoxon procedures
144. Mann-Whitney test
145. Kruskal-Wallis test
146. Empirical distribution functions
147. Kernel density estimation
148. Bandwidth selection
149. Quantile regression
150. Robust location estimators
151. Robust scale estimators
152. M-estimators
153. Huber loss
154. Influence functions
155. Breakdown point
156. Robust covariance estimation
157. Sandwich standard errors
158. Semiparametric inference foundations

## Mathematics

Empirical distribution function:

\[
\hat F_n(x)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}(X_i\le x)
\]

Bootstrap approximation:

```text
Observed Dataset
      |
      v
Resample According to Sampling Design
      |
      v
Recompute Statistic
      |
      v
Repeat Many Times
      |
      v
Approximate Sampling Distribution
```

Huber loss, untuk residual \(r\):

\[
\rho_\delta(r)
=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta\\
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta
\end{cases}
\]

## Advanced Concepts

### A. Bootstrap Is Not One Universal Procedure

Pilih resampling scheme sesuai struktur data.

- IID observations: ordinary bootstrap dapat sesuai.
- Clustered observations: pertimbangkan cluster bootstrap.
- Time series: pertimbangkan block bootstrap.
- Randomized experiments: randomization inference mengikuti desain eksperimen.
- Heteroscedastic regression: wild bootstrap dapat relevan.

Mengacak observasi secara sembarang dapat menghancurkan dependence structure dan menghasilkan uncertainty estimates yang tidak valid.

### B. Robustness

Pahami:

- Sensitivity to outliers
- Influence functions
- Breakdown point
- Heavy-tailed observations
- Heteroscedasticity
- Robust standard errors
- Robust regression
- Robust covariance estimation

Robust standard errors dan robust estimators tidak menyelesaikan semua bentuk model misspecification.

## Algorithm Mechanism

- Bootstrap resampling
- Permutation tests
- Jackknife estimation
- Kernel smoothing
- Quantile regression
- M-estimation
- Huber regression
- Median absolute deviation
- Sandwich variance estimation

## From Scratch

1. Implement percentile bootstrap CI.
2. Implement basic BCa bootstrap conceptually and computationally.
3. Implement permutation tests.
4. Implement jackknife standard errors.
5. Implement robust location estimators.
6. Implement Huber loss and optimization.
7. Compare ordinary and robust regression.
8. Implement cluster bootstrap.
9. Implement block bootstrap untuk time series sederhana.

## Experiments

### Experiment 1 — Outlier Contamination

Bandingkan:

```text
Mean vs Median
OLS vs Robust Regression
Standard Deviation vs MAD
Ordinary SE vs Appropriate Robust SE
```

Ukur bias, variance, coverage, dan sensitivity terhadap contaminating observations.

### Experiment 2 — Bootstrap Under Dependence

Gunakan time series dengan autocorrelation.

Bandingkan ordinary bootstrap dan block bootstrap.

Tujuannya adalah mengamati akibat menghancurkan dependence structure saat resampling.

## Failure Modes

- Bootstrap under non-identifiability
- Invalid bootstrap under nonregular problems
- IID bootstrap for dependent data
- Ignoring cluster structure
- Treating nonparametric methods as assumption-free
- Using rank tests without understanding their null hypotheses
- Estimating bandwidth without sensitivity analysis
- Assuming robust methods are automatically more efficient

## Case — Financial Return Risk

Return finansial dapat memiliki outlier, skewness, volatility clustering, dan tail risk.

Gunakan:

- Quantile estimates
- Robust scale
- Bootstrap uncertainty
- Block resampling
- Tail-focused diagnostics

Tujuan:

Memahami bagaimana pilihan estimator dan resampling method memengaruhi estimasi risiko.

---

# DAY 7 — Bayesian Statistics, Hierarchical Inference & Computational Bayes

## Objective

Memahami Bayesian inference sebagai prosedur untuk menggabungkan likelihood dengan prior information guna menghasilkan posterior distributions.

## Core

159. Prior distributions
160. Likelihood
161. Posterior distributions
162. Posterior odds
163. Posterior predictive distributions
164. Conjugate priors
165. Informative priors
166. Weakly informative priors
167. Prior predictive checks
168. Posterior predictive checks
169. Credible intervals
170. Highest posterior density regions
171. Bayesian decision theory
172. Hierarchical Bayesian models
173. Partial pooling
174. Empirical Bayes
175. Markov chain Monte Carlo
176. Metropolis-Hastings
177. Gibbs sampling
178. Hamiltonian Monte Carlo
179. No-U-Turn Sampler
180. Effective sample size
181. Potential scale reduction diagnostics
182. MCMC autocorrelation
183. Posterior geometry
184. Sensitivity to priors
185. Model comparison
186. Bayesian model averaging

## Mathematics

Bayes' theorem:

\[
p(\theta\mid y)
=
\frac{p(y\mid\theta)p(\theta)}{p(y)}
\]

Posterior predictive distribution:

\[
p(y_{\text{new}}\mid y)
=
\int p(y_{\text{new}}\mid\theta)
p(\theta\mid y)\,d\theta
\]

Posterior expected loss for a decision \(a\):

\[
\mathbb E[L(\theta,a)\mid y]
\]

## Advanced Concepts

### A. Prior Predictive Reasoning

Periksa model sebelum mengamati outcome.

Pertanyaan:

- Apakah prior memungkinkan nilai yang tidak realistis?
- Apakah prior predictive distribution sesuai dengan pengetahuan domain?
- Apakah parameterisasi menghasilkan perilaku yang masuk akal?

### B. Posterior Predictive Checking

Setelah fitting model:

1. Generate replicated data dari posterior predictive distribution.
2. Bandingkan data replicated dengan data aktual.
3. Identifikasi pola yang tidak dapat dijelaskan model.
4. Lakukan model revision jika diperlukan.

### C. Hierarchical Models

Contoh struktur:

\[
y_{ij}\sim p(y_{ij}\mid\theta_j)
\]

\[
\theta_j\sim p(\theta_j\mid\phi)
\]

Parameter individual dapat berbagi informasi melalui population-level distribution.

Pelajari kapan pooling penuh, tidak ada pooling, atau partial pooling lebih tepat.

## Algorithm Mechanism

- Conjugate Bayesian updating
- Grid approximation
- Numerical integration
- Rejection sampling
- Importance sampling
- Metropolis-Hastings
- Gibbs sampling
- Hamiltonian Monte Carlo
- Variational inference
- Posterior predictive simulation

## Diagnostics

- Trace plots
- Chain mixing
- Effective sample size
- \(\hat R\)
- Autocorrelation
- Divergent transitions
- Monte Carlo standard error
- Prior sensitivity
- Posterior predictive discrepancy
- Identifiability
- Multimodality
- Posterior correlations

## From Scratch

1. Implement Beta-Binomial updating.
2. Implement Normal-Normal updating.
3. Implement grid-based posterior estimation.
4. Implement Metropolis-Hastings untuk model sederhana.
5. Compare posterior samples with numerical integration.
6. Build a hierarchical model simulation.
7. Demonstrate partial pooling.
8. Perform prior predictive checks.
9. Perform posterior predictive checks.

Gunakan library seperti PyMC dan ArviZ untuk workflow profesional, tetapi tetap implementasikan model kecil dari nol untuk memahami mekanismenya.

## Experiment

### Prior Sensitivity

Bandingkan posterior inference dengan beberapa prior yang masuk akal.

Periksa:

- Posterior location
- Posterior spread
- Extreme parameter values
- Predictive performance
- Stability of substantive conclusions

### Partial Pooling

Simulasikan beberapa kelompok dengan jumlah observasi yang berbeda.

Bandingkan:

```text
No Pooling
    |
    v
Independent Group Estimates

Complete Pooling
    |
    v
One Shared Estimate

Partial Pooling
    |
    v
Group Estimates + Shared Population Information
```

## Failure Modes

- Treating priors as irrelevant
- Using implausible prior scales
- Failing to check convergence
- Ignoring divergent transitions
- Confusing posterior uncertainty with Monte Carlo error
- Treating credible intervals as confidence intervals
- Overinterpreting posterior probabilities under model misspecification
- Assuming MCMC output automatically implies valid inference
- Ignoring weak identifiability and posterior multimodality

## Case — Personalized Learning

Misalkan setiap learner memiliki estimasi probabilitas menyelesaikan materi.

Sebagian learner memiliki banyak histori aktivitas, sedangkan lainnya memiliki data yang sangat sedikit.

Gunakan hierarchical Bayesian modeling untuk:

```text
Individual Learning History
        |
        v
Individual Parameter Estimate
        |
        v
Population-Level Distribution
        |
        v
Partial Pooling
        |
        v
Personalized Probability Estimate
        |
        v
Posterior Predictive Uncertainty
```

Tujuan:

Mengembangkan pemahaman statistik untuk personalization tanpa menganggap estimasi dari data yang sedikit memiliki presisi setara dengan estimasi dari data yang banyak.

---

# DAY 8 — Regression Inference, Generalized Linear Models & Model Diagnostics

## Objective

Mempelajari regression sebagai statistical model, bukan hanya fungsi yang menghasilkan prediksi.

Fokus hari ini adalah inferensi, error structure, model diagnostics, estimation uncertainty, dan konsekuensi asumsi distribusi.

## Core

187. Statistical regression models
188. Conditional expectation
189. Conditional variance
190. Linear model assumptions
191. Gaussian linear models
192. Heteroscedasticity
193. Autocorrelated errors
194. Model misspecification
195. Residual diagnostics
196. Leverage
197. Influence
198. Cook's distance
199. Heteroscedasticity-consistent covariance estimators
200. Generalized linear models
201. Exponential-family regression
202. Link functions
203. Logistic regression
204. Poisson regression
205. Negative binomial regression
206. Gamma regression
207. Quasi-likelihood
208. Overdispersion
209. Separation
210. Deviance
211. Likelihood-based model comparison
212. Calibration
213. Prediction intervals
214. Model uncertainty
215. Penalized regression inference

## Mathematics

Conditional mean model:

\[
\mathbb E[Y\mid X]
=
m(X;\theta)
\]

Linear model:

\[
Y=X\beta+\varepsilon
\]

Generalized linear model:

\[
g(\mathbb E[Y\mid X])
=
X\beta
\]

Logistic regression:

\[
\operatorname{logit}\left(
P(Y=1\mid X)
\right)
=
X\beta
\]

Poisson regression:

\[
\log(\mathbb E[Y\mid X])
=
X\beta
\]

## Statistical Assumptions

Bedakan asumsi yang dibutuhkan untuk:

- Unbiasedness
- Consistency
- Standard error estimation
- Exact finite-sample inference
- Asymptotic inference
- Prediction
- Causal interpretation

Contohnya, normality of errors bukan asumsi yang sama dengan zero conditional mean.

Homoscedasticity juga tidak identik dengan independence.

## Advanced Topics

### A. Heteroscedasticity

Bandingkan ordinary standard errors dengan variance estimators yang konsisten terhadap heteroscedasticity di bawah kondisi yang sesuai.

Secara skematis:

\[
\widehat{\operatorname{Var}}(\hat\beta)
=
(X^TX)^{-1}
X^T\hat\Omega X
(X^TX)^{-1}
\]

Bentuk \(\hat\Omega\) bergantung pada variance estimator yang digunakan.

### B. GLM Diagnostics

Untuk GLM, periksa:

- Link function
- Conditional mean
- Conditional variance
- Deviance residuals
- Pearson residuals
- Overdispersion
- Calibration
- Separation
- Influence
- Extrapolation

### C. Statistical vs Predictive Modeling

Model dengan predictive performance tinggi tidak otomatis memiliki parameter yang dapat ditafsirkan secara causal.

Model dengan koefisien yang signifikan juga belum tentu mempunyai predictive performance tinggi.

## Algorithm Mechanism

- Maximum likelihood for regression
- Iteratively reweighted least squares
- Robust covariance estimation
- Quasi-likelihood estimation
- Profile likelihood
- Residual analysis
- Calibration analysis
- Penalized likelihood

## From Scratch

1. Implement linear regression likelihood.
2. Implement logistic regression dengan gradient-based optimization.
3. Implement Poisson regression sederhana.
4. Compare ordinary dan robust standard errors.
5. Implement deviance calculations.
6. Implement residual diagnostics.
7. Detect influential observations.
8. Simulate heteroscedasticity dan evaluate inferential coverage.
9. Demonstrate complete separation in logistic regression.

## Experiments

### Experiment 1 — Heteroscedasticity

Bangun data dengan:

\[
\operatorname{Var}(Y\mid X=x)
\]

yang berubah terhadap \(x\).

Bandingkan:

- Coefficient estimates
- Ordinary standard errors
- Heteroscedasticity-consistent standard errors
- Confidence interval coverage

### Experiment 2 — Model Misspecification

Generate data menggunakan nonlinear mean structure.

Fit linear model dan periksa:

- Residual patterns
- Predictive errors
- Coefficient interpretation
- Confidence intervals
- Sensitivity to model specification

## Failure Modes

- Treating residual normality as the only relevant diagnostic
- Using ordinary standard errors under incompatible variance assumptions
- Ignoring dependence
- Confusing an association coefficient with a causal effect
- Interpreting logistic coefficients as direct probability changes
- Ignoring overdispersion in count models
- Ignoring separation in logistic regression
- Extrapolating outside observed covariate support
- Selecting models only through p-values

## Case — Healthcare Claims Review

Gunakan data fiktif dengan outcome seperti probability of documentation discrepancy.

Tujuan model bukan otomatis menggantikan verifikator.

Analisis statistik seharusnya mencakup:

1. Outcome definition.
2. Sampling and label quality.
3. Logistic regression or another appropriate model.
4. Calibration.
5. Uncertainty in estimated effects.
6. Subgroup performance.
7. Influence of rare events.
8. False-positive and false-negative consequences.
9. External validation.

---

# DAY 9 — Hierarchical Models, Multilevel Statistics & Dependent Observations

## Objective

Memahami model ketika data memiliki struktur kelompok atau dependence yang tidak dapat diabaikan.

## Core

216. Independent vs dependent observations
217. Clustered data
218. Repeated measurements
219. Longitudinal dependence
220. Random intercept models
221. Random slope models
222. Multilevel regression
223. Partial pooling
224. Intraclass correlation
225. Variance components
226. Hierarchical generalized linear models
227. Generalized linear mixed models
228. Generalized estimating equations
229. Population-averaged effects
230. Subject-specific effects
231. Cluster-robust inference
232. Crossed random effects
233. Nested random effects
234. Bayesian multilevel models
235. Shrinkage across groups
236. Group-level confounding
237. Aggregation bias
238. Simpson's paradox

## Mathematics

Random intercept model:

\[
Y_{ij}
=
X_{ij}^T\beta
+
u_j
+
\varepsilon_{ij}
\]

Dengan:

- \(i\): observation
- \(j\): group
- \(u_j\): group-specific random effect
- \(\varepsilon_{ij}\): observation-level error

Intraclass correlation untuk model random-intercept sederhana:

\[
\operatorname{ICC}
=
\frac{\sigma_u^2}
{\sigma_u^2+\sigma_\varepsilon^2}
\]

Interpretasinya bergantung pada model dan struktur variance yang digunakan.

## Advanced Concepts

### A. Partial Pooling

Pelajari hubungan multilevel models dengan shrinkage dan empirical Bayes.

### B. Population-Averaged vs Subject-Specific Effects

Generalized estimating equations dan generalized linear mixed models umumnya menargetkan parameter dengan interpretasi yang berbeda.

Pemilihan metode harus mengikuti estimand yang ingin dipelajari.

### C. Aggregation Bias

Periksa mengapa suatu hubungan pada tingkat individu dapat berbeda dari hubungan pada tingkat kelompok.

Jangan menganggap group-level correlation dapat digunakan sebagai pengganti individual-level association.

## Algorithm Mechanism

- Mixed-effects maximum likelihood
- Restricted maximum likelihood
- Bayesian hierarchical inference
- GEE estimation
- Cluster-robust variance estimation
- Variance component estimation
- Partial pooling

## Diagnostics

- Cluster count
- Cluster size imbalance
- Intraclass correlation
- Random-effect distribution
- Variance component boundaries
- Within-group residual dependence
- Between-group heterogeneity
- Model convergence
- Sensitivity to group specification

## From Scratch

1. Simulate clustered data.
2. Estimate the effect of intraclass correlation on standard errors.
3. Implement a simple random-intercept likelihood.
4. Compare pooled and unpooled estimates.
5. Simulate Simpson's paradox.
6. Compare conditional dan marginal interpretations.
7. Evaluate consequences of ignoring cluster structure.

## Experiment

### Independence Violation

Bangun data dengan ukuran cluster yang bervariasi.

Bandingkan:

```text
Naive Independent Model
        |
        v
Potentially Misestimated Uncertainty

Cluster-Aware / Multilevel Model
        |
        v
Explicit Dependence Structure
```

## Failure Modes

- Treating repeated measures as independent
- Assuming all groups are identical
- Overinterpreting variance components
- Ignoring few-cluster problems
- Confusing marginal and conditional effects
- Using cluster-robust standard errors without considering cluster count
- Ignoring informative cluster size
- Assuming random effects automatically remove confounding

## Case — Personalized Learner Modeling

Data learner biasanya terdiri atas banyak interaksi per individu.

Observasi dari learner yang sama cenderung saling berhubungan.

Gunakan multilevel reasoning untuk memisahkan:

- Population-level behavior
- Learner-specific heterogeneity
- Lesson-level differences
- Repeated-measurement effects
- Sampling uncertainty

---

# DAY 10 — Missing Data, Measurement Error, Selection Bias & Sensitivity Analysis

## Objective

Memahami bahwa masalah statistik tidak selalu berasal dari estimator.

Sering kali masalah utamanya berasal dari proses pengumpulan, pengukuran, seleksi, atau hilangnya data.

## Core

239. Missing data mechanisms
240. MCAR
241. MAR
242. MNAR
243. Complete-case analysis
244. Inverse probability weighting
245. Multiple imputation
246. Imputation models
247. Missingness indicators
248. Measurement error
249. Misclassification
250. Selection bias
251. Nonresponse bias
252. Collider bias
253. Informative censoring
254. Latent measurement models
255. Reliability and validity
256. Measurement invariance
257. Sensitivity analysis
258. Quantitative bias analysis
259. Tipping-point analysis
260. Pattern-mixture models
261. Selection models
262. Missing-not-at-random sensitivity analysis

## Statistical Mechanisms

### A. Missing Data

MCAR:

Missingness tidak bergantung pada observed maupun unobserved data, di bawah formal definition yang sesuai.

MAR:

Missingness dapat dijelaskan oleh observed variables setelah conditioning yang relevan.

MNAR:

Missingness masih bergantung pada unobserved information setelah mempertimbangkan observed variables.

MAR dan MNAR bukan label yang dapat ditentukan hanya dengan melihat bahwa sebagian nilai hilang.

### B. Multiple Imputation

Secara konseptual:

```text
Observed Data
      |
      v
Imputation Model
      |
      v
Multiple Plausible Completed Datasets
      |
      v
Analysis on Each Dataset
      |
      v
Combine Estimates and Uncertainty
```

Multiple imputation harus mencerminkan analysis model, missingness assumptions, dan struktur data yang relevan.

### C. Measurement Error

Observed measurement dapat direpresentasikan secara sederhana:

\[
W=X+U
\]

di mana \(X\) adalah nilai yang ingin diketahui dan \(U\) adalah measurement error.

Formula sederhana ini tidak mencakup semua jenis measurement-error mechanisms.

Pelajari classical measurement error, differential error, misclassification, dan latent measurement models.

## Advanced Concepts

- Inverse probability weighting
- Multiple imputation
- Selection models
- Pattern-mixture models
- Sensitivity parameters
- Quantitative bias analysis
- Measurement invariance
- Differential misclassification
- Missingness in longitudinal data
- Informative observation processes

## Diagnostics

- Missingness patterns
- Missingness by subgroup
- Distribution changes between complete and incomplete cases
- Imputation diagnostics
- Extreme inverse probability weights
- Positivity violations
- Measurement reliability
- Model sensitivity
- Sensitivity of conclusions to unverifiable assumptions

## From Scratch

1. Simulate MCAR, MAR, and MNAR processes.
2. Compare complete-case analysis with an appropriate imputation approach.
3. Implement simple inverse probability weighting.
4. Simulate attenuation from classical measurement error.
5. Quantify the effect of misclassification.
6. Build a sensitivity analysis across plausible assumptions.
7. Demonstrate selection bias using a causal diagram.

## Failure Modes

- Assuming missingness is MAR without justification
- Treating imputed values as observed facts
- Ignoring uncertainty from imputation
- Using unstable inverse probability weights
- Confusing missingness indicators with a universal solution
- Ignoring differential measurement error
- Treating complete-case analysis as automatically unbiased
- Claiming to identify MNAR behavior from observed data alone

## Case — Healthcare Data Quality

Misalkan informasi dokumentasi klinis tidak selalu tercatat untuk seluruh pasien.

Analisis harus mempertimbangkan apakah missing documentation berkaitan dengan:

- Patient characteristics
- Disease severity
- Healthcare facility
- Workflow
- Outcome
- Observation process

Tujuan:

Memahami bagaimana data missing dan proses pencatatan dapat memengaruhi estimasi risiko dan kesimpulan penelitian.

---

# DAY 11 — Causal Inference, Potential Outcomes & Causal Identification

## Objective

Membangun kemampuan membedakan prediction dan association dari causal inference.

Ini adalah salah satu bagian paling penting dalam advanced statistics.

## Core

263. Association vs causation
264. Potential outcomes framework
265. Average treatment effect
266. Average treatment effect on the treated
267. Conditional average treatment effect
268. Consistency
269. Exchangeability
270. Positivity
271. Stable unit treatment value assumptions
272. Treatment assignment mechanisms
273. Confounding
274. Mediators
275. Colliders
276. Directed acyclic graphs
277. Backdoor adjustment
278. Standardization
279. G-computation
280. Propensity scores
281. Inverse probability weighting
282. Matching
283. Doubly robust estimation
284. Augmented inverse probability weighting
285. Instrumental variables
286. Regression discontinuity
287. Difference-in-differences
288. Synthetic controls
289. Longitudinal causal inference
290. Time-varying confounding
291. Marginal structural models
292. Targeted learning
293. Sensitivity analysis for unmeasured confounding

## Mathematics

Potential outcomes:

\[
Y(1),\qquad Y(0)
\]

Average treatment effect:

\[
ATE
=
\mathbb E[Y(1)-Y(0)]
\]

Hasil potensial untuk satu unit tidak secara bersamaan teramati dalam kedua kondisi.

Masalah tersebut merupakan bagian fundamental dari causal inference.

## Identification Assumptions

### Consistency

Observed outcome harus sesuai dengan potential outcome untuk treatment yang benar-benar diterima, sesuai definisi treatment dan estimand.

### Conditional Exchangeability

Dalam bentuk tertentu:

\[
(Y(1),Y(0))\perp A\mid X
\]

Artinya, setelah conditioning pada covariates yang sesuai, treatment assignment dianggap exchangeable terhadap potential outcomes.

Asumsi ini tidak otomatis dijamin oleh regression atau propensity-score adjustment.

### Positivity

Setiap treatment yang relevan harus mempunyai probabilitas positif pada covariate profiles dalam target population.

### Interference

Perlu diperiksa apakah treatment pada satu unit dapat memengaruhi outcome unit lain.

## Algorithm Mechanism

- Backdoor adjustment
- G-computation
- Propensity score estimation
- Inverse probability weighting
- Matching
- AIPW
- Instrumental-variable estimation
- Regression discontinuity
- Difference-in-differences
- Synthetic control
- Marginal structural models
- Targeted maximum likelihood estimation

## Advanced Concepts

### A. Propensity Score

Propensity score:

\[
e(X)=P(A=1\mid X)
\]

Pahami:

- Common support
- Positivity
- Weight instability
- Covariate balance
- Model misspecification
- Effective sample size
- Treatment effect target

Model propensity score dengan classification accuracy tinggi tidak otomatis menghasilkan covariate balance yang baik.

### B. Doubly Robust Estimation

Dalam setting yang sesuai, estimator doubly robust menggabungkan outcome model dan treatment model.

Untuk ATE, salah satu bentuk estimasi augmented inverse probability weighting menggunakan kontribusi:

\[
\hat\psi
=
\frac{1}{n}\sum_{i=1}^n
\left[
\hat m_1(X_i)-\hat m_0(X_i)
+
\frac{A_i(Y_i-\hat m_1(X_i))}{\hat e(X_i)}
-
\frac{(1-A_i)(Y_i-\hat m_0(X_i))}
{1-\hat e(X_i)}
\right]
\]

Interpretasi dan jaminan estimasi bergantung pada identification assumptions, regularity conditions, dan estimand yang dituju.

### C. Causal Identification vs Estimation

Bedakan:

```text
CAUSAL QUESTION
      |
      v
TARGET ESTIMAND
      |
      v
IDENTIFICATION ASSUMPTIONS
      |
      v
IDENTIFIED FUNCTIONAL
      |
      v
STATISTICAL ESTIMATION
      |
      v
UNCERTAINTY QUANTIFICATION
      |
      v
SENSITIVITY ANALYSIS
```

Estimator yang canggih tidak dapat memperbaiki estimand yang tidak terdefinisi atau identification assumptions yang tidak masuk akal.

## Diagnostics

- Covariate balance
- Overlap
- Positivity
- Effective sample size
- Extreme weights
- Model specification
- Pre-treatment trends
- Instrument validity arguments
- Regression discontinuity bandwidth sensitivity
- Unmeasured confounding sensitivity
- Negative controls
- Placebo tests
- Treatment heterogeneity

## From Scratch

1. Simulate treatment assignment with confounding.
2. Compare naive association with randomized treatment effects.
3. Implement standardization.
4. Implement propensity score weighting.
5. Diagnose covariate balance before and after weighting.
6. Implement simple AIPW estimation.
7. Simulate positivity violations.
8. Demonstrate collider bias.
9. Simulate difference-in-differences with and without parallel trends.
10. Perform a basic unmeasured-confounding sensitivity analysis.

## Experiments

### Experiment 1 — Confounding

```text
Confounder
    |       \
    v        v
Treatment -> Outcome
```

Bangun data ketika treatment tidak diacak.

Bandingkan:

- Naive association
- Stratified estimates
- Regression adjustment
- Inverse probability weighting
- Randomized benchmark

### Experiment 2 — Positivity Violation

Bangun situasi ketika beberapa covariate profiles hampir selalu menerima treatment yang sama.

Amati:

- Extreme weights
- Effective sample size
- Estimate instability
- Sensitivity to trimming or alternative target populations

## Failure Modes

- Treating predictive accuracy as causal validity
- Adjusting for colliders
- Ignoring unmeasured confounding
- Ignoring positivity
- Interpreting matching as a guarantee of causal validity
- Using instrumental variables without defensible assumptions
- Applying difference-in-differences without assessing parallel trends
- Ignoring treatment versions
- Ignoring interference
- Confusing an identified estimand with its estimate

## Case — Personalized Learning Intervention

Pertanyaan:

Apakah adaptive learning benar-benar meningkatkan mastery dibandingkan metode standar?

Bandingkan:

- Randomized experiment
- Observational comparison
- Covariate adjustment
- Propensity weighting
- Doubly robust estimation
- Sensitivity analysis

Pisahkan peningkatan yang mungkin disebabkan oleh intervensi dari perbedaan awal antarlearner.

---

# DAY 12 — Experimental Design, A/B Testing & Adaptive Experiments

## Objective

Memahami bahwa validitas statistik dimulai dari desain data collection, bukan setelah data terkumpul.

## Core

294. Scientific hypothesis
295. Estimand specification
296. Randomized experiments
297. Blocking
298. Stratified randomization
299. Factorial design
300. Cluster randomization
301. Crossover design
302. Repeated-measures experiments
303. Randomization inference
304. Sample size calculation
305. Statistical power
306. Minimum detectable effect
307. Effect-size specification
308. Multiple endpoints
309. Interim analysis
310. Group-sequential designs
311. Adaptive experiments
312. Multi-armed bandits
313. Experiment contamination
314. Noncompliance
315. Attrition
316. Covariate adjustment
317. CUPED and variance reduction
318. Heterogeneous treatment effects
319. Experiment replication
320. Operational versus statistical success

## Experiment Design Logic

```text
Research Question
      |
      v
Target Estimand
      |
      v
Experimental Unit
      |
      v
Assignment Mechanism
      |
      v
Outcome Definition
      |
      v
Sample Size / Power
      |
      v
Pre-Specified Analysis Plan
      |
      v
Data Collection
      |
      v
Statistical Analysis
      |
      v
Decision
```

## Advanced Concepts

### A. Unit of Randomization

Unit of randomization harus sesuai dengan mekanisme intervensi.

Jika treatment diberikan kepada satu kelas, tetapi analisis menganggap setiap siswa independen tanpa memperhatikan cluster, inferensi dapat menjadi tidak valid.

### B. Adaptive Experimentation

Adaptive designs mengubah keputusan eksperimen berdasarkan data interim.

Metode ini membutuhkan inferential procedure yang sesuai dengan aturan adaptasinya.

### C. Variance Reduction

Pelajari bagaimana penggunaan baseline covariates atau variance-reduction techniques dapat meningkatkan presisi estimasi ketika asumsi yang diperlukan terpenuhi.

### D. Treatment Effect Heterogeneity

Bedakan:

- Average treatment effect
- Conditional average treatment effect
- Subgroup effect
- Individual treatment effect prediction

Estimasi individual treatment effects umumnya menghadapi ketidakpastian yang jauh lebih besar dibanding estimasi efek rata-rata.

## Diagnostics

- Randomization balance
- Sample ratio mismatch
- Treatment compliance
- Attrition
- Outcome measurement
- Interference
- Experiment contamination
- Power
- Multiplicity
- Peeking
- Covariate balance
- Treatment effect heterogeneity
- Replication stability

## From Scratch

1. Simulate a randomized A/B test.
2. Implement random assignment.
3. Estimate difference in means.
4. Compare standard errors with randomization-based uncertainty.
5. Conduct power simulations.
6. Estimate minimum detectable effects.
7. Compare unadjusted and baseline-adjusted analysis.
8. Simulate cluster randomization.
9. Simulate sequential monitoring under the null.
10. Compare fixed-horizon and adaptive designs.

## Case — AI Personalization Experiment

Misalkan sistem rekomendasi materi baru dibandingkan dengan baseline.

Primary outcome:

- Assessment score after a fixed period.

Secondary outcomes:

- Completion rate
- Time to mastery
- Retention
- User engagement

Desain perlu mendefinisikan:

- Unit of assignment
- Target population
- Primary estimand
- Treatment duration
- Handling of missing outcomes
- Sample size
- Multiple endpoints
- Stopping rule
- Subgroup analysis
- Long-term follow-up

Jangan menganggap kenaikan engagement otomatis berarti peningkatan learning outcomes.

---

# DAY 13 — Time Series, Longitudinal Data & Survival Analysis

## Objective

Memahami statistik ketika urutan waktu, repeated observations, censoring, dan dynamic dependency memengaruhi inferensi.

## PART A — Time Series

### Core

321. Time-indexed observations
322. Stationarity
323. Autocorrelation
324. Partial autocorrelation
325. Trend and seasonality
326. AR models
327. MA models
328. ARMA models
329. ARIMA models
330. Seasonal ARIMA
331. State-space models
332. Kalman filtering
333. Innovations
334. Forecasting distributions
335. Forecast intervals
336. Conditional heteroscedasticity
337. ARCH and GARCH
338. Volatility clustering
339. Structural breaks
340. Regime switching
341. Model residual diagnostics
342. Backtesting
343. Rolling-origin evaluation
344. Time-series cross-validation

## Mathematics

Autoregressive model sederhana:

\[
Y_t
=
c+\phi Y_{t-1}+\varepsilon_t
\]

Forecasting tidak hanya membutuhkan point forecast.

Model juga perlu mengevaluasi uncertainty distribution dari forecast tersebut.

## Advanced Time-Series Concepts

- Weak stationarity
- Ergodicity
- Autocovariance
- Spectral density
- Unit roots
- Cointegration
- State-space inference
- Kalman filtering
- Structural break detection
- Conditional volatility
- Forecast calibration
- Temporal distribution shift

## Failure Modes

- Random train-test splits for temporal prediction
- Leakage from future observations
- Ignoring seasonality
- Assuming stationarity without assessment
- Ignoring autocorrelation in uncertainty estimates
- Choosing models using only in-sample fit
- Ignoring changing data-generating processes

## PART B — Survival Analysis

### Core

345. Time-to-event outcomes
346. Event times
347. Censoring
348. Right censoring
349. Left censoring
350. Interval censoring
351. Survival functions
352. Hazard functions
353. Cumulative hazard
354. Kaplan-Meier estimator
355. Nelson-Aalen estimator
356. Log-rank tests
357. Cox proportional hazards model
358. Proportional hazards assumption
359. Time-varying covariates
360. Competing risks
361. Cause-specific hazards
362. Subdistribution hazards
363. Accelerated failure-time models
364. Parametric survival models
365. Frailty models
366. Informative censoring
367. Immortal time bias
368. Restricted mean survival time

## Mathematics

Survival function:

\[
S(t)=P(T>t)
\]

Hazard function:

\[
h(t)
=
\lim_{\Delta t\to 0}
\frac{
P(t\le T<t+\Delta t\mid T\ge t)
}{
\Delta t
}
\]

Hazard merupakan instantaneous event-rate concept, bukan probabilitas langsung bahwa event terjadi sebelum waktu tertentu.

## Algorithm Mechanism

- Time-series likelihood
- ARIMA estimation
- Kalman filtering
- Forecast uncertainty estimation
- Kaplan-Meier estimation
- Cox partial likelihood
- Survival regression
- Competing-risks analysis

## Diagnostics

- Residual autocorrelation
- Forecast errors
- Forecast interval coverage
- Temporal leakage
- Structural breaks
- Proportional hazards checks
- Censoring assumptions
- Risk-set construction
- Competing-event handling
- Sensitivity to follow-up definition

## From Scratch

1. Simulate autocorrelated time series.
2. Estimate an autoregressive model.
3. Implement rolling-origin validation.
4. Build a basic Kalman-filter simulation.
5. Implement Kaplan-Meier estimation.
6. Simulate right censoring.
7. Examine bias from ignoring censoring.
8. Compare survival curves across groups.
9. Demonstrate immortal time bias.

## Case 1 — Learner Retention

Gunakan survival analysis untuk mempelajari waktu sampai learner berhenti aktif.

Pembedaan:

- Probability of remaining active
- Hazard of inactivity
- Censoring from incomplete observation
- Treatment effect on time-to-event
- Competing definitions of inactivity

## Case 2 — Financial Volatility

Gunakan time-series reasoning untuk mempelajari perubahan volatilitas dan ketidakpastian return dari waktu ke waktu.

Pisahkan forecasting performance, risk estimation, dan causal interpretation.

---

# DAY 14 — Multivariate Statistics, Latent Variables & High-Dimensional Dependence

## Objective

Memahami distribusi multivariat, covariance structures, latent representations, mixture models, dan estimasi ketika jumlah fitur besar.

Linear algebra menyediakan struktur matriksnya. Fokus di sini adalah model statistik dan interpretasi probabilistiknya.

## Core

369. Multivariate distributions
370. Multivariate normal models
371. Multivariate t-distributions
372. Covariance estimation
373. Correlation structures
374. Covariance shrinkage
375. Precision matrices
376. Partial correlations
377. Gaussian graphical models
378. Elliptical distributions
379. Copulas
380. Tail dependence
381. Principal component analysis
382. Factor analysis
383. Latent variable models
384. Measurement models
385. Mixture models
386. Latent class analysis
387. Expectation-Maximization
388. Identifiability of latent models
389. Factor rotation
390. Reliability analysis
391. Measurement invariance
392. Multivariate hypothesis testing
393. Multiple-response data
394. Missing multivariate data
395. High-dimensional covariance estimation
396. Sparse graphical models
397. False discovery control in high dimensions
398. Robust multivariate statistics

## Advanced Concepts

### A. Covariance Estimation

Sample covariance dapat menjadi tidak stabil ketika dimension tinggi dan sample size relatif kecil.

Pelajari:

- Shrinkage covariance
- Sparse precision matrices
- Robust covariance
- Regularized covariance
- Covariance conditioning
- High-dimensional concentration
- Estimation under dependence

### B. PCA vs Factor Analysis

PCA terutama membangun komponen berdasarkan variasi total dalam data.

Factor analysis menggunakan latent-factor model dengan struktur error tertentu untuk merepresentasikan covariance.

Kedua metode tersebut tidak dapat dipertukarkan secara otomatis.

### C. Gaussian Graphical Models

Precision matrix dapat digunakan untuk merepresentasikan conditional-dependence structures dalam Gaussian models.

Nonzero entries pada precision matrix mempunyai interpretasi tertentu terhadap conditional dependence di bawah asumsi model Gaussian.

### D. Mixture Models

Mixture models dapat merepresentasikan heterogenitas melalui latent group membership.

Namun, cluster yang dihasilkan tidak otomatis merupakan kategori alami yang nyata.

Periksa:

- Identifiability
- Number of components
- Label switching
- Local optima
- Component overlap
- Stability across initialization
- Model selection uncertainty

## Algorithm Mechanism

- Multivariate likelihood
- Covariance shrinkage
- Factor analysis
- PCA from a statistical perspective
- Gaussian mixture estimation
- EM algorithm
- Latent class estimation
- Graphical-model estimation
- Sparse precision estimation
- Multiple-testing correction

## Diagnostics

- Covariance conditioning
- Component stability
- Factor loading interpretation
- Measurement invariance
- Residual covariance
- Posterior predictive fit
- Mixture-component separation
- Sensitivity to initialization
- Number-of-component uncertainty
- High-dimensional false discovery rate

## From Scratch

1. Simulate a multivariate Gaussian distribution.
2. Estimate covariance under different sample sizes.
3. Compare sample covariance with shrinkage covariance.
4. Implement a simple factor-model simulation.
5. Implement Gaussian mixture EM.
6. Compare mixture estimation across initializations.
7. Simulate a latent-class model.
8. Examine sensitivity of covariance estimates when \(p\) approaches or exceeds \(n\).
9. Demonstrate false discoveries in many simultaneous tests.

## Case — Multivariate Financial Anomaly Detection

```text
Financial Feature Vector
        |
        v
Data Quality / Scaling
        |
        v
Multivariate Distribution
        |
        v
Covariance / Dependence Estimation
        |
        v
Conditional or Joint Anomaly Score
        |
        v
Uncertainty and Sensitivity Analysis
```

Jangan menganggap anomaly score sebagai probabilitas fraud tanpa model probabilistik dan validation yang mendukung interpretasi tersebut.

---

# DAY 15 — Statistical Learning, Generalization & Predictive Uncertainty

## Objective

Menghubungkan statistics dengan modern machine learning melalui generalization, statistical learning theory, calibration, uncertainty quantification, dan distribution shift.

## Core

399. Statistical learning problems
400. Prediction targets
401. Empirical risk minimization
402. Population risk
403. Generalization error
404. Bias-variance trade-off
405. Overfitting
406. Underfitting
407. Model selection
408. Cross-validation
409. Nested cross-validation
410. Selection-induced optimism
411. Data leakage
412. Proper scoring rules
413. Probabilistic prediction
414. Calibration
415. Reliability diagrams
416. Brier score
417. Log loss
418. ROC analysis
419. Precision-recall analysis
420. Threshold selection
421. Predictive intervals
422. Aleatoric uncertainty
423. Epistemic uncertainty
424. Conformal prediction
425. Distribution shift
426. Covariate shift
427. Label shift
428. Concept drift
429. Selective prediction
430. Out-of-distribution evaluation
431. Statistical learning bounds
432. VC dimension
433. Rademacher complexity
434. PAC-style generalization reasoning
435. Model evaluation under dependence
436. Statistical evaluation of AI systems

## Statistical Learning Theory

Empirical risk:

\[
\hat R(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
L(y_i,f(x_i))
\]

Population risk:

\[
R(f)
=
\mathbb E_{(X,Y)}
[L(Y,f(X))]
\]

Objective utama bukan sekadar meminimalkan observed training loss.

Tujuan deployment adalah mendapatkan performa yang memadai pada target population dan operational environment.

## Advanced Concepts

### A. Evaluation Uncertainty

Metrics seperti accuracy, AUROC, F1, Brier score, dan log loss memiliki sampling uncertainty.

Jika model dibandingkan pada observasi yang sama, paired analysis umumnya lebih relevan daripada memperlakukan kedua model sebagai evaluasi yang independen.

Pilih uncertainty estimation sesuai evaluation design:

- IID test observations
- Clustered observations
- Repeated measurements
- Temporal test sets
- Paired model evaluations

### B. Calibration

Untuk probabilistic prediction, calibration menilai apakah predicted probabilities selaras dengan observed frequencies pada kondisi yang relevan.

Calibration berbeda dari discrimination.

Model dapat membedakan kelompok berisiko tinggi dan rendah tetapi memiliki probability estimates yang kurang terkalibrasi.

### C. Conformal Prediction

Conformal prediction dapat menghasilkan prediction sets atau intervals menggunakan calibration data.

Pada split-conformal regression sederhana, residual calibration digunakan untuk menentukan interval width.

Coverage guarantee standar bergantung pada kondisi seperti exchangeability antara calibration observations dan target observations.

Coverage marginal tidak berarti coverage bersyarat dijamin pada setiap subgroup atau setiap individual.

### D. Distribution Shift

Bedakan:

- Covariate shift
- Label shift
- Concept drift
- Temporal shift
- Population shift
- Dataset construction shift

Evaluasi yang valid pada satu test distribution tidak menjamin performa yang sama ketika deployment distribution berubah.

### E. LLM and AI Evaluation Statistics

Statistical tools dapat diterapkan pada:

- Pairwise model comparisons
- Human preference evaluations
- Benchmark uncertainty
- Prompt variation
- Repeated generations
- Inter-rater reliability
- Evaluation-set contamination
- Subgroup reliability
- Calibration of confidence estimates
- Abstention behavior
- Error-rate estimation

LLM evaluation memerlukan unit analisis dan sampling assumptions yang eksplisit.

Contohnya, banyak jawaban dari prompt yang sama bukan otomatis merupakan banyak observasi independen tentang populasi prompt yang ingin dievaluasi.

## Algorithm Mechanism

- Cross-validation
- Bootstrap evaluation
- Paired bootstrap
- Group-aware validation
- Time-aware validation
- Probability calibration
- Conformal prediction
- Threshold optimization
- Selective prediction
- Distribution-shift diagnostics
- Generalization-bound estimation

## Diagnostics

- Train-test contamination
- Validation design
- Metric confidence intervals
- Calibration error
- Proper scoring rules
- Subgroup performance
- Data leakage
- Covariate shift
- Temporal drift
- Prediction-set coverage
- Prediction-set width
- Human-rater agreement
- Model comparison uncertainty

## From Scratch

1. Implement train-validation-test separation.
2. Implement paired bootstrap for model comparisons.
3. Implement a reliability diagram.
4. Calculate Brier score and log loss.
5. Implement probability threshold analysis.
6. Implement simple split-conformal prediction.
7. Measure empirical coverage and interval width.
8. Simulate covariate shift.
9. Compare random split and group-aware validation.
10. Evaluate statistical uncertainty of an AI benchmark.
11. Compare repeated observations with independent observations.
12. Demonstrate model-selection optimism.

## Experiment

### Statistical Model Comparison

Bandingkan dua model dengan paired outcomes.

Analisis:

```text
Same Evaluation Units
        |
        v
Paired Performance Differences
        |
        v
Uncertainty Estimate
        |
        v
Confidence Interval
        |
        v
Practical Effect Size
        |
        v
Deployment Decision
```

Jangan memilih model hanya karena mempunyai satu angka metric yang lebih tinggi.

## Failure Modes

- Data leakage
- Reusing test data for model selection
- Misinterpreting calibration
- Assuming predictive confidence equals correctness probability
- Treating repeated generations as independent when they are not
- Overinterpreting benchmark differences
- Ignoring evaluation-set uncertainty
- Reporting aggregate performance without subgroup analysis
- Ignoring distribution shift
- Treating conformal coverage as a universal individual-level guarantee

## Case — AI Learning Recommendation

Model digunakan untuk memperkirakan materi yang sesuai bagi seorang learner.

Evaluasi perlu membedakan:

1. Ranking performance.
2. Probability calibration.
3. Prediction uncertainty.
4. Learning outcome improvement.
5. Subgroup reliability.
6. Temporal generalization.
7. Distribution shift.
8. Causal impact of the recommendation.

Predictive quality dan causal impact adalah dua target evaluasi yang berbeda.

---

# DAY 16 — Statistical Computing, Simulation Science & Research Integration

## Objective

Menyatukan teori statistik, computational methods, diagnostics, experimental evaluation, dan reproducible research.

Hari ini berfungsi sebagai integrasi seluruh roadmap.

## Core

437. Statistical computing
438. Numerical likelihood optimization
439. Monte Carlo methods
440. Monte Carlo standard error
441. Variance reduction
442. Control variates
443. Antithetic sampling
444. Importance sampling
445. Rare-event simulation
446. Random number generation
447. Random seed management
448. Reproducible simulation
449. Simulation-based calibration
450. Posterior predictive simulation
451. Parametric bootstrap
452. Numerical integration
453. Optimization diagnostics
454. Parallel simulation
455. Computational scalability
456. Statistical software validation
457. Unit testing of estimators
458. Simulation-based testing
459. Reproducible research
460. Sensitivity analysis
461. Scientific reporting
462. Statistical workflow automation
463. Research documentation

## Core Mathematics

Monte Carlo estimate:

\[
\hat\mu_{\mathrm{MC}}
=
\frac{1}{M}\sum_{m=1}^{M}h(X_m)
\]

Untuk observasi simulasi IID dengan finite variance:

\[
\operatorname{MCSE}(\hat\mu_{\mathrm{MC}})
\approx
\frac{s_h}{\sqrt M}
\]

di mana \(s_h\) adalah sample standard deviation dari \(h(X_m)\).

Untuk correlated Monte Carlo samples, seperti output MCMC, perhitungan error perlu memperhitungkan dependence antar-sampel.

## Statistical Computing Principles

### A. Statistical Correctness

Kode yang berjalan bukan bukti bahwa algoritma benar.

Verifikasi:

- Mathematical derivation
- Known analytical solutions
- Simulation under known data-generating processes
- Edge cases
- Parameter boundaries
- Invariance properties
- Numerical stability
- Reproducibility

### B. Simulation-Based Validation

Ketika kebenaran analitik tersedia, gunakan untuk membandingkan estimator.

Ketika kebenaran analitik tidak tersedia, buat controlled data-generating processes dengan parameter yang diketahui.

Ukur:

- Bias
- Variance
- MSE
- Coverage
- Type I error
- Power
- Calibration
- Convergence
- Computational cost

### C. Statistical Software Testing

Setiap implementation dari estimator perlu diuji dengan:

1. Small controlled example.
2. Known analytical solution.
3. Edge-case parameters.
4. Large and small sample sizes.
5. Different random seeds.
6. Alternative data distributions.
7. Numerical precision checks.
8. Regression tests.

## From Scratch

1. Build a reusable Monte Carlo simulation framework.
2. Implement statistical estimators with unit tests.
3. Implement simulation-based coverage analysis.
4. Implement simulation-based power analysis.
5. Implement a parametric bootstrap.
6. Estimate Monte Carlo standard errors.
7. Compare numerical estimators against analytical solutions.
8. Build reproducible simulation reports.
9. Automate diagnostic plots.
10. Benchmark multiple statistical methods under controlled conditions.

## Failure Modes

- Using one seed as the only validation
- Confusing Monte Carlo error with sampling uncertainty
- Ignoring numerical error
- Failing to distinguish algorithmic convergence from model validity
- Reporting simulation results without simulation assumptions
- Comparing methods under different data-generating processes
- Ignoring computation-induced bias
- Failing to test boundary cases
- Reproducing code without reproducing its assumptions

---

# 03. ADVANCED STATISTICAL THEORY EXTENSIONS

Modul 1–16 membangun core yang luas.

Setelah itu, lanjutkan ke teori lanjutan berikut sesuai fokus riset.

## 03.1 Measure-Theoretic Probability

- Sigma-algebras
- Probability measures
- Measurable functions
- Lebesgue integration
- Conditional expectation as a random variable
- Radon-Nikodym theorem
- Modes of stochastic convergence
- Uniform integrability
- Martingales
- Martingale convergence
- Stopping times
- Optional stopping conditions
- Weak convergence
- Characteristic-function methods

Tujuan:

Memahami fondasi matematis yang digunakan dalam mathematical statistics, stochastic processes, Bayesian theory, dan asymptotic inference.

## 03.2 Asymptotic Statistics

- M-estimation
- Z-estimation
- Asymptotic linearity
- Influence functions
- Empirical processes
- Asymptotic efficiency
- Local asymptotic normality
- Likelihood asymptotics
- Semiparametric models
- Nuisance parameters
- Orthogonal estimating equations
- Double machine learning
- Debiased estimation
- Post-selection inference

## 03.3 Information-Theoretic Statistics

- Shannon entropy
- Conditional entropy
- Mutual information
- Kullback-Leibler divergence
- Hellinger distance
- Total variation distance
- Fisher information
- Information geometry
- Minimum description length
- Information criteria
- Fano's inequality
- Le Cam's methods
- Minimax lower bounds
- Statistical identifiability

## 03.4 Extreme Value Statistics

- Extreme order statistics
- Generalized extreme-value distributions
- Generalized Pareto distributions
- Threshold selection
- Tail-index estimation
- Return levels
- Rare-event extrapolation
- Tail dependence
- Extreme risk estimation
- Uncertainty in tail estimates

## 03.5 Spatial Statistics

- Spatial dependence
- Spatial covariance functions
- Geostatistics
- Kriging
- Spatial Gaussian processes
- Point-process models
- Spatial sampling
- Spatial confounding
- Spatial random effects
- Spatially structured missingness
- Spatiotemporal models

## 03.6 Advanced Longitudinal and Causal Models

- Potential outcomes with repeated treatments
- Time-varying confounding
- Marginal structural models
- Sequential exchangeability
- G-methods
- Longitudinal mediation
- Time-dependent treatment effects
- Targeted maximum likelihood estimation
- Causal forests
- Heterogeneous treatment effect estimation
- Interference and spillovers
- Sensitivity analysis under unmeasured confounding

## 03.7 Sequential and Online Statistics

- Sequential hypothesis testing
- Sequential confidence sequences
- Optional stopping
- Adaptive data collection
- Online change-point detection
- Online estimation
- Nonstationary populations
- Exploration-exploitation
- Contextual bandits
- Adaptive experimentation
- Online calibration
- Delayed outcome feedback

## 03.8 Advanced Statistical Learning Theory

- Uniform convergence
- Rademacher complexity
- VC theory
- PAC learning
- Stability-based generalization
- Double descent
- Post-selection inference
- Distribution-free prediction
- Conformal risk control
- Selective classification
- Uncertainty under distribution shift
- Statistical guarantees for adaptive model evaluation

---

# 04. STATISTICAL MATHEMATICS CHECKLIST

Gunakan bagian ini sebagai checklist penguasaan matematis.

## Probability Theory

- Conditional probability
- Bayes' theorem
- Random variables
- Joint and conditional distributions
- Expectation operators
- Conditional expectation
- Variance decomposition
- Covariance
- Independence
- Exchangeability
- Moment-generating functions
- Characteristic functions
- Probability inequalities
- Stochastic convergence
- LLN
- CLT
- Concentration inequalities

## Statistical Estimation

- Bias
- Variance
- MSE
- Consistency
- Efficiency
- Sufficiency
- Completeness
- Cramér-Rao lower bound
- Fisher information
- Method of moments
- Maximum likelihood
- Profile likelihood
- Shrinkage
- Decision-theoretic risk
- Minimax estimation

## Statistical Inference

- Sampling distributions
- Standard errors
- Confidence intervals
- Confidence regions
- Prediction intervals
- Hypothesis testing
- Likelihood-ratio tests
- Wald tests
- Score tests
- Bootstrap inference
- Permutation tests
- Multiple testing
- Sequential testing
- Power analysis
- Selective inference

## Bayesian Statistics

- Prior
- Likelihood
- Posterior
- Posterior predictive
- Conjugate priors
- Hierarchical priors
- Partial pooling
- Bayes factors
- Posterior model comparison
- MCMC
- HMC
- NUTS
- Variational inference
- Posterior predictive checking

## Regression and Statistical Modeling

- Conditional expectation
- Conditional variance
- Linear models
- GLM
- Logistic regression
- Poisson regression
- Negative binomial regression
- Gamma regression
- Likelihood-based inference
- Robust standard errors
- Model specification
- Calibration
- Overdispersion
- Influence functions
- Penalized likelihood

## Robust and Nonparametric Statistics

- Empirical distribution functions
- Quantile methods
- Bootstrap
- Jackknife
- Permutation inference
- Kernel density estimation
- Nonparametric regression
- Robust location
- Robust scale
- M-estimation
- Huber estimation
- Influence functions
- Breakdown point
- Sandwich variance

## Causal Inference

- Potential outcomes
- Causal estimands
- DAGs
- Confounding
- Consistency
- Exchangeability
- Positivity
- Backdoor adjustment
- Standardization
- G-computation
- Propensity scores
- IPW
- AIPW
- Instrumental variables
- Difference-in-differences
- Regression discontinuity
- Synthetic controls
- Sensitivity analysis

## Experimental Design

- Randomization
- Blocking
- Cluster assignment
- Factorial designs
- Sample size
- Statistical power
- MDE
- Covariate adjustment
- Multiple endpoints
- Sequential designs
- Adaptive experiments
- Treatment effect heterogeneity
- Noncompliance
- Attrition
- Replication

## Multivariate and High-Dimensional Statistics

- Multivariate distributions
- Covariance estimation
- Precision matrix
- Covariance shrinkage
- Gaussian graphical models
- PCA
- Factor analysis
- Mixture models
- EM
- Latent variables
- Copulas
- Tail dependence
- Sparse estimation
- High-dimensional multiple testing
- Robust covariance

## Longitudinal and Time-to-Event Statistics

- Time-series dependence
- Stationarity
- Autocorrelation
- ARIMA
- State-space models
- Kalman filtering
- Forecast distributions
- Structural breaks
- Survival functions
- Hazard functions
- Kaplan-Meier
- Cox regression
- Competing risks
- Informative censoring

## Statistical Learning and Uncertainty

- Empirical risk
- Population risk
- Generalization
- Model selection
- Cross-validation
- Nested cross-validation
- Calibration
- Proper scoring rules
- Prediction intervals
- Conformal prediction
- Distribution shift
- Selective prediction
- Statistical benchmark uncertainty
- Adaptive evaluation

---

# 05. STATISTICAL ASSUMPTIONS

Asumsi adalah bagian dari metode statistik, bukan catatan tambahan setelah perhitungan selesai.

| Assumption | Meaning | Typical Risk if Violated |
|---|---|---|
| Independence | Observations satisfy the required independence structure | Incorrect standard errors and invalid inference |
| Identical distribution | Observations share a common distribution where required | Misleading pooled estimates |
| Exchangeability | Relevant observations can be permuted without changing their joint distribution | Invalid resampling or conformal guarantees |
| Random sampling | Sampling process supports the intended population inference | Selection bias |
| Random assignment | Treatment assignment follows the specified randomized design | Invalid experimental analysis if design is ignored |
| Consistency | Observed outcome matches the relevant potential outcome | Ambiguous causal interpretation |
| Positivity | Relevant treatment or exposure levels are available across target covariate profiles | Unstable or unidentified causal estimates |
| Correct likelihood | Model represents the relevant data-generating distribution adequately | Invalid likelihood-based inference |
| Identifiability | Different parameter values imply distinguishable observable distributions | Non-unique inference |
| Correct link and mean structure | Conditional mean is adequately represented | Biased or misleading interpretation |
| Variance specification | Variance model matches the estimation procedure's requirements | Misestimated uncertainty |
| Non-informative censoring | Censoring mechanism meets the relevant survival-analysis assumptions | Biased survival estimates |
| Missing at random | Missingness can be explained by observed data under the specified model | Invalid imputation or likelihood inference |
| Stable evaluation distribution | Evaluation design reflects the target population | Misleading generalization claims |

Tidak semua metode memerlukan seluruh asumsi tersebut.

Kumpulkan hanya asumsi yang benar-benar dibutuhkan untuk target estimand, estimator, standard errors, dan inferential guarantees yang dipakai.

---

# 06. STATISTICAL HYPERPARAMETERS AND DESIGN CHOICES

Istilah *hyperparameter* dalam statistik dapat berarti hal berbeda, tergantung konteks.

Sebagian adalah parameter model yang ditentukan sebelum fitting, sebagian merupakan pilihan desain penelitian, dan sebagian merupakan tuning parameter komputasi.

## Bayesian

- Prior location
- Prior scale
- Prior degrees of freedom
- Hyperprior specification
- Number of MCMC chains
- Warmup iterations
- Sampling iterations
- Target acceptance rate
- Parameterization
- Variational approximation family

## Frequentist

- Significance level
- Confidence level
- Bootstrap repetitions
- Resampling scheme
- Multiple-testing procedure
- Sequential stopping rule
- Confidence interval construction
- Power target

## Regression and Statistical Learning

- Regularization strength
- Link function
- Spline degrees of freedom
- Kernel bandwidth
- Cross-validation scheme
- Model complexity
- Decision threshold
- Calibration method
- Feature selection procedure

## Causal Inference

- Propensity model specification
- Weight truncation or stabilization choices
- Matching ratio
- Matching caliper
- Covariate adjustment set
- Regression discontinuity bandwidth
- Difference-in-differences time window
- Sensitivity parameter
- Cross-fitting scheme

## Time Series

- AR order
- MA order
- Seasonal orders
- Forecast horizon
- Rolling window length
- Volatility-model specification
- Structural-break procedure

## Survival Analysis

- Time scale
- Follow-up window
- Functional form of covariates
- Baseline hazard specification
- Time-varying effects
- Competing-risk estimand

Setiap pilihan perlu didokumentasikan beserta alasan, prosedur pemilihan, dan sensitivity analysis yang relevan.

---

# 07. STATISTICAL DIAGNOSTICS

Jangan membatasi diagnostics pada p-values dan residual plots.

## Estimator Diagnostics

- Bias
- Variance
- MSE
- Consistency
- Efficiency
- Identifiability
- Sensitivity
- Robustness

## Sampling Diagnostics

- Representativeness
- Sampling frame coverage
- Cluster structure
- Missingness
- Effective sample size
- Weight distribution
- Dependence

## Inference Diagnostics

- Confidence interval coverage
- Type I error
- Statistical power
- Multiplicity
- P-value calibration
- Sensitivity to assumptions
- Finite-sample behavior

## Model Diagnostics

- Residual structure
- Heteroscedasticity
- Autocorrelation
- Overdispersion
- Model fit
- Influence
- Calibration
- Predictive checks
- Distributional assumptions

## Bayesian Diagnostics

- Chain mixing
- Effective sample size
- R-hat
- Monte Carlo error
- Posterior predictive checks
- Prior sensitivity
- Divergent transitions
- Identifiability

## Causal Diagnostics

- Covariate balance
- Positivity
- Overlap
- Weight instability
- Pre-treatment trends
- Sensitivity to unmeasured confounding
- Placebo tests
- Alternative model specifications

## Machine Learning Diagnostics

- Leakage
- Calibration
- Generalization
- Distribution shift
- Data contamination
- Metric uncertainty
- Subgroup errors
- Conformal coverage
- Temporal performance decay

---

# 08. STATISTICAL FAILURE MODES

## Probability

- Incorrect dependence assumptions
- Misinterpreted conditional probabilities
- Ignored tail behavior
- Incorrect distributional approximation
- Incorrect Monte Carlo uncertainty

## Sampling and Inference

- Selection bias
- Nonresponse bias
- Invalid standard errors
- Incorrect confidence interval interpretation
- P-hacking
- Multiple-testing errors
- Invalid sequential stopping
- Underpowered studies

## Estimation

- Non-identifiability
- Inconsistent estimation under the assumed framework
- Excessive variance
- Numerical optimization failure
- Boundary estimates
- Incorrect loss function
- Sensitivity to outliers

## Bayesian

- Implausible prior predictive behavior
- Poor posterior geometry
- MCMC convergence problems
- Weak identification
- Prior sensitivity
- Posterior predictive mismatch
- Invalid interpretation of model-dependent uncertainty

## Modeling

- Misspecified likelihood
- Wrong conditional mean structure
- Dependence ignored
- Heteroscedasticity ignored
- Overdispersion
- Separation
- Influential observations
- Overfitting

## Data Quality

- Missing not at random
- Measurement error
- Differential misclassification
- Selection bias
- Information leakage
- Incorrect aggregation
- Dataset shift

## Causal Inference

- Unmeasured confounding
- Collider adjustment
- Positivity violations
- Incorrect identification assumptions
- Invalid instrumental variables
- Unsupported parallel-trends assumptions
- Ignored interference

## Advanced Machine Learning

- Evaluation contamination
- Test-set reuse
- Benchmark overfitting
- Miscalibration
- Distribution shift
- Unsupported uncertainty claims
- Confusing prediction with causation

---

# 09. FROM-SCRATCH IMPLEMENTATION TRACK

Tujuan implementation dari nol bukan mengganti library profesional.

Tujuannya adalah memastikan mekanisme matematis benar-benar dipahami.

## Level 1 — Core Probability

- Random variable simulator
- Conditional probability calculator
- Distribution sampler
- Monte Carlo estimator
- Empirical distribution functions
- Sampling distribution simulator

## Level 2 — Statistical Inference

- Method-of-moments estimator
- Maximum likelihood estimator
- Numerical score function
- Fisher information approximation
- Confidence interval construction
- Hypothesis testing
- Power simulation
- Bootstrap
- Permutation testing

## Level 3 — Robust Statistics

- Quantile estimator
- MAD
- Huber loss
- Robust location estimation
- Robust regression
- Cluster bootstrap
- Block bootstrap

## Level 4 — Bayesian Computation

- Beta-Binomial updating
- Normal-Normal updating
- Grid-based posterior
- Rejection sampling
- Metropolis-Hastings
- Posterior predictive simulation
- Hierarchical partial pooling

## Level 5 — Regression and Causal Inference

- Linear regression likelihood
- Logistic regression likelihood
- Poisson regression
- Robust standard errors
- Propensity scores
- IPW
- AIPW
- Basic sensitivity analysis

## Level 6 — Advanced Data Structures

- Multilevel model simulation
- Gaussian mixture EM
- Survival-function estimation
- Kaplan-Meier estimation
- Time-series bootstrap
- Forecast uncertainty analysis
- Covariance shrinkage

## Level 7 — Statistical Learning

- Cross-validation
- Paired model comparison
- Calibration
- Bootstrap evaluation
- Split-conformal prediction
- Distribution-shift diagnostics
- Subgroup uncertainty analysis

---

# 10. EXPERIMENT LAB

Setiap experiment harus mempunyai:

```text
QUESTION
    |
    v
DATA-GENERATING PROCESS
    |
    v
KNOWN TARGET / GROUND TRUTH
    |
    v
METHODS TO COMPARE
    |
    v
REPETITIONS
    |
    v
PERFORMANCE METRICS
    |
    v
DIAGNOSTICS
    |
    v
RESULTS
    |
    v
INTERPRETATION
    |
    v
LIMITATIONS
```

## Experiment 1 — Sampling Distribution

Compare:

- Different sample sizes
- Different underlying distributions
- Different tail behaviors
- Independent vs dependent observations

Metrics:

- Bias
- Standard error
- Coverage
- Convergence behavior

## Experiment 2 — Estimator Performance

Compare:

- Method of moments
- Maximum likelihood
- Bayesian estimation
- Robust estimation
- Shrinkage estimation

Metrics:

- Bias
- Variance
- MSE
- Robustness
- Computational cost

## Experiment 3 — Frequentist Calibration

Generate data under a known null model.

Evaluate:

- Type I error
- P-value behavior
- Confidence interval coverage
- Effect of sample size
- Effect of model misspecification

## Experiment 4 — Bootstrap Validity

Compare:

- IID bootstrap
- Cluster bootstrap
- Block bootstrap
- Parametric bootstrap

Gunakan data-generating processes dengan dependence structures yang diketahui.

## Experiment 5 — Bayesian Model Validation

Evaluate:

- Prior predictive behavior
- Posterior recovery
- MCMC diagnostics
- Posterior predictive checks
- Sensitivity to prior choices

## Experiment 6 — Missing Data

Generate MCAR, MAR, dan MNAR datasets.

Compare:

- Complete-case analysis
- Multiple imputation
- Inverse probability weighting
- Sensitivity analysis

Jangan menganggap satu metode selalu lebih baik untuk semua missingness mechanisms.

## Experiment 7 — Causal Identification

Compare:

- Randomized benchmark
- Naive association
- Regression adjustment
- Propensity weighting
- Doubly robust estimation

Manipulasikan:

- Confounding strength
- Treatment overlap
- Sample size
- Treatment prevalence
- Unmeasured confounding

## Experiment 8 — Experimental Design

Evaluate:

- Sample size
- Power
- Minimum detectable effect
- Cluster assignment
- Repeated testing
- Outcome multiplicity
- Attrition

## Experiment 9 — Distribution Shift

Compare model reliability under:

- In-distribution testing
- Covariate shift
- Temporal shift
- Label shift
- Increased noise
- Changed outcome prevalence

## Experiment 10 — Statistical AI Evaluation

Compare two AI systems using:

- Paired outcomes
- Bootstrap confidence intervals
- Calibration
- Subgroup performance
- Human-rater agreement
- Prompt-level variability
- Evaluation-set sensitivity

## Experiment 11 — Survival Analysis

Evaluate:

- Kaplan-Meier estimation
- Cox model assumptions
- Independent versus informative censoring
- Competing risks
- Survival probability calibration

## Experiment 12 — High-Dimensional Estimation

Compare covariance and regression estimation when:

\[
p \ll n,\qquad p\approx n,\qquad p>n
\]

Amati:

- Estimation error
- Variance
- Conditioning
- Regularization effects
- Feature selection instability

---

# 11. CASE STUDY PORTFOLIO

Setiap case study harus menggunakan data nyata atau simulated data yang proses pembentukannya didokumentasikan.

Jangan menyimpulkan causal effect dari data observasional hanya karena model regression menghasilkan koefisien yang signifikan.

## Case 1 — Healthcare Risk Estimation

Questions:

- Bagaimana mengestimasi probabilitas event?
- Bagaimana uncertainty berubah pada kasus yang jarang?
- Bagaimana missing documentation memengaruhi estimasi?
- Bagaimana calibration dievaluasi?
- Apakah model cukup reliable untuk target population?

Methods:

- GLM
- Logistic regression
- Bayesian estimation
- Robust inference
- Calibration
- Missing-data sensitivity analysis

## Case 2 — Financial Anomaly and Tail Risk

Questions:

- Apakah sebuah observasi menyimpang dari model baseline?
- Bagaimana dependence dan volatility memengaruhi anomaly score?
- Bagaimana tail risk diestimasi?
- Seberapa sensitif hasil terhadap extreme observations?

Methods:

- Robust statistics
- Multivariate models
- Extreme value statistics
- Time-series analysis
- Covariance estimation
- Bootstrap and stress scenarios

## Case 3 — Personalized Learning

Questions:

- Bagaimana mengestimasi individual learning outcomes?
- Bagaimana menggabungkan individual dan population evidence?
- Apakah personalized intervention menyebabkan peningkatan?
- Bagaimana ketidakpastian berubah ketika histori learner sedikit?

Methods:

- Hierarchical models
- Bayesian inference
- Causal inference
- Randomized experiments
- Survival analysis
- Sequential experimentation

## Case 4 — AI Benchmark Evaluation

Questions:

- Apakah model A benar-benar lebih baik daripada model B?
- Seberapa besar sampling uncertainty?
- Apakah perbedaan konsisten di berbagai prompt categories?
- Bagaimana human-rating variability memengaruhi hasil?
- Apakah benchmark representative terhadap penggunaan nyata?

Methods:

- Paired inference
- Bootstrap
- Multiple testing
- Calibration
- Hierarchical models
- Statistical decision theory

## Case 5 — Healthcare Intervention Evaluation

Questions:

- Berapa average treatment effect?
- Apakah randomization berhasil?
- Apakah missing outcome memengaruhi kesimpulan?
- Apakah efek berbeda pada subgroup yang telah ditentukan?
- Apakah hasil robust terhadap analisis alternatif yang masuk akal?

Methods:

- Randomized trials
- Regression adjustment
- Causal estimands
- Confidence intervals
- Multiple imputation
- Sensitivity analysis

## Case 6 — Time-to-Event Prediction

Questions:

- Berapa probability event belum terjadi pada waktu \(t\)?
- Bagaimana censoring memengaruhi estimasi?
- Apakah hazard berubah seiring waktu?
- Apakah terdapat competing events?

Methods:

- Kaplan-Meier
- Cox models
- Parametric survival models
- Competing risks
- Calibration
- Censoring diagnostics

## Case 7 — Statistical Evaluation of Self-Improving AI

Questions:

- Apakah versi model baru menunjukkan peningkatan yang konsisten?
- Apakah perubahan data memengaruhi evaluasi?
- Apakah metric improvement berasal dari genuine generalization atau benchmark overfitting?
- Apakah confidence estimates terkalibrasi?
- Apakah perbedaan performa hanya muncul pada subgroup tertentu?

Methods:

- Paired evaluations
- Bootstrap uncertainty
- Sequential testing
- Calibration
- Distribution-shift analysis
- Multiple testing
- Hierarchical modeling
- Reproducible simulation

---

# 12. IMPLEMENTATION STACK

## Core Numerical and Data Stack

- NumPy
- SciPy
- pandas
- Matplotlib

## Classical Statistical Modeling

- statsmodels
- SciPy statistics
- scikit-learn
- patsy

## Bayesian Statistics

- PyMC
- ArviZ

## Survival Analysis

- lifelines

## Time-Series Analysis

- statsmodels
- SciPy
- Specialized forecasting libraries when justified by the task

## Causal Inference

- statsmodels
- DoWhy
- EconML
- Appropriate specialized econometrics libraries

## Research Workflow

- Jupyter
- pytest
- Git
- Reproducible environments
- Simulation notebooks
- Versioned datasets
- Automated diagnostics
- Documented analysis plans

Library pilihan perlu disesuaikan dengan model, versi, dan kebutuhan analisis.

Jangan menggunakan library sebagai pengganti pemahaman asumsi statistik.

---

# 13. REPRODUCIBLE STATISTICAL RESEARCH

Setiap experiment harus memiliki dokumentasi minimum:

```text
experiment/
├── README.md
├── assumptions.md
├── data_generating_process.py
├── estimators.py
├── diagnostics.py
├── simulation.py
├── tests/
│   └── test_estimators.py
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── metrics.csv
│   └── plots/
└── report.md
```

Untuk data empiris, tambahkan dokumentasi data provenance, sampling design, variable definitions, missingness, privacy, dan limitations.

## Experiment Report

Setiap report harus menjawab:

1. Apa pertanyaan penelitian?
2. Apa target estimand?
3. Apa data-generating process atau sampling mechanism?
4. Apa asumsi utama?
5. Mengapa metode dipilih?
6. Apa alternatif metode yang masuk akal?
7. Bagaimana uncertainty diukur?
8. Apa diagnostics yang diperiksa?
9. Apakah hasil sensitif terhadap asumsi?
10. Bagaimana hasil berubah pada subgroup yang relevan?
11. Apa keterbatasan kesimpulan?
12. Apa yang dapat dan tidak dapat disimpulkan?

---

# 14. FINAL INTEGRATION

Pada akhir roadmap, seluruh materi harus terhubung sebagai satu sistem.

## Statistical Foundations

```text
PROBABILITY SPACE
        |
        v
RANDOM VARIABLES
        |
        v
JOINT / CONDITIONAL DISTRIBUTIONS
        |
        v
EXPECTATION / VARIANCE / DEPENDENCE
        |
        v
SAMPLING DISTRIBUTIONS
        |
        v
LLN / CLT / ASYMPTOTIC THEORY
```

## Statistical Estimation

```text
DATA-GENERATING PROCESS
        |
        v
TARGET PARAMETER / ESTIMAND
        |
        v
ESTIMATOR
        |
        v
BIAS / VARIANCE / MSE
        |
        v
EFFICIENCY / ROBUSTNESS
        |
        v
ESTIMATION UNCERTAINTY
```

## Likelihood and Frequentist Inference

```text
PROBABILITY MODEL
        |
        v
LIKELIHOOD
        |
        v
MLE / SCORE / FISHER INFORMATION
        |
        v
CONFIDENCE INTERVAL / HYPOTHESIS TEST
        |
        v
POWER / MULTIPLICITY / SENSITIVITY
```

## Bayesian Inference

```text
PRIOR
    |
    v
LIKELIHOOD
    |
    v
POSTERIOR
    |
    v
POSTERIOR PREDICTIVE
    |
    v
DECISION UNDER UNCERTAINTY
```

## Regression and Statistical Modeling

```text
DATA
    |
    v
CONDITIONAL DISTRIBUTION
    |
    v
MODEL SPECIFICATION
    |
    v
PARAMETER ESTIMATION
    |
    v
RESIDUAL / MODEL DIAGNOSTICS
    |
    v
UNCERTAINTY QUANTIFICATION
    |
    v
SCIENTIFIC INTERPRETATION
```

## Causal Inference

```text
CAUSAL QUESTION
        |
        v
CAUSAL ESTIMAND
        |
        v
IDENTIFICATION ASSUMPTIONS
        |
        v
IDENTIFICATION STRATEGY
        |
        v
ESTIMATOR
        |
        v
UNCERTAINTY QUANTIFICATION
        |
        v
SENSITIVITY ANALYSIS
```

## Experimental Design

```text
RESEARCH QUESTION
        |
        v
TARGET ESTIMAND
        |
        v
RANDOMIZATION / SAMPLING DESIGN
        |
        v
POWER AND SAMPLE SIZE
        |
        v
DATA COLLECTION
        |
        v
PRE-SPECIFIED ANALYSIS
        |
        v
STATISTICAL EVIDENCE
        |
        v
DECISION
```

## Modern Statistical Learning

```text
TRAINING DATA
        |
        v
LEARNING PROCEDURE
        |
        v
EMPIRICAL RISK
        |
        v
MODEL SELECTION
        |
        v
GENERALIZATION EVALUATION
        |
        v
CALIBRATION / UNCERTAINTY
        |
        v
DISTRIBUTION SHIFT ANALYSIS
        |
        v
DEPLOYMENT DECISION
```

---

# 15. FINAL KNOWLEDGE MAP

```text
STATISTICS
|
+-- 01. Probability Theory
|   +-- Random Variables
|   +-- Joint / Conditional Distributions
|   +-- Expectation
|   +-- Dependence
|   +-- Stochastic Convergence
|
+-- 02. Sampling Theory
|   +-- Sampling Distributions
|   +-- Standard Errors
|   +-- LLN / CLT
|   +-- Concentration Inequalities
|
+-- 03. Estimation Theory
|   +-- Bias / Variance / MSE
|   +-- Consistency
|   +-- Efficiency
|   +-- Decision Theory
|
+-- 04. Likelihood Theory
|   +-- MLE
|   +-- Fisher Information
|   +-- Likelihood Ratio
|   +-- Profile Likelihood
|
+-- 05. Statistical Inference
|   +-- Confidence Intervals
|   +-- Hypothesis Testing
|   +-- Statistical Power
|   +-- Multiple Testing
|
+-- 06. Resampling and Robust Statistics
|   +-- Bootstrap
|   +-- Permutation Tests
|   +-- Nonparametric Methods
|   +-- Influence Functions
|
+-- 07. Bayesian Statistics
|   +-- Priors and Posteriors
|   +-- Hierarchical Models
|   +-- MCMC
|   +-- Posterior Predictive Checks
|
+-- 08. Statistical Modeling
|   +-- Regression
|   +-- GLM
|   +-- Robust Inference
|   +-- Model Diagnostics
|
+-- 09. Multilevel Statistics
|   +-- Partial Pooling
|   +-- Random Effects
|   +-- Repeated Measures
|   +-- Clustered Data
|
+-- 10. Data Quality and Bias
|   +-- Missing Data
|   +-- Measurement Error
|   +-- Selection Bias
|   +-- Sensitivity Analysis
|
+-- 11. Causal Inference
|   +-- Potential Outcomes
|   +-- DAGs
|   +-- Propensity Scores
|   +-- Doubly Robust Estimation
|   +-- Quasi-Experimental Design
|
+-- 12. Experimental Design
|   +-- Randomization
|   +-- Power
|   +-- A/B Testing
|   +-- Sequential Experiments
|
+-- 13. Temporal Statistics
|   +-- Time Series
|   +-- Forecasting
|   +-- Survival Analysis
|   +-- Censoring
|
+-- 14. Multivariate Statistics
|   +-- Covariance Estimation
|   +-- PCA
|   +-- Factor Analysis
|   +-- Mixture Models
|   +-- Graphical Models
|
+-- 15. Statistical Learning
|   +-- Generalization
|   +-- Calibration
|   +-- Conformal Prediction
|   +-- Distribution Shift
|   +-- Uncertainty Quantification
|
+-- 16. Statistical Computing
    +-- Simulation
    +-- Monte Carlo
    +-- Numerical Validation
    +-- Reproducible Research
```

---

# 16. MASTERY CRITERIA

Roadmap ini dianggap berhasil dipahami secara mendalam ketika kamu mampu melakukan hal-hal berikut tanpa sekadar mengikuti tutorial.

## Mathematical Mastery

- Derive estimators from a stated statistical model.
- Explain the assumptions needed for a theorem or approximation.
- Distinguish finite-sample results from asymptotic results.
- Explain why a confidence interval has its intended coverage.
- Explain what a likelihood-based procedure estimates.
- Derive simple Bayesian posterior distributions.
- Distinguish causal identification from statistical estimation.

## Computational Mastery

- Implement core estimators from scratch.
- Validate implementations against analytical solutions.
- Build simulation studies with known ground truth.
- Estimate bias, variance, MSE, power, and coverage.
- Diagnose Monte Carlo error.
- Implement resampling procedures that respect the data structure.
- Test numerical stability and edge cases.

## Analytical Mastery

- Identify selection bias and missing-data risks.
- Diagnose model misspecification.
- Evaluate uncertainty beyond a single metric.
- Analyze the effects of multiplicity and adaptive decisions.
- Justify a causal inference strategy.
- Evaluate uncertainty under distribution shift.
- Perform sensitivity analysis when assumptions cannot be verified directly.

## Research Mastery

- Define a precise estimand.
- Formulate explicit assumptions.
- Design a reproducible experiment.
- Compare appropriate competing methods.
- Document diagnostics and limitations.
- Explain why the chosen method is appropriate.
- Identify where available evidence is insufficient.

## Final Standard

```text
KNOW THE FORMULA
       |
       v
DERIVE THE METHOD
       |
       v
UNDERSTAND THE ASSUMPTIONS
       |
       v
IMPLEMENT FROM SCRATCH
       |
       v
VALIDATE USING SIMULATION
       |
       v
DIAGNOSE FAILURE MODES
       |
       v
COMPARE ALTERNATIVE METHODS
       |
       v
APPLY TO REAL DATA
       |
       v
REPORT UNCERTAINTY AND LIMITATIONS
       |
       v
MAKE A DEFENSIBLE SCIENTIFIC CONCLUSION
```

---

# 17. REFERENCE MATERIAL

Gunakan referensi berikut untuk memperdalam modul berdasarkan kebutuhan.

## Mathematical Statistics and Inference

- [MIT OpenCourseWare — Mathematical Statistics](https://ocw.mit.edu/courses/18-655-mathematical-statistics-spring-2016/)
- [MIT OpenCourseWare — Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/)

Fokus bacaan:

- Estimation theory
- Likelihood
- Asymptotic efficiency
- Hypothesis testing
- Confidence regions
- Exponential families
- Generalized linear models

## Causal Inference

- [Causal Inference: What If — Miguel Hernán and James Robins](https://miguelhernan.org/whatifbook)

Fokus bacaan:

- Potential outcomes
- Causal estimands
- Identification assumptions
- Causal inference with models
- Longitudinal causal inference

## Bayesian Statistics

- [Statistical Rethinking — 2026 Course Materials](https://github.com/rmcelreath/stat_rethinking_2026)

Fokus bacaan:

- Generative models
- Bayesian reasoning
- Multilevel models
- Causal models
- Posterior predictive checking
- Computational Bayesian statistics

## Statistical Learning

- [The Elements of Statistical Learning — Springer](https://link.springer.com/book/10.1007/978-0-387-21606-5)

Fokus bacaan:

- Statistical learning theory
- Model selection
- Regression and classification
- Regularization
- Model evaluation

## Modern Uncertainty Quantification

- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)
- [A Tutorial on Distribution-Free Uncertainty Quantification Using Conformal Prediction — 2025](https://journals.sagepub.com/doi/10.1177/25152459251380452)

Fokus bacaan:

- Conformal prediction
- Prediction intervals
- Distribution-free uncertainty quantification
- Exchangeability
- Calibration and coverage
- Limitations under distribution shift

## Statistical Methods and Experimental Design

- [NIST/SEMATECH Engineering Statistics Handbook](https://www.nist.gov/programs-projects/nistsematech-engineering-statistics-handbook)

Fokus bacaan:

- Experimental design
- Regression diagnostics
- Statistical process analysis
- Measurement
- Reliability
- Uncertainty

---

# FINAL PRINCIPLE

Statistics should not be treated as a collection of tests to run after data collection.

The complete discipline is a chain of reasoning:

```text
SCIENTIFIC QUESTION
        |
        v
DATA-GENERATING PROCESS
        |
        v
PROBABILITY MODEL
        |
        v
SAMPLING DESIGN
        |
        v
STATISTICAL ESTIMAND
        |
        v
ESTIMATION / INFERENCE
        |
        v
UNCERTAINTY QUANTIFICATION
        |
        v
DIAGNOSTICS / SENSITIVITY
        |
        v
VALIDATION
        |
        v
INTERPRETATION
        |
        v
SCIENTIFIC OR OPERATIONAL DECISION
```

The objective is not merely to obtain a statistically significant result.

The objective is to understand:

- What the data can establish.
- What the data cannot establish.
- Which assumptions are necessary.
- How much uncertainty remains.
- Whether the conclusion survives reasonable changes in assumptions.
- Whether the result generalizes to the target population.
- Whether the evidence supports the intended decision.

**Statistics becomes powerful when mathematical theory, computational implementation, experimental design, and scientific reasoning are treated as one integrated system.**