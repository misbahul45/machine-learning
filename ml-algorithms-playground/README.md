# FULL REPOSITORY LEARNING PLAN — MACHINE LEARNING BEFORE DEEP LEARNING

```text
Goal:
Memahami Machine Learning klasik secara kompleks dan sistematis sebelum masuk Deep Learning.

Core learning flow:
1. Understand problem
2. Understand data
3. Analyze data
4. Preprocess data
5. Engineer features
6. Build model
7. Evaluate model
8. Diagnose bias-variance
9. Interpret model
10. Apply to real case
```

---

# 00_FOUNDATION

## 00.1 Data Preprocessing

```text
Path:
00_foundation/data_preprocessing.ipynb
```

### Plan

```text
1. Understand raw data problem
2. Understand dataset structure
3. Understand feature and target
4. Check data type
5. Check missing value
6. Check duplicate data
7. Check inconsistent data
8. Check invalid value
9. Check outlier
10. Handle missing numerical data
11. Handle missing categorical data
12. Handle duplicate rows
13. Handle noisy data
14. Handle wrong data type
15. Handle outlier using IQR
16. Handle outlier using Z-score
17. Compare raw data vs clean data
18. Prepare clean dataset for model
```

### Sub Materi

```text
- Dataset shape
- Column meaning
- Numerical feature
- Categorical feature
- Binary feature
- Ordinal feature
- Target variable
- Missing value
- Duplicate data
- Outlier
- Data leakage
- Invalid value
- Data consistency
- Mean imputation
- Median imputation
- Mode imputation
- Drop missing value
- IQR method
- Z-score method
```

### Case

```text
1. Cleaning student performance dataset
2. Cleaning restaurant rating dataset
3. Cleaning credit risk dataset
4. Cleaning churn prediction dataset
```

---

## 00.2 Feature Engineering

```text
Path:
00_foundation/feature_engineering.ipynb
```

### Plan

```text
1. Understand why raw features are not enough
2. Analyze feature-target relationship
3. Create numerical features
4. Create ratio features
5. Create interaction features
6. Create polynomial features
7. Create categorical encoded features
8. Create ordinal features
9. Create binary indicator features
10. Create date/time features
11. Create text-derived features
12. Compare model before and after feature engineering
13. Select useful engineered features
14. Remove useless features
```

### Sub Materi

```text
- Feature meaning
- Feature transformation
- Feature interaction
- Polynomial features
- Ratio features
- Difference features
- Aggregation features
- Binning
- Log transformation
- One-hot encoding
- Label encoding
- Ordinal encoding
- Frequency encoding
- Date extraction
- Text count feature
- Feature selection basic
```

### Case

```text
1. price × discount for purchase prediction
2. loan_amount / income for credit risk
3. study_hours² for student score
4. tenure × monthly_spend for churn prediction
5. alcohol × acidity for wine quality
```

---

## 00.3 Scaling & Normalization

```text
Path:
00_foundation/scaling_normalization.ipynb
```

### Plan

```text
1. Understand feature scale problem
2. Analyze feature range
3. Compare small-scale and large-scale features
4. Apply min-max scaling
5. Apply standardization
6. Apply robust scaling
7. Apply log scaling
8. Compare model sensitive to scaling
9. Compare model not sensitive to scaling
10. Understand effect on KNN
11. Understand effect on SVM
12. Understand effect on Logistic Regression
13. Understand effect on PCA
14. Choose proper scaling method
```

### Sub Materi

```text
- Feature range
- Feature distribution
- Min-max scaling
- Standard scaler
- Robust scaler
- Log transform
- Mean
- Standard deviation
- Median
- IQR
- Distance-based model sensitivity
- Gradient-based model sensitivity
- Tree-based model scale independence
```

### Case

```text
1. KNN before vs after scaling
2. SVM before vs after scaling
3. Logistic Regression before vs after scaling
4. PCA before vs after scaling
```

---

## 00.4 Train-Test Split & Cross Validation

```text
Path:
00_foundation/train_test_split_cross_validation.ipynb
```

### Plan

```text
1. Understand why model needs unseen data
2. Split dataset into train and test
3. Split dataset into train, validation, and test
4. Understand random split
5. Understand stratified split
6. Understand K-Fold Cross Validation
7. Understand Stratified K-Fold
8. Understand time-based split
9. Prevent data leakage
10. Compare single split vs cross validation
11. Measure model stability
12. Calculate mean validation score
13. Calculate standard deviation score
14. Choose best validation strategy
```

### Sub Materi

```text
- Train set
- Validation set
- Test set
- Holdout validation
- K-Fold
- Stratified K-Fold
- Time-based split
- Random seed
- Data leakage
- Model stability
- Mean CV score
- Standard deviation CV score
```

### Case

```text
1. Regression K-Fold on house price
2. Stratified K-Fold on credit default
3. Time-based split on stock prediction
4. Leakage example on churn prediction
```

---

# 01_SUPERVISED_LEARNING

# 01.1 REGRESSION

## 01.1.1 Linear Regression

```text
Path:
01_supervised_learning/regression/linear_regression.ipynb
```

### Plan

```text
1. Understand regression problem
2. Understand continuous target
3. Understand linear relationship
4. Learn hypothesis function
5. Learn coefficient and intercept
6. Learn residual
7. Learn MSE loss
8. Learn MAE loss
9. Learn gradient descent
10. Learn normal equation
11. Train simple linear regression
12. Train multiple linear regression
13. Evaluate with MAE
14. Evaluate with MSE
15. Evaluate with RMSE
16. Evaluate with R2
17. Analyze residual
18. Interpret coefficient
19. Detect underfitting
20. Detect overfitting
```

### Sub Materi

```text
- Regression
- Continuous target
- Independent variable
- Dependent variable
- Slope
- Intercept
- Residual
- Error
- Cost function
- MSE
- MAE
- RMSE
- R2 score
- Gradient descent
- Normal equation
- Coefficient interpretation
```

### Case

```text
1. House price prediction
2. Student final score prediction
3. Restaurant rating prediction
4. Delivery time prediction
```

---

## 01.1.2 Polynomial Regression

```text
Path:
01_supervised_learning/regression/polynomial_regression.ipynb
```

### Plan

```text
1. Understand nonlinear relationship
2. Understand limitation of linear regression
3. Create x² feature
4. Create x³ feature
5. Create interaction feature
6. Train polynomial degree 2
7. Train polynomial degree 3
8. Compare train error and test error
9. Understand model complexity
10. Diagnose underfitting
11. Diagnose overfitting
12. Select best polynomial degree
13. Apply regularization if overfit
```

### Sub Materi

```text
- Nonlinear pattern
- Polynomial feature
- Degree
- Interaction term
- Model complexity
- Bias
- Variance
- Underfitting
- Overfitting
- Validation curve
```

### Case

```text
1. Study hours vs final grade
2. Price vs demand
3. Customer count vs restaurant sales
4. Distance vs delivery time
```

---

## 01.1.3 Ridge Regression

```text
Path:
01_supervised_learning/regression/ridge_regression.ipynb
```

### Plan

```text
1. Understand overfitting in linear model
2. Understand regularization
3. Understand L2 penalty
4. Add penalty to MSE loss
5. Train Ridge with small alpha
6. Train Ridge with medium alpha
7. Train Ridge with large alpha
8. Compare coefficient shrinkage
9. Compare train and test error
10. Analyze multicollinearity
11. Use cross validation for alpha
12. Interpret stable coefficient
```

### Sub Materi

```text
- Regularization
- L2 penalty
- Alpha
- Lambda
- Coefficient shrinkage
- Multicollinearity
- Bias-variance tradeoff
- Cross-validation tuning
```

### Case

```text
1. Wine quality prediction
2. Air quality prediction
3. House price prediction with correlated features
4. Student grade prediction
```

---

## 01.1.4 Lasso Regression

```text
Path:
01_supervised_learning/regression/lasso_regression.ipynb
```

### Plan

```text
1. Understand L1 regularization
2. Understand sparse coefficient
3. Understand automatic feature selection
4. Train Lasso with different alpha
5. Observe coefficient becoming zero
6. Compare with Linear Regression
7. Compare with Ridge Regression
8. Select useful features
9. Remove unimportant features
10. Evaluate simpler model
```

### Sub Materi

```text
- L1 penalty
- Sparse model
- Feature selection
- Coefficient zeroing
- Subgradient
- Alpha tuning
- Interpretability
```

### Case

```text
1. Wine quality feature selection
2. Credit score feature selection
3. Churn feature selection
4. Air quality pollutant selection
```

---

## 01.1.5 Elastic Net

```text
Path:
01_supervised_learning/regression/elastic_net.ipynb
```

### Plan

```text
1. Understand limitation of Ridge
2. Understand limitation of Lasso
3. Combine L1 and L2 penalty
4. Tune alpha
5. Tune L1 ratio
6. Compare Ridge vs Lasso vs Elastic Net
7. Analyze correlated features
8. Analyze selected features
9. Evaluate performance
10. Interpret final model
```

### Sub Materi

```text
- L1 regularization
- L2 regularization
- Elastic Net
- Alpha
- L1 ratio
- Sparse coefficient
- Stable coefficient
- Correlated feature handling
```

### Case

```text
1. High-dimensional tabular regression
2. Wine quality prediction
3. Air quality prediction
4. Product rating prediction
```

---

# 01.2 CLASSIFICATION

## 01.2.1 Logistic Regression

```text
Path:
01_supervised_learning/classification/logistic_regression.ipynb
```

### Plan

```text
1. Understand binary classification
2. Understand probability output
3. Understand sigmoid function
4. Understand logit
5. Understand decision threshold
6. Learn binary cross entropy
7. Learn Bernoulli log-likelihood
8. Train Logistic Regression
9. Predict probability
10. Convert probability into class
11. Tune classification threshold
12. Evaluate confusion matrix
13. Evaluate accuracy
14. Evaluate precision
15. Evaluate recall
16. Evaluate F1-score
17. Interpret coefficient
18. Interpret odds ratio
19. Diagnose false positive
20. Diagnose false negative
```

### Sub Materi

```text
- Binary classification
- Sigmoid
- Logit
- Probability
- Decision threshold
- Binary cross entropy
- Bernoulli likelihood
- Log loss
- Odds ratio
- Confusion matrix
- Precision
- Recall
- F1-score
```

### Case

```text
1. Credit default prediction
2. Churn prediction
3. Disease prediction
4. Purchase prediction
5. Student dropout prediction
```

---

## 01.2.2 KNN Classifier

```text
Path:
01_supervised_learning/classification/knn_classifier.ipynb
```

### Plan

```text
1. Understand instance-based learning
2. Understand distance-based prediction
3. Learn Euclidean distance
4. Learn Manhattan distance
5. Understand K nearest neighbors
6. Understand majority voting
7. Understand weighted voting
8. Train KNN with small K
9. Train KNN with large K
10. Compare decision boundary
11. Understand effect of scaling
12. Understand effect of irrelevant features
13. Tune K using validation
14. Evaluate classification metrics
```

### Sub Materi

```text
- Instance-based learning
- Distance metric
- Euclidean distance
- Manhattan distance
- Cosine distance
- K value
- Majority voting
- Weighted voting
- Curse of dimensionality
- Scaling sensitivity
```

### Case

```text
1. Digit classification
2. Fruit classification
3. Customer group classification
4. Medical classification baseline
```

---

## 01.2.3 Decision Tree Classifier

```text
Path:
01_supervised_learning/classification/decision_tree.ipynb
```

### Plan

```text
1. Understand tree-based decision
2. Understand root node
3. Understand internal node
4. Understand leaf node
5. Learn Gini impurity
6. Learn entropy
7. Learn information gain
8. Learn recursive splitting
9. Build tree manually
10. Control max depth
11. Control minimum samples split
12. Detect overfitting
13. Prune tree conceptually
14. Interpret decision path
15. Extract feature importance
```

### Sub Materi

```text
- Decision tree
- Root node
- Leaf node
- Split rule
- Gini impurity
- Entropy
- Information gain
- Recursive split
- Max depth
- Min samples split
- Pruning
- Feature importance
```

### Case

```text
1. Loan approval prediction
2. Student dropout prediction
3. Marketing response prediction
4. Fraud detection
```

---

## 01.2.4 Gaussian Naive Bayes

```text
Path:
01_supervised_learning/classification/naive_bayes/gaussian_nb.ipynb
```

### Plan

```text
1. Understand Bayes theorem
2. Understand prior probability
3. Understand likelihood
4. Understand posterior probability
5. Understand conditional independence
6. Understand Gaussian assumption
7. Calculate class prior
8. Calculate mean per class
9. Calculate variance per class
10. Calculate likelihood
11. Calculate posterior
12. Predict class
13. Evaluate model
14. Analyze independence assumption
```

### Sub Materi

```text
- Bayes theorem
- Prior
- Likelihood
- Posterior
- Gaussian distribution
- Conditional probability
- Independence assumption
- Mean per class
- Variance per class
- Log probability
```

### Case

```text
1. Disease prediction
2. Credit risk classification
3. Audio gender classification
4. Wine class prediction
```

---

## 01.2.5 Multinomial Naive Bayes

```text
Path:
01_supervised_learning/classification/naive_bayes/multinomial_nb.ipynb
```

### Plan

```text
1. Understand text classification
2. Build vocabulary
3. Convert text to word count
4. Calculate class prior
5. Calculate word frequency per class
6. Apply Laplace smoothing
7. Calculate word likelihood
8. Calculate document likelihood
9. Use log probability
10. Predict class
11. Evaluate text classifier
12. Analyze important words
```

### Sub Materi

```text
- Text classification
- Vocabulary
- Bag of Words
- Word count
- Class prior
- Word likelihood
- Laplace smoothing
- Log probability
- Multinomial distribution
```

### Case

```text
1. Spam email classification
2. News category classification
3. Sentiment classification
4. Toxic comment classification
```

---

## 01.2.6 Linear SVM

```text
Path:
01_supervised_learning/classification/svm/linear_svm.ipynb
```

### Plan

```text
1. Understand margin-based classification
2. Understand separating hyperplane
3. Understand support vectors
4. Understand maximum margin
5. Learn hinge loss
6. Learn regularization C
7. Train linear SVM
8. Compare with Logistic Regression
9. Analyze margin
10. Analyze misclassified points
11. Evaluate classification metrics
```

### Sub Materi

```text
- Hyperplane
- Margin
- Support vector
- Hinge loss
- Regularization C
- Linear boundary
- Max margin classifier
```

### Case

```text
1. Spam classification with TF-IDF
2. Disease classification
3. Credit default classification
4. Image feature classification
```

---

## 01.2.7 Kernel SVM

```text
Path:
01_supervised_learning/classification/svm/kernel_svm.ipynb
```

### Plan

```text
1. Understand nonlinear classification
2. Understand limitation of linear SVM
3. Understand kernel trick
4. Learn polynomial kernel
5. Learn RBF kernel
6. Learn gamma parameter
7. Train polynomial kernel SVM
8. Train RBF kernel SVM
9. Compare decision boundary
10. Diagnose overfitting
11. Tune C and gamma
12. Evaluate model
```

### Sub Materi

```text
- Kernel trick
- Polynomial kernel
- RBF kernel
- Gamma
- C parameter
- Nonlinear boundary
- Support vectors
- Overfitting in kernel model
```

### Case

```text
1. Nonlinear synthetic classification
2. Medical nonlinear classification
3. Image feature classification
4. Customer churn nonlinear pattern
```

---

# 02_UNSUPERVISED_LEARNING

# 02.1 CLUSTERING

## 02.1.1 K-Means

```text
Path:
02_unsupervised_learning/clustering/k_means.ipynb
```

### Plan

```text
1. Understand unlabeled data
2. Understand clustering goal
3. Understand centroid
4. Initialize centroids
5. Calculate distance to centroid
6. Assign points to nearest centroid
7. Update centroid
8. Repeat until convergence
9. Calculate inertia
10. Use elbow method
11. Analyze cluster quality
12. Interpret each cluster
```

### Sub Materi

```text
- Unsupervised learning
- Cluster
- Centroid
- Euclidean distance
- Assignment step
- Update step
- Inertia
- Elbow method
- Cluster interpretation
```

### Case

```text
1. Customer segmentation
2. Product segmentation
3. Student behavior grouping
4. Image color clustering
```

---

## 02.1.2 Hierarchical Clustering

```text
Path:
02_unsupervised_learning/clustering/hierarchical_clustering.ipynb
```

### Plan

```text
1. Understand hierarchy-based clustering
2. Calculate distance matrix
3. Understand agglomerative clustering
4. Merge closest points
5. Merge closest clusters
6. Learn single linkage
7. Learn complete linkage
8. Learn average linkage
9. Learn Ward linkage
10. Build dendrogram
11. Choose cluster cut
12. Interpret hierarchy
```

### Sub Materi

```text
- Agglomerative clustering
- Distance matrix
- Single linkage
- Complete linkage
- Average linkage
- Ward linkage
- Dendrogram
- Cluster hierarchy
```

### Case

```text
1. Document similarity grouping
2. Customer hierarchy segmentation
3. Product hierarchy grouping
4. Small medical patient clustering
```

---

## 02.1.3 DBSCAN

```text
Path:
02_unsupervised_learning/clustering/dbscan.ipynb
```

### Plan

```text
1. Understand density-based clustering
2. Understand epsilon radius
3. Understand min samples
4. Identify core points
5. Identify border points
6. Identify noise points
7. Expand cluster from core point
8. Detect outliers
9. Tune epsilon
10. Tune min samples
11. Compare with K-Means
12. Interpret noise as anomaly
```

### Sub Materi

```text
- Density clustering
- Epsilon
- Min samples
- Core point
- Border point
- Noise point
- Arbitrary shape cluster
- Outlier detection
```

### Case

```text
1. Fraud anomaly grouping
2. GPS location clustering
3. Customer anomaly detection
4. Network traffic anomaly
```

---

## 02.1.4 Gaussian Mixture Model

```text
Path:
02_unsupervised_learning/clustering/gaussian_mixture_model.ipynb
```

### Plan

```text
1. Understand probabilistic clustering
2. Understand Gaussian component
3. Understand mixture weight
4. Understand soft assignment
5. Initialize Gaussian parameters
6. Perform expectation step
7. Perform maximization step
8. Calculate log-likelihood
9. Repeat until convergence
10. Predict cluster probability
11. Compare with K-Means
12. Interpret uncertainty
```

### Sub Materi

```text
- Gaussian distribution
- Mixture model
- Soft clustering
- Expectation-Maximization
- Cluster probability
- Log-likelihood
- AIC
- BIC
```

### Case

```text
1. Customer segmentation with overlap
2. Medical patient subtype discovery
3. Fraud probability cluster
4. Market behavior grouping
```

---

# 02.2 DIMENSIONALITY REDUCTION

## 02.2.1 PCA

```text
Path:
02_unsupervised_learning/dimensionality_reduction/pca.ipynb
```

### Plan

```text
1. Understand high-dimensional data
2. Understand feature redundancy
3. Standardize data
4. Calculate covariance matrix
5. Calculate eigenvalues
6. Calculate eigenvectors
7. Sort principal components
8. Calculate explained variance
9. Select top components
10. Transform data
11. Visualize 2D projection
12. Analyze information loss
```

### Sub Materi

```text
- Curse of dimensionality
- Covariance matrix
- Eigenvalue
- Eigenvector
- Principal component
- Explained variance
- Projection
- Reconstruction error
```

### Case

```text
1. PCA on medical features
2. PCA on image pixels
3. PCA on audio features
4. PCA before Logistic Regression
```

---

## 02.2.2 LDA

```text
Path:
02_unsupervised_learning/dimensionality_reduction/lda.ipynb
```

### Plan

```text
1. Understand supervised dimensionality reduction
2. Understand class separation
3. Calculate class mean
4. Calculate overall mean
5. Calculate within-class scatter
6. Calculate between-class scatter
7. Find projection direction
8. Transform data
9. Visualize class separation
10. Compare PCA vs LDA
11. Train classifier after LDA
```

### Sub Materi

```text
- Class mean
- Overall mean
- Within-class scatter
- Between-class scatter
- Projection
- Class separability
- Supervised projection
```

### Case

```text
1. Breast cancer class separation
2. Credit score class separation
3. Digit classification visualization
4. Wine class projection
```

---

## 02.2.3 t-SNE

```text
Path:
02_unsupervised_learning/dimensionality_reduction/tsne.ipynb
```

### Plan

```text
1. Understand nonlinear visualization
2. Understand high-dimensional similarity
3. Understand low-dimensional similarity
4. Understand local neighborhood
5. Understand perplexity
6. Understand KL divergence
7. Visualize dataset in 2D
8. Analyze cluster separation
9. Compare with PCA
10. Understand limitation of t-SNE
```

### Sub Materi

```text
- Nonlinear projection
- Pairwise similarity
- Local structure
- Perplexity
- KL divergence
- 2D embedding
- Visualization limitation
```

### Case

```text
1. Visualize digit data
2. Visualize text TF-IDF data
3. Visualize customer clusters
4. Visualize medical data
```

---

## 02.2.4 UMAP

```text
Path:
02_unsupervised_learning/dimensionality_reduction/umap.ipynb
```

### Plan

```text
1. Understand manifold learning
2. Understand nearest neighbor graph
3. Understand local structure
4. Understand global structure
5. Tune n_neighbors
6. Tune min_dist
7. Generate 2D embedding
8. Compare with PCA
9. Compare with t-SNE
10. Interpret visualization
```

### Sub Materi

```text
- Manifold learning
- Neighbor graph
- Local structure
- Global structure
- n_neighbors
- min_dist
- 2D embedding
```

### Case

```text
1. Customer behavior visualization
2. Text document visualization
3. Image feature visualization
4. Medical feature visualization
```

---

# 03_ENSEMBLE_LEARNING

## 03.1 Bagging

```text
Path:
03_ensemble_learning/bagging.ipynb
```

### Plan

```text
1. Understand weak model instability
2. Understand bootstrap sampling
3. Create multiple training subsets
4. Train multiple base models
5. Aggregate predictions
6. Use voting for classification
7. Use averaging for regression
8. Compare single model vs bagging
9. Analyze variance reduction
10. Evaluate stability
```

### Sub Materi

```text
- Bootstrap
- Base learner
- Ensemble
- Voting
- Averaging
- Variance reduction
- Model diversity
```

### Case

```text
1. Bagging Decision Tree for churn
2. Bagging Regression for house price
3. Bagging for credit risk
4. Bagging for fraud detection
```

---

## 03.2 Random Forest

```text
Path:
03_ensemble_learning/random_forest.ipynb
```

### Plan

```text
1. Understand Decision Tree overfitting
2. Understand bagging trees
3. Understand bootstrap sample
4. Understand random feature subset
5. Train multiple trees
6. Aggregate predictions
7. Calculate feature importance
8. Compare with Decision Tree
9. Analyze variance reduction
10. Tune number of trees
11. Tune max depth
12. Evaluate generalization
```

### Sub Materi

```text
- Bagging
- Bootstrap
- Random feature subset
- Decision tree ensemble
- Majority voting
- Averaging
- Feature importance
- OOB error concept
```

### Case

```text
1. Credit default prediction
2. Fraud detection
3. Churn prediction
4. Wine quality prediction
5. Air quality prediction
```

---

## 03.3 Gradient Boosting

```text
Path:
03_ensemble_learning/gradient_boosting.ipynb
```

### Plan

```text
1. Understand boosting concept
2. Understand sequential learning
3. Start with simple prediction
4. Calculate residual
5. Train weak learner on residual
6. Update prediction
7. Repeat boosting iteration
8. Tune learning rate
9. Tune number of estimators
10. Tune tree depth
11. Analyze loss curve
12. Diagnose overfitting
```

### Sub Materi

```text
- Boosting
- Weak learner
- Residual
- Additive model
- Learning rate
- Number of estimators
- Shallow tree
- Loss reduction
- Bias reduction
```

### Case

```text
1. Restaurant sales prediction
2. Air quality prediction
3. Wine quality prediction
4. Churn prediction
```

---

## 03.4 Stacking & Voting

```text
Path:
03_ensemble_learning/stacking_voting.ipynb
```

### Plan

```text
1. Understand model diversity
2. Train multiple different models
3. Compare individual models
4. Implement hard voting
5. Implement soft voting
6. Understand meta-model
7. Generate out-of-fold predictions
8. Train stacking model
9. Prevent leakage in stacking
10. Compare voting vs stacking
11. Interpret ensemble result
```

### Sub Materi

```text
- Ensemble diversity
- Hard voting
- Soft voting
- Meta learner
- Out-of-fold prediction
- Stacking
- Blending
- Leakage prevention
```

### Case

```text
1. Ensemble for breast cancer classification
2. Ensemble for credit risk
3. Ensemble for churn prediction
4. Ensemble benchmark across models
```

---

## 03.5 XGBoost & LightGBM Concept

```text
Path:
03_ensemble_learning/xgboost_lightgbm.ipynb
```

### Plan

```text
1. Understand advanced gradient boosting
2. Understand regularized objective
3. Understand tree boosting
4. Understand shrinkage
5. Understand row subsampling
6. Understand column subsampling
7. Understand split gain concept
8. Understand pruning concept
9. Understand histogram-based split
10. Compare concept with manual Gradient Boosting
11. Implement simplified boosting version
12. Analyze why boosting dominates tabular data
```

### Sub Materi

```text
- Gradient boosted trees
- Regularized objective
- Shrinkage
- Subsampling
- Column sampling
- Split gain
- Tree pruning
- Histogram split
- Leaf-wise growth
```

### Case

```text
1. Credit score classification
2. Churn prediction
3. Fraud detection
4. Air quality prediction
```

---

# 04_INSTANCE_BASED_LEARNING

## 04.1 KNN Recommender System

```text
Path:
04_instance_based_learning/knn_recommender_system.ipynb
```

### Plan

```text
1. Understand recommendation problem
2. Understand user-item interaction
3. Build user-item matrix
4. Understand sparse matrix
5. Calculate user similarity
6. Calculate item similarity
7. Use cosine similarity
8. Use Pearson similarity
9. Find nearest users
10. Find nearest items
11. Predict rating
12. Generate Top-N recommendation
13. Evaluate recommendation
14. Analyze cold start problem
```

### Sub Materi

```text
- User
- Item
- Rating
- Interaction
- Explicit feedback
- Implicit feedback
- User-item matrix
- Sparsity
- Collaborative filtering
- Content-based filtering
- Cosine similarity
- Pearson similarity
- Top-N recommendation
- Precision@K
- Recall@K
```

### Case

```text
1. Movie recommendation
2. Product recommendation
3. Restaurant recommendation
4. Course recommendation
5. Book recommendation
```

---

# 05_MODEL_EVALUATION

## 05.1 Bias-Variance Tradeoff

```text
Path:
05_model_evaluation/bias_variance_tradeoff.ipynb
```

### Plan

```text
1. Understand model error
2. Understand bias
3. Understand variance
4. Understand irreducible error
5. Train simple model
6. Train complex model
7. Compare train error
8. Compare validation error
9. Build learning curve
10. Build validation curve
11. Diagnose underfitting
12. Diagnose overfitting
13. Apply regularization
14. Apply model complexity control
15. Choose balanced model
```

### Sub Materi

```text
- Bias
- Variance
- Irreducible error
- Underfitting
- Overfitting
- Learning curve
- Validation curve
- Model complexity
- Regularization
- Cross validation
```

### Case

```text
1. Polynomial degree comparison
2. Decision Tree max depth comparison
3. Ridge alpha comparison
4. KNN K value comparison
```

---

## 05.2 Classification Metrics

```text
Path:
05_model_evaluation/classification_metrics.ipynb
```

### Plan

```text
1. Understand classification output
2. Understand confusion matrix
3. Calculate TP
4. Calculate TN
5. Calculate FP
6. Calculate FN
7. Calculate accuracy
8. Calculate precision
9. Calculate recall
10. Calculate specificity
11. Calculate F1-score
12. Calculate macro average
13. Calculate micro average
14. Calculate weighted average
15. Analyze imbalanced classification
16. Choose metric based on problem
```

### Sub Materi

```text
- Confusion matrix
- True positive
- True negative
- False positive
- False negative
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- Macro average
- Micro average
- Weighted average
- Class imbalance
```

### Case

```text
1. Fraud detection metric selection
2. Breast cancer recall analysis
3. Spam precision analysis
4. Churn F1-score analysis
```

---

## 05.3 Regression Metrics

```text
Path:
05_model_evaluation/regression_metrics.ipynb
```

### Plan

```text
1. Understand regression error
2. Calculate residual
3. Calculate MAE
4. Calculate MSE
5. Calculate RMSE
6. Calculate R2
7. Calculate Adjusted R2
8. Analyze residual distribution
9. Analyze large error
10. Compare models using metrics
11. Choose metric based on case
```

### Sub Materi

```text
- Residual
- Absolute error
- Squared error
- MAE
- MSE
- RMSE
- R2
- Adjusted R2
- Residual plot
- Error distribution
```

### Case

```text
1. House price prediction evaluation
2. Wine quality prediction evaluation
3. Air quality prediction evaluation
4. Delivery time prediction evaluation
```

---

## 05.4 ROC-AUC & Confusion Matrix

```text
Path:
05_model_evaluation/roc_auc_confusion_matrix.ipynb
```

### Plan

```text
1. Understand probability prediction
2. Understand threshold
3. Generate predictions at different thresholds
4. Calculate TPR
5. Calculate FPR
6. Build ROC curve
7. Calculate AUC
8. Build precision-recall curve
9. Compare ROC-AUC and PR-AUC
10. Tune threshold
11. Analyze false positive cost
12. Analyze false negative cost
13. Select threshold based on risk
```

### Sub Materi

```text
- Probability output
- Threshold
- TPR
- FPR
- ROC curve
- AUC
- Precision-recall curve
- PR-AUC
- Threshold tuning
- Cost-sensitive evaluation
```

### Case

```text
1. Breast cancer threshold tuning
2. Credit default risk threshold
3. Fraud detection PR-AUC
4. Churn campaign threshold
```

---

# 06_REAL_WORLD_CASE_STUDY

## 06.1 Air Quality Prediction

```text
Path:
06_real_world_case_study/air_quality_prediction.ipynb
```

### Plan

```text
1. Understand air pollution problem
2. Define target as AQI regression
3. Understand pollutant features
4. Analyze missing sensor values
5. Analyze outlier pollutant values
6. Analyze correlation between pollutants and AQI
7. Engineer time/weather features
8. Train baseline Linear Regression
9. Train Ridge Regression
10. Train Decision Tree
11. Train Random Forest
12. Train Gradient Boosting
13. Evaluate MAE/RMSE/R2
14. Diagnose bias-variance
15. Interpret most important pollutants
16. Build final insight
```

### Sub Materi

```text
- Regression
- Sensor data
- Missing value
- Outlier
- Correlation
- Multicollinearity
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Regression metrics
```

### Case Output

```text
- Predict AQI value
- Identify pollutant impact
- Compare linear vs tree-based model
- Recommend best model
```

---

## 06.2 Breast Cancer Classification

```text
Path:
06_real_world_case_study/breast_cancer_classification.ipynb
```

### Plan

```text
1. Understand medical classification problem
2. Define target benign vs malignant
3. Analyze class distribution
4. Analyze feature distribution
5. Analyze feature correlation
6. Handle scaling
7. Train Logistic Regression
8. Train Gaussian Naive Bayes
9. Train Linear SVM
10. Train Decision Tree
11. Train Random Forest
12. Evaluate accuracy
13. Evaluate precision
14. Evaluate recall
15. Evaluate F1-score
16. Evaluate ROC-AUC
17. Focus false negative analysis
18. Tune threshold
19. Interpret important features
20. Build final medical insight
```

### Sub Materi

```text
- Binary classification
- Medical risk
- Recall priority
- False negative
- Logistic Regression
- Gaussian Naive Bayes
- Linear SVM
- Decision Tree
- Random Forest
- ROC-AUC
- Threshold tuning
```

### Case Output

```text
- Predict benign/malignant
- Reduce false negative
- Compare interpretable vs complex model
- Recommend threshold
```

---

## 06.3 Churn Prediction

```text
Path:
06_real_world_case_study/churn_prediction.ipynb
```

### Plan

```text
1. Understand churn business problem
2. Define target churn vs not churn
3. Analyze customer behavior data
4. Analyze tenure
5. Analyze monthly charge
6. Analyze contract type
7. Analyze support ticket
8. Encode categorical features
9. Scale numerical features
10. Train Logistic Regression
11. Train Decision Tree
12. Train Random Forest
13. Train Gradient Boosting
14. Evaluate precision/recall/F1
15. Evaluate ROC-AUC
16. Tune campaign threshold
17. Analyze high-risk customer segment
18. Interpret churn drivers
19. Build retention recommendation
```

### Sub Materi

```text
- Binary classification
- Customer analytics
- Categorical encoding
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- ROC-AUC
- Threshold tuning
- Business interpretation
```

### Case Output

```text
- Predict churn probability
- Identify churn risk factors
- Segment high-risk users
- Recommend retention action
```

---

## 06.4 Credit Score Classification

```text
Path:
06_real_world_case_study/credit_score_classification.ipynb
```

### Plan

```text
1. Understand credit risk problem
2. Define target poor/standard/good
3. Analyze income
4. Analyze debt ratio
5. Analyze loan amount
6. Analyze payment delay
7. Analyze previous default
8. Detect financial outliers
9. Encode categorical features
10. Scale numerical features
11. Train Logistic Regression multiclass
12. Train Gaussian Naive Bayes
13. Train Decision Tree
14. Train Random Forest
15. Train SVM
16. Evaluate macro F1
17. Evaluate weighted F1
18. Evaluate confusion matrix
19. Analyze class-wise error
20. Interpret risk factor
21. Build final credit risk insight
```

### Sub Materi

```text
- Multiclass classification
- Credit risk
- Financial outlier
- Logistic Regression multiclass
- Softmax Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- SVM
- Macro F1
- Weighted F1
- Class-wise recall
```

### Case Output

```text
- Predict credit score class
- Analyze poor-credit recall
- Identify risk factors
- Recommend model for credit scoring
```

---

## 06.5 Wine Quality Prediction

```text
Path:
06_real_world_case_study/wine_quality_prediction.ipynb
```

### Plan

```text
1. Understand wine quality problem
2. Define target quality score
3. Analyze chemical features
4. Analyze alcohol feature
5. Analyze acidity feature
6. Analyze sulphates feature
7. Analyze feature correlation
8. Detect chemical outliers
9. Compare regression vs classification framing
10. Train Linear Regression
11. Train Ridge Regression
12. Train Lasso Regression
13. Train Decision Tree
14. Train Random Forest
15. Train Gradient Boosting
16. Evaluate MAE/RMSE/R2
17. Evaluate class metrics if categorized
18. Interpret chemical impact
19. Build final insight
```

### Sub Materi

```text
- Regression
- Optional multiclass classification
- Chemical feature analysis
- Correlation
- Multicollinearity
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Regression metrics
```

### Case Output

```text
- Predict wine quality score
- Identify strongest chemical features
- Compare regression vs classification
- Recommend best modeling approach
```

---

# ADDITIONAL CORE ALGORITHMS TO ADD BEFORE DEEP LEARNING

## A. Softmax Regression

```text
Recommended Path:
01_supervised_learning/classification/softmax_regression.ipynb
```

### Plan

```text
1. Understand multiclass classification
2. Understand one-hot label
3. Understand softmax function
4. Understand multiclass cross entropy
5. Compute class probabilities
6. Train with gradient descent
7. Predict class
8. Evaluate macro F1
9. Evaluate confusion matrix
10. Compare with One-vs-Rest Logistic Regression
```

### Case

```text
1. Digit classification
2. News category classification
3. Credit score classification
4. Wine quality class prediction
```

---

## B. KNN Regressor

```text
Recommended Path:
01_supervised_learning/regression/knn_regressor.ipynb
```

### Plan

```text
1. Understand regression using neighbors
2. Calculate distance
3. Select K nearest samples
4. Average target values
5. Use weighted average
6. Compare different K
7. Analyze scaling effect
8. Evaluate RMSE
```

### Case

```text
1. House price prediction
2. Restaurant rating prediction
3. Delivery time prediction
4. Wine quality prediction
```

---

## C. Decision Tree Regressor

```text
Recommended Path:
01_supervised_learning/regression/decision_tree_regressor.ipynb
```

### Plan

```text
1. Understand tree for regression
2. Understand MSE split
3. Find best split
4. Create leaf prediction using mean target
5. Control max depth
6. Control min samples
7. Detect overfitting
8. Compare with Linear Regression
```

### Case

```text
1. Restaurant sales prediction
2. House price prediction
3. Delivery time prediction
4. Air quality prediction
```

---

## D. AdaBoost

```text
Recommended Path:
03_ensemble_learning/adaboost.ipynb
```

### Plan

```text
1. Understand boosting with sample weights
2. Train weak classifier
3. Calculate weighted error
4. Calculate model weight
5. Increase weight for wrong samples
6. Decrease weight for correct samples
7. Combine weak classifiers
8. Evaluate final weighted voting
```

### Case

```text
1. Fraud detection
2. Churn prediction
3. Breast cancer classification
4. Credit default prediction
```

---

## E. Bernoulli Naive Bayes

```text
Recommended Path:
01_supervised_learning/classification/naive_bayes/bernoulli_nb.ipynb
```

### Plan

```text
1. Understand binary feature
2. Convert text to word presence
3. Calculate class prior
4. Calculate probability of word present per class
5. Apply Laplace smoothing
6. Predict using log probability
7. Compare with Multinomial Naive Bayes
```

### Case

```text
1. Spam classification
2. Toxic comment classification
3. Fake news classification
4. Sentiment classification
```

---

## F. Anomaly Detection

```text
Recommended Path:
02_unsupervised_learning/anomaly_detection/
```

### Plan

```text
1. Understand anomaly problem
2. Detect anomaly using Z-score
3. Detect anomaly using IQR
4. Detect anomaly using distance to centroid
5. Detect anomaly using KNN distance
6. Detect anomaly using DBSCAN noise
7. Detect anomaly using GMM low probability
8. Evaluate anomaly detection
```

### Case

```text
1. Fraud transaction anomaly
2. Network intrusion anomaly
3. Credit risk anomaly
4. Sensor anomaly
```

---

# FULL LEARNING ORDER FOR THIS REPOSITORY

```text
1. data_preprocessing
2. feature_engineering
3. scaling_normalization
4. train_test_split_cross_validation

5. linear_regression
6. polynomial_regression
7. ridge_regression
8. lasso_regression
9. elastic_net
10. knn_regressor
11. decision_tree_regressor

12. logistic_regression
13. softmax_regression
14. knn_classifier
15. decision_tree
16. gaussian_nb
17. multinomial_nb
18. bernoulli_nb
19. linear_svm
20. kernel_svm

21. regression_metrics
22. classification_metrics
23. roc_auc_confusion_matrix
24. bias_variance_tradeoff

25. k_means
26. hierarchical_clustering
27. dbscan
28. gaussian_mixture_model
29. anomaly_detection

30. pca
31. lda
32. tsne
33. umap

34. bagging
35. random_forest
36. gradient_boosting
37. adaboost
38. stacking_voting
39. xgboost_lightgbm_concept

40. knn_recommender_system

41. air_quality_prediction
42. breast_cancer_classification
43. churn_prediction
44. credit_score_classification
45. wine_quality_prediction
```

---

# FINAL MASTER PLAN BEFORE DEEP LEARNING

```text
Stage 1:
Understand data, preprocessing, feature engineering, scaling, splitting.

Stage 2:
Master regression algorithms:
Linear Regression, Polynomial Regression, Ridge, Lasso, Elastic Net, KNN Regressor, Decision Tree Regressor.

Stage 3:
Master classification algorithms:
Logistic Regression, Softmax Regression, KNN, Decision Tree, Gaussian NB, Multinomial NB, Bernoulli NB, Linear SVM, Kernel SVM.

Stage 4:
Master evaluation:
Regression metrics, classification metrics, confusion matrix, ROC-AUC, PR curve, threshold tuning, bias-variance.

Stage 5:
Master unsupervised learning:
K-Means, Hierarchical, DBSCAN, GMM, anomaly detection.

Stage 6:
Master dimensionality reduction:
PCA, LDA, t-SNE, UMAP.

Stage 7:
Master ensemble learning:
Bagging, Random Forest, Gradient Boosting, AdaBoost, Voting, Stacking, XGBoost/LightGBM concept.

Stage 8:
Master instance-based recommendation:
KNN recommender, user-based filtering, item-based filtering, cosine similarity.

Stage 9:
Apply real-world case studies:
Air quality, breast cancer, churn, credit score, wine quality.

Stage 10:
Only after this, enter Deep Learning:
Perceptron, MLP, backpropagation, CNN, RNN, Transformer.
```
