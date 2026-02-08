# 🎯 IDE PROJECT + TARGET PEMAHAMAN SETIAP ALGORITMA

## 📘 00_FOUNDATION

### 1. Gradient Descent

**5 Ide Project:**
* project → Optimasi rute pengiriman paket
* project → Training model linear regression manual
* project → Optimasi portfolio investasi
* project → Tuning hyperparameter neural network
* project → Minimalisasi biaya produksi pabrik

**🎯 Target Pemahaman:**
* ✅ Paham kenapa learning rate terlalu besar bikin divergen, terlalu kecil bikin lambat
* ✅ Bisa jelaskan perbedaan batch, mini-batch, stochastic GD (kapan pakai mana)
* ✅ Mengerti konsep local minima vs global minima
* ✅ Paham momentum & adaptive learning rate (Adam, RMSprop)
* ✅ Bisa visualisasikan loss surface & trajectory optimasi
* ✅ Tahu kapan GD gagal (non-convex, saddle point, plateau)

---

### 2. Distance Metrics

**5 Ide Project:**
* project → Sistem rekomendasi film berdasarkan rating
* project → Face recognition similarity checker
* project → Deteksi plagiarisme dokumen
* project → Product recommendation engine
* project → DNA sequence similarity analysis

**🎯 Target Pemahaman:**
* ✅ Tahu kapan pakai Euclidean (magnitude matters), Manhattan (grid-based), Cosine (direction matters)
* ✅ Paham kenapa cosine bagus untuk text, euclidean untuk spatial data
* ✅ Mengerti curse of dimensionality & dampaknya ke distance
* ✅ Bisa jelaskan kenapa normalisasi penting sebelum hitung distance
* ✅ Paham Minkowski distance sebagai generalisasi
* ✅ Tahu kapan distance metric gagal (high-dim, sparse data)

---

## 📘 01_SUPERVISED_LEARNING

### 3. Linear Regression

**5 Ide Project:**
* project → Prediksi harga rumah
* project → Prediksi salary berdasarkan pengalaman
* project → Forecasting penjualan bulanan
* project → Prediksi konsumsi listrik
* project → Estimasi biaya marketing vs revenue

**🎯 Target Pemahaman:**
* ✅ Paham asumsi linear relationship & kapan asumsi ini gagal
* ✅ Bisa implementasi closed-form solution (Normal Equation) vs Gradient Descent
* ✅ Mengerti trade-off: closed-form cepat tapi ga scalable, GD lambat tapi scalable
* ✅ Paham R², MSE, RMSE & interpretasinya
* ✅ Tahu kapan linear regression cocok (linear trend, continuous target)
* ✅ Bisa deteksi multicollinearity & dampaknya ke coefficient

---

### 4. Polynomial Regression

**5 Ide Project:**
* project → Prediksi pertumbuhan populasi
* project → Modeling kurva belajar siswa
* project → Prediksi suhu harian
* project → Trajectory prediction bola basket
* project → Economic growth forecasting

**🎯 Target Pemahaman:**
* ✅ Paham kenapa degree tinggi → overfitting, degree rendah → underfitting
* ✅ Bisa pilih degree optimal pakai validation curve
* ✅ Mengerti bahwa polynomial regression = linear regression dengan feature engineering
* ✅ Tahu kapan polynomial lebih baik dari linear (non-linear pattern)
* ✅ Paham feature scaling wajib untuk polynomial features
* ✅ Bisa jelaskan trade-off complexity vs interpretability

---

### 5. Ridge Regression (L2 Regularization)

**5 Ide Project:**
* project → Prediksi harga saham dengan banyak fitur
* project → Medical cost prediction dengan multicollinearity
* project → Real estate valuation dengan fitur berkorelasi
* project → Student performance prediction
* project → Energy consumption forecasting

**🎯 Target Pemahaman:**
* ✅ Paham kenapa L2 shrink weights tapi ga bikin 0 (smooth penalty)
* ✅ Bisa jelaskan efek lambda: besar → underfitting, kecil → overfitting
* ✅ Mengerti kapan pakai Ridge (banyak fitur berkorelasi, multicollinearity)
* ✅ Tahu cara pilih alpha optimal (cross-validation)
* ✅ Paham kenapa Ridge lebih stabil dari Linear Regression
* ✅ Bisa visualisasikan weight shrinkage effect

---

### 6. Lasso Regression (L1 Regularization)

**5 Ide Project:**
* project → Feature selection untuk prediksi diabetes
* project → Identifikasi faktor penting penjualan
* project → Customer churn prediction dengan auto-feature selection
* project → Gene selection untuk disease prediction
* project → Sensor data filtering untuk IoT

**🎯 Target Pemahaman:**
* ✅ Paham kenapa L1 bisa bikin weight = 0 (sparse solution)
* ✅ Bisa jelaskan perbedaan L1 vs L2: Lasso → feature selection, Ridge → shrinkage
* ✅ Mengerti kapan pakai Lasso (butuh interpretability, banyak fitur irrelevant)
* ✅ Tahu Elastic Net = kombinasi L1 + L2
* ✅ Paham geometri L1 penalty (diamond shape) vs L2 (circle)
* ✅ Bisa deteksi fitur penting dari coefficient path

---

### 7. Logistic Regression

**5 Ide Project:**
* project → Email spam classifier
* project → Customer churn prediction
* project → Loan default prediction
* project → Disease diagnosis (diabetes/heart)
* project → Click-through rate prediction

**🎯 Target Pemahaman:**
* ✅ Paham kenapa pakai sigmoid (output probabilitas 0-1)
* ✅ Bisa jelaskan log-loss / cross-entropy loss
* ✅ Mengerti decision boundary & threshold tuning
* ✅ Tahu perbedaan linear regression (continuous) vs logistic (binary)
* ✅ Paham class imbalance problem & solusinya (SMOTE, class weight)
* ✅ Bisa interpretasi coefficient sebagai log-odds

---

### 8. K-Nearest Neighbors (KNN)

**5 Ide Project:**
* project → Handwritten digit recognition
* project → Movie recommendation system
* project → Credit risk assessment
* project → Plant species classification
* project → Medical diagnosis berdasarkan symptoms

**🎯 Target Pemahaman:**
* ✅ Paham lazy learning (no training phase)
* ✅ Bisa jelaskan bias-variance trade-off: K kecil → high variance, K besar → high bias
* ✅ Mengerti kenapa scaling/normalisasi wajib
* ✅ Tahu computational cost tinggi saat prediction (brute force)
* ✅ Paham curse of dimensionality pada KNN
* ✅ Bisa pilih K optimal pakai elbow method / cross-validation

---

### 9. Decision Tree

**5 Ide Project:**
* project → Customer segmentation untuk marketing
* project → Loan approval system
* project → Medical treatment recommendation
* project → Employee attrition prediction
* project → Game player behavior classification

**🎯 Target Pemahaman:**
* ✅ Paham Gini Impurity vs Entropy (kapan pakai mana)
* ✅ Bisa jelaskan greedy splitting strategy & kenapa ga optimal
* ✅ Mengerti overfitting pada tree dalam & solusinya (pruning, max_depth)
* ✅ Tahu kapan Decision Tree cocok (interpretability, non-linear, categorical data)
* ✅ Paham feature importance dari split frequency
* ✅ Bisa visualisasikan & interpretasi tree structure

---

### 10. Naive Bayes – Gaussian

**5 Ide Project:**
* project → Disease diagnosis dari medical test results
* project → Weather prediction
* project → Iris flower classification
* project → Gender classification dari biometric data
* project → Student admission prediction

**🎯 Target Pemahaman:**
* ✅ Paham asumsi conditional independence (kenapa "naive")
* ✅ Bisa jelaskan Bayes' Theorem & posterior probability
* ✅ Mengerti kapan asumsi independence gagal (correlated features)
* ✅ Tahu kenapa pakai Gaussian distribution untuk continuous data
* ✅ Paham zero-frequency problem & Laplace smoothing
* ✅ Bisa bandingkan dengan logistic regression

---

### 11. Naive Bayes – Multinomial

**5 Ide Project:**
* project → Sentiment analysis untuk product reviews
* project → News article categorization
* project → Email spam detection
* project → Language detection
* project → Topic modeling untuk social media posts

**🎯 Target Pemahaman:**
* ✅ Paham kenapa cocok untuk text (word count/frequency)
* ✅ Bisa jelaskan TF (term frequency) vs TF-IDF
* ✅ Mengerti log probability untuk avoid underflow
* ✅ Tahu solusi zero-frequency (add-one smoothing)
* ✅ Paham bag-of-words assumption (order doesn't matter)
* ✅ Bisa bandingkan dengan Gaussian Naive Bayes

---

### 12. Support Vector Machine (Linear)

**5 Ide Project:**
* project → Face detection (face vs non-face)
* project → Cancer classification (benign vs malignant)
* project → Image classification (cat vs dog)
* project → Fraud detection pada transaksi
* project → Handwriting recognition

**🎯 Target Pemahaman:**
* ✅ Paham konsep maximum margin & kenapa penting
* ✅ Bisa jelaskan support vectors (data points yang define boundary)
* ✅ Mengerti hard margin vs soft margin (C parameter)
* ✅ Tahu kapan SVM cocok (clear margin, high-dimensional data)
* ✅ Paham kernel trick untuk non-linear (teaser untuk SVM kernel)
* ✅ Bisa tuning C parameter (large C → hard margin, small C → soft margin)

---

## 📗 02_UNSUPERVISED_LEARNING

### 13. K-Means Clustering

**5 Ide Project:**
* project → Customer segmentation untuk e-commerce
* project → Image compression
* project → Document clustering
* project → Market segmentation
* project → Anomaly detection dalam network traffic

**🎯 Target Pemahaman:**
* ✅ Paham algoritma: assignment → update centroid → repeat
* ✅ Bisa jelaskan kenapa sensitif terhadap inisialisasi (K-Means++)
* ✅ Mengerti elbow method & silhouette score untuk pilih K optimal
* ✅ Tahu limitasi: spherical clusters, sensitive to outliers, hard assignment
* ✅ Paham kenapa scaling wajib (Euclidean distance-based)
* ✅ Bisa deteksi konvergen & local minima problem

---

### 14. K-Medoids (PAM)

**5 Ide Project:**
* project → Robust customer profiling dengan outliers
* project → Gene expression clustering
* project → Sensor network clustering
* project → Image segmentation
* project → City clustering berdasarkan demographics

**🎯 Target Pemahaman:**
* ✅ Paham perbedaan: K-Means (mean) vs K-Medoids (actual data point)
* ✅ Bisa jelaskan kenapa lebih robust terhadap outliers
* ✅ Mengerti trade-off: lebih robust tapi lebih lambat (O(n²) vs O(n))
* ✅ Tahu kapan pakai K-Medoids (data dengan outliers, non-Euclidean metric)
* ✅ Paham swap strategy untuk optimize medoids
* ✅ Bisa bandingkan computational cost vs K-Means

---

### 15. Hierarchical Clustering

**5 Ide Project:**
* project → Phylogenetic tree construction
* project → Social network community detection
* project → Product categorization hierarchy
* project → Document taxonomy creation
* project → Customer hierarchy analysis

**🎯 Target Pemahaman:**
* ✅ Paham agglomerative (bottom-up) vs divisive (top-down)
* ✅ Bisa jelaskan linkage types: single, complete, average, Ward (kapan pakai mana)
* ✅ Mengerti dendrogram interpretation & cutting strategy
* ✅ Tahu kapan pakai Hierarchical (butuh hierarchy structure, small dataset)
* ✅ Paham computational cost O(n³) → tidak scalable
* ✅ Bisa deteksi optimal number of clusters dari dendrogram

---

### 16. DBSCAN

**5 Ide Project:**
* project → Anomaly detection dalam sensor data
* project → Geospatial clustering (restaurant locations)
* project → Network intrusion detection
* project → Noise filtering dalam image processing
* project → Traffic pattern analysis

**🎯 Target Pemahaman:**
* ✅ Paham density-based clustering (vs centroid/hierarchical)
* ✅ Bisa jelaskan epsilon (radius) & minPts (minimum points)
* ✅ Mengerti core points, border points, noise points
* ✅ Tahu kapan DBSCAN cocok (arbitrary shapes, noise handling, no need to specify K)
* ✅ Paham limitasi: varying density, parameter tuning sulit
* ✅ Bisa pilih epsilon optimal (k-distance graph)

---

### 17. Gaussian Mixture Model (GMM)

**5 Ide Project:**
* project → Speaker identification
* project → Image segmentation dengan soft boundaries
* project → Customer behavior modeling
* project → Background subtraction dalam video
* project → Multi-modal data clustering

**🎯 Target Pemahaman:**
* ✅ Paham soft clustering (probabilistic membership) vs hard clustering
* ✅ Bisa jelaskan EM algorithm: Expectation → Maximization
* ✅ Mengerti kenapa cocok untuk overlapping clusters
* ✅ Tahu perbedaan GMM vs K-Means (Gaussian vs spherical assumption)
* ✅ Paham BIC/AIC untuk pilih number of components
* ✅ Bisa interpretasi mixture weights, means, covariances

---

### 18. Principal Component Analysis (PCA)

**5 Ide Project:**
* project → Face recognition dengan eigenfaces
* project → Data compression untuk image storage
* project → Feature reduction untuk big data
* project → Visualization high-dimensional datasets
* project → Noise reduction dalam signal processing

**🎯 Target Pemahaman:**
* ✅ Paham variance maximization & orthogonality constraint
* ✅ Bisa jelaskan eigenvector (direction) & eigenvalue (importance)
* ✅ Mengerti scree plot untuk pilih number of components
* ✅ Tahu kapan PCA cocok (linear correlations, remove redundancy)
* ✅ Paham limitasi: linear only, interpretability loss
* ✅ Bisa rekonstruksi data dari principal components

---

### 19. Singular Value Decomposition (SVD)

**5 Ide Project:**
* project → Recommender system (Netflix-style)
* project → Image compression
* project → Latent semantic analysis
* project → Data imputation untuk missing values
* project → Collaborative filtering

**🎯 Target Pemahaman:**
* ✅ Paham matrix factorization: A = UΣV^T
* ✅ Bisa jelaskan low-rank approximation & information loss
* ✅ Mengerti hubungan SVD dengan PCA (PCA = SVD pada centered data)
* ✅ Tahu kapan SVD cocok (recommender system, missing data, compression)
* ✅ Paham singular values sebagai importance ranking
* ✅ Bisa pilih truncated rank untuk compression vs accuracy trade-off

---

### 20. Linear Discriminant Analysis (LDA)

**5 Ide Project:**
* project → Face recognition dengan class separation
* project → Handwriting recognition optimization
* project → Medical image classification
* project → Speech recognition preprocessing
* project → Biometric authentication system

**🎯 Target Pemahaman:**
* ✅ Paham supervised dimensionality reduction (vs PCA unsupervised)
* ✅ Bisa jelaskan maximize between-class variance, minimize within-class variance
* ✅ Mengerti projection direction yang maximize class separability
* ✅ Tahu kapan LDA > PCA (labeled data, classification task)
* ✅ Paham limitasi: max (n_classes - 1) components, Gaussian assumption
* ✅ Bisa bandingkan dengan PCA untuk classification

---

### 21. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**5 Ide Project:**
* project → Visualization MNIST dataset
* project → Gene expression visualization
* project → Word embedding visualization
* project → Customer segment exploration
* project → High-dimensional data exploration tool

**🎯 Target Pemahaman:**
* ✅ Paham preserve local structure (neighbor relationships)
* ✅ Bisa jelaskan perplexity parameter & efeknya
* ✅ Mengerti kenapa global structure tidak terjaga
* ✅ Tahu kapan pakai t-SNE (visualization only, bukan preprocessing)
* ✅ Paham non-deterministic (different run → different result)
* ✅ Bisa interpretasi cluster distance (jarak antar cluster ga meaningful)

---

### 22. UMAP (Uniform Manifold Approximation and Projection)

**5 Ide Project:**
* project → Single-cell RNA visualization
* project → Scalable image dataset visualization
* project → Document embedding visualization
* project → Protein structure analysis
* project → Large-scale customer journey mapping

**🎯 Target Pemahaman:**
* ✅ Paham graph-based manifold learning
* ✅ Bisa jelaskan n_neighbors & min_dist parameters
* ✅ Mengerti preservasi global + local structure (better than t-SNE)
* ✅ Tahu kapan UMAP > t-SNE (scalability, faster, preserve more structure)
* ✅ Paham bisa untuk preprocessing (not just visualization)
* ✅ Bisa tune hyperparameter untuk balance local vs global

---

## 📕 03_ENSEMBLE_LEARNING

### 23. Random Forest

**5 Ide Project:**
* project → Credit scoring system
* project → Disease prediction dari symptoms
* project → Stock market prediction
* project → Customer lifetime value prediction
* project → Fraud detection system

**🎯 Target Pemahaman:**
* ✅ Paham bagging (bootstrap aggregating) untuk reduce variance
* ✅ Bisa jelaskan feature randomness & kenapa penting (decorrelate trees)
* ✅ Mengerti out-of-bag (OOB) error sebagai validation
* ✅ Tahu kapan Random Forest cocok (reduce overfitting, robust, feature importance)
* ✅ Paham trade-off: accuracy vs interpretability
* ✅ Bisa tune n_estimators, max_features, max_depth

---

### 24. AdaBoost (Adaptive Boosting)

**5 Ide Project:**
* project → Face detection system (Viola-Jones)
* project → Weak signal classification
* project → Imbalanced dataset classification
* project → Pedestrian detection
* project → Rare disease prediction

**🎯 Target Pemahaman:**
* ✅ Paham sequential learning (focus on misclassified samples)
* ✅ Bisa jelaskan sample weighting & weak learner combination
* ✅ Mengerti kenapa fokus pada error → bias reduction
* ✅ Tahu kapan AdaBoost cocok (weak learners, binary classification)
* ✅ Paham sensitivity terhadap noise & outliers
* ✅ Bisa tune learning rate & n_estimators

---

### 25. Gradient Boosting

**5 Ide Project:**
* project → House price prediction (Kaggle-style)
* project → Click-through rate optimization
* project → Sales forecasting
* project → Customer churn prediction
* project → Insurance claim prediction

**🎯 Target Pemahaman:**
* ✅ Paham fit residual errors sequentially
* ✅ Bisa jelaskan gradient descent in function space
* ✅ Mengerti learning rate untuk prevent overfitting
* ✅ Tahu kapan Gradient Boosting > Random Forest (tabular data, want best accuracy)
* ✅ Paham regularization (subsample, max_depth, min_samples_split)
* ✅ Bisa bandingkan XGBoost, LightGBM, CatBoost

---

## 🧠 04_NEURAL_NETWORK

### 26. Perceptron

**5 Ide Project:**
* project → Binary logic gate simulator (AND, OR)
* project → Simple pattern recognition
* project → Linear classifier untuk 2D data
* project → Early spam filter
* project → Binary sentiment classifier

**🎯 Target Pemahaman:**
* ✅ Paham single neuron architecture (weights, bias, activation)
* ✅ Bisa jelaskan linear separability constraint
* ✅ Mengerti kenapa ga bisa solve XOR problem (non-linearly separable)
* ✅ Tahu update rule: w = w + α(y - ŷ)x
* ✅ Paham limitasi: binary classification, linear only
* ✅ Bisa visualisasikan decision boundary

---

### 27. Feedforward Neural Network (Multilayer Perceptron)

**5 Ide Project:**
* project → Handwritten digit recognition (MNIST)
* project → Wine quality prediction
* project → Fashion item classification
* project → Student grade prediction
* project → XOR problem solver

**🎯 Target Pemahaman:**
* ✅ Paham hidden layer role (learn non-linear representations)
* ✅ Bisa jelaskan universal approximation theorem
* ✅ Mengerti depth vs width trade-off
* ✅ Tahu aktivasi function role (inject non-linearity)
* ✅ Paham forward propagation flow
* ✅ Bisa tune architecture (layers, neurons, activation)

---

### 28. Backpropagation

**5 Ide Project:**
* project → Training visualizer untuk neural networks
* project → Gradient flow debugger
* project → Custom loss function optimizer
* project → Learning rate scheduler
* project → Weight update tracker

**🎯 Target Pemahaman:**
* ✅ Paham chain rule untuk compute gradients
* ✅ Bisa trace gradient propagation backward
* ✅ Mengerti vanishing gradient problem (deep networks, sigmoid/tanh)
* ✅ Tahu exploding gradient problem & gradient clipping
* ✅ Paham computational graph & autodiff
* ✅ Bisa implement backprop from scratch

---

### 29. Activation Functions

**5 Ide Project:**
* project → Activation function comparison tool
* project → Non-linearity simulator
* project → Gradient vanishing detector
* project → Custom activation function tester
* project → Performance benchmarking dashboard

**🎯 Target Pemahaman:**
* ✅ Paham Sigmoid (vanishing gradient), Tanh (zero-centered), ReLU (dead neurons)
* ✅ Bisa jelaskan Leaky ReLU, ELU, GELU variants
* ✅ Mengerti kapan pakai mana (output layer vs hidden layer)
* ✅ Tahu dying ReLU problem & solusi (Leaky ReLU, He initialization)
* ✅ Paham gradient saturation & derivatives
* ✅ Bisa visualisasikan activation & gradient curves

---

## 🧾 05_ASSOCIATION_RULE_LEARNING

### 30. Apriori Algorithm

**5 Ide Project:**
* project → Market basket analysis untuk supermarket
* project → Product bundling recommendation
* project → Cross-selling strategy optimizer
* project → Web clickstream analysis
* project → Medical symptom co-occurrence finder

**🎯 Target Pemahaman:**
* ✅ Paham support, confidence, lift metrics & interpretasinya
* ✅ Bisa jelaskan candidate generation & pruning strategy
* ✅ Mengerti minimum support & confidence threshold
* ✅ Tahu kapan Apriori cocok (frequent pattern mining, association rules)
* ✅ Paham computational complexity & scalability issue
* ✅ Bisa interpretasi rules untuk business insight

---

### 31. ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal)

**5 Ide Project:**
* project → Fast transaction pattern mining
* project → Efficient market basket analysis
* project → Large-scale recommendation system
* project → Sequential pattern mining
* project → E-commerce behavior analysis

**🎯 Target Pemahaman:**
* ✅ Paham vertical data format (item → transaction list)
* ✅ Bisa jelaskan intersection-based support counting
* ✅ Mengerti kenapa lebih efisien dari Apriori (no candidate generation)
* ✅ Tahu trade-off: memory usage vs computation speed
* ✅ Paham depth-first search strategy
* ✅ Bisa bandingkan efficiency dengan Apriori

---

## 🎁 BONUS PEMAHAMAN UMUM

**🧠 Meta-Learning Skills (Paling Penting!):**
* ✅ **Bias-Variance Trade-off** → Paham untuk semua model
* ✅ **Overfitting vs Underfitting** → Bisa deteksi & solusi
* ✅ **Cross-Validation** → K-Fold, Stratified, Time-Series split
* ✅ **Feature Engineering** → Scaling, encoding, creation
* ✅ **Hyperparameter Tuning** → Grid search, Random search, Bayesian optimization
* ✅ **Model Evaluation** → Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
* ✅ **Data Leakage** → Deteksi & prevent
* ✅ **Class Imbalance** → SMOTE, class weights, resampling

---

**💡 Cara Validasi Pemahaman:**
1. ✅ Bisa jelaskan ke orang awam (Feynman Technique)
2. ✅ Bisa implement from scratch (tanpa library)
3. ✅ Bisa debug kenapa model ga perform
4. ✅ Bisa pilih algoritma yang tepat untuk problem baru
5. ✅ Bisa tune hyperparameter dengan reasoning jelas

---

**Total: 31 algoritma × 5 project × 6 pemahaman = MASSIVE LEARNING PATH! 🚀**