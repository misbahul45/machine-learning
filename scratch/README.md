# 45 Project Basic Data Science From Scratch

## Day 1

## 1. Restaurant Rating Prediction

**Goal:** Prediksi rating restoran dari harga, pelayanan, kebersihan, promo, parkir, halal, dan jarak.

**Input fitur:**

```text
price
service_score
cleanliness_score
food_quality_score
parking_available
halal_certified
promo_available
distance
```

**Target:**

```text
rating
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Variance Threshold
Polynomial Features
IQR Outlier Detection
K-Fold Cross Validation
Bias-Variance Analysis
Gaussian Log-Likelihood
p-value
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

## 2. House Price Prediction

**Goal:** Prediksi harga rumah berdasarkan luas, kamar, lokasi, umur bangunan, dan fasilitas.

**Input fitur:**

```text
area
bedrooms
bathrooms
location_score
building_age
has_garage
has_garden
distance_to_city
```

**Target:**

```text
house_price
```

**Penerapan materi:**

```text
Covariance Matrix
Pearson Correlation
Variance Threshold
Polynomial area²
IQR Outlier Detection
K-Fold Cross Validation
Learning Curve
Gaussian Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

# Day 2

## 3. E-Commerce Purchase Prediction

**Goal:** Prediksi apakah user membeli produk atau tidak.

**Input fitur:**

```text
price
discount
rating
review_count
is_free_shipping
is_official_store
product_rank
clicked
```

**Target:**

```text
purchased
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial price × discount
Variance Threshold
Stratified K-Fold
Bias-Variance Tradeoff
Bernoulli Log-Likelihood
p-value
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

## 4. Credit Default Prediction

**Goal:** Prediksi apakah nasabah gagal bayar atau tidak.

**Input fitur:**

```text
income
loan_amount
loan_duration
age
employment_years
debt_ratio
has_previous_default
credit_score
```

**Target:**

```text
default
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
High-Bias vs High-Variance
Bernoulli Log-Likelihood
p-value
Odds Ratio
Confidence Interval
Feature Importance
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Gaussian
```

---

# Day 3

## 5. Student Final Grade Prediction

**Goal:** Prediksi nilai akhir siswa.

**Input fitur:**

```text
study_hours
attendance_rate
previous_score
assignment_score
parent_education_level
internet_access
motivation_score
```

**Target:**

```text
final_grade
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial study_hours²
IQR Outlier Detection
K-Fold Cross Validation
Learning Curve
Gaussian Log-Likelihood
p-value
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

## 6. Student Dropout Risk Prediction

**Goal:** Prediksi apakah siswa berisiko dropout.

**Input fitur:**

```text
attendance_rate
study_hours
assignment_missing
financial_issue
distance_to_school
previous_grade
motivation_score
```

**Target:**

```text
dropout
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial study_hours²
IQR Outlier Detection
Stratified K-Fold
Learning Curve
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

# Day 4

## 7. Medical Disease Prediction

**Goal:** Prediksi apakah pasien memiliki penyakit tertentu atau tidak.

**Input fitur:**

```text
age
bmi
blood_pressure
cholesterol
glucose
smoking
family_history
physical_activity_score
```

**Target:**

```text
disease
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
PCA Manual
Stratified K-Fold
Bias-Variance Comparison
Bernoulli Log-Likelihood
Likelihood Ratio Test
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. PCA + Logistic Regression
```

---

## 8. Loan Approval Prediction

**Goal:** Prediksi apakah pengajuan pinjaman disetujui.

**Input fitur:**

```text
income
loan_amount
credit_score
age
employment_years
debt_ratio
has_collateral
previous_default
```

**Target:**

```text
approved
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bernoulli Log-Likelihood
p-value
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. SVM Linear
```

---

# Day 5

## 9. Customer Churn Prediction

**Goal:** Prediksi apakah pelanggan akan berhenti memakai layanan.

**Input fitur:**

```text
tenure
monthly_spend
complaint_count
support_ticket_count
last_login_days
is_premium
discount_used
usage_frequency
```

**Target:**

```text
churn
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bias-Variance Analysis
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Gaussian
```

---

## 10. Marketing Campaign Response Prediction

**Goal:** Prediksi apakah user akan merespons campaign.

**Input fitur:**

```text
age
income
previous_purchases
email_open_rate
click_rate
discount_amount
is_member
last_purchase_days
```

**Target:**

```text
responded
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance
Polynomial discount_amount²
CV-based Feature Selection
Stratified K-Fold
Bias-Variance Tradeoff
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

# Day 6

## 11. Restaurant Sales Prediction

**Goal:** Prediksi penjualan harian restoran.

**Input fitur:**

```text
day_of_week
is_weekend
weather_score
promo
customer_count
avg_order_value
holiday
online_order_count
```

**Target:**

```text
daily_sales
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial customer_count²
IQR Outlier Detection
K-Fold Cross Validation
Learning Curve
Gaussian Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Decision Tree Regressor
```

---

## 12. Delivery Time Prediction

**Goal:** Prediksi waktu pengiriman pesanan.

**Input fitur:**

```text
distance
traffic_score
weather_score
driver_experience
order_size
is_peak_hour
restaurant_delay
vehicle_type_score
```

**Target:**

```text
delivery_time
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial distance²
IQR Outlier Detection
Variance Threshold
K-Fold Cross Validation
Bias-Variance Analysis
Gaussian Log-Likelihood
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

# Day 7

## 13. Fraud Transaction Detection

**Goal:** Prediksi apakah transaksi merupakan fraud.

**Input fitur:**

```text
transaction_amount
transaction_hour
location_risk_score
device_risk_score
failed_attempts
is_new_device
account_age_days
transaction_frequency
```

**Target:**

```text
fraud
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Precision-Recall Analysis
Bernoulli Log-Likelihood
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

## 14. Employee Attrition Prediction

**Goal:** Prediksi apakah karyawan akan resign.

**Input fitur:**

```text
age
salary
years_at_company
promotion_gap
work_life_balance
job_satisfaction
overtime
training_hours
```

**Target:**

```text
attrition
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial salary²
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Gaussian
```

---

# Day 8

## 15. Employee Performance Prediction

**Goal:** Prediksi skor performa karyawan.

**Input fitur:**

```text
experience_years
training_hours
task_completed
manager_rating
peer_rating
remote_work_days
overtime_hours
team_score
```

**Target:**

```text
performance_score
```

**Penerapan materi:**

```text
Kendall Tau
Covariance Matrix
Variance Threshold
Polynomial training_hours²
IQR Outlier Detection
K-Fold Cross Validation
Learning Curve
Gaussian Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

## 16. Public Service Satisfaction Prediction

**Goal:** Prediksi rating kepuasan layanan publik.

**Input fitur:**

```text
waiting_time
staff_friendliness
facility_score
queue_system_score
complaint_submitted
service_type_score
accessibility_score
cost_score
```

**Target:**

```text
satisfaction_rating
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial waiting_time²
IQR Outlier Detection
Variance Threshold
K-Fold Cross Validation
Gaussian Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Decision Tree Regressor
```

---

# Day 9

## 17. Fitness Progress Prediction

**Goal:** Prediksi perubahan berat badan berdasarkan pola hidup.

**Input fitur:**

```text
calorie_intake
workout_minutes
sleep_hours
water_intake
stress_level
protein_intake
steps_per_day
consistency_score
```

**Target:**

```text
weight_change
```

**Penerapan materi:**

```text
Kendall Tau
Covariance
Polynomial workout_minutes²
IQR Outlier Detection
Variance Threshold
K-Fold Cross Validation
Learning Curve
Gaussian Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

## 18. Sleep Quality Classification

**Goal:** Prediksi apakah kualitas tidur baik atau buruk.

**Input fitur:**

```text
sleep_duration
screen_time
caffeine_intake
stress_level
exercise_minutes
room_noise
late_meal
wake_up_count
```

**Target:**

```text
good_sleep
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial screen_time²
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. SVM Linear
```

---

# Day 10

## 19. Stock Return Direction Prediction

**Goal:** Prediksi apakah return saham besok naik atau turun.

**Input fitur:**

```text
return_1d
return_3d
return_7d
volume_change
volatility
moving_average_gap
rsi_score
market_sentiment_score
```

**Target:**

```text
target_up
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance Matrix
IQR Outlier Detection
Variance Threshold
Rolling Validation
Bias-Variance Analysis
Bernoulli Log-Likelihood
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Gaussian
```

---

## 20. Poverty Risk Prediction

**Goal:** Prediksi apakah rumah tangga memiliki risiko miskin.

**Input fitur:**

```text
income
family_size
education_level
employment_status_score
housing_quality
access_to_water
electricity_access
dependent_count
```

**Target:**

```text
poverty_risk
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

# Day 11

## 21. Spam Email Classification

**Goal:** Klasifikasi email spam atau bukan spam.

**Input fitur:**

```text
word_count
capital_word_count
link_count
money_keyword_count
urgent_keyword_count
exclamation_count
sender_reputation_score
has_attachment
```

**Target:**

```text
spam
```

**Penerapan materi:**

```text
Bag of Words Manual
Point-Biserial Correlation
Covariance
Variance Threshold
TF Manual
Stratified K-Fold
Bernoulli Log-Likelihood
Precision-Recall Analysis
```

**Algoritma manual:**

```text
1. Naive Bayes Bernoulli
2. Logistic Regression
```

---

## 22. Sentiment Analysis Basic

**Goal:** Klasifikasi sentimen teks menjadi positif atau negatif.

**Input fitur:**

```text
positive_word_count
negative_word_count
text_length
exclamation_count
question_count
emoji_count
uppercase_ratio
sentiment_lexicon_score
```

**Target:**

```text
sentiment
```

**Penerapan materi:**

```text
Bag of Words Manual
Point-Biserial Correlation
Covariance
Variance Threshold
TF-IDF Manual
Stratified K-Fold
Bernoulli Log-Likelihood
Odds Ratio
```

**Algoritma manual:**

```text
1. Naive Bayes Bernoulli
2. Logistic Regression
```

---

# Day 12

## 23. News Category Classification

**Goal:** Klasifikasi berita ke kategori tertentu.

**Input fitur:**

```text
politics_word_count
sports_word_count
business_word_count
technology_word_count
health_word_count
title_length
body_word_count
keyword_density
```

**Target:**

```text
news_category
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Covariance
Variance Threshold
K-Fold Cross Validation
Multiclass Log-Likelihood
Feature Importance
```

**Algoritma manual:**

```text
1. Multinomial Naive Bayes
2. Softmax Regression
```

---

## 24. Movie Review Sentiment Prediction

**Goal:** Prediksi sentimen review film.

**Input fitur:**

```text
positive_word_count
negative_word_count
intensifier_count
negation_count
review_length
rating_mentioned
exclamation_count
sentiment_score
```

**Target:**

```text
positive_review
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Point-Biserial Correlation
Polynomial negative_word_count²
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. Logistic Regression
2. SVM Linear
```

---

# Day 13

## 25. Fake News Detection Basic

**Goal:** Prediksi apakah berita fake atau real.

**Input fitur:**

```text
title_length
body_word_count
clickbait_word_count
source_reputation_score
sentiment_extreme_score
uppercase_ratio
link_count
claim_count
```

**Target:**

```text
fake_news
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Point-Biserial Correlation
Covariance
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Bernoulli
```

---

## 26. Toxic Comment Classification

**Goal:** Prediksi apakah komentar toxic atau tidak.

**Input fitur:**

```text
toxic_word_count
insult_word_count
uppercase_ratio
exclamation_count
comment_length
negative_sentiment_score
personal_attack_keyword
repeated_character_count
```

**Target:**

```text
toxic
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Point-Biserial Correlation
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Precision-Recall Analysis
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Bernoulli
```

---

# Day 14

## 27. Question Intent Classification

**Goal:** Klasifikasi intent pertanyaan user.

**Input fitur:**

```text
what_keyword
how_keyword
why_keyword
where_keyword
question_length
verb_count
noun_count
keyword_density
```

**Target:**

```text
intent_class
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Covariance
Variance Threshold
K-Fold Cross Validation
Multiclass Log-Likelihood
Confidence Score Analysis
```

**Algoritma manual:**

```text
1. Multinomial Naive Bayes
2. Softmax Regression
```

---

## 28. Simple Search Engine Ranking

**Goal:** Ranking dokumen paling relevan terhadap query.

**Input fitur:**

```text
term_frequency
inverse_document_frequency
query_term_overlap
document_length
title_match_score
keyword_position_score
cosine_similarity
popularity_score
```

**Target:**

```text
relevance_score
```

**Penerapan materi:**

```text
TF-IDF Manual
Cosine Similarity
Kendall Tau Ranking Evaluation
Covariance
Variance Threshold
Polynomial query_overlap²
K-Fold Cross Validation
```

**Algoritma manual:**

```text
1. TF-IDF + Cosine Similarity
2. Linear Regression Ranking Score
```

---

# Day 15

## 29. Product Review Rating Prediction

**Goal:** Prediksi rating produk dari teks review sederhana.

**Input fitur:**

```text
positive_word_count
negative_word_count
review_length
exclamation_count
delivery_word_count
quality_word_count
price_word_count
sentiment_score
```

**Target:**

```text
product_rating
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Kendall Tau
Covariance
Polynomial sentiment_score²
IQR Outlier Detection
K-Fold Cross Validation
Gaussian Log-Likelihood
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

## 30. Resume Job Match Scoring

**Goal:** Prediksi skor kecocokan CV dengan lowongan kerja.

**Input fitur:**

```text
skill_overlap_count
education_match
experience_years
keyword_similarity
project_relevance_score
certification_count
job_title_similarity
text_cosine_similarity
```

**Target:**

```text
match_score
```

**Penerapan materi:**

```text
TF-IDF Manual
Cosine Similarity
Kendall Tau
Point-Biserial Correlation
Covariance
Polynomial skill_overlap²
K-Fold Cross Validation
Gaussian Log-Likelihood
```

**Algoritma manual:**

```text
1. TF-IDF + Cosine Similarity
2. Linear Regression
```

---

# Day 16

## 31. Digit Classification Basic

**Goal:** Klasifikasi digit tulisan tangan dari pixel sederhana.

**Input fitur:**

```text
pixel_1
pixel_2
pixel_3
...
pixel_n
mean_intensity
center_of_mass_x
center_of_mass_y
```

**Target:**

```text
digit_label
```

**Penerapan materi:**

```text
Pixel Flattening
Variance Threshold
PCA Manual
Euclidean Distance
K-Fold Cross Validation
Confusion Matrix
Multiclass Accuracy
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```

---

## 32. Cat vs Dog Image Classification Basic

**Goal:** Klasifikasi gambar kucing atau anjing menggunakan fitur dasar.

**Input fitur:**

```text
mean_red
mean_green
mean_blue
brightness
edge_density
texture_score
aspect_ratio
color_variance
```

**Target:**

```text
cat_or_dog
```

**Penerapan materi:**

```text
Color Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
Stratified K-Fold
Bernoulli Log-Likelihood
Precision-Recall Analysis
```

**Algoritma manual:**

```text
1. Logistic Regression
2. K-Nearest Neighbors
```

---

# Day 17

## 33. Fruit Image Classification

**Goal:** Klasifikasi jenis buah berdasarkan warna dan bentuk sederhana.

**Input fitur:**

```text
mean_red
mean_green
mean_blue
dominant_color_score
roundness
area
perimeter
texture_score
```

**Target:**

```text
fruit_class
```

**Penerapan materi:**

```text
Color Histogram Manual
Shape Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Accuracy
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Naive Bayes Gaussian
```

---

## 34. Face vs Non-Face Classification Basic

**Goal:** Prediksi apakah gambar mengandung wajah atau bukan.

**Input fitur:**

```text
mean_intensity
edge_density
symmetry_score
skin_color_ratio
eye_region_darkness
texture_variance
aspect_ratio
center_brightness
```

**Target:**

```text
face_present
```

**Penerapan materi:**

```text
Image Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
Stratified K-Fold
Bernoulli Log-Likelihood
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. Logistic Regression
2. SVM Linear
```

---

# Day 18

## 35. Handwritten Letter Classification

**Goal:** Klasifikasi huruf tulisan tangan dari fitur pixel sederhana.

**Input fitur:**

```text
pixel_1
pixel_2
pixel_3
...
pixel_n
stroke_density
center_of_mass_x
center_of_mass_y
symmetry_score
```

**Target:**

```text
letter_class
```

**Penerapan materi:**

```text
Pixel Flattening
Variance Threshold
PCA Manual
Covariance Matrix
K-Fold Cross Validation
Confusion Matrix
Multiclass Log-Likelihood
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```

---

## 36. Image Brightness Quality Classification

**Goal:** Klasifikasi kualitas gambar: terlalu gelap, normal, atau terlalu terang.

**Input fitur:**

```text
mean_brightness
brightness_variance
dark_pixel_ratio
bright_pixel_ratio
contrast_score
mean_red
mean_green
mean_blue
```

**Target:**

```text
brightness_quality_class
```

**Penerapan materi:**

```text
Histogram Feature Extraction
Covariance
Variance Threshold
IQR Outlier Detection
K-Fold Cross Validation
Multiclass Evaluation
```

**Algoritma manual:**

```text
1. Decision Tree Classifier
2. Naive Bayes Gaussian
```

---

# Day 19

## 37. Plant Leaf Disease Classification Basic

**Goal:** Klasifikasi daun sehat atau sakit dari warna dan tekstur.

**Input fitur:**

```text
green_ratio
yellow_ratio
brown_spot_ratio
texture_variance
edge_density
leaf_area
color_variance
symmetry_score
```

**Target:**

```text
leaf_disease
```

**Penerapan materi:**

```text
Color Histogram Manual
Texture Feature Extraction
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
PCA Manual
Stratified K-Fold
```

**Algoritma manual:**

```text
1. Logistic Regression
2. K-Nearest Neighbors
```

---

## 38. Traffic Sign Classification Basic

**Goal:** Klasifikasi rambu lalu lintas dari bentuk dan warna dasar.

**Input fitur:**

```text
red_ratio
blue_ratio
yellow_ratio
edge_density
shape_roundness
corner_count
brightness
contrast_score
```

**Target:**

```text
traffic_sign_class
```

**Penerapan materi:**

```text
Color Feature Extraction
Shape Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Accuracy
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```

---

# Day 20

## 39. Image Similarity Search Basic

**Goal:** Mencari gambar yang paling mirip dengan query image.

**Input fitur:**

```text
color_histogram_red
color_histogram_green
color_histogram_blue
brightness_histogram
texture_score
edge_density
dominant_color_score
aspect_ratio
```

**Target:**

```text
similarity_rank
```

**Penerapan materi:**

```text
Color Histogram Manual
Cosine Similarity
Euclidean Distance
Kendall Tau Ranking Evaluation
Covariance Matrix
Variance Threshold
PCA Manual
```

**Algoritma manual:**

```text
1. Cosine Similarity Ranking
2. K-Nearest Neighbors Search
```

---

## 40. Medical X-Ray Normal vs Abnormal Basic

**Goal:** Klasifikasi X-Ray normal atau abnormal menggunakan fitur gambar sederhana.

**Input fitur:**

```text
mean_intensity
contrast_score
edge_density
dark_region_ratio
bright_region_ratio
texture_variance
symmetry_score
region_density_score
```

**Target:**

```text
abnormal
```

**Penerapan materi:**

```text
Image Histogram Features
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
PCA Manual
Stratified K-Fold
Precision-Recall Analysis
```

**Algoritma manual:**

```text
1. Logistic Regression
2. SVM Linear
```

---

# Day 21

## 41. Speech Gender Classification Basic

**Goal:** Klasifikasi suara laki-laki atau perempuan dari fitur audio dasar.

**Input fitur:**

```text
zero_crossing_rate
mean_amplitude
energy
pitch_estimate
spectral_centroid
spectral_bandwidth
duration
silence_ratio
```

**Target:**

```text
gender_class
```

**Penerapan materi:**

```text
Audio Feature Extraction
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
PCA Manual
Stratified K-Fold
Bernoulli Log-Likelihood
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Gaussian
```

---

## 42. Music Genre Classification Basic

**Goal:** Klasifikasi genre musik sederhana dari fitur audio.

**Input fitur:**

```text
tempo_estimate
energy
zero_crossing_rate
spectral_centroid
spectral_bandwidth
beat_strength
mean_amplitude
duration
```

**Target:**

```text
music_genre
```

**Penerapan materi:**

```text
Audio Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Accuracy
Confusion Matrix
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Naive Bayes Gaussian
```

---

# Day 22

## 43. Audio Emotion Classification Basic

**Goal:** Klasifikasi emosi audio sederhana seperti happy, sad, angry, neutral.

**Input fitur:**

```text
mean_amplitude
energy
pitch_estimate
zero_crossing_rate
spectral_centroid
speech_rate
silence_ratio
duration
```

**Target:**

```text
emotion_class
```

**Penerapan materi:**

```text
Audio Feature Extraction
Kendall Tau
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Log-Likelihood
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```

---

## 44. Clap vs Non-Clap Sound Classification

**Goal:** Klasifikasi apakah audio berisi suara tepuk tangan atau bukan.

**Input fitur:**

```text
peak_amplitude
short_time_energy
zero_crossing_rate
duration
silence_ratio
spectral_centroid
attack_time
energy_variance
```

**Target:**

```text
clap_sound
```

**Penerapan materi:**

```text
Audio Energy Feature
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bernoulli Log-Likelihood
```

**Algoritma manual:**

```text
1. Energy Threshold Classifier
2. Logistic Regression
```

---

# Day 23

## 45. Environmental Sound Classification Basic

**Goal:** Klasifikasi suara lingkungan seperti hujan, kendaraan, manusia, hewan, dan mesin.

**Input fitur:**

```text
zero_crossing_rate
short_time_energy
spectral_centroid
spectral_bandwidth
mean_amplitude
energy_variance
duration
silence_ratio
```

**Target:**

```text
sound_class
```

**Penerapan materi:**

```text
Audio Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Accuracy
Confusion Matrix
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```
# Day 24

## 46. Voice Command Classification Basic

**Goal:** Klasifikasi perintah suara sederhana seperti start, stop, yes, no, left, right.

**Input fitur:**

```text
zero_crossing_rate
short_time_energy
spectral_centroid
spectral_bandwidth
mean_amplitude
duration
silence_ratio
peak_amplitude
```

**Target:**

```text
voice_command_class
```

**Penerapan materi:**

```text
Audio Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Log-Likelihood
Confusion Matrix
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```

---

## 47. Speaker Identification Basic

**Goal:** Memprediksi speaker berdasarkan karakteristik suara dasar.

**Input fitur:**

```text
pitch_estimate
mean_amplitude
energy
zero_crossing_rate
spectral_centroid
spectral_bandwidth
speech_rate
silence_ratio
```

**Target:**

```text
speaker_id
```

**Penerapan materi:**

```text
Audio Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
K-Fold Cross Validation
Multiclass Accuracy
Confusion Matrix
Feature Importance
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Naive Bayes Gaussian
```

---

# Day 25

## 48. Audio Noise Detection Basic

**Goal:** Klasifikasi audio bersih atau noise.

**Input fitur:**

```text
mean_amplitude
energy
energy_variance
zero_crossing_rate
spectral_centroid
spectral_bandwidth
silence_ratio
signal_to_noise_score
```

**Target:**

```text
noisy_audio
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bernoulli Log-Likelihood
Precision-Recall Analysis
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. SVM Linear
```

---

## 49. Audio Event Detection Basic

**Goal:** Deteksi apakah audio mengandung event tertentu seperti alarm, bell, siren, atau knock.

**Input fitur:**

```text
peak_amplitude
short_time_energy
zero_crossing_rate
spectral_centroid
attack_time
duration
energy_variance
silence_ratio
```

**Target:**

```text
event_detected
```

**Penerapan materi:**

```text
Audio Energy Feature
Point-Biserial Correlation
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bernoulli Log-Likelihood
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. Energy Threshold Classifier
2. Logistic Regression
```

---

# Day 26

## 50. Multimodal Product Quality Prediction

**Goal:** Prediksi kualitas produk dari kombinasi data tabular, teks review, dan fitur gambar sederhana.

**Input fitur:**

```text
price
discount
seller_rating
review_positive_word_count
review_negative_word_count
image_brightness
image_sharpness_score
color_variance
```

**Target:**

```text
product_quality_score
```

**Penerapan materi:**

```text
TF-IDF Manual
Image Feature Extraction
Covariance Matrix
Point-Biserial Correlation
Polynomial price × discount
Variance Threshold
K-Fold Cross Validation
Gaussian Log-Likelihood
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. Linear Regression
2. Ridge Regression
```

---

## 51. Multimodal Purchase Prediction

**Goal:** Prediksi apakah user membeli produk berdasarkan fitur produk, teks review, dan gambar produk.

**Input fitur:**

```text
price
discount
rating
review_sentiment_score
review_length
image_brightness
image_quality_score
shipping_available
```

**Target:**

```text
purchased
```

**Penerapan materi:**

```text
TF-IDF Manual
Image Feature Extraction
Point-Biserial Correlation
Covariance Matrix
Polynomial price × discount
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Odds Ratio
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

# Day 27

## 52. Multimodal Restaurant Recommendation

**Goal:** Prediksi apakah restoran layak direkomendasikan berdasarkan rating, teks review, dan gambar makanan.

**Input fitur:**

```text
price
service_score
cleanliness_score
review_sentiment_score
positive_word_count
negative_word_count
food_image_brightness
food_color_score
```

**Target:**

```text
recommended
```

**Penerapan materi:**

```text
Kendall Tau
Point-Biserial Correlation
TF-IDF Manual
Image Color Feature
Covariance Matrix
Variance Threshold
Stratified K-Fold
Bernoulli Log-Likelihood
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Naive Bayes Gaussian
```

---

## 53. Multimodal Food Rating Prediction

**Goal:** Prediksi rating makanan dari harga, review teks, dan fitur visual makanan.

**Input fitur:**

```text
price
portion_size
review_positive_word_count
review_negative_word_count
review_length
food_color_variance
brightness
texture_score
```

**Target:**

```text
food_rating
```

**Penerapan materi:**

```text
TF-IDF Manual
Image Feature Extraction
Kendall Tau
Covariance Matrix
Polynomial price²
IQR Outlier Detection
K-Fold Cross Validation
Gaussian Log-Likelihood
```

**Algoritma manual:**

```text
1. Linear Regression
2. Decision Tree Regressor
```

---

# Day 28

## 54. Simple OCR Digit Recognition Basic

**Goal:** Mengenali digit dari gambar angka sederhana berbasis pixel.

**Input fitur:**

```text
pixel_1
pixel_2
pixel_3
...
pixel_n
stroke_density
center_of_mass_x
center_of_mass_y
symmetry_score
```

**Target:**

```text
digit_class
```

**Penerapan materi:**

```text
Pixel Flattening
Variance Threshold
PCA Manual
Euclidean Distance
K-Fold Cross Validation
Multiclass Accuracy
Confusion Matrix
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Softmax Regression
```

---

## 55. Simple OCR Letter Recognition Basic

**Goal:** Mengenali huruf kapital sederhana dari gambar berbasis pixel.

**Input fitur:**

```text
pixel_1
pixel_2
pixel_3
...
pixel_n
vertical_stroke_score
horizontal_stroke_score
curve_score
symmetry_score
```

**Target:**

```text
letter_class
```

**Penerapan materi:**

```text
Pixel Flattening
Variance Threshold
PCA Manual
Covariance Matrix
K-Fold Cross Validation
Multiclass Log-Likelihood
Confusion Matrix
Error Analysis
```

**Algoritma manual:**

```text
1. K-Nearest Neighbors
2. Naive Bayes Gaussian
```

---

# Day 29

## 56. Basic Text Similarity Plagiarism Detection

**Goal:** Deteksi apakah dua teks mirip atau berpotensi plagiarisme.

**Input fitur:**

```text
tfidf_cosine_similarity
jaccard_similarity
common_word_ratio
sentence_length_difference
keyword_overlap
ngram_overlap_score
document_length_ratio
rare_word_overlap
```

**Target:**

```text
similar_text
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Cosine Similarity
Jaccard Similarity
Point-Biserial Correlation
Covariance Matrix
Stratified K-Fold
Bernoulli Log-Likelihood
```

**Algoritma manual:**

```text
1. TF-IDF + Cosine Similarity Classifier
2. Logistic Regression
```

---

## 57. Basic Document Clustering

**Goal:** Mengelompokkan dokumen berdasarkan kemiripan topik tanpa label.

**Input fitur:**

```text
tfidf_feature_1
tfidf_feature_2
tfidf_feature_3
...
tfidf_feature_n
document_length
keyword_density
title_match_score
```

**Target:**

```text
cluster_id
```

**Penerapan materi:**

```text
Bag of Words Manual
TF-IDF Manual
Cosine Similarity
Covariance Matrix
Variance Threshold
PCA Manual
Elbow Method
Cluster Evaluation Manual
```

**Algoritma manual:**

```text
1. K-Means Clustering
2. Hierarchical Clustering Basic
```

---

# Day 30

## 58. End-to-End Basic Multimodal Classifier

**Goal:** Klasifikasi data sederhana dari gabungan tabular, teks, gambar, dan audio.

**Input fitur:**

```text
numeric_score
binary_feature
text_sentiment_score
text_tfidf_similarity
image_brightness
image_edge_density
audio_energy
audio_zero_crossing_rate
```

**Target:**

```text
class_label
```

**Penerapan materi:**

```text
Tabular Feature Engineering
TF-IDF Manual
Image Feature Extraction
Audio Feature Extraction
Covariance Matrix
Variance Threshold
PCA Manual
Stratified K-Fold
Multiclass Log-Likelihood
Bias-Variance Analysis
```

**Algoritma manual:**

```text
1. Softmax Regression
2. K-Nearest Neighbors
```

---

## 59. End-to-End Basic Risk Scoring Engine

**Goal:** Membuat sistem scoring risiko sederhana dari data tabular, teks laporan, dan sinyal perilaku.

**Input fitur:**

```text
income
debt_ratio
transaction_frequency
report_negative_word_count
report_risk_keyword_count
behavior_score
anomaly_score
history_default
```

**Target:**

```text
risk_level
```

**Penerapan materi:**

```text
Point-Biserial Correlation
Kendall Tau
TF-IDF Manual
Covariance Matrix
Variance Threshold
IQR Outlier Detection
Stratified K-Fold
Bernoulli Log-Likelihood
Odds Ratio
Confidence Interval
```

**Algoritma manual:**

```text
1. Logistic Regression
2. Decision Tree Classifier
```

---

## 60. Final Mini AutoML From Scratch Benchmark

**Goal:** Membandingkan beberapa algoritma manual pada dataset basic untuk memilih model terbaik.

**Input fitur:**

```text
feature_1
feature_2
feature_3
feature_4
feature_5
feature_6
feature_7
feature_8
```

**Target:**

```text
target
```

**Penerapan materi:**

```text
Variance Threshold
Polynomial Features
IQR Outlier Detection
K-Fold Cross Validation
Stratified K-Fold
Learning Curve
Bias-Variance Analysis
Log-Likelihood
Model Comparison
Feature Importance
```

**Algoritma manual:**

```text
1. Linear Regression / Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Ridge Regression
```
