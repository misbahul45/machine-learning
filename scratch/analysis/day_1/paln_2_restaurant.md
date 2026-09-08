Tambahkan satu bagian khusus: **Location Impact Analysis**. Fokusnya bukan langsung “lokasi menyebabkan rating”, tapi **seberapa besar lokasi membantu menjelaskan atau memprediksi rating**.

Pakai plan seperti ini:

```md
## Location Impact Analysis

Tujuan analisis ini adalah mengukur apakah lokasi restoran dan jarak user ke restoran memiliki hubungan terhadap nilai rating.

Kolom yang digunakan:

- latitude_x, longitude_x -> lokasi restoran
- latitude_y, longitude_y -> lokasi user
- city, state, country -> lokasi administratif restoran
- rating -> target utama
```

Urutan analisisnya:

```text
1. Analisis rating berdasarkan city/state
2. Hitung jarak user ke restoran
3. Buat kategori jarak: near, medium, far
4. Analisis rating berdasarkan jarak
5. Buat location cluster dari koordinat restoran
6. Bandingkan rata-rata rating tiap cluster lokasi
7. Uji apakah lokasi memberi pengaruh prediktif lewat model comparison
```

Fitur baru yang bisa kamu buat:

```text
distance_user_restaurant_km
distance_category
restaurant_location_cluster
```

Contoh logika analisis:

```text
Analisis city/state terhadap rating
menggunakan groupby mean rating dan bar chart
untuk melihat apakah restoran di lokasi tertentu cenderung memiliki rating lebih tinggi.
```

```text
Analisis jarak user ke restoran terhadap rating
menggunakan Haversine distance, boxplot, dan correlation
untuk melihat apakah user yang lebih dekat cenderung memberi rating lebih tinggi.
```

```text
Analisis cluster lokasi restoran terhadap rating
menggunakan K-Means clustering pada latitude dan longitude
untuk melihat apakah area geografis tertentu memiliki pola rating berbeda.
```

Bagian terpenting untuk mengukur “seberapa ngaruh” adalah **model comparison**:

```text
Model A: tanpa fitur lokasi
Model B: dengan fitur lokasi
```

Contohnya:

```text
Model A features:
food_rating, service_rating, price, budget, alcohol, parking, ambience

Model B features:
food_rating, service_rating, price, budget, alcohol, parking, ambience,
distance_user_restaurant_km, city, state, restaurant_location_cluster
```

Lalu bandingkan hasilnya:

```text
Jika Model B memiliki RMSE lebih rendah dan R2 lebih tinggi daripada Model A,
berarti fitur lokasi membantu model memprediksi rating.
```

Tambahkan section ini di notebook:

```md
## 2.9 Location Impact Analysis

### 2.9.1 Restaurant Location Distribution
### 2.9.2 Rating by City and State
### 2.9.3 User-to-Restaurant Distance
### 2.9.4 Rating by Distance Category
### 2.9.5 Restaurant Location Clustering
### 2.9.6 Location Feature Impact on Model Performance
```

Kesimpulan yang nanti ingin kamu cari:

```text
Apakah restoran di kota/state tertentu punya rata-rata rating lebih tinggi?
Apakah jarak user ke restoran memengaruhi rating?
Apakah cluster lokasi restoran memiliki pola rating berbeda?
Apakah fitur lokasi meningkatkan performa model?
```

Jadi alurnya jelas:

```text
Lokasi restoran
↓
Jarak user ke restoran
↓
Cluster area restoran
↓
Hubungan dengan rating
↓
Uji dampaknya ke model
```
