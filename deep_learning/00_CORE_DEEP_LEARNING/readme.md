# 📘 00_CORE_DEEP_LEARNING — EXPANDED (WITH PER-PROJECT GOALS)

## 🏗️ Topik yang Dicakup:
- CNN (Convolutional Neural Networks)
- RNN (Recurrent Neural Networks)
- LSTM/GRU
- Transformer Architecture
- Generative Models (VAE, GAN)
- Reinforcement Learning Basics

---

## 🔹 CNN (Convolutional Neural Networks)

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. Binary Image Classifier** — Bedakan kucing vs anjing (dataset Kaggle). Train dari scratch dengan arsitektur sederhana 3-layer CNN.
- 🎯 **Goals:**
  - Memahami pipeline end-to-end: load image → preprocess → forward pass → loss → backprop → update
  - Mengerti perbedaan antara konvolusi dan fully connected: kenapa conv jauh lebih efisien untuk data spasial
  - Bisa debug overfitting dasar: train acc tinggi tapi val acc rendah → tambah dropout atau kurangi kapasitas
  - Memahami binary cross-entropy loss dan sigmoid output
  - Bisa interpret confusion matrix untuk binary classification
  - Mengerti kenapa image augmentation (flip, crop, brightness) meningkatkan generalisasi
  - Bisa memilih optimizer (Adam vs SGD) dan menjelaskan perbedaan efek training-nya

**2. Digit Recognizer (MNIST)** — Klasifikasi tulisan tangan 0-9. Implementasi CNN pertama dengan Conv → Pool → FC.
- 🎯 **Goals:**
  - Memahami arsitektur Conv → Pool → FC secara intuitif: extract features → compress → classify
  - Mengerti operasi convolution secara matematis: dot product antara kernel dan patch gambar
  - Memahami max pooling untuk spatial downsampling dan mengapa ini membantu translational invariance
  - Bisa menjelaskan parameter count di setiap layer (conv vs FC)
  - Mengerti softmax + categorical cross-entropy untuk multi-class output
  - Bisa visualisasikan output feature map setelah setiap conv layer
  - Memahami flatten operation sebelum FC layer

**3. Color Detector** — Klasifikasi warna dominan pada gambar.
- 🎯 **Goals:**
  - Memahami bagaimana low-level feature (warna, edge) ditangkap oleh conv layer pertama
  - Mengerti representasi RGB dan bagaimana filter belajar sensitif terhadap channel tertentu
  - Bisa visualisasikan dan interpret learned filters di layer awal
  - Memahami bahwa conv layer awal belajar detector edge/color, bukan semantik
  - Mengerti pengaruh ukuran kernel (3x3 vs 5x5) terhadap receptive field dan smoothness feature
  - Bisa menjelaskan mengapa spatial features di gambar lebih informatif dari raw pixels

**4. Image Brightness Classifier** — Prediksi apakah gambar terang/gelap/sedang.
- 🎯 **Goals:**
  - Memahami global average pooling sebagai alternatif flatten + FC
  - Mengerti konsep spatial averaging vs spatial max dalam feature aggregation
  - Bisa mengidentifikasi kapan task sangat sederhana sehingga deep network overkill (overparameterization)
  - Memahami hubungan antara channel mean dan brightness prediction
  - Mengerti bias-variance tradeoff dalam konteks model capacity vs task complexity

**5. Chest X-Ray Normal vs Pneumonia** — Binary classification medical imaging.
- 🎯 **Goals:**
  - Memahami tantangan class imbalance: ketika satu kelas jauh lebih banyak dari yang lain
  - Bisa gunakan weighted loss function atau oversampling (SMOTE) untuk imbalanced data
  - Mengerti metrik yang tepat untuk medical imaging: sensitivity, specificity, AUC-ROC (bukan hanya accuracy)
  - Memahami transfer learning dasar: mengapa pretrained model lebih baik untuk data medis yang kecil
  - Bisa menggunakan Grad-CAM untuk memvisualisasikan area yang dilihat model saat prediksi
  - Mengerti implikasi false negative vs false positive dalam konteks medis (cost asymmetry)

---

#### 🟡 INTERMEDIATE

**6. CIFAR-10 Classifier** — 10-class classification.
- 🎯 **Goals:**
  - Memahami efek kedalaman jaringan: mengapa 10-layer CNN lebih baik dari 3-layer untuk dataset kompleks
  - Mengerti regularisasi melalui dropout: berapa probability terbaik dan di mana menempatkannya
  - Bisa implementasi data augmentation pipeline: flip horizontal, random crop, color jitter
  - Memahami learning rate scheduling: step decay, cosine annealing, warmup
  - Mengerti perbedaan train/val/test split dan kenapa test set hanya digunakan sekali
  - Bisa analisis kesalahan (error analysis): kelas mana yang sering salah dan mengapa
  - Memahami trade-off antara model size, training time, dan accuracy

**7. CIFAR-100 dengan Transfer Learning** — Fine-tune ResNet18/VGG16 pretrained ImageNet.
- 🎯 **Goals:**
  - Memahami konsep transfer learning: feature dari domain sumber bisa berguna di domain target
  - Mengerti perbedaan feature extraction (freeze base) vs full fine-tuning (unfreeze semua)
  - Bisa memilih layer mana yang di-freeze dan mana yang di-train berdasarkan task similarity
  - Memahami kenapa learning rate fine-tuning harus jauh lebih kecil dari training from scratch
  - Mengerti representasi hierarkis: layer awal = textures, layer tengah = parts, layer akhir = objects
  - Bisa menggunakan ImageNet normalization statistics dan mengapa ini penting
  - Memahami keterbatasan transfer learning ketika domain source dan target sangat berbeda (domain shift)

**8. Custom Object Detector (YOLO-lite)** — Deteksi 2-3 objek sederhana dengan sliding window.
- 🎯 **Goals:**
  - Memahami perbedaan classification vs detection: output adalah koordinat bounding box + class
  - Mengerti sliding window approach dan kelemahannya (computationally expensive, fixed scale)
  - Bisa implementasi bounding box regression: prediksi (x, y, w, h) relatif terhadap anchor
  - Memahami IoU (Intersection over Union) sebagai metrik overlap antar bounding box
  - Mengerti Non-Maximum Suppression (NMS) untuk menghilangkan duplicate detection
  - Bisa definisikan multi-task loss: classification loss + localization loss
  - Memahami kenapa anchor boxes penting untuk menangkap objek dengan berbagai aspek rasio

**9. Face vs Non-Face Detector** — Binary detector dengan dataset Labeled Faces in the Wild.
- 🎯 **Goals:**
  - Memahami konsep hard negative mining: mengambil false positive paling sulit untuk re-training
  - Mengerti bagaimana model bisa belajar bias dari data (mis. semua wajah dari skin tone tertentu)
  - Bisa implementasi precision-recall curve dan memilih threshold berdasarkan kebutuhan aplikasi
  - Memahami multi-scale detection: wajah bisa muncul dalam berbagai ukuran dalam gambar
  - Mengerti bootstrap sampling untuk membangun dataset negatif yang challenging
  - Bisa mengukur dan mitigasi gender/race bias dalam face detector

**10. Plant Disease Classifier** — 38 kelas penyakit tanaman dari dataset PlantVillage.
- 🎯 **Goals:**
  - Memahami strategi untuk dataset imbalanced berskala besar: focal loss, class-weighted sampling
  - Bisa implementasi hierarchical classification: tanaman → penyakit (coarse-to-fine)
  - Mengerti bagaimana texture (spots, lesions, discoloration) sebagai diskriminatif feature untuk penyakit
  - Memahami cara mengukur per-class accuracy dan menemukan kelas yang underperform
  - Bisa menggunakan confusion matrix untuk menemukan pasangan kelas yang sering salah klasifikasi
  - Mengerti real-world deployment challenge: variasi pencahayaan, sudut kamera, background

**11. Traffic Sign Recognition** — Dataset GTSRB.
- 🎯 **Goals:**
  - Memahami bagaimana menangani variasi kondisi real-world: pencahayaan, blur, okluasi, sudut pandang
  - Bisa implementasi augmentasi yang domain-specific: color jitter (warna lampu), perspective transform
  - Mengerti pentingnya class weighting untuk dataset yang class-imbalanced
  - Memahami confidence calibration: apakah model yang confident 95% benar-benar akurat 95% dari waktu?
  - Bisa mengukur model robustness terhadap corruptions (Gaussian noise, motion blur)
  - Mengerti konsep safety-critical classifier dan threshold yang sesuai

**12. Neural Style Transfer** — Gabungkan content + style dengan Gram matrix.
- 🎯 **Goals:**
  - Memahami bahwa CNN features encode baik content (apa yang ada) maupun style (bagaimana tampilannya)
  - Mengerti Gram matrix sebagai cara mengukur feature correlations antar spatial locations (style)
  - Bisa implementasi optimization di image space (bukan parameter space): gradient descent pada pixel
  - Memahami perceptual loss vs pixel loss dan mengapa perceptual lebih baik untuk visual quality
  - Mengerti layer depth dan representasi: layer dangkal = textures, layer dalam = semantic content
  - Bisa control trade-off content vs style melalui loss weights
  - Memahami bahwa visualisasi ini adalah cara memeriksa apa yang "dipelajari" CNN

---

#### 🔴 ADVANCED

**13. YOLOv1 from Scratch** — Implementasi full YOLO architecture tanpa library detection.
- 🎯 **Goals:**
  - Memahami sepenuhnya grid cell approach: gambar dibagi ke grid S×S, setiap cell prediksi B box + C kelas
  - Bisa implementasi multi-part loss function YOLO: coordinate loss + confidence loss + class loss
  - Mengerti kenapa YOLO jauh lebih cepat dari region-based detector (R-CNN, Faster R-CNN)
  - Memahami anchor box design: bagaimana clustering bounding box di training set menentukan anchor shape
  - Bisa debug training instability: NaN loss, gradient explosion, dead neurons
  - Mengerti bagaimana inference pipeline bekerja: forward pass → NMS → threshold filtering
  - Memahami limitation YOLO v1: kesulitan deteksi objek kecil yang berkumpul

**14. U-Net Medical Image Segmentation** — Pixel-wise segmentation dari MRI/CT.
- 🎯 **Goals:**
  - Memahami encoder-decoder architecture: encoder compress, decoder upsample kembali
  - Mengerti skip connections dalam U-Net: mengapa menggabungkan feature encoder dan decoder penting
  - Bisa implementasi transposed convolution (deconvolution) untuk upsampling yang dipelajari
  - Memahami Dice loss untuk segmentasi medis dan mengapa lebih baik dari BCE pada imbalanced masks
  - Bisa evaluasi dengan IoU, Dice coefficient, Hausdorff distance
  - Mengerti bagaimana patch-based training digunakan untuk gambar medis yang sangat besar
  - Memahami uncertainty estimation: aleatoric vs epistemic uncertainty dalam prediksi medis

**15. ResNet from Scratch + Ablation Study** — Bangun ResNet-18/34 manual.
- 🎯 **Goals:**
  - Memahami secara matematis mengapa residual connection menyelesaikan degradation problem
  - Bisa implementasi identity shortcut dan projection shortcut (untuk dimension mismatch)
  - Mengerti mengapa sangat deep network tanpa residual connection tidak bisa dilatih (gradient vanishing)
  - Bisa lakukan ablation study: hitung akurasi dengan/tanpa residual, dengan berbagai kedalaman
  - Memahami batch normalization placement: sebelum atau sesudah activation (pre-activation vs post-activation)
  - Mengerti bottleneck block (1×1 → 3×3 → 1×1) dan mengapa mengurangi parameter
  - Bisa vizualize gradient magnitude per layer untuk membuktikan gradient highway

**16. EfficientNet Scaling Experiments** — Eksperimen compound scaling.
- 🎯 **Goals:**
  - Memahami tiga dimensi scaling: depth (lebih banyak layer), width (lebih banyak channel), resolution (input lebih besar)
  - Mengerti mengapa scaling satu dimensi saja memberikan diminishing returns
  - Bisa mengukur trade-off accuracy vs FLOPs vs parameter count secara sistematis
  - Memahami Neural Architecture Search (NAS) sebagai cara menemukan baseline architecture
  - Mengerti MBConv block: depthwise separable convolution dengan squeeze-and-excitation
  - Bisa plot Pareto frontier: model terbaik pada setiap titik resource budget
  - Memahami kenapa compound scaling lebih efisien: depth, width, resolution saling bergantung

**17. Face Recognition System (ArcFace)** — End-to-end: detection → alignment → embedding → metric learning.
- 🎯 **Goals:**
  - Memahami face recognition sebagai embedding problem, bukan classification problem
  - Mengerti metric learning: intraclass distance kecil, interclass distance besar
  - Bisa implementasi ArcFace loss: additive angular margin pada unit hypersphere
  - Memahami perbedaan face verification (1:1) vs face identification (1:N)
  - Mengerti face alignment: mengapa normalisasi pose (Procrustes analysis, 5-point landmark) meningkatkan akurasi
  - Bisa evaluasi dengan TAR@FAR curve (True Accept Rate at given False Accept Rate)
  - Memahami open-set vs closed-set recognition: known vs unknown identities

**18. Real-time Object Detection Pipeline** — YOLO + OpenCV + webcam.
- 🎯 **Goals:**
  - Memahami inference optimization: model quantization (FP32 → INT8), pruning, dan dampaknya pada accuracy
  - Bisa convert model ke ONNX dan jalankan dengan ONNX Runtime untuk backend-agnostic inference
  - Mengerti latency vs throughput trade-off: batch size 1 vs batching untuk real-time applications
  - Memahami TensorRT optimization: layer fusion, precision reduction untuk NVIDIA GPU
  - Bisa profiling bottleneck: preprocessing time vs inference time vs postprocessing time
  - Mengerti streaming pipeline: frame drop strategy ketika GPU tidak bisa keep up dengan camera FPS
  - Memahami hardware-aware model design: mengapa mobile-optimized models (MobileNet, NanoDet) penting

**19. Semi-Supervised Learning dengan Pseudo-Labels** — Gunakan unlabeled data.
- 🎯 **Goals:**
  - Memahami mengapa labeled data mahal dan unlabeled data berlimpah di dunia nyata
  - Bisa implementasi self-training loop: train on labeled → predict unlabeled → add high-confidence as pseudo-labels
  - Mengerti risiko confirmation bias: model salah yang mengajar dirinya sendiri kesalahan yang sama
  - Memahami confidence thresholding: mengapa memilih threshold yang tepat sangat kritis
  - Bisa implementasi FixMatch: consistency regularization + pseudo-labeling combined
  - Mengerti Mean Teacher: model EMA sebagai teacher yang lebih stabil dari student model
  - Memahami bagaimana mengukur gain dari unlabeled data secara fair (controlled experiment)

**20. Adversarial Robustness Study** — Generate FGSM/PGD examples + adversarial training.
- 🎯 **Goals:**
  - Memahami mengapa small perturbations imperceptible to humans bisa fool CNN
  - Bisa implementasi FGSM: perturbasi searah gradient sign, bounded oleh epsilon-ball
  - Mengerti PGD attack: FGSM iteratif, lebih kuat dari single-step attack
  - Memahami adversarial training: augmentasi dengan adversarial examples meningkatkan robustness
  - Bisa ukur accuracy di bawah attack dengan berbagai epsilon menggunakan certified robustness evaluator
  - Mengerti certified robustness vs empirical robustness: mengapa adaptive attacks lebih jujur
  - Memahami trade-off robustness vs accuracy: robust model biasanya sedikit kurang akurat pada clean data

---

## 🔹 RNN (Recurrent Neural Networks)

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. Palindrome Checker** — Klasifikasikan apakah string adalah palindrom.
- 🎯 **Goals:**
  - Memahami bagaimana RNN memproses sequence: hidden state diperbarui di setiap timestep
  - Mengerti parameter sharing: bobot yang sama digunakan di semua timestep (tidak seperti FC)
  - Bisa implementasi forward pass RNN manual: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
  - Memahami mengapa RNN bisa "ingat" informasi dari awal sequence (melalui hidden state)
  - Bisa debug vanishing hidden state: setelah banyak timestep, informasi awal hilang

**2. Count Vowels in Sequence** — Hitung jumlah vokal menggunakan hidden state accumulation.
- 🎯 **Goals:**
  - Memahami hidden state sebagai "running counter" atau "accumulator" yang berevolusi seiring waktu
  - Mengerti bagaimana RNN bisa belajar arithmetic sederhana dari sequential input
  - Bisa visualisasikan hidden state per timestep untuk memverifikasi bahwa model belajar mengakumulasi
  - Memahami many-to-one architecture: sequence of inputs → single output di akhir

**3. Binary Sequence Parity** — Prediksi parity jumlah 1 dalam binary sequence.
- 🎯 **Goals:**
  - Memahami XOR problem yang diperluas ke sequence: parity adalah canonical sequential task
  - Mengerti bagaimana RNN bisa implement finite state machine melalui hidden state
  - Bisa visualisasikan bahwa hidden state "flips" setiap kali menemukan angka 1
  - Memahami mengapa task ini sulit untuk model non-sequential (bag-of-words)
  - Mengerti generalisasi sequence length: dapatkah model train pada panjang 10 generalize ke panjang 20?

**4. Temperature Trend Predictor** — Prediksi naik/turun/tetap dari sequence suhu harian.
- 🎯 **Goals:**
  - Memahami perbedaan regression vs classification output untuk time series
  - Bisa implementasi sliding window: gunakan T timestep sebelumnya untuk prediksi berikutnya
  - Mengerti normalisasi data time series: mean subtraction dan std division penting untuk training stability
  - Memahami bagaimana hidden state encode "trend momentum" dari data historis

**5. Character-level Autocomplete** — Lengkapi satu karakter dari prefix kata.
- 🎯 **Goals:**
  - Memahami character-level language model: output adalah distribusi probabilitas atas vocabulary
  - Bisa implementasi teacher forcing: gunakan ground truth sebagai input (bukan prediksi sebelumnya) saat training
  - Mengerti one-hot encoding untuk karakter dan embedding layer sebagai alternatif
  - Memahami argmax vs sampling dari distribusi untuk generation (greedy vs stochastic)

---

#### 🟡 INTERMEDIATE

**6. Sentiment Analysis Sequential** — Klasifikasi review positif/negatif word-by-word.
- 🎯 **Goals:**
  - Memahami embedding layer: dari token ID ke dense vector representation yang dipelajari
  - Bisa bandingkan RNN vs bag-of-words: RNN menangkap "not good" (negation), BOW tidak
  - Mengerti padding dan masking: sequence panjang berbeda-beda, padding mengisi dengan zero
  - Memahami pengaruh panjang sequence: panjang sequence melemahkan hidden state RNN
  - Bisa implementasi global average pooling atas output semua timestep (vs hanya last hidden state)

**7. Character-level Text Generation** — Generate teks baru dari model bahasa.
- 🎯 **Goals:**
  - Memahami language model sebagai conditional probability: P(karakter berikutnya | semua karakter sebelumnya)
  - Bisa implementasi temperature sampling: temperature tinggi → lebih random, rendah → lebih deterministik
  - Mengerti konsep perplexity sebagai ukuran kualitas language model (lower is better)
  - Memahami mode collapse dalam generation: kenapa model bisa terjebak mengulang frasa yang sama
  - Bisa generate teks dengan berbagai seed string dan analisis koherensi output per epoch training

**8. Stock Price Direction Prediction** — Prediksi naik/turun dari 30-day window.
- 🎯 **Goals:**
  - Memahami look-ahead bias: mengapa memastikan tidak ada fitur dari masa depan dalam training sangat kritis
  - Bisa implementasi proper time series split: bukan random split, harus chronological
  - Mengerti feature engineering untuk time series: returns, moving averages, RSI, Bollinger bands
  - Memahami mengapa prediksi harga saham sangat sulit: efficient market hypothesis
  - Bisa evaluasi dengan financial metrics: Sharpe ratio, max drawdown (bukan hanya accuracy)

**9. Morse Code Encoder/Decoder** — Train RNN untuk encode/decode Morse.
- 🎯 **Goals:**
  - Memahami many-to-many architecture dengan aligned output (setiap input punya output)
  - Bisa implementasi bidirectional RNN untuk decoder yang perlu konteks kedua arah
  - Mengerti sequence-to-sequence tanpa attention: encoder state harus compress semua informasi
  - Memahami CTC loss sebagai alternatif untuk alignment yang tidak diketahui
  - Bisa visualisasikan output sequence dan hitung character error rate (CER)

**10. Spam Classifier dengan RNN** — Email spam detection.
- 🎯 **Goals:**
  - Memahami truncation strategy: ketika sequence terlalu panjang, potong dari mana (awal/akhir/tengah)?
  - Bisa bandingkan padding strategies: pre-padding vs post-padding dan efeknya pada gradient flow
  - Mengerti mengapa subject line mungkin lebih informatif dari body untuk spam detection
  - Memahami attention-weighted pooling sebagai cara fokus pada kata paling diskriminatif

**11. Language Identification** — Identifikasi bahasa dari sentence.
- 🎯 **Goals:**
  - Memahami perbedaan character-level vs word-level model untuk language ID
  - Mengerti mengapa character n-gram sangat efektif: bahasa berbeda = pola karakter berbeda
  - Bisa implementasi dan bandingkan character RNN vs FastText vs logistic regression pada character n-gram
  - Memahami bahwa untuk language ID, model sederhana seringkali cukup baik (no need for deep RNN)
  - Bisa uji robustness terhadap code-switching (campuran dua bahasa dalam satu kalimat)

**12. Simple Music Generation** — Generate melodi MIDI dari sequence note.
- 🎯 **Goals:**
  - Memahami representasi musik sebagai discrete sequence: pitch + duration + rest
  - Bisa implementasi sampling dengan constraints musikalitas (mis. hanya nada dalam skala)
  - Mengerti priming sequence: mulai dari motif musik tertentu untuk guide generation
  - Memahami evaluasi generative model musik: novelty vs memorization, harmonic consistency

---

#### 🔴 ADVANCED

**13. Seq2Seq Language Translation (No Attention)** — English → Indonesian tanpa attention.
- 🎯 **Goals:**
  - Memahami information bottleneck problem: semua informasi source harus masuk ke satu hidden vector
  - Bisa quantifikasi penurunan performa seiring panjang sequence meningkat (kenapa attention diperlukan)
  - Mengerti beam search decoding: menjaga K hypothesis terbaik versus greedy decoding
  - Memahami BLEU score: precision dari n-gram overlap antara prediction dan reference
  - Bisa analisis error: translation kalimat pendek vs panjang, konkret vs abstrak
  - Mengerti bahwa model ini adalah baseline untuk motivasi attention mechanism

**14. Handwriting Synthesis** — Generate sequence koordinat pen stroke (MDN-RNN).
- 🎯 **Goals:**
  - Memahami Mixture Density Network: output adalah parameter distribusi campuran, bukan single value
  - Bisa implementasi Gaussian mixture model output untuk prediksi koordinat kontinyu
  - Mengerti pen-up/pen-down binary output bersama dengan koordinat x, y
  - Memahami bagaimana conditioning pada teks memandu synthesis ke karakter tertentu
  - Bisa visualisasikan sampling dari distribusi mixture dan tuning temperature

**15. Anomaly Detection Time Series** — Deteksi anomali sensor IoT.
- 🎯 **Goals:**
  - Memahami reconstruction-based anomaly detection: normal data direkonstruksi dengan baik, anomali tidak
  - Bisa implementasi LSTM autoencoder: encoder compress ke latent, decoder rekonstruksi sequence
  - Mengerti cara menentukan threshold anomaly dari distribusi reconstruction error pada validation set
  - Memahami konsep point anomaly vs contextual anomaly vs collective anomaly
  - Bisa evaluasi dengan precision@k dan F1 untuk anomaly detection (bukan accuracy)

**16. Neural Machine Translation dengan BPTT Debug** — Implementasi BPTT manual.
- 🎯 **Goals:**
  - Bisa implementasi Backpropagation Through Time secara manual langkah demi langkah
  - Memahami mengapa gradient vanish: perkalian matriks berulang menyebabkan nilai mengecil secara eksponensial
  - Bisa visualisasikan gradient magnitude per timestep menggunakan gradient norm histogram
  - Mengerti truncated BPTT: mengapa memotong gradient flow pada interval tertentu membantu practical training
  - Memahami gradient clipping: teknik pragmatis untuk mencegah exploding gradient

**17. Algorithmic Task Solver** — RNN belajar melakukan operasi aritmatika dari digit sequence.
- 🎯 **Goals:**
  - Memahami perbedaan interpolation generalisasi (dalam training range) vs extrapolation (di luar range)
  - Bisa eksperimen: apakah RNN yang dilatih pada addition panjang 1-10 bisa generalize ke panjang 15?
  - Mengerti bahwa standard RNN gagal pada algoritma yang butuh presisi memory panjang
  - Memahami mengapa Neural Turing Machine / Differentiable Neural Computer diciptakan untuk task ini

**18. Text Adventure Game AI** — RNN model dunia game text-based.
- 🎯 **Goals:**
  - Memahami world model sebagai sequence-to-sequence task: action → state description
  - Bisa implementasi conditional generation: output bergantung pada current state + action
  - Mengerti long-range dependency dalam game: keputusan awal mempengaruhi outcome jauh kemudian
  - Memahami mengapa game ini adalah benchmark yang baik untuk memory dalam sequence model

**19. Real-time Gesture Recognition** — Klasifikasi gerakan tangan dari sequence landmark.
- 🎯 **Goals:**
  - Memahami representasi gesture sebagai sequence koordinat landmark (bukan raw video)
  - Bisa implementasi sliding window inference untuk real-time classification
  - Mengerti pre- dan post-processing: normalisasi koordinat relatif ke palm center dan size
  - Memahami latency requirement dalam real-time system dan cara optimize inference

**20. Reservoir Computing / Echo State Network** — Implementasi random fixed RNN.
- 🎯 **Goals:**
  - Memahami bahwa hanya output layer yang dilatih: hidden weights tetap random dan fixed
  - Mengerti mengapa ini bekerja: reservoir project input ke high-dimensional nonlinear feature space
  - Bisa bandingkan dengan trained RNN: kapan reservoir cukup baik? kapan trained lebih baik?
  - Memahami spectral radius: kontrol stabilitas dan memory capacity reservoir
  - Mengerti hubungan dengan kernel methods: random features sebagai implicit kernel mapping

---

## 🔹 LSTM / GRU

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. Word-level Next Word Prediction** — Prediksi kata berikutnya dalam kalimat.
- 🎯 **Goals:**
  - Memahami word-level language model: vocabulary sebagai output classes, next word sebagai label
  - Bisa implementasi embedding lookup table dan mengerti mengapa ini bisa dipelajari end-to-end
  - Mengerti perplexity: exp(cross-entropy loss) sebagai metrik standar language model
  - Memahami bagaimana LSTM mempertahankan konteks jangka panjang vs vanillah RNN
  - Bisa visualisasikan prediction probability distribution atas vocabulary

**2. Simple Time Series Forecasting** — Prediksi 1-step univariate.
- 🎯 **Goals:**
  - Memahami window-based forecasting: gunakan T timestep sebelumnya untuk prediksi t+1
  - Bisa bandingkan LSTM vs ARIMA vs exponential smoothing pada dataset AirPassengers
  - Mengerti stationarity dan mengapa differencing membantu: remove trend dan seasonality
  - Memahami MSE vs MAE loss untuk regression: MSE lebih sensitif terhadap outlier
  - Bisa plot actual vs predicted dengan confidence interval

**3. Emoji Sentiment Classifier** — Prediksi emoji yang cocok untuk kalimat pendek.
- 🎯 **Goals:**
  - Memahami LSTM sebagai feature extractor untuk teks: last hidden state sebagai sentence representation
  - Bisa implementasi pre-trained word embeddings (GloVe/FastText) dan fine-tune vs freeze
  - Mengerti mengapa emoji prediction adalah richer supervised signal daripada binary sentiment
  - Memahami perbedaan character-level vs word-level LSTM untuk teks pendek

**4. Sequence Copy Task** — LSTM belajar copy input sequence ke output setelah delay.
- 🎯 **Goals:**
  - Memahami bahwa copy task adalah eksperimen diagnostic untuk long-term memory
  - Bisa kuantifikasi: pada delay berapa timestep LSTM masih bisa copy akurat?
  - Mengerti mengapa vanilla RNN gagal total pada copy task dengan delay panjang
  - Memahami cell state sebagai "conveyor belt": informasi bisa mengalir tanpa transformasi besar

**5. Password Strength Scorer** — Klasifikasi kekuatan password dari sequence karakter.
- 🎯 **Goals:**
  - Memahami karakter sequence sebagai input alami untuk LSTM (bukan kata)
  - Bisa implementasi character-level LSTM dengan embedding layer untuk karakter
  - Mengerti bagaimana LSTM bisa belajar rules implisit: panjang, diversity, entropy karakter
  - Memahami ordinal regression (weak < medium < strong) vs binary classification

---

#### 🟡 INTERMEDIATE

**6. Named Entity Recognition (NER)** — Tag entitas dalam teks.
- 🎯 **Goals:**
  - Memahami sequence labeling: setiap token mendapat label (B-PER, I-PER, O, dll. dalam BIO scheme)
  - Bisa implementasi BiLSTM-CRF: BiLSTM extract features, CRF model label dependencies
  - Mengerti mengapa CRF layer penting: mencegah output ilegal (mis. I-PER setelah O)
  - Memahami Viterbi algorithm untuk CRF decoding
  - Bisa evaluasi dengan entity-level F1 (bukan token-level accuracy)
  - Mengerti CoNLL evaluation format dan scoring

**7. Multivariate Time Series Forecasting** — Prediksi dari banyak sensor.
- 🎯 **Goals:**
  - Memahami multivariate input: setiap timestep adalah vektor fitur (bukan scalar)
  - Bisa implementasi LSTM dengan multiple output steps (multi-step forecasting)
  - Mengerti feature correlation: bagaimana LSTM belajar relasi antar sensor secara implicit
  - Memahami missing data imputation sebelum LSTM: interpolasi linear vs model-based imputation
  - Bisa evaluasi RMSE per variabel dan identify variable mana yang paling sulit diprediksi

**8. Headline Generator** — Generate judul berita dari ringkasan artikel.
- 🎯 **Goals:**
  - Memahami conditional text generation: generate headline yang relevan dengan konten article
  - Bisa implementasi sequence-to-sequence LSTM sederhana: encode article → decode headline
  - Mengerti length normalization dalam beam search: mencegah model prefer kalimat pendek
  - Memahami ROUGE score sebagai evaluasi summarization: recall-oriented overlap
  - Bisa analisis diversity vs relevance trade-off dalam generated headlines

**9. Code Completion (Simple)** — Autocomplete Python/JS satu baris.
- 🎯 **Goals:**
  - Memahami tokenization khusus kode: token berbeda dari bahasa alami (operator, indentasi)
  - Bisa implementasi line-level LSTM yang kondisional pada context sebelumnya
  - Mengerti mengapa kode memiliki struktur yang lebih rigid dan predictable dari teks biasa
  - Memahami exact match accuracy vs token-level accuracy untuk code completion

**10. Bidirectional LSTM Sentiment** — Klasifikasi sentimen menggunakan BiLSTM.
- 🎯 **Goals:**
  - Memahami BiLSTM secara intuitif: forward pass baca kiri-ke-kanan, backward baca kanan-ke-kiri
  - Bisa bandingkan last hidden state BiLSTM vs unidirectional LSTM: apa keuntungannya?
  - Mengerti concatenation vs sum vs average untuk menggabungkan forward dan backward hidden states
  - Memahami mengapa "I thought the movie would be bad, but it was great" perlu konteks kedua arah

**11. Anomaly Detection ECG** — Deteksi aritmia dari signal ECG.
- 🎯 **Goals:**
  - Memahami LSTM autoencoder secara mendalam: encode normal patterns, anomali memiliki reconstruction error tinggi
  - Bisa implementasi time-distributed dense layer untuk per-timestep reconstruction
  - Mengerti bagaimana menset threshold: percentile dari error distribution pada normal validation data
  - Memahami clinical metric: sensitivity (recall) lebih penting dari specificity untuk aritmia detection
  - Bisa visualisasikan reconstruction overlay: predicted waveform vs actual pada anomaly region

**12. GRU vs LSTM Benchmark** — Bandingkan pada 3 dataset.
- 🎯 **Goals:**
  - Memahami perbedaan arsitektur GRU (2 gates: reset, update) vs LSTM (3 gates: forget, input, output)
  - Bisa ukur secara empiris: parameter count GRU ~75% LSTM, training speed ~1.3x faster
  - Mengerti bahwa GRU performa lebih baik pada dataset kecil/sequence pendek, LSTM pada dataset besar
  - Memahami update gate GRU sebagai gabungan forget+input gate LSTM
  - Bisa lakukan benchmark yang adil: sama hyperparameter budget, berbeda arsitektur

---

#### 🔴 ADVANCED

**13. Speech Recognition (phoneme-level)** — Klasifikasi phoneme dari MFCC features dengan BiLSTM + CTC.
- 🎯 **Goals:**
  - Memahami MFCC extraction: dari raw waveform ke mel-spectrogram ke cepstral coefficients
  - Bisa implementasi CTC loss: memungkinkan training tanpa explicit alignment antara audio dan teks
  - Mengerti CTC decoding: forward-backward algorithm, prefix beam search
  - Memahami bahwa CTC output adalah "blank" + phoneme, dan collapse rule untuk decoding
  - Bisa mengukur Phoneme Error Rate (PER) dan Word Error Rate (WER)

**14. Video Captioning** — Deskripsikan konten video pendek.
- 🎯 **Goals:**
  - Memahami CNN+LSTM pipeline: CNN extract visual features per frame, LSTM generate caption
  - Bisa implementasi mean pooling vs attention over frame features
  - Mengerti temporal modeling: sequence of frames mengandung motion information
  - Memahami evaluation: METEOR, CIDEr lebih baik dari BLEU untuk captioning
  - Bisa handle video dengan panjang berbeda-beda menggunakan fixed-rate sampling

**15. Temporal Anomaly Detection Multi-sensor** — Deteksi anomali kompleks dari data pabrik.
- 🎯 **Goals:**
  - Memahami multivariate time series anomaly detection: korelasi antar sensor sebagai sinyal
  - Bisa implementasi graph-based anomaly: sensor sebagai node, korelasi sebagai edge
  - Mengerti distinction: univariate anomaly (satu sensor keluar range) vs multivariate (pola korelasi rusak)
  - Memahami evaluation pada industrial dataset: precision, recall, point-adjust metrics

**16. Transformer vs LSTM Ablation** — Ukur performa keduanya across sequence lengths.
- 🎯 **Goals:**
  - Bisa ukur secara sistematis: accuracy, training time per epoch, memory usage untuk L=16,64,128,512,1024
  - Memahami mengapa Transformer dominan untuk L > 256: parallelism + O(n²) vs sequential LSTM
  - Mengerti mengapa LSTM masih kompetitif untuk L < 64 dan pada dataset kecil
  - Bisa explain melalui plot: training curve stability, convergence speed, parameter efficiency

**17. Neural Turing Machine (Simplified)** — Implementasi NTM dengan external memory.
- 🎯 **Goals:**
  - Memahami differentiable memory: continuous, soft addressing memungkinkan end-to-end training
  - Bisa implementasi content-based addressing: cosine similarity antara query dan memory slots
  - Mengerti location-based addressing: konvolusi shift atas memory untuk "geser pointer"
  - Memahami mengapa NTM bisa solve algorithmic tasks yang gagal pada standard LSTM
  - Bisa visualisasikan read dan write attention weights atas memory slots per timestep

**18. LSTM Language Model Perplexity Optimization** — Fine-tune hiperparameter.
- 🎯 **Goals:**
  - Memahami perplexity sebagai fungsi dari cross-entropy loss dan bagaimana keduanya berkorelasi
  - Bisa lakukan systematic grid search: layers (1,2,3), units (256,512,1024), dropout (0.2,0.3,0.5)
  - Mengerti zoneout sebagai alternatif dropout untuk RNN: drop state transitions bukan activations
  - Memahami Averaged SGD: average parameter over training trajectory memberikan smoother generalization
  - Bisa mengidentifikasi diminishing returns: titik mana yang model tidak perlu lebih besar

**19. Multi-task LSTM** — Satu model shared untuk NER + POS + dependency parsing.
- 🎯 **Goals:**
  - Memahami hard vs soft parameter sharing: shared LSTM vs task-specific tower
  - Bisa implementasi task-specific output heads di atas shared encoder
  - Mengerti negative transfer: ketika task-task saling mengganggu dan tidak membantu
  - Memahami bagaimana multi-task learning sebagai regularization: mencegah overfitting ke satu task
  - Bisa menggunakan loss weighting dan curriculum (prioritas task tertentu) untuk multi-task training

**20. Online Learning LSTM** — Update model secara incremental tanpa catastrophic forgetting.
- 🎯 **Goals:**
  - Memahami catastrophic forgetting: model overwrite knowledge lama saat belajar data baru
  - Bisa implementasi Elastic Weight Consolidation (EWC): regularize bobot penting berdasarkan Fisher information
  - Mengerti reservoir sampling: simpan subset representatif data lama untuk replay
  - Memahami progressive neural networks: beku jaringan lama, tambah kapasitas baru
  - Bisa ukur backward transfer dan forward transfer secara empiris

---

## 🔹 Transformer Architecture

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. Sentiment Classifier dengan Pre-trained BERT** — Fine-tune BERT untuk sentimen review.
- 🎯 **Goals:**
  - Memahami HuggingFace API: tokenizer, model, trainer
  - Bisa explain apa itu [CLS] token dan mengapa digunakan sebagai sentence representation
  - Mengerti subword tokenization (WordPiece/BPE): cara menangani OOV (out-of-vocabulary) words
  - Memahami perbedaan fine-tuning semua layer vs hanya classification head
  - Mengerti learning rate sangat kecil untuk fine-tuning (1e-5 vs 1e-3 for training from scratch)

**2. Text Similarity (Cosine dari Embeddings)** — Hitung kesamaan antar kalimat.
- 🎯 **Goals:**
  - Memahami sentence embedding: dari token embeddings ke fixed-size sentence vector
  - Mengerti mean pooling vs [CLS] sebagai sentence representation: mana yang lebih baik?
  - Bisa implementasi cosine similarity dan explain mengapa normalisasi ke unit sphere penting
  - Memahami semantic similarity vs lexical similarity: kalimat berbeda kata bisa maknanya sama
  - Bisa evaluasi dengan STS (Semantic Textual Similarity) benchmark: Spearman correlation

**3. Simple Q&A dengan Extractive Reading** — Temukan jawaban dari paragraph.
- 🎯 **Goals:**
  - Memahami extractive QA: jawaban adalah span dalam passage, bukan generated text
  - Mengerti format input: [CLS] question [SEP] passage [SEP] sebagai single sequence
  - Bisa implementasi start/end span prediction head: dua linear classifiers atas token representations
  - Memahami bahwa fine-tuning BERT pada SQuAD mengubahnya dari "understand text" ke "find answers"

**4. Language Detection** — Fine-tune mBERT untuk identifikasi bahasa.
- 🎯 **Goals:**
  - Memahami multilingual BERT: satu model untuk 100+ bahasa, shared vocabulary dengan cross-lingual representations
  - Mengerti zero-shot cross-lingual transfer: train di satu bahasa, test di bahasa lain
  - Bisa bandingkan mBERT vs XLM-R: perbedaan training data dan performa di low-resource languages
  - Memahami bahwa language ID adalah task relatif mudah bahkan untuk model sederhana

**5. Spam Detection dengan DistilBERT** — Klasifikasi spam dengan lightweight transformer.
- 🎯 **Goals:**
  - Memahami knowledge distillation: DistilBERT adalah BERT yang di-distill (40% lebih kecil, 60% lebih cepat)
  - Bisa bandingkan inference speed dan accuracy: DistilBERT vs BERT vs LSTM
  - Mengerti trade-off: kapan cukup pakai model kecil dan kapan perlu model besar?
  - Memahami quantization tambahan untuk edge deployment

---

#### 🟡 INTERMEDIATE

**6. Machine Translation En-Id** — Fine-tune Helsinki-NLP opus-mt model.
- 🎯 **Goals:**
  - Memahami encoder-decoder Transformer architecture untuk sequence-to-sequence tasks
  - Bisa implementasi BLEU score evaluasi dengan SacreBLEU
  - Mengerti cross-attention: decoder "attends" ke semua encoder representations
  - Memahami mengapa teacher forcing digunakan saat training tapi autoregressive saat inference
  - Bisa identifikasi error pattern: terjemahan kalimat panjang, proper noun, idiom

**7. Text Summarization Abstractive** — Fine-tune BART/T5 untuk summarize berita.
- 🎯 **Goals:**
  - Memahami BART pre-training: denoising autoencoder yang corrupt dan rekonstruksi text
  - Mengerti abstractive vs extractive: abstractive bisa generate kata baru, bukan hanya copy
  - Bisa evaluasi dengan ROUGE-1, ROUGE-2, ROUGE-L dan interpretasi apa yang mereka ukur
  - Memahami length penalty dalam beam search: cegah summary terlalu pendek atau panjang
  - Bisa bandingkan T5 vs BART: pre-training objective berbeda, performa pada summarization

**8. Code Generation (Function Level)** — Fine-tune CodeT5 untuk generate fungsi Python dari docstring.
- 🎯 **Goals:**
  - Memahami CodeT5 pre-training: identifier-aware pre-training untuk kode
  - Bisa implementasi CodeBLEU: BLEU + syntax match + semantic match untuk kode
  - Mengerti tantangan code generation: sintaks valid + semantik benar + efficient
  - Memahami unit test sebagai evaluation: kode yang lulus test > BLEU score tinggi
  - Bisa analisis error: common mistake seperti wrong variable name, off-by-one error

**9. Document Classification Long Text** — Klasifikasi dokumen panjang dengan Longformer.
- 🎯 **Goals:**
  - Memahami mengapa standard BERT gagal pada dokumen > 512 token (quadratic memory)
  - Mengerti Longformer's sliding window attention: O(n) complexity vs O(n²)
  - Bisa implementasi BigBird's block sparse attention sebagai alternatif
  - Memahami strategi alternatif: chunk-and-aggregate vs hierarchical model vs truncation
  - Bisa bandingkan memory usage dan accuracy: Longformer vs chunk-mean-pooling vs truncation

**10. Transformer from Scratch (Toy Task)** — Implementasi full transformer tanpa library.
- 🎯 **Goals:**
  - Bisa implementasi scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V dari nol
  - Mengerti multi-head attention: project ke H subspaces, attend parallel, concatenate
  - Bisa implementasi positional encoding: sinusoidal formula PE(pos, 2i) = sin(pos/10000^(2i/d_model))
  - Memahami feed-forward sublayer: dua linear transformation dengan ReLU
  - Bisa implementasi pre-norm (LayerNorm sebelum sublayer) vs post-norm
  - Memahami masking: padding mask untuk encoder, causal mask untuk decoder

**11. Cross-lingual Transfer** — Zero-shot transfer ke Bahasa Indonesia.
- 🎯 **Goals:**
  - Memahami mengapa multilingual model bisa zero-shot transfer: shared multilingual representation space
  - Bisa ukur zero-shot gap: performa pada Indonesian tanpa fine-tuning vs dengan fine-tuning
  - Mengerti curse of multilinguality: lebih banyak bahasa → kapasitas per bahasa berkurang
  - Memahami faktor yang mempengaruhi transfer success: script similarity, typological similarity, resource amount

**12. Retrieval-Augmented QA** — Gabungkan DPR + reader untuk open-domain QA.
- 🎯 **Goals:**
  - Memahami RAG architecture: retriever mengambil relevant passages, reader extract answer
  - Bisa implementasi dense retriever dengan FAISS: encode passages ke vectors, nearest neighbor search
  - Mengerti DPR training: contrastive learning antara question dan positive/negative passages
  - Memahami end-to-end evaluation: Exact Match dan F1 untuk open-domain QA
  - Bisa analisis retrieval failure vs reading failure sebagai dua sumber error yang berbeda

---

#### 🔴 ADVANCED

**13. Attention Visualization & Probing** — Visualisasikan attention heads + probe tiap layer.
- 🎯 **Goals:**
  - Bisa implementasi probing classifier: linear model di atas frozen representations untuk prediksi linguistic property
  - Memahami bahwa attention ≠ explanation: attention weights tidak selalu menunjukkan "mengapa" prediksi
  - Bisa visualisasikan attention pattern untuk berbagai heads: induction heads, previous token heads, syntactic heads
  - Mengerti probing methodology: mengapa layer yang berbeda encode informasi linguistik yang berbeda
  - Memahami mutual information sebagai cara mengukur berapa banyak informasi yang tersimpan di setiap layer

**14. Efficient Transformer Comparison** — Implementasi Linformer/Performer/Reformer.
- 🎯 **Goals:**
  - Memahami Linformer: approximate attention dengan low-rank projection (O(n) memory)
  - Mengerti Performer: random feature approximation dari softmax attention (FAVOR+)
  - Bisa implementasi Reformer: Locality-Sensitive Hashing untuk approximate nearest neighbor attention
  - Bisa benchmark secara fair: perband sequence length vs memory vs speed vs accuracy tradeoff
  - Memahami mengapa vanilla attention masih dominan di banyak tasks meski ada efficient alternatives

**15. Multi-modal Transformer (Text + Image)** — Bangun CLIP-like model sederhana.
- 🎯 **Goals:**
  - Memahami contrastive learning: image dan caption yang matched harus memiliki embedding yang dekat
  - Bisa implementasi symmetric InfoNCE loss untuk text-image contrastive training
  - Mengerti zero-shot image classification via text prompts: "a photo of a [CLASS]"
  - Memahami temperature parameter dalam contrastive loss dan efeknya pada training
  - Bisa evaluasi dengan zero-shot classification accuracy dan image-text retrieval R@K

**16. Instruction Fine-tuning (SFT)** — Fine-tune GPT2 dengan format instruksi.
- 🎯 **Goals:**
  - Memahami format instruksi: "### Instruction: ... ### Response: ..." dan mengapa konsisten
  - Bisa curate instruction dataset kecil sendiri dan fine-tune GPT2
  - Mengerti perbedaan base model (predict next token) vs instruction-tuned model (follow instructions)
  - Memahami exposure bias: gap antara teacher forcing training dan autoregressive inference
  - Bisa evaluasi dengan human preference dan automatic metrics seperti MT-Bench

**17. Knowledge Distillation Transformer** — Distill BERT-base ke model 50% lebih kecil.
- 🎯 **Goals:**
  - Memahami soft labels dari teacher: mengapa distribution probability lebih informatif dari one-hot
  - Bisa implementasi KD loss: kombinasi task loss + KL divergence dari teacher logits
  - Mengerti intermediate distillation: distill hidden states dan attention maps antar layer
  - Memahami temperature scaling dalam distillation: T > 1 membuat distribusi teacher lebih "soft"
  - Bisa plot accuracy vs model size Pareto frontier untuk berbagai distillation konfigurasi

**18. Prefix Tuning & LoRA** — Parameter-efficient fine-tuning.
- 🎯 **Goals:**
  - Memahami prefix tuning: tambahkan learned "virtual tokens" ke key dan value, bukan update semua param
  - Bisa implementasi LoRA: approximate weight update ΔW = BA, hanya B dan A yang dilatih
  - Mengerti mengapa LoRA efektif: weight update pada model pretrained cenderung low-rank
  - Bisa bandingkan full fine-tuning vs prefix tuning vs LoRA: params trained, accuracy, training speed
  - Memahami kapan PEFT lebih baik: resource terbatas, banyak task berbeda, catastrophic forgetting

**19. Constitutional AI Alignment (Toy)** — Implementasi RLHF pipeline sederhana.
- 🎯 **Goals:**
  - Memahami tiga tahap RLHF: SFT → reward model → PPO (RL) fine-tuning
  - Bisa implementasi reward model: fine-tune LM untuk predict human preference score
  - Mengerti PPO objective untuk language model: optimize reward sambil stay close ke reference policy
  - Memahami KL penalty: cegah model diverge terlalu jauh dari SFT model (reward hacking)
  - Bisa identifikasi reward hacking: model temukan cara maximize reward yang tidak sesuai intent

**20. Speculative Decoding Accelerator** — Implementasi speculative decoding.
- 🎯 **Goals:**
  - Memahami speculative decoding: draft model kecil propose K tokens, besar verifikasi sekaligus
  - Bisa implementasi rejection sampling scheme: accept/reject token draft berdasarkan probability ratio
  - Mengerti mengapa ini mempercepat inference: target model verifikasi K token dalam satu forward pass
  - Memahami trade-off: speedup bergantung pada acceptance rate dan perbedaan ukuran model
  - Bisa ukur speedup empiris dan analyze acceptance rate per domain

---

## 🔹 Generative Models — VAE

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. MNIST VAE** — Generate digit baru dari latent space 2D.
- 🎯 **Goals:**
  - Memahami encoder: memprediksi mean (μ) dan log-variance (log σ²), bukan single point
  - Bisa implementasi reparameterization trick: z = μ + σ * ε, ε ~ N(0,I) untuk enable backprop
  - Mengerti ELBO loss: reconstruction term + KL divergence term
  - Memahami KL divergence sebagai regularizer: mendorong posterior mendekati prior N(0,I)
  - Bisa visualisasikan 2D latent space sebagai scatter plot dengan warna per digit class
  - Mengerti mengapa cluster digit terpisah di latent space tanpa supervisi eksplisit

**2. Latent Space Interpolation** — Interpolasi antara dua gambar di latent space.
- 🎯 **Goals:**
  - Memahami bahwa latent space VAE continuous: bisa interpolasi antara dua encoding
  - Bisa implement spherical interpolation (slerp) vs linear interpolation dan perbedaannya
  - Mengerti mengapa interpolasi menghasilkan gambar yang bermakna (bukan noise): structured latent space
  - Memahami smoothness sebagai bukti disentanglement: transisi halus tanpa discontinuity

**3. Image Denoising VAE** — Rekonstruksi gambar bersih dari yang noisy.
- 🎯 **Goals:**
  - Memahami denoising VAE sebagai conditional generative model: P(x_clean | x_noisy)
  - Bisa bandingkan VAE reconstruction vs classical denoising: bilateral filter, BM3D, Gaussian blur
  - Mengerti PSNR dan SSIM sebagai objective metrics untuk image quality
  - Memahami bahwa reconstruction loss mendorong sharpness vs KL mendorong regularization

**4. Tabular Data Synthesis** — Generate synthetic tabular data menggunakan VAE.
- 🎯 **Goals:**
  - Memahami bagaimana encode dan decode mixed data types: continuous + categorical di satu VAE
  - Bisa evaluasi synthetic data quality: statistical fidelity (mean, std, correlation), utility (downstream ML)
  - Mengerti privacy concern: apakah synthetic data benar-benar private? membership inference attack
  - Memahami Gower distance untuk perbandingan distribusi data tabular yang memiliki mixed types

**5. Anomaly Detection MNIST** — Flag digit asing dari reconstruction error.
- 🎯 **Goals:**
  - Memahami mengapa anomaly memiliki reconstruction error tinggi: model belum melihat pola ini
  - Bisa definisikan threshold berdasarkan distribution dari reconstruction error pada normal data
  - Mengerti ELBO sebagai anomaly score: kombinasi reconstruction + KL
  - Bisa evaluasi dengan AUC-ROC: bagaimana membandingkan VAE anomaly detection vs isolation forest

---

#### 🟡 INTERMEDIATE

**6. Face Generation VAE** — Generate wajah baru dari dataset CelebA.
- 🎯 **Goals:**
  - Memahami bagaimana VAE seringkali menghasilkan gambar yang blur: MSE loss mendorong averaging
  - Bisa implementasi perceptual loss (VGG features) sebagai alternatif pixel-wise MSE
  - Mengerti atribut manipulation: encode dua gambar (dengan/tanpa senyum), hitung direction di latent space
  - Memahami disentanglement secara intuitif: dimensi yang ideally hanya kontrol satu faktor

**7. Conditional VAE (cVAE)** — Generate gambar dengan kondisi label kelas.
- 🎯 **Goals:**
  - Memahami cara inject label kondisi: concatenate ke input encoder dan decoder input
  - Bisa generate gambar untuk class tertentu: "generate digit 7 dengan style tertentu"
  - Mengerti posterior collapse: ketika decoder mengabaikan latent z dan hanya pakai kondisi label
  - Memahami auxiliary classifier sebagai cara enforce conditioning

**8. β-VAE Disentanglement Study** — Eksperimen beta dari 0.1 hingga 10.
- 🎯 **Goals:**
  - Memahami β-VAE: multiply KL term dengan β > 1 untuk stronger regularization → better disentanglement
  - Bisa ukur disentanglement secara kuantitatif: Mutual Information Gap (MIG), DCI score
  - Mengerti trade-off: beta tinggi → lebih disentangled tapi reconstruction lebih buruk
  - Bisa visualisasikan latent traversals: ubah satu dimensi, biarkan yang lain fixed, plot hasil

**9. VAE untuk Drug Discovery** — Generate molekul valid menggunakan SMILES string VAE.
- 🎯 **Goals:**
  - Memahami SMILES representasi: ASCII string encoding molekul sebagai path graph traversal
  - Bisa evaluasi validity (% SMILES yang valid), novelty (% tidak ada di training set), uniqueness
  - Mengerti Bayesian optimization di latent space untuk molecular property optimization
  - Memahami grammar VAE sebagai cara ensure grammatical validity dari generated SMILES

**10. Music VAE** — Generate melodi baru menggunakan hierarchical VAE.
- 🎯 **Goals:**
  - Memahami hierarchical VAE: high-level latent encode global structure, low-level encode details
  - Bisa implementasi bar-level encoding kemudian sequence-level encoder diatasnya
  - Mengerti musical evaluation: note density, pitch range, rhythmic diversity
  - Memahami mengapa flat VAE gagal untuk musik panjang: posterior collapse lebih sering terjadi

**11. Point Cloud VAE** — Generate 3D objek dari point cloud.
- 🎯 **Goals:**
  - Memahami point cloud sebagai unordered set: berbeda dari grid (voxel) atau mesh
  - Bisa implementasi PointNet encoder: symmetric function (max pooling) untuk permutation invariance
  - Mengerti Chamfer Distance dan Earth Mover's Distance sebagai reconstruction loss untuk point cloud
  - Memahami fold-based decoder: unfold 2D grid ke 3D surface menggunakan learned deformation

**12. Cross-modal VAE** — Latih shared latent space antara text dan gambar.
- 🎯 **Goals:**
  - Memahami product of experts: kombinasi posterior dari dua modalitas secara principled
  - Bisa implementasi unimodal inference: encode dari satu modalitas saja ke shared latent
  - Mengerti cross-modal generation: encode teks → sample z → decode gambar
  - Memahami alignment loss: mendorong representasi gambar dan teks yang matched mendekati satu sama lain

---

#### 🔴 ADVANCED

**13. VQ-VAE** — Implementasi discrete latent space.
- 🎯 **Goals:**
  - Memahami vector quantization: setiap latent vector di-snap ke nearest codebook entry
  - Bisa implementasi straight-through estimator untuk gradient melalui argmin yang non-differentiable
  - Mengerti commitment loss dan codebook loss: latih encoder dan codebook bersamaan
  - Memahami codebook collapse dan solusinya: random restart, exponential moving average update
  - Bisa ukur codebook utilization: berapa banyak dari K entries benar-benar digunakan

**14. Hierarchical VAE (NVAE/HVAE)** — VAE dengan multiple stochastic layers.
- 🎯 **Goals:**
  - Memahami bahwa stacked stochastic layers bisa model distribusi yang lebih ekspresif
  - Bisa implementasi top-down inference: prior dari level atas mempengaruhi posterior level bawah
  - Mengerti residual normal distribution: posterior = prior + residual untuk training stability
  - Memahami bagaimana hierarchical structure menghasilkan gambar yang lebih tajam dan coherent

**15. VAE + Flow-based Posterior** — Ganti Gaussian posterior dengan normalizing flow.
- 🎯 **Goals:**
  - Memahami mengapa Gaussian posterior terlalu restrictive untuk komplex posterior
  - Bisa implementasi Inverse Autoregressive Flow (IAF) sebagai posterior approximation
  - Mengerti ELBO dengan flow: log p(x) ≥ E[log p(x|z)] - KL(q_flow(z|x) || p(z))
  - Memahami bahwa tighter ELBO menghasilkan representasi yang lebih baik

**16. World Model (VAE + MDN-RNN)** — Implementasi world model à la David Ha.
- 🎯 **Goals:**
  - Memahami tiga komponen: V (VAE vision), M (MDN-RNN memory), C (controller)
  - Bisa implementasi MDN-RNN: predict distribusi Gaussian campuran atas next latent state
  - Mengerti training C dalam dream (simulasi di latent space, bukan environment nyata)
  - Memahami limitation: simulation gap antara dream dan real environment

**17. Information-theoretic Disentanglement** — Implementasi FactorVAE atau TCVAE.
- 🎯 **Goals:**
  - Memahami Total Correlation: mengukur statistical independence antar dimensi latent
  - Bisa implementasi FactorVAE: adversarial penalty untuk minimize total correlation
  - Mengerti TCVAE: decompose KL menjadi MI term + TC term + dimension-wise KL
  - Memahami mengapa TC minimization lebih directly connected ke disentanglement daripada beta-VAE

**18. Latent Diffusion Model (Simplified)** — Latih diffusion process di latent space.
- 🎯 **Goals:**
  - Memahami mengapa diffusion di latent space (bukan pixel): jauh lebih efisien secara komputasi
  - Bisa implementasi forward process (add noise) dan reverse process (denoise) di VAE latent space
  - Mengerti DDPM objective: predict noise ε dari noisy latent z_t
  - Memahami conditioning: how to inject class label atau text embedding ke diffusion process

**19. Out-of-Distribution Detection Benchmark** — Bandingkan VAE reconstruction vs alternatives.
- 🎯 **Goals:**
  - Memahami OOD detection sebagai density estimation problem: in-distribution memiliki high likelihood
  - Bisa implementasi multiple baselines: VAE ELBO, energy-based, flow-based, Mahalanobis distance
  - Mengerti mengapa high-dimensional OOD detection sulit: typical set phenomenon
  - Bisa evaluasi dengan AUROC pada multiple OOD datasets: near-OOD vs far-OOD

**20. VAE for Time Series Imputation** — Inputasi missing values dengan probabilistic VAE.
- 🎯 **Goals:**
  - Memahami missing data mechanisms: MCAR, MAR, MNAR dan implikasi untuk model
  - Bisa implementasi GPVAE: Gaussian process prior untuk smooth temporal latent trajectories
  - Mengerti uncertainty propagation: imputed values harus datang dengan credible interval
  - Memahami evaluation: MSE pada masked values, tapi juga downstream task performance setelah imputation

---

## 🔹 Generative Models — GAN

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. MNIST GAN** — Generate digit handwritten dengan vanilla GAN.
- 🎯 **Goals:**
  - Memahami minimax game: G minimizes, D maximizes log(D(x)) + log(1-D(G(z)))
  - Bisa implement training loop: alternate D update dan G update
  - Mengerti mode collapse secara visual: generated samples sangat mirip satu sama lain
  - Memahami training instability: G loss dan D loss harus roughly balanced
  - Bisa diagnose: D terlalu kuat (G tidak bisa belajar) vs D terlalu lemah (G tidak mendapat sinyal)

**2. 1D Distribution GAN** — Latih GAN untuk match distribusi 1D.
- 🎯 **Goals:**
  - Memahami GAN sebagai implicit density estimator: belajar distribusi tanpa explicitly modeling PDF
  - Bisa visualisasikan training progression: generated distribution vs target distribution per 100 steps
  - Mengerti mode collapse pada 1D: GAN capture satu mode dari bimodal distribution
  - Memahami Nash equilibrium concept secara konkret: titik di mana D tidak bisa membedakan G dan real

**3. Fashion-MNIST DCGAN** — DCGAN untuk generate pakaian.
- 🎯 **Goals:**
  - Memahami DCGAN guidelines: strided convolution untuk downsampling, transposed conv untuk upsample
  - Bisa implementasi BatchNorm di G dan LeakyReLU di D sebagai best practice
  - Mengerti checkerboard artifact: muncul karena transposed conv dengan stride, solusi: upsample + conv
  - Memahami latent vector sampling: standard normal vs uniform dan perbedaan coverage

**4. Conditional GAN (digit class)** — Generate digit tertentu berdasarkan class label.
- 🎯 **Goals:**
  - Memahami cara conditioning: concatenate one-hot embedding ke noise (G) dan ke gambar (D)
  - Bisa generate grid: setiap kolom satu kelas, setiap baris satu random z
  - Mengerti bahwa cGAN harus verify bahwa kondisi label dipatuhi, bukan hanya realism
  - Memahami auxiliary classifier GAN (ACGAN) sebagai alternatif conditioning approach

**5. GAN Failure Mode Study** — Sengaja trigger mode collapse dan instability.
- 🎯 **Goals:**
  - Bisa trigger mode collapse by design: learning rate D terlalu tinggi atau capacity D >> G
  - Mengerti vanishing gradient problem: ketika D terlalu kuat, G gradients mendekati zero
  - Bisa trigger checkerboard artifact dengan pixel shuffle / transposed conv yang tidak tepat
  - Memahami solusi praktis: label smoothing (0.9 bukan 1.0), noise ke D input, gradient penalty

---

#### 🟡 INTERMEDIATE

**6. CelebA Face Generation (DCGAN)** — Generate wajah realistis 64x64.
- 🎯 **Goals:**
  - Memahami hyperparameter sensitivity GAN: learning rate, beta1 Adam, batch size
  - Bisa implementasi truncation trick: sample dari |z| < threshold untuk higher quality at cost of diversity
  - Mengerti FID sebagai proxy untuk visual quality: Fréchet distance antara feature distributions
  - Memahami mengapa 64x64 lebih mudah dari 256x256: training stability decreases dengan resolution

**7. WGAN & WGAN-GP** — Implementasi Wasserstein GAN dengan gradient penalty.
- 🎯 **Goals:**
  - Memahami Earth Mover's Distance vs Jensen-Shannon divergence: mengapa EM lebih baik?
  - Bisa implementasi weight clipping (WGAN) dan gradient penalty (WGAN-GP)
  - Mengerti 1-Lipschitz constraint: mengapa diperlukan untuk WGAN dan bagaimana memaksanya
  - Memahami critic vs discriminator: WGAN critic output score (bukan probability), tidak ada sigmoid
  - Bisa bandingkan training curve stability: WGAN-GP jauh lebih stabil dari vanilla GAN

**8. Pix2Pix Image Translation** — Edge map → foto, siang → malam.
- 🎯 **Goals:**
  - Memahami paired image translation: input dan target adalah paired, conditional GAN
  - Bisa implementasi U-Net generator dan PatchGAN discriminator (classify patches, bukan full image)
  - Mengerti PatchGAN: lebih efisien dan fokus pada local texture realism
  - Memahami L1 loss + adversarial loss combination: L1 ensures global structure, GAN adds texture
  - Bisa evaluasi dengan FID dan user study: manakah yang lebih fotorealistik?

**9. Super Resolution GAN (SRGAN)** — Upscale gambar 4x.
- 🎯 **Goals:**
  - Memahami perceptual loss: mengukur similarity di feature space VGG, bukan pixel space
  - Bisa bandingkan: MSE loss only (blurry) vs perceptual loss (sharp tapi bisa artifact) vs SRGAN (realistic)
  - Mengerti sub-pixel convolution (pixel shuffle) untuk efficient learned upsampling
  - Memahami SSIM dan LPIPS sebagai evaluation metrics yang lebih aligned dengan perceptual quality

**10. Text-to-Image (Simple)** — GAN yang generate gambar dari deskripsi teks.
- 🎯 **Goals:**
  - Memahami conditioning GAN dengan embedding: sentence embedding sebagai kondisi untuk G dan D
  - Bisa implementasi mismatched example loss: D juga harus reject gambar yang real tapi text-nya salah
  - Mengerti semantic alignment challenge: sama text tapi gambar beda (diversity vs alignment trade-off)
  - Memahami bahwa ini adalah precursor ke DALL-E/Stable Diffusion

**11. GAN Evaluation Suite** — Implementasi IS & FID dari scratch.
- 🎯 **Goals:**
  - Bisa implementasi Inception Score: IS = exp(E[KL(p(y|x) || p(y))]) — tinggi berarti diverse dan sharp
  - Mengerti FID: measure Fréchet distance antara Inception features dari real dan generated samples
  - Bisa explain limitation IS: tidak compare to real data, bisa di-game
  - Memahami Precision dan Recall untuk generative models: quality vs diversity yang terpisah
  - Mengerti mengapa FID lebih reliable dari IS tapi masih memiliki blind spots

**12. InfoGAN** — Latih GAN dengan interpretable latent code.
- 🎯 **Goals:**
  - Memahami mutual information maximization: latent code c dan output G(z,c) harus highly dependent
  - Bisa implementasi auxiliary network Q(c|x) yang prediksi latent code dari generated image
  - Mengerti variational lower bound untuk mutual information (mengapa MI langsung tidak bisa dioptimize)
  - Bisa visualisasikan disentanglement: ubah satu dimension c, lihat apa yang berubah di gambar

---

#### 🔴 ADVANCED

**13. CycleGAN** — Unpaired image translation: kuda → zebra.
- 🎯 **Goals:**
  - Memahami cycle consistency loss: F(G(x)) ≈ x dan G(F(y)) ≈ y untuk constrain translation
  - Bisa implementasi identity loss: G(y) ≈ y untuk preserve color composition
  - Mengerti mengapa unpaired training works: cycle consistency sebagai proxy supervision
  - Memahami limitation: CycleGAN bisa "cheat" dengan hiding information (steganography)
  - Bisa bandingkan CycleGAN vs Pix2Pix pada task yang sama dengan paired data

**14. StyleGAN2 (Simplified)** — Implementasi key innovations.
- 🎯 **Goals:**
  - Memahami mapping network: z → w untuk disentangle latent space dari generator architecture
  - Bisa implementasi AdaIN (Adaptive Instance Normalization): inject style ke setiap layer
  - Mengerti progressive growing: train pada resolusi rendah dulu, tambah layer secara bertahap
  - Memahami weight demodulation: alternatif AdaIN yang lebih stabil (fix blob artifacts)
  - Bisa generate images dengan style mixing: mix styles dari dua w vectors di berbagai layer

**15. Self-Attention GAN (SAGAN)** — Tambahkan self-attention layer ke GAN.
- 🎯 **Goals:**
  - Memahami mengapa convolutional GAN gagal capture long-range dependencies: receptive field terbatas
  - Bisa implementasi self-attention dalam convolutional network: query/key/value dari feature map
  - Mengerti spectral normalization untuk D: stabilize training dan enforce Lipschitz constraint
  - Memahami attention map visualization: apa yang model "attend" ketika generate setiap part gambar

**16. BigGAN (Class-conditional Large-scale)** — Class-conditional GAN pada dataset besar.
- 🎯 **Goals:**
  - Memahami class conditioning dengan class embedding: inject ke setiap BatchNorm layer (cBN)
  - Bisa implementasi truncation trick: sample dari truncated normal untuk quality vs diversity control
  - Mengerti orthogonal regularization: encourage weight matrices yang orthogonal untuk training stability
  - Memahami scaling challenges: mengapa large-batch training memerlukan careful learning rate warmup

**17. GAN Inversion** — Encode gambar nyata ke latent space StyleGAN.
- 🎯 **Goals:**
  - Memahami optimization-based inversion: minimize reconstruction loss over z per gambar (slow tapi akurat)
  - Bisa implementasi encoder-based inversion: train E(x) → w untuk fast inference
  - Mengerti editability vs reconstruction tradeoff: W space vs W+ space vs extended latent spaces
  - Memahami semantic attribute editing: find attribute direction di latent space menggunakan binary classifier

**18. Data Augmentation GAN Pipeline** — Gunakan GAN untuk augmentasi dataset medis.
- 🎯 **Goals:**
  - Bisa implementasi controlled experiment: classifier trained on real-only vs real+synthetic
  - Mengerti FID tidak menjamin downstream improvement: synthetic quality ≠ useful diversity
  - Memahami training data leakage risk: pastikan synthetic data dari training set GAN tidak "contaminate" test
  - Bisa compare dengan classical augmentation: synthetic yang costly vs simple augmentation

**19. Generative Data Poisoning Defense** — Latih detector GAN-generated vs real.
- 🎯 **Goals:**
  - Memahami deepfake detection sebagai binary classification problem
  - Bisa implementasi frequency-domain features: GAN artifacts seringkali terlihat di spectrum
  - Mengerti arms race: detector di-fool oleh adversarial perturbation, GAN improve dari feedback
  - Memahami evaluasi yang proper: bukan hanya accuracy, tapi juga cross-generator generalization

**20. Diffusion vs GAN Comparison** — Training stability, FID, diversity, speed.
- 🎯 **Goals:**
  - Bisa implementasi controlled comparison pada dataset yang sama (CIFAR-10, CelebA)
  - Memahami mengapa diffusion model menghasilkan FID lebih baik: tidak ada training instability
  - Mengerti sampling speed gap: GAN = satu forward pass, Diffusion = 50-1000 forward passes
  - Bisa quantify diversity gap: Recall metric dan intra-class diversity comparison
  - Memahami kapan GAN masih disukai: real-time application, fine-grained control, faster inference

---

## 🔹 Reinforcement Learning Basics

### 📂 Projects (Easy → Advanced)

#### 🟢 EASY

**1. Multi-Armed Bandit** — Eksplorasi ε-greedy, UCB, Thompson Sampling.
- 🎯 **Goals:**
  - Memahami exploration-exploitation dilemma: kapan mencoba arm baru vs stick dengan yang terbaik
  - Bisa implementasi ε-greedy dan ukur cumulative regret per timestep
  - Mengerti UCB: confidence bound memastikan arm yang jarang dicoba selalu ada kesempatan dicoba
  - Memahami Thompson Sampling: Bayesian approach, sample dari posterior belief per arm
  - Bisa plot regret curves semua algorithm dan explain trade-off teoritis vs empiris

**2. FrozenLake Policy Iteration** — Solve grid world dengan dynamic programming.
- 🎯 **Goals:**
  - Memahami MDP formalism: S, A, R, P(s'|s,a), γ secara konkret di grid world
  - Bisa implementasi policy evaluation: solve sistem persamaan linear Bellman untuk V^π
  - Mengerti policy improvement: untuk setiap state, pilih action yang maximize expected return
  - Memahami convergence: kenapa policy iteration selalu converge ke optimal policy dalam finite MDP

**3. CartPole Q-Learning (Discretized)** — Tabular Q-Learning pada state yang di-discretize.
- 🎯 **Goals:**
  - Memahami Q-Learning update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
  - Bisa implementasi discretization: binning continuous state variables
  - Mengerti epsilon-decay schedule: explore aggressively di awal, exploit lebih di akhir
  - Memahami limitasi Q-table: state space besar → table terlalu besar → need function approximation (DQN)

**4. Grid World dengan Value Iteration** — Solve custom grid world.
- 🎯 **Goals:**
  - Memahami Bellman optimality equation dan Value Iteration sebagai iterative solver
  - Bisa design reward function untuk different behaviors: reward shaping effects
  - Mengerti discount factor γ: γ mendekati 0 = myopic agent, γ mendekati 1 = far-sighted
  - Memahami bahwa value iteration memerlukan full model (transition dan reward): model-based RL

**5. Tic-Tac-Toe Self-Play** — Latih Q-agent melalui self-play.
- 🎯 **Goals:**
  - Memahami self-play sebagai cara generate unlimited training data tanpa human opponent
  - Bisa visualisasikan Q-table convergence: nilai Q berubah seiring training
  - Mengerti sparse reward challenge: reward hanya di akhir game (win/lose/draw)
  - Memahami symmetry exploitation: rotate/flip board untuk data augmentation

---

#### 🟡 INTERMEDIATE

**6. CartPole DQN** — Implementasi Deep Q-Network penuh.
- 🎯 **Goals:**
  - Memahami mengapa neural network sebagai Q-function approximator bermasalah tanpa tricks
  - Bisa implementasi experience replay: uniformly sample minibatch dari replay buffer
  - Mengerti target network: frozen copy dari Q-network untuk stable target computation
  - Memahami Double DQN motivation: overestimation bias dari standard DQN
  - Bisa plot training curve dan diagnose: apakah Q-values diverge? apakah reward plateau?

**7. Atari Pong DQN** — Latih DQN bermain Pong dari pixel.
- 🎯 **Goals:**
  - Memahami preprocessing pipeline: grayscale, resize, frame stacking (temporal information)
  - Bisa implementasi reward clipping: clip ke {-1, 0, +1} untuk training stability
  - Mengerti episode reward vs game score: episode bisa terdiri dari banyak game point
  - Memahami mengapa DQN dari raw pixel adalah milestone: end-to-end RL without feature engineering

**8. LunarLander Double DQN** — Double DQN + Dueling architecture.
- 🎯 **Goals:**
  - Memahami Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A) untuk better value estimation
  - Bisa implementasi Double DQN: online network pilih action, target network evaluate
  - Mengerti Prioritized Experience Replay: sample transitions dengan high TD error lebih sering
  - Memahami Rainbow DQN sebagai combination dari semua improvements
  - Bisa landingkan rocket dengan reward function yang sudah ada, lalu analyze what agent learned

**9. Traffic Signal Control** — RL agent kontrol lampu hijau untuk minimize queue.
- 🎯 **Goals:**
  - Memahami custom environment design: definisi state space, action space, dan reward function
  - Bisa implement reward shaping: intermediate rewards untuk guide learning pada sparse reward
  - Mengerti multi-objective reward: trade-off antara throughput, fairness, waiting time
  - Memahami evaluation: cumulative vehicle waiting time sebagai primary metric
  - Bisa bandingkan RL agent dengan fixed-time controller dan actuated controller

**10. Portfolio Optimization RL** — Agent alokasikan aset.
- 🎯 **Goals:**
  - Memahami continuous action space: alokasi persentase portfolio sebagai action vector
  - Bisa design reward function yang proper: risk-adjusted return (Sharpe ratio bukan raw return)
  - Mengerti transaction costs sebagai penalty: frequent rebalancing memiliki real cost
  - Memahami look-ahead bias risk: menggunakan informasi masa depan saat mendefinisikan state
  - Bisa bandingkan RL vs Markowitz portfolio optimization dalam out-of-sample testing

**11. MountainCar dengan Reward Shaping** — Solve MountainCar dengan sparse reward.
- 🎯 **Goals:**
  - Memahami sparse reward problem: agent hanya dapat reward ketika mencapai goal (sangat jarang)
  - Bisa implementasi potential-based reward shaping: tambah φ(s') - φ(s) tanpa mengubah optimal policy
  - Mengerti curriculum learning: train dulu dari starting position dekat goal, lalu perlahan jauhkan
  - Memahami count-based exploration: visit count sebagai bonus reward untuk mendorong exploration
  - Bisa compare: no shaping (stuck) vs potential-based (solve dalam lebih sedikit steps)

**12. Taxi Driver (Multi-objective)** — Modifikasi Taxi-v3 dengan multiple objectives.
- 🎯 **Goals:**
  - Memahami multi-objective RL: tidak ada satu reward function yang optimal
  - Bisa implementasi Pareto front visualization: trade-off antara speed, fuel, comfort
  - Mengerti scalarization: linear combination dari objectives (tapi ini hanya finds convex Pareto points)
  - Memahami constrained MDP: maximize satu objective subject to constraint pada yang lain

---

#### 🔴 ADVANCED

**13. Atari Breakout DQN Full** — Rainbow DQN, ukur sample efficiency.
- 🎯 **Goals:**
  - Bisa implementasi 6 Rainbow components: Double DQN + Dueling + PER + n-step returns + Noisy Net + Distributional RL
  - Memahami distributional RL (C51/QR-DQN): predict full return distribution, bukan hanya mean
  - Mengerti n-step returns: trade off bias (farther target) vs variance (more signal)
  - Bisa ukur sample efficiency: total environment steps untuk mencapai threshold reward
  - Memahami Noisy Nets: parametric noise sebagai replacement ε-greedy untuk exploration

**14. Policy Gradient (REINFORCE)** — Implementasi vanilla PG dari scratch.
- 🎯 **Goals:**
  - Memahami policy gradient theorem: ∇J(θ) = E[∇log π(a|s) * G_t]
  - Bisa implementasi REINFORCE: accumulate log probabilities, multiply by return, update
  - Mengerti high variance problem: return G_t bisa bervariasi sangat besar antar episode
  - Bisa implementasi baseline: kurangi state value baseline untuk reduce variance tanpa bias
  - Memahami perbedaan fundamental value-based vs policy-based: yang mana optimal untuk apa?

**15. Proximal Policy Optimization (PPO)** — Implementasi PPO dari scratch.
- 🎯 **Goals:**
  - Memahami clipped objective: min(r_t(θ) A_t, clip(r_t, 1-ε, 1+ε) A_t) mencegah policy update terlalu besar
  - Bisa implementasi GAE (Generalized Advantage Estimation): bias-variance trade-off untuk advantage
  - Mengerti entropy bonus: dorong exploration dengan tambahkan H(π) ke objective
  - Memahami mengapa PPO lebih stabil dari TRPO: simpler constraint (clip) vs KL constraint
  - Bisa train MuJoCo HalfCheetah dan reach competitive reward dalam wall-clock time tertentu

**16. Robot Arm Control (PyBullet)** — Kontrol 3-DoF arm untuk reach target.
- 🎯 **Goals:**
  - Memahami continuous control dengan deterministic policy (DDPG) atau stochastic (SAC)
  - Bisa implementasi Hindsight Experience Replay (HER): replay dengan achieved goal sebagai target
  - Mengerti mengapa HER sangat efektif untuk sparse reward robotics: setiap episode selalu ada "success"
  - Memahami sim-to-real gap: domain randomization sebagai cara membuat policy transfer ke real robot
  - Bisa ukur success rate sebagai primary metric (bukan return) untuk reach task

**17. Multi-Agent RL (Cooperative)** — QMIX atau MAPPO untuk cooperative task.
- 🎯 **Goals:**
  - Memahami centralized training with decentralized execution (CTDE) paradigm
  - Bisa implementasi QMIX: joint Q function yang monotonically factorizes per-agent Q functions
  - Mengerti credit assignment problem dalam multi-agent: siapa yang berkontribusi pada reward bersama?
  - Memahami emergent communication: apakah agents develop meaningful communication protocol?
  - Bisa evaluate team performance vs individual performance untuk cooperative scenarios

**18. Model-Based RL (Dyna-Q)** — Pelajari model lingkungan secara explicit.
- 🎯 **Goals:**
  - Memahami Dyna architecture: real experience update Q + n simulated updates dari learned model
  - Bisa implementasi model learning: tabular atau neural network untuk predict (s', r) dari (s, a)
  - Mengerti planning n: berapa banyak model rollout memberikan best data efficiency?
  - Memahami model error propagation: compound error dalam multi-step rollout dari imperfect model
  - Bisa compare: model-free DQN vs Dyna-Q dalam terms of sample efficiency dan wall-clock time

**19. AlphaZero (Simplified)** — MCTS + neural network untuk Connect4.
- 🎯 **Goals:**
  - Memahami MCTS: selection (UCB tree policy) → expansion → simulation → backpropagation
  - Bisa implementasi PUCT formula: UCB yang menggunakan prior policy dari neural network
  - Mengerti joint training: MCTS improved policy digunakan untuk train policy network, saling iterate
  - Memahami self-play data generation: setiap game generate training examples (state, π, z)
  - Bisa evaluate Elo rating progression seiring training: bukti bahwa agent improve dari self-play

**20. Safe RL dengan Constraint** — CPO atau Lagrangian method.
- 🎯 **Goals:**
  - Memahami Constrained MDP (CMDP): maximize reward subject to cost constraint C ≤ d
  - Bisa implementasi Lagrangian relaxation: dual variable λ mengontrol trade-off reward vs safety
  - Mengerti CPO: policy update yang guaranteed tidak melanggar constraint (projection ke safe policy set)
  - Memahami safety vs performance trade-off: safe policy biasanya lebih konservatif dan lower reward
  - Bisa evaluate pada Safety Gym atau custom environment: track cumulative constraint violations

---

## 📊 Ringkasan Progress Tracker

| Algoritma | Easy (5) | Intermediate (8) | Advanced (7) | Total |
|-----------|----------|-----------------|--------------|-------|
| CNN | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| RNN | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| LSTM/GRU | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| Transformer | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| VAE | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| GAN | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| RL | ☐☐☐☐☐ | ☐☐☐☐☐☐☐☐ | ☐☐☐☐☐☐☐ | 20 |
| **TOTAL** | | | | **140** |

---

> 🔑 **Tips Belajar**: Selesaikan semua EASY terlebih dahulu sebelum lompat ke INTERMEDIATE. Setiap project wajib disertai **write-up singkat** (apa yang berhasil, apa yang gagal, insight baru). Portfolio > sekedar checklist. Goals tiap project bukan hanya untuk diceklis — jadikan sebagai **self-quiz**: bisa kamu jelaskan konsep ini kepada orang lain tanpa melihat notes?
