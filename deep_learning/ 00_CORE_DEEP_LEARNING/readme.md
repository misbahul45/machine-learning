# 📘 00_CORE_DEEP_LEARNING

## 🏗️ Topik yang Dicakup:
- CNN (Convolutional Neural Networks)
- RNN (Recurrent Neural Networks)
- LSTM/GRU
- Transformer Architecture
- Generative Models (VAE, GAN)
- Reinforcement Learning Basics

---

### 🔹 CNN (Convolutional Neural Networks)

**5 Ide Project:**
* project → Image Classification (CIFAR-10/100)
* project → Object Detection (YOLO-style detector)
* project → Image Segmentation (U-Net untuk medical images)
* project → Style Transfer (Neural Style Transfer)
* project → Face Recognition System

**🎯 Target Pemahaman:**
* ✅ Paham kenapa convolution > fully connected (local connectivity, parameter sharing)
* ✅ Bisa jelaskan filter/kernel operation & feature map
* ✅ Mengerti pooling (max vs average) untuk spatial reduction
* ✅ Tahu receptive field concept & deep network advantage
* ✅ Paham padding (valid vs same), stride effect ke output size
* ✅ Bisa visualisasikan learned filters per layer
* ✅ Mengerti CNN architecture evolution (LeNet → AlexNet → VGG → ResNet → EfficientNet)
* ✅ Tahu kapan pakai 1x1 convolution (channel reduction, add non-linearity)
* ✅ Paham batch normalization placement & effect

---

### 🔹 RNN (Recurrent Neural Networks)

**5 Ide Project:**
* project → Text Generation (Character-level RNN)
* project → Sentiment Analysis dengan Sequential Processing
* project → Stock Price Prediction
* project → Music Generation
* project → Language Translation (Seq2Seq)

**🎯 Target Pemahaman:**
* ✅ Paham sequential data processing & hidden state concept
* ✅ Bisa jelaskan parameter sharing across time steps
* ✅ Mengerti vanishing/exploding gradient problem di RNN
* ✅ Tahu backpropagation through time (BPTT)
* ✅ Paham truncated BPTT untuk long sequences
* ✅ Bisa implementasi dari scratch (forward & backward pass)
* ✅ Mengerti many-to-one, one-to-many, many-to-many architectures
* ✅ Tahu limitasi RNN (short-term memory, slow training)

---

### 🔹 LSTM/GRU

**5 Ide Project:**
* project → Named Entity Recognition (NER)
* project → Time Series Forecasting (multivariate)
* project → Speech Recognition
* project → Video Captioning
* project → Anomaly Detection in Sequences

**🎯 Target Pemahaman:**
* ✅ Paham gating mechanism (forget, input, output gates di LSTM)
* ✅ Bisa jelaskan cell state vs hidden state
* ✅ Mengerti kenapa LSTM solve vanishing gradient (gradient highway)
* ✅ Tahu perbedaan LSTM vs GRU (gates, parameters, speed)
* ✅ Paham bidirectional LSTM untuk context dari both directions
* ✅ Bisa visualisasikan gate activations
* ✅ Mengerti peephole connections (optional LSTM variant)
* ✅ Tahu kapan pakai LSTM vs GRU (LSTM → complex, GRU → faster)

---

### 🔹 Transformer Architecture

**5 Ide Project:**
* project → Machine Translation (English-Indonesian)
* project → Text Summarization
* project → Question Answering System
* project → Code Generation
* project → Document Classification

**🎯 Target Pemahaman:**
* ✅ Paham self-attention mechanism (Query, Key, Value)
* ✅ Bisa jelaskan multi-head attention & kenapa penting
* ✅ Mengerti positional encoding (sinusoidal vs learned)
* ✅ Tahu encoder-decoder architecture
* ✅ Paham masked attention di decoder (autoregressive generation)
* ✅ Bisa hitung complexity: O(n²) vs RNN O(n)
* ✅ Mengerti layer normalization & residual connections
* ✅ Tahu feed-forward network role
* ✅ Paham kenapa parallelizable (vs sequential RNN)

---

### 🔹 Generative Models - VAE (Variational Autoencoder)

**5 Ide Project:**
* project → Image Generation (Fashion/Faces)
* project → Anomaly Detection
* project → Data Augmentation
* project → Latent Space Interpolation
* project → Image Denoising

**🎯 Target Pemahaman:**
* ✅ Paham encoder-decoder architecture
* ✅ Bisa jelaskan latent space & probabilistic encoding
* ✅ Mengerti reparameterization trick (backprop through sampling)
* ✅ Tahu loss function: reconstruction + KL divergence
* ✅ Paham KL divergence role (regularization ke standard normal)
* ✅ Bisa sample dari latent space untuk generation
* ✅ Mengerti disentangled representations
* ✅ Tahu β-VAE untuk better disentanglement

---

### 🔹 Generative Models - GAN (Generative Adversarial Networks)

**5 Ide Project:**
* project → Face Generation (DCGAN)
* project → Image-to-Image Translation (Pix2Pix)
* project → Super Resolution (SRGAN)
* project → Style Transfer (CycleGAN)
* project → Data Augmentation

**🎯 Target Pemahaman:**
* ✅ Paham adversarial training (generator vs discriminator)
* ✅ Bisa jelaskan minimax game & Nash equilibrium
* ✅ Mengerti mode collapse problem & solusinya
* ✅ Tahu loss functions: vanilla GAN, WGAN, LSGAN
* ✅ Paham training instability & tips (label smoothing, feature matching)
* ✅ Bisa implementasi progressive growing
* ✅ Mengerti conditional GAN (cGAN)
* ✅ Tahu evaluation metrics (IS, FID)

---

### 🔹 Reinforcement Learning Basics

**5 Ide Project:**
* project → Game AI (CartPole, Atari)
* project → Grid World Navigation
* project → Traffic Light Control
* project → Portfolio Optimization
* project → Robot Arm Control (simulation)

**🎯 Target Pemahaman:**
* ✅ Paham MDP (Markov Decision Process): state, action, reward, policy
* ✅ Bisa jelaskan value function & Q-function
* ✅ Mengerti exploration vs exploitation trade-off (ε-greedy)
* ✅ Tahu Q-Learning & DQN (Deep Q-Network)
* ✅ Paham experience replay & target network
* ✅ Bisa implementasi policy gradient methods
* ✅ Mengerti actor-critic architecture
* ✅ Tahu reward shaping & credit assignment problem

---
