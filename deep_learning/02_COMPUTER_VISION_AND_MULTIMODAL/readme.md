

# 📙 02_COMPUTER_VISION_AND_MULTIMODAL

## 🏗️ Topik yang Dicakup:
- Advanced CV (Object Detection, Segmentation)
- 3D Vision
- Neural Rendering
- Domain Adaptation
- Self-Supervised Learning
- Vision Transformers

---

### 🔹 Object Detection

**5 Ide Project:**
* project → Custom YOLO Detector (traffic signs)
* project → Faster R-CNN Implementation
* project → Real-time Person Detection
* project → Small Object Detection (drone imagery)
* project → Multi-Class Detection System

**🎯 Target Pemahaman:**
* ✅ Paham two-stage (R-CNN family) vs one-stage (YOLO, SSD)
* ✅ Bisa jelaskan anchor boxes & IoU (Intersection over Union)
* ✅ Mengerti Non-Maximum Suppression (NMS)
* ✅ Tahu feature pyramid networks (FPN)
* ✅ Paham loss function: classification + localization
* ✅ Bisa handle class imbalance (focal loss)
* ✅ Mengerti evaluation metrics (mAP, AP50, AP75)
* ✅ Tahu trade-off accuracy vs speed (EfficientDet)

---

### 🔹 Image Segmentation

**5 Ide Project:**
* project → Semantic Segmentation (Cityscapes)
* project → Instance Segmentation (Mask R-CNN)
* project → Medical Image Segmentation (U-Net)
* project → Panoptic Segmentation
* project → Interactive Segmentation (SAM-style)

**🎯 Target Pemahaman:**
* ✅ Paham semantic vs instance vs panoptic segmentation
* ✅ Bisa jelaskan U-Net architecture (skip connections)
* ✅ Mengerti encoder-decoder structure
* ✅ Tahu atrous/dilated convolution untuk receptive field
* ✅ Paham loss functions (Dice, IoU, cross-entropy)
* ✅ Bisa handle class imbalance di pixel level
* ✅ Mengerti post-processing (CRF)
* ✅ Tahu evaluation metrics (IoU, Dice coefficient)

---

### 🔹 3D Vision

**5 Ide Project:**
* project → Depth Estimation dari Single Image
* project → 3D Object Reconstruction
* project → Point Cloud Processing
* project → Stereo Matching
* project → SLAM (Simultaneous Localization and Mapping)

**🎯 Target Pemahaman:**
* ✅ Paham monocular vs stereo depth estimation
* ✅ Bisa jelaskan epipolar geometry
* ✅ Mengerti point cloud representations (PointNet)
* ✅ Tahu voxel-based vs mesh-based 3D
* ✅ Paham camera intrinsic & extrinsic parameters
* ✅ Bisa implement structure from motion (SfM)
* ✅ Mengerti multi-view geometry
* ✅ Tahu NeRF (Neural Radiance Fields) basics

---

### 🔹 Neural Rendering

**5 Ide Project:**
* project → NeRF Implementation (novel view synthesis)
* project → 3D Gaussian Splatting
* project → Neural Style Transfer (advanced)
* project → Image Inpainting
* project → Super Resolution Network

**🎯 Target Pemahaman:**
* ✅ Paham implicit neural representations
* ✅ Bisa jelaskan volumetric rendering
* ✅ Mengerti positional encoding untuk high-freq details
* ✅ Tahu ray marching & sampling strategies
* ✅ Paham differentiable rendering
* ✅ Bisa optimize NeRF training (hashgrid encoding)
* ✅ Mengerti 3D Gaussian Splatting vs NeRF
* ✅ Tahu applications (VR, AR, digital twins)

---

### 🔹 Domain Adaptation

**5 Ide Project:**
* project → Sim-to-Real Transfer
* project → Style Transfer untuk Domain Shift
* project → Unsupervised Domain Adaptation
* project → Few-Shot Domain Adaptation
* project → Cross-Domain Object Detection

**🎯 Target Pemahaman:**
* ✅ Paham domain shift problem (distribution mismatch)
* ✅ Bisa jelaskan adversarial domain adaptation
* ✅ Mengerti self-training & pseudo-labeling
* ✅ Tahu CycleGAN untuk unpaired translation
* ✅ Paham domain confusion loss
* ✅ Bisa implement feature alignment
* ✅ Mengerti source vs target domain
* ✅ Tahu evaluation (target domain accuracy)

---

### 🔹 Self-Supervised Learning (SSL)

**5 Ide Project:**
* project → Contrastive Learning (SimCLR, MoCo)
* project → Masked Autoencoder (MAE)
* project → BYOL Implementation
* project → Self-Supervised Pre-training
* project → SSL Evaluation Benchmark

**🎯 Target Pemahaman:**
* ✅ Paham pretext tasks (rotation, jigsaw, colorization)
* ✅ Bisa jelaskan contrastive learning (positive vs negative pairs)
* ✅ Mengerti momentum encoder (MoCo)
* ✅ Tahu masked image modeling (MAE, BEiT)
* ✅ Paham BYOL (no negative pairs)
* ✅ Bisa design augmentations untuk SSL
* ✅ Mengerti linear probing evaluation
* ✅ Tahu SSL vs supervised pre-training trade-off

---

### 🔹 Vision Transformers (ViT)

**5 Ide Project:**
* project → ViT from Scratch (ImageNet)
* project → Swin Transformer Implementation
* project → DeiT (Data-efficient ViT)
* project → Vision-Language Pre-training
* project → ViT vs CNN Comparison

**🎯 Target Pemahaman:**
* ✅ Paham patch embedding & tokenization
* ✅ Bisa jelaskan positional encoding untuk images
* ✅ Mengerti self-attention untuk vision (quadratic complexity)
* ✅ Tahu hierarchical vision transformers (Swin)
* ✅ Paham data efficiency problem (ViT needs more data)
* ✅ Bisa implement distillation (DeiT)
* ✅ Mengerti hybrid architectures (CNN + Transformer)
* ✅ Tahu inductive bias: CNN (locality) vs ViT (global)

---

## 📄 README.md Structure untuk 02_COMPUTER_VISION_AND_MULTIMODAL

```markdown
# 👁️ Computer Vision & Multimodal Portfolio

## 📋 Overview
Advanced CV topics: dari detection sampai neural rendering.
Fokus: **state-of-the-art implementations + production deployment**.

---

## 🗂️ Project Showcase

### 1. Object Detection Suite
- **Traffic Sign Detector**: YOLOv8 custom dataset
  - *Metric*: mAP@50 = 89%, real-time (30 FPS)
  - *Challenge*: Small object detection in varying lighting

### 2. Segmentation Projects
- **Medical Segmentation**: Tumor detection (U-Net++)
  - *Metric*: Dice = 0.92 on test set
  - *Challenge*: Class imbalance (95% background)

### 3. 3D Vision
- **Depth Estimation**: Monocular depth (MiDaS)
  - *Insight*: Zero-shot generalization impressive
  - *Challenge*: Scale ambiguity

### 4. Neural Rendering
- **NeRF Implementation**: 100 views → novel views
  - *Training*: 4 hours on single GPU
  - *Challenge*: Unbounded scenes

### 5. SSL Experiments
- **SimCLR Pre-training**: ImageNet-100 subset
  - *Result*: 92% linear probe accuracy
  - *Finding*: Strong augmentation = key

### 6. Vision Transformer
- **ViT-B/16**: Trained from scratch
  - *Insight*: Needs 10x data vs ResNet
  - *Challenge*: Compute requirements
