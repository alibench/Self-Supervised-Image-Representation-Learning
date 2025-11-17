# Self-Supervised Image Representation Learning

This project was completed as part of **CS-461: Foundations Models and Generative AI (EPFL)**.  
It explores **self-supervised learning (SSL)** for image representation using a contrastive objective and evaluates **out-of-distribution (OOD) generalization**.

---

## 🧠 Project Overview

The goal of this project is to train an **image encoder** from scratch using self-supervised methods on a subset of **ImageNet-1k**, without using labels.  
The learned representations are then evaluated using **k-NN** and **linear probes** on both in-distribution and OOD datasets to measure generalization.

Key steps:
- Implemented an **encoder** and **projection head** for self-supervised learning  
- Designed a **custom contrastive loss** for representation alignment  
- Trained on 200 ImageNet classes (train + validation splits)  
- Evaluated transfer and OOD performance  
- Visualized embeddings with **t-SNE** and **UMAP**

---

## ⚙️ Methodology

1. **Model Architecture**
   - Encoder: CNN-based (ResNet-style backbone)
   - Projection head: MLP with normalization and dropout
2. **Training Objective**
   - Contrastive loss on augmented image pairs
3. **Evaluation**
   - In-distribution (200 ImageNet classes used for training)
   - Out-of-distribution (200 unseen classes)
   - k-NN and linear probes on frozen embeddings
4. **Visualization**
   - t-SNE and UMAP plots for learned feature space

---

## 📦 Data Setup

The datasets used for this project are derived from **ImageNet-1k**, but due to size constraints, they are **not included in this repository**.

To reproduce the experiments:

1. Request access to ImageNet-1k from the official site:  
   🔗 https://www.image-net.org/

2. Once downloaded, prepare the following subsets:
   - `train/`: 200 selected ImageNet classes (≈100k images)  
   - `val/`: 200 classes, 50 images per class (≈10k images)  
   - `ood/`: 200 **unseen** ImageNet classes, 50 images each (≈10k images)

3. Place the datasets as follows:

cs461_assignment1_data/
├── ood.npz
└── train.npz
└── test.npz
