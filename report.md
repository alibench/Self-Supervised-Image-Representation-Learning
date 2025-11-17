# Report

> CS-461 Foundation Models and Generative AI

# Self-Supervised Learning and OOD Generalization

### Approach Overview

This project extends the SimCLR-style framework explored in Exercise Sessions 1–2. Our goal was to design a model trained from scratch on the 200-class ImageNet subset that learns invariant visual representations generalizing to unseen (OOD) classes. Rather than reusing the baseline pipeline, we systematically re-evaluated each design choice from data augmentation to projection head structure to improve feature quality and OOD transfer.

### Model Design

We used a **ResNet-18 style encoder (that we encoded from scratch as asked for)** instead of the smaller convolutional network used in exercise session 1 because it provided richer hierarchical features while remaining computationally feasible (we then also tried ResNet-34 and ResNet-50 style but the results and computational time showed that it wasn't worth it). We added a **two-layer projection head** (`Linear → ReLU → Linear → BatchNorm`) producing 128-D embeddings. Batch normalization was deliberately placed **after** the second linear layer to stabilize contrastive similarity scores — in early tests without it, representation norms drifted, hurting contrastive alignment.

We compared projection heads of size 64 and 128; the latter yielded smoother loss curves and stronger linear probe accuracy. We also normalized feature vectors before computing similarities, which improved convergence stability and made the loss temperature less sensitive.

### Loss and Training

While the exercise notebooks relied on the standard SimCLR NT-Xent loss, we re-implemented it with **explicit temperature control (τ = 0.5)** and experimented with **cosine-margin** and **triplet losses**. Both alternatives produced slower convergence and poorer clustering, confirming the robustness of NT-Xent for this dataset. We trained for **150 epochs** using **Adam (lr = 1e-3)** with a **cosine annealing scheduler**; the scheduler consistently yielded higher linear probe accuracy compared to a fixed learning rate.

We used stronger augmentations than the exercises to promote invariance. Removing blur degraded OOD accuracy, indicating its benefit for texture-independent representations.

### Evaluation and Results

Feature extraction was done in frozen mode with normalization. Two probe classifiers were trained:

- **k-NN (k=5)** on cosine distance,
- **Logistic Regression** for linear probing.

Our final model achieved:

- **In-Distribution:** k-NN = 28.95%, Linear = 34.82%
- **Out-of-Distribution:** k-NN = 19.45%, Linear = 23.55%

### Visualization and Interpretation

All visualizations (loss curves, t-SNE embeddings, and sample predictions) are included in Section 6 of the submitted notebook. They show progressively tighter feature clusters and consistent semantic grouping, confirming stable learning and minimal overfitting. OOD embeddings remain more dispersed but show partial alignment with similar ID concepts.
