# ArtStylometry: Consolidated System Architecture & Mathematical Specification

## 1. Executive Summary & Foundational Principles

**ArtStylometry** is an open-source, high-throughput computational stylometry and quantitative connoisseurship engine designed to evaluate fine art paintings. It extracts native-scale painterly facture (brushstroke ductility, impasto profile, sfumato gradations) from high-resolution museum image archives (IIIF) to generate **spatial attribution heatmaps** and perform **unsupervised collection-scale clustering**.

### Core Axiom: Inverting the Attribution Problem
Traditional commercial art authentication attempts to solve the ill-posed problem of proving authenticity:
$$\mathcal{P}(\text{Authentic} \mid \text{Image}) = 1.0$$
In historical reality, master workshops operated collaboratively, with multiple assistants executing draperies, backgrounds, or secondary figures. ArtStylometry inverts the problem into **multivariate statistical anomaly detection**:
$$\mathcal{H}_0: \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{master}}, \boldsymbol{\Sigma}_{\text{master}}) \quad \text{vs.} \quad \mathcal{H}_1: \mathbf{x} \notin \mathcal{D}_{\text{master}}$$

---

## 2. Multi-Stage Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1 : High-Resolution Ingestion               │
│  IIIF Manifests / Local Gigapixel Images (.tif, .png, .jpg)            │
│         │                                                              │
│         ▼ (`src/acquisition/`)                                         │
│  1. Manifest Parser & Level-of-Detail Resolver                         │
│  2. Tile Assembler & Deep-Zoom Reconstructor                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2 : Preprocessing & Noise Neutralization       │
│  Full-Resolution Raw Artwork                                           │
│         │                                                              │
│         ▼ (`src/preprocessing/`)                                       │
│  1. Micro-Patch Slicer (256x256 / 512x512 non-destructive tiling)      │
│  2. Chromatic Decoupling (Luminance gradients, Sobel operators)        │
│  3. Craquelure Masking (Morphological top-hat & crack exclusion)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 3 : Dual-Scale Feature Extraction                │
│  Cleaned Micro-Patches + Full Canvas                                   │
│         │                                                              │
│         ▼ (`src/embeddings/`)                                          │
│  1. Micro-Texture Extractor (DINOv2 patch tokens, Gram matrices)       │
│  2. Macro-Composition Extractor (Structural proportions & pose)        │
│  3. Patch Aggregator (VLAD, Fisher Vectors, Statistical Pooling)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│ STAGE 4A: Intra-Canvas Audit    │   │ STAGE 4B: Unsupervised Cluster  │
│ (`src/attribution/`)            │   │ (`src/clustering/`)             │
│ - Master reference distribution │   │ - Global artwork embeddings     │
│ - Sliding window tile distances │   │ - UMAP manifold projection      │
│ - Spatial multi-hand heatmap    │   │ - HDBSCAN cluster & outlier sep │
└─────────────────────────────────┘   └─────────────────────────────────┘
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 5 : Visualization & Reporting                  │
│  - Heatmap Overlays on Original Artwork (`src/visualization/`)         │
│  - Interactive HTML Inspection Dashboard (`src/visualization/`)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations

### 3.1 Chromatic Decoupling & Spatial Luminance Derivatives
To prevent models from clustering paintings according to museum camera sensors, color temperature, or yellowed varnish, RGB images are transformed to standardized Rec. 709 luminance:
$$Y(x, y) = 0.2126 R(x, y) + 0.7152 G(x, y) + 0.0722 B(x, y)$$

Spatial directional gradients are evaluated using Sobel kernels:
$$G_x = Y * K_x, \quad G_y = Y * K_y$$
$$\text{Magnitude: } M(x, y) = \sqrt{G_x^2 + G_y^2}$$
$$\text{Orientation: } \theta(x, y) = \text{arctan2}(G_y, G_x)$$

The brushstroke orientation distribution is captured via magnitude-weighted circular histograms across $B=16$ bins:
$$H_b = \sum_{(x, y) \in \text{bin}_b} M(x, y)$$

### 3.2 Craquelure Network Masking via Morphological Black Top-Hat
Aging cracks appear as dark, narrow fissures across the paint layer. Morphological closing with structuring footprint $S$:
$$\text{Closing}(Y, S) = (Y \oplus S) \ominus S$$
$$\text{Black Top-Hat: } T_{\text{black}}(Y) = \text{Closing}(Y, S) - Y$$

A binary crack mask $\mathcal{B}_{\text{crack}}$ is constructed using percentile thresholding $\tau$:
$$\mathcal{B}_{\text{crack}}(x, y) = \mathbb{I}(T_{\text{black}}(x, y) \ge \tau)$$
The patch cleanliness metric $\mathcal{M}_{\text{clean}}$ downweights contaminated patches:
$$\mathcal{M}_{\text{clean}} = 1.0 - \frac{1}{|\Omega|} \sum_{(x, y) \in \Omega} \mathcal{B}_{\text{crack}}(x, y)$$

### 3.3 Micro-Facture Embeddings
Each patch $\mathbf{p}_i$ is mapped to a normalized feature vector $\mathbf{e}_i \in \mathbb{R}^D$ ($D=128$ in structural mode, $D=384$ or $768$ in DINOv2 mode).
In structural mode, the representation fuses:
1. **Directional gradient statistics** (orientation histogram, radial energy profile, gradient skewness and kurtosis).
2. **Gram matrix of spatial derivatives**:
   $$\mathbf{G} = \frac{1}{N} \mathbf{F} \mathbf{F}^T, \quad \text{where } \mathbf{F} = [Y, G_x, G_y, M, \nabla^2 Y]^T$$
3. **Steerable directional filter bank energy** across multiple scales:
   $$\mathcal{R}_{\sigma, \theta} = \| \cos\theta \cdot G_x + \sin\theta \cdot G_y \|_{\sigma}$$
4. **Spatial frequency pyramid** measuring local standard deviation.

### 3.4 Artist Reference Profiling & Mahalanobis Manifold
Given $N$ reference patches from undisputed masterworks:
$$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{e}_i, \quad \boldsymbol{\Sigma} = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{e}_i - \boldsymbol{\mu})(\mathbf{e}_i - \boldsymbol{\mu})^T$$
With Ledoit-Wolf Tikhonov regularization:
$$\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \lambda \mathbf{I}$$

The Mahalanobis distance from any candidate patch $\mathbf{e}_{\text{test}}$ to the artist profile is:
$$d_{\mathcal{M}}(\mathbf{e}_{\text{test}}) = \sqrt{(\mathbf{e}_{\text{test}} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}_{\text{reg}}^{-1} (\mathbf{e}_{\text{test}} - \boldsymbol{\mu})}$$

Attribution confidence $C(\mathbf{e}) \in [0, 1]$ is evaluated through a calibrated sigmoid:
$$C(\mathbf{e}) = \frac{1}{1 + \exp\left( \frac{d_{\mathcal{M}} - d_{\text{median}}}{\text{scale}} \cdot 1.5 \right)}$$

### 3.5 Intra-Canvas Sliding Window Heatmap
A sliding window of size $W \times W$ with stride $S$ traverses the canvas. Confidence values are accumulated with Gaussian bell weights $G_\sigma(x, y)$:
$$\mathcal{H}(x, y) = \frac{\sum_{k} C(\mathbf{e}_k) \cdot G_\sigma(x - x_k, y - y_k)}{\sum_{k} G_\sigma(x - x_k, y - y_k)}$$

Contiguous low-confidence zones ($C < \tau_{\text{outlier}}$) are grouped into **Anomaly Zones** indicating studio hands, repentirs (pentimenti), or modern restorations.

### 3.6 Unsupervised Collection Clustering
For batch analysis across $M$ paintings, patch embeddings are pooled into a single artwork descriptor $\mathbf{d}_m$ using statistical moments or VLAD (Vector of Locally Aggregated Descriptors):
$$\mathbf{v}_k = \sum_{i: q(\mathbf{e}_i) = k} (\mathbf{e}_i - \mathbf{c}_k)$$

The collection matrix $\mathbf{D} \in \mathbb{R}^{M \times D_{\text{global}}}$ is projected into 2D/3D manifolds via UMAP / PCA and clustered using HDBSCAN:
- Core artist clusters: $\mathcal{C}_1, \mathcal{C}_2, \dots$
- Outlier / noise paintings: $\mathcal{C}_{-1}$
