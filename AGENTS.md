# AGENTS.md — Developer & AI Agent Collaboration Directives for ArtStylometry

## 🎯 1. Core Project Identity & Scope

**ArtStylometry is a high-throughput computational stylometry and quantitative connoisseurship engine for fine art paintings, transforming high-resolution museum image archives (IIIF) into statistical artist profiles, intra-canvas multi-hand attribution heatmaps, and unsupervised stylistic clusters.**

### The Core Problem:
Art history and computational connoisseurship suffer from acute technical and epistemological bottlenecks:
1. **The Semantic Classifier Illusion**: Standard deep convolutional networks (e.g. ResNet, VGG) trained on ImageNet classify the *subject matter* (a Madonna and Child, a landscape, a red drapery) or global color palette, completely failing to capture the *facture*—the artist's physical handling of the brush, impasto density, and stroke ductility.
2. **Sensor & Digitization Confounding**: Museum digitizations vary wildly across institutions. If painter A is scanned at 1,200 DPI under warm halogen lighting at the Louvre, and painter B is photographed at 300 DPI under cool LED at the Prado, standard models cluster by camera sensor and ICC color profile rather than artist style.
3. **Material Degradation & Craquelures**: Centuries of varnish oxidation, wood warping, relining, and micro-crack networks (*craquelure*) inject high-frequency noise that easily tricks naive texture extractors into clustering works by conservation state rather than painterly touch.
4. **Workshop Reality vs. The Myth of the Solitary Genius**: Renaissance and Baroque masters operated collaborative studios. A master planned the composition and painted primary faces or hands, while pupils and journeymen painted backgrounds, draperies, or secondary figures. Attribution cannot be a binary scalar; it requires spatial intra-canvas segmentation.
5. **The Proprietary Paywall & Academic Graveyard**: Commercial authentication services operate as opaque, closed-source black boxes charging thousands of euros per certificate, while academic vision papers leave behind unmaintained, non-reproducible research scripts locked to private datasets.

### ArtStylometry Boundary Principles:
* **IN SCOPE**:
  1. **IIIF Deep-Zoom Harvesting (`src/acquisition/`)**: Harvesting native gigapixel image tiles from museum open-access IIIF servers (National Gallery London, Rijksmuseum, The Met, Art Institute of Chicago, Gallica).
  2. **Micro-Texture Patch Extraction (`src/preprocessing/`)**: Slicing ultra-high-resolution paintings into $256 \times 256$ and $512 \times 512$ tiles at native scale without destructive downsampling.
  3. **Chromatic Decoupling & Gradient Analysis (`src/preprocessing/`)**: Isolating spatial luminance derivatives (Sobel gradients, directional filter responses) from chrominance to evaluate pure drawing and volume transitions independent of pigment aging.
  4. **Craquelure & Varnish Filtering (`src/preprocessing/`)**: Applying morphological filters and top-hat transforms to mask crack networks and varnish oxidation artifacts.
  5. **Dual-Scale Self-Supervised Embeddings (`src/embeddings/`)**:
     - *Micro-Scale*: Self-supervised ViT patch tokens (DINOv2) and Gram matrix texture descriptors capturing brushwork ductility and sfumato transitions.
     - *Macro-Scale*: Global composition and structural proportions.
  6. **Statistical Reference Profiling (`src/attribution/`)**: Modeling consensus masterworks into multivariate Gaussian distributions, computing Mahalanobis and cosine distance manifolds.
  7. **Intra-Canvas Anomaly Heatmaps (`src/attribution/`)**: Sliding-window spatial audit highlighting studio interventions, secondary hands, pentimenti, or modern overpaints.
  8. **Unsupervised Collection Clustering (`src/clustering/`)**: Projecting aggregated image descriptors via UMAP and HDBSCAN to group unknown lots and isolate questionable outliers.
  9. **Interactive Connoisseurship Dashboard (`src/visualization/`)**: Inspecting high-resolution overlays, patch-level distance metrics, and cluster topologies.
* **OUT OF SCOPE / STRICTLY REJECTED**:
  - Unconstrained generative inpainting (no diffusion hallucinations modifying original brushwork).
  - Subject classification or semantic tagging ("is this a portrait or a landscape?").
  - Claiming definitive judicial or legal finality on single 2D visible-light scans without multimodal physical diagnostics (infrared reflectography, X-ray radiography, XRF spectroscopy).
  - Proprietary lock-in: all features, embeddings, and models must remain fully transparent, scriptable, and open source.

---

## ⚡ 2. The Computational & Yield Filter

ArtStylometry operates under the principle that **attribution in visible light is an anomaly detection problem, not an authoritarian judicial verdict**:

$$\text{Attribution Confidence} = \mathcal{D}_{\text{Mahalanobis}}(\mathbf{e}_{\text{patch}}, \boldsymbol{\mu}_{\text{master}}, \boldsymbol{\Sigma}_{\text{master}}^{-1}) \times \mathcal{M}_{\text{crack\_cleanliness}} \times \mathcal{S}_{\text{gradient\_coherence}}$$

* **Rule of Statistical Neutrality**: The system evaluates geometric and textural signal distributions without awareness of historical market valuations.
* **The "Outlier Inversion" Principle**: Rather than attempting to prove indisputable authenticity (vulnerable to workshop ambiguity), ArtStylometry identifies when an artwork or specific region represents a statistically significant anomaly within an established corpus.

---

## 🔄 3. Multi-Engine Architecture

All AI agents and developers implementing features in ArtStylometry must adhere to the modular pipeline:

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
│  - Interactive HTML Inspection Dashboard (`scripts/`)                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 4. Technical Documentation Standard

* **Strict English Language Requirement**: ALL documentation (`README.md`, `docs/*.md`, `AGENTS.md`, `PROJECT_STATE.md`), code docstrings, typing annotations, test names, and commit messages MUST be written strictly in English.
* **Reproducible Protocols**: All museum references must include canonical institutional catalog IDs, IIIF manifest URLs, or accession numbers (e.g., Rijksmuseum `SK-A-399`, National Gallery `NG6376`).
* **Test-Driven Rigor**: Every engine module must have a corresponding test suite in `tests/` verifying both synthetic edge cases and real image arrays.

---

## 🤖 5. Specialized Modules & Components

1. **`Acquisition` (`src/acquisition/`)**:
   - `iiif_client.py`: Interrogates IIIF Presentation and Image API endpoints (v2 and v3).
   - `downloader.py`: Assembles deep-zoom tiles at full resolution with rate-limiting and cached local storage.
2. **`Preprocessing` (`src/preprocessing/`)**:
   - `tiler.py`: Slices gigapixel canvases into overlapping or regular patches with border preservation.
   - `color_decoupling.py`: Strips chromatic camera bias; computes spatial gradients, directional filter banks, and normalized grayscale textures.
   - `craquelure_filter.py`: Detects and masks age-induced craquelure networks via morphological black top-hat filtering.
3. **`Embeddings` (`src/embeddings/`)**:
   - `micro_extractor.py`: Extracts patch-level ViT embeddings (DINOv2 `dinov2_vits14`, `dinov2_vitb14`) and local gradient statistics.
   - `macro_extractor.py`: Extracts canvas-wide structural composition features.
   - `aggregator.py`: Performs statistical pooling (mean, std, skew) and VLAD / Fisher vector aggregation for canvas-level representation.
4. **`Attribution` (`src/attribution/`)**:
   - `reference_profile.py`: Fits reference artist distributions (centroid, covariance matrix, Mahalanobis metric).
   - `heatmap_generator.py`: Computes tile-by-tile attribution confidence across a canvas to detect studio assistants or overpaints.
   - `anomaly_detector.py`: Evaluates one-class SVM and Isolation Forest anomaly boundaries.
5. **`Clustering` (`src/clustering/`)**:
   - `dimension_reduction.py`: Projects high-dimensional artwork descriptors to 2D/3D manifolds via UMAP / PCA.
   - `clusterer.py`: Performs density-based clustering via HDBSCAN to segregate artist clusters and isolate contested works into noise classes.
6. **`Visualization` (`src/visualization/`)**:
   - `heatmap_overlay.py`: Blends attribution confidence heatmaps with original artwork pixels.
   - `dashboard.py`: Exports interactive HTML/JavaScript visual exploration interfaces.

---

## 🏗️ 6. Source & Test Separation Directives

* **Domain-Agnostic Generic `src/`**: All modules in `src/` MUST remain 100% generic, reusable, and free of hardcoded paths or single-artist assumptions.
* **Mirrored `tests/` Hierarchy**: All test fixtures, mock data generators, and unit tests MUST reside under `tests/` mirroring `src/` (`tests/test_acquisition.py`, `tests/test_preprocessing.py`, `tests/test_embeddings.py`, `tests/test_attribution.py`, `tests/test_clustering.py`, `tests/test_pipeline.py`).
* **Offline Fallbacks**: Tests MUST execute and pass without requiring gigabyte-sized external model downloads or active internet connections by supporting synthetic feature mocks and lightweight deterministic backends.
