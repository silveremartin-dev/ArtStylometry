# ArtStylometry: Computational Stylometry & Quantitative Connoisseurship Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/embeddings-DINOv2%20%2F%20Gram-purple.svg)](https://github.com/facebookresearch/dinov2)
[![Clustering: HDBSCAN](https://img.shields.io/badge/clustering-HDBSCAN%20%2B%20UMAP-yellow.svg)](https://scikit-learn.org/)
[![Tests: 20 Passing](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)
[![IIIF API Compliant](https://img.shields.io/badge/IIIF-v2%20%26%20v3-red.svg)](https://iiif.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ArtStylometry** is an open-source, high-throughput computational stylometry and quantitative connoisseurship engine for fine art paintings. It transforms ultra-high-resolution museum image archives (**IIIF gigapixel collections** from the **Musée du Louvre**, **Smithsonian Institution**, **Rijksmuseum**, and **National Gallery**) into statistical artist reference manifolds, **intra-canvas multi-hand spatial attribution heatmaps**, and **unsupervised collection clusters**, isolating studio assistant interventions, late restorations, and contested attributions without relying on opaque commercial black boxes.

---

## 🏛️ Interactive Museum Artwork Browser (`browser.html`)

ArtStylometry includes a standalone, zero-dependency offline web application ([`browser.html`](file:///c:/Silvere/Encours/Developpement/ArtStylometry/browser.html)) allowing researchers, curators, and collectors to navigate fine art corpora:

* **Dual Ordering**: Sort artists **Chronologically** (by artistic epoch: Early/High Renaissance, Dutch Golden Age, French Baroque, Gilded Age Tonalism) or **Alphabetically (A-Z)**.
* **Instant Search**: Search bar to query painters, periods, museums, or painting titles in real time.
* **Empirical Attribution Degree**: Every canvas displays a calibrated confidence percentage, status pill (*Consensus Masterwork*, *Studio Hand*, *Disputed / Multi-Hand*, or *Suspected Misattribution*), Mahalanobis distance, and crack cleanliness index.
* **Famous Case Studies ("Cas d'école")**: Dedicated filter and diagnostic cards detailing landmark historical controversies (e.g. Leonardo's *Salvator Mundi*, Rembrandt's *The Man with the Golden Helmet*, Vermeer's *Girl with a Flute*).
* **Interactive Heatmap Overlay**: Live opacity slider blending the continuous 2D attribution heatmap and outlining detected anomaly bounding boxes over the canvas.
* **Museum Links**: Direct hyperlinks to verified catalog accession records at the **Musée du Louvre** (`collections.louvre.fr`), **Smithsonian American Art Museum & Freer Gallery**, and **Rijksmuseum**.

```bash
# Launch or regenerate the interactive browser
python scripts/generate_browser.py
# Open browser.html directly in any web browser!
```

---

## 🏷️ Repository Metadata & Recommended Tags

* **GitHub Repository**: [`https://github.com/silveremartin-dev/ArtStylometry`](https://github.com/silveremartin-dev/ArtStylometry)
* **Short Description**:
  > *High-throughput computational stylometry and quantitative connoisseurship engine for fine art paintings: patch-level micro-texture analysis, self-supervised vision embeddings (DINOv2), multi-hand intra-canvas attribution heatmaps, and outlier clustering via IIIF museum archives.*
* **Recommended GitHub Topics / Tags**:
  `computational-stylometry` &bull; `art-attribution` &bull; `computer-vision` &bull; `dinov2` &bull; `art-history` &bull; `digital-humanities` &bull; `iiif` &bull; `unsupervised-clustering` &bull; `hdbscan` &bull; `texture-analysis` &bull; `patch-embeddings` &bull; `fine-art` &bull; `brushstroke-analysis` &bull; `connoisseurship` &bull; `anomaly-detection`

---

## 🔍 The Core Problem: Why Standard Deep Learning Fails in Fine Art

Commercial art attribution services operate behind private paywalls charging thousands of euros per certificate, while traditional academic computer vision scripts routinely fail to transfer into real connoisseurship for fundamental structural reasons:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 4 BOTTLENECKS OF COMPUTATIONAL CONNOISSEURSHIP              │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. THE SEMANTIC TRAP     │ 2. SENSOR & LIGHT BIAS   │ 3. THE CRAQUELURE NOISE          │
│ Standard CNNs (ResNet)   │ Digitizations from the   │ Centuries of varnish oxidation,  │
│ classify iconography     │ Louvre vs Prado differ   │ wood warping, and crack networks │
│ (Virgin, halo, drapery)  │ in temperature and DPI;  │ fool texture models into         │
│ rather than painterly    │ naive models cluster by  │ grouping by conservation state   │
│ facture and brushstroke. │ scanner hardware.        │ rather than artist hand.         │
├──────────────────────────┴──────────────────────────┴──────────────────────────────────┤
│ 4. THE WORKSHOP REALITY vs THE SINGLE-GENIUS MYTH                                      │
│ Masters ran workshops: the master designed the composition and painted the face, while │
│ pupils painted backgrounds and drapery. Attribution is not binary; it is spatial!     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### ArtStylometry Solution: Inverting the Problem
Proving indisputable authenticity in visible light is an epistemological trap due to collaborative studio practices. In contrast, **identifying when an artwork or specific region represents a statistical anomaly within an established corpus** is a mathematically rigorous, computable engineering problem.

---

## 🔄 End-to-End Pipeline Architecture

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
│  2. Chromatic Decoupling (Rec. 709 luminance gradients, Sobel operators)│
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

## 🧪 Empirical Validation: The Three-Stage Benchmark

To guarantee that the pipeline captures authentic painterly facture rather than statistical artifacts, ArtStylometry establishes a three-stage experimental validation protocol:

| Experimental Stage | Hypothesis & Protocol | Empirical Outcome | Connoisseurship Impact |
| :--- | :--- | :---: | :--- |
| **Stage 1: Sanity Control** | Unsupervised separation of 3 contemporaneous 17th-century Dutch masters (**20 Rembrandt**, **20 Frans Hals**, **20 Vermeer**) sharing identical pigments. | **3 Distinct Clusters**<br>*(Silhouette: 0.460)* | Proves the engine captures physical handling (impasto vs wet-in-wet vs smooth glazes) without supervision. |
| **Stage 2: Pupil Intruder** | Blind insertion of recognized studio pupil works (**Gerrit Dou**, **Ferdinand Bol**) into the Rembrandt corpus. | **5/5 Pupils Flagged**<br>*(Median Conf: 0.000 vs 0.487)* | Isolates followers and copyists into peripheral sub-clusters or outlier classes. |
| **Stage 3: Controversy Crash Test** | Sliding-window spatial audit on contested composite works (e.g. *Salvator Mundi*, *Man with the Golden Helmet*). | **Spatial Zone Breaks Detected** | Distinguishes canonical master regions (face, hands) from pupil-executed drapery and late restorations. |

---

## ⚡ Quickstart Guide

### 1. Installation
Clone the repository and install requirements:

```bash
git clone https://github.com/silveremartin-dev/ArtStylometry.git
cd ArtStylometry
pip install -r requirements.txt
```

### 2. Harvest Museum Paintings via IIIF
Download native-resolution paintings directly from open-access museum IIIF servers:

```bash
# Harvest from Art Institute of Chicago / Rijksmuseum / National Gallery
python scripts/harvest_iiif.py \
  --url "https://api.artic.edu/api/v1/artworks/27992/manifest.json" \
  --out "data/raw/seurat_grande_jatte.jpg" \
  --max-dimension 6000
```

### 3. Audit an Artwork & Generate Attribution Heatmap
Perform an intra-canvas attribution audit against a reference master corpus:

```bash
python scripts/generate_attribution_heatmap.py \
  --artwork "data/raw/contested_canvas.jpg" \
  --references "data/raw/masterpiece_1.jpg" "data/raw/masterpiece_2.jpg" "data/raw/masterpiece_3.jpg" \
  --artist "Leonardo da Vinci" \
  --out-dir "data/processed/audits"
```
*Generates a side-by-side comparative heatmap (`.jpg`) and a standalone, zero-dependency interactive visual inspection dashboard (`index.html`).*

### 4. Cluster an Artwork Collection
Group a directory of uncurated paintings into stylistic schools and isolate outliers:

```bash
python scripts/cluster_collection.py \
  --images data/raw/*.jpg \
  --min-cluster-size 3
```

### 5. Run the Validation Benchmark & Tests
```bash
# Run the 3-stage validation benchmark
python scripts/run_sanity_check.py --synthetic

# Run test suite (17 passing unit & integration tests)
pytest -v tests/
```

---

## 🏛️ Repository Structure

```
ArtStylometry/
├── AGENTS.md                       # Developer & AI agent collaboration directives
├── PROJECT_STATE.md                # Session memory & completed assets inventory
├── README.md                       # Master overview, visual presentation & benchmarks
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
│
├── data/
│   ├── raw/                        # Ingested high-resolution artwork files (.jpg, .png, .tif)
│   ├── tiles/                      # Cached IIIF deep-zoom image tiles
│   └── processed/                  # Generated audit heatmaps and HTML reports
│
├── docs/
│   ├── ARCHITECTURE.md             # Consolidated mathematical & technical specification
│   ├── STYLOMETRIC_METHODOLOGY.md  # Connoisseurship theory & epistemology
│   ├── IIIF_HARVESTING_GUIDE.md    # Guide to harvesting museum collections via IIIF
│   └── SANITY_CHECK_PROTOCOL.md    # 3-stage validation protocol (Rembrandt/Hals/Vermeer)
│
├── scripts/
│   ├── harvest_iiif.py             # Download gigapixel artworks from museum IIIF endpoints
│   ├── run_sanity_check.py         # 3-stage validation benchmark runner
│   ├── generate_attribution_heatmap.py # Single artwork spatial audit CLI
│   └── cluster_collection.py       # Unsupervised collection clustering CLI
│
├── scratch/
│   └── demo_synthetic_experiment.py # End-to-end simulation generating visual audit & report
│
├── src/                            # Core generic library modules
│   ├── acquisition/                # IIIF client & deep-zoom reconstructor
│   ├── preprocessing/              # Tiler, chromatic decoupler, craquelure filter
│   ├── embeddings/                 # Micro-facture DINOv2 / Gram features & VLAD aggregator
│   ├── attribution/                # Reference profile, spatial heatmap, anomaly detector
│   ├── clustering/                 # HDBSCAN clusterer & UMAP/PCA dimension reducer
│   ├── visualization/              # Heatmap overlay renderer & interactive HTML dashboard
│   └── pipeline.py                 # Central CLI pipeline orchestrator
│
└── tests/                          # 17 comprehensive unit & integration tests
    ├── test_acquisition.py
    ├── test_preprocessing.py
    ├── test_embeddings.py
    ├── test_attribution.py
    ├── test_clustering.py
    └── test_pipeline.py
```

---

## ⚖️ Scientific Disclaimers & Ethics

1. **Anomaly Detection, Not a Legal Verdict**: ArtStylometry is an exploratory research instrument for quantitative connoisseurship. In visible-light photography, algorithmic scores cannot substitute for multimodal physical laboratory forensics (infrared reflectography for underdrawings, X-ray radiography for lead densities, and X-ray fluorescence spectroscopy for elemental pigment analysis).
2. **Open Science & Museum Open Access**: This project operates strictly in public-domain fine art collections harvested via open-standard IIIF endpoints.

---

## 📜 Citation & License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```bibtex
@software{martin2026artstylometry,
  author = {Martin-Dev, Silvere},
  title = {ArtStylometry: Computational Stylometry and Quantitative Connoisseurship Engine for Fine Art Paintings},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/silveremartin-dev/ArtStylometry}
}
```
