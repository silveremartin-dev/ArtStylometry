# Stylometric Sanity Check & Validation Protocol

## 1. Overview & Scientific Purpose

To ensure that the algorithmic pipeline captures true painterly facture rather than statistical artifacts, ArtStylometry defines a **three-stage experimental protocol** for scientific validation without circular reasoning.

---

## 2. The Three-Stage Experimental Protocol

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1 : Control Group                         │
│  20 Rembrandt + 20 Frans Hals + 20 Vermeer                             │
│  (Same 17th-century Dutch pigments, contemporaneous workshops)         │
│         │                                                              │
│         ▼                                                              │
│  Unsupervised Clustering (DINOv2 / Gram + HDBSCAN)                     │
│  Goal: Spontaneous separation into 3 distinct clusters (Silhouette > 0.45)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2 : Studio Pupil Intruder                      │
│  Rembrandt Corpus + Injected Gerrit Dou / Ferdinand Bol Paintings      │
│         │                                                              │
│         ▼                                                              │
│  One-Class Anomaly Detection & Reference Profiling                     │
│  Goal: Pupil paintings flagged as statistical outliers or sub-cluster │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 3 : The Controversy Crash Test                   │
│  High-Stakes Disputed Works (e.g. Salvator Mundi, Golden Helmet)       │
│         │                                                              │
│         ▼                                                              │
│  Sliding-Window Spatial Attribution Heatmap                            │
│  Goal: Reveal intra-canvas multi-hand zones (master face vs assistant) │
└────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: The Sanity Control (Dutch Golden Age Masters)
* **Corpus**:
  - 20 uncontested paintings by **Rembrandt van Rijn**
  - 20 uncontested paintings by **Frans Hals**
  - 20 uncontested paintings by **Johannes Vermeer**
* **Rationale**: These three masters worked in the same region during the 17th century using identical Dutch pigments (lead white, bone black, smalt, ochres, vermilion). Their subjects frequently overlapped (portraits, genre scenes). However, their physical handling was radically distinct:
  - Rembrandt: Heavy impasto, scraped glazes, chiaroscuro lighting.
  - Frans Hals: Virtuosic wet-in-wet direct brushstrokes, open painterly touch (*alla prima*).
  - Johannes Vermeer: Delicate translucent glazes, pointillist specular highlights (*camera obscura* effect), subtle luminosity.
* **Criterion**: An unsupervised pipeline (patch tiling + gradient/texture embeddings + HDBSCAN) MUST spontaneously separate these works into three distinct clusters without human supervision.

### Stage 2: The Studio Pupil Intruder Test
* **Corpus**: The 20 Rembrandt works from Stage 1, into which 5 recognized works by his direct studio pupils (**Gerrit Dou**, **Ferdinand Bol**, **Govert Flinck**) are blind-injected.
* **Rationale**: Pupils trained to replicate Rembrandt's style, palette, and themes. However, their stroke ductility, fine hatching, and anatomical transitions deviate from the master's signature.
* **Criterion**: The statistical reference profile fitted on canonical Rembrandts must isolate the pupil works into a peripheral sub-cluster or flag them as outliers ($p < 0.05$).

### Stage 3: The Controversy Crash Test
* **Corpus**: Famous disputed artworks where scholarly consensus has shifted or remains contested:
  - *The Man with the Golden Helmet* (formerly Rembrandt, re-attributed to Circle of Rembrandt / workshop).
  - *Salvator Mundi* (attributed by some to Leonardo, by others to Boltraffio or studio collaboration).
* **Rationale**: Multi-hand execution occurs when the master conceives the figure and paints the primary focal point (face/eyes), while an assistant executes the drapery, background, and secondary elements.
* **Criterion**: The spatial attribution heatmap should highlight localized zone breaks, distinguishing canonical master regions from assistant-executed zones.

---

## 3. Running the Benchmark

The entire three-stage validation protocol can be run locally using the provided benchmark script:

```bash
# Run synthetic calibration benchmark
python scripts/run_sanity_check.py --synthetic
```
