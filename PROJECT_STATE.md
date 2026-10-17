# PROJECT_STATE.md — ArtStylometry Project State & Session Memory

## 🎯 1. Project Identity & Repository
* **Project Name**: ArtStylometry
* **Repository**: [`https://github.com/silveremartin-dev/ArtStylometry`](https://github.com/silveremartin-dev/ArtStylometry)
* **Tagline**: Computational Stylometry & Quantitative Connoisseurship Engine for Fine Art Paintings.
* **License**: MIT License (2026 Silvere Martin-Dev / ArtStylometry Contributors).

---

## 🏛️ 2. Core Modules & Implementation Status

| Module | Location | Status | Key Features |
| :--- | :--- | :---: | :--- |
| **`Acquisition`** | `src/acquisition/` | ✅ Complete | IIIF Presentation API (v2/v3), Image API `info.json`, tile grid calculation, deep-zoom tile downloader, local canvas loader |
| **`Preprocessing`** | `src/preprocessing/` | ✅ Complete | Non-destructive micro-patch tiler, Rec. 709 chromatic decoupler, Sobel spatial derivatives, morphological black top-hat craquelure filter |
| **`Embeddings`** | `src/embeddings/` | ✅ Complete | Micro-facture extractor (DINOv2 patch tokens + deterministic Gram/Gabor gradient fallback), macro-composition layout, VLAD / statistical moment aggregator |
| **`Attribution`** | `src/attribution/` | ✅ Complete | Artist reference profiling with Mahalanobis & Cosine manifolds, sliding-window spatial heatmap generator, One-Class SVM / Isolation Forest anomaly detector |
| **`Clustering`** | `src/clustering/` | ✅ Complete | High-dimensional artwork clustering (HDBSCAN/DBSCAN), UMAP / PCA manifold projection, noise & outlier isolation |
| **`Visualization`**| `src/visualization/` | ✅ Complete | Heatmap overlay renderer with zone bounding boxes, side-by-side comparative views, standalone zero-dependency interactive HTML dashboard |
| **`Browser`**      | `src/browser/`       | ✅ Complete | Interactive browser (`browser.html`) with chronological/alphabetical sorting, real-time search, Musée du Louvre & Smithsonian collections, confidence badges, interactive heatmap slider, and famous controversy case studies |
| **`Pipeline`**     | `src/pipeline.py`    | ✅ Complete | Central CLI orchestrator for auditing individual paintings, clustering multi-painting collections, and building the interactive browser |

---

## 🧪 3. Test Suite & Validation Status

* **Status**: 100% Passing (20 of 20 tests passed).
* **Test Command**: `py -3 -m pytest -v tests/`

```
tests/test_acquisition.py::test_iiif_image_service_url_construction PASSED
tests/test_acquisition.py::test_iiif_client_parse_v2_manifest PASSED
tests/test_acquisition.py::test_iiif_client_parse_v3_manifest PASSED
tests/test_acquisition.py::test_load_local_canvas PASSED
tests/test_attribution.py::test_reference_profile_mahalanobis_calibration PASSED
tests/test_attribution.py::test_heatmap_generator_detects_anomaly_region PASSED
tests/test_attribution.py::test_stylistic_anomaly_detector PASSED
tests/test_browser.py::test_catalog_data_integrity PASSED
tests/test_browser.py::test_browser_html_rendering PASSED
tests/test_browser.py::test_browser_export PASSED
tests/test_clustering.py::test_dimension_reducer_pca_and_umap PASSED
tests/test_clustering.py::test_collection_clusterer_discovers_groups_and_outliers PASSED
tests/test_embeddings.py::test_micro_facture_extractor PASSED
tests/test_embeddings.py::test_macro_composition_extractor PASSED
tests/test_embeddings.py::test_patch_aggregator_statistical_and_vlad PASSED
tests/test_pipeline.py::test_pipeline_audit_and_clustering PASSED
tests/test_preprocessing.py::test_artwork_tiler_slice_dimensions PASSED
tests/test_preprocessing.py::test_tiler_filters_blank_margins PASSED
tests/test_preprocessing.py::test_chromatic_decoupler_luminance_and_gradients PASSED
tests/test_preprocessing.py::test_craquelure_detector_synthetic_crack PASSED
============================= 20 passed in 10.60s =============================
```

---

## 📊 4. Three-Stage Benchmark & Experimental Verification

Operationalized in `scripts/run_sanity_check.py` and demonstrated in `scratch/demo_synthetic_experiment.py`:

1. **Stage 1: Dutch Masters Separation**:
   - 20 Rembrandt + 20 Frans Hals + 20 Vermeer.
   - Spontaneously discovered 3 distinct clusters (Silhouette: 0.460).
2. **Stage 2: Pupil Intruder Detection**:
   - Rembrandt core median confidence: `0.487`.
   - Gerrit Dou (pupil) median confidence: `0.000`.
   - 5 of 5 pupil works successfully flagged as statistical outliers.
3. **Stage 3: Composite Artwork Spatial Heatmap**:
   - Evaluated synthetic Leonardo vs pupil composite painting (`salvator_mundi_sim_composite`).
   - Sfumato face identified as canonical master touch; pupil drapery and background highlighted as anomaly zone.
   - Overall artwork isolated into outlier noise class `[-1]` in collection clustering.

---

## 📂 5. Directory Inventory

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
