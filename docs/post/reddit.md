# Reddit launch post — ArtStylometry

## Suggested subreddits

- **r/MachineLearning**: post under the `[P]` (Project) tag. Good for feedback on shrinkage covariance with N << D, DINOv2 patch features for texture, and HDBSCAN outlier handling.
- **r/computervision**: the core technical questions are about texture vs. semantics, scanner/DPI confounds, and patch boundary artifacts.
- **r/DigitalHumanities**: the right audience for the IIIF harvesting side and the epistemology of computational connoisseurship.
- **r/ArtHistory**: useful for domain critique (workshop practice, which test cases make sense). Keep the post less technical there and check the sub's self-promotion rules first.
- **r/opensource** or **r/Python**: optional, a lighter version for people who might contribute.

---

**Title:** [P] ArtStylometry: an open-source pipeline that treats painting attribution as patch-level anomaly detection, not classification (early stage, feedback welcome)

---

Hi all,

I've just open-sourced **ArtStylometry**, a Python pipeline for computational stylometry on paintings. It's an early-stage project and I'd really like critical feedback before I put more work into validation.

### Context / motivation

Most "AI art attribution" either lives behind commercial paywalls or ends up as one-off research scripts tied to private datasets. Naive approaches also tend to fail in the same few ways:

- **Semantic trap**: ImageNet-style features pick up *subject matter* (a Madonna, a drapery) more than *facture* (how the brush was handled).
- **Acquisition confounds**: different museums use different DPI, lighting and colour profiles, so models end up clustering by scanner.
- **Craquelure/varnish**: crack networks add high-frequency texture, so works get grouped by conservation state rather than by hand.
- **Workshops**: masters and pupils often painted parts of the same canvas, so a single "authentic / not authentic" label is the wrong output.

So I flipped the question. Instead of trying to *prove* authorship, the tool asks: **which regions of a canvas are statistical outliers compared with a reference corpus of consensus works?**

### What it does

1. **Acquisition**: IIIF Presentation/Image API (v2 and v3) client, deep-zoom tile reassembly, and a registry of endpoints for 14 institutions (Rijksmuseum, NGA, National Gallery London, Met, AIC, Getty, etc.).
2. **Preprocessing**: patches cut at *physical* scale (mm, using IIIF physical dimensions where available) rather than fixed pixels; Rec. 709 luminance/gradient decoupling; DoG/FFT bandpass and morphological top-hat to suppress craquelure; optional alignment of IRR/X-ray layers.
3. **Embeddings**: DINOv2 patch tokens plus Gram/Gabor gradient statistics, with Hann/Gaussian apodization at patch borders. There's a deterministic non-GPU fallback, so tests run offline. Patches are aggregated with VLAD / statistical pooling.
4. **Attribution**: per-artist reference profile using **Ledoit-Wolf / OAS shrinkage covariance** (masters often have only 15–30 works, so the empirical covariance is useless), Mahalanobis distance, a sliding-window heatmap, and One-Class SVM / Isolation Forest.
5. **Clustering**: UMAP/PCA followed by HDBSCAN, with the noise class used as an "outlier" bucket.
6. **Output**: heatmap overlays and standalone HTML dashboards.

There are 28 unit/integration tests, all passing. The licence is MIT.

### Results (please read the caveat)

The three-stage benchmark script (`scripts/run_sanity_check.py`) currently runs on **synthetic 64-D feature vectors**: seeded Gaussian clusters that only *stand in* for Rembrandt / Hals / Vermeer. They are not real image embeddings. On that synthetic setup:

- Stage 1 (3 "artists" × 20 works): 3 clusters recovered, silhouette **0.460**
- Stage 2 (5 "pupil" samples placed near the "Rembrandt" centroid): median attribution confidence **0.000** vs **0.487** for core samples; **5/5** flagged as anomalies
- Stage 3: a synthetic composite canvas (smooth "sfumato" region + hatched "pupil" region) gives a visible zone break in the heatmap, and the canvas lands in the HDBSCAN noise class.

To be clear, these numbers only show that **the plumbing works as intended**. They say nothing yet about whether the features separate real painters. Running the same protocol on real IIIF scans is the next milestone.

The repo also includes a demo HTML browser of museum works and famous case studies. Its scores and rationales are **hand-curated illustrative records**, not outputs of the pipeline, so please don't read them as findings.

### What it is NOT

- Not a certificate of authenticity, and it makes no definitive attribution claims about any specific painting.
- Not a replacement for technical art history: IRR, X-radiography, XRF and dendrochronology see things a visible-light scan can't.
- Not a subject classifier, and it does no generative inpainting.

### Feedback I'm looking for

- **Validation design**: what would be a convincing real-data test? Ideally sets with well-documented workshop involvement and good open IIIF scans.
- **Confounds**: beyond DPI normalization and bandpass filtering, how would you control for acquisition and conservation bias? Per-institution stratification? Same-museum-only splits?
- **Features**: are DINOv2 patch tokens a reasonable texture descriptor here, or should I lean on handcrafted/Gram features or a texture-specific SSL model?
- **Domain critique** from art historians or conservators: where is the framing naive?

Repo: https://github.com/silveremartin-dev/ArtStylometry

Happy to answer questions, and harsh but constructive criticism is very welcome. Thanks for reading!

— Silvère Martin-Michiellot
