# Stylometric Methodology: Epistemological & Computational Principles

## 1. The Epistemological Crisis in Art History Attribution

For two centuries, art attribution has relied heavily on the connoisseur's subjective eye—a practice formalized in the 19th century by Giovanni Morelli and Bernard Berenson. While Morelli introduced the crucial insight that an artist reveals their unconscious habits in minor, non-expressive details (earlobes, fingernails, drapery folds), traditional connoisseurship suffers from structural limitations:

1. **Reputational and Market Conflicts of Interest**: Commercial attributions in major auction houses involve financial stakes worth tens of millions of dollars. Attributing a canvas to Leonardo da Vinci rather than his pupil Boltraffio multiplies its financial valuation by a factor of 1,000.
2. **The "Solitary Genius" Anachronism**: Renaissance and Baroque masters did not paint in isolation. Masters such as Raphael, Rembrandt, Rubens, and Leonardo ran large, industrial workshops. The master established the composition (*invenzione* and *disegno*), painted the principal faces and hands, while apprentices, pupils, and journeymen painted background landscapes, draperies, still-life details, and minor figures.
3. **The Cognitive Memory Bottleneck**: An expert human eye can compare only two or three physical canvases simultaneously, relying on memory for other works. A computational vector space can simultaneously evaluate hundreds of works across dozens of contemporary workshops.

---

## 2. The Semantic Classifier Trap in Computer Vision

When computer vision researchers first applied deep learning to fine art (e.g., standard ResNet, VGG, or Inception networks trained on ImageNet), the results were largely disappointing for connoisseurship:

* **Iconographic Dominance**: Supervised networks learn high-level semantic shortcuts. Given a dataset of Renaissance paintings, a standard network classifies whether the image portrays a "Madonna and Child", a "Crucifixion", or a "Bacchus". It does not look at the brushstroke; it looks at the presence of a halo, a red cloak, or a mountain range.
* **Color Temperature Confounding**: Standard CNNs are highly sensitive to low-frequency chromatic shifts. If the National Gallery's scans are slightly cooler than the Louvre's scans, the network clusters all London paintings together and all Paris paintings together.
* **Destructive Downsampling**: Resizing a $10,000 \times 8,000$ pixel gigapixel scan down to $224 \times 224$ pixels destroys the physical footprint of the brush, obliterating sfumato glazes, paint layer thickness, and impasto ridges.

---

## 3. The ArtStylometry Solution: Dual-Scale & Chromatic Decoupling

To overcome these structural pitfalls, ArtStylometry enforces three foundational design principles:

### A. Non-Destructive Micro-Tiling
The canvas is sliced into native-resolution patches ($256 \times 256$ or $512 \times 512$ pixels) without downsampling. Each patch operates as an independent observation of painterly facture:
* Impasto ductus and tool edge sharpness
* Sfumato transition smoothness
* Hatching and cross-hatching stroke density

### B. Chromatic Decoupling
Color palettes vary wildly based on pigment oxidation, centuries of yellowed varnish, and museum lighting setups. By stripping chromatic information and isolating spatial luminance derivatives ($G_x, G_y$), the engine evaluates the *disegno* (structural volume transitions and drawing touch) rather than modern color temperature.

### C. Morphological Craquelure Masking
Aging cracks (*craquelure*) are mechanical fracture networks resulting from drying oils, fluctuating relative humidity, and wooden panel expansion. A naive convolutional model often groups paintings by their age and conservation state rather than artist style. ArtStylometry applies morphological black top-hat filters to detect and downweight crack networks, ensuring the features reflect original artist strokes rather than conservation history.

---

## 4. Anomaly Detection Over Binary Classification

ArtStylometry avoids the trap of binary supervised classifiers ("Is this painting 100% Rembrandt? Yes/No"):
* There is no undisputed, universally agreed ground-truth dataset in early modern painting. Contested works like the *Salvator Mundi* or Rembrandt's *Man with the Golden Helmet* have divided the scholarly community for decades.
* Anomaly detection provides a scientifically defensible alternative:
  1. Fit a statistical manifold $\mathcal{M}$ on consensus works that all scholars agree are by the master.
  2. Map any candidate canvas across this manifold.
  3. If a canvas or region diverges beyond the $99^{\text{th}}$ percentile of the master's distance distribution, it is flagged as a statistical outlier.
  4. The model produces a spatial heatmap indicating exactly *where* on the canvas the divergent touch occurs (e.g., the face is canonical, but the drapery and hands belong to an assistant).
