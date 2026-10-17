# Museum IIIF Harvesting Guide

## 1. What is IIIF?

The **International Image Interoperability Framework (IIIF)** is an open standard adopted by major global museums, national libraries, and research archives. It provides uniform APIs to query artwork metadata, level-of-detail image pyramids, and gigapixel deep-zoom image tiles.

By leveraging IIIF endpoints, ArtStylometry accesses raw, native-resolution museum image tiles without requiring proprietary commercial databases or third-party image licensing.

---

## 2. Supported Museum Collections & Endpoints

| Museum / Institution | IIIF Presentation Manifest URL Pattern | IIIF Image API Base URL |
| :--- | :--- | :--- |
| **Rijksmuseum (Amsterdam)** | `https://data.rijksmuseum.nl/object/{OBJECT_ID}/manifest` | `https://lh3.googleusercontent.com/...` (via info.json) |
| **National Gallery (London)** | `https://media.nationalgallery.org.uk/iiif/{ID}/manifest.json` | `https://media.nationalgallery.org.uk/iiif/{ID}` |
| **The Metropolitan Museum (NYC)** | Public Domain Open Access / IIIF manifests | `https://images.metmuseum.org/...` |
| **Art Institute of Chicago** | `https://api.artic.edu/api/v1/artworks/{ID}/manifest.json` | `https://www.artic.edu/iiif/2/{IMAGE_ID}` |
| **BnF / Gallica (Paris)** | `https://gallica.bnf.fr/iiif/ark:/12148/{ARK_ID}/manifest.json` | `https://gallica.bnf.fr/iiif/ark:/12148/{ARK_ID}/f1` |

---

## 3. Harvesting Workflow

ArtStylometry includes a dedicated harvester tool in `scripts/harvest_iiif.py`:

```bash
# Harvest a high-resolution painting from a museum IIIF manifest
python scripts/harvest_iiif.py \
  --url "https://api.artic.edu/api/v1/artworks/27992/manifest.json" \
  --out "data/raw/sunday_afternoon_seurat.jpg" \
  --max-dimension 6000
```

### Direct Image API `info.json` Query
If you have the direct Image API endpoint (ending in `info.json`):

```bash
python scripts/harvest_iiif.py \
  --url "https://media.nationalgallery.org.uk/iiif/NG6376/info.json" \
  --out "data/raw/rembrandt_self_portrait.jpg" \
  --max-dimension 8000
```

---

## 4. Tile Resolution & Level-of-Detail Mathematics

A IIIF Image Service exposes multi-resolution scale factors:
$$\text{Scale Factors} = \{1, 2, 4, 8, 16\}$$
* **Scale Factor 1**: Native sensor pixel resolution (100% full scale). Slices of $256 \times 256$ pixels directly capture canvas weave, paint layers, and micro-craquelure.
* **Scale Factor 2 or 4**: Half or quarter scale. Used for rapid overview and coarse macro-compositional layout.

The harvester script automatically evaluates the requested `--max-dimension`, computes the optimal scale factor, downloads the required rectangular grid of tiles with local caching in `data/cache/tiles/`, and stitches them seamlessly into the output file.
