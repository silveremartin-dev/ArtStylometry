"""
Contested Masters Corpus Harvester.
Targets high-resolution museum paintings for historical masters with workshop
ambiguity, student interventions, and connoisseurship controversies:
- Johannes Vermeer
- Rembrandt van Rijn
- Frans Hals
- Lucas Cranach the Elder
- Leonardo da Vinci
"""

from __future__ import annotations

import concurrent.futures
import csv
import io
import json
import logging
import socket
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

# Global socket timeout to avoid hung connections
socket.setdefaulttimeout(25.0)

logger = logging.getLogger(__name__)


@dataclass
class ContestedArtworkRecord:
    """Detailed metadata for a painting in a contested master's corpus."""
    artwork_id: str
    artist_canonical: str
    title: str
    official_attribution: str
    attribution_category: str  # 'autograph', 'workshop', 'follower', 'imitator_fake'
    creation_date: str
    medium: str
    dimensions: str
    museum: str
    image_url: str
    local_path: str
    width: int
    height: int
    filesize_bytes: int


LEONARDO_CURATED_HOLDINGS: List[Dict[str, Any]] = [
    {
        "artwork_id": "leo_mona_lisa",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Mona Lisa (La Gioconda)",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1503–1519",
        "medium": "Oil on poplar panel",
        "dimensions": "77 x 53 cm",
        "museum": "Musée du Louvre, Paris",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Mona%20Lisa%2C%20by%20Leonardo%20da%20Vinci%2C%20from%20C2RMF%20retouched.jpg?width=1280",
    },
    {
        "artwork_id": "leo_virgin_rocks_louvre",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Virgin of the Rocks (Louvre version)",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "1483–1486",
        "medium": "Oil on panel transferred to canvas",
        "dimensions": "199 x 122 cm",
        "museum": "Musée du Louvre, Paris",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%20-%20Virgin%20of%20the%20Rocks%20%28Louvre%29.jpg?width=1280",
    },
    {
        "artwork_id": "leo_lady_ermine",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Lady with an Ermine (Cecilia Gallerani)",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1489–1490",
        "medium": "Oil on walnut panel",
        "dimensions": "54 x 39 cm",
        "museum": "Czartoryski Museum, Kraków",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Lady%20with%20an%20Ermine%20-%20Leonardo%20da%20Vinci%20-%20Google%20Art%20Project.jpg?width=1280",
    },
    {
        "artwork_id": "leo_st_john_baptist",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Saint John the Baptist",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1513–1516",
        "medium": "Oil on walnut wood",
        "dimensions": "69 x 57 cm",
        "museum": "Musée du Louvre, Paris",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%20-%20Saint%20John%20the%20Baptist%20C2RMF%20retouched.jpg?width=1280",
    },
    {
        "artwork_id": "leo_la_belle_ferronniere",
        "artist_canonical": "Leonardo da Vinci",
        "title": "La Belle Ferronnière",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1490–1496",
        "medium": "Oil on walnut panel",
        "dimensions": "62 x 44 cm",
        "museum": "Musée du Louvre, Paris",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/La%20Belle%20Ferronni%C3%A8re.jpg?width=1280",
    },
    {
        "artwork_id": "leo_annunciation",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Annunciation",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1472–1475",
        "medium": "Oil and tempera on poplar panel",
        "dimensions": "98 x 217 cm",
        "museum": "Uffizi Gallery, Florence",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Annunciation%20%28Leonardo%29.jpg?width=1280",
    },
    {
        "artwork_id": "leo_st_anne",
        "artist_canonical": "Leonardo da Vinci",
        "title": "The Virgin and Child with Saint Anne",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1503–1519",
        "medium": "Oil on wood",
        "dimensions": "168 x 130 cm",
        "museum": "Musée du Louvre, Paris",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%20-%20Virgin%20and%20Child%20with%20St%20Anne%20C2RMF%20retouched.jpg?width=1280",
    },
    {
        "artwork_id": "leo_portrait_musician",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Portrait of a Musician",
        "official_attribution": "Leonardo da Vinci",
        "attribution_category": "autograph",
        "creation_date": "c. 1483–1487",
        "medium": "Oil on walnut panel",
        "dimensions": "44.7 x 32 cm",
        "museum": "Pinacoteca Ambrosiana, Milan",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%20-%20Portrait%20of%20a%20Musician.jpg?width=1280",
    },
    {
        "artwork_id": "leo_baptism_christ",
        "artist_canonical": "Leonardo da Vinci",
        "title": "The Baptism of Christ",
        "official_attribution": "Andrea del Verrocchio and Leonardo da Vinci",
        "attribution_category": "workshop",
        "creation_date": "c. 1472–1475",
        "medium": "Oil and tempera on wood",
        "dimensions": "177 x 151 cm",
        "museum": "Uffizi Gallery, Florence",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Verrocchio%2C%20Leonardo%20da%20Vinci%20-%20Battesimo%20di%20Cristo%20-%20Google%20Art%20Project.jpg?width=1280",
    },
    {
        "artwork_id": "leo_virgin_rocks_london",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Virgin of the Rocks (London version)",
        "official_attribution": "Leonardo da Vinci and Workshop (Ambrogio de Predis)",
        "attribution_category": "workshop",
        "creation_date": "c. 1495–1508",
        "medium": "Oil on poplar panel",
        "dimensions": "189.5 x 120 cm",
        "museum": "The National Gallery, London",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%20Virgin%20of%20the%20Rocks%20%28National%20Gallery%20London%29.jpg?width=1280",
    },
    {
        "artwork_id": "leo_bacchus",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Bacchus (Saint John in the Wilderness)",
        "official_attribution": "Workshop of Leonardo da Vinci (Cesare da Sesto)",
        "attribution_category": "workshop",
        "creation_date": "c. 1510–1515",
        "medium": "Oil on panel transferred to canvas",
        "dimensions": "177 x 115 cm",
        "museum": "Musée du Louvre, Paris",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bacchus%20-%20Leonardo%20da%20Vinci%20workshop%20-%20Louvre%20INV%20780.jpg?width=1280",
    },
    {
        "artwork_id": "leo_madonna_litta",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Madonna Litta",
        "official_attribution": "Attributed to Leonardo da Vinci and Giovanni Antonio Boltraffio",
        "attribution_category": "workshop",
        "creation_date": "c. 1490",
        "medium": "Tempera on canvas (transferred from wood)",
        "dimensions": "42 x 33 cm",
        "museum": "State Hermitage Museum, Saint Petersburg",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%20attributed%20-%20Madonna%20Litta.jpg?width=1280",
    },
    {
        "artwork_id": "leo_salvator_mundi",
        "artist_canonical": "Leonardo da Vinci",
        "title": "Salvator Mundi",
        "official_attribution": "Attributed to Leonardo da Vinci (contested workshop intervention)",
        "attribution_category": "imitator_fake",
        "creation_date": "c. 1500",
        "medium": "Oil on walnut panel",
        "dimensions": "65.6 x 45.4 cm",
        "museum": "Private Collection / Abu Dhabi (contested)",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo%20da%20Vinci%2C%20Salvator%20Mundi%2C%20c.1500%2C%20oil%20on%20walnut%2C%2045.4%20%C3%97%2065.6%20cm.jpg?width=1280",
    },
    {
        "artwork_id": "leo_bella_principessa",
        "artist_canonical": "Leonardo da Vinci",
        "title": "La Bella Principessa (Portrait of Bianca Sforza)",
        "official_attribution": "Attributed to Leonardo da Vinci (disputed modern forgery)",
        "attribution_category": "imitator_fake",
        "creation_date": "c. 1495 (contested 19th/20th century)",
        "medium": "Chalk and ink on vellum",
        "dimensions": "33 x 24 cm",
        "museum": "Private Collection (contested)",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Profile%20of%20a%20Young%20Fiancee%20-%20da%20Vinci.jpg?width=1280",
    },
    {
        "artwork_id": "leo_isleworth_mona_lisa",
        "artist_canonical": "Leonardo da Vinci",
        "title": "The Isleworth Mona Lisa",
        "official_attribution": "Attributed to Leonardo da Vinci (contested early version or copy)",
        "attribution_category": "imitator_fake",
        "creation_date": "c. 1503–1516",
        "medium": "Oil on canvas",
        "dimensions": "86 x 64.5 cm",
        "museum": "Mona Lisa Foundation (contested)",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/The%20Isleworth%20Mona%20Lisa.jpg?width=1280",
    },
]

# Hand-curated holdings for works absent from NGA/CMA open data.
# Catalogue-raisonné-scale corpora for all other masters are resolved
# authoritatively through Wikidata (see ``wikidata_corpus.py``).
CURATED_HOLDINGS_BY_ARTIST: Dict[str, List[Dict[str, Any]]] = {
    "leonardo": LEONARDO_CURATED_HOLDINGS,
}


class ContestedMastersHarvester:
    """Harvests complete mono-artist painting corpora from NGA, CMA, and curated archives."""

    NGA_OBJECTS_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv"
    NGA_IMAGES_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"
    CMA_API_BASE = "https://openaccess-api.clevelandart.org/api/artworks/"

    ARTIST_CONFIGS = {
        "vermeer": {
            "name": "Johannes Vermeer",
            "nga_query": "vermeer",
            "cma_query": "Vermeer",
        },
        "rembrandt": {
            "name": "Rembrandt van Rijn",
            "nga_query": "rembrandt",
            "cma_query": "Rembrandt",
        },
        "hals": {
            "name": "Frans Hals",
            "nga_query": "hals",
            "cma_query": "Frans Hals",
        },
        "cranach": {
            "name": "Lucas Cranach the Elder",
            "nga_query": "cranach",
            "cma_query": "Cranach",
        },
        "leonardo": {
            "name": "Leonardo da Vinci",
            "nga_query": "vinci",
            "cma_query": "Leonardo da Vinci",
        },
        "rubens": {
            "name": "Peter Paul Rubens",
            "nga_query": "rubens",
            "cma_query": "Rubens",
        },
        "titian": {
            "name": "Titian (Tiziano Vecellio)",
            "nga_query": "titian",
            "cma_query": "Titian",
        },
        "greco": {
            "name": "El Greco",
            "nga_query": "greco",
            "cma_query": "El Greco",
        },
        "raphael": {
            "name": "Raphael (Raffaello Sanzio)",
            "nga_query": "raphael",
            "cma_query": "Raphael",
            "exclude": ["peale", "soyer", "mengs"],
        },
        "goya": {
            "name": "Francisco de Goya",
            "nga_query": "goya",
            "cma_query": "Goya",
        },
        "velazquez": {
            "name": "Diego Velázquez",
            "nga_query": "diego vel",
            "cma_query": "Velázquez",
        },
        "bosch": {
            "name": "Hieronymus Bosch",
            "nga_query": "hieronymus bosch",
            "cma_query": "Bosch",
        },
    }

    def __init__(
        self,
        base_dir: str | Path = "data/raw/contested_masters",
        max_dimension: int = 4000,
        concurrency: int = 4,
        user_agent: str = "ArtStylometry/1.0 (academic connoisseurship; https://github.com/ArtStylometry)",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.max_dimension = max_dimension
        self.concurrency = concurrency
        self.user_agent = user_agent
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path("data/cache/nga")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def categorize_attribution(raw_attr: str) -> str:
        """Classifies official museum attribution into connoisseurship tiers."""
        attr_lower = raw_attr.lower()
        if any(w in attr_lower for w in ["imitator", "fake", "forgery", "after", "copy", "style of"]):
            return "imitator_fake"
        elif any(w in attr_lower for w in ["workshop", "studio", "circle", "pupil", "possibly", "attributed to", " and ", "collaborat"]):
            return "workshop"
        elif any(w in attr_lower for w in ["follower", "school of"]):
            return "follower"
        else:
            return "autograph"

    def _ensure_nga_cache(self) -> tuple[Path, Path]:
        """Ensures NGA objects and images CSV files are cached locally."""
        obj_file = self.cache_dir / "objects.csv"
        img_file = self.cache_dir / "published_images.csv"

        if not obj_file.exists() or obj_file.stat().st_size == 0:
            print("  [*] Downloading and caching NGA objects database...", flush=True)
            req = urllib.request.Request(self.NGA_OBJECTS_URL, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=45) as r:
                obj_file.write_bytes(r.read())

        if not img_file.exists() or img_file.stat().st_size == 0:
            print("  [*] Downloading and caching NGA images database...", flush=True)
            req = urllib.request.Request(self.NGA_IMAGES_URL, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=60) as r:
                img_file.write_bytes(r.read())

        return obj_file, img_file

    def fetch_nga_candidates_for_artist(self, artist_key: str) -> List[Dict[str, Any]]:
        """Harvests candidate painting records and IIIF URLs from NGA OpenData."""
        cfg = self.ARTIST_CONFIGS[artist_key]
        query = cfg["nga_query"]

        obj_file, img_file = self._ensure_nga_cache()

        # 1. Fetch objects from cache
        with open(obj_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matched_objs = {}
            for row in reader:
                attr = row.get("attribution", "").lower()
                cls = row.get("classification", "").lower()
                if "painting" not in cls:
                    continue
                # Specific filter to exclude American painter Peale family members (Rembrandt Peale, Titian Ramsay Peale)
                if artist_key in ("rembrandt", "titian") and "peale" in attr:
                    continue
                if any(ex in attr for ex in cfg.get("exclude", [])):
                    continue
                if query in attr:
                    matched_objs[row["objectid"]] = row

        if not matched_objs:
            return []

        # 2. Fetch images from cache
        matched_images = {}
        with open(img_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = row.get("depictstmsobjectid")
                if oid in matched_objs and oid not in matched_images:
                    matched_images[oid] = row.get("iiifurl")

        candidates = []
        for oid, obj in matched_objs.items():
            iiif_base = matched_images.get(oid)
            if not iiif_base:
                continue

            attr = obj.get("attribution", cfg["name"])
            category = self.categorize_attribution(attr)
            image_url = f"{iiif_base}/full/!{self.max_dimension},{self.max_dimension}/0/default.jpg"

            candidates.append({
                "artwork_id": f"nga_{oid}",
                "artist_canonical": cfg["name"],
                "title": obj.get("title", "Untitled"),
                "official_attribution": attr,
                "attribution_category": category,
                "creation_date": obj.get("displaydate", "Unknown"),
                "medium": obj.get("medium", "Oil on canvas"),
                "dimensions": obj.get("dimensions", "Unknown"),
                "museum": "National Gallery of Art, Washington D.C.",
                "image_url": image_url,
            })

        return candidates

    def fetch_cma_candidates_for_artist(self, artist_key: str) -> List[Dict[str, Any]]:
        """Harvests candidate painting records from Cleveland Museum of Art API."""
        cfg = self.ARTIST_CONFIGS[artist_key]
        query = cfg["cma_query"]

        params = {
            "q": query,
            "type": "Painting",
            "has_image": "1",
            "limit": "50",
        }
        url = f"{self.CMA_API_BASE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        candidates = []
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                payload = json.loads(r.read().decode("utf-8"))
                for item in payload.get("data", []):
                    creators = item.get("creators", [])
                    raw_attr = creators[0].get("description", "Unknown") if creators else "Unknown"
                    if query.lower() not in raw_attr.lower():
                        continue
                    if artist_key in ("rembrandt", "titian") and "peale" in raw_attr.lower():
                        continue
                    if any(ex in raw_attr.lower() for ex in cfg.get("exclude", [])):
                        continue

                    images = item.get("images", {})
                    target_img = images.get("print") or images.get("web")
                    if not target_img or not target_img.get("url"):
                        continue

                    dim_val = item.get("dimensions", "Unknown")
                    if isinstance(dim_val, dict):
                        unframed = dim_val.get("unframed", {})
                        dim_str = f"{unframed.get('height', '')} x {unframed.get('width', '')} m" if unframed else str(dim_val)
                    else:
                        dim_str = str(dim_val)

                    candidates.append({
                        "artwork_id": f"cma_{item.get('id', item.get('accession_number'))}",
                        "artist_canonical": cfg["name"],
                        "title": item.get("title", "Untitled"),
                        "official_attribution": raw_attr,
                        "attribution_category": self.categorize_attribution(raw_attr),
                        "creation_date": str(item.get("creation_date", "Unknown")),
                        "medium": str(item.get("technique", "Oil on canvas")),
                        "dimensions": dim_str,
                        "museum": "Cleveland Museum of Art",
                        "image_url": target_img["url"],
                    })
        except Exception as e:
            logger.error(f"Error querying CMA for {artist_key}: {e}")

        return candidates

    def download_image(self, candidate: Dict[str, Any], artist_dir: Path) -> Optional[ContestedArtworkRecord]:
        """Downloads and rescales a single high-resolution painting."""
        artwork_id = candidate["artwork_id"]
        out_file = artist_dir / f"{artwork_id}.jpg"

        if out_file.exists() and out_file.stat().st_size > 30000:
            try:
                with Image.open(out_file) as img:
                    w, h = img.size
                return ContestedArtworkRecord(
                    artwork_id=artwork_id,
                    artist_canonical=candidate["artist_canonical"],
                    title=candidate["title"],
                    official_attribution=candidate["official_attribution"],
                    attribution_category=candidate["attribution_category"],
                    creation_date=candidate["creation_date"],
                    medium=candidate["medium"],
                    dimensions=candidate["dimensions"],
                    museum=candidate["museum"],
                    image_url=candidate["image_url"],
                    local_path=str(out_file.resolve()),
                    width=w,
                    height=h,
                    filesize_bytes=out_file.stat().st_size,
                )
            except Exception:
                pass  # Corrupt file, re-download below

        req = urllib.request.Request(
            candidate["image_url"],
            headers={
                "User-Agent": self.user_agent,
                "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw_data = r.read()

            if len(raw_data) < 10000:
                return None

            with Image.open(io.BytesIO(raw_data)) as pil_img:
                pil_rgb = pil_img.convert("RGB")
                orig_w, orig_h = pil_rgb.size

                if max(orig_w, orig_h) > self.max_dimension:
                    scale = self.max_dimension / max(orig_w, orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    pil_rgb = pil_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    new_w, new_h = orig_w, orig_h

                pil_rgb.save(out_file, format="JPEG", quality=92, optimize=True)

            return ContestedArtworkRecord(
                artwork_id=artwork_id,
                artist_canonical=candidate["artist_canonical"],
                title=candidate["title"],
                official_attribution=candidate["official_attribution"],
                attribution_category=candidate["attribution_category"],
                creation_date=candidate["creation_date"],
                medium=candidate["medium"],
                dimensions=candidate["dimensions"],
                museum=candidate["museum"],
                image_url=candidate["image_url"],
                local_path=str(out_file.resolve()),
                width=new_w,
                height=new_h,
                filesize_bytes=out_file.stat().st_size,
            )
        except Exception as e:
            logger.warning(f"Error downloading {artwork_id}: {e}")
            return None

    def harvest_corpus_for_artist(self, artist_key: str) -> List[ContestedArtworkRecord]:
        """Harvests the complete painting corpus for a target master."""
        cfg = self.ARTIST_CONFIGS[artist_key]
        artist_dir = self.base_dir / artist_key
        artist_dir.mkdir(parents=True, exist_ok=True)
        catalog_file = artist_dir / "catalog.json"

        print(f"\n[{artist_key.upper()}] Discovering holdings for {cfg['name']}...", flush=True)
        candidates_nga = self.fetch_nga_candidates_for_artist(artist_key)
        candidates_cma = self.fetch_cma_candidates_for_artist(artist_key)
        candidates = candidates_nga + candidates_cma

        curated = CURATED_HOLDINGS_BY_ARTIST.get(artist_key, [])
        seen_ids = {c["artwork_id"] for c in candidates}
        candidates.extend(c for c in curated if c["artwork_id"] not in seen_ids)

        print(f"  [+] Found {len(candidates)} paintings (NGA: {len(candidates_nga)}, CMA: {len(candidates_cma)}, Curated: {len(curated)})", flush=True)

        records: List[ContestedArtworkRecord] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_cand = {executor.submit(self.download_image, c, artist_dir): c for c in candidates}
            for future in concurrent.futures.as_completed(future_to_cand):
                cand = future_to_cand[future]
                try:
                    rec = future.result(timeout=25)
                    if rec:
                        records.append(rec)
                        mb = rec.filesize_bytes / (1024 * 1024)
                        safe_title = rec.title[:25].encode("ascii", "replace").decode("ascii")
                        print(f"    [{len(records):2d}/{len(candidates)}] [{rec.attribution_category.upper():12s}] {safe_title} ({rec.width}x{rec.height}px, {mb:.1f}MB)", flush=True)
                except Exception as e:
                    logger.warning(f"Download task exception for {cand['artwork_id']}: {e}")

        # Save catalog
        with open(catalog_file, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)

        print(f"  [OK] Saved {len(records)} paintings to: {artist_dir}", flush=True)
        print(f"  [OK] Metadata catalog: {catalog_file}", flush=True)
        return records

    def harvest_all_contested_masters(self) -> Dict[str, List[ContestedArtworkRecord]]:
        """Harvests corpora for all configured contested masters."""
        all_corpora = {}
        for key in self.ARTIST_CONFIGS:
            all_corpora[key] = self.harvest_corpus_for_artist(key)
        return all_corpora

