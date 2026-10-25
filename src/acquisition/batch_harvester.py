"""
Batch Painting Harvester for High-Throughput Museum Ingestion.
Harvests high-resolution fine art paintings (up to 4,000px) from open-access
museum APIs (Cleveland Museum of Art, National Gallery of Art, etc.)
with polite concurrency, automatic resumability, and structured catalog persistence.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class HarvestedPaintingRecord:
    """Metadata record for a downloaded museum painting."""
    artwork_id: str
    title: str
    artist: str
    creation_date: str
    medium: str
    dimensions: str
    museum: str
    source_url: str
    image_url: str
    local_path: str
    width: int
    height: int
    filesize_bytes: int
    download_timestamp: float = field(default_factory=time.time)


class BatchHarvester:
    """
    Orchestrates high-resolution batch downloading of fine art paintings
    from open-access cultural institutions.
    """

    CMA_API_BASE = "https://openaccess-api.clevelandart.org/api/artworks/"

    def __init__(
        self,
        dest_dir: str | Path = "data/raw/paintings",
        catalog_path: str | Path = "data/raw/paintings_catalog.json",
        max_dimension: int = 4000,
        concurrency: int = 4,
        user_agent: str = "ArtStylometry/1.0 (academic computational stylometry)",
    ) -> None:
        self.dest_dir = Path(dest_dir)
        self.catalog_path = Path(catalog_path)
        self.max_dimension = max_dimension
        self.concurrency = concurrency
        self.user_agent = user_agent
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self._catalog: Dict[str, HarvestedPaintingRecord] = self._load_catalog()

    def _load_catalog(self) -> Dict[str, HarvestedPaintingRecord]:
        """Loads existing catalog records to ensure idempotent resuming."""
        if not self.catalog_path.exists():
            return {}
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = {}
            for item in data:
                rec = HarvestedPaintingRecord(**item)
                records[rec.artwork_id] = rec
            return records
        except Exception as e:
            logger.warning(f"Could not load existing catalog from {self.catalog_path}: {e}")
            return {}

    def _save_catalog(self) -> None:
        """Persists current catalog records to disk."""
        records_list = [asdict(rec) for rec in list(self._catalog.values())]
        temp_path = self.catalog_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(records_list, f, indent=2, ensure_ascii=False)
        temp_path.replace(self.catalog_path)

    def fetch_cma_painting_candidates(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Queries the Cleveland Museum of Art Open Access API for paintings with high-res assets.
        Paginates until the target limit is reached.
        """
        candidates: List[Dict[str, Any]] = []
        page_size = min(100, limit)
        skip = 0

        while len(candidates) < limit:
            params = {
                "type": "Painting",
                "has_image": "1",
                "limit": str(page_size),
                "skip": str(skip),
            }
            url = f"{self.CMA_API_BASE}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    data = payload.get("data", [])
                    if not data:
                        break

                    for item in data:
                        images = item.get("images", {})
                        # Prefer 'print' (~3000-4000px) or 'full' (TIF)
                        print_img = images.get("print")
                        web_img = images.get("web")
                        target_img = print_img or web_img

                        if not target_img or not target_img.get("url"):
                            continue

                        creators = item.get("creators", [])
                        artist_name = creators[0].get("description", "Unknown") if creators else "Unknown"

                        dim_val = item.get("dimensions", "Unknown")
                        if isinstance(dim_val, dict):
                            unframed = dim_val.get("unframed", {})
                            if unframed:
                                dim_str = f"{unframed.get('height', '')} x {unframed.get('width', '')} m"
                            else:
                                dim_str = str(dim_val)
                        else:
                            dim_str = str(dim_val)

                        candidates.append({
                            "artwork_id": f"cma_{item.get('id', item.get('accession_number'))}",
                            "title": item.get("title", "Untitled"),
                            "artist": artist_name,
                            "creation_date": str(item.get("creation_date", "Unknown")),
                            "medium": str(item.get("technique", "Oil on canvas")),
                            "dimensions": dim_str,
                            "museum": "Cleveland Museum of Art",
                            "source_url": item.get("url", ""),
                            "image_url": target_img["url"],
                        })

                        if len(candidates) >= limit:
                            break

                    print(f"  [+] Discovered {len(candidates)}/{limit} candidate paintings...", flush=True)
                    skip += len(data)
                    time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error fetching CMA candidates page (skip={skip}): {e}")
                break

        return candidates

    def fetch_nga_painting_candidates(self, limit: int = 4500) -> List[Dict[str, Any]]:
        """
        Loads NGA open-access paintings from local cached CSVs (data/cache/nga/objects.csv
        and data/cache/nga/published_images.csv) and resolves IIIF endpoints at max_dimension.
        """
        import pandas as pd
        obj_file = Path("data/cache/nga/objects.csv")
        img_file = Path("data/cache/nga/published_images.csv")
        if not obj_file.exists() or not img_file.exists():
            logger.warning("NGA cache CSVs not found in data/cache/nga/")
            return []

        df_obj = pd.read_csv(obj_file, low_memory=False)
        df_img = pd.read_csv(img_file, low_memory=False)

        df_paint = df_obj[df_obj["classification"].astype(str).str.lower() == "painting"]
        merged = pd.merge(df_paint, df_img, left_on="objectid", right_on="depictstmsobjectid", how="inner")
        merged = merged.drop_duplicates(subset=["objectid"])

        candidates: List[Dict[str, Any]] = []
        for _, row in merged.iterrows():
            if len(candidates) >= limit:
                break
            obj_id = str(row["objectid"])
            iiif_base = str(row["iiifurl"]).rstrip("/")
            candidates.append({
                "artwork_id": f"nga_{obj_id}",
                "title": str(row.get("title", "Untitled")),
                "artist": str(row.get("attribution", "Unknown")),
                "creation_date": str(row.get("displaydate", "Unknown")),
                "medium": str(row.get("medium", "Oil on canvas")),
                "dimensions": str(row.get("dimensions", "Unknown")),
                "museum": "National Gallery of Art, Washington",
                "source_url": f"https://www.nga.gov/collection/art-object-page.{obj_id}.html",
                "image_url": f"{iiif_base}/full/!{self.max_dimension},{self.max_dimension}/0/default.jpg",
            })

        logger.info(f"Loaded {len(candidates)} NGA painting candidates.")
        return candidates

    MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
    #: Met departments holding paintings: American Wing, Asian Art, European Paintings, Robert Lehman, Modern & Contemporary
    MET_PAINTING_DEPARTMENTS = (11, 1, 19, 21, 6)

    #: Wikidata QIDs of major museums whose public-domain paintings are mirrored on Wikimedia Commons
    WIKIDATA_MUSEUMS = {
        "Q190804": "Rijksmuseum, Amsterdam",
        "Q160236": "The Metropolitan Museum of Art",
        "Q19675": "Musée du Louvre, Paris",
        "Q160112": "Museo del Prado, Madrid",
        "Q180788": "The National Gallery, London",
        "Q51252": "Uffizi Gallery, Florence",
        "Q95569": "Kunsthistorisches Museum, Vienna",
        "Q165631": "Gemäldegalerie, Berlin",
        "Q221092": "Mauritshuis, The Hague",
        "Q132783": "State Hermitage Museum, Saint Petersburg",
        "Q154568": "Alte Pinakothek, Munich",
        "Q4692": "Gemäldegalerie Alte Meister, Dresden",
        "Q671384": "Statens Museum for Kunst, Copenhagen",
        "Q1471477": "Koninklijk Museum voor Schone Kunsten Antwerpen",
        "Q842858": "Nationalmuseum, Stockholm",
    }

    def fetch_met_painting_candidates(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Resolves public-domain Met paintings through the Collection API (department listing +
        per-object lookup). Only objects classified as paintings with a primary image are kept.
        """
        ids: List[int] = []
        for dep in self.MET_PAINTING_DEPARTMENTS:
            url = f"{self.MET_API_BASE}/objects?departmentIds={dep}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    ids.extend(json.loads(resp.read()).get("objectIDs") or [])
            except Exception as e:
                logger.error(f"Met department {dep} listing failed: {e}")

        def lookup(oid: int) -> Optional[Dict[str, Any]]:
            req = urllib.request.Request(f"{self.MET_API_BASE}/objects/{oid}", headers={"User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    o = json.loads(resp.read())
            except Exception:
                return None
            if not o.get("isPublicDomain") or not o.get("primaryImage"):
                return None
            if "painting" not in str(o.get("classification", "")).lower():
                return None
            return {
                "artwork_id": f"met_{oid}",
                "title": o.get("title", "Untitled"),
                "artist": o.get("artistDisplayName") or "Unknown",
                "creation_date": str(o.get("objectDate", "Unknown")),
                "medium": str(o.get("medium", "Unknown")),
                "dimensions": str(o.get("dimensions", "Unknown")),
                "museum": "The Metropolitan Museum of Art",
                "source_url": o.get("objectURL", ""),
                "image_url": o["primaryImage"],
            }

        candidates: List[Dict[str, Any]] = []
        print(f"  [*] Met: resolving {len(ids)} object records...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for i, c in enumerate(ex.map(lookup, ids)):
                if c:
                    candidates.append(c)
                if (i + 1) % 500 == 0:
                    print(f"  [+] Met: {i + 1}/{len(ids)} resolved, {len(candidates)} paintings", flush=True)
                if len(candidates) >= limit:
                    break
        return candidates[:limit]

    def fetch_wikidata_museum_candidates(self, limit: int = 20000) -> List[Dict[str, Any]]:
        """
        Queries Wikidata for paintings (Q3305213) with a Commons image (P18) held by major museums.
        Images are resolved through Commons ``Special:FilePath`` thumbnails at max_dimension width.
        """
        candidates: List[Dict[str, Any]] = []
        seen: set = set()
        for qid, museum in self.WIKIDATA_MUSEUMS.items():
            if len(candidates) >= limit:
                break
            q = (
                "SELECT ?item ?itemLabel ?image ?creatorLabel ?inception WHERE { "
                f"?item wdt:P31 wd:Q3305213 ; wdt:P195 wd:{qid} ; wdt:P18 ?image . "
                "OPTIONAL { ?item wdt:P170 ?creator } OPTIONAL { ?item wdt:P571 ?inception } "
                'SERVICE wikibase:label { bd:serviceParam wikibase:language "en,fr,nl,de,es,it". } }'
            )
            url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(q)
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/sparql-results+json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    rows = json.loads(resp.read())["results"]["bindings"]
            except Exception as e:
                logger.error(f"Wikidata query failed for {museum}: {e}")
                continue
            n0 = len(candidates)
            for b in rows:
                item = b["item"]["value"].rsplit("/", 1)[-1]
                if item in seen:
                    continue
                seen.add(item)
                inc = b.get("inception", {}).get("value", "")
                candidates.append({
                    "artwork_id": f"wd_{item}",
                    "title": b.get("itemLabel", {}).get("value", item),
                    "artist": b.get("creatorLabel", {}).get("value", "Unknown"),
                    "creation_date": inc[:4].lstrip("+") if inc else "Unknown",
                    "medium": "Unknown",
                    "dimensions": "Unknown",
                    "museum": museum,
                    "source_url": b["item"]["value"],
                    "image_url": b["image"]["value"].replace("http://", "https://") + f"?width={min(self.max_dimension, 3840)}",
                })
            print(f"  [+] Wikidata {museum}: {len(candidates) - n0} paintings (total {len(candidates)})", flush=True)
            time.sleep(1.0)
        return candidates[:limit]

    def fetch_all_candidates(self, limit: int = 10000, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Discovers candidates across multiple open-access institutions (NGA, CMA, Met, Wikidata museums).
        """
        if sources is None:
            sources = ["nga", "cma"]

        all_cands: List[Dict[str, Any]] = []
        for src in sources:
            if len(all_cands) >= limit:
                break
            remaining = limit - len(all_cands)
            src = src.lower()
            if src == "cma":
                cands = self.fetch_cma_painting_candidates(limit=remaining)
            elif src == "nga":
                cands = self.fetch_nga_painting_candidates(limit=remaining)
            elif src == "met":
                cands = self.fetch_met_painting_candidates(limit=remaining)
            elif src == "wikidata":
                cands = self.fetch_wikidata_museum_candidates(limit=remaining)
            else:
                continue
            all_cands.extend(cands)

        return all_cands[:limit]

    def get_dest_dir_size_bytes(self) -> int:
        """Computes current size in bytes of all files in destination directory."""
        if not self.dest_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self.dest_dir.glob("*.jpg") if f.is_file())

    def download_painting(
        self,
        candidate: Dict[str, Any],
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> Optional[HarvestedPaintingRecord]:
        """
        Downloads a single high-resolution painting, ensures valid JPEG format,
        rescales if necessary to max_dimension, and logs to the catalog.
        """
        artwork_id = candidate["artwork_id"]
        safe_filename = f"{artwork_id}.jpg"
        out_path = self.dest_dir / safe_filename

        # If already recorded in catalog and exists on disk with valid size, skip
        if artwork_id in self._catalog and out_path.exists() and out_path.stat().st_size > 50000:
            return self._catalog[artwork_id]

        img_url = candidate["image_url"]
        req = urllib.request.Request(
            img_url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8",
            },
        )

        try:
            data = b""
            delay = 5.0
            for _attempt in range(6):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    break
                except urllib.error.HTTPError as http_err:
                    # Polite exponential back-off for rate-limited hosts (e.g. Wikimedia Commons)
                    if http_err.code in (429, 503):
                        time.sleep(delay)
                        delay = min(delay * 2, 120.0)
                        continue
                    raise

            if len(data) < 10000:
                logger.warning(f"Downloaded payload too small ({len(data)} bytes) for {artwork_id}")
                return None

            # Open with PIL to verify integrity and rescale if needed
            import io
            with Image.open(io.BytesIO(data)) as pil_img:
                pil_img.verify()

            with Image.open(io.BytesIO(data)) as pil_img:
                pil_rgb = pil_img.convert("RGB")
                orig_w, orig_h = pil_rgb.size

                # Resize if exceeds max_dimension
                if max(orig_w, orig_h) > self.max_dimension:
                    scale = self.max_dimension / max(orig_w, orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    pil_rgb = pil_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    new_w, new_h = orig_w, orig_h

                pil_rgb.save(out_path, format="JPEG", quality=92, optimize=True)

            filesize = out_path.stat().st_size
            record = HarvestedPaintingRecord(
                artwork_id=artwork_id,
                title=candidate["title"],
                artist=candidate["artist"],
                creation_date=candidate["creation_date"],
                medium=candidate["medium"],
                dimensions=candidate["dimensions"],
                museum=candidate["museum"],
                source_url=candidate["source_url"],
                image_url=img_url,
                local_path=str(out_path.resolve()),
                width=new_w,
                height=new_h,
                filesize_bytes=filesize,
            )

            self._catalog[artwork_id] = record
            return record

        except Exception as e:
            logger.error(f"Failed to harvest {artwork_id} from {img_url}: {e}")
            return None

    def harvest_batch(
        self,
        total_limit: int = 1000,
        source: str = "all",
        target_gb: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, HarvestedPaintingRecord], None]] = None,
    ) -> List[HarvestedPaintingRecord]:
        """
        Executes multi-threaded batch ingestion up to total_limit or target_gb disk quota.
        Periodically saves the catalog every 10 downloads.
        """
        print(f"[*] Discovering public-domain painting candidates (source: {source}, target: {total_limit})...", flush=True)
        if source == "cma":
            candidates = self.fetch_cma_painting_candidates(limit=total_limit)
        elif source == "nga":
            candidates = self.fetch_nga_painting_candidates(limit=total_limit)
        elif source == "met":
            candidates = self.fetch_met_painting_candidates(limit=total_limit)
        elif source == "wikidata":
            candidates = self.fetch_wikidata_museum_candidates(limit=total_limit)
        elif source == "extended":
            candidates = self.fetch_all_candidates(limit=total_limit, sources=["met", "wikidata"])
        else:
            candidates = self.fetch_all_candidates(limit=total_limit, sources=["nga", "cma"])

        print(f"[+] Found {len(candidates)} candidates ready for ingestion.", flush=True)

        successful_records: List[HarvestedPaintingRecord] = []
        completed_count = 0

        # Check existing
        already_done = 0
        to_download = []
        for c in candidates:
            aid = c["artwork_id"]
            local_f = self.dest_dir / f"{aid}.jpg"
            if aid in self._catalog and local_f.exists() and local_f.stat().st_size > 50000:
                already_done += 1
                successful_records.append(self._catalog[aid])
            else:
                to_download.append(c)

        current_bytes = self.get_dest_dir_size_bytes()
        current_gb = current_bytes / (1024 ** 3)
        print(f"[*] Already harvested in local cache: {already_done} / {len(candidates)} (Current corpus size: {current_gb:.2f} GB)", flush=True)
        print(f"[*] Pending downloads: {len(to_download)} (concurrency: {self.concurrency})", flush=True)

        if not to_download:
            return successful_records

        target_bytes = (target_gb * (1024 ** 3)) if target_gb else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_cand = {executor.submit(self.download_painting, cand): cand for cand in to_download}

            for future in concurrent.futures.as_completed(future_to_cand):
                cand = future_to_cand[future]
                completed_count += 1
                try:
                    record = future.result()
                    if record:
                        successful_records.append(record)
                        current_bytes += record.filesize_bytes
                        current_gb = current_bytes / (1024 ** 3)
                        if progress_callback:
                            progress_callback(len(successful_records), len(candidates), record)
                        else:
                            mb = record.filesize_bytes / (1024 * 1024)
                            safe_artist = record.artist[:20].encode("ascii", "replace").decode("ascii")
                            safe_title = record.title[:25].encode("ascii", "replace").decode("ascii")
                            print(
                                f"  [{len(successful_records):4d}/{len(candidates)}] Harvested: {safe_artist} - '{safe_title}' "
                                f"({record.width}x{record.height}px, {mb:.1f} MB, Total: {current_gb:.2f} GB)",
                                flush=True,
                            )

                        if target_bytes and current_bytes >= target_bytes:
                            print(f"\n[!] Target storage quota of {target_gb:.1f} GB reached ({current_gb:.2f} GB on disk). Pausing further downloads.", flush=True)
                            for f in future_to_cand:
                                f.cancel()
                            break
                except Exception as exc:
                    logger.error(f"Task generated an exception for {cand['artwork_id']}: {exc}")

                # Persist catalog every 10 records
                if completed_count % 10 == 0:
                    self._save_catalog()

        # Final catalog flush
        self._save_catalog()
        print(f"\n[OK] Batch harvest complete! {len(successful_records)} paintings saved to: {self.dest_dir} ({current_gb:.2f} GB)", flush=True)
        print(f"[OK] Master catalog updated: {self.catalog_path}", flush=True)
        return successful_records
