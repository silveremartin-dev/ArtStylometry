"""
Wikidata / Wikimedia Commons Catalogue-Raisonné Harvester.

Builds near-exhaustive mono-artist painting corpora by querying the Wikidata
SPARQL endpoint for every painting (Q3305213) linked to a master, including
attribution qualifiers that encode connoisseurship status:

    P170 creator (no qualifier)            -> 'autograph'
    P170 creator + P5102 nature-of-claim   -> 'workshop'   (attributed / presumably / disputed)
    P170 qualifier P1773 attributed to     -> 'workshop'
    P170 qualifier P1774 workshop of       -> 'workshop'
    P170 qualifier P1780 school of         -> 'follower'
    P170 qualifier P1777 manner of         -> 'imitator_fake'
    P1877 after a work by                  -> 'imitator_fake' (copy)

Images are resolved through Wikimedia Commons ``Special:FilePath`` (P18).
Downloads are polite (low concurrency, exponential back-off on HTTP 429)
and idempotent (existing files are reused).
"""

from __future__ import annotations

import io
import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .contested_masters_harvester import ContestedArtworkRecord

Image.MAX_IMAGE_PIXELS = None  # Commons originals can exceed PIL's decompression-bomb guard

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

#: Wikidata QIDs of the masters audited by ArtStylometry.
MASTER_QIDS: Dict[str, Dict[str, str]] = {
    "vermeer": {"qid": "Q41264", "name": "Johannes Vermeer"},
    "rembrandt": {"qid": "Q5598", "name": "Rembrandt van Rijn"},
    "hals": {"qid": "Q167654", "name": "Frans Hals"},
    "cranach": {"qid": "Q191748", "name": "Lucas Cranach the Elder"},
    "leonardo": {"qid": "Q762", "name": "Leonardo da Vinci"},
    "rubens": {"qid": "Q5599", "name": "Peter Paul Rubens"},
    "titian": {"qid": "Q47551", "name": "Titian (Tiziano Vecellio)"},
    "greco": {"qid": "Q301", "name": "El Greco"},
    "raphael": {"qid": "Q5597", "name": "Raphael (Raffaello Sanzio)"},
    "goya": {"qid": "Q5432", "name": "Francisco de Goya"},
    "velazquez": {"qid": "Q297", "name": "Diego Velázquez"},
    "bosch": {"qid": "Q130531", "name": "Hieronymus Bosch"},
}

_CATEGORY_RANK = {"autograph": 0, "workshop": 1, "follower": 2, "imitator_fake": 3}

SPARQL_FAST_TEMPLATE = """
SELECT ?item ?itemLabel ?image ?collLabel ?inception ?rel WHERE {{
  ?item wdt:P31 wd:Q3305213 ;
        wdt:P18 ?image ;
        wdt:P170 wd:{qid} .
  BIND("autograph" AS ?rel)
  OPTIONAL {{ ?item wdt:P195 ?coll }}
  OPTIONAL {{ ?item wdt:P571 ?inception }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr,de,es,it,nl". }}
}}
"""

SPARQL_TEMPLATE = """
SELECT ?item ?itemLabel ?image ?collLabel ?inception ?inv ?rel WHERE {{
  {{ ?item p:P170 ?st . ?st ps:P170 wd:{qid} .
     OPTIONAL {{ ?st pq:P5102 ?nature }}
     BIND(IF(BOUND(?nature), "workshop", "autograph") AS ?rel) }}
  UNION {{ ?item p:P170 ?st . ?st pq:P1773 wd:{qid} . BIND("workshop" AS ?rel) }}
  UNION {{ ?item p:P170 ?st . ?st pq:P1774 wd:{qid} . BIND("workshop" AS ?rel) }}
  UNION {{ ?item p:P170 ?st . ?st pq:P1780 wd:{qid} . BIND("follower" AS ?rel) }}
  UNION {{ ?item p:P170 ?st . ?st pq:P1777 wd:{qid} . BIND("imitator_fake" AS ?rel) }}
  UNION {{ ?item wdt:P1877 wd:{qid} . BIND("imitator_fake" AS ?rel) }}
  ?item wdt:P31 wd:Q3305213 ; wdt:P18 ?image .
  OPTIONAL {{ ?item wdt:P195 ?coll }}
  OPTIONAL {{ ?item wdt:P571 ?inception }}
  OPTIONAL {{ ?item wdt:P217 ?inv }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr,de,es,it,nl". }}
}}
"""


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


class WikidataCorpusHarvester:
    """Harvests catalogue-raisonné-scale painting corpora from Wikidata + Commons."""

    def __init__(
        self,
        base_dir: str | Path = "data/raw/contested_masters",
        max_dimension: int = 4000,
        concurrency: int = 2,
        user_agent: str = "ArtStylometry/1.0 (academic computational stylometry; https://github.com/silveremartin-dev/ArtStylometry)",
        thumb_width: int = 3840,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.max_dimension = max_dimension
        self.concurrency = concurrency
        self.user_agent = user_agent
        self.thumb_width = thumb_width
        self._throttle_lock = threading.Lock()
        self._next_allowed = 0.0

    # ------------------------------------------------------------------ query
    def query_artist(self, artist_key: str) -> List[Dict[str, Any]]:
        """Returns one deduplicated candidate per Wikidata painting item."""
        cfg = MASTER_QIDS[artist_key]
        for tmpl in [SPARQL_FAST_TEMPLATE, SPARQL_TEMPLATE]:
            query = tmpl.format(qid=cfg["qid"])
            url = f"{SPARQL_ENDPOINT}?format=json&query={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/sparql-results+json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    rows = json.loads(r.read())["results"]["bindings"]
                return self.rows_to_candidates(rows, artist_key)
            except Exception as e:
                logger.warning(f"Query attempt with template failed for {artist_key}: {e}. Retrying with fallback template...")
        return []

    @staticmethod
    def rows_to_candidates(rows: List[Dict[str, Any]], artist_key: str) -> List[Dict[str, Any]]:
        """Collapses SPARQL rows into candidates, keeping the most autograph-leaning category."""
        cfg = MASTER_QIDS[artist_key]
        by_item: Dict[str, Dict[str, Any]] = {}
        for b in rows:
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            rel = b.get("rel", {}).get("value", "autograph")
            title = b.get("itemLabel", {}).get("value", qid)
            existing = by_item.get(qid)
            if existing and _CATEGORY_RANK[existing["attribution_category"]] <= _CATEGORY_RANK[rel]:
                continue
            inception = b.get("inception", {}).get("value", "")
            year = inception[:4].lstrip("+") if inception else "Unknown"
            label_map = {"autograph": cfg["name"], "workshop": f"Attributed to / workshop of {cfg['name']}",
                         "follower": f"School of {cfg['name']}", "imitator_fake": f"Manner of / after {cfg['name']}"}
            by_item[qid] = {
                "artwork_id": f"wd_{qid}",
                "artist_canonical": cfg["name"],
                "title": title,
                "official_attribution": label_map[rel],
                "attribution_category": rel,
                "creation_date": year,
                "medium": "Unknown",
                "dimensions": "Unknown",
                "museum": b.get("collLabel", {}).get("value", "Unknown"),
                "image_url": b["image"]["value"].replace("http://", "https://"),
                "wikidata_qid": qid,
                "inventory_number": b.get("inv", {}).get("value", ""),
            }
        return list(by_item.values())

    # --------------------------------------------------------------- download
    def _throttle(self, min_interval: float = 0.6) -> None:
        with self._throttle_lock:
            now = time.time()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = max(now, self._next_allowed) + min_interval

    def _fetch(self, url: str, retries: int = 6) -> Optional[bytes]:
        delay = 5.0
        for _ in range(retries):
            self._throttle()
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    retry_after = e.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    time.sleep(sleep_s + random.random())
                    delay = min(delay * 2, 120)
                    continue
                if e.code in (400, 404) and "width=" in url:
                    return None
                logger.warning("HTTP %s for %s", e.code, url)
                return None
            except Exception as e:  # network hiccup
                time.sleep(delay)
                delay = min(delay * 2, 120)
                logger.debug("Retrying %s after %s", url, e)
        return None

    def download(self, cand: Dict[str, Any], artist_dir: Path) -> Optional[ContestedArtworkRecord]:
        out_file = artist_dir / f"{_safe_id(cand['artwork_id'])}.jpg"
        if out_file.exists() and out_file.stat().st_size > 30000:
            try:
                with Image.open(out_file) as img:
                    w, h = img.size
                return self._record(cand, out_file, w, h)
            except Exception:
                pass

        base = cand["image_url"]
        sep = "&" if "?" in base else "?"
        data = self._fetch(f"{base}{sep}width={self.thumb_width}") or self._fetch(base)
        if not data or len(data) < 10000:
            return None
        try:
            with Image.open(io.BytesIO(data)) as pil:
                rgb = pil.convert("RGB")
            if max(rgb.size) > self.max_dimension:
                s = self.max_dimension / max(rgb.size)
                rgb = rgb.resize((int(rgb.width * s), int(rgb.height * s)), Image.Resampling.LANCZOS)
            rgb.save(out_file, format="JPEG", quality=92, optimize=True)
            return self._record(cand, out_file, rgb.width, rgb.height)
        except Exception as e:
            logger.warning("Could not decode %s: %s", cand["artwork_id"], e)
            return None

    @staticmethod
    def _record(cand: Dict[str, Any], out_file: Path, w: int, h: int) -> ContestedArtworkRecord:
        return ContestedArtworkRecord(
            artwork_id=cand["artwork_id"],
            artist_canonical=cand["artist_canonical"],
            title=cand["title"],
            official_attribution=cand["official_attribution"],
            attribution_category=cand["attribution_category"],
            creation_date=cand["creation_date"],
            medium=cand["medium"],
            dimensions=cand["dimensions"],
            museum=cand["museum"],
            image_url=cand["image_url"],
            local_path=str(out_file.resolve()),
            width=w,
            height=h,
            filesize_bytes=out_file.stat().st_size,
        )

    # ---------------------------------------------------------------- harvest
    def harvest_artist(self, artist_key: str, merge_existing: bool = True, limit: Optional[int] = None) -> List[ContestedArtworkRecord]:
        """Downloads the Wikidata corpus and merges it with any existing NGA/CMA catalog."""
        import concurrent.futures

        artist_dir = self.base_dir / artist_key
        artist_dir.mkdir(parents=True, exist_ok=True)
        catalog_file = artist_dir / "catalog.json"

        existing: Dict[str, Dict[str, Any]] = {}
        if merge_existing and catalog_file.exists():
            for rec in json.loads(catalog_file.read_text(encoding="utf-8")):
                existing[rec["artwork_id"]] = rec

        cands = self.query_artist(artist_key)
        if limit and len(cands) > limit:
            cands = cands[:limit]
        counts = {k: sum(1 for c in cands if c["attribution_category"] == k) for k in _CATEGORY_RANK}
        print(f"[{artist_key.upper()}] Wikidata: {len(cands)} paintings {counts}", flush=True)

        records: List[ContestedArtworkRecord] = []
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(self.download, c, artist_dir): c for c in cands}
            for fut in concurrent.futures.as_completed(futures):
                done += 1
                rec = fut.result()
                if rec:
                    records.append(rec)
                if done % 25 == 0:
                    print(f"  [{artist_key}] {done}/{len(cands)} processed, {len(records)} saved", flush=True)
                    self._save(catalog_file, existing, records)

        self._save(catalog_file, existing, records)
        print(f"  [OK] {artist_key}: {len(records)} Wikidata paintings + {len(existing)} prior records", flush=True)
        return records

    @staticmethod
    def _save(catalog_file: Path, existing: Dict[str, Dict[str, Any]], records: List[ContestedArtworkRecord]) -> None:
        merged = dict(existing)
        for r in records:
            merged[r.artwork_id] = asdict(r)
        tmp = catalog_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(catalog_file)
