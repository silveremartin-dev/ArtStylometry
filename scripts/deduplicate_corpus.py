"""
Corpus Deduplication and Cleansing Script.
Removes duplicate photographic instances of the same physical painting (e.g. seeded vs Wikidata),
eliminates non-painting documentary photographs (e.g. Sala delle Asse museum hall photo),
clarifies workshop copies (e.g. Chambéry copy vs Louvre original),
and resolves historical misattributions (e.g. Rubens's copy of Battle of Anghiari).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deduplicate_corpus")

# Known non-paintings, documentary photos, or duplicate photo entries to remove
EXCLUDED_ARTWORK_IDS: Set[str] = {
    # Leonardo da Vinci:
    "wd_Q3800774",      # Sala delle Asse (wide-angle photo of castle room & ceiling with museum spot lights)
    "wd_Q131414918",    # Codex Atlanticus (manuscript text page, not a painting)
    "leo_mona_lisa",    # Duplicate of wd_Q12418 (Louvre Mona Lisa)
    "leo_la_belle_ferronniere", # Duplicate of wd_Q656434 (Louvre La Belle Ferronnière)
    "leo_st_anne",      # Duplicate of wd_Q563727 (Louvre Virgin and Child with Saint Anne)
    "leo_virgin_rocks_louvre", # Duplicate of wd_Q11935346 (Louvre Virgin of the Rocks)
    "leo_virgin_rocks_london", # Duplicate of wd_Q4009972 (London Virgin of the Rocks)
    "leo_lady_ermine",  # Duplicate of wd_Q474338 (Czartoryski Lady with an Ermine)
    "leo_annunciation", # Duplicate of wd_Q1217213 (Uffizi Annunciation)
    "leo_portrait_musician", # Duplicate of wd_Q2338404 (Ambrosiana Musician)
    "leo_madonna_litta", # Duplicate of wd_Q1327676 (Hermitage Madonna Litta)
    "leo_bacchus",      # Duplicate of wd_Q798034 (Louvre Bacchus)
    "leo_st_john_baptist", # Duplicate of wd_Q783215 (Louvre Saint John the Baptist)
    "leo_baptism_christ", # Duplicate of wd_Q1059211 (Uffizi Baptism of Christ)
    "leo_salvator_mundi", # Duplicate of wd_Q1892745 (Louvre Abu Dhabi Salvator Mundi)
    "leo_isleworth_mona_lisa", # Duplicate of wd_Q4390923 (Isleworth Mona Lisa)
    "wd_Q1267893",      # Duplicate of nga_50724 (Ginevra de' Benci [obverse])
    "wd_Q20174114",     # Duplicate of nga_50725 (Wreath of Laurel, Palm, and Juniper [reverse])

    # Rembrandt:
    "wd_Q21505931",     # Duplicate of cma_122348 ("A Young Man with a Chain" at Cleveland Museum)

    # El Greco:
    "nga_91",           # Duplicate of nga_1164 ("Saint Martin and the Beggar" at NGA)

    # Goya:
    "wd_Q20539657",     # Duplicate of wd_Q118142042 ("Time, Truth and History" at MFA Boston)
    "wd_Q11770765",     # Duplicate of wd_Q29959236 ("Plucked Turkey" at Munich)
    "wd_Q5965178",      # Duplicate of wd_Q27766122 ("The Holy Family" at Prado)
    "wd_Q3825327",      # Duplicate of wd_Q5966799 ("Blind Man's Bluff" at Prado)
    "wd_Q104522002",    # Duplicate of wd_Q68975875 ("Don Juan Bautista de Goicoechea" at Karlsruhe)
}

# Explicit title & attribution corrections for famous workshop / copy cases
METADATA_CORRECTIONS: Dict[str, Dict[str, Any]] = {
    "wd_Q18633460": {
        "title": "La Belle Ferronnière (Chambéry Workshop Copy)",
        "museum": "Musée des Beaux-Arts de Chambéry (on loan from Louvre)",
        "attribution_category": "workshop",
        "official_attribution": "Workshop / Follower of Leonardo da Vinci (Chambéry Collection)",
    },
    "wd_Q656434": {
        "title": "La Belle Ferronnière (Louvre Masterpiece)",
        "museum": "Musée du Louvre, Paris",
        "attribution_category": "autograph",
        "official_attribution": "Leonardo da Vinci",
    },
    "wd_Q2045726": {
        "title": "The Battle of Anghiari (Copy after Leonardo by Peter Paul Rubens)",
        "attribution_category": "follower",
        "official_attribution": "Peter Paul Rubens (after Leonardo da Vinci)",
    },
    "wd_Q27955518": {
        "title": "Salome with the Head of John the Baptist",
        "attribution_category": "workshop",
        "official_attribution": "Attributed to Cesare da Sesto (Circle of Leonardo da Vinci)",
    },
    "wd_Q2126369": {
        "title": "Mona Lisa (Prado Studio Version)",
        "attribution_category": "workshop",
        "official_attribution": "Workshop of Leonardo da Vinci (Museo del Prado)",
    },
    "wd_Q4390923": {
        "title": "Isleworth Mona Lisa",
        "attribution_category": "follower",
        "official_attribution": "Attributed to Leonardo da Vinci and Workshop (Contested)",
    },
    "wd_Q105339423": {
        "title": "Virgin and Child with Saint Anne (Berlin Workshop Copy)",
        "attribution_category": "imitator_fake",
        "official_attribution": "Follower of Leonardo da Vinci (Gemäldegalerie Berlin)",
    },
    "wd_Q59682078": {
        "title": "Virgin and Child with Saint Anne (Prado Workshop Copy)",
        "attribution_category": "imitator_fake",
        "official_attribution": "Follower of Leonardo da Vinci (Museo del Prado)",
    },
    "wd_Q27697256": {
        "title": "Saint Anne, the Virgin and the Christ Child (Prado Studio Version)",
        "attribution_category": "workshop",
        "official_attribution": "Workshop of Leonardo da Vinci (Museo del Prado)",
    },
}


def normalize_str(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\[.*?\]|\(.*?\)", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def clean_artist_catalog(artist_dir: Path, source_records: List[Dict[str, Any]]) -> Tuple[int, int]:
    cat_file = artist_dir / "catalog.json"
    orig_count = len(source_records)
    kept_records: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for r in source_records:
        aid = r.get("artwork_id", "")
        if aid in EXCLUDED_ARTWORK_IDS:
            logger.info("Excluding known duplicate / non-painting: %s (%s)", aid, r.get("title"))
            continue

        # Create clean copy
        rec = {
            "artwork_id": aid,
            "artist_canonical": r.get("artist_canonical"),
            "title": r.get("title"),
            "official_attribution": r.get("official_attribution"),
            "attribution_category": r.get("attribution_category"),
            "creation_date": str(r.get("creation_date", "")),
            "medium": r.get("medium", "Oil on canvas / panel"),
            "dimensions": r.get("dimensions", "Standard"),
            "museum": r.get("museum", "Museum Collection"),
            "local_path": r.get("local_path", ""),
        }

        # Apply metadata corrections if defined
        if aid in METADATA_CORRECTIONS:
            rec.update(METADATA_CORRECTIONS[aid])

        # Deduplication key by normalized title + museum
        norm_title = normalize_str(rec.get("title", ""))
        norm_mus = normalize_str(rec.get("museum", ""))
        dedup_key = f"{norm_title}::{norm_mus}"

        # If identical title in the exact same museum, keep only the first/best one
        if dedup_key in seen_keys and len(norm_title) > 4 and len(norm_mus) > 3:
            logger.info("Excluding duplicate title in same museum: %s - '%s' at %s", aid, rec.get("title"), rec.get("museum"))
            continue

        seen_keys.add(dedup_key)
        kept_records.append(rec)

    # Save cleaned catalog
    with open(cat_file, "w", encoding="utf-8") as f:
        json.dump(kept_records, f, indent=2, ensure_ascii=False)

    removed = orig_count - len(kept_records)
    logger.info("[%s] Catalog cleaned: %d -> %d records (%d removed)", artist_dir.name, orig_count, len(kept_records), removed)
    return orig_count, len(kept_records)


def main() -> None:
    audit_file = Path("data/processed/contested_masters_audit.json")
    if not audit_file.exists():
        raise FileNotFoundError("contested_masters_audit.json not found")

    with open(audit_file, "r", encoding="utf-8") as f:
        all_audit_records: List[Dict[str, Any]] = json.load(f)

    # Group by artist directory name
    records_by_artist: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_audit_records:
        lp = r.get("local_path", "")
        # Extract artist folder name from local_path
        m = re.search(r"contested_masters[/\\]([^/\\]+)", lp)
        if m:
            art_dir_name = m.group(1).lower()
            records_by_artist.setdefault(art_dir_name, []).append(r)

    masters_dir = Path("data/raw/contested_masters")
    total_before = 0
    total_after = 0
    for adir in sorted(masters_dir.iterdir()):
        if adir.is_dir() and adir.name.lower() in records_by_artist:
            b, a = clean_artist_catalog(adir, records_by_artist[adir.name.lower()])
            total_before += b
            total_after += a

    print(f"\n[+] Global Corpus Deduplication Complete: {total_before} -> {total_after} paintings ({total_before - total_after} duplicates removed).")


if __name__ == "__main__":
    main()
