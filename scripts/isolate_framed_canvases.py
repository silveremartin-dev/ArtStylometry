"""
Automated Picture Frame, Passe-Partout Mount, and Archive Banner Isolation Script.
Scans harvested museum paintings, detects and strips outer frames/mounts/text banners
via ArtworkTiler.detect_canvas_bbox, and invalidates cached feature vectors for all
cropped works so downstream stylometric audits and manifold clustering evaluate only
the painted canvas.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.tiler import ArtworkTiler


def process_single_painting(item: Tuple[str, Path, Optional[Path]]) -> Optional[Dict[str, Any]]:
    artwork_id, img_path, cache_npz = item
    if not img_path.exists():
        return None
    try:
        with Image.open(img_path) as im:
            rgb_im = im.convert("RGB")
            arr = np.asarray(rgb_im)
        h, w = arr.shape[:2]
        if h < 80 or w < 80:
            return None

        x0, y0, x1, y1 = ArtworkTiler.detect_canvas_bbox(arr)
        new_w = x1 - x0
        new_h = y1 - y0
        kept_ratio = (new_w * new_h) / float(w * h)
        max_inset = max(x0, y0, w - x1, h - y1)

        if kept_ratio < 0.965 or max_inset >= 18:
            cropped_arr = arr[y0:y1, x0:x1]
            Image.fromarray(cropped_arr).save(img_path, quality=94)
            if cache_npz and cache_npz.exists():
                try:
                    cache_npz.unlink()
                except Exception:
                    pass
            return {
                "artwork_id": artwork_id,
                "path": str(img_path),
                "orig_size": (w, h),
                "cropped_size": (new_w, new_h),
                "kept_pct": round(kept_ratio * 100.0, 1),
            }
    except Exception:
        return None
    return None


def main() -> None:
    t0 = time.time()
    tasks: List[Tuple[str, Path, Optional[Path]]] = []

    # 1. All 12 Contested Master Workshops (1,137 paintings)
    masters_root = Path("data/raw/contested_masters")
    if masters_root.exists():
        for artist_dir in sorted(masters_root.iterdir()):
            if not artist_dir.is_dir():
                continue
            cat_file = artist_dir / "catalog.json"
            cache_dir = artist_dir / ".features_cache"
            if cat_file.exists():
                records = json.loads(cat_file.read_text(encoding="utf-8"))
                for r in records:
                    aid = r.get("artwork_id", "")
                    lp = Path(r.get("local_path", ""))
                    if not lp.exists():
                        lp = artist_dir / lp.name
                    c_npz = cache_dir / f"{aid}.npz"
                    tasks.append((aid, lp, c_npz))

    # 2. General Museum Collection Paintings
    cat_path = Path("data/raw/paintings_catalog.json")
    if cat_path.exists():
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        for c in cat:
            aid = c.get("artwork_id", "")
            lp = Path(c.get("local_path", ""))
            if aid and lp.exists():
                tasks.append((aid, lp, None))

    print(f"[*] Scanning {len(tasks)} paintings for frames, mounts, and text banners...")
    cropped_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(process_single_painting, tasks):
            if res is not None:
                cropped_results.append(res)

    # 3. Invalidate global manifold facture cache entries for all cropped paintings
    global_cache_path = Path("data/embeddings/cache/facture_cache_all.npz")
    if global_cache_path.exists() and cropped_results:
        cropped_ids = {r["artwork_id"] for r in cropped_results}
        try:
            with np.load(global_cache_path) as data:
                cache_dict = {k: data[k] for k in data.files if k not in cropped_ids}
            np.savez_compressed(global_cache_path, **cache_dict)
            print(f"[*] Invalidated {len(cropped_ids)} stale entries in {global_cache_path}")
        except Exception as e:
            print(f"[!] Warning updating global cache: {e}")

    elapsed = time.time() - t0
    print(f"[+] Isolated painted canvas on {len(cropped_results)} framed artworks in {elapsed:.1f}s.")
    for r in cropped_results[:25]:
        print(
            f"    - {r['artwork_id']} ({r['path']}): {r['orig_size'][0]}x{r['orig_size'][1]} -> "
            f"{r['cropped_size'][0]}x{r['cropped_size'][1]} (kept {r['kept_pct']}%)"
        )


if __name__ == "__main__":
    main()
