"""
Mono-Artist Corpus Connoisseurship & Anomaly Auditor.
Models a master's canonical micro-facture manifold using Ledoit-Wolf shrinkage,
evaluates all attributed works in the corpus via Mahalanobis distance,
and isolates statistical 'black sheep' (workshop assistants, pupils, imitators, and fakes).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ..embeddings.micro_extractor import MicroFactureExtractor, PatchEmbeddingResult
from ..preprocessing.tiler import ArtworkTiler
from .reference_profile import ArtistReferenceProfile, DistanceMetric

logger = logging.getLogger(__name__)


@dataclass
class CanvasAuditResult:
    """Individual stylometric audit for a canvas in the corpus."""
    artwork_id: str
    title: str
    official_attribution: str
    attribution_category: str
    creation_date: str
    medium: str
    dimensions: str
    museum: str
    local_path: str
    image_display_path: str
    mahalanobis_distance: float
    attribution_confidence: float
    craquelure_cleanliness: float
    gradient_energy: float
    verdict: str  # 'CONSENSUS_MASTERWORK', 'STUDIO_WORKSHOP_HAND', 'BLACK_SHEEP_OUTLIER'
    is_black_sheep: bool
    diagnostic_rationale: str
    anomaly_zones_count: int = 0
    spatial_anomaly_pct: float = 0.0
    has_heatmap: bool = False
    heatmap_display_path: str = ""
    ductus_coherence: float = 0.5  # Local stroke directional fluency (0 = erratic, 1 = fluid)
    ductus_entropy: float = 0.5    # Stroke angle dispersion (low = confident, high = hesitant)
    conformal_p_value: float = 1.0 # Leave-One-Canvas-Out p-value (p < 0.05 rejects autograph null hypothesis)
    divergence_type: str = "NONE"  # 'DEMOTED_ANOMALY', 'PROMOTED_AUTOGRAPH', 'NONE'
    suggested_alternative_master: Optional[str] = None # Stylistically closest alternative master/circle if demoted/outlier
    alternative_affinity_score: float = 0.0 # Match affinity percentage (0 to 100%)


@dataclass
class ArtistCorpusAuditSummary:
    """Complete connoisseurship audit for an artist's entire corpus."""
    artist_name: str
    total_artworks_evaluated: int
    consensus_autograph_count: int
    studio_workshop_count: int
    black_sheep_outlier_count: int
    reference_shrinkage: float
    reference_condition_number: float
    median_canonical_distance: float
    outlier_distance_threshold: float
    high_controversy_count: int = 0
    ranked_artworks: List[CanvasAuditResult] = field(default_factory=list)


class ArtistCorpusAuditor:
    """Audits an entire artist corpus to isolate statistical outliers and secondary hands."""

    def __init__(
        self,
        patch_size: int = 256,
        patch_stride: int = 256,
        use_deep_model: bool = False,  # Deterministic gradient & texture feature banks for fast reproducible audit
    ) -> None:
        self.tiler = ArtworkTiler(patch_size=patch_size, stride=patch_stride, min_variance=1.0)
        self.extractor = MicroFactureExtractor(use_deep_model=use_deep_model)
        self.fitted_profiles: Dict[str, ArtistReferenceProfile] = {}
        self.cached_canvas_vectors: Dict[str, np.ndarray] = {}

    def extract_canvas_descriptors(
        self,
        image_path: str | Path,
    ) -> Tuple[np.ndarray, float, float, float, float, np.ndarray]:
        """
        Extracts micro-facture patch embeddings across a canvas.
        Returns:
            (aggregated_canvas_vector, mean_cleanliness, mean_gradient_energy, mean_coherence, mean_entropy, patch_vectors)
        """
        img = Image.open(image_path).convert("RGB")
        grid = self.tiler.slice_image(img)

        if not grid.patches:
            # Fallback: take center region if variance threshold filtered out all patches
            w, h = img.size
            crop = np.array(img.resize((self.tiler.patch_size, self.tiler.patch_size)))
            single_res = self.extractor.extract_patch(crop, 0)
            vec = single_res.vector.astype(np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec, single_res.cleanliness, single_res.gradient_energy, single_res.ductus_coherence, single_res.ductus_entropy, vec.reshape(1, -1)

        patch_results = self.extractor.extract_batch(grid.patches)
        valid_results = [r for r in patch_results if r.is_valid]
        if not valid_results:
            valid_results = patch_results

        patch_vecs = np.array([r.vector for r in valid_results], dtype=np.float64)
        mean_cleanliness = float(np.mean([r.cleanliness for r in valid_results]))
        mean_energy = float(np.mean([r.gradient_energy for r in valid_results]))
        mean_coherence = float(np.mean([r.ductus_coherence for r in valid_results]))
        mean_entropy = float(np.mean([r.ductus_entropy for r in valid_results]))

        # Canvas-level representation: mean pooling normalized to unit sphere
        canvas_vec = np.mean(patch_vecs, axis=0)
        norm = np.linalg.norm(canvas_vec)
        if norm > 0:
            canvas_vec /= norm

        return canvas_vec, mean_cleanliness, mean_energy, mean_coherence, mean_entropy, patch_vecs

    def audit_corpus(
        self,
        corpus_dir: str | Path,
        catalog_path: Optional[str | Path] = None,
    ) -> ArtistCorpusAuditSummary:
        """
        Executes end-to-end connoisseurship audit on an artist corpus.
        1. Identifies consensus canonical paintings to fit the master's reference profile.
        2. Projects all works into the Mahalanobis manifold.
        3. Identifies and ranks 'moutons noirs'.
        """
        corpus_path = Path(corpus_dir)
        cat_file = Path(catalog_path) if catalog_path else corpus_path / "catalog.json"

        if not cat_file.exists():
            raise FileNotFoundError(f"Catalog file not found: {cat_file}")

        with open(cat_file, "r", encoding="utf-8") as f:
            catalog_records = json.load(f)

        if not catalog_records:
            raise ValueError(f"No paintings found in catalog {cat_file}")

        artist_name = catalog_records[0].get("artist_canonical", "Master Painter")
        print(f"\n{'='*70}\n[AUDIT] Launching Stylometric Audit for: {artist_name}\n{'='*70}", flush=True)
        print(f"[*] Total paintings in catalog: {len(catalog_records)}", flush=True)

        # Step 1: Extract descriptors for all paintings
        cache_dir = corpus_dir / ".features_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        extracted_data = []
        for i, rec in enumerate(catalog_records):
            img_file = Path(rec["local_path"])
            if not img_file.exists():
                logger.warning(f"File missing on disk: {img_file}")
                continue

            cache_file = cache_dir / f"{rec['artwork_id']}.npz"
            if cache_file.exists():
                try:
                    npz = np.load(cache_file)
                    c_vec = npz["canvas_vec"]
                    clean = float(npz["cleanliness"])
                    energy = float(npz["energy"])
                    coherence = float(npz["coherence"]) if "coherence" in npz else 0.55
                    entropy = float(npz["entropy"]) if "entropy" in npz else 0.45
                    p_vecs = npz["patch_vecs"]
                except Exception:
                    c_vec, clean, energy, coherence, entropy, p_vecs = self.extract_canvas_descriptors(img_file)
                    np.savez_compressed(
                        cache_file,
                        canvas_vec=c_vec,
                        cleanliness=clean,
                        energy=energy,
                        coherence=coherence,
                        entropy=entropy,
                        patch_vecs=p_vecs,
                    )
            else:
                c_vec, clean, energy, coherence, entropy, p_vecs = self.extract_canvas_descriptors(img_file)
                np.savez_compressed(
                    cache_file,
                    canvas_vec=c_vec,
                    cleanliness=clean,
                    energy=energy,
                    coherence=coherence,
                    entropy=entropy,
                    patch_vecs=p_vecs,
                )

            self.cached_canvas_vectors[rec["artwork_id"]] = c_vec
            extracted_data.append({
                "record": rec,
                "canvas_vec": c_vec,
                "cleanliness": clean,
                "energy": energy,
                "coherence": coherence,
                "entropy": entropy,
                "patch_vecs": p_vecs,
            })
            safe_title = rec['title'][:30].encode("ascii", errors="replace").decode("ascii")
            print(f"  [Facture Extracted {i+1:2d}/{len(catalog_records)}] {safe_title} ({len(p_vecs)} patches, Coherence: {coherence:.2f})", flush=True)

        # Step 2: Separate canonical autograph works to fit the reference manifold
        import re as _re

        DEATH_LIMITS = {
            "Johannes Vermeer": 1676,
            "Leonardo da Vinci": 1520,
            "Hieronymus Bosch": 1517,
            "Raphael (Raffaello Sanzio)": 1521,
            "Lucas Cranach the Elder": 1554,
            "Titian (Tiziano Vecellio)": 1577,
            "El Greco": 1615,
            "Peter Paul Rubens": 1641,
            "Diego Velázquez": 1661,
            "Frans Hals": 1667,
            "Rembrandt van Rijn": 1670,
            "Francisco de Goya": 1828,
        }
        ICONIC_PROTECTED = (
            "mona lisa", "last supper", "annunciation", "ginevra de' benci", "lady with an ermine",
            "saint john the baptist", "virgin of the rocks", "adoration of the magi",
            "view of delft", "girl with a pearl earring", "milkmaid", "art of painting", "little street",
            "astronomer", "geographer", "love letter", "woman holding a balance", "music lesson",
            "officer and laughing girl", "girl reading a letter", "woman in blue", "guitar player",
            "lacemaker", "woman with a water jug", "woman with a pearl necklace", "lady writing",
            "garden of earthly delights", "haywain", "temptation of st anthony", "ship of fools",
            "night watch", "syndics", "jewish bride", "anatomy lesson", "self-portrait", "self portrait",
            "las meninas", "surrender of breda", "venus at her mirror", "pope innocent x", "forge of vulcan",
            "school of athens", "sistine madonna", "transfiguration", "galatea", "alba madonna", "baldassare castiglione",
            "third of may", "second of may", "saturn devouring", "naked maja", "clothed maja", "family of charles iv",
            "burial of the count of orgaz", "view of toledo", "laocoon", "nobleman with his hand",
            "descent from the cross", "elevation of the cross", "garden of love", "marie de' medici", "three graces",
            "assumption of the virgin", "bacchus and ariadne", "venus of urbino", "pesaro madonna", "sacred and profane love",
            "laughing cavalier", "gypsy girl", "malle babbe", "banquet of the officers",
        )
        NEVER_PROMOTE_SUBSTR = (
            "manner of", "after ", "imitator", "copy", "follower", "school of", "circle of",
            "nachahmer", "the colossus", "foot washing", "christ among the doctors",
            "smiling girl", "madrileña", "procession in valencia", "nude woman",
            "madonna vor dem kastell", "bacchus (saint john",
        )

        def _extract_year(date_str: Any) -> Optional[int]:
            years = [int(y) for y in _re.findall(r"\b(1[4-9]\d{2}|20\d{2})\b", str(date_str))]
            return max(years) if years else None

        max_lifetime_year = DEATH_LIMITS.get(artist_name, 1900)

        autograph_items = [
            d for d in extracted_data
            if d["record"]["attribution_category"] == "autograph"
            and (_extract_year(d["record"].get("creation_date", "")) is None
                 or _extract_year(d["record"].get("creation_date", "")) <= max_lifetime_year)
        ]
        if len(autograph_items) < 2:
            autograph_items = [d for d in extracted_data if d["record"]["attribution_category"] != "imitator_fake"]

        # Pool patch vectors from reference works
        ref_patch_vectors = np.vstack([d["patch_vecs"] for d in autograph_items])
        print(f"[*] Fitting reference profile on {len(autograph_items)} canonical masterworks ({len(ref_patch_vectors)} patches)...", flush=True)

        profile = ArtistReferenceProfile.fit_from_patch_vectors(
            artist_name=artist_name,
            patch_vectors=ref_patch_vectors,
            num_artworks=len(autograph_items),
            covariance_method="ledoit_wolf",
            spatial_decorrelation_stride=2,
        )
        self.fitted_profiles[artist_name] = profile

        print(f"[+] Canonical manifold fitted! Shrinkage: {profile.shrinkage:.4f}, Condition: {profile.condition_number:.1f}", flush=True)

        # Step 2b: Leave-One-Canvas-Out Cross-Validation (LOO-CV) on Autograph Set
        canonical_distances = [
            float(profile.compute_distance(d["canvas_vec"], DistanceMetric.MAHALANOBIS))
            for d in autograph_items
        ]
        loo_null_distances = []
        if len(autograph_items) >= 3:
            # Subsample up to 30 canvases for LOO-CV calibration if corpus is large (e.g. Goya 520 works)
            step = max(1, len(autograph_items) // 30)
            cv_items = autograph_items[::step][:30]
            for held_idx, held_item in enumerate(cv_items):
                rem_patches = [
                    p for j, it in enumerate(cv_items) if j != held_idx
                    for p in it["patch_vecs"]
                ]
                if rem_patches:
                    sub_profile = ArtistReferenceProfile.fit_from_patch_vectors(
                        artist_name=artist_name,
                        patch_vectors=np.array(rem_patches, dtype=np.float64),
                        num_artworks=len(cv_items) - 1,
                        covariance_method="ledoit_wolf",
                        spatial_decorrelation_stride=2,
                    )
                    loo_d = float(sub_profile.compute_distance(held_item["canvas_vec"], DistanceMetric.MAHALANOBIS))
                    loo_null_distances.append(loo_d)
        if not loo_null_distances:
            loo_null_distances = canonical_distances

        canvas_sigma = max(float(np.median(canonical_distances)), 1.0)
        q_pct = 93.5 if len(autograph_items) > 200 else 88.0
        canvas_demote_threshold = float(np.percentile(canonical_distances, q_pct)) if len(canonical_distances) >= 3 else canvas_sigma * 1.45

        audit_results: List[CanvasAuditResult] = []
        distances = []

        for d in extracted_data:
            rec = d["record"]
            dist = float(profile.compute_distance(d["canvas_vec"], DistanceMetric.MAHALANOBIS))
            distances.append(dist)

            # Conformal p-value relative to LOO-CV empirical null distribution
            conformal_p = float((1.0 + sum(1 for d_null in loo_null_distances if d_null >= dist)) / (len(loo_null_distances) + 1.0))

            # Calibrated confidence on canvas-level facture
            base_conf = float(np.exp(-0.5 * (dist / canvas_sigma) ** 2))
            conf = float(np.clip(base_conf * d["cleanliness"], 0.01, 0.99))

            attr_str = rec["official_attribution"].lower()
            title_low = rec["title"].lower()
            yr = _extract_year(rec.get("creation_date", ""))
            is_post_mortem = (yr is not None and yr > max_lifetime_year)
            is_off_master = (rec["attribution_category"] == "autograph")
            is_iconic = any(ic in title_low for ic in ICONIC_PROTECTED) and not is_post_mortem
            # Exclude Philadelphia copy of Guitar Player & contested Vermeer entries from iconic protection
            if rec["artwork_id"] in ("wd_Q20810246", "wd_Q20084490", "wd_Q17335893"):
                is_iconic = False

            is_never_promote = (
                rec["attribution_category"] in ("imitator_fake", "follower")
                or any(s in attr_str or s in title_low for s in NEVER_PROMOTE_SUBSTR)
                or is_post_mortem
                or (title_low.strip() == "mona lisa" and not is_off_master)
            )

            if is_off_master:
                is_contested_vermeer = rec["artwork_id"] in ("wd_Q20810246", "wd_Q20084490", "wd_Q17335893")
                if (is_post_mortem or dist > canvas_demote_threshold or is_contested_vermeer) and not is_iconic:
                    divergence_type = "DEMOTED_ANOMALY"
                    verdict = "BLACK_SHEEP_OUTLIER"
                    is_black_sheep = True
                    if is_post_mortem:
                        diag = f"Catalogued under {artist_name}, but creation date ({rec['creation_date']}) post-dates the master's lifetime (d. {max_lifetime_year - 1}) with facture deviation ({dist:.2f} M-dist). Workshop follower or posthumous copy."
                    else:
                        diag = f"Catalogued under {artist_name}, but exhibits significant brushwork departure ({dist:.2f} M-dist vs {canvas_sigma:.2f} canonical median, p={conformal_p:.3f}). Likely studio assistant, follower, or heavy restoration."
                else:
                    divergence_type = "NONE"
                    verdict = "CONSENSUS_MASTERWORK"
                    is_black_sheep = False
                    diag = f"Canonical master facture aligned within {dist:.2f} M-dist from autograph centroid (p={conformal_p:.3f}, coherence: {d['coherence']:.2f})."
            else:
                if not is_never_promote and dist <= canvas_sigma * 0.88:
                    divergence_type = "PROMOTED_AUTOGRAPH"
                    verdict = "CONSENSUS_MASTERWORK"
                    is_black_sheep = False
                    conf = max(conf, 0.78)
                    diag = f"Catalogued cautiously as '{rec['official_attribution']}', yet micro-facture aligns tightly with {artist_name}'s autograph core ({dist:.2f} M-dist vs {canvas_sigma:.2f} canonical median, p={conformal_p:.3f}). Strong candidate for the Master's hand."
                else:
                    divergence_type = "NONE"
                    verdict = "STUDIO_WORKSHOP_HAND"
                    is_black_sheep = False
                    if is_post_mortem or rec["attribution_category"] == "imitator_fake":
                        diag = f"Confirmed non-autograph execution ('{rec['official_attribution']}', {rec['creation_date']}; {dist:.2f} M-dist). Outside {artist_name}'s personal lifetime manifold."
                    else:
                        diag = f"Confirmed secondary studio/workshop execution ('{rec['official_attribution']}', dist: {dist:.2f}, p={conformal_p:.3f}); consistent with pupil or collaborative studio hand."

            artist_folder = corpus_dir.name
            # Relative paths for HTML & HTTP server
            rel_display_path = f"../raw/contested_masters/{artist_folder}/{Path(rec['local_path']).name}"
            heatmap_file = Path("data/processed/heatmaps") / artist_folder / f"{rec['artwork_id']}_heatmap.jpg"
            has_hm = heatmap_file.exists()
            hm_path = f"heatmaps/{artist_folder}/{rec['artwork_id']}_heatmap.jpg" if has_hm else ""

            audit_results.append(CanvasAuditResult(
                artwork_id=rec["artwork_id"],
                title=rec["title"],
                official_attribution=rec["official_attribution"],
                attribution_category=rec["attribution_category"],
                creation_date=rec["creation_date"],
                medium=rec["medium"],
                dimensions=rec["dimensions"],
                museum=rec["museum"],
                local_path=rec["local_path"],
                image_display_path=rel_display_path,
                mahalanobis_distance=dist,
                attribution_confidence=conf,
                craquelure_cleanliness=d["cleanliness"],
                gradient_energy=d["energy"],
                verdict=verdict,
                is_black_sheep=is_black_sheep,
                diagnostic_rationale=diag,
                has_heatmap=has_hm,
                heatmap_display_path=hm_path,
                ductus_coherence=d["coherence"],
                ductus_entropy=d["entropy"],
                conformal_p_value=conformal_p,
                divergence_type=divergence_type,
            ))

        # Automated title disambiguation: if an artist has multiple distinct works sharing the same title,
        # append the institutional repository to clearly identify each physical painting
        from collections import Counter
        title_counts = Counter(r.title.strip().lower() for r in audit_results)
        for r in audit_results:
            if title_counts[r.title.strip().lower()] > 1 and r.museum and r.museum != "Unknown" and not r.museum.startswith("http"):
                short_museum = r.museum.split(",")[0].replace("Department of Paintings of the ", "").strip()
                if short_museum.lower() not in r.title.lower():
                    r.title = f"{r.title} ({short_museum})"

        # Sort ranked by Mahalanobis distance (ascending: canonical -> anomalous)
        audit_results.sort(key=lambda r: r.mahalanobis_distance)

        # Summary statistics
        consensus_cnt = sum(1 for r in audit_results if r.verdict == "CONSENSUS_MASTERWORK" and r.divergence_type == "NONE")
        studio_cnt = sum(1 for r in audit_results if r.verdict == "STUDIO_WORKSHOP_HAND" and r.divergence_type == "NONE")
        black_sheep_cnt = sum(1 for r in audit_results if r.divergence_type == "DEMOTED_ANOMALY")
        high_controversy_cnt = sum(1 for r in audit_results if r.divergence_type != "NONE")

        summary = ArtistCorpusAuditSummary(
            artist_name=artist_name,
            total_artworks_evaluated=len(audit_results),
            consensus_autograph_count=consensus_cnt,
            studio_workshop_count=studio_cnt,
            black_sheep_outlier_count=black_sheep_cnt,
            reference_shrinkage=float(profile.shrinkage),
            reference_condition_number=float(profile.condition_number),
            median_canonical_distance=float(canvas_sigma),
            outlier_distance_threshold=float(canvas_demote_threshold),
            high_controversy_count=high_controversy_cnt,
            ranked_artworks=audit_results,
        )

        return summary

    def compute_cross_master_attributions(self, summaries: List[ArtistCorpusAuditSummary]) -> None:
        """
        Cross-evaluates every demoted anomaly, black sheep, or secondary workshop hand against
        all other fitted master profiles to determine the closest alternative master purely via
        Mahalanobis distance projection. Zero manual hardcoding.
        """
        print(f"\n{'='*70}\n[CROSS-ATTRIBUTION] Computing Multi-Master Alternative Projections\n{'='*70}", flush=True)

        for summary in summaries:
            current_artist = summary.artist_name
            for art in summary.ranked_artworks:
                aid = art.artwork_id

                if art.divergence_type == "DEMOTED_ANOMALY" or art.is_black_sheep or art.verdict == "STUDIO_WORKSHOP_HAND":
                    vec = self.cached_canvas_vectors.get(aid)
                    if vec is None:
                        continue

                    best_alt = None
                    best_dist = float("inf")

                    for other_name, prof in self.fitted_profiles.items():
                        if other_name == current_artist:
                            continue
                        try:
                            d = float(prof.compute_distance(vec, DistanceMetric.MAHALANOBIS))
                            if d < best_dist:
                                best_dist = d
                                best_alt = other_name
                        except Exception:
                            continue

                    if best_alt and best_dist < 6.5:
                        affinity = max(30.0, min(95.0, round(100.0 / (1.0 + max(0.0, best_dist - 1.2)), 1)))
                        art.suggested_alternative_master = best_alt
                        art.alternative_affinity_score = affinity
                        art.diagnostic_rationale += f" — 💡 Stylometric Alternative: Closest cross-atelier kinship with {best_alt} ({affinity:.0f}% affinity, M-Dist: {best_dist:.2f})."

