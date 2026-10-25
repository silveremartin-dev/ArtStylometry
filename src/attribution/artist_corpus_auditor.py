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
        autograph_items = [d for d in extracted_data if d["record"]["attribution_category"] == "autograph"]
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
        canvas_p95_threshold = float(np.percentile(canonical_distances, 95)) * 1.5 if len(canonical_distances) >= 3 else canvas_sigma * 1.8

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

            # Verdict categorization
            if dist <= canvas_sigma * 1.25 and rec["attribution_category"] == "autograph":
                verdict = "CONSENSUS_MASTERWORK"
                is_black_sheep = False
                diag = f"Pivotal master facture aligned within {dist:.2f} M-dist from canonical centroid (p={conformal_p:.3f}, coherence: {d['coherence']:.2f})."
            elif dist <= canvas_sigma * 1.8:
                verdict = "STUDIO_WORKSHOP_HAND"
                is_black_sheep = (rec["attribution_category"] in ("workshop", "imitator_fake"))
                diag = f"Partial alignment with master ductus (dist: {dist:.2f}, p={conformal_p:.3f}); characteristic of collaborative workshop execution."
            else:
                verdict = "BLACK_SHEEP_OUTLIER"
                is_black_sheep = True
                diag = f"Severe statistical departure ({dist:.2f} M-dist, p={conformal_p:.3f}, baseline: {canvas_sigma:.2f}). Non-autograph touch, pupil hand, or modern intervention."

            # Divergence classification (Novel findings vs museum consensus)
            attr_str = rec["official_attribution"].lower()
            is_off_master = (rec["attribution_category"] == "autograph" or attr_str == artist_name.lower())
            is_off_demoted = (rec["attribution_category"] in ("workshop", "circle_follower", "disputed", "imitator_fake") or
                              any(k in attr_str for k in ("workshop", "circle", "follower", "attributed", "school")))

            if is_off_master and is_black_sheep:
                divergence_type = "DEMOTED_ANOMALY"
            elif is_off_demoted and (verdict == "CONSENSUS_MASTERWORK" or conf >= 0.70):
                divergence_type = "PROMOTED_AUTOGRAPH"
            else:
                divergence_type = "NONE"

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

        # Sort ranked by Mahalanobis distance (ascending: canonical -> anomalous)
        audit_results.sort(key=lambda r: r.mahalanobis_distance)

        # Summary statistics
        consensus_cnt = sum(1 for r in audit_results if r.verdict == "CONSENSUS_MASTERWORK")
        studio_cnt = sum(1 for r in audit_results if r.verdict == "STUDIO_WORKSHOP_HAND")
        black_sheep_cnt = sum(1 for r in audit_results if r.is_black_sheep)
        high_controversy_cnt = sum(1 for r in audit_results if r.divergence_type != "NONE")

        summary = ArtistCorpusAuditSummary(
            artist_name=artist_name,
            total_artworks_evaluated=len(audit_results),
            consensus_autograph_count=consensus_cnt,
            studio_workshop_count=studio_cnt,
            black_sheep_outlier_count=black_sheep_cnt,
            reference_shrinkage=float(profile.shrinkage),
            reference_condition_number=float(profile.condition_number),
            median_canonical_distance=float(profile.median_distance),
            outlier_distance_threshold=float(profile.outlier_threshold_p95),
            high_controversy_count=high_controversy_cnt,
            ranked_artworks=audit_results,
        )

        return summary
