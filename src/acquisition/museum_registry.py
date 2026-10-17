"""
Comprehensive Museum Registry and IIIF Endpoint Resolver.
Provides preset connectors, API schema resolvers, and physical dimension parsers
for the world's leading open-access fine art museum collections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MuseumPreset:
    """Configuration preset for an institutional museum collection API."""
    code: str
    name: str
    country: str
    city: str
    api_type: str  # "iiif_presentation", "iiif_image", "custom_rest", "direct_deepzoom"
    iiif_version: str  # "2.0", "2.1", "3.0", "custom"
    portal_url: str
    manifest_template: Optional[str] = None
    image_service_template: Optional[str] = None
    search_endpoint: Optional[str] = None
    supports_physical_dimensions: bool = False
    rate_limit_req_per_sec: float = 5.0
    notes: str = ""
    example_identifiers: List[str] = field(default_factory=list)


# Registry of worldwide museum collections with open-access or IIIF deep-zoom endpoints
MUSEUM_REGISTRY: Dict[str, MuseumPreset] = {
    "rijksmuseum": MuseumPreset(
        code="rijksmuseum",
        name="Rijksmuseum",
        country="Netherlands",
        city="Amsterdam",
        api_type="iiif_presentation",
        iiif_version="2.1",
        portal_url="https://www.rijksmuseum.nl/en/research/collection-api",
        manifest_template="https://data.rijksmuseum.nl/object/{id}/manifest",
        image_service_template="https://lh3.googleusercontent.com/{id}",
        search_endpoint="https://www.rijksmuseum.nl/api/en/collection",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=10.0,
        notes="Unrivaled collection of Rembrandt, Frans Hals, and Vermeer with high-precision color profiling.",
        example_identifiers=["SK-C-5", "SK-A-399", "SK-A-2815"],
    ),
    "met": MuseumPreset(
        code="met",
        name="The Metropolitan Museum of Art",
        country="United States",
        city="New York",
        api_type="custom_rest",
        iiif_version="custom",
        portal_url="https://www.metmuseum.org",
        manifest_template="https://collectionapi.metmuseum.org/public/collection/v1/objects/{id}",
        image_service_template=None,
        search_endpoint="https://collectionapi.metmuseum.org/public/collection/v1/search",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=15.0,
        notes="400,000+ public-domain high-resolution artworks with dimensions and open REST API.",
        example_identifiers=["437984", "436535", "436533"],
    ),
    "nga": MuseumPreset(
        code="nga",
        name="National Gallery of Art",
        country="United States",
        city="Washington, D.C.",
        api_type="iiif_presentation",
        iiif_version="3.0",
        portal_url="https://www.nga.gov",
        manifest_template="https://media.nga.gov/iiif/public/objects/{id}/manifest.json",
        image_service_template="https://media.nga.gov/iiif/public/objects/{id}/full/full/0/default.jpg",
        search_endpoint="https://www.nga.gov/collection/collection-search.html",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=8.0,
        notes="High-fidelity color calibration charts; includes Vermeer's Girl with a Flute.",
        example_identifiers=["1942.9.97", "1937.1.40"],
    ),
    "national_gallery": MuseumPreset(
        code="national_gallery",
        name="The National Gallery",
        country="United Kingdom",
        city="London",
        api_type="iiif_presentation",
        iiif_version="2.0",
        portal_url="https://www.nationalgallery.org.uk",
        manifest_template="https://api.nationalgallery.org.uk/iiif/{id}/manifest.json",
        image_service_template="https://api.nationalgallery.org.uk/iiif/{id}",
        search_endpoint="https://www.nationalgallery.org.uk/paintings/search-the-collection",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=5.0,
        notes="Pivotal Leonardo (Virgin of the Rocks), Holbein, and Rembrandt holdings.",
        example_identifiers=["NG6376", "NG1093", "NG775"],
    ),
    "louvre": MuseumPreset(
        code="louvre",
        name="Musée du Louvre & C2RMF",
        country="France",
        city="Paris",
        api_type="custom_rest",
        iiif_version="custom",
        portal_url="https://collections.louvre.fr",
        manifest_template="https://collections.louvre.fr/ark:/53355/{id}.json",
        image_service_template="https://collections.louvre.fr/media/cache/original/{id}",
        search_endpoint="https://collections.louvre.fr/recherche",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=4.0,
        notes="Core corpus of Leonardo da Vinci, Raphael, and French Classical masters.",
        example_identifiers=["cl010062370", "cl010066723"],
    ),
    "smithsonian": MuseumPreset(
        code="smithsonian",
        name="Smithsonian Institution (Freer, SAAM, NPG)",
        country="United States",
        city="Washington, D.C.",
        api_type="iiif_image",
        iiif_version="2.1",
        portal_url="https://www.si.edu/openaccess",
        manifest_template=None,
        image_service_template="https://ids.si.edu/ids/iiif/{id}/info.json",
        search_endpoint="https://api.si.edu/openaccess/v1.0/search",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=10.0,
        notes="4.5M open-access records; extensive Whistler, Sargent, and American Gilded Age holdings.",
        example_identifiers=["F1904.61", "1909.7.35"],
    ),
    "artic": MuseumPreset(
        code="artic",
        name="Art Institute of Chicago",
        country="United States",
        city="Chicago",
        api_type="iiif_presentation",
        iiif_version="2.0",
        portal_url="https://www.artic.edu",
        manifest_template="https://api.artic.edu/api/v1/artworks/{id}/manifest.json",
        image_service_template="https://www.artic.edu/iiif/2/{image_id}/info.json",
        search_endpoint="https://api.artic.edu/api/v1/artworks/search",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=10.0,
        notes="Ultra-high-definition open IIIF Image API; canonical Impressionist and Post-Impressionist masterworks.",
        example_identifiers=["27992", "20684", "111628"],
    ),
    "gallica": MuseumPreset(
        code="gallica",
        name="Bibliothèque nationale de France (Gallica)",
        country="France",
        city="Paris",
        api_type="iiif_presentation",
        iiif_version="2.0",
        portal_url="https://gallica.bnf.fr",
        manifest_template="https://gallica.bnf.fr/iiif/ark:/12148/{id}/manifest.json",
        image_service_template="https://gallica.bnf.fr/iiif/ark:/12148/{id}/f1/info.json",
        search_endpoint="https://gallica.bnf.fr/services/engine/search/sru",
        supports_physical_dimensions=False,
        rate_limit_req_per_sec=5.0,
        notes="Illuminated manuscripts, rare drawings, engravings, and Renaissance prints.",
        example_identifiers=["btv1b550073228", "btv1b8451634x"],
    ),
    "prado": MuseumPreset(
        code="prado",
        name="Museo Nacional del Prado",
        country="Spain",
        city="Madrid",
        api_type="direct_deepzoom",
        iiif_version="custom",
        portal_url="https://www.museodelprado.es",
        manifest_template=None,
        image_service_template="https://content3.museodelprado.es/imagenes/{id}/info.json",
        search_endpoint="https://www.museodelprado.es/coleccion/obras-de-arte",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=3.0,
        notes="Velázquez, Goya, Bosch, and El Greco deep-zoom high-resolution gigapixel digitizations.",
        example_identifiers=["P001170", "P001174", "P000508"],
    ),
    "uffizi": MuseumPreset(
        code="uffizi",
        name="Gallerie degli Uffizi",
        country="Italy",
        city="Florence",
        api_type="direct_deepzoom",
        iiif_version="custom",
        portal_url="https://www.uffizi.it",
        manifest_template=None,
        image_service_template="https://digital.uffizi.it/iiif/{id}/info.json",
        search_endpoint="https://www.uffizi.it/opere",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=3.0,
        notes="Epicenter of the Italian Renaissance: Botticelli, Leonardo, Michelangelo, Caravaggio.",
        example_identifiers=["1890-001438", "1890-001472"],
    ),
    "getty": MuseumPreset(
        code="getty",
        name="J. Paul Getty Museum",
        country="United States",
        city="Los Angeles",
        api_type="iiif_presentation",
        iiif_version="2.0",
        portal_url="https://www.getty.edu/art/collection",
        manifest_template="https://media.getty.edu/iiif/manifest/{id}",
        image_service_template="https://media.getty.edu/iiif/image/{id}/info.json",
        search_endpoint="https://www.getty.edu/art/collection/search",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=10.0,
        notes="Getty Open Content Program; deep European painting collection with IIIF manifests.",
        example_identifiers=["877", "901", "102834"],
    ),
    "cleveland": MuseumPreset(
        code="cleveland",
        name="Cleveland Museum of Art",
        country="United States",
        city="Cleveland",
        api_type="custom_rest",
        iiif_version="2.0",
        portal_url="https://www.clevelandart.org/art/collection",
        manifest_template=None,
        image_service_template="https://openaccess-api.clevelandart.org/api/artworks/{id}",
        search_endpoint="https://openaccess-api.clevelandart.org/api/artworks",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=10.0,
        notes="Completely unrestricted Open Access API (Creative Commons Zero CC0).",
        example_identifiers=["138767", "154677"],
    ),
    "harvard": MuseumPreset(
        code="harvard",
        name="Harvard Art Museums",
        country="United States",
        city="Cambridge",
        api_type="iiif_image",
        iiif_version="2.0",
        portal_url="https://harvardartmuseums.org",
        manifest_template="https://api.harvardartmuseums.org/object/{id}/manifest",
        image_service_template="https://nrs.harvard.edu/urn-3:HUAM:{id}",
        search_endpoint="https://api.harvardartmuseums.org/object",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=8.0,
        notes="Straus Center for Conservation and Technical Studies; rich technical documentation.",
        example_identifiers=["299843", "299844"],
    ),
    "yale": MuseumPreset(
        code="yale",
        name="Yale Center for British Art",
        country="United States",
        city="New Haven",
        api_type="iiif_presentation",
        iiif_version="2.0",
        portal_url="https://britishart.yale.edu",
        manifest_template="https://manifests.collections.yale.edu/ycba/obj/{id}",
        image_service_template="https://images.collections.yale.edu/iiif/2/{id}/info.json",
        search_endpoint="https://collections.britishart.yale.edu",
        supports_physical_dimensions=True,
        rate_limit_req_per_sec=8.0,
        notes="Unrivaled collection of Turner, Constable, and British portraiture.",
        example_identifiers=["120", "121"],
    ),
}


def list_supported_museums() -> List[Dict[str, Any]]:
    """Returns a list of summary dictionaries for all supported museum connectors."""
    return [
        {
            "code": m.code,
            "name": m.name,
            "location": f"{m.city}, {m.country}",
            "api_type": m.api_type,
            "iiif_version": m.iiif_version,
            "portal_url": m.portal_url,
            "physical_dimensions": m.supports_physical_dimensions,
            "notes": m.notes,
        }
        for m in MUSEUM_REGISTRY.values()
    ]


def get_museum(code: str) -> Optional[MuseumPreset]:
    """Retrieves a museum preset by its unique institutional code."""
    return MUSEUM_REGISTRY.get(code.lower().strip())


def resolve_artwork_endpoint(museum_code: str, object_identifier: str) -> Dict[str, Any]:
    """
    Resolves the canonical endpoint (manifest URL or direct image API URL) for a given
    artwork accession ID or institutional record.
    """
    preset = get_museum(museum_code)
    if not preset:
        raise ValueError(
            f"Unsupported museum code '{museum_code}'. "
            f"Supported codes: {list(MUSEUM_REGISTRY.keys())}"
        )

    clean_id = str(object_identifier).strip()
    manifest_url = (
        preset.manifest_template.format(id=clean_id)
        if preset.manifest_template
        else None
    )
    image_url = (
        preset.image_service_template.format(id=clean_id, image_id=clean_id)
        if preset.image_service_template
        else None
    )

    return {
        "museum": preset.name,
        "code": preset.code,
        "api_type": preset.api_type,
        "identifier": clean_id,
        "manifest_url": manifest_url,
        "image_service_url": image_url,
        "rate_limit_req_per_sec": preset.rate_limit_req_per_sec,
    }


def extract_physical_scale_from_manifest(
    manifest_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Parses IIIF Presentation manifest services looking for the official
    IIIF Physical Dimensions Annex (`http://iiif.io/api/annex/services/physdim`).
    
    Returns:
        Dict with 'physical_scale', 'physical_units', and computed 'pixels_per_mm',
        or None if physical dimensions service is absent.
    """
    services = []

    # Check root service
    if "service" in manifest_data:
        srv = manifest_data["service"]
        services.extend(srv if isinstance(srv, list) else [srv])

    # Check canvas / sequence services
    sequences = manifest_data.get("sequences", [])
    for seq in sequences:
        canvases = seq.get("canvases", [])
        for c in canvases:
            if "service" in c:
                srv = c["service"]
                services.extend(srv if isinstance(srv, list) else [srv])

    # Check IIIF v3 items
    items = manifest_data.get("items", [])
    for itm in items:
        if "service" in itm:
            srv = itm["service"]
            services.extend(srv if isinstance(srv, list) else [srv])

    for s in services:
        if not isinstance(s, dict):
            continue
        profile = s.get("profile", "")
        if "physdim" in str(profile) or "physicalDimensions" in str(profile):
            scale = s.get("physicalScale")
            units = s.get("physicalUnits", "mm").lower()
            if scale is not None and scale > 0:
                # scale is physical dimension per pixel (e.g. 0.05 mm per pixel)
                # pixels_per_unit = 1.0 / scale
                factor_to_mm = 1.0
                if units in ("cm", "centimeter", "centimeters"):
                    factor_to_mm = 10.0
                elif units in ("in", "inch", "inches"):
                    factor_to_mm = 25.4

                mm_per_pixel = scale * factor_to_mm
                pixels_per_mm = 1.0 / mm_per_pixel if mm_per_pixel > 0 else None

                return {
                    "physical_scale": scale,
                    "physical_units": units,
                    "mm_per_pixel": mm_per_pixel,
                    "pixels_per_mm": pixels_per_mm,
                }

    return None
