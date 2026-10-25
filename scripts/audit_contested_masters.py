"""
Connoisseurship Audit CLI for Contested Masters.
Audits the complete mono-artist corpora of 12 high-controversy master workshops:
- Johannes Vermeer
- Rembrandt van Rijn
- Frans Hals
- Lucas Cranach the Elder
- Leonardo da Vinci
- Peter Paul Rubens
- Titian (Tiziano Vecellio)
- El Greco
- Raphael (Raffaello Sanzio)
- Francisco de Goya
- Diego Velázquez
- Hieronymus Bosch

Ranks works by Mahalanobis distance to the master's manifold, isolates statistical anomalies ('moutons noirs'),
highlights connoisseurship divergences (canon works demoted, studio works promoted), and exports an interactive Connoisseurship Report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition.contested_masters_harvester import ContestedMastersHarvester
from src.attribution.artist_corpus_auditor import ArtistCorpusAuditor, ArtistCorpusAuditSummary


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArtStylometry — Contested Masters Connoisseurship Audit</title>
    <style>
        :root {
            --bg-base: #0b1120;
            --bg-surface: #131d32;
            --bg-card: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-purple: #c084fc;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-base); color: var(--text-primary); padding: 2rem; line-height: 1.5; }
        
        .header { margin-bottom: 2rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1.5rem; }
        .header h1 { font-size: 2.1rem; color: var(--accent-cyan); display: flex; align-items: center; gap: 0.75rem; }
        .header p { color: var(--text-secondary); margin-top: 0.5rem; font-size: 0.95rem; max-width: 1000px; line-height: 1.6; }

        /* FILTER TOOLBAR */
        .filters-container {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem 1.75rem;
            margin-bottom: 2.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .filter-row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
        .filter-label { font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); margin-right: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
        
        .artist-pill, .status-pill {
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .artist-pill:hover, .status-pill:hover { background: #334155; color: #fff; }
        .artist-pill.active, .status-pill.active { background: #38bdf8; color: #0b1120; border-color: #38bdf8; font-weight: 700; }
        .status-pill-divergence.active { background: #c084fc; color: #0b1120; border-color: #c084fc; }
        .status-pill-demoted.active { background: #f43f5e; color: #fff; border-color: #f43f5e; }
        .status-pill-promoted.active { background: #10b981; color: #0b1120; border-color: #10b981; }

        .search-input {
            flex: 1;
            min-width: 250px;
            background: #1e293b;
            color: #fff;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.85rem;
        }

        /* ARTIST SECTION */
        .artist-section { margin-bottom: 3.5rem; background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }
        .artist-header { padding: 1.25rem 1.75rem; background: linear-gradient(90deg, #162035 0%, #1e293b 100%); border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
        .artist-title h2 { font-size: 1.5rem; color: #fff; }
        .artist-meta { font-size: 0.8rem; color: var(--accent-cyan); margin-top: 0.25rem; }
        
        .stat-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; }
        .stat-badge { padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }
        .stat-badge:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.4); }
        .badge-total { background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
        .badge-canonical { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid var(--accent-emerald); }
        .badge-divergence { background: rgba(192, 132, 252, 0.2); color: var(--accent-purple); border: 1px solid var(--accent-purple); }
        .badge-outlier { background: rgba(244, 63, 94, 0.18); color: var(--accent-rose); border: 1px solid var(--accent-rose); }

        /* CARDS GRID */
        .cards-grid { padding: 1.75rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1.5rem; }
        .art-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; transition: transform 0.2s, border-color 0.2s; position: relative; }
        .art-card:hover { transform: translateY(-3px); }
        .art-card.is-mouton-noir { border: 2px solid var(--accent-rose); box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15); }
        .art-card.is-divergence-promoted { border: 2px solid var(--accent-emerald); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15); }

        .card-img-wrapper { height: 230px; background: #0f172a; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        .card-img { width: 100%; height: 100%; object-fit: cover; }
        
        .status-tag { position: absolute; top: 10px; right: 10px; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; z-index: 2; }
        .tag-consensus { background: rgba(16, 185, 129, 0.92); color: #fff; }
        .tag-studio { background: rgba(245, 158, 11, 0.92); color: #0b1120; }
        .tag-outlier { background: rgba(244, 63, 94, 0.95); color: #fff; }
        .rank-tag { position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); color: #f8fafc; font-size: 0.72rem; font-weight: 800; padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; z-index: 2; }

        /* DIVERGENCE BANNER */
        .divergence-banner {
            padding: 6px 12px;
            font-size: 0.72rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            letter-spacing: 0.02em;
        }
        .banner-demoted { background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); border-bottom: 1px solid rgba(244, 63, 94, 0.4); }
        .banner-promoted { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border-bottom: 1px solid rgba(16, 185, 129, 0.4); }

        .card-content { padding: 1.25rem; display: flex; flex-direction: column; flex: 1; gap: 0.65rem; }
        .card-title { font-size: 1.05rem; font-weight: 700; color: #fff; line-height: 1.3; }
        .card-attr { font-size: 0.8rem; color: #e2e8f0; font-weight: 600; }
        .card-museum { font-size: 0.75rem; color: var(--text-secondary); }
        
        .gauge-container { margin: 0.4rem 0; }
        .gauge-track { height: 6px; background: #0f172a; border-radius: 3px; overflow: hidden; }
        .gauge-fill { height: 100%; }
        .gauge-labels { display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-secondary); margin-top: 2px; }
        
        .diag-box { background: rgba(15, 23, 42, 0.6); padding: 0.6rem 0.75rem; border-radius: 6px; font-size: 0.75rem; color: #cbd5e1; border-left: 3px solid var(--accent-cyan); margin-top: auto; }
        .diag-box.is-mouton { border-left-color: var(--accent-rose); background: rgba(244, 63, 94, 0.08); }
        .diag-box.is-promoted { border-left-color: var(--accent-emerald); background: rgba(16, 185, 129, 0.08); }
        .mouton-alert { color: var(--accent-rose); font-weight: 800; display: flex; align-items: center; gap: 0.3rem; margin-bottom: 0.2rem; }
        .promoted-alert { color: var(--accent-emerald); font-weight: 800; display: flex; align-items: center; gap: 0.3rem; margin-bottom: 0.2rem; }

        .heatmap-btn { position: absolute; bottom: 8px; right: 8px; background: rgba(14, 165, 233, 0.9); color: white; font-size: 0.7rem; font-weight: 700; padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.3); cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.5); transition: all 0.2s ease; z-index: 5; }
        .heatmap-btn:hover { background: #0284c7; transform: translateY(-1px); }
        .heatmap-btn.active { background: #e11d48; }
        .hm-badge { display: inline-block; font-size: 0.68rem; background: rgba(14, 165, 233, 0.15); color: var(--accent-cyan); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 4px; padding: 2px 6px; margin-top: 2px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Quantitative Connoisseurship: Contested Masters Audit</h1>
        <p>Mono-artist corpus isolation and multi-hand attribution across 12 master workshops (Johannes Vermeer, Rembrandt van Rijn, Frans Hals, Lucas Cranach, Leonardo da Vinci, Peter Paul Rubens, Titian, El Greco, Raphael, Francisco de Goya, Diego Velázquez, Hieronymus Bosch). Each corpus is projected into its master manifold via Ledoit-Wolf shrinkage covariance to objectively separate canonical touch from workshop pupils, followers, and modern interventions.</p>
    </div>

    <!-- TOP FILTER TOOLBAR -->
    <div class="filters-container">
        <!-- ROW 1: MASTER ATELIER SELECTOR -->
        <div class="filter-row">
            <span class="filter-label">👨‍🎨 Atelier:</span>
            <button class="artist-pill active" onclick="filterByArtist('all', this)">All 12 Ateliers (1,137)</button>
            <button class="artist-pill" onclick="filterByArtist('Johannes Vermeer', this)">Vermeer (52)</button>
            <button class="artist-pill" onclick="filterByArtist('Rembrandt van Rijn', this)">Rembrandt (223)</button>
            <button class="artist-pill" onclick="filterByArtist('Peter Paul Rubens', this)">Rubens (204)</button>
            <button class="artist-pill" onclick="filterByArtist('Lucas Cranach', this)">Cranach (141)</button>
            <button class="artist-pill" onclick="filterByArtist('Titian', this)">Titian (116)</button>
            <button class="artist-pill" onclick="filterByArtist('Francisco de Goya', this)">Goya (111)</button>
            <button class="artist-pill" onclick="filterByArtist('Raphael', this)">Raphael (83)</button>
            <button class="artist-pill" onclick="filterByArtist('Diego Velázquez', this)">Velázquez (64)</button>
            <button class="artist-pill" onclick="filterByArtist('Frans Hals', this)">Hals (58)</button>
            <button class="artist-pill" onclick="filterByArtist('Hieronymus Bosch', this)">Bosch (42)</button>
            <button class="artist-pill" onclick="filterByArtist('Leonardo da Vinci', this)">Leonardo da Vinci (26)</button>
            <button class="artist-pill" onclick="filterByArtist('El Greco', this)">El Greco (17)</button>
        </div>

        <!-- ROW 2: CONNOISSEURSHIP FINDINGS & DIVERGENCES -->
        <div class="filter-row" style="border-top: 1px solid #1e293b; padding-top: 0.75rem;">
            <span class="filter-label">⚖️ Verdict:</span>
            <button class="status-pill active" onclick="filterByStatus('all', this)">All Works</button>
            <button class="status-pill status-pill-divergence" onclick="filterByStatus('Divergences', this)">⚡ All Divergences (108)</button>
            <button class="status-pill status-pill-demoted" onclick="filterByStatus('Demoted', this)">⚠️ Canon Flagged as Anomaly (91)</button>
            <button class="status-pill status-pill-promoted" onclick="filterByStatus('Promoted', this)">✨ Disputed Promoted to Master (17)</button>
            <button class="status-pill" onclick="filterByStatus('Canonical Master', this)">🏛️ Consensus Autograph (902)</button>
            <button class="status-pill" onclick="filterByStatus('Studio / Assistant', this)">👥 Consensus Studio (127)</button>
            
            <input type="text" id="auditSearchInput" class="search-input" placeholder="🔍 Search title, museum, year, rationale..." oninput="handleAuditSearch()" />
            <div id="filterCountBadge" style="font-size:0.85rem; font-weight:700; color:var(--accent-cyan);">1,137 Artworks Displayed</div>
        </div>
    </div>

    <!-- ARTIST SECTIONS -->
    {% for summary in summaries %}
    <div class="artist-section" data-artist="{{ summary.artist_name }}">
        <div class="artist-header">
            <div class="artist-title">
                <h2>{{ summary.artist_name }}</h2>
                <div class="artist-meta">Manifold Shrinkage: {{ "%.4f"|format(summary.reference_shrinkage) }} &bull; Outlier Threshold: {{ "%.2f"|format(summary.outlier_distance_threshold) }} M-dist</div>
            </div>
            <div class="stat-badges">
                <span class="stat-badge badge-total" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'all')">{{ summary.total_artworks_evaluated }} Audited</span>
                <span class="stat-badge badge-canonical" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Canonical Master')">{{ summary.consensus_autograph_count }} Canonical Master</span>
                <span class="stat-badge badge-divergence" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Divergences')">⚡ {{ summary.high_controversy_count }} Controversies</span>
                <span class="stat-badge badge-outlier" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Demoted')">⚠️ {{ summary.black_sheep_outlier_count }} Outliers</span>
            </div>
        </div>

        <div class="cards-grid">
            {% for art in summary.ranked_artworks %}
            <div class="art-card {% if art.is_black_sheep %}is-mouton-noir{% endif %} {% if art.divergence_type == 'PROMOTED_AUTOGRAPH' %}is-divergence-promoted{% endif %}" data-divergence="{{ art.divergence_type }}">
                
                {% if art.divergence_type == 'DEMOTED_ANOMALY' %}
                <div class="divergence-banner banner-demoted">
                    ⚠️ HIGH-CONTROVERSY: Museum Canon Flagged as Statistical Outlier (p={{ "%.3f"|format(art.conformal_p_value) }})
                </div>
                {% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}
                <div class="divergence-banner banner-promoted">
                    ✨ RE-ATTRIBUTION: Studio/Disputed Re-Evaluated as Autograph Quality ({{ (art.attribution_confidence * 100)|round }}%)
                </div>
                {% endif %}

                <div class="card-img-wrapper" id="wrap-{{ art.artwork_id }}">
                    <img src="{{ art.image_display_path }}" alt="{{ art.title }}" class="card-img" id="img-{{ art.artwork_id }}" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\' fill=\\'%231e293b\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%230f172a\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2364748b\\' font-size=\\'12\\' text-anchor=\\'middle\\'>Preview Unavailable</text></svg>';" />
                    <span class="rank-tag">#{{ loop.index }}</span>
                    {% if art.verdict == 'CONSENSUS_MASTERWORK' %}
                    <span class="status-tag tag-consensus">Canonical Master</span>
                    {% elif art.verdict == 'STUDIO_WORKSHOP_HAND' %}
                    <span class="status-tag tag-studio">Studio / Assistant</span>
                    {% else %}
                    <span class="status-tag tag-outlier">⚠️ Black Sheep</span>
                    {% endif %}
                    {% if art.has_heatmap %}
                    <button class="heatmap-btn" onclick="toggleHeatmap('{{ art.artwork_id }}', '{{ art.image_display_path }}', '{{ art.heatmap_display_path }}', this)">🗺️ 2D Heatmap</button>
                    {% endif %}
                </div>
                <div class="card-content">
                    <div class="card-title">{{ art.title }}</div>
                    <div class="card-attr">{{ art.official_attribution }} ({{ art.creation_date }})</div>
                    <div class="card-museum">🏛️ {{ art.museum }}</div>
                    {% if art.has_heatmap %}
                    <div><span class="hm-badge">🗺️ 2D Intra-Canvas Heatmap Available</span></div>
                    {% endif %}

                    <div class="gauge-container">
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width: {{ (art.attribution_confidence * 100)|round }}%; background: {% if art.attribution_confidence >= 0.70 %}var(--accent-emerald){% elif art.attribution_confidence >= 0.40 %}var(--accent-amber){% else %}var(--accent-rose){% endif %};"></div>
                        </div>
                        <div class="gauge-labels">
                            <span>Confidence: {{ (art.attribution_confidence * 100)|round }}%</span>
                            <span>M-Dist: {{ "%.2f"|format(art.mahalanobis_distance) }} (p={{ "%.3f"|format(art.conformal_p_value) }})</span>
                        </div>
                        <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 3px; display: flex; justify-content: space-between;">
                            <span>Ductus Coherence: {{ "%.2f"|format(art.ductus_coherence) }}</span>
                            <span>Stroke Entropy: {{ "%.2f"|format(art.ductus_entropy) }}</span>
                        </div>
                    </div>

                    <div class="diag-box {% if art.is_black_sheep %}is-mouton{% endif %} {% if art.divergence_type == 'PROMOTED_AUTOGRAPH' %}is-promoted{% endif %}">
                        {% if art.is_black_sheep %}
                        <div class="mouton-alert">⚠️ STATISTICAL ANOMALY / DEPARTURE</div>
                        {% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}
                        <div class="promoted-alert">✨ REDISCOVERED MASTER TOUCH</div>
                        {% endif %}
                        {{ art.diagnostic_rationale }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}

    <!-- CLIENT CONTROLLER SCRIPT -->
    <script>
    function resolveSrc(p) {
        if (!p) return '';
        if (location.protocol.startsWith('http')) {
            return p.replace(/^[a-zA-Z]:\\/.*?\\/ArtStylometry\\//, '/').replace(/^\\/?/, '/');
        }
        return p.startsWith('file:///') ? p : ('file:///' + p);
    }

    function toggleHeatmap(artId, origSrc, hmSrc, btn) {
        const img = document.getElementById('img-' + artId);
        if (!img) return;
        if (btn.classList.contains('active')) {
            img.src = origSrc;
            btn.classList.remove('active');
            btn.innerText = '🗺️ 2D Heatmap';
        } else {
            img.src = hmSrc;
            btn.classList.add('active');
            btn.innerText = '🖼️ Original Canvas';
        }
    }

    let currentArtist = 'all';
    let currentStatus = 'all';
    let currentQuery = '';

    function filterByArtist(artist, btn) {
        document.querySelectorAll('.artist-pill').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');
        currentArtist = artist;
        applyFilters();
    }

    function filterByStatus(status, btn) {
        document.querySelectorAll('.status-pill').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');
        currentStatus = status;
        applyFilters();
    }

    function filterByArtistAndStatus(artist, status) {
        // Select artist pill
        document.querySelectorAll('.artist-pill').forEach(b => {
            if (b.textContent.toLowerCase().includes(artist.toLowerCase().split(' ')[0])) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });
        currentArtist = artist;

        // Select status pill
        document.querySelectorAll('.status-pill').forEach(b => {
            if (status === 'all' && b.textContent.includes('All Works')) b.classList.add('active');
            else if (status === 'Canonical Master' && b.textContent.includes('Consensus Autograph')) b.classList.add('active');
            else if (status === 'Divergences' && b.textContent.includes('All Divergences')) b.classList.add('active');
            else if (status === 'Demoted' && b.textContent.includes('Canon Flagged')) b.classList.add('active');
            else b.classList.remove('active');
        });
        currentStatus = status;
        applyFilters();

        // Scroll to artist section
        const targetSec = Array.from(document.querySelectorAll('.artist-section')).find(s => s.getAttribute('data-artist').toLowerCase().includes(artist.toLowerCase()));
        if (targetSec) targetSec.scrollIntoView({ behavior: 'smooth' });
    }

    function handleAuditSearch() {
        currentQuery = document.getElementById('auditSearchInput').value.toLowerCase().trim();
        applyFilters();
    }

    function applyFilters() {
        let totalVisible = 0;
        const sections = document.querySelectorAll('.artist-section');
        sections.forEach(sec => {
            const artistName = sec.getAttribute('data-artist');
            const artistMatch = (currentArtist === 'all' || artistName.toLowerCase().includes(currentArtist.toLowerCase()));
            
            if (!artistMatch) {
                sec.style.display = 'none';
                return;
            }

            let sectionVisibleCards = 0;
            const cards = sec.querySelectorAll('.art-card');
            cards.forEach(card => {
                const title = card.querySelector('.card-title').textContent.toLowerCase();
                const museum = card.querySelector('.card-museum').textContent.toLowerCase();
                const attr = card.querySelector('.card-attr').textContent.toLowerCase();
                const statusTag = card.querySelector('.status-tag').textContent.trim();
                const divType = card.getAttribute('data-divergence');

                let statusMatch = (currentStatus === 'all');
                if (currentStatus === 'Canonical Master') statusMatch = statusTag.includes('Canonical');
                else if (currentStatus === 'Studio / Assistant') statusMatch = statusTag.includes('Studio');
                else if (currentStatus === 'Mouton Noir' || currentStatus === 'Demoted') statusMatch = (divType === 'DEMOTED_ANOMALY' || card.classList.contains('is-mouton-noir'));
                else if (currentStatus === 'Promoted') statusMatch = (divType === 'PROMOTED_AUTOGRAPH');
                else if (currentStatus === 'Divergences') statusMatch = (divType === 'DEMOTED_ANOMALY' || divType === 'PROMOTED_AUTOGRAPH');

                let queryMatch = (!currentQuery || title.includes(currentQuery) || museum.includes(currentQuery) || attr.includes(currentQuery));

                if (statusMatch && queryMatch) {
                    card.style.display = '';
                    sectionVisibleCards++;
                    totalVisible++;
                } else {
                    card.style.display = 'none';
                }
            });

            sec.style.display = (sectionVisibleCards > 0) ? '' : 'none';
        });

        document.getElementById('filterCountBadge').textContent = totalVisible.toLocaleString() + ' Artworks Displayed';
    }
    </script>
</body>
</html>
"""


def export_html_report(summaries: list[ArtistCorpusAuditSummary], output_path: str | Path = "data/processed/contested_masters_audit.html") -> Path:
    from jinja2 import Template
    t = Template(HTML_REPORT_TEMPLATE)
    html = t.render(summaries=summaries)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")
    return out_file


def export_audit_tabular_data(summaries: list[ArtistCorpusAuditSummary], output_dir: str | Path = "data/processed") -> dict[str, Path]:
    import pandas as pd
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for s in summaries:
        for idx, a in enumerate(s.ranked_artworks):
            records.append({
                "artist_canonical": s.artist_name,
                "rank_in_corpus": idx + 1,
                "artwork_id": a.artwork_id,
                "title": a.title,
                "official_attribution": a.official_attribution,
                "attribution_category": a.attribution_category,
                "creation_date": a.creation_date,
                "medium": a.medium,
                "dimensions": str(a.dimensions),
                "museum": a.museum,
                "mahalanobis_distance": round(float(a.mahalanobis_distance), 4),
                "attribution_confidence": round(float(a.attribution_confidence), 4),
                "craquelure_cleanliness": round(float(a.craquelure_cleanliness), 4),
                "gradient_energy": round(float(a.gradient_energy), 4),
                "ductus_coherence": round(float(a.ductus_coherence), 4),
                "ductus_entropy": round(float(a.ductus_entropy), 4),
                "conformal_p_value": round(float(a.conformal_p_value), 4),
                "verdict": a.verdict,
                "is_black_sheep": bool(a.is_black_sheep),
                "divergence_type": a.divergence_type,
                "diagnostic_rationale": a.diagnostic_rationale,
                "anomaly_zones_count": int(a.anomaly_zones_count),
                "spatial_anomaly_pct": round(float(a.spatial_anomaly_pct), 2),
                "has_heatmap": bool(a.has_heatmap),
                "heatmap_display_path": str(a.heatmap_display_path),
                "reference_shrinkage": round(float(s.reference_shrinkage), 6),
                "reference_condition_number": round(float(s.reference_condition_number), 2),
                "outlier_threshold": round(float(s.outlier_distance_threshold), 4),
                "local_path": str(a.local_path),
            })

    df = pd.DataFrame(records)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    parquet_file = out_dir / "contested_masters_audit.parquet"
    csv_file = out_dir / "contested_masters_audit.csv"
    json_file = out_dir / "contested_masters_audit.json"

    df.to_parquet(parquet_file, index=False)
    df.to_csv(csv_file, index=False, encoding="utf-8")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return {
        "parquet": parquet_file,
        "csv": csv_file,
        "json": json_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit mono-artist corpora for contested masters")
    parser.add_argument(
        "--artists",
        nargs="+",
        default=["vermeer", "hals", "cranach", "rembrandt", "leonardo", "rubens", "titian", "greco", "raphael", "goya", "velazquez", "bosch"],
        help="Artists to audit",
    )
    parser.add_argument("--skip-harvest", action="store_true", help="Skip harvesting if already downloaded")
    parser.add_argument("--output-report", default="data/processed/contested_masters_audit.html", help="HTML report output path")

    args = parser.parse_args()

    harvester = ContestedMastersHarvester()
    auditor = ArtistCorpusAuditor()

    summaries = []

    for key in args.artists:
        if not args.skip_harvest:
            harvester.harvest_corpus_for_artist(key)

        corpus_dir = Path("data/raw/contested_masters") / key
        catalog_file = corpus_dir / "catalog.json"

        if not catalog_file.exists():
            print(f"Skipping {key}: no catalog found at {catalog_file}")
            continue

        summary = auditor.audit_corpus(corpus_dir, catalog_file)
        summaries.append(summary)

        print(f"\\n[SUMMARY: {summary.artist_name}]")
        print(f"  Total Paintings   : {summary.total_artworks_evaluated}")
        print(f"  Canonical Core    : {summary.consensus_autograph_count}")
        print(f"  Studio Hands      : {summary.studio_workshop_count}")
        print(f"  Moutons Noirs     : {summary.black_sheep_outlier_count}")
        print(f"  Divergences       : {summary.high_controversy_count}")

    report_path = export_html_report(summaries, args.output_report)
    tabular_paths = export_audit_tabular_data(summaries, Path(args.output_report).parent)
    print(f"\\n{'='*70}\\n[OK] Connoisseurship report successfully generated: {report_path.resolve()}")
    print(f"[OK] Parquet dataset generated: {tabular_paths['parquet'].resolve()}")
    print(f"[OK] CSV dataset generated:     {tabular_paths['csv'].resolve()}")
    print(f"[OK] JSON dataset generated:    {tabular_paths['json'].resolve()}\\n{'='*70}")


if __name__ == "__main__":
    main()
