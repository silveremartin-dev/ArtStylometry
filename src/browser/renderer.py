"""
ArtStylometry Interactive Artwork & Connoisseurship Visual Browser.
Generates a standalone, zero-dependency HTML application allowing users to
browse museum holdings (Louvre, Smithsonian, Rijksmuseum, NGA), sort painters
chronologically or alphabetically, inspect attribution confidence, and explore
famous historical misattribution case studies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from jinja2 import Template

from .catalog_data import ArtistCatalogEntry, ArtworkRecord, get_curated_artists_catalog


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArtStylometry — Museum Artwork Browser & Connoisseurship Navigator</title>
    <style>
        :root {
            --bg-base: #0b1120;
            --bg-surface: #131d32;
            --bg-card: #1e293b;
            --bg-hover: #293548;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-dim: #64748b;
            --accent-cyan: #38bdf8;
            --accent-gold: #fbbf24;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-purple: #a855f7;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background: var(--bg-base);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* SIDEBAR (Artists Navigation) */
        #sidebar {
            width: 380px;
            background: var(--bg-surface);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }

        .sidebar-header {
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            background: rgba(15, 23, 42, 0.6);
        }

        .sidebar-header h1 {
            font-size: 1.25rem;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .sidebar-header p {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.35rem;
            line-height: 1.3;
        }

        /* Controls & Sorting */
        .controls-panel {
            padding: 0.85rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-surface);
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }

        .search-row {
            display: flex;
            gap: 0.5rem;
        }

        .search-input {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            color: #fff;
            font-size: 0.85rem;
            outline: none;
        }

        .search-input:focus {
            border-color: var(--accent-cyan);
        }

        .sort-row {
            display: flex;
            gap: 0.4rem;
        }

        .btn-sort {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.4rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            text-align: center;
            transition: all 0.15s ease;
        }

        .btn-sort.active, .btn-sort:hover {
            background: var(--accent-cyan);
            color: #0b1120;
            border-color: var(--accent-cyan);
        }

        /* Artist List */
        .artist-list {
            flex: 1;
            overflow-y: auto;
            list-style: none;
        }

        .artist-item {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            cursor: pointer;
            transition: background 0.15s ease;
        }

        .artist-item:hover {
            background: var(--bg-hover);
        }

        .artist-item.active {
            background: var(--bg-card);
            border-left: 4px solid var(--accent-cyan);
        }

        .artist-name {
            font-size: 0.95rem;
            font-weight: 700;
            color: #f1f5f9;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .artist-meta {
            font-size: 0.75rem;
            color: var(--accent-gold);
            margin-top: 0.25rem;
        }

        .artist-stats {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.4rem;
            font-size: 0.7rem;
        }

        .badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .badge-canonical {
            background: rgba(16, 185, 129, 0.18);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .badge-contested {
            background: rgba(244, 63, 94, 0.18);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.35);
        }

        .badge-museum {
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.25);
        }

        /* MAIN WORKSPACE */
        #workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            background: var(--bg-base);
        }

        /* Artist Banner */
        .artist-banner {
            background: linear-gradient(180deg, #162035 0%, #0f172a 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .banner-info h2 {
            font-size: 1.7rem;
            color: #fff;
        }

        .banner-subtitle {
            font-size: 0.85rem;
            color: var(--accent-cyan);
            margin-top: 0.3rem;
            display: flex;
            gap: 0.75rem;
        }

        .banner-bio {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.6rem;
            max-width: 800px;
            line-height: 1.45;
        }

        .filter-tabs {
            display: flex;
            gap: 0.5rem;
            margin-top: 1.25rem;
        }

        .tab-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.4rem 0.85rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }

        .tab-btn.active, .tab-btn:hover {
            background: var(--accent-cyan);
            color: #0b1120;
            border-color: var(--accent-cyan);
        }

        /* Artwork Catalog Grid */
        .artworks-grid {
            padding: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
            gap: 1.75rem;
        }

        .artwork-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, border-color 0.2s;
        }

        .artwork-card:hover {
            transform: translateY(-2px);
            border-color: #475569;
        }

        .artwork-card.is-controversy {
            border: 1px solid rgba(244, 63, 94, 0.45);
            box-shadow: 0 4px 20px rgba(244, 63, 94, 0.08);
        }

        /* Canvas & Heatmap Viewport */
        .canvas-viewport {
            position: relative;
            width: 100%;
            height: 320px;
            background: #050811;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .canvas-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        .heatmap-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.0;
            transition: opacity 0.25s ease;
            mix-blend-mode: screen;
        }

        .anomaly-box {
            position: absolute;
            border: 2px dashed var(--accent-rose);
            background: rgba(244, 63, 94, 0.15);
            pointer-events: auto;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(244, 63, 94, 0.5);
        }

        .anomaly-tag {
            position: absolute;
            top: -20px;
            left: 0;
            background: var(--accent-rose);
            color: #fff;
            font-size: 0.65rem;
            font-weight: 800;
            padding: 2px 5px;
            border-radius: 3px;
            white-space: nowrap;
        }

        /* Card Content */
        .card-body {
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            flex: 1;
            gap: 0.85rem;
        }

        .card-header-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.5rem;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
            line-height: 1.3;
        }

        .confidence-pill {
            padding: 4px 9px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .card-metadata {
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Scientific Metrics Bar */
        .metrics-bar {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            background: var(--bg-card);
            padding: 0.6rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        .metric-cell {
            text-align: center;
        }

        .metric-cell-label {
            font-size: 0.65rem;
            color: var(--text-dim);
            text-transform: uppercase;
            font-weight: 700;
        }

        .metric-cell-val {
            font-size: 0.85rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-top: 2px;
        }

        /* Case Study Banner */
        .case-study-box {
            background: rgba(244, 63, 94, 0.08);
            border: 1px solid rgba(244, 63, 94, 0.3);
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
        }

        .case-study-title {
            color: var(--accent-rose);
            font-weight: 800;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .case-study-text {
            color: #cbd5e1;
            line-height: 1.4;
        }

        /* Interactive Heatmap Controls */
        .card-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 0.75rem;
            margin-top: auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .heatmap-toggle-wrapper {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .heatmap-slider {
            width: 100px;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        .museum-link {
            font-size: 0.75rem;
            color: var(--accent-cyan);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-weight: 600;
        }

        .museum-link:hover {
            text-decoration: underline;
        }

        /* Empty State */
        .empty-state {
            grid-column: 1 / -1;
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-dim);
        }
    </style>
</head>
<body>

    <!-- SIDEBAR: ARTISTS DIRECTORY -->
    <aside id="sidebar">
        <div class="sidebar-header">
            <h1>🎨 ArtStylometry</h1>
            <p>Quantitative Connoisseurship & Attribution Explorer &bull; Louvre & Smithsonian Collections</p>
        </div>

        <div class="controls-panel">
            <div class="search-row">
                <input type="text" id="artistSearchInput" class="search-input" placeholder="Search painter, period, museum..." />
            </div>
            <div class="sort-row">
                <button class="btn-sort active" id="btnSortChronological" onclick="setArtistSorting('chronological')">
                    ⏳ Chronological (Epoch)
                </button>
                <button class="btn-sort" id="btnSortAlphabetical" onclick="setArtistSorting('alphabetical')">
                    🔤 Alphabetical (A-Z)
                </button>
            </div>
        </div>

        <ul class="artist-list" id="artistListContainer">
            <!-- Rendered dynamically by JS -->
        </ul>
    </aside>

    <!-- MAIN WORKSPACE: CORPUS & ATTRIBUTION AUDIT -->
    <main id="workspace">
        <div class="artist-banner" id="artistBanner">
            <div class="banner-info">
                <h2 id="bannerArtistName">Artist Name</h2>
                <div class="banner-subtitle">
                    <span id="bannerArtistPeriod">Period</span> &bull; 
                    <span id="bannerArtistDates">Dates</span> &bull; 
                    <span id="bannerArtistMuseum">Primary Museum</span>
                </div>
                <p class="banner-bio" id="bannerArtistBio">Biography and connoisseurship notes...</p>
                <div class="filter-tabs">
                    <button class="tab-btn active" onclick="setArtworkFilter('all')">All Artworks (<span id="countAll">0</span>)</button>
                    <button class="tab-btn" onclick="setArtworkFilter('canonical')">Consensus Masterworks (<span id="countCanonical">0</span>)</button>
                    <button class="tab-btn" onclick="setArtworkFilter('cases')">⭐ Cas d'école / Controversies (<span id="countCases">0</span>)</button>
                    <button class="tab-btn" onclick="setArtworkFilter('anomalies')">Suspected Outliers &lt; 50% (<span id="countAnomalies">0</span>)</button>
                </div>
            </div>
        </div>

        <div class="artworks-grid" id="artworksGrid">
            <!-- Artworks populated dynamically by JS -->
        </div>
    </main>

    <!-- EMBEDDED JSON CATALOG -->
    <script id="catalogData" type="application/json">
        {{ catalog_json | safe }}
    </script>

    <script>
        const catalog = JSON.parse(document.getElementById('catalogData').textContent);
        let currentSort = 'chronological';
        let currentFilter = 'all';
        let selectedArtistId = catalog.length > 0 ? catalog[0].id : null;
        let searchQuery = '';

        // DOM elements
        const artistListContainer = document.getElementById('artistListContainer');
        const searchInput = document.getElementById('artistSearchInput');
        const btnSortChrono = document.getElementById('btnSortChronological');
        const btnSortAlpha = document.getElementById('btnSortAlphabetical');
        const artworksGrid = document.getElementById('artworksGrid');

        // Banner elements
        const bannerName = document.getElementById('bannerArtistName');
        const bannerPeriod = document.getElementById('bannerArtistPeriod');
        const bannerDates = document.getElementById('bannerArtistDates');
        const bannerMuseum = document.getElementById('bannerArtistMuseum');
        const bannerBio = document.getElementById('bannerArtistBio');

        // Initialize search listener
        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase().trim();
            renderArtistList();
        });

        function setArtistSorting(mode) {
            currentSort = mode;
            if (mode === 'chronological') {
                btnSortChrono.classList.add('active');
                btnSortAlpha.classList.remove('active');
            } else {
                btnSortAlpha.classList.add('active');
                btnSortChrono.classList.remove('active');
            }
            renderArtistList();
        }

        function setArtworkFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderArtworkGrid();
        }

        function getSortedArtists() {
            let artists = [...catalog];

            if (searchQuery) {
                artists = artists.filter(a => 
                    a.name.toLowerCase().includes(searchQuery) ||
                    a.period.toLowerCase().includes(searchQuery) ||
                    a.primary_museum.toLowerCase().includes(searchQuery) ||
                    a.artworks.some(art => art.title.toLowerCase().includes(searchQuery))
                );
            }

            if (currentSort === 'chronological') {
                artists.sort((a, b) => a.birth_year - b.birth_year);
            } else {
                artists.sort((a, b) => a.name.localeCompare(b.name));
            }

            return artists;
        }

        function renderArtistList() {
            const artists = getSortedArtists();
            artistListContainer.innerHTML = '';

            if (artists.length === 0) {
                artistListContainer.innerHTML = '<li style="padding: 2rem; text-align: center; color: var(--text-dim); font-size: 0.85rem;">No painters found matching search.</li>';
                return;
            }

            // Ensure selection exists in filtered list
            if (!artists.some(a => a.id === selectedArtistId)) {
                selectedArtistId = artists[0].id;
            }

            artists.forEach(artist => {
                const li = document.createElement('li');
                li.className = `artist-item ${artist.id === selectedArtistId ? 'active' : ''}`;
                li.onclick = () => selectArtist(artist.id);

                li.innerHTML = `
                    <div class="artist-name">
                        <span>${artist.name}</span>
                        ${artist.contested_count > 0 ? `<span class="badge badge-contested" title="${artist.contested_count} famous case studies / controversies">⚠️ ${artist.contested_count} Cas d'école</span>` : ''}
                    </div>
                    <div class="artist-meta">
                        ${artist.birth_year}–${artist.death_year} &bull; ${artist.period}
                    </div>
                    <div class="artist-stats">
                        <span class="badge badge-museum">${artist.primary_museum.split(',')[0]}</span>
                        <span class="badge badge-canonical">${artist.canonical_count} Canonical</span>
                    </div>
                `;
                artistListContainer.appendChild(li);
            });

            updateBannerAndGrid();
        }

        function selectArtist(artistId) {
            selectedArtistId = artistId;
            document.querySelectorAll('.artist-item').forEach(el => el.classList.remove('active'));
            renderArtistList();
        }

        function updateBannerAndGrid() {
            const artist = catalog.find(a => a.id === selectedArtistId);
            if (!artist) return;

            bannerName.textContent = artist.name;
            bannerPeriod.textContent = artist.period;
            bannerDates.textContent = `${artist.birth_year}–${artist.death_year}`;
            bannerMuseum.textContent = artist.primary_museum;
            bannerBio.textContent = artist.biography_snippet;

            // Update tab counters
            document.getElementById('countAll').textContent = artist.artworks.length;
            document.getElementById('countCanonical').textContent = artist.artworks.filter(a => a.status === 'canonical').length;
            document.getElementById('countCases').textContent = artist.artworks.filter(a => a.is_case_study).length;
            document.getElementById('countAnomalies').textContent = artist.artworks.filter(a => a.attribution_confidence < 0.50).length;

            renderArtworkGrid();
        }

        function renderArtworkGrid() {
            const artist = catalog.find(a => a.id === selectedArtistId);
            if (!artist) return;

            artworksGrid.innerHTML = '';

            let filtered = artist.artworks;
            if (currentFilter === 'canonical') {
                filtered = filtered.filter(a => a.status === 'canonical');
            } else if (currentFilter === 'cases') {
                filtered = filtered.filter(a => a.is_case_study);
            } else if (currentFilter === 'anomalies') {
                filtered = filtered.filter(a => a.attribution_confidence < 0.50);
            }

            if (filtered.length === 0) {
                artworksGrid.innerHTML = '<div class="empty-state">No paintings matching the current filter.</div>';
                return;
            }

            filtered.forEach(art => {
                const card = document.createElement('div');
                card.className = `artwork-card ${art.is_case_study ? 'is-controversy' : ''}`;

                const confPercent = Math.round(art.attribution_confidence * 100);
                const confColor = art.status_color;

                // Anomaly zone SVG overlay
                let anomalyBoxesHtml = '';
                if (art.anomaly_zones && art.anomaly_zones.length > 0) {
                    art.anomaly_zones.forEach(z => {
                        const leftPct = (z.bounding_box[0] / 600) * 100;
                        const topPct = (z.bounding_box[1] / 600) * 100;
                        const widthPct = (z.bounding_box[2] / 600) * 100;
                        const heightPct = (z.bounding_box[3] / 600) * 100;

                        anomalyBoxesHtml += `
                            <div class="anomaly-box" style="left: ${leftPct}%; top: ${topPct}%; width: ${widthPct}%; height: ${heightPct}%;" title="${z.title}: ${z.description}">
                                <span class="anomaly-tag">Zone ${z.zone_id} (${Math.round(z.confidence * 100)}%)</span>
                            </div>
                        `;
                    });
                }

                // Dynamic SVG gradient heatmap simulating local confidence
                const heatmapSvg = `
                    <svg viewBox="0 0 600 600" class="heatmap-layer" id="heatmap-${art.id}">
                        <defs>
                            <radialGradient id="grad-${art.id}" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="${art.attribution_confidence >= 0.8 ? '#10b981' : '#f43f5e'}" stop-opacity="0.8"/>
                                <stop offset="60%" stop-color="${art.attribution_confidence >= 0.8 ? '#38bdf8' : '#fbbf24'}" stop-opacity="0.6"/>
                                <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.3"/>
                            </radialGradient>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#grad-${art.id})" />
                    </svg>
                `;

                card.innerHTML = `
                    <div class="canvas-viewport">
                        <img src="${art.image_url}" alt="${art.title}" class="canvas-img" loading="lazy" />
                        ${heatmapSvg}
                        ${anomalyBoxesHtml}
                    </div>

                    <div class="card-body">
                        <div class="card-header-row">
                            <h3 class="card-title">${art.title}</h3>
                            <span class="confidence-pill" style="background: ${confColor}22; color: ${confColor}; border: 1px solid ${confColor}66;">
                                ${confPercent}% ${art.status_label}
                            </span>
                        </div>

                        <div class="card-metadata">
                            <strong>${art.creation_date}</strong> &bull; ${art.medium}<br/>
                            🏛️ <em>${art.institution}</em> (${art.collection_id})
                        </div>

                        <div class="metrics-bar">
                            <div class="metric-cell">
                                <div class="metric-cell-label">Mahalanobis Dist</div>
                                <div class="metric-cell-val">${art.mahalanobis_distance.toFixed(2)}</div>
                            </div>
                            <div class="metric-cell">
                                <div class="metric-cell-label">Crack Cleanliness</div>
                                <div class="metric-cell-val">${Math.round(art.craquelure_cleanliness * 100)}%</div>
                            </div>
                            <div class="metric-cell">
                                <div class="metric-cell-label">Facture Detail</div>
                                <div class="metric-cell-val">${Math.round(art.gradient_energy * 100)}%</div>
                            </div>
                        </div>

                        ${art.is_case_study ? `
                            <div class="case-study-box">
                                <div class="case-study-title">🔍 ${art.case_study_title}</div>
                                <div class="case-study-text">${art.case_study_summary}</div>
                            </div>
                        ` : ''}

                        <div class="card-footer">
                            <div class="heatmap-toggle-wrapper">
                                <span>Heatmap</span>
                                <input type="range" min="0" max="100" value="0" class="heatmap-slider" 
                                    oninput="document.getElementById('heatmap-${art.id}').style.opacity = this.value / 100" />
                            </div>
                            <a href="${art.museum_url}" target="_blank" rel="noopener" class="museum-link">
                                Museum Catalog ↗
                            </a>
                        </div>
                    </div>
                `;

                artworksGrid.appendChild(card);
            });
        }

        // Initial render
        renderArtistList();
    </script>
</body>
</html>
"""


class ArtworkBrowserRenderer:
    """Renders the standalone interactive artwork and connoisseurship browser."""

    def __init__(self, catalog: Optional[List[ArtistCatalogEntry]] = None) -> None:
        self.catalog = catalog or get_curated_artists_catalog()

    def render_html(self) -> str:
        """Renders the complete HTML template populated with the curated catalog."""
        # Convert dataclasses to serializable dicts
        catalog_dicts = []
        for entry in self.catalog:
            entry_dict = {
                "id": entry.id,
                "name": entry.name,
                "birth_year": entry.birth_year,
                "death_year": entry.death_year,
                "period": entry.period,
                "nationality": entry.nationality,
                "primary_museum": entry.primary_museum,
                "canonical_count": entry.canonical_count,
                "contested_count": entry.contested_count,
                "biography_snippet": entry.biography_snippet,
                "artworks": [],
            }
            for art in entry.artworks:
                art_dict = {
                    "id": art.id,
                    "title": art.title,
                    "artist": art.artist,
                    "creation_date": art.creation_date,
                    "medium": art.medium,
                    "dimensions": art.dimensions,
                    "institution": art.institution,
                    "collection_id": art.collection_id,
                    "museum_url": art.museum_url,
                    "image_url": art.image_url,
                    "attribution_confidence": art.attribution_confidence,
                    "status": art.status,
                    "status_label": art.status_label,
                    "status_color": art.status_color,
                    "is_case_study": art.is_case_study,
                    "case_study_title": art.case_study_title,
                    "case_study_summary": art.case_study_summary,
                    "mahalanobis_distance": art.mahalanobis_distance,
                    "craquelure_cleanliness": art.craquelure_cleanliness,
                    "gradient_energy": art.gradient_energy,
                    "anomaly_zones": [
                        {
                            "zone_id": z.zone_id,
                            "bounding_box": z.bounding_box,
                            "area_pct": z.area_pct,
                            "confidence": z.confidence,
                            "title": z.title,
                            "description": z.description,
                        }
                        for z in art.anomaly_zones
                    ],
                }
                entry_dict["artworks"].append(art_dict)
            catalog_dicts.append(entry_dict)

        catalog_json = json.dumps(catalog_dicts, ensure_ascii=False)
        template = Template(HTML_TEMPLATE)
        return template.render(catalog_json=catalog_json)

    def export(self, output_file: str | Path = "data/processed/browser.html") -> Path:
        """Exports the interactive browser to the designated output file."""
        html_content = self.render_html()
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding="utf-8")
        return out_path
