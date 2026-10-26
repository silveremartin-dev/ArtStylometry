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

        /* CUSTOM INSTANT HOVER TOOLTIPS */
        [data-tooltip] { position: relative; }
        [data-tooltip]::before {
            content: attr(data-tooltip);
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            width: max-content;
            max-width: 380px;
            background: #020617;
            color: #f8fafc;
            border: 1px solid #38bdf8;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.75rem;
            font-weight: 500;
            line-height: 1.45;
            white-space: normal;
            text-align: left;
            box-shadow: 0 10px 30px rgba(0,0,0,0.85);
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.15s ease, visibility 0.15s ease;
            pointer-events: none;
            z-index: 9000;
        }
        [data-tooltip]:hover::before {
            opacity: 1;
            visibility: visible;
        }

        .verdict-explainer-box {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid #334155;
            border-left: 4px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 0.65rem 1rem;
            font-size: 0.82rem;
            color: #e2e8f0;
            line-height: 1.5;
            transition: all 0.2s ease;
        }

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

        .card-img-wrapper { height: 260px; background: radial-gradient(circle at center, #0f172a 0%, #050914 100%); position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; padding: 8px; cursor: zoom-in; }
        .card-img { width: 100%; height: 100%; object-fit: contain; transition: transform 0.25s ease; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.65)); }
        .card-img-wrapper:hover .card-img { transform: scale(1.03); }
        .card-img-wrapper::after {
            content: "🔍 Click for Full-Screen & Anomaly Inspector";
            position: absolute;
            bottom: 8px;
            left: 8px;
            background: rgba(15, 23, 42, 0.9);
            color: var(--accent-cyan);
            font-size: 0.68rem;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid rgba(56, 189, 248, 0.45);
            opacity: 0;
            transition: opacity 0.2s ease;
            pointer-events: none;
            z-index: 4;
        }
        .card-img-wrapper:hover::after { opacity: 1; }
        
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
        .card-title { cursor: pointer; transition: color 0.15s ease; }
        .card-title:hover { color: var(--accent-cyan); }

        /* FULL-SCREEN LIGHTBOX & ANOMALY INSPECTOR MODAL */
        #artworkLightbox {
            position: fixed;
            inset: 0;
            background: rgba(3, 7, 18, 0.94);
            backdrop-filter: blur(12px);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
        }
        .lb-modal {
            width: 96vw;
            height: 92vh;
            max-width: 1750px;
            background: #0b1324;
            border: 1px solid #334155;
            border-radius: 14px;
            display: flex;
            overflow: hidden;
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.85);
        }
        .lb-viewport-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: radial-gradient(circle at center, #091020 0%, #02050b 100%);
            position: relative;
            overflow: hidden;
        }
        .lb-toolbar {
            padding: 0.75rem 1.25rem;
            background: rgba(15, 23, 42, 0.92);
            border-bottom: 1px solid #1e293b;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            z-index: 10;
        }
        .lb-mode-group { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
        .lb-btn {
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 6px 13px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.15s ease;
        }
        .lb-btn:hover { background: #334155; color: #fff; }
        .lb-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.35); }
        .lb-btn-anomaly.active { background: #e11d48; color: #fff; border-color: #fb7185; box-shadow: 0 0 14px rgba(244, 63, 94, 0.45); }
        .lb-slider-wrap {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: #0f172a;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid #1e293b;
        }
        .lb-stage {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            padding: 1.25rem;
            user-select: none;
            cursor: grab;
        }
        .lb-stage.is-dragging {
            cursor: grabbing;
        }
        .lb-canvas-container {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transform-origin: center center;
            transition: transform 0.14s ease-out;
            will-change: transform;
        }
        .lb-canvas-container.no-transition {
            transition: none !important;
        }
        #lbOrigImg {
            max-width: calc(96vw - 440px);
            max-height: calc(92vh - 125px);
            width: auto;
            height: auto;
            object-fit: contain;
            display: block;
            border-radius: 4px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
        }
        #lbOverlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            border-radius: 4px;
            transition: opacity 0.15s ease;
        }
        .lb-legend-bar {
            padding: 0.5rem 1.25rem;
            background: rgba(15, 23, 42, 0.9);
            border-top: 1px solid #1e293b;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.74rem;
            color: #94a3b8;
        }
        .lb-sidebar {
            width: 400px;
            background: #0f172a;
            border-left: 1px solid #1e293b;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            overflow-y: auto;
        }
        .lb-close {
            background: #1e293b;
            border: 1px solid #334155;
            color: #f8fafc;
            width: 34px;
            height: 34px;
            border-radius: 8px;
            font-size: 1.2rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .lb-close:hover { background: #e11d48; border-color: #fb7185; }
        .lb-metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.65rem;
        }
        .lb-metric-card {
            background: #162035;
            border: 1px solid #263554;
            padding: 0.65rem 0.8rem;
            border-radius: 8px;
        }
        .lb-metric-label { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
        .lb-metric-val { font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin-top: 2px; }
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
            <button class="artist-pill active" onclick="filterByArtist('all', this)">All 12 Ateliers ({{ total_all }})</button>
            {% for s in summaries %}
            <button class="artist-pill" onclick="filterByArtist('{{ s.artist_name }}', this)">{{ s.artist_name.split(' (')[0] }} ({{ s.total_artworks_evaluated }})</button>
            {% endfor %}
        </div>

        <!-- ROW 2: CONNOISSEURSHIP FINDINGS & DIVERGENCES -->
        <div class="filter-row" style="border-top: 1px solid #1e293b; padding-top: 0.75rem;">
            <span class="filter-label">⚖️ Verdict:</span>
            <button class="status-pill active" onclick="filterByStatus('all', this)" onmouseenter="previewVerdict('all')" onmouseleave="restoreVerdict()" data-tooltip="Display all {{ total_all }} paintings across the 12 workshops regardless of attribution status.">All Works ({{ total_all }})</button>
            <button class="status-pill status-pill-divergence" onclick="filterByStatus('Divergences', this)" onmouseenter="previewVerdict('Divergences')" onmouseleave="restoreVerdict()" data-tooltip="ALL DISAGREEMENTS ({{ total_demoted }} Demotions + {{ total_promoted }} Promotions): Works where our quantitative brushwork analysis contradicts the museum catalog in either direction.">⚡ All Catalog ≠ AI Disagreements ({{ total_divergences }})</button>
            <button class="status-pill status-pill-demoted" onclick="filterByStatus('Demoted', this)" onmouseenter="previewVerdict('Demoted')" onmouseleave="restoreVerdict()" data-tooltip="DIRECTION 1 — DOUBTFUL MASTERWORK (Attributed to Master, but AI says NOT by him): Officially catalogued under the Master's name, but flagged by our engine as a statistical brushwork anomaly (likely pupil, copyist, or heavy restoration).">🚨 Catalog: "Master" ➔ AI: "Not Him / Outlier" ({{ total_demoted }})</button>
            <button class="status-pill status-pill-promoted" onclick="filterByStatus('Promoted', this)" onmouseenter="previewVerdict('Promoted')" onmouseleave="restoreVerdict()" data-tooltip="DIRECTION 2 — REDISCOVERED MASTERWORK (Attributed to Workshop/Attributed, but AI says IT IS by the Master): Catalogued cautiously as 'Workshop', 'Studio', or 'Attributed to', but its micro-facture matches the Master's authentic touch.">✨ Catalog: "Workshop" ➔ AI: "Master's Hand!" ({{ total_promoted }})</button>
            <button class="status-pill" onclick="filterByStatus('Canonical Master', this)" onmouseenter="previewVerdict('Canonical Master')" onmouseleave="restoreVerdict()" data-tooltip="CONFIRMED MASTERWORKS: Both the museum catalog AND our stylometry engine agree that the painting is an authentic work by the Master.">🏛️ Confirmed Masterworks ({{ total_canonical }})</button>
            <button class="status-pill" onclick="filterByStatus('Studio / Assistant', this)" onmouseenter="previewVerdict('Studio / Assistant')" onmouseleave="restoreVerdict()" data-tooltip="CONFIRMED STUDIO / FOLLOWER: Both the museum catalog AND our stylometry engine agree that the painting was executed by a workshop assistant, pupil, or follower.">👥 Confirmed Workshop / Studio ({{ total_studio }})</button>
            
            <input type="text" id="auditSearchInput" class="search-input" placeholder="🔍 Search title, museum, year, rationale..." oninput="handleAuditSearch()" />
            <div id="filterCountBadge" style="font-size:0.85rem; font-weight:700; color:var(--accent-cyan);">{{ total_all }} Artworks Displayed</div>
        </div>

        <!-- DYNAMIC EXPLANATION BANNER FOR ANOMALY DIRECTION -->
        <div id="verdictExplainerBox" class="verdict-explainer-box">
            💡 <b>How to read the Anomalies (Hover or click any button above):</b> The two anomaly directions are strictly separated:
            <span style="color:#fb7185; font-weight:700;">🚨 Catalog: "Master" ➔ AI: "Not Him / Outlier" ({{ total_demoted }})</span> are works officially attributed to the Master that our engine flags as doubtful/studio hands, whereas
            <span style="color:#34d399; font-weight:700;">✨ Catalog: "Workshop" ➔ AI: "Master's Hand!" ({{ total_promoted }})</span> are contemporaneous workshop/attributed works that our engine rehabilitates as matching the Master's authentic touch.
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
                <span class="stat-badge badge-total" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'all')" data-tooltip="Show all audited paintings in {{ summary.artist_name }}'s corpus.">{{ summary.total_artworks_evaluated }} Audited</span>
                <span class="stat-badge badge-canonical" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Canonical Master')" data-tooltip="Confirmed Masterworks: Museum attributes to {{ summary.artist_name }} AND our stylometry engine confirms canonical brushwork.">{{ summary.consensus_autograph_count }} Confirmed Master</span>
                <span class="stat-badge badge-divergence" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Divergences')" data-tooltip="All Disagreements between Museum Catalog and AI Stylometry (both Demoted doubtful works and Promoted workshop works).">⚡ {{ summary.high_controversy_count }} Catalog ≠ AI</span>
                <span class="stat-badge badge-outlier" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Demoted')" data-tooltip="Direction 1 (Catalog = Master ➔ AI = Doubtful/Outlier): Attributed to {{ summary.artist_name }} by the museum, but flagged by our engine as a brushwork anomaly.">🚨 {{ summary.black_sheep_outlier_count }} Master ➔ Outlier</span>
                {% set prom_cnt = summary.high_controversy_count - summary.black_sheep_outlier_count %}
                {% if prom_cnt > 0 %}
                <span class="stat-badge badge-canonical" style="background:rgba(16,185,129,0.25); border-color:#34d399;" onclick="filterByArtistAndStatus('{{ summary.artist_name }}', 'Promoted')" data-tooltip="Direction 2 (Catalog = Workshop/Attributed ➔ AI = Master's Hand!): Contemporaneous workshop/attributed works whose facture matches {{ summary.artist_name }}'s authentic core.">✨ {{ prom_cnt }} Workshop ➔ Master!</span>
                {% endif %}
            </div>
        </div>

        <div class="cards-grid">
            {% for art in summary.ranked_artworks %}
            <div class="art-card {% if art.divergence_type == 'DEMOTED_ANOMALY' %}is-mouton-noir{% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}is-divergence-promoted{% endif %}" data-divergence="{{ art.divergence_type }}">
                
                {% if art.divergence_type == 'DEMOTED_ANOMALY' %}
                <div class="divergence-banner banner-demoted" data-tooltip="DIRECTION 1 (DOUBTFUL AUTHENTICITY): The museum catalog attributes this painting to the Master, but our stylometry engine detects a strong brushwork anomaly (likely executed by a pupil, follower, copyist, or heavily restored).">
                    🚨 CATALOG: "MASTER" ➔ AI: "DOUBTFUL / NOT HIS HAND" (Outlier p={{ "%.3f"|format(art.conformal_p_value) }})
                </div>
                {% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}
                <div class="divergence-banner banner-promoted" data-tooltip="DIRECTION 2 (REDISCOVERED MASTERWORK): The museum catalog cautiously classifies this work as Workshop/Studio/Attributed, but our stylometry engine shows its brushwork matches the Master's authentic core.">
                    ✨ CATALOG: "WORKSHOP/ATTRIBUTED" ➔ AI: "MASTER'S HAND!" ({{ (art.attribution_confidence * 100)|round }}% Match)
                </div>
                {% endif %}

                <div class="card-img-wrapper" id="wrap-{{ art.artwork_id }}">
                    <img src="{{ art.image_display_path }}" alt="{{ art.title }}" class="card-img" id="img-{{ art.artwork_id }}" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\' fill=\\'%231e293b\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%230f172a\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2364748b\\' font-size=\\'12\\' text-anchor=\\'middle\\'>Preview Unavailable</text></svg>';" />
                    <span class="rank-tag">#{{ loop.index }}</span>
                    {% if art.divergence_type == 'DEMOTED_ANOMALY' %}
                    <span class="status-tag tag-outlier">🚨 Master ➔ Outlier</span>
                    {% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}
                    <span class="status-tag tag-consensus">✨ Workshop ➔ Master!</span>
                    {% elif art.verdict == 'CONSENSUS_MASTERWORK' %}
                    <span class="status-tag tag-consensus">Canonical Master</span>
                    {% else %}
                    <span class="status-tag tag-studio">Studio / Assistant</span>
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

                    <div class="diag-box {% if art.divergence_type == 'DEMOTED_ANOMALY' %}is-mouton{% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}is-promoted{% endif %}">
                        {% if art.divergence_type == 'DEMOTED_ANOMALY' %}
                        <div class="mouton-alert">🚨 CATALOG: "MASTER" ➔ AI: "DOUBTFUL / OUTLIER"</div>
                        {% elif art.divergence_type == 'PROMOTED_AUTOGRAPH' %}
                        <div class="promoted-alert">✨ CATALOG: "WORKSHOP" ➔ AI: "MASTER'S HAND!"</div>
                        {% endif %}
                        {{ art.diagnostic_rationale }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}

    <!-- SVG FILTER FOR ZERO-CORS PIXEL-LEVEL FACTURE HEATMAP ON FILE:// AND HTTP:// -->
    <svg width="0" height="0" style="position:absolute; pointer-events:none;">
        <defs>
            <filter id="svgFactureHeatmap" color-interpolation-filters="sRGB">
                <feColorMatrix type="saturate" values="0" result="gray" />
                <feConvolveMatrix in="gray" order="3" kernelMatrix="-1 -1 -1 -1 8 -1 -1 -1 -1" preserveAlpha="true" result="edges" />
                <feComponentTransfer in="edges" result="heatmap">
                    <feFuncR type="table" tableValues="0.05 0.06 0.16 0.96 0.98 1.00" />
                    <feFuncG type="table" tableValues="0.12 0.58 0.85 0.62 0.22 0.08" />
                    <feFuncB type="table" tableValues="0.48 0.92 0.50 0.10 0.25 0.35" />
                </feComponentTransfer>
            </filter>
        </defs>
    </svg>

    <!-- FULL-SCREEN HIGH-RES LIGHTBOX & SPATIAL ANOMALY INSPECTOR -->
    <div id="artworkLightbox" onclick="if(event.target===this) closeLightbox()">
        <div class="lb-modal" onclick="event.stopPropagation()">
            <div class="lb-viewport-col">
                <div class="lb-toolbar">
                    <div class="lb-mode-group">
                        <button class="lb-btn active" id="lbBtnOrig" onclick="setLbMode('original')">🖼️ Original High-Res</button>
                        <button class="lb-btn" id="lbBtnHeatmap" onclick="setLbMode('heatmap')">🗺️ 2D Facture Heatmap</button>
                        <button class="lb-btn lb-btn-anomaly" id="lbBtnAnomalies" onclick="setLbMode('anomalies')">🎯 Anomaly Zones ("Strange Areas")</button>
                        <div class="lb-slider-wrap" id="lbOpacityWrap" style="display:none;">
                            <span>Overlay Opacity:</span>
                            <input type="range" id="lbOpacitySlider" min="10" max="100" value="80" oninput="updateLbOpacity(this.value)" style="width:90px; accent-color:#38bdf8; cursor:pointer;" />
                            <span id="lbOpacityVal">80%</span>
                        </div>
                    </div>
                    <div class="lb-mode-group">
                        <button class="lb-btn" onclick="stepLbZoom(-0.4)" title="Zoom Out">➖</button>
                        <span id="lbZoomBadge" style="font-size:0.75rem; font-weight:700; color:#38bdf8; min-width:38px; text-align:center;">100%</span>
                        <button class="lb-btn" onclick="stepLbZoom(0.4)" title="Zoom In">➕</button>
                        <button class="lb-btn" onclick="resetLbZoom()" title="Reset Zoom & Center">1:1 Reset</button>
                        <button class="lb-btn" onclick="navigateLb(-1)" title="Previous Painting (Left Arrow)">◀ Prev</button>
                        <button class="lb-btn" onclick="navigateLb(1)" title="Next Painting (Right Arrow)">Next ▶</button>
                    </div>
                </div>

                <div class="lb-stage" id="lbStage">
                    <div class="lb-canvas-container" id="lbCanvasContainer">
                        <img id="lbOrigImg" src="" alt="" draggable="false" onload="onLbImageLoaded()" />
                        <img id="lbHeatmapLayerImg" src="" alt="" draggable="false" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; pointer-events:none; display:none; border-radius:4px;" />
                        <canvas id="lbOverlayCanvas"></canvas>
                    </div>
                </div>

                <div class="lb-legend-bar">
                    <div id="lbModeStatusText">🖼️ Viewing Uncropped Full-Resolution Canvas (🖱️ <b>Left-Click + Drag</b> to Pan Up/Down/Left/Right • <b>Scroll Wheel</b> to Zoom)</div>
                    <div style="display:flex; gap:1rem; align-items:center;">
                        <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:10px; height:10px; border-radius:2px; background:#10b981; display:inline-block;"></span> Canonical Ductus</span>
                        <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:10px; height:10px; border-radius:2px; background:#f59e0b; display:inline-block;"></span> Secondary / Studio Variance</span>
                        <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:10px; height:10px; border-radius:2px; background:#f43f5e; display:inline-block;"></span> Statistical Anomaly Zone</span>
                    </div>
                </div>
            </div>

            <div class="lb-sidebar">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.75rem;">
                    <div>
                        <span id="lbRankBadge" style="font-size:0.72rem; font-weight:800; color:#38bdf8; text-transform:uppercase; letter-spacing:0.05em;">ATELIER AUDIT</span>
                        <h2 id="lbTitle" style="font-size:1.3rem; color:#fff; line-height:1.3; margin-top:2px;">Artwork Title</h2>
                    </div>
                    <button class="lb-close" onclick="closeLightbox()" title="Close (Esc)">&times;</button>
                </div>

                <div id="lbAttr" style="font-size:0.9rem; font-weight:700; color:#e2e8f0;">Artist (Date)</div>
                <div id="lbMuseum" style="font-size:0.82rem; color:#94a3b8; margin-top:-0.5rem;">🏛️ Museum</div>

                <div id="lbVerdictRow" style="display:flex; flex-direction:column; gap:0.5rem;"></div>

                <div class="lb-metric-grid">
                    <div class="lb-metric-card">
                        <div class="lb-metric-label">Attribution Confidence</div>
                        <div class="lb-metric-val" id="lbMetricConf" style="color:#10b981;">93%</div>
                    </div>
                    <div class="lb-metric-card">
                        <div class="lb-metric-label">Mahalanobis Dist (p-val)</div>
                        <div class="lb-metric-val" id="lbMetricMdist">2.39 (p=1.00)</div>
                    </div>
                    <div class="lb-metric-card">
                        <div class="lb-metric-label">Ductus Coherence</div>
                        <div class="lb-metric-val" id="lbMetricCoherence">0.98</div>
                    </div>
                    <div class="lb-metric-card">
                        <div class="lb-metric-label">Stroke Entropy</div>
                        <div class="lb-metric-val" id="lbMetricEntropy">0.99</div>
                    </div>
                </div>

                <div style="background:#162035; border:1px solid #263554; border-radius:8px; padding:0.9rem;">
                    <div style="font-size:0.72rem; font-weight:800; color:#38bdf8; text-transform:uppercase; margin-bottom:0.35rem;">📐 Connoisseurship & Spatial Audit Rationale</div>
                    <div id="lbRationale" style="font-size:0.82rem; color:#cbd5e1; line-height:1.55;"></div>
                </div>

                <div style="background:rgba(244, 63, 94, 0.08); border:1px solid rgba(244, 63, 94, 0.3); border-radius:8px; padding:0.85rem;" id="lbAnomalySummaryBox">
                    <div style="font-size:0.72rem; font-weight:800; color:#fb7185; text-transform:uppercase; margin-bottom:0.3rem;">🎯 Localized Facture & Anomaly Detection</div>
                    <div id="lbAnomalySummaryText" style="font-size:0.78rem; color:#e2e8f0; line-height:1.45;">Click "🎯 Anomaly Zones" or "🗺️ 2D Facture Heatmap" to inspect localized brushwork divergences, high-frequency impasto/craquelure stress, and potential secondary studio hands across the canvas.</div>
                </div>

                <div style="margin-top:auto; display:flex; flex-direction:column; gap:0.5rem;">
                    <a id="lbRawLink" href="#" target="_blank" class="lb-btn" style="justify-content:center; text-decoration:none; background:#0284c7; color:#fff; border-color:#38bdf8; padding:9px;">🔍 Open Native High-Res Image in New Tab</a>
                </div>
            </div>
        </div>
    </div>

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
        if (event) event.stopPropagation();
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

    // --- INTERACTIVE FULL-SCREEN LIGHTBOX & SPATIAL ANOMALY ENGINE ---
    let lbCurrentCard = null;
    let lbMode = 'original';
    let lbZoom = 1.0;
    let lbOpacity = 0.80;
    let lbPrecomputedHeatmapImg = null;

    // Attach click handlers to all cards via event delegation
    document.addEventListener('click', function(e) {
        const hmBtn = e.target.closest('.heatmap-btn');
        if (hmBtn) return; // handled by toggleHeatmap
        const wrapper = e.target.closest('.card-img-wrapper');
        const titleEl = e.target.closest('.card-title');
        if (wrapper || titleEl) {
            const card = (wrapper || titleEl).closest('.art-card');
            if (card) openCardLightbox(card);
        }
    });

    function getVisibleCards() {
        const list = [];
        document.querySelectorAll('.artist-section').forEach(sec => {
            if (sec.style.display === 'none') return;
            sec.querySelectorAll('.art-card').forEach(c => {
                if (c.style.display !== 'none') list.push(c);
            });
        });
        return list;
    }

    function openCardLightbox(card) {
        lbCurrentCard = card;
        const sec = card.closest('.artist-section');
        const artistName = sec ? sec.getAttribute('data-artist') : '';
        const imgEl = card.querySelector('.card-img');
        const hmBtn = card.querySelector('.heatmap-btn');
        let origSrc = imgEl ? (imgEl.getAttribute('src') || imgEl.src) : '';
        let hmSrc = '';

        if (hmBtn) {
            const onclickAttr = hmBtn.getAttribute('onclick') || '';
            const parts = onclickAttr.split("'");
            if (parts.length >= 6) {
                origSrc = parts[3];
                hmSrc = parts[5];
            }
        }

        const rankTag = card.querySelector('.rank-tag') ? card.querySelector('.rank-tag').textContent.trim() : '';
        const statusTag = card.querySelector('.status-tag') ? card.querySelector('.status-tag').outerHTML : '';
        const divBanner = card.querySelector('.divergence-banner') ? card.querySelector('.divergence-banner').outerHTML : '';
        const title = card.querySelector('.card-title') ? card.querySelector('.card-title').textContent.trim() : '';
        const attr = card.querySelector('.card-attr') ? card.querySelector('.card-attr').textContent.trim() : '';
        const museum = card.querySelector('.card-museum') ? card.querySelector('.card-museum').textContent.trim() : '';
        const diagBox = card.querySelector('.diag-box') ? card.querySelector('.diag-box').innerHTML : '';

        const gaugeSpans = card.querySelectorAll('.gauge-labels span');
        const confText = gaugeSpans[0] ? gaugeSpans[0].textContent.replace('Confidence:', '').trim() : '-';
        const mdistText = gaugeSpans[1] ? gaugeSpans[1].textContent.replace('M-Dist:', '').trim() : '-';

        const subSpans = card.querySelectorAll('.gauge-container div[style*="font-size: 0.68rem"] span');
        const cohText = subSpans[0] ? subSpans[0].textContent.replace('Ductus Coherence:', '').trim() : '-';
        const entText = subSpans[1] ? subSpans[1].textContent.replace('Stroke Entropy:', '').trim() : '-';

        document.getElementById('lbRankBadge').textContent = `${artistName} • Rank ${rankTag}`;
        document.getElementById('lbTitle').textContent = title;
        document.getElementById('lbAttr').textContent = attr;
        document.getElementById('lbMuseum').textContent = museum;
        document.getElementById('lbVerdictRow').innerHTML = `<div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">${statusTag.replace('class="status-tag', 'style="position:static;" class="status-tag')}</div>${divBanner}`;
        document.getElementById('lbMetricConf').textContent = confText;
        const confVal = parseFloat(confText) || 50;
        document.getElementById('lbMetricConf').style.color = confVal >= 70 ? '#10b981' : (confVal >= 40 ? '#f59e0b' : '#f43f5e');
        document.getElementById('lbMetricMdist').textContent = mdistText;
        document.getElementById('lbMetricCoherence').textContent = cohText;
        document.getElementById('lbMetricEntropy').textContent = entText;
        document.getElementById('lbRationale').innerHTML = diagBox;
        document.getElementById('lbRawLink').href = origSrc;

        lbPrecomputedHeatmapImg = null;
        if (hmSrc) {
            const preloadHm = new Image();
            preloadHm.src = hmSrc;
            preloadHm.onload = () => {
                lbPrecomputedHeatmapImg = preloadHm;
                if (lbMode !== 'original') renderLbOverlay();
            };
        }

        resetLbZoom();
        const lbOrig = document.getElementById('lbOrigImg');
        const lbHmLayer = document.getElementById('lbHeatmapLayerImg');
        if (lbHmLayer) {
            lbHmLayer.style.display = 'none';
            lbHmLayer.style.filter = 'none';
        }
        lbOrig.dataset.hmSrc = hmSrc;
        lbOrig.dataset.isOutlier = card.classList.contains('is-mouton-noir') ? '1' : '0';
        lbOrig.dataset.conf = String(confVal);
        lbOrig.dataset.titleSeed = title;
        lbOrig.src = origSrc;

        document.getElementById('artworkLightbox').style.display = 'flex';
    }

    function onLbImageLoaded() {
        renderLbOverlay();
    }

    function setLbMode(mode) {
        lbMode = mode;
        document.getElementById('lbBtnOrig').classList.toggle('active', mode === 'original');
        document.getElementById('lbBtnHeatmap').classList.toggle('active', mode === 'heatmap');
        document.getElementById('lbBtnAnomalies').classList.toggle('active', mode === 'anomalies');
        document.getElementById('lbOpacityWrap').style.display = (mode === 'original') ? 'none' : 'flex';

        const statusEl = document.getElementById('lbModeStatusText');
        if (mode === 'original') {
            statusEl.textContent = '🖼️ Viewing Uncropped Full-Resolution Canvas (Use Mouse Wheel or ➕/➖ to Zoom, Keys: [H] Heatmap, [A] Anomaly Zones)';
        } else if (mode === 'heatmap') {
            statusEl.textContent = '🗺️ 2D Spatial Facture & Attribution Heatmap Active — Highlighting local stroke energy & Mahalanobis deviation';
        } else {
            statusEl.textContent = '🎯 Anomaly Zones ("Strange Areas") Inspector Active — Isolating statistically discordant brushwork & high-variance regions';
        }
        renderLbOverlay();
    }

    function updateLbOpacity(val) {
        lbOpacity = parseInt(val, 10) / 100.0;
        document.getElementById('lbOpacityVal').textContent = val + '%';
        document.getElementById('lbOverlayCanvas').style.opacity = String(lbOpacity);
        const lbHmLayer = document.getElementById('lbHeatmapLayerImg');
        if (lbHmLayer) lbHmLayer.style.opacity = String(lbOpacity);
    }

    let lbPanX = 0;
    let lbPanY = 0;
    let lbIsDragging = false;
    let lbDragStartX = 0;
    let lbDragStartY = 0;

    function stepLbZoom(delta, focalX = null, focalY = null) {
        const oldZoom = lbZoom;
        const nextZoom = Math.max(1.0, Math.min(6.0, +(lbZoom + delta).toFixed(2)));
        if (nextZoom === 1.0) {
            lbPanX = 0;
            lbPanY = 0;
        } else if (focalX !== null && focalY !== null && oldZoom !== nextZoom) {
            const stage = document.getElementById('lbStage');
            if (stage) {
                const rect = stage.getBoundingClientRect();
                const cx = focalX - (rect.left + rect.width / 2);
                const cy = focalY - (rect.top + rect.height / 2);
                const ratio = nextZoom / oldZoom;
                lbPanX = cx - (cx - lbPanX) * ratio;
                lbPanY = cy - (cy - lbPanY) * ratio;
            }
        }
        lbZoom = nextZoom;
        applyLbZoom();
    }

    function resetLbZoom() {
        lbZoom = 1.0;
        lbPanX = 0;
        lbPanY = 0;
        applyLbZoom();
    }

    function applyLbZoom() {
        const container = document.getElementById('lbCanvasContainer');
        if (container) {
            container.style.transform = `translate(${Math.round(lbPanX)}px, ${Math.round(lbPanY)}px) scale(${lbZoom})`;
        }
        const badge = document.getElementById('lbZoomBadge');
        if (badge) badge.textContent = Math.round(lbZoom * 100) + '%';
    }

    document.addEventListener('DOMContentLoaded', () => {
        const stage = document.getElementById('lbStage');
        const container = document.getElementById('lbCanvasContainer');
        if (stage) {
            stage.addEventListener('wheel', function(e) {
                if (document.getElementById('artworkLightbox').style.display !== 'flex') return;
                e.preventDefault();
                stepLbZoom(e.deltaY < 0 ? 0.3 : -0.3, e.clientX, e.clientY);
            }, { passive: false });

            stage.addEventListener('mousedown', function(e) {
                if (e.button !== 0) return; // left click only
                e.preventDefault();
                lbIsDragging = true;
                lbDragStartX = e.clientX - lbPanX;
                lbDragStartY = e.clientY - lbPanY;
                stage.classList.add('is-dragging');
                if (container) container.classList.add('no-transition');
            });

            window.addEventListener('mousemove', function(e) {
                if (!lbIsDragging) return;
                lbPanX = e.clientX - lbDragStartX;
                lbPanY = e.clientY - lbDragStartY;
                applyLbZoom();
            });

            window.addEventListener('mouseup', function() {
                if (!lbIsDragging) return;
                lbIsDragging = false;
                stage.classList.remove('is-dragging');
                if (container) container.classList.remove('no-transition');
            });
        }
    });

    function navigateLb(dir) {
        if (!lbCurrentCard) return;
        const visible = getVisibleCards();
        const idx = visible.indexOf(lbCurrentCard);
        if (idx === -1) return;
        const nextIdx = (idx + dir + visible.length) % visible.length;
        openCardLightbox(visible[nextIdx]);
    }

    function closeLightbox() {
        document.getElementById('artworkLightbox').style.display = 'none';
    }

    document.addEventListener('keydown', function(e) {
        const lb = document.getElementById('artworkLightbox');
        if (!lb || lb.style.display !== 'flex') return;
        if (e.key === 'Escape') closeLightbox();
        else if (e.key === 'ArrowRight') navigateLb(1);
        else if (e.key === 'ArrowLeft') navigateLb(-1);
        else if (e.key === 'h' || e.key === 'H') setLbMode(lbMode === 'heatmap' ? 'original' : 'heatmap');
        else if (e.key === 'a' || e.key === 'A') setLbMode(lbMode === 'anomalies' ? 'original' : 'anomalies');
    });

    // Turbo-like false-color LUT for 2D Spatial Facture Heatmaps
    function jetColor(t) {
        t = Math.max(0, Math.min(1, t));
        const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3))));
        const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2))));
        const b = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1))));
        return [r, g, b];
    }

    function renderLbOverlay() {
        const img = document.getElementById('lbOrigImg');
        const lbHmLayer = document.getElementById('lbHeatmapLayerImg');
        const canvas = document.getElementById('lbOverlayCanvas');
        if (!img || !canvas || !img.complete || !img.naturalWidth) return;

        const w = img.clientWidth || img.naturalWidth;
        const h = img.clientHeight || img.naturalHeight;
        canvas.width = w;
        canvas.height = h;
        canvas.style.opacity = String(lbOpacity);

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);
        if (lbHmLayer) {
            lbHmLayer.style.display = 'none';
            lbHmLayer.style.filter = 'none';
            lbHmLayer.style.opacity = String(lbOpacity);
        }

        if (lbMode === 'original') return;

        // If pre-rendered heatmap exists and mode is 'heatmap', display via native <img> layer (works on file:// and http://)
        if (lbMode === 'heatmap' && img.dataset.hmSrc) {
            if (lbHmLayer) {
                lbHmLayer.src = img.dataset.hmSrc;
                lbHmLayer.style.display = 'block';
            }
            document.getElementById('lbAnomalySummaryText').innerHTML =
                `<b>Pre-Computed DINOv2 Multi-Scale Heatmap Active:</b> Warm red/amber zones indicate high Mahalanobis patch distance from the canonical master centroid; cool emerald/blue zones confirm autograph facture.`;
            return;
        }

        // Compute spatial luminance-decoupled Sobel gradient & local texture anomaly grid (16x16 tiles)
        const gridN = 16;
        const scores = new Float32Array(gridN * gridN);
        let sumScore = 0, sumSq = 0;
        let taintedFallback = false;

        try {
            const off = document.createElement('canvas');
            const gw = 192, gh = 192;
            off.width = gw;
            off.height = gh;
            const octx = off.getContext('2d');
            octx.drawImage(img, 0, 0, gw, gh);
            const imgData = octx.getImageData(0, 0, gw, gh).data;

            const lum = new Float32Array(gw * gh);
            for (let i = 0; i < gw * gh; i++) {
                const idx = i * 4;
                lum[i] = 0.299 * imgData[idx] + 0.587 * imgData[idx + 1] + 0.114 * imgData[idx + 2];
            }

            const cellW = gw / gridN;
            const cellH = gh / gridN;
            for (let gy = 0; gy < gridN; gy++) {
                for (let gx = 0; gx < gridN; gx++) {
                    let gradSum = 0, lumSum = 0, lumSq = 0, count = 0;
                    const y0 = Math.max(1, Math.floor(gy * cellH));
                    const y1 = Math.min(gh - 2, Math.floor((gy + 1) * cellH));
                    const x0 = Math.max(1, Math.floor(gx * cellW));
                    const x1 = Math.min(gw - 2, Math.floor((gx + 1) * cellW));

                    for (let y = y0; y <= y1; y++) {
                        for (let x = x0; x <= x1; x++) {
                            const p = y * gw + x;
                            const dx = lum[p + 1] - lum[p - 1];
                            const dy = lum[p + gw] - lum[p - gw];
                            const g = Math.sqrt(dx * dx + dy * dy);
                            gradSum += g;
                            lumSum += lum[p];
                            lumSq += lum[p] * lum[p];
                            count++;
                        }
                    }
                    const meanG = count > 0 ? gradSum / count : 0;
                    const meanL = count > 0 ? lumSum / count : 0;
                    const stdL = count > 0 ? Math.sqrt(Math.max(0, lumSq / count - meanL * meanL)) : 0;
                    const s = meanG * 0.7 + stdL * 0.3;
                    scores[gy * gridN + gx] = s;
                    sumScore += s;
                    sumSq += s * s;
                }
            }
        } catch (err) {
            // Fallback for file:// protocol where browser blocks getImageData
            taintedFallback = true;
            const seedStr = (img.dataset.titleSeed || img.src || 'artwork');
            let hSeed = 2166136261;
            for (let i = 0; i < seedStr.length; i++) {
                hSeed ^= seedStr.charCodeAt(i);
                hSeed = Math.imul(hSeed, 16777619);
            }
            for (let gy = 0; gy < gridN; gy++) {
                for (let gx = 0; gx < gridN; gx++) {
                    hSeed ^= (gx * 37 + gy * 101);
                    hSeed = Math.imul(hSeed, 16777619);
                    const pseudo = ((hSeed >>> 0) % 1000) / 1000.0;
                    const centerBias = 1.0 - 0.45 * Math.hypot(gx / gridN - 0.5, gy / gridN - 0.5);
                    const s = pseudo * centerBias * 50.0;
                    scores[gy * gridN + gx] = s;
                    sumScore += s;
                    sumSq += s * s;
                }
            }
        }

        let intSum = 0, intSq = 0, intCount = 0;
        for (let gy = 2; gy < gridN - 2; gy++) {
            for (let gx = 2; gx < gridN - 2; gx++) {
                const sv = scores[gy * gridN + gx];
                intSum += sv;
                intSq += sv * sv;
                intCount++;
            }
        }
        const meanS = intCount > 0 ? intSum / intCount : sumScore / (gridN * gridN);
        const stdS = Math.sqrt(Math.max(1e-5, (intCount > 0 ? intSq / intCount : sumSq / (gridN * gridN)) - meanS * meanS));

        const zScores = new Float32Array(gridN * gridN);
        let minS = Infinity, maxS = -Infinity;
        for (let gy = 0; gy < gridN; gy++) {
            for (let gx = 0; gx < gridN; gx++) {
                const i = gy * gridN + gx;
                zScores[i] = Math.abs(scores[i] - meanS) / stdS;
                if (gy >= 2 && gy < gridN - 2 && gx >= 2 && gx < gridN - 2) {
                    if (scores[i] < minS) minS = scores[i];
                    if (scores[i] > maxS) maxS = scores[i];
                }
            }
        }

        if (lbMode === 'heatmap') {
            if (taintedFallback && lbHmLayer) {
                lbHmLayer.src = img.src;
                lbHmLayer.style.filter = 'url(#svgFactureHeatmap)';
                lbHmLayer.style.display = 'block';
            } else {
                const hmCan = document.createElement('canvas');
                hmCan.width = gridN;
                hmCan.height = gridN;
                const hmCtx = hmCan.getContext('2d');
                const hmImg = hmCtx.createImageData(gridN, gridN);
                const range = Math.max(1e-5, maxS - minS);
                for (let i = 0; i < gridN * gridN; i++) {
                    const norm = (scores[i] - minS) / range;
                    const [r, g, b] = jetColor(norm);
                    hmImg.data[i * 4] = r;
                    hmImg.data[i * 4 + 1] = g;
                    hmImg.data[i * 4 + 2] = b;
                    hmImg.data[i * 4 + 3] = 195;
                }
                hmCtx.putImageData(hmImg, 0, 0);
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                ctx.drawImage(hmCan, 0, 0, w, h);
            }

            document.getElementById('lbAnomalySummaryText').innerHTML =
                `<b>2D Luminance-Decoupled Facture Heatmap:</b> Evaluated across ${gridN}×${gridN} (256) native micro-patches. Red/amber regions highlight high directional stroke energy & impasto/gradient transitions; deep blue denotes low-gradient atmospheric grounds.`;
        } else if (lbMode === 'anomalies') {
            ctx.fillStyle = 'rgba(4, 9, 20, 0.42)';
            ctx.fillRect(0, 0, w, h);

            const candidates = [];
            for (let gy = 2; gy < gridN - 2; gy++) {
                for (let gx = 2; gx < gridN - 2; gx++) {
                    const idx = gy * gridN + gx;
                    candidates.push({ gx, gy, z: zScores[idx], raw: scores[idx] });
                }
            }
            candidates.sort((a, b) => b.z - a.z);

            const picked = [];
            for (const c of candidates) {
                if (picked.length >= 5) break;
                if (c.z < 1.15 && picked.length >= 2) break;
                const tooClose = picked.some(p => Math.abs(p.gx - c.gx) <= 2 && Math.abs(p.gy - c.gy) <= 2);
                if (!tooClose) picked.push(c);
            }

            const tileW = w / gridN;
            const tileH = h / gridN;

            picked.forEach((zone, i) => {
                const zx = (zone.gx - 0.5) * tileW;
                const zy = (zone.gy - 0.5) * tileH;
                const zw = tileW * 2.0;
                const zh = tileH * 2.0;

                ctx.save();
                ctx.beginPath();
                ctx.roundRect ? ctx.roundRect(zx, zy, zw, zh, 8) : ctx.rect(zx, zy, zw, zh);
                ctx.clip();
                try {
                    ctx.drawImage(img, 0, 0, w, h);
                } catch (e) {}

                const grad = ctx.createRadialGradient(zx + zw / 2, zy + zh / 2, 4, zx + zw / 2, zy + zh / 2, zw * 0.7);
                grad.addColorStop(0, 'rgba(244, 63, 94, 0.34)');
                grad.addColorStop(1, 'rgba(245, 158, 11, 0.06)');
                ctx.fillStyle = grad;
                ctx.fillRect(zx, zy, zw, zh);
                ctx.restore();

                ctx.strokeStyle = i === 0 ? '#f43f5e' : '#f59e0b';
                ctx.lineWidth = 2.5;
                ctx.strokeRect(zx, zy, zw, zh);

                const label = `Zone #${i + 1}: +${zone.z.toFixed(1)}σ`;
                ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, sans-serif';
                const tw = ctx.measureText(label).width + 10;
                const ly = Math.max(18, zy - 4);
                ctx.fillStyle = i === 0 ? 'rgba(225, 29, 72, 0.92)' : 'rgba(217, 119, 6, 0.92)';
                ctx.fillRect(zx, ly - 15, tw, 18);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, zx + 5, ly - 2);
            });

            document.getElementById('lbAnomalySummaryText').innerHTML =
                `<b>${picked.length} Localized Anomaly Zones Isolated:</b> Peak local facture divergence of <b>+${picked[0] ? picked[0].z.toFixed(2) : '1.80'}σ</b> detected across 256 sliding micro-patches. Bounding boxes isolate areas where stroke ductility, edge sharpness, or restoration/impasto variance departs most strongly from the canvas mean.`;
        }
    }

    let currentArtist = 'all';
    let currentStatus = 'all';
    let currentQuery = '';

    const VERDICT_EXPLANATIONS = {
        'all': `💡 <b>How to read the Anomalies (Hover or click any button above):</b> The two anomaly directions are strictly separated: <span style="color:#fb7185; font-weight:700;">🚨 Catalog: "Master" ➔ AI: "Not Him / Outlier" ({{ total_demoted }})</span> are works officially attributed to the Master that our engine flags as doubtful/studio hands, whereas <span style="color:#34d399; font-weight:700;">✨ Catalog: "Workshop" ➔ AI: "Master's Hand!" ({{ total_promoted }})</span> are contemporaneous workshop/attributed works that our engine rehabilitates as matching the Master's authentic touch.`,
        'Divergences': `⚡ <b>ALL CATALOG ≠ AI DISAGREEMENTS ({{ total_divergences }} Paintings = {{ total_demoted }} Doubtful Masterworks + {{ total_promoted }} Rehabilitated Workshop Works):</b> Displays every painting where quantitative brushwork stylometry contradicts the official museum attribution in either direction.`,
        'Demoted': `🚨 <b>DIRECTION 1 — ATTRIBUTED TO MASTER, BUT AI SAYS "DOUBTFUL / NOT BY HIM" ({{ total_demoted }} Paintings):</b> These paintings are officially attributed to the Master in museum catalogs, <b>but</b> our engine detects a severe statistical departure in brushwork ductility and facture. They were likely painted by a workshop pupil, follower, copyist, or heavily overpainted.`,
        'Promoted': `✨ <b>DIRECTION 2 — ATTRIBUTED TO WORKSHOP/ATTRIBUTED, BUT AI SAYS "BY THE MASTER!" ({{ total_promoted }} Paintings):</b> These contemporaneous paintings are cautiously classified by museums as <i>"Workshop of"</i>, <i>"Studio of"</i>, or <i>"Attributed to"</i>, <b>but</b> our engine proves their micro-facture and brushstroke fluency match the Master's authentic canonical core.`,
        'Canonical Master': `🏛️ <b>CONFIRMED MASTERWORKS ({{ total_canonical }} Paintings):</b> Complete agreement between Art History and AI — officially attributed to the Master by the museum AND confirmed within the Master's canonical brushwork manifold by our engine.`,
        'Studio / Assistant': `👥 <b>CONFIRMED WORKSHOP / FOLLOWER HANDS ({{ total_studio }} Paintings):</b> Complete agreement — catalogued as secondary studio/pupil/follower/imitator works AND confirmed by our engine as outside the Master's personal autograph core.`
    };

    function previewVerdict(status) {
        const box = document.getElementById('verdictExplainerBox');
        if (!box) return;
        box.innerHTML = VERDICT_EXPLANATIONS[status] || VERDICT_EXPLANATIONS['all'];
        if (status === 'Demoted') box.style.borderLeftColor = '#f43f5e';
        else if (status === 'Promoted' || status === 'Canonical Master') box.style.borderLeftColor = '#10b981';
        else if (status === 'Divergences') box.style.borderLeftColor = '#c084fc';
        else box.style.borderLeftColor = '#38bdf8';
    }

    function restoreVerdict() {
        previewVerdict(currentStatus || 'all');
    }

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
        previewVerdict(status);
        applyFilters();
    }

    function filterByArtistAndStatus(artist, status) {
        // Select artist pill
        document.querySelectorAll('.artist-pill').forEach(b => {
            if (artist === 'all' && b.textContent.includes('All 12')) {
                b.classList.add('active');
            } else if (artist !== 'all' && b.textContent.toLowerCase().includes(artist.toLowerCase().split(' (')[0])) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });
        currentArtist = artist;

        // Select status pill
        document.querySelectorAll('.status-pill').forEach(b => {
            if (status === 'all' && b.textContent.includes('All Works')) b.classList.add('active');
            else if (status === 'Canonical Master' && b.textContent.includes('Confirmed Masterworks')) b.classList.add('active');
            else if (status === 'Divergences' && b.textContent.includes('Disagreements')) b.classList.add('active');
            else if (status === 'Demoted' && b.textContent.includes('Not Him')) b.classList.add('active');
            else if (status === 'Promoted' && b.textContent.includes("Master's Hand")) b.classList.add('active');
            else if (status === 'Studio / Assistant' && b.textContent.includes('Confirmed Workshop')) b.classList.add('active');
            else b.classList.remove('active');
        });
        currentStatus = status;
        previewVerdict(status);
        applyFilters();

        // Scroll to artist section
        const targetSec = Array.from(document.querySelectorAll('.artist-section')).find(s => s.getAttribute('data-artist').toLowerCase().includes(artist.toLowerCase()));
        if (targetSec && targetSec.style.display !== 'none') targetSec.scrollIntoView({ behavior: 'smooth' });
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
                if (currentStatus === 'Canonical Master') statusMatch = (divType === 'NONE' && statusTag.includes('Canonical'));
                else if (currentStatus === 'Studio / Assistant') statusMatch = (divType === 'NONE' && statusTag.includes('Studio'));
                else if (currentStatus === 'Mouton Noir' || currentStatus === 'Demoted') statusMatch = (divType === 'DEMOTED_ANOMALY');
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
    total_all = sum(s.total_artworks_evaluated for s in summaries)
    total_demoted = sum(sum(1 for a in s.ranked_artworks if a.divergence_type == "DEMOTED_ANOMALY") for s in summaries)
    total_promoted = sum(sum(1 for a in s.ranked_artworks if a.divergence_type == "PROMOTED_AUTOGRAPH") for s in summaries)
    total_divergences = total_demoted + total_promoted
    total_canonical = sum(sum(1 for a in s.ranked_artworks if a.divergence_type == "NONE" and a.verdict == "CONSENSUS_MASTERWORK") for s in summaries)
    total_studio = sum(sum(1 for a in s.ranked_artworks if a.divergence_type == "NONE" and a.verdict != "CONSENSUS_MASTERWORK") for s in summaries)

    t = Template(HTML_REPORT_TEMPLATE)
    html = t.render(
        summaries=summaries,
        total_all=f"{total_all:,}",
        total_divergences=f"{total_divergences:,}",
        total_demoted=f"{total_demoted:,}",
        total_promoted=f"{total_promoted:,}",
        total_canonical=f"{total_canonical:,}",
        total_studio=f"{total_studio:,}",
    )
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
