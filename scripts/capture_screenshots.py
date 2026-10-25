"""
Automated UI Screenshot Capture with Playwright.
Generates publication-quality screenshots of ArtStylometry's views for the documentation.
"""

from __future__ import annotations

import functools
import http.server
import os
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def start_test_server(port: int = 8899) -> http.server.ThreadingHTTPServer:
    root_dir = Path(__file__).resolve().parent.parent
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def capture_all_views() -> None:
    port = 8899
    server = start_test_server(port)
    time.sleep(1)

    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Capturing screenshots from http://127.0.0.1:{port}/ ...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.set_default_timeout(60000)

        # 1. Capture Master Portal Dashboard (Tab 1: 12 Masters Audit)
        print("  - Capturing Master Portal Audit View...")
        page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="domcontentloaded")
        time.sleep(3)
        page.screenshot(path=str(out_dir / "master_portal_dashboard.png"))

        # 2. Capture 12 Masters Audit with Divergences filtered
        print("  - Capturing Contested Masters Audit (Divergences Filtered)...")
        page.goto(f"http://127.0.0.1:{port}/data/processed/contested_masters_audit.html", wait_until="domcontentloaded")
        time.sleep(3)
        # Click on All Divergences pill
        try:
            page.click("button:has-text('All Divergences')")
            time.sleep(1)
        except Exception as e:
            print("    Warning clicking filter:", e)
        page.screenshot(path=str(out_dir / "contested_masters_audit.png"))

        # 3. Capture 3D WebGL Manifold Explorer
        print("  - Capturing 3D WebGL Collection Manifold...")
        page.goto(f"http://127.0.0.1:{port}/data/processed/collection_manifold_3d.html", wait_until="domcontentloaded")
        time.sleep(3)
        page.screenshot(path=str(out_dir / "collection_manifold_3d.png"))

        # 4. Capture Global Gallery
        print("  - Capturing Global Gallery...")
        page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="domcontentloaded")
        time.sleep(2)
        try:
            page.click("button:has-text('Global Gallery')")
            time.sleep(3)
        except Exception as e:
            print("    Warning clicking Global Gallery tab:", e)
        page.screenshot(path=str(out_dir / "global_gallery.png"))

        # 4b. Capture Data & Methodology Viewport
        print("  - Capturing Data & Methodology Viewport...")
        try:
            page.click("button:has-text('Data & Methodology')")
            time.sleep(2)
            page.screenshot(path=str(out_dir / "data_and_methodology_tab.png"))
        except Exception as e:
            print("    Warning clicking Data & Methodology tab:", e)

        # 5. Capture Close-up: Top Filter Toolbar & Atelier Badges
        print("  - Capturing Filter Toolbar & Badges...")
        page.goto(f"http://127.0.0.1:{port}/data/processed/contested_masters_audit.html", wait_until="domcontentloaded")
        time.sleep(2)
        try:
            filters_el = page.locator(".filters-container")
            filters_el.screenshot(path=str(out_dir / "filter_toolbar_and_badges.png"))
        except Exception as e:
            print("    Warning capturing filter toolbar:", e)

        # 6. Capture Close-up: Demoted Canon Anomaly Card (Frans Hals Portrait of a Man)
        print("  - Capturing Demotion Divergence Card...")
        try:
            demote_card = page.locator(".art-card:has-text('Portrait of a man')").first
            demote_card.scroll_into_view_if_needed()
            time.sleep(1)
            demote_card.screenshot(path=str(out_dir / "divergence_demotion_detail.png"))
        except Exception as e:
            print("    Warning capturing demoted card:", e)

        # 7. Capture Close-up: Promoted Disputed Work Card (Vermeer Saint Praxedis)
        print("  - Capturing Promotion Divergence Card...")
        try:
            promote_card = page.locator(".art-card:has-text('Saint Praxedis')").first
            promote_card.scroll_into_view_if_needed()
            time.sleep(1)
            promote_card.screenshot(path=str(out_dir / "divergence_promotion_detail.png"))
        except Exception as e:
            print("    Warning capturing promoted card:", e)

        # 8. Capture Close-up: 2D Spatial Heatmap Overlay Active (Girl with a Flute)
        print("  - Capturing 2D Heatmap Overlay Active...")
        try:
            flute_card = page.locator(".art-card:has-text('Girl with a Flute')").first
            flute_card.scroll_into_view_if_needed()
            time.sleep(1)
            # Click the 2D Heatmap button on this card
            btn = flute_card.locator("button.heatmap-btn")
            btn.click()
            time.sleep(1)
            flute_card.screenshot(path=str(out_dir / "spatial_heatmap_overlay_active.png"))
        except Exception as e:
            print("    Warning capturing heatmap card:", e)

        # 9. Capture 3D Manifold with Hover Inspection Tooltip
        print("  - Capturing 3D Manifold with Hover Tooltip...")
        page.goto(f"http://127.0.0.1:{port}/data/processed/collection_manifold_3d.html", wait_until="domcontentloaded")
        time.sleep(3)
        try:
            page.evaluate("""() => {
                const cv = document.getElementById('c');
                if (window.proj && window.proj.length > 0) {
                    // Pick a prominent visible point near center of canvas
                    const w = window.innerWidth, h = window.innerHeight;
                    const candidates = window.proj.filter(p => p[0] > w * 0.35 && p[0] < w * 0.65 && p[1] > h * 0.35 && p[1] < h * 0.65);
                    const pt = candidates.length > 0 ? candidates[Math.floor(candidates.length / 2)] : window.proj[0];
                    const evt = new MouseEvent('mousemove', {
                        clientX: pt[0],
                        clientY: pt[1],
                        bubbles: true
                    });
                    cv.dispatchEvent(evt);
                }
            }""")
            time.sleep(1)
            page.screenshot(path=str(out_dir / "manifold_3d_inspection_hover.png"))
        except Exception as e:
            print("    Warning capturing 3D hover tooltip:", e)

        browser.close()

    server.shutdown()
    print(f"[OK] All screenshots saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    capture_all_views()
