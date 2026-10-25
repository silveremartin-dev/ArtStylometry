@echo off
chcp 65001 > nul
title ArtStylometry — Connoisseurship Engine
echo =========================================================================
echo       ARTSTYLOMETRY - COMPUTATIONAL STYLOMETRY ^& CONNOISSEURSHIP
echo =========================================================================
echo  [+] Launching local HTTP server at http://localhost:8080/
echo  [+] Automatically launching dashboard in your web browser...
echo  [!] To stop the server, press Ctrl+C in this window.
echo =========================================================================

py -3 scripts/serve.py 2>nul || python scripts/serve.py
pause
