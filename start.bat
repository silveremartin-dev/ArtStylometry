@echo off
chcp 65001 > nul
title ArtStylometry — Connoisseurship Engine
echo =========================================================================
echo       ARTSTYLOMETRY - COMPUTATIONAL STYLOMETRY ^& CONNOISSEURSHIP
echo =========================================================================
echo  [+] Launching local HTTP server at http://localhost:8080/
echo  [+] Automatically launching dashboard in your web browser...
echo  [i] Tip: To browse AND populate your local database in background,
echo           use 'start_and_harvest.bat' instead.
echo  [!] To stop the server, press Ctrl+C in this window.
echo =========================================================================

py -3 scripts/serve.py 2>nul || python scripts/serve.py
pause
