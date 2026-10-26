@echo off
chcp 65001 > nul
title ArtStylometry — Server & Background Artwork Harvester
echo =========================================================================
echo       ARTSTYLOMETRY - SERVER ^& LOCAL DATABASE HARVESTER
echo =========================================================================
echo  [+] Launching local HTTP server at http://localhost:8080/
echo  [+] Launching dashboard in your default web browser...
echo  [+] Background harvester will download fine art paintings up to 60 GB
echo      from open museum archives (Cleveland, NGA, Met, Wikidata, etc.)
echo  [!] To stop the server and downloads safely, press Ctrl+C.
echo =========================================================================

py -3 scripts/serve_and_harvest.py %* 2>nul || python scripts/serve_and_harvest.py %*
pause
