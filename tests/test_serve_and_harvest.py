"""
Unit tests for the dual-engine serve and harvest script (scripts/serve_and_harvest.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.serve_and_harvest import (
    BackgroundHarvesterThread,
    find_available_port,
    is_port_free,
    parse_args,
)


def test_port_helper_functions():
    port = find_available_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_background_harvester_thread_init(tmp_path):
    dest = tmp_path / "paintings"
    cat = tmp_path / "catalog.json"
    worker = BackgroundHarvesterThread(
        dest_dir=dest,
        catalog_path=cat,
        max_dimension=2000,
        concurrency=2,
        source="cma",
        limit=10,
        target_gb=5.0,
    )
    assert worker.limit == 10
    assert worker.target_gb == 5.0
    assert worker.concurrency == 2
    assert worker.source == "cma"
    assert worker.daemon is True


def test_parse_args_defaults():
    with patch("sys.argv", ["serve_and_harvest.py"]):
        args = parse_args()
        assert args.limit == 25000
        assert args.target_gb == 60.0
        assert args.concurrency == 4
        assert args.source == "full"
        assert args.max_dimension == 4000
        assert args.no_browser is False


def test_parse_args_custom():
    with patch(
        "sys.argv",
        [
            "serve_and_harvest.py",
            "--port", "9999",
            "--target-gb", "30.5",
            "--limit", "500",
            "--source", "extended",
            "--concurrency", "8",
            "--no-browser",
        ],
    ):
        args = parse_args()
        assert args.port == 9999
        assert args.target_gb == 30.5
        assert args.limit == 500
        assert args.source == "extended"
        assert args.concurrency == 8
        assert args.no_browser is True
