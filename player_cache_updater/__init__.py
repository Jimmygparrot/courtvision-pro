"""Dry-run updater for the 2026-27 player game-log cache.

The package downloads the sportsdataverse player game logs, validates and
normalizes them to the nba_api ``PlayerGameLog`` schema used by
``data/nba_cache``, and writes *proposed* CSVs to a scratch directory only.
"""
