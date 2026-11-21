"""Writing proposed files, self-checks and run reports."""

import json
from pathlib import Path

import pandas as pd

from .models import Failure
from .schema import KEY_COLUMNS, LOCAL_COLUMNS, SEASON


def ensure_scratch_directory(out_dir, protected):
    """Refuse to write inside the repository's data or cache directories."""
    out_dir = Path(out_dir).resolve()
    for path in protected:
        path = Path(path).resolve()
        if out_dir == path or path in out_dir.parents:
            raise ValueError(
                f"Output directory {out_dir} must not be inside {path}."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_proposed(plan, out_dir):
    target = Path(out_dir) / "proposed" / SEASON
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{plan.player_id}.csv"
    plan.proposed[LOCAL_COLUMNS].to_csv(path, index=False, lineterminator="\n")
    return path


def validate_proposed_file(path):
    """Re-read a proposed file and check schema, order and duplicates."""
    problems = []
    frame = pd.read_csv(path, dtype={"Game_ID": str})
    if list(frame.columns) != LOCAL_COLUMNS:
        problems.append("columns differ from the cache schema")
        return problems
    if frame.duplicated(KEY_COLUMNS).any():
        problems.append("duplicate player/game rows")
    if not frame["Game_ID"].str.fullmatch(r"\d{10}").all():
        problems.append("Game_ID is not 10 digits")
    dates = pd.to_datetime(frame["GAME_DATE"], format="mixed", errors="coerce")
    if dates.isna().any():
        problems.append("unparseable GAME_DATE")
    elif not dates.is_monotonic_decreasing:
        problems.append("rows are not newest first")
    if frame.isna().any().any():
        problems.append("contains empty values")
    return problems


def build_report(context):
    plans = context["plans"]
    failures = context["failures"]
    reasons = {}
    for failure in failures:
        reasons[failure.reason] = reasons.get(failure.reason, 0) + 1

    changed = [plan for plan in plans if plan.changed]
    summary = {
        "source_rows": context["source_rows"],
        "valid_source_rows": context["valid_rows"],
        "validation_failures": len(failures),
        "validation_failure_reasons": reasons,
        "source_duplicates_removed": context["source_duplicates_removed"],
        "missing_percentages_filled": context["pct_filled"],
        "players_in_source": context["players_in_source"],
        "unknown_players_skipped": len(context["unknown_players"]),
        "players_compared": len(plans),
        "players_with_changes": len(changed),
        "players_unchanged": len(plans) - len(changed),
        "new_player_files": sum(not plan.existing_file for plan in changed),
        "new_games": sum(len(plan.new_games) for plan in plans),
        "corrected_records": sum(len(plan.corrections) for plan in plans),
        "corrected_cells": sum(
            len(item["changes"]) for plan in plans for item in plan.corrections
        ),
        "local_duplicates_removed": sum(p.duplicates_removed for p in plans),
        "local_ids_normalized": sum(p.ids_normalized for p in plans),
        "unchanged_records": sum(plan.unchanged for plan in plans),
        "local_only_records": sum(plan.local_only for plan in plans),
        "cache_players_absent_from_source": context["absent_from_source"],
        "proposed_files_written": len(context["written"]),
        "proposed_file_problems": len(context["output_problems"]),
    }
    details = {
        "source": context["source_info"],
        "warnings": context["warnings"],
        "summary": summary,
        "unknown_players": context["unknown_players"],
        "validation_failures": [vars(item) for item in failures],
        "output_problems": context["output_problems"],
        "players": [
            {
                "player_id": plan.player_id,
                "name": plan.name,
                "existing_file": plan.existing_file,
                "new_games": plan.new_games,
                "corrections": plan.corrections,
                "local_duplicates_removed": plan.duplicates_removed,
                "local_ids_normalized": plan.ids_normalized,
                "unchanged_records": plan.unchanged,
            }
            for plan in changed
        ],
    }
    return details


def write_report(report, out_dir):
    path = Path(out_dir) / "report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def render_text(report, out_dir):
    summary = report["summary"]
    lines = [
        "CourtVision Pro player cache DRY RUN (no cache files were modified)",
        f"Scratch directory: {out_dir}",
    ]
    source = report["source"]
    if source.get("http_status"):
        lines.append(
            f"Source: HTTP {source['http_status']}, {source['bytes']:,} bytes "
            f"in {source['seconds']}s"
        )
    else:
        lines.append(f"Source file: {source.get('path')}")
    lines.append(f"Latest source game date: {source.get('latest_game_date')}")
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append("")
    labels = [
        ("source_rows", "Source rows"),
        ("valid_source_rows", "Valid source rows"),
        ("validation_failures", "Validation failures (rows excluded)"),
        ("source_duplicates_removed", "Source duplicates removed"),
        ("missing_percentages_filled", "Missing percentages filled (0 attempts)"),
        ("players_in_source", "Players in source"),
        ("unknown_players_skipped", "Unknown players skipped"),
        ("players_with_changes", "Players with proposed changes"),
        ("players_unchanged", "Players unchanged"),
        ("new_player_files", "New player files proposed"),
        ("new_games", "New games"),
        ("corrected_records", "Corrected records"),
        ("corrected_cells", "Corrected cells"),
        ("local_duplicates_removed", "Cached duplicates removed"),
        ("local_ids_normalized", "Cached Game_IDs re-padded"),
        ("unchanged_records", "Unchanged records"),
        ("local_only_records", "Cached records absent from source"),
        ("cache_players_absent_from_source", "Cached players absent from source"),
        ("proposed_files_written", "Proposed CSVs written"),
        ("proposed_file_problems", "Proposed file self-check problems"),
    ]
    for key, label in labels:
        lines.append(f"{label}: {summary[key]}")
    if summary["validation_failure_reasons"]:
        lines.append("Validation failure reasons:")
        for reason, count in summary["validation_failure_reasons"].items():
            lines.append(f"  - {reason}: {count}")
    corrections = [
        (player["name"], item)
        for player in report["players"]
        for item in player["corrections"]
    ]
    if corrections:
        lines.append("Corrected records (first 10):")
        for name, item in corrections[:10]:
            changes = ", ".join(
                f"{column} {old}->{new}"
                for column, (old, new) in item["changes"].items()
            )
            lines.append(
                f"  - {name} {item['game_date']} {item['matchup']}: {changes}"
            )
    for problem in report["output_problems"][:10]:
        lines.append(f"OUTPUT PROBLEM: {problem}")
    return "\n".join(lines)


def failure(scope, player_id, game_id, reason):
    return Failure(scope, str(player_id), str(game_id), reason)
