import json

import pandas as pd
import pytest

from player_cache_updater import cli, compare, normalize
from player_cache_updater.models import SourceError
from player_cache_updater.schema import LOCAL_COLUMNS, SOURCE_TO_LOCAL

TODAY = pd.Timestamp("2026-11-21").date()


def source_row(game_id="0022600001", date="2026-10-20", player_id=1628401, **overrides):
    row = {
        "season_id": 22026, "player_id": player_id, "game_id": game_id,
        "game_date": date, "matchup": "BOS @ DET", "wl": "W", "min": 30,
        "fgm": 4, "fga": 8, "fg_pct": 0.5, "fg3m": 1, "fg3a": 4,
        "fg3_pct": 0.25, "ftm": 2, "fta": 2, "ft_pct": 1.0, "oreb": 1,
        "dreb": 3, "reb": 4, "ast": 5, "stl": 1, "blk": 0, "tov": 2,
        "pf": 1, "pts": 11, "plus_minus": 6, "video_available": 1,
    }
    row.update(overrides)
    return row


def source_frame(*rows):
    return pd.DataFrame(list(rows), dtype=object).astype(
        {"game_id": str, "player_id": str}
    )


def local_frame(*rows):
    converted = []
    for row in rows:
        local = {SOURCE_TO_LOCAL[key]: value for key, value in row.items()}
        local["GAME_DATE"] = normalize.format_game_date(pd.Timestamp(row["game_date"]))
        local["Game_ID"] = row["game_id"]
        converted.append(local)
    return pd.DataFrame(converted)[LOCAL_COLUMNS]


def normalized_source(*rows):
    return normalize.normalize_source_frame(source_frame(*rows), 2026, TODAY)


def test_duplicate_game_ids_collapse_to_one_padded_row():
    cached = normalize.normalize_local_frame(
        local_frame(
            source_row(game_id="22600001"),
            source_row(game_id="0022600001"),
        )
    )
    incoming = normalized_source(source_row(game_id="22600001"))

    assert cached.duplicates_removed == 1
    assert list(cached.rows["Game_ID"]) == ["0022600001"]
    assert list(incoming.rows["Game_ID"]) == ["0022600001"]

    plan = compare.plan_player(1628401, "Derrick White", cached, incoming.rows)
    assert plan.changed and plan.duplicates_removed == 1
    assert plan.new_games == [] and plan.corrections == []
    assert list(plan.proposed["Game_ID"]) == ["0022600001"]


def test_missing_percentages_are_filled_only_without_attempts():
    zero_attempts = source_row(
        game_id="0022600002", fga=0, fgm=0, fg_pct=None, fg3m=0, fg3a=0,
        fg3_pct=None, ftm=0, fta=0, ft_pct=None, oreb=0, dreb=0, reb=0, pts=0,
    )
    has_attempts = source_row(game_id="0022600003", ft_pct=None)
    result = normalized_source(zero_attempts, has_attempts)

    assert result.pct_filled == 3
    kept = result.rows.iloc[0]
    assert (kept["FG_PCT"], kept["FG3_PCT"], kept["FT_PCT"]) == (0.0, 0.0, 0.0)
    assert [item.game_id for item in result.failures] == ["0022600003"]
    assert result.failures[0].reason == "missing FT_PCT despite attempts"
    assert list(result.rows["Game_ID"]) == ["0022600002"]


def test_unchanged_records_produce_no_proposal():
    row = source_row()
    cached = normalize.normalize_local_frame(local_frame(row))
    plan = compare.plan_player(
        1628401, "Derrick White", cached, normalized_source(row).rows
    )

    assert not plan.changed
    assert plan.proposed is None
    assert plan.unchanged == 1


def test_new_games_and_corrections_keep_newest_first_order():
    older = source_row(game_id="0022600001", date="2026-10-20")
    cached = normalize.normalize_local_frame(local_frame(older))
    corrected = source_row(game_id="0022600001", date="2026-10-20", min=32)
    newer = source_row(game_id="0022600005", date="2026-10-25")
    plan = compare.plan_player(
        1628401, "Derrick White", cached, normalized_source(corrected, newer).rows
    )

    assert [game["game_id"] for game in plan.new_games] == ["0022600005"]
    assert plan.corrections[0]["changes"] == {"MIN": [30, 32]}
    assert list(plan.proposed["Game_ID"]) == ["0022600005", "0022600001"]
    assert list(plan.proposed["GAME_DATE"]) == ["Oct 25, 2026", "Oct 20, 2026"]


def test_invalid_source_rows_are_reported_and_excluded():
    result = normalized_source(
        source_row(game_id="0022600001"),
        source_row(game_id="0022600002", reb=9),
        source_row(game_id="0022600003", date="2026-12-25"),
        source_row(game_id="abc"),
    )

    assert list(result.rows["Game_ID"]) == ["0022600001"]
    assert sorted(item.reason for item in result.failures) == sorted([
        "game_date is in the future",
        "invalid game_id",
        "REB does not equal OREB + DREB",
    ])


def test_conflicting_source_duplicates_are_rejected_identical_ones_collapse():
    same = source_row(game_id="0022600001")
    result = normalized_source(
        same, dict(same),
        source_row(game_id="0022600002"), source_row(game_id="0022600002", pts=30),
    )

    assert result.duplicates_removed == 1
    assert list(result.rows["Game_ID"]) == ["0022600001"]
    assert {item.reason for item in result.failures} == {
        "conflicting duplicate source rows"
    }


def test_dry_run_skips_unknown_players_and_never_touches_the_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    known_old = source_row(game_id="0022600001", date="2026-10-20")
    local_frame(known_old).to_csv(cache_dir / "1628401.csv", index=False)
    before = (cache_dir / "1628401.csv").read_bytes()

    source_rows = [
        source_row(game_id="0022600001", date="2026-10-20"),
        source_row(game_id="0022600005", date="2026-10-25"),
        source_row(game_id="0022600005", date="2026-10-25", player_id=999999999),
    ]
    source_rows += [
        source_row(game_id=f"00226{n:05d}", date="2026-10-21", player_id=1700000 + n)
        for n in range(1, 120)
    ]
    source_path = tmp_path / "source.csv"
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    roster = tmp_path / "players.csv"
    roster.write_text(
        "id,full_name,first_name,last_name,is_active\n"
        "1628401,Derrick White,Derrick,White,True\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "scratch"

    status = cli.main([
        "--source-file", str(source_path), "--cache-dir", str(cache_dir),
        "--players-csv", str(roster), "--out-dir", str(out_dir),
    ])

    assert status == 0
    assert (cache_dir / "1628401.csv").read_bytes() == before
    assert sorted(path.name for path in cache_dir.iterdir()) == ["1628401.csv"]
    summary = json.loads((out_dir / "report.json").read_text())["summary"]
    assert summary["new_games"] == 1
    assert summary["unknown_players_skipped"] == 120
    proposed = pd.read_csv(out_dir / "proposed" / "2026-27" / "1628401.csv")
    assert list(proposed.columns) == LOCAL_COLUMNS
    assert list(proposed["GAME_DATE"]) == ["Oct 25, 2026", "Oct 20, 2026"]


def test_output_directory_inside_data_is_rejected(tmp_path):
    cache_dir = tmp_path / "data" / "nba_cache" / "2026-27"
    cache_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        cli.report.ensure_scratch_directory(cache_dir / "scratch", [cache_dir])


def test_source_missing_columns_is_fatal():
    with pytest.raises(SourceError):
        normalize.normalize_source_frame(pd.DataFrame({"player_id": ["1"]}), 2026, TODAY)


WHITE, TATUM, NEWBIE = 1628401, 1628369, 1700001


def make_workspace(tmp_path, bad_source_row=False):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    local_frame(
        source_row(game_id="0022600001", date="2026-10-20", player_id=WHITE)
    ).to_csv(cache_dir / f"{WHITE}.csv", index=False)
    local_frame(
        source_row(game_id="22600002", date="2026-10-21", player_id=TATUM),
        source_row(game_id="0022600002", date="2026-10-21", player_id=TATUM),
    ).to_csv(cache_dir / f"{TATUM}.csv", index=False)

    rows = [
        source_row(game_id="0022600001", date="2026-10-20", player_id=WHITE),
        source_row(game_id="0022600005", date="2026-10-25", player_id=WHITE),
        source_row(game_id="0022600002", date="2026-10-21", player_id=TATUM),
        source_row(game_id="0022600006", date="2026-10-26", player_id=TATUM),
        source_row(game_id="0022600007", date="2026-10-27", player_id=NEWBIE),
    ]
    rows += [
        source_row(game_id=f"00226{n:05d}", date="2026-10-21", player_id=1800000 + n)
        for n in range(1, 120)
    ]
    if bad_source_row:
        rows.append(source_row(game_id="0022600009", player_id=WHITE, reb=99))
    source_path = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False)
    roster = tmp_path / "players.csv"
    roster.write_text(
        "id,full_name,first_name,last_name,is_active\n"
        f"{WHITE},Derrick White,Derrick,White,True\n"
        f"{TATUM},Jayson Tatum,Jayson,Tatum,True\n"
        f"{NEWBIE},New Player,New,Player,True\n",
        encoding="utf-8",
    )
    return cache_dir, [
        "--source-file", str(source_path), "--cache-dir", str(cache_dir),
        "--players-csv", str(roster), "--out-dir", str(tmp_path / "scratch"),
    ]


def snapshot(cache_dir):
    return {path.name: path.read_bytes() for path in cache_dir.iterdir()}


def test_apply_updates_cache_with_backup_and_rolls_back_exactly(tmp_path):
    cache_dir, base = make_workspace(tmp_path)
    original = snapshot(cache_dir)

    assert cli.main(base + ["--apply"]) == 0

    white = pd.read_csv(cache_dir / f"{WHITE}.csv", dtype={"Game_ID": str})
    tatum = pd.read_csv(cache_dir / f"{TATUM}.csv", dtype={"Game_ID": str})
    assert list(white["Game_ID"]) == ["0022600005", "0022600001"]
    assert list(tatum["Game_ID"]) == ["0022600006", "0022600002"]
    assert not (cache_dir / f"{NEWBIE}.csv").exists()
    assert not [p for p in cache_dir.iterdir() if p.name.endswith(".tmp")]

    backup = tmp_path / "scratch" / "backup"
    assert (backup / "files" / f"{TATUM}.csv").read_bytes() == original[
        f"{TATUM}.csv"
    ]
    assert cli.main(["--rollback", str(backup), "--cache-dir", str(cache_dir)]) == 0
    assert snapshot(cache_dir) == original


def test_apply_creates_new_player_files_only_when_requested_and_rolls_back(tmp_path):
    cache_dir, base = make_workspace(tmp_path)
    original = snapshot(cache_dir)

    assert cli.main(base + ["--apply", "--include-new-players"]) == 0
    assert (cache_dir / f"{NEWBIE}.csv").exists()

    backup = tmp_path / "scratch" / "backup"
    assert cli.main(["--rollback", str(backup), "--cache-dir", str(cache_dir)]) == 0
    assert snapshot(cache_dir) == original


def test_apply_is_refused_and_cache_untouched_when_validation_fails(tmp_path):
    cache_dir, base = make_workspace(tmp_path, bad_source_row=True)
    original = snapshot(cache_dir)

    assert cli.main(base + ["--apply"]) == 3

    assert snapshot(cache_dir) == original
    assert not (tmp_path / "scratch" / "backup").exists()
    info = json.loads((tmp_path / "scratch" / "report.json").read_text())["apply"]
    assert info["gate_passed"] is False and info["applied"] is False


def test_failure_during_apply_restores_every_file(tmp_path, monkeypatch):
    cache_dir, base = make_workspace(tmp_path)
    original = snapshot(cache_dir)
    real_replace = cli.apply._replace_file
    calls = []

    def fail_on_second(proposed, destination, expected):
        calls.append(destination.name)
        if len(calls) == 2:
            raise OSError("disk full")
        real_replace(proposed, destination, expected)

    monkeypatch.setattr(cli.apply, "_replace_file", fail_on_second)

    assert cli.main(base + ["--apply"]) == 4
    assert snapshot(cache_dir) == original
