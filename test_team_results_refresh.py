from datetime import date

import pandas as pd
import pytest
from nba_api.stats.endpoints import leaguegamelog

import nba_predictions
from nba_predictions import refresh_current_season_team_snapshot


TEAM_IDS = list(range(1610612737, 1610612767))
SNAPSHOT_COLUMNS = ["GAME_ID", "TEAM_ID", "GAME_DATE", "WL"]


def completed_team_games(game_date, first_game_id):
    rows = []
    for game_number, first_index in enumerate(range(0, 30, 2)):
        game_id = f"{first_game_id + game_number:010d}"
        home_team_id = TEAM_IDS[first_index]
        away_team_id = TEAM_IDS[first_index + 1]
        rows.extend([
            {
                "GAME_ID": game_id,
                "TEAM_ID": home_team_id,
                "GAME_DATE": game_date,
                "WL": "W",
            },
            {
                "GAME_ID": game_id,
                "TEAM_ID": away_team_id,
                "GAME_DATE": game_date,
                "WL": "L",
            },
        ])
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def mock_api(monkeypatch, games):
    class LeagueGameLog:
        def __init__(self, **kwargs):
            assert kwargs == {
                "player_or_team_abbreviation": "T",
                "season": "2026-27",
                "season_type_all_star": "Regular Season",
                "timeout": 60,
            }

        def get_data_frames(self):
            return [games]

    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", LeagueGameLog)


def test_bundled_snapshot_passes_completeness_validation():
    snapshot = (
        nba_predictions.ROOT
        / "data"
        / "historical"
        / f"nba_games_{nba_predictions.CURRENT_SEASON}.csv"
    )
    source = pd.read_csv(snapshot)

    results, latest_game_date = nba_predictions._validate_complete_team_results(
        source
    )

    assert len(results) == len(source)
    assert results["GAME_ID"].nunique() * 2 == len(results)
    assert results["TEAM_ID"].nunique() == nba_predictions.NBA_TEAM_COUNT
    assert latest_game_date == pd.to_datetime(
        source["GAME_DATE"],
        format="mixed",
    ).max().date()


def test_refresh_writes_complete_newer_results_atomically(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "nba_games_2026-27.csv"
    previous_games = completed_team_games("2026-11-18", 22600001)
    previous_games.to_csv(snapshot, index=False)
    api_games = pd.concat([
        previous_games,
        completed_team_games("2026-11-20", 22600016),
    ], ignore_index=True)
    mock_api(monkeypatch, api_games)
    monkeypatch.setattr(
        nba_predictions,
        "_today_eastern",
        lambda: date(2026, 11, 21),
    )

    latest_game_date = refresh_current_season_team_snapshot(snapshot)

    refreshed = pd.read_csv(snapshot, dtype={"GAME_ID": str})
    assert latest_game_date == date(2026, 11, 20)
    assert len(refreshed) == 60
    assert refreshed["GAME_DATE"].max() == "2026-11-20"
    assert refreshed["TEAM_ID"].nunique() == 30
    assert set(refreshed["WL"]) == {"W", "L"}
    assert not list(tmp_path.glob(".*.tmp"))


def test_api_timeout_preserves_valid_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "nba_games_2026-27.csv"
    previous_games = completed_team_games("2026-11-18", 22600001)
    previous_games.to_csv(snapshot, index=False)
    original_bytes = snapshot.read_bytes()

    def timeout(**kwargs):
        raise TimeoutError("NBA API timed out")

    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", timeout)
    monkeypatch.setattr(
        nba_predictions,
        "_today_eastern",
        lambda: date(2026, 11, 21),
    )

    with pytest.raises(TimeoutError, match="timed out"):
        refresh_current_season_team_snapshot(snapshot)

    assert snapshot.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("invalid_row", "expected_error"),
    [
        ({"TEAM_ID": "invalid"}, "invalid team IDs"),
        ({"GAME_DATE": "not-a-date"}, "invalid game dates"),
        ({"WL": "T"}, "invalid win/loss outcomes"),
    ],
)
def test_invalid_api_data_does_not_replace_snapshot(
    monkeypatch,
    tmp_path,
    invalid_row,
    expected_error,
):
    snapshot = tmp_path / "nba_games_2026-27.csv"
    previous_games = completed_team_games("2026-11-18", 22600001)
    previous_games.to_csv(snapshot, index=False)
    original_bytes = snapshot.read_bytes()
    new_games = completed_team_games("2026-11-20", 22600016)
    for column, value in invalid_row.items():
        new_games[column] = new_games[column].astype("object")
        new_games.loc[new_games.index[0], column] = value
    api_games = pd.concat([
        previous_games,
        new_games,
    ], ignore_index=True)
    mock_api(monkeypatch, api_games)
    monkeypatch.setattr(
        nba_predictions,
        "_today_eastern",
        lambda: date(2026, 11, 21),
    )

    with pytest.raises(ValueError, match=expected_error):
        refresh_current_season_team_snapshot(snapshot)

    assert snapshot.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".*.tmp"))


def test_incomplete_game_does_not_replace_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "nba_games_2026-27.csv"
    previous_games = completed_team_games("2026-11-18", 22600001)
    previous_games.to_csv(snapshot, index=False)
    original_bytes = snapshot.read_bytes()
    incomplete_games = pd.concat([
        previous_games,
        completed_team_games("2026-11-20", 22600016).iloc[:-1],
    ], ignore_index=True)
    mock_api(monkeypatch, incomplete_games)
    monkeypatch.setattr(
        nba_predictions,
        "_today_eastern",
        lambda: date(2026, 11, 21),
    )

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        refresh_current_season_team_snapshot(snapshot)

    assert snapshot.read_bytes() == original_bytes


def test_older_api_data_does_not_replace_newer_snapshot(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "nba_games_2026-27.csv"
    current_games = pd.concat([
        completed_team_games("2026-11-18", 22600001),
        completed_team_games("2026-11-20", 22600016),
    ], ignore_index=True)
    current_games.to_csv(snapshot, index=False)
    original_bytes = snapshot.read_bytes()
    mock_api(
        monkeypatch,
        completed_team_games("2026-11-18", 22600001),
    )
    monkeypatch.setattr(
        nba_predictions,
        "_today_eastern",
        lambda: date(2026, 11, 21),
    )

    with pytest.raises(ValueError, match="older than the current snapshot"):
        refresh_current_season_team_snapshot(snapshot)

    assert snapshot.read_bytes() == original_bytes
