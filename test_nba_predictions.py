import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from backtesting.nba_daily_predictions import build_prediction_pdf
from nba_predictions import (
    build_pregame_predictions,
    extract_completed_results,
    load_todays_matchups,
    prediction_accuracy,
    read_prediction_history,
    save_prediction_snapshots,
    update_prediction_results,
)


NOW = datetime(2026, 11, 18, 16, 0, tzinfo=timezone.utc)


def schedule_rows():
    return pd.DataFrame([
        {
            "game_id": "game-1",
            "game_date": "2026-11-18",
            "tipoff_utc": "2026-11-18T23:00:00Z",
            "home_team_id": "1",
            "home_team": "Home",
            "away_team_id": "2",
            "away_team": "Away",
            "status": "scheduled",
        },
        {
            "game_id": "game-started",
            "game_date": "2026-11-18",
            "tipoff_utc": "2026-11-18T15:00:00Z",
            "home_team_id": "1",
            "home_team": "Home",
            "away_team_id": "3",
            "away_team": "Started opponent",
            "status": "in_progress",
        },
        {
            "game_id": "stale-status-started",
            "game_date": "2026-11-18",
            "tipoff_utc": "2026-11-18T15:00:00Z",
            "home_team_id": "8",
            "home_team": "Stale-status home",
            "away_team_id": "9",
            "away_team": "Stale-status away",
            "status": "scheduled",
        },
        {
            "game_id": "game-final",
            "game_date": "2026-11-18",
            "tipoff_utc": "2026-11-18T14:00:00Z",
            "home_team_id": "4",
            "home_team": "Final home",
            "away_team_id": "5",
            "away_team": "Final away",
            "status": "final",
        },
        {
            "game_id": "unknown-tipoff",
            "game_date": "2026-11-18",
            "tipoff_utc": None,
            "home_team_id": "6",
            "home_team": "Unknown-time home",
            "away_team_id": "7",
            "away_team": "Unknown-time away",
            "status": "scheduled",
        },
    ])


def test_pregame_model_uses_only_prior_date_records():
    current = pd.DataFrame([
        {"TEAM_ID": 1, "GAME_DATE": "NOV 17, 2026", "WL": "W"},
        {"TEAM_ID": 1, "GAME_DATE": "NOV 18, 2026", "WL": "L"},
        {"TEAM_ID": 2, "GAME_DATE": "NOV 17, 2026", "WL": "L"},
        {"TEAM_ID": 2, "GAME_DATE": "NOV 18, 2026", "WL": "W"},
    ])
    previous = pd.DataFrame([
        {"TEAM_ID": 1, "GAME_DATE": "APR 10, 2026", "WL": "L"},
        {"TEAM_ID": 2, "GAME_DATE": "APR 10, 2026", "WL": "W"},
    ])

    predictions = build_pregame_predictions(
        schedule_rows(),
        current,
        previous,
        now=NOW,
    )

    assert predictions["game_id"].tolist() == ["game-1"]
    # November 18 outcomes are excluded: the ratings are 6/11 and 5/11.
    expected_probability = 1 / (
        1 + math.exp(
            -(0.1982 + 4.1196 * (1 / 11))
        )
    )
    assert predictions.iloc[0]["home_win_probability"] == pytest.approx(
        expected_probability
    )
    assert predictions.iloc[0]["rating_source"] == (
        "current-season results through prior day"
    )


def test_history_is_immutable_and_results_only_update_outcomes(tmp_path):
    history_path = tmp_path / "predictions.sqlite3"
    prediction = build_pregame_predictions(
        schedule_rows(),
        pd.DataFrame(),
        pd.DataFrame(),
        now=NOW,
    )
    assert save_prediction_snapshots(prediction, history_path) == 1

    changed_prediction = prediction.copy()
    changed_prediction.loc[0, "predicted_winner"] = "Away"
    changed_prediction.loc[0, "created_at"] = (
        datetime(2026, 11, 18, 16, 1, tzinfo=timezone.utc).isoformat()
    )
    assert save_prediction_snapshots(changed_prediction, history_path) == 0

    completed = pd.DataFrame([
        {
            "GAME_ID": "game-1",
            "TEAM_ID": "1",
            "WL": "L",
            "MATCHUP": "Home vs. Away",
        },
        {
            "GAME_ID": "game-1",
            "TEAM_ID": "2",
            "WL": "W",
            "MATCHUP": "Away @ Home",
        },
        {
            "GAME_ID": "incomplete",
            "TEAM_ID": "3",
            "WL": "W",
            "MATCHUP": "One vs. Two",
        },
    ])
    actuals = extract_completed_results(completed)
    assert actuals == {"game-1": "2"}
    assert update_prediction_results(actuals, history_path) == 1

    history = read_prediction_history(history_path)
    assert history.iloc[0]["predicted_winner"] == prediction.iloc[0][
        "predicted_winner"
    ]
    assert history.iloc[0]["created_at"] == prediction.iloc[0]["created_at"]
    assert history.iloc[0]["actual_winner"] == "2"
    assert prediction_accuracy(history) == (1, 0.0)


def test_started_or_completed_games_are_not_saved(tmp_path):
    predictions = build_pregame_predictions(
        schedule_rows(),
        pd.DataFrame(),
        pd.DataFrame(),
        now=NOW,
    )
    assert predictions["game_id"].tolist() == ["game-1"]
    assert save_prediction_snapshots(
        predictions,
        tmp_path / "predictions.sqlite3",
    ) == 1


def test_pdf_report_is_generated_in_memory_for_letter_page():
    predictions = build_pregame_predictions(
        schedule_rows().iloc[[0]],
        pd.DataFrame(),
        pd.DataFrame(),
        now=NOW,
    )
    report = build_prediction_pdf(predictions, "2026-11-18")

    assert report.startswith(b"%PDF-")
    assert len(report) > 500


def test_todays_schedule_falls_back_to_saved_fixture_on_api_error(
    monkeypatch, tmp_path
):
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame([{
        "gameId": "fixture",
        "gameDateTimeUTC": "2026-11-18T23:00:00Z",
        "gameStatus": 1,
        "homeTeam_teamId": 1,
        "homeTeam_teamCity": "Boston",
        "homeTeam_teamName": "Celtics",
        "awayTeam_teamId": 2,
        "awayTeam_teamCity": "New York",
        "awayTeam_teamName": "Knicks",
    }]).to_csv(schedule_path, index=False)

    def unavailable(*args, **kwargs):
        raise TimeoutError("service unavailable")

    monkeypatch.setattr("nba_predictions._scoreboard_schedule", unavailable)
    games, source, warning = load_todays_matchups(
        "2026-11-18",
        schedule_path,
    )

    assert source == "saved schedule"
    assert games.iloc[0]["game_id"] == "fixture"
    assert games.iloc[0]["home_team"] == "Boston Celtics"
    assert "TimeoutError" in warning
