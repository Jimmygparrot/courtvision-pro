from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import nba_predictions
from backtesting import nba_daily_predictions


@pytest.mark.parametrize(
    ("is_stale", "days_since_latest_game", "expected_freshness"),
    [
        (False, 1, "fresh source data"),
        (True, 3, "stale saved snapshot"),
        (False, 3, "stale source data"),
    ],
)
def test_game_winners_separates_current_and_original_predictions(
    monkeypatch,
    is_stale,
    days_since_latest_game,
    expected_freshness,
):
    now = datetime.now(timezone.utc)
    today = now.astimezone(nba_predictions.EASTERN).date()
    game_date = today.isoformat()
    latest_game_date = (
        today - timedelta(days=days_since_latest_game)
    ).isoformat()
    tipoff = now + timedelta(hours=2)
    game_id = "0022600001"
    games = pd.DataFrame([{
        "game_id": game_id,
        "game_date": game_date,
        "tipoff_utc": tipoff,
        "home_team_id": "1",
        "home_team": "Boston Celtics",
        "away_team_id": "2",
        "away_team": "New York Knicks",
        "status": "scheduled",
    }])
    team_results = pd.DataFrame([{
        "TEAM_ID": "1",
        "GAME_DATE": latest_game_date,
        "WL": "W",
    }])
    team_results.attrs.update({
        "latest_game_date": latest_game_date,
        "is_stale": is_stale,
        "source": (
            "saved current-season snapshot" if is_stale else "NBA API"
        ),
        "warning": (
            "NBA API refresh failed; using the saved snapshot."
            if is_stale else None
        ),
    })
    saved_at = (now - timedelta(hours=1)).isoformat()
    saved_predictions = pd.DataFrame([{
        "game_id": game_id,
        "game_date": game_date,
        "tipoff_utc": tipoff.isoformat(),
        "home_team_id": "1",
        "home_team": "Boston Celtics",
        "away_team_id": "2",
        "away_team": "New York Knicks",
        "predicted_winner": "New York Knicks",
        "home_win_probability": 0.549,
        "away_win_probability": 0.451,
        "win_probability": 0.549,
        "created_at": saved_at,
        "model_version": nba_predictions.MODEL_VERSION,
        "rating_source": "neutral early-season baseline",
        "actual_winner": None,
        "completed_at": None,
    }])

    monkeypatch.setattr(
        nba_predictions,
        "load_todays_matchups",
        lambda requested_date: (games, "test schedule", None),
    )
    monkeypatch.setattr(
        nba_predictions,
        "load_current_season_team_games",
        lambda: team_results,
    )
    monkeypatch.setattr(
        nba_predictions,
        "load_previous_season_team_games",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        nba_predictions,
        "build_pregame_predictions",
        lambda *_args, **_kwargs: pd.DataFrame([{
            "game_id": game_id,
            "game_date": game_date,
            "tipoff_utc": tipoff,
            "home_team_id": "1",
            "home_team": "Boston Celtics",
            "away_team_id": "2",
            "away_team": "New York Knicks",
            "predicted_winner": "Boston Celtics",
            "home_win_probability": 0.672,
            "away_win_probability": 0.328,
            "win_probability": 0.672,
            "created_at": now.isoformat(),
            "model_version": nba_predictions.MODEL_VERSION,
            "rating_source": (
                "current-season results through prior day"
            ),
            "status": "scheduled",
        }]),
    )
    monkeypatch.setattr(
        nba_predictions,
        "save_prediction_snapshots",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        nba_predictions,
        "read_prediction_history",
        lambda: saved_predictions,
    )
    monkeypatch.setattr(
        nba_daily_predictions,
        "build_prediction_pdf",
        lambda *_args, **_kwargs: b"%PDF-test",
    )

    st.cache_data.clear()
    app_path = Path(__file__).with_name("app.py")
    app = AppTest.from_file(str(app_path), default_timeout=10)
    app.session_state["cv_authenticated"] = True
    app.session_state["cv_dashboard_view"] = "Game winners"
    app.run()

    assert not app.exception
    assert "Current Prediction" in [item.value for item in app.subheader]
    assert "Original Saved Prediction" in [
        item.value for item in app.subheader
    ]

    current_table = next(
        frame.value
        for frame in app.dataframe
        if "Current home win probability" in frame.value.columns
    )
    saved_table = next(
        frame.value
        for frame in app.dataframe
        if "Original saved home win probability" in frame.value.columns
    )
    assert current_table.iloc[0]["Current home win probability"] == "67.2%"
    assert current_table.iloc[0]["Current away win probability"] == "32.8%"
    assert saved_table.iloc[0]["Original saved home win probability"] == "54.9%"
    assert saved_table.iloc[0]["Original saved away win probability"] == "45.1%"
    assert saved_table.iloc[0]["Original saved at (ET)"] == (
        nba_predictions.format_eastern_datetime(saved_at)
    )
    latest_game_label = pd.Timestamp(latest_game_date).strftime(
        "%b %d, %Y"
    ).replace(" 0", " ")
    assert any(
        f"through {latest_game_label}" in caption.value
        and expected_freshness in caption.value
        for caption in app.caption
    )
