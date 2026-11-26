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


# --- Today / Tomorrow selector ------------------------------------------------

def _day_fixtures(monkeypatch, games_by_date, results=None):
    requested = []
    saved = []

    def matchups(requested_date):
        requested.append(requested_date)
        return games_by_date.get(requested_date, pd.DataFrame()), "test schedule", None

    def save(predictions):
        saved.append(predictions)
        return 0

    if results is None:
        results = pd.DataFrame()
    monkeypatch.setattr(nba_predictions, "load_todays_matchups", matchups)
    monkeypatch.setattr(
        nba_predictions, "load_current_season_team_games", lambda: results
    )
    monkeypatch.setattr(
        nba_predictions, "load_previous_season_team_games", lambda: pd.DataFrame()
    )
    monkeypatch.setattr(nba_predictions, "save_prediction_snapshots", save)
    monkeypatch.setattr(
        nba_predictions, "read_prediction_history", lambda: pd.DataFrame()
    )
    return requested, saved


def _run_game_winners(day_value=None):
    today = datetime.now(nba_predictions.EASTERN).date()
    st.cache_data.clear()
    app = AppTest.from_file(
        str(Path(__file__).with_name("app.py")), default_timeout=30
    )
    app.session_state["cv_authenticated"] = True
    app.session_state["cv_dashboard_view"] = "Game winners"
    if day_value:
        app.session_state[f"cv_game_winners_day_{today}"] = day_value
    app.run()
    return app, today


def _games(game_date, count, start_hour_et=19):
    from nba_api.stats.static import teams

    team_list = teams.get_teams()[: count * 2]
    tip = pd.Timestamp(f"{game_date} {start_hour_et}:00", tz=nba_predictions.EASTERN)
    return pd.DataFrame([{
        "game_id": f"00226{index:05d}",
        "game_date": str(game_date),
        "tipoff_utc": (tip + pd.Timedelta(minutes=10 * index)).tz_convert("UTC"),
        "home_team_id": str(team_list[index]["id"]),
        "home_team": team_list[index]["full_name"],
        "away_team_id": str(team_list[index + count]["id"]),
        "away_team": team_list[index + count]["full_name"],
        "status": "scheduled",
    } for index in range(count)])


def _team_results(games, end_date, leak_date=None):
    rows = []
    team_ids = list(games["home_team_id"]) + list(games["away_team_id"])
    for position, team_id in enumerate(team_ids):
        wins = 2 + (position * 3) % 7
        for game_number in range(10):
            rows.append({
                "TEAM_ID": team_id,
                "GAME_DATE": (end_date - timedelta(days=game_number)).isoformat(),
                "WL": "W" if game_number < wins else "L",
            })
    if leak_date is not None:
        rows += [{"TEAM_ID": team_ids[0], "GAME_DATE": leak_date.isoformat(),
                  "WL": "W"}] * 30
    frame = pd.DataFrame(rows)
    frame.attrs.update({
        "latest_game_date": end_date.isoformat(), "is_stale": False,
        "source": "saved snapshot + ESPN results", "warning": None,
    })
    return frame


def _pick_table(app):
    return next(
        frame.value for frame in app.dataframe if "Pick" in frame.value.columns
    )


def test_selector_defaults_to_today_when_games_are_scheduled(monkeypatch):
    today = datetime.now(nba_predictions.EASTERN).date()
    games = _games(today + timedelta(days=0), 2, start_hour_et=23)
    requested, saved = _day_fixtures(monkeypatch, {str(today): games})
    monkeypatch.setattr(
        nba_predictions, "build_pregame_predictions",
        lambda *_a, **_k: pd.DataFrame(),
    )

    app, _ = _run_game_winners()

    assert not app.exception
    selector = app.get_by_key(f"cv_game_winners_day_{today}")
    assert selector.value == "Today"
    assert str(today + timedelta(days=1)) not in requested
    assert len(saved) == 1


def test_selector_defaults_to_tomorrow_and_shows_all_predictions(monkeypatch):
    today = datetime.now(nba_predictions.EASTERN).date()
    tomorrow = today + timedelta(days=1)
    games = _games(tomorrow, 11)
    results = _team_results(games, today - timedelta(days=1))
    requested, saved = _day_fixtures(monkeypatch, {str(tomorrow): games}, results)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Tomorrow's preview must not touch the history database")

    monkeypatch.setattr(nba_predictions, "save_prediction_snapshots", forbidden)
    monkeypatch.setattr(nba_predictions, "read_prediction_history", forbidden)

    app, _ = _run_game_winners()

    assert not app.exception
    assert app.get_by_key(f"cv_game_winners_day_{today}").value == "Tomorrow"
    assert any("Tomorrow" in item.value for item in app.subheader)
    assert any("Preview only" in item.value for item in app.caption)
    table = _pick_table(app)
    assert list(table.columns) == ["Away", "Home", "Pick", "Win %", "Tipoff"]
    assert len(table) == 11
    assert all(float(value.rstrip("%")) >= 50 for value in table["Win %"])
    assert all(value.endswith(" ET") for value in table["Tipoff"])
    assert not app.error


def test_tomorrow_preview_ignores_results_on_or_after_its_date(monkeypatch):
    today = datetime.now(nba_predictions.EASTERN).date()
    tomorrow = today + timedelta(days=1)
    games = _games(tomorrow, 3)
    clean = _team_results(games, today - timedelta(days=1))
    leaky = _team_results(
        games, today - timedelta(days=1), leak_date=tomorrow
    )
    expected = nba_predictions.build_pregame_predictions(games, clean, pd.DataFrame())
    _day_fixtures(monkeypatch, {str(tomorrow): games}, leaky)

    app, _ = _run_game_winners()

    assert not app.exception
    table = _pick_table(app)
    shown = sorted(zip(table["Away"], table["Home"], table["Win %"]))
    want = sorted(zip(
        expected["away_team"], expected["home_team"],
        expected["win_probability"].map(lambda value: f"{value:.1%}"),
    ))
    assert shown == want


def test_choosing_tomorrow_manually_does_not_save_or_change_todays_rows(monkeypatch):
    today = datetime.now(nba_predictions.EASTERN).date()
    tomorrow = today + timedelta(days=1)
    todays = _games(today, 1, start_hour_et=23)
    tomorrows = _games(tomorrow, 2)
    requested, saved = _day_fixtures(
        monkeypatch, {str(today): todays, str(tomorrow): tomorrows},
        _team_results(pd.concat([todays, tomorrows]), today - timedelta(days=1)),
    )

    app, _ = _run_game_winners("Tomorrow")

    assert not app.exception
    assert str(tomorrow) in requested
    assert saved == []
    assert len(_pick_table(app)) == 2


def test_tomorrow_without_games_says_so(monkeypatch):
    _day_fixtures(monkeypatch, {})

    app, today = _run_game_winners()

    assert not app.exception
    assert app.get_by_key(f"cv_game_winners_day_{today}").value == "Tomorrow"
    assert any("No NBA games are scheduled for tomorrow" in i.value for i in app.info)
