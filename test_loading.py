"""Regression checks for independent stats/odds loading and Arrow-safe tables.

Run with: python -m pytest test_loading.py
"""
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import nba_stats
import live_odds
import nba_predictions


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.chdir(ROOT)
    st.cache_data.clear()

    def blocked(*args, **kwargs):
        raise AssertionError("Unexpected external request")

    monkeypatch.setattr(requests, 'get', blocked)
    monkeypatch.setattr(nba_stats, 'get_player_games', blocked)
    yield
    st.cache_data.clear()


def app():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90)
    at.session_state['cv_authenticated'] = True
    at.run()
    assert not at.exception
    return at


def button(at, label):
    return next(item for item in at.button if item.label == label)


def game_logs():
    return pd.DataFrame({
        'Game_ID': [str(100 + i) for i in range(20)],
        'GAME_DATE': pd.date_range('2026-01-01', periods=20),
        'MATCHUP': ['BOS vs. NYK'] * 20,
        'MIN': [35] * 20,
        'PTS': [20 + i % 10 for i in range(20)],
        'REB': [3 + i % 5 for i in range(20)],
        'AST': [2 + i % 6 for i in range(20)],
    })


def configure_odds(monkeypatch, player, team, calls):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self.payload

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith('/events/'):
            return Response([{
                'id': 'event1', 'home_team': team, 'away_team': 'New York Knicks',
                'commence_time': '2099-01-01T00:00:00Z'
            }])
        return Response({'bookmakers': [{'title': 'DraftKings', 'markets': [{
            'key': 'player_points', 'outcomes': [
                {'description': player, 'name': 'Over', 'point': 22.5, 'price': -110},
                {'description': player, 'name': 'Under', 'point': 22.5, 'price': -105},
            ]
        }]}]})
    monkeypatch.setattr(live_odds, 'get_api_key', lambda: 'test-private-key')
    monkeypatch.setattr(requests, 'get', get)


def test_startup_renders_controls_without_external_requests():
    at = app()
    assert button(at, 'Load Player Statistics')
    assert button(at, 'Load / Refresh Live Odds')
    assert any(item.label == 'Save Odds' for item in at.button)
    assert at.metric[0].value == '—'
    for table in at.dataframe:
        pa.Table.from_pandas(table.value)


def test_game_winners_view_shows_empty_schedule_without_external_requests(
    monkeypatch,
):
    monkeypatch.setattr(
        nba_predictions,
        'load_todays_matchups',
        lambda game_date: (pd.DataFrame(), 'test schedule', None),
    )
    monkeypatch.setattr(
        nba_predictions,
        'load_current_season_team_games',
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        nba_predictions,
        'read_prediction_history',
        lambda: pd.DataFrame(),
    )
    at = app()
    at.get_by_key('cv_dashboard_view').set_value('Game winners').run()
    assert not at.exception
    today = datetime.now(ZoneInfo("America/New_York")).date()
    at.get_by_key(f'cv_game_winners_day_{today}').set_value('Today').run()
    assert not at.exception
    assert any(
        'No current or original pregame predictions' in item.value
        for item in at.info
    )


def test_game_winners_view_renders_predictions_and_pdf(monkeypatch):
    now = datetime.now(ZoneInfo("America/New_York"))
    games = pd.DataFrame([{
        "game_id": "fixture",
        "game_date": str(now.date()),
        "tipoff_utc": (now + timedelta(hours=1)).astimezone(
            timezone.utc
        ).isoformat(),
        "home_team_id": "1",
        "home_team": "Boston Celtics",
        "away_team_id": "2",
        "away_team": "New York Knicks",
        "status": "scheduled",
    }])
    predictions = nba_predictions.build_pregame_predictions(
        games,
        pd.DataFrame(),
        pd.DataFrame(),
        now=now,
    )
    monkeypatch.setattr(
        nba_predictions,
        'load_todays_matchups',
        lambda game_date: (games, 'test schedule', None),
    )
    monkeypatch.setattr(
        nba_predictions,
        'load_current_season_team_games',
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        nba_predictions,
        'load_previous_season_team_games',
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        nba_predictions,
        'save_prediction_snapshots',
        lambda rows: 0,
    )
    monkeypatch.setattr(
        nba_predictions,
        'read_prediction_history',
        lambda: predictions,
    )

    at = app()
    at.get_by_key('cv_dashboard_view').set_value('Game winners').run()

    assert not at.exception
    prediction_table = next(
        table.value for table in at.dataframe
        if 'Current predicted winner' in table.value.columns
    )
    assert 'Current home win probability' in prediction_table.columns
    assert 'Current away win probability' in prediction_table.columns
    assert 'Calculated at (ET)' in prediction_table.columns
    assert 'Game Tipoff (ET)' in prediction_table.columns
    assert 'Prediction timestamp (UTC)' not in prediction_table.columns
    assert predictions.iloc[0]['created_at'].endswith('+00:00')
    assert prediction_table.iloc[0]['Calculated at (ET)'].endswith(
        ('EST', 'EDT')
    )
    assert prediction_table.iloc[0]['Game Tipoff (ET)'].endswith(
        ('EST', 'EDT')
    )
    assert prediction_table.iloc[0]['Current home win probability'].endswith(
        '%'
    )
    assert prediction_table.iloc[0]['Current away win probability'].endswith(
        '%'
    )

    original_table = next(
        table.value for table in at.dataframe
        if 'Original saved predicted winner' in table.value.columns
    )
    assert 'Original saved home win probability' in original_table.columns
    assert 'Original saved away win probability' in original_table.columns
    assert 'Original saved at (ET)' in original_table.columns
    assert original_table.iloc[0]['Original saved home win probability'] \
        .endswith('%')
    assert any(
        item.value == 'Original Saved Prediction' for item in at.subheader
    )
    assert not at.error
    assert len(at.get('download_button')) == 1


def test_prediction_history_renders_eastern_times_and_both_probabilities(
    monkeypatch,
):
    now = datetime.now(ZoneInfo("America/New_York"))
    games = pd.DataFrame([{
        "game_id": "history-fixture",
        "game_date": str(now.date()),
        "tipoff_utc": (now + timedelta(hours=1)).astimezone(
            timezone.utc
        ).isoformat(),
        "home_team_id": "1",
        "home_team": "Boston Celtics",
        "away_team_id": "2",
        "away_team": "New York Knicks",
        "status": "scheduled",
    }])
    history = nba_predictions.build_pregame_predictions(
        games,
        pd.DataFrame(),
        pd.DataFrame(),
        now=now,
    )
    history['actual_winner'] = None
    history['completed_at'] = None
    monkeypatch.setattr(
        nba_predictions,
        'load_current_season_team_games',
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        nba_predictions,
        'extract_completed_results',
        lambda team_games: {},
    )
    monkeypatch.setattr(
        nba_predictions,
        'update_prediction_results',
        lambda actual_winners: 0,
    )
    monkeypatch.setattr(
        nba_predictions,
        'read_prediction_history',
        lambda: history,
    )

    at = app()
    at.get_by_key('cv_dashboard_view').set_value('Prediction history').run()

    assert not at.exception
    history_table = next(
        table.value for table in at.dataframe
        if 'Actual winner' in table.value.columns
    )
    assert 'Home win probability' in history_table.columns
    assert 'Away win probability' in history_table.columns
    assert 'Prediction timestamp (ET)' in history_table.columns
    assert 'Game Tipoff (ET)' in history_table.columns
    assert history_table.iloc[0]['Prediction timestamp (ET)'].endswith(
        ('EST', 'EDT')
    )
    assert history_table.iloc[0]['Game Tipoff (ET)'].endswith(
        ('EST', 'EDT')
    )
    assert history_table.iloc[0]['Home win probability'].endswith('%')
    assert history_table.iloc[0]['Away win probability'].endswith('%')


def test_history_outage_shows_error_without_fallback(monkeypatch):
    import db

    def unavailable(*args, **kwargs):
        raise db.HistoryUnavailable("Could not connect to the prediction history database.")

    monkeypatch.setattr(nba_predictions, 'load_current_season_team_games', lambda: pd.DataFrame())
    monkeypatch.setattr(nba_predictions, 'extract_completed_results', lambda team_games: {})
    monkeypatch.setattr(nba_predictions, 'update_prediction_results', unavailable)
    monkeypatch.setattr(nba_predictions, 'read_prediction_history', unavailable)

    at = app()
    at.get_by_key('cv_dashboard_view').set_value('Prediction history').run()

    assert not at.exception
    assert any('temporarily unavailable' in item.value for item in at.error)
    assert not any('Actual winner' in t.value.columns for t in at.dataframe)


def test_statistics_failure_can_retry_and_widgets_do_not_reload(monkeypatch):
    calls = []
    def download(*args, **kwargs):
        calls.append(args)
        return None if len(calls) <= 2 else game_logs()
    monkeypatch.setattr(nba_stats, 'get_player_games', download)
    at = app()
    button(at, 'Load Player Statistics').click().run()
    assert not at.exception
    assert any('could not be downloaded' in item.value for item in at.warning)
    assert button(at, 'Load / Refresh Live Odds')
    button(at, 'Load Player Statistics').click().run()
    assert not at.exception
    assert len(calls) == 3
    assert at.metric[0].value != '—'
    recent_games = next(
        table.value for table in at.dataframe
        if 'GAME_DATE' in table.value.columns
    )
    assert len(recent_games) == 20
    assert recent_games['GAME_DATE'].is_monotonic_decreasing
    next(item for item in at.selectbox if item.label == 'Select Statistic').select('PRA').run()
    assert not at.exception
    assert len(calls) == 3


def test_odds_load_without_stats_and_mixed_tables_serialize(monkeypatch):
    at = app()
    player = at.selectbox(key='cv_player_select').value
    directory = pd.read_csv(ROOT / 'nba_players_teams.csv')
    team = directory[directory.full_name == player].iloc[0].team_name
    calls = []
    configure_odds(monkeypatch, player, team, calls)
    button(at, 'Load / Refresh Live Odds').click().run()
    assert not at.exception
    assert len(calls) == 2
    assert at.session_state['cv_live_results'][0]['Sportsbook'] == 'DraftKings'
    assert at.metric[0].value == '—'
    assert any('vs.' in item.value for item in at.markdown)
    for table in at.dataframe:
        pa.Table.from_pandas(table.value)
    assert calls[0][1]['timeout'] == (3.05, 8)
    at.number_input(key='ev_prop_line_' + player + '_Points').set_value(25.5).run()
    assert not at.exception
    assert len(calls) == 2
    button(at, 'Load / Refresh Live Odds').click().run()
    assert len(calls) == 4
    next(item for item in at.selectbox if item.label == 'Select Statistic').select('Rebounds').run()
    assert not at.exception
    assert at.session_state['cv_live_results'] == []
    assert at.session_state['cv_selected_event'] is None
    assert len(calls) == 4


def test_odds_error_does_not_expose_api_key_or_block_manual_entry(monkeypatch):
    at = app()
    monkeypatch.setattr(live_odds, 'get_api_key', lambda: 'super-secret')
    def fail(*args, **kwargs):
        raise requests.HTTPError('https://example.com?apiKey=super-secret')
    monkeypatch.setattr(requests, 'get', fail)
    button(at, 'Load / Refresh Live Odds').click().run()
    assert not at.exception
    assert any('Live odds could not be loaded' in item.value for item in at.error)
    assert all('super-secret' not in item.value for item in at.error)
    button(at, 'Save Odds').click().run()
    assert not at.exception
    assert len(at.session_state['cv_manual_odds']) == 1
    for table in at.dataframe:
        pa.Table.from_pandas(table.value)


def test_statistics_and_live_odds_enable_ev_without_more_downloads(monkeypatch):
    stats_calls = []
    def download(*args, **kwargs):
        stats_calls.append(args)
        return game_logs()
    monkeypatch.setattr(nba_stats, 'get_player_games', download)
    at = app()
    player = at.selectbox(key='cv_player_select').value
    directory = pd.read_csv(ROOT / 'nba_players_teams.csv')
    team = directory[directory.full_name == player].iloc[0].team_name
    calls = []
    configure_odds(monkeypatch, player, team, calls)
    button(at, 'Load Player Statistics').click().run()
    button(at, 'Load / Refresh Live Odds').click().run()
    assert not at.exception
    ev_tables = [table.value for table in at.dataframe if 'Over EV %' in table.value.columns]
    assert len(ev_tables) == 1
    assert ev_tables[0]['Over EV %'].notna().all()
    assert len(stats_calls) == 1
    assert len(calls) == 2
    for table in at.dataframe:
        pa.Table.from_pandas(table.value)


def test_accented_names_match_odds():
    outcomes = [
        {'description': 'Luka Dončić', 'name': 'Over', 'point': 25.5, 'price': -110},
        {'description': 'Luka Dončić', 'name': 'Under', 'point': 25.5, 'price': -110}
    ]
    data = {'bookmakers': [{'title': 'DraftKings', 'markets': [
        {'key': 'player_points', 'outcomes': outcomes}
    ]}]}
    rows = live_odds.compare_player('Luka Doncic', data)
    assert len(rows) == 1
    assert rows[0]['Player'] == 'Luka Doncic'


def test_empty_current_season_uses_previous_season(monkeypatch):
    calls = []
    def download(player_id, season, **kwargs):
        calls.append(season)
        return pd.DataFrame() if season == '2026-27' else game_logs()
    monkeypatch.setattr(nba_stats, 'get_player_games', download)
    at = app()
    button(at, 'Load Player Statistics').click().run()
    assert not at.exception
    assert calls == ['2026-27', '2025-26']
    assert at.metric[0].value != '—'
    player_box = at.selectbox(key='cv_player_select')
    player_box.select(player_box.options[1]).run()
    assert not at.exception
    assert at.metric[0].value == '—'
    assert at.session_state['cv_live_results'] == []
    assert calls == ['2026-27', '2025-26']


def test_nba_wrapper_distinguishes_empty_season_and_download_error(
    monkeypatch, tmp_path
):
    class EmptyLog:
        def get_data_frames(self):
            return [pd.DataFrame()]
    monkeypatch.setattr(nba_stats.playergamelog, 'PlayerGameLog', lambda **kw: EmptyLog())
    # Exercise the real wrapper; the fixture replaces the app entry point.
    import importlib.util
    spec = importlib.util.spec_from_file_location('nba_stats_wrapper_test', ROOT / 'nba_stats.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        'cache_path',
        lambda player_id, season: tmp_path / season / f'{player_id}.csv',
    )
    assert module.get_player_games(1).empty
    def fail(**kwargs):
        raise requests.Timeout('request timed out')
    monkeypatch.setattr(module.playergamelog, 'PlayerGameLog', fail)
    with pytest.raises(RuntimeError, match='could not be downloaded'):
        module.get_player_games(1, force_refresh=True)
