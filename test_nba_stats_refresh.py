from datetime import datetime, timedelta, timezone
import os

import pandas as pd

import nba_stats


def use_test_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nba_stats,
        "cache_path",
        lambda player_id, season: (
            tmp_path / season / f"{player_id}.csv"
        ),
    )
    nba_stats._failed_refresh_until.clear()
    nba_stats._empty_refresh_until.clear()


def test_stale_cache_survives_api_failure_and_failure_is_throttled(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(7, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    cached = pd.DataFrame({"GAME_ID": ["old-game"], "PTS": [12]})
    cached.to_csv(path, index=False)
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    os.utime(path, (old_time.timestamp(), old_time.timestamp()))
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("NBA API unavailable")

    monkeypatch.setattr(nba_stats.playergamelog, "PlayerGameLog", fail)

    first = nba_stats.get_player_games(7, nba_stats.CURRENT_SEASON)
    second = nba_stats.get_player_games(7, nba_stats.CURRENT_SEASON)

    assert first["GAME_ID"].tolist() == ["old-game"]
    assert first.attrs["refresh_error"]
    assert second["GAME_ID"].tolist() == ["old-game"]
    assert len(calls) == 1
    assert pd.read_csv(path)["GAME_ID"].tolist() == ["old-game"]


def test_successful_refresh_replaces_stale_cache_atomically(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(8, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GAME_ID": ["old-game"]}).to_csv(path, index=False)
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    os.utime(path, (old_time.timestamp(), old_time.timestamp()))

    class GameLog:
        def get_data_frames(self):
            return [pd.DataFrame({"GAME_ID": ["new-game"], "PTS": [25]})]

    monkeypatch.setattr(
        nba_stats.playergamelog,
        "PlayerGameLog",
        lambda **kwargs: GameLog(),
    )
    games = nba_stats.get_player_games(8, nba_stats.CURRENT_SEASON)

    assert games["GAME_ID"].tolist() == ["new-game"]
    assert games.attrs["refresh_error"] is None
    assert pd.read_csv(path)["GAME_ID"].tolist() == ["new-game"]
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_empty_refresh_keeps_existing_data_without_repeating_api_calls(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(10, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GAME_ID": ["last-success"], "PTS": [18]}).to_csv(
        path,
        index=False,
    )
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    os.utime(path, (old_time.timestamp(), old_time.timestamp()))
    calls = []

    class EmptyLog:
        def get_data_frames(self):
            return [pd.DataFrame()]

    def empty(**kwargs):
        calls.append(kwargs)
        return EmptyLog()

    monkeypatch.setattr(nba_stats.playergamelog, "PlayerGameLog", empty)
    first = nba_stats.get_player_games(10, nba_stats.CURRENT_SEASON)
    second = nba_stats.get_player_games(10, nba_stats.CURRENT_SEASON)

    assert first["GAME_ID"].tolist() == ["last-success"]
    assert first.attrs["refresh_error"] is None
    assert second["GAME_ID"].tolist() == ["last-success"]
    assert len(calls) == 1
    assert pd.read_csv(path)["GAME_ID"].tolist() == ["last-success"]


def test_historical_cache_is_not_refreshed(monkeypatch, tmp_path):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(9, "2025-26")
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GAME_ID": ["historical"]}).to_csv(path, index=False)

    def unexpected(**kwargs):
        raise AssertionError("Historical seasons should use local cache")

    monkeypatch.setattr(nba_stats.playergamelog, "PlayerGameLog", unexpected)
    assert nba_stats.get_player_games(9, "2025-26")["GAME_ID"].tolist() == [
        "historical"
    ]
