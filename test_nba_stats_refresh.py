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
    nba_stats._validated_at.clear()


def test_stale_cache_survives_api_failure_and_failure_is_throttled(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(7, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    cached = pd.DataFrame({
        "GAME_ID": ["old-game"],
        "GAME_DATE": ["Nov 10, 2026"],
        "PTS": [12],
    })
    cached.to_csv(path, index=False)
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
    assert second.attrs["refresh_error"]
    assert len(calls) == 1
    assert pd.read_csv(path)["GAME_ID"].tolist() == ["old-game"]


def test_recent_file_with_old_game_data_refreshes_and_merges_atomically(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(8, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "GAME_ID": ["old-game", "prior-game"],
        "GAME_DATE": ["Nov 10, 2026", "Nov 9, 2026"],
        "PTS": [12, 8],
    }).to_csv(path, index=False)

    class GameLog:
        def get_data_frames(self):
            return [pd.DataFrame({
                "Game_ID": ["old-game", "new-game"],
                "GAME_DATE": ["Nov 10, 2026", "Nov 17, 2026"],
                "PTS": [13, 25],
            })]

    calls = []
    monkeypatch.setattr(
        nba_stats.playergamelog,
        "PlayerGameLog",
        lambda **kwargs: (calls.append(kwargs), GameLog())[1],
    )
    games = nba_stats.get_player_games(8, nba_stats.CURRENT_SEASON)

    assert games["GAME_ID"].tolist() == [
        "new-game",
        "old-game",
        "prior-game",
    ]
    assert games["PTS"].tolist() == [25, 13, 8]
    assert games.attrs["refresh_error"] is None
    assert len(calls) == 1
    assert pd.read_csv(path)["GAME_ID"].tolist() == [
        "new-game",
        "old-game",
        "prior-game",
    ]
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_matching_latest_game_validates_cache_without_rewriting_or_repeating(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(11, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    cached = pd.DataFrame({
        "Game_ID": ["latest", "prior"],
        "GAME_DATE": ["Nov 19, 2026", "Nov 17, 2026"],
        "PTS": [20, 12],
    })
    cached.to_csv(path, index=False)
    original_bytes = path.read_bytes()
    calls = []

    class GameLog:
        def get_data_frames(self):
            return [cached.copy()]

    def refresh(**kwargs):
        calls.append(kwargs)
        return GameLog()

    monkeypatch.setattr(
        nba_stats.playergamelog,
        "PlayerGameLog",
        refresh,
    )

    first = nba_stats.get_player_games(11, nba_stats.CURRENT_SEASON)
    second = nba_stats.get_player_games(11, nba_stats.CURRENT_SEASON)

    assert first["Game_ID"].tolist() == ["latest", "prior"]
    assert second["Game_ID"].tolist() == ["latest", "prior"]
    assert first.attrs["refresh_error"] is None
    assert len(calls) == 1
    assert path.read_bytes() == original_bytes


def test_padded_and_unpadded_game_ids_merge_as_one_game(
    monkeypatch, tmp_path
):
    use_test_cache(monkeypatch, tmp_path)
    path = nba_stats.cache_path(12, nba_stats.CURRENT_SEASON)
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "Game_ID": ["22600240", "22600200"],
        "GAME_DATE": ["Nov 15, 2026", "Nov 08, 2026"],
        "PTS": [16, 9],
    }).to_csv(path, index=False)

    cached = nba_stats.read_cached_games(12, nba_stats.CURRENT_SEASON)
    assert cached["Game_ID"].tolist() == ["0022600240", "0022600200"]

    class GameLog:
        def get_data_frames(self):
            return [pd.DataFrame({
                "Game_ID": ["0022600245", "0022600240"],
                "GAME_DATE": ["Nov 16, 2026", "Nov 15, 2026"],
                "PTS": [9, 16],
            })]

    monkeypatch.setattr(
        nba_stats.playergamelog,
        "PlayerGameLog",
        lambda **kwargs: GameLog(),
    )
    games = nba_stats.get_player_games(
        12, nba_stats.CURRENT_SEASON, force_refresh=True
    )

    expected = ["0022600245", "0022600240", "0022600200"]
    assert games["Game_ID"].tolist() == expected
    saved = pd.read_csv(path, dtype={"Game_ID": str})
    assert saved["Game_ID"].tolist() == expected


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
    pd.DataFrame({
        "GAME_ID": ["last-success"],
        "GAME_DATE": ["Nov 10, 2026"],
        "PTS": [18],
    }).to_csv(path, index=False)
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
    assert "returned no current-season games" in first.attrs["refresh_error"]
    assert second["GAME_ID"].tolist() == ["last-success"]
    assert "returned no current-season games" in second.attrs["refresh_error"]
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

