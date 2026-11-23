import pandas as pd

import todays_schedule as ts


def test_format_tipoff_converts_to_eastern():
    assert ts.format_tipoff("2026-11-22T00:30:00Z") == "7:30 PM ET"
    assert ts.format_tipoff("2026-07-01T23:00:00Z") == "7:00 PM ET"
    assert ts.format_tipoff(None) == "TBD"


def test_rows_sorted_and_formatted(monkeypatch):
    games = pd.DataFrame({
        "away_team": ["B", "A"],
        "home_team": ["D", "C"],
        "tipoff_utc": ["2026-11-23T02:00:00Z", "2026-11-23T00:00:00Z"],
    })
    monkeypatch.setattr(
        "nba_predictions.load_todays_matchups",
        lambda d: (games, "x", None),
    )
    rows, error = ts.fetch_schedule_rows(None)
    assert not error
    assert [r["Away"] for r in rows] == ["A", "B"]
    assert rows[0]["Tipoff"] == "7:00 PM ET"


def test_empty_and_failure(monkeypatch):
    monkeypatch.setattr(
        "nba_predictions.load_todays_matchups",
        lambda d: (pd.DataFrame(), "NBA scoreboard", None),
    )
    assert ts.fetch_schedule_rows(None) == ([], False)

    monkeypatch.setattr(
        "nba_predictions.load_todays_matchups",
        lambda d: (pd.DataFrame(), "unavailable", "could not be loaded (X)."),
    )
    assert ts.fetch_schedule_rows(None) == ([], True)

    def boom(d):
        raise RuntimeError

    monkeypatch.setattr("nba_predictions.load_todays_matchups", boom)
    assert ts.fetch_schedule_rows(None) == ([], True)


def _g(status, text="", period=0, clock="", a=0, h=0):
    return {"gameStatus": status, "gameStatusText": text, "period": period,
            "gameClock": clock, "awayTeam": {"score": a}, "homeTeam": {"score": h}}


def test_live_display_states():
    assert ts.live_display(_g(1, "7:00 pm ET"), "7:00 PM ET") == ("", "7:00 PM ET")
    assert ts.live_display(_g(2, "Q3 5:42", 3, "PT05M42.00S", 70, 75), "x") == ("70-75", "Q3 5:42")
    assert ts.live_display(_g(2, "Half", 2, "PT00M00.00S", 50, 55), "x") == ("50-55", "Halftime")
    assert ts.live_display(_g(3, "Final", 4, "", 100, 98), "x") == ("100-98", "Final")
    assert ts.live_display(_g(3, "Final/OT", 5, "", 110, 108), "x")[1] == "Final/OT"


def test_build_rows_merges_live_by_game_id():
    rows = [{"game_id": "0022600001", "Away": "A", "Home": "B", "Tipoff": "7:00 PM ET"},
            {"game_id": "0022600002", "Away": "C", "Home": "D", "Tipoff": "9:00 PM ET"}]
    out = ts.build_display_rows(rows, {"0022600001": _g(3, "Final", 4, "", 90, 80)})
    assert out[0]["Score"] == "90-80" and out[0]["Status"] == "Final"
    assert out[1] == {"Away": "C", "Home": "D", "Score": "", "Status": "9:00 PM ET"}
    assert ts.has_active_games({"x": _g(2, "Half")})
    assert not ts.has_active_games({"x": _g(3, "Final")})


def test_stale_scoreboard_rejected(monkeypatch):
    import datetime as dt
    import nba_api.live.nba.endpoints.scoreboard as sb

    class Fake:
        def __init__(self, *a, **k): pass
        def get_dict(self):
            return {"scoreboard": {"gameDate": "2026-11-21", "games": [{"gameId": "1"}]}}

    monkeypatch.setattr(sb, "ScoreBoard", Fake)
    try:
        ts.fetch_live_scoreboard(dt.date(2026, 11, 22))
        assert False
    except ValueError:
        pass
    assert "0000000001" in ts.fetch_live_scoreboard(dt.date(2026, 11, 21))


def test_api_failure_renders_schedule_without_crash(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def app():
        import todays_schedule as t
        t.cached_schedule_rows = lambda k: [
            {"game_id": "0022600001", "Away": "A", "Home": "B", "Tipoff": "7:00 PM ET"}]
        def boom(k): raise RuntimeError("down")
        t.cached_live_scoreboard = boom
        t.render_todays_schedule()

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert any("unavailable" in c.value for c in at.caption)
