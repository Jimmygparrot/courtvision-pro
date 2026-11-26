from datetime import date

import pandas as pd
import pytest

import espn_results
import nba_predictions
from nba_api.stats.endpoints import leaguegamelog

TEAMS = {t["abbreviation"]: t for t in espn_results.teams.get_teams()}


def team(abbr):
    return TEAMS[abbr]


def scheduled(*games):
    """games: (game_id, et_date, home_abbr, away_abbr)."""
    return pd.DataFrame([{
        "gameId": game_id,
        "game_date": day,
        "homeTeam_teamId": str(team(home)["id"]),
        "awayTeam_teamId": str(team(away)["id"]),
        "gameDateTimeUTC": f"{day}T23:00:00Z",
        "homeTeam_teamCity": team(home)["city"] if "city" in team(home) else "",
        "homeTeam_teamName": team(home)["nickname"],
        "awayTeam_teamCity": "",
        "awayTeam_teamName": team(away)["nickname"],
        "gameStatus": 1,
    } for game_id, day, home, away in games])


def event(home, away, home_score, away_score, status="STATUS_FINAL", name=None):
    def side(abbr, score, home_away, other):
        return {
            "homeAway": home_away,
            "team": {"displayName": name or team(abbr)["full_name"]},
            "score": str(score),
            "winner": score > other,
        }
    return {"competitions": [{
        "status": {"type": {"name": status}},
        "competitors": [side(home, home_score, "home", away_score),
                        side(away, away_score, "away", home_score)],
    }]}


def board(*events):
    return {"events": list(events)}


def fetcher(boards, calls=None):
    def fetch(day):
        if calls is not None:
            calls.append(day)
        value = boards[day]
        if isinstance(value, Exception):
            raise value
        return value
    return fetch


def run(sched, boards, after=date(2026, 11, 20), through=date(2026, 11, 25),
        calls=None):
    return espn_results.fetch_missing_team_results(
        sched, after, through, fetch=fetcher(boards, calls)
    )


def test_confirmed_days_become_snapshot_rows_with_schedule_ids():
    sched = scheduled(("0022600055", "2026-11-21", "PHI", "MIA"))
    rows, through = run(
        sched,
        {date(2026, 11, 21): board(event("PHI", "MIA", 111, 104))},
        through=date(2026, 11, 21),
    )

    assert through == date(2026, 11, 21)
    assert set(rows["GAME_ID"]) == {"0022600055"}
    assert rows.set_index("TEAM_ABBREVIATION")["WL"].to_dict() == {
        "PHI": "W", "MIA": "L"}
    assert rows.set_index("TEAM_ABBREVIATION")["MATCHUP"].to_dict() == {
        "PHI": "PHI vs. MIA", "MIA": "MIA @ PHI"}
    assert (rows["GAME_DATE"] == "2026-11-21").all()


def test_clippers_alias_and_days_without_games_need_no_request():
    sched = scheduled(("0022600063", "2026-11-23", "LAC", "SAS"))
    calls = []
    rows, through = run(
        sched,
        {date(2026, 11, 23): board(event("LAC", "SAS", 100, 90, name=None))},
        through=date(2026, 11, 23), calls=calls,
    )
    # 11/21 and 11/22 have no scheduled games, so only 11/23 is requested.
    assert calls == [date(2026, 11, 23)]
    assert through == date(2026, 11, 23)
    assert len(rows) == 2

    alias_board = board(event("LAC", "SAS", 100, 90))
    alias_board["events"][0]["competitions"][0]["competitors"][0]["team"][
        "displayName"] = "LA Clippers"
    alias_board["events"][0]["competitions"][0]["competitors"][1]["team"][
        "displayName"] = "San Antonio Spurs"
    rows, _ = run(sched, {date(2026, 11, 23): alias_board},
                  through=date(2026, 11, 23))
    assert "LAC" in set(rows["TEAM_ABBREVIATION"])


def test_unfinished_game_stops_at_the_last_complete_day():
    sched = scheduled(
        ("0022600055", "2026-11-21", "PHI", "MIA"),
        ("0022600056", "2026-11-22", "CHA", "CHI"),
        ("0022600057", "2026-11-23", "NYK", "CLE"),
    )
    boards = {
        date(2026, 11, 21): board(event("PHI", "MIA", 111, 104)),
        date(2026, 11, 22): board(event("CHA", "CHI", 50, 48, status="STATUS_IN_PROGRESS")),
        date(2026, 11, 23): board(event("NYK", "CLE", 121, 102)),
    }
    rows, through = run(sched, boards, through=date(2026, 11, 23))

    assert through == date(2026, 11, 21)
    assert set(rows["GAME_ID"]) == {"0022600055"}


@pytest.mark.parametrize("bad_board", [
    board(event("PHI", "MIA", 111, 104), event("PHI", "MIA", 111, 104)),
    board(event("PHI", "MIA", 111, 104), event("NYK", "CLE", 121, 102)),
    board(event("PHI", "MIA", 100, 100)),
])
def test_duplicate_unscheduled_or_tied_results_are_rejected(bad_board):
    sched = scheduled(("0022600055", "2026-11-21", "PHI", "MIA"))
    rows, through = run(sched, {date(2026, 11, 21): bad_board},
                        through=date(2026, 11, 21))

    assert rows.empty
    assert through == date(2026, 11, 20)


def test_non_regular_season_games_are_not_added_but_are_tolerated():
    sched = scheduled(
        ("0022600055", "2026-11-21", "PHI", "MIA"),
        ("0062600001", "2026-11-21", "NYK", "CLE"),
    )
    rows, through = run(
        sched,
        {date(2026, 11, 21): board(
            event("PHI", "MIA", 111, 104), event("NYK", "CLE", 121, 102))},
        through=date(2026, 11, 21),
    )

    assert set(rows["GAME_ID"]) == {"0022600055"}
    assert through == date(2026, 11, 21)


def test_first_request_failure_is_reported_as_unavailable():
    sched = scheduled(("0022600055", "2026-11-21", "PHI", "MIA"))
    with pytest.raises(espn_results.EspnUnavailable):
        run(sched, {date(2026, 11, 21): espn_results.EspnUnavailable("Timeout")},
            through=date(2026, 11, 21))


def test_later_failure_keeps_earlier_confirmed_days():
    sched = scheduled(
        ("0022600055", "2026-11-21", "PHI", "MIA"),
        ("0022600056", "2026-11-22", "CHA", "CHI"),
    )
    rows, through = run(sched, {
        date(2026, 11, 21): board(event("PHI", "MIA", 111, 104)),
        date(2026, 11, 22): espn_results.EspnUnavailable("Timeout"),
    }, through=date(2026, 11, 22))

    assert through == date(2026, 11, 21)
    assert len(rows) == 2


# --- integration with the snapshot loader -----------------------------------

def write_loader_files(tmp_path, snapshot_date, sched):
    historical = tmp_path / "data" / "historical"
    historical.mkdir(parents=True)
    pd.DataFrame([
        {"SEASON_ID": "22026", "TEAM_ID": team("PHI")["id"], "GAME_ID": "0022600001",
         "GAME_DATE": snapshot_date, "MATCHUP": "PHI vs. MIA", "WL": "W"},
        {"SEASON_ID": "22026", "TEAM_ID": team("MIA")["id"], "GAME_ID": "0022600001",
         "GAME_DATE": snapshot_date, "MATCHUP": "MIA @ PHI", "WL": "L"},
    ]).to_csv(historical / "nba_games_2026-27.csv", index=False)
    sched.to_csv(tmp_path / "data" / "nba_schedule_2026-27.csv", index=False)


def patch_loader(monkeypatch, tmp_path, today, boards, calls=None):
    monkeypatch.setattr(nba_predictions, "ROOT", tmp_path)
    monkeypatch.setattr(nba_predictions, "_today_eastern", lambda: today)
    monkeypatch.setattr(espn_results, "fetch_scoreboard", fetcher(boards, calls))

    def no_nba_api(**kwargs):
        raise AssertionError("The NBA API must not be called when ESPN works")

    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", no_nba_api)


def loader_schedule():
    sched = scheduled(("0022600055", "2026-11-21", "CHA", "CHI"))
    sched["gameDateTimeUTC"] = "2026-11-22T00:00:00Z"
    return sched


def test_loader_extends_stale_snapshot_without_nba_api(monkeypatch, tmp_path):
    write_loader_files(tmp_path, "2026-11-20", loader_schedule())
    patch_loader(monkeypatch, tmp_path, date(2026, 11, 22),
                 {date(2026, 11, 21): board(event("CHA", "CHI", 105, 107))})

    results = nba_predictions.load_current_season_team_games()

    assert results.attrs["source"] == "saved snapshot + ESPN results"
    assert results.attrs["latest_game_date"] == "2026-11-21"
    assert results.attrs["complete_through"] == "2026-11-22"
    assert not results.duplicated(["GAME_ID", "TEAM_ID"]).any()
    winners = nba_predictions.extract_completed_results(results)
    assert winners["0022600055"] == str(team("CHI")["id"])
    assert winners["0022600001"] == str(team("PHI")["id"])


def test_loader_off_day_is_fresh_without_any_request(monkeypatch, tmp_path):
    write_loader_files(tmp_path, "2026-11-21", loader_schedule())
    calls = []
    # The snapshot already has 11/21; 11/22 and 11/23 have no scheduled games.
    patch_loader(monkeypatch, tmp_path, date(2026, 11, 24), {}, calls)

    results = nba_predictions.load_current_season_team_games()

    assert calls == []
    assert results.attrs["latest_game_date"] == "2026-11-21"
    assert results.attrs["complete_through"] == "2026-11-24"
    assert results.attrs["is_stale"] is False


def test_loader_includes_todays_games_only_once_all_are_final(
    monkeypatch, tmp_path
):
    sched = scheduled(("0022600055", "2026-11-22", "CHA", "CHI"))
    sched["gameDateTimeUTC"] = "2026-11-22T23:00:00Z"
    write_loader_files(tmp_path, "2026-11-20", sched)
    patch_loader(monkeypatch, tmp_path, date(2026, 11, 22), {
        date(2026, 11, 22): board(event("CHA", "CHI", 50, 48, status="STATUS_IN_PROGRESS"))})

    live = nba_predictions.load_current_season_team_games()
    assert live.attrs["latest_game_date"] == "2026-11-20"
    assert live.attrs["complete_through"] == "2026-11-21"
    assert live.attrs["is_stale"] is False

    patch_loader(monkeypatch, tmp_path, date(2026, 11, 22), {
        date(2026, 11, 22): board(event("CHA", "CHI", 105, 107))})
    final = nba_predictions.load_current_season_team_games()
    assert final.attrs["latest_game_date"] == "2026-11-22"
    assert final.attrs["complete_through"] == "2026-11-22"
    # Same-day results never feed a same-day prediction: CHA and CHI have only
    # their 11/22 game on record, so they have no rating for 11/22.
    strengths, _ = nba_predictions.team_strengths_for_date(
        final, pd.DataFrame(), "2026-11-22")
    assert str(team("CHA")["id"]) not in strengths
    assert str(team("CHI")["id"]) not in strengths
    strengths_next, _ = nba_predictions.team_strengths_for_date(
        final, pd.DataFrame(), "2026-11-23")
    assert str(team("CHI")["id"]) in strengths_next


def test_loader_reports_stale_when_latest_day_not_final(monkeypatch, tmp_path):
    write_loader_files(tmp_path, "2026-11-20", loader_schedule())
    patch_loader(monkeypatch, tmp_path, date(2026, 11, 23), {
        date(2026, 11, 21): board(event("CHA", "CHI", 50, 48, status="STATUS_IN_PROGRESS"))})

    results = nba_predictions.load_current_season_team_games()

    assert results.attrs["is_stale"] is True
    assert results.attrs["complete_through"] == "2026-11-20"
    assert "confirmed only through Nov 20, 2026" in results.attrs["warning"]
    assert len(results) == 2


def test_loader_falls_back_to_nba_api_when_espn_is_down(monkeypatch, tmp_path):
    write_loader_files(tmp_path, "2026-11-20", loader_schedule())
    monkeypatch.setattr(nba_predictions, "ROOT", tmp_path)
    monkeypatch.setattr(nba_predictions, "_today_eastern", lambda: date(2026, 11, 22))
    monkeypatch.setattr(
        espn_results, "fetch_scoreboard",
        fetcher({date(2026, 11, 21): espn_results.EspnUnavailable("Timeout")}))
    seen = []

    def unavailable(**kwargs):
        seen.append(kwargs["timeout"])
        raise TimeoutError("NBA API unavailable")

    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", unavailable)

    results = nba_predictions.load_current_season_team_games()

    assert seen == [nba_predictions.NBA_API_FALLBACK_TIMEOUT_SECONDS]
    assert results.attrs["is_stale"] is True
    assert "TimeoutError" in results.attrs["warning"]
