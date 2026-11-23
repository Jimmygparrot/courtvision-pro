import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

EASTERN = ZoneInfo("America/New_York")
NO_GAMES_MESSAGE = "No NBA games scheduled today."
UNAVAILABLE_MESSAGE = (
    "Today's NBA schedule is temporarily unavailable. Please try again later."
)


def eastern_today():
    return datetime.now(EASTERN).date()


def format_tipoff(tipoff_utc):
    tipoff = pd.to_datetime(tipoff_utc, utc=True, errors="coerce")
    if pd.isna(tipoff):
        return "TBD"
    local = tipoff.tz_convert(EASTERN)
    return f"{local.hour % 12 or 12}:{local:%M} {local:%p} ET"


def fetch_schedule_rows(game_date):
    """Return (rows, error). Uses the project's existing schedule loader."""
    try:
        from nba_predictions import load_todays_matchups

        games, _source, _warning = load_todays_matchups(game_date)
    except Exception:
        return [], True
    if games is None or games.empty:
        # An empty frame with a warning means the lookup itself failed.
        return [], bool(_warning and "could not be loaded" in _warning)

    games = games.assign(
        _tipoff=pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce")
    ).sort_values("_tipoff", na_position="last")
    rows = [
        {
            "game_id": str(game.get("game_id", "")).zfill(10),
            "Away": str(game["away_team"]),
            "Home": str(game["home_team"]),
            "Tipoff": format_tipoff(game["tipoff_utc"]),
        }
        for _, game in games.iterrows()
    ]
    return rows, False


# Keyed by date so it rolls over at midnight ET; short TTL picks up changes.
@st.cache_data(ttl=900, show_spinner=False)
def cached_schedule_rows(date_key):
    rows, error = fetch_schedule_rows(datetime.strptime(date_key, "%Y-%m-%d").date())
    if error:
        # Raising prevents caching the failure so the next run retries.
        raise RuntimeError("schedule unavailable")
    return rows


def fetch_live_scoreboard(today):
    """Return {game_id: live game dict} from the NBA live scoreboard.

    Raises if the API fails or the board is not for `today` (ET), so stale
    games never appear and failures are never cached.
    """
    from nba_api.live.nba.endpoints.scoreboard import ScoreBoard

    board = ScoreBoard(timeout=10).get_dict()["scoreboard"]
    if str(board.get("gameDate")) != today.isoformat():
        raise ValueError("stale scoreboard")
    return {str(g["gameId"]).zfill(10): g for g in board.get("games", [])}


# Short TTL dedupes calls across reruns and sessions; key rolls with the ET date.
@st.cache_data(ttl=30, show_spinner=False)
def cached_live_scoreboard(date_key):
    games = fetch_live_scoreboard(datetime.strptime(date_key, "%Y-%m-%d").date())
    return {"games": games, "updated": datetime.now(EASTERN)}


def _clock(raw):
    match = re.fullmatch(r"PT(\d+)M(\d+)(?:\.\d+)?S", str(raw or ""))
    return f"{int(match.group(1))}:{match.group(2)}" if match else ""


def game_state(live):
    """Return 'scheduled', 'live', 'halftime' or 'final'."""
    status = live.get("gameStatus")
    if status == 3:
        return "final"
    if status == 2:
        if "half" in str(live.get("gameStatusText", "")).lower():
            return "halftime"
        return "live"
    return "scheduled"


def live_display(live, tipoff_text):
    """Return (score, status) strings for a game."""
    state = game_state(live)
    if state == "scheduled":
        return "", tipoff_text
    score = f"{live['awayTeam'].get('score', 0)}-{live['homeTeam'].get('score', 0)}"
    period = int(live.get("period") or 0)
    if state == "final":
        return score, "Final" if period <= 4 else "Final/OT"
    if state == "halftime":
        return score, "Halftime"
    label = f"Q{period}" if period <= 4 else f"OT{period - 4 if period > 5 else ''}"
    return score, f"{label} {_clock(live.get('gameClock'))}".strip()


def build_display_rows(rows, live_games):
    out = []
    for row in rows:
        live = live_games.get(row["game_id"])
        score, status = live_display(live, row["Tipoff"]) if live else ("", row["Tipoff"])
        out.append({
            "Away": row["Away"], "Home": row["Home"],
            "Score": score, "Status": status,
        })
    return out


def has_active_games(live_games):
    return any(game_state(g) in {"live", "halftime"} for g in live_games.values())


def _live_or_none():
    """Return (live data, warning). Falls back to the last good result."""
    key = eastern_today().isoformat()
    try:
        data = cached_live_scoreboard(key)
        st.session_state["cv_live_last_good"] = (key, data)
        return data, None
    except Exception:
        last = st.session_state.get("cv_live_last_good")
        if last and last[0] == key:
            return last[1], "Live scores are delayed; showing the last update."
        return None, "Live scores are temporarily unavailable."


def _render_table(rows):
    data, warning = _live_or_none()
    live_games = data["games"] if data else {}
    st.dataframe(
        build_display_rows(rows, live_games), hide_index=True, width="stretch"
    )
    if warning:
        st.caption(warning)
    if data:
        st.caption(
            f"Last updated {data['updated']:%I:%M:%S %p} ET · Tipoffs in Eastern Time"
        )
    else:
        st.caption("Tipoff times shown in Eastern Time (ET).")
    return has_active_games(live_games)


def render_todays_schedule():
    st.subheader("Today's NBA Schedule")
    try:
        rows = cached_schedule_rows(eastern_today().isoformat())
    except Exception:
        st.info(UNAVAILABLE_MESSAGE)
        return
    if not rows:
        st.write(NO_GAMES_MESSAGE)
        return

    # Only a fragment reruns each minute, so the prediction model is untouched.
    was_active = st.session_state.get("cv_live_active", False)

    def body():
        active = _render_table(rows)
        st.session_state["cv_live_active"] = active
        if active != was_active:
            st.rerun()

    st.fragment(run_every=60 if was_active else None)(body)()
