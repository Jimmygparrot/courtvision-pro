import logging

from nba_defense import get_defense_factors
import streamlit as st
import pandas as pd
from nba_stats import get_player_id, get_player_games
import plotly.express as px
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CourtVision Pro",
    page_icon="static/courtvision-pro-logo.png",
    layout="wide"
)

st.html(
    """
    <script>
    (() => {
        let manifestLink = document.querySelector('link[rel="manifest"]');
        if (!manifestLink) {
            manifestLink = document.createElement("link");
            manifestLink.rel = "manifest";
            document.head.appendChild(manifestLink);
        }
        manifestLink.href = "/app/static/courtvision-pro.webmanifest";
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)


# CourtVision Pro private login
import hashlib
import hmac

def courtvision_login():
    if st.session_state.get("cv_authenticated", False):
        if st.sidebar.button("Log out"):
            st.session_state["cv_authenticated"] = False
            st.rerun()
        return

    st.title("🏀 CourtVision Pro")
    st.subheader("Private Login")

    with st.form("courtvision_login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        salt = bytes.fromhex(st.secrets["CV_PASSWORD_SALT"])
        expected = bytes.fromhex(st.secrets["CV_PASSWORD_HASH"])
        iterations = int(st.secrets["CV_PASSWORD_ITERATIONS"])

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt,
            iterations
        )

        if hmac.compare_digest(actual, expected):
            st.session_state["cv_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()

courtvision_login()

# Load NBA player and team directory
nba_players = pd.read_csv("nba_players_teams.csv")

team_names = sorted(
    nba_players["team_name"].dropna().unique()
)

# Create team-filtered player directory


# Application heading
st.image(
    "static/courtvision-pro-logo.png",
    width=250,
    alt="CourtVision Pro basketball analytics logo",
)
st.title("🏀 CourtVision Pro")
st.subheader("NBA Player Prop Analytics")

st.write("Version 0.1 — Interactive Basketball Dashboard")

st.divider()

dashboard_view = st.segmented_control(
    "Analytics",
    ["Player props", "Game winners", "Prediction history"],
    default="Player props",
    key="cv_dashboard_view",
    width="stretch",
    wrap=True,
)

if dashboard_view in {"Game winners", "Prediction history"}:
    from nba_predictions import (
        EASTERN,
        MODEL_VERSION,
        build_pregame_predictions,
        extract_completed_results,
        format_eastern_datetime,
        load_current_season_team_games,
        load_previous_season_team_games,
        load_todays_matchups,
        prediction_accuracy,
        read_prediction_history,
        save_prediction_snapshots,
        update_prediction_results,
    )
    from backtesting.nba_daily_predictions import build_prediction_pdf

    today = datetime.now(EASTERN).date()

    @st.cache_data(ttl=900, max_entries=4, show_spinner=False)
    def cached_todays_matchups(game_date):
        return load_todays_matchups(game_date)

    @st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
    def cached_team_game_logs():
        try:
            results = load_current_season_team_games()
            return results, results.attrs.get("warning")
        except Exception as error:
            logger.exception(
                "Current-season team-results loader failed; returning no "
                "team records to the prediction model."
            )
            return (
                pd.DataFrame(),
                "Current-season results are unavailable; predictions use "
                "the prior-season baseline where available "
                f"({type(error).__name__}).",
            )

    if dashboard_view == "Game winners":
        st.subheader(f"NBA game winners — {today:%B %d, %Y}")
        st.caption(
            "Pregame estimates use only team results from dates before each "
            "game date. Same-day results and live standings are not used."
        )
        if st.button(
            "Refresh games and model inputs",
            key="cv_refresh_game_predictions",
            icon=":material/refresh:",
        ):
            cached_todays_matchups.clear()
            cached_team_game_logs.clear()
            st.rerun()
        games_today, schedule_source, schedule_warning = (
            cached_todays_matchups(str(today))
        )
        if schedule_warning:
            st.warning(schedule_warning)
        if (
            not games_today.empty
            and games_today.loc[
                games_today["status"] == "scheduled", "tipoff_utc"
            ].isna().any()
        ):
            st.warning(
                "Some scheduled matchups have no verifiable tipoff time; "
                "pregame predictions for those games are withheld."
            )
        current_results, results_warning = cached_team_game_logs()
        if current_results.empty:
            logger.warning(
                "Game Winners team-results cache returned no rows; "
                "loader_warning=%s",
                results_warning,
            )
        else:
            logger.info(
                "Game Winners team-results cache returned: "
                "rows=%d latest=%s stale=%s source=%s",
                len(current_results),
                current_results.attrs.get("latest_game_date"),
                current_results.attrs.get("is_stale"),
                current_results.attrs.get("source"),
            )
        if results_warning:
            st.warning(results_warning)
        previous_results = load_previous_season_team_games()

        fresh_predictions = build_pregame_predictions(
            games_today,
            current_results,
            previous_results,
        )
        save_prediction_snapshots(fresh_predictions)
        history = read_prediction_history()
        if not history.empty:
            saved_predictions = history[
                history["game_date"] == str(today)
            ].copy()
        else:
            saved_predictions = pd.DataFrame()

        st.subheader("Current Prediction")
        latest_game_date = current_results.attrs.get("latest_game_date")
        if latest_game_date:
            latest_game = pd.Timestamp(latest_game_date).date()
            latest_game_label = pd.Timestamp(latest_game).strftime(
                "%b %d, %Y"
            ).replace(" 0", " ")
            data_age = (today - latest_game).days
            if current_results.attrs.get("is_stale") is True:
                freshness = "stale saved snapshot"
            elif data_age > 1:
                freshness = "stale source data"
            elif data_age < 0:
                freshness = "freshness unknown"
            elif current_results.attrs.get("is_stale") is False:
                freshness = "fresh source data"
            else:
                freshness = "freshness unknown"
            source = current_results.attrs.get("source")
            source_label = f" ({source})" if source else ""
            st.caption(
                f"Team statistics used: through {latest_game_label} — "
                f"{freshness}{source_label}."
            )
        elif current_results.empty:
            st.caption(
                "Team statistics are unavailable; prior-season or neutral "
                "ratings may be used."
            )
        else:
            st.caption(
                "The latest team-statistics date and freshness are "
                "unavailable."
            )

        def prediction_display_table(
            predictions,
            label,
            timestamp_label,
        ):
            table = predictions[[
                "game_date",
                "away_team",
                "home_team",
                "predicted_winner",
                "home_win_probability",
                "away_win_probability",
                "created_at",
                "tipoff_utc",
            ]].copy()
            table["created_at"] = table["created_at"].map(
                format_eastern_datetime
            )
            table["tipoff_utc"] = table["tipoff_utc"].map(
                format_eastern_datetime
            )
            table["home_win_probability"] = table[
                "home_win_probability"
            ].map(lambda value: f"{value:.1%}")
            table["away_win_probability"] = table[
                "away_win_probability"
            ].map(lambda value: f"{value:.1%}")
            return table.rename(columns={
                "game_date": "Game date",
                "away_team": "Away",
                "home_team": "Home",
                "predicted_winner": f"{label} predicted winner",
                "home_win_probability": (
                    f"{label} home win probability"
                ),
                "away_win_probability": (
                    f"{label} away win probability"
                ),
                "created_at": timestamp_label,
                "tipoff_utc": "Game Tipoff (ET)",
            })

        if fresh_predictions.empty:
            if saved_predictions.empty:
                st.info(
                    "No current or original pregame predictions are "
                    "available. Predictions are not reconstructed after "
                    "tipoff."
                )
            else:
                st.info(
                    "No current pregame predictions are available. "
                    "Predictions are not reconstructed after tipoff."
                )
        else:
            st.caption(
                f"Model: {MODEL_VERSION} · Schedule: {schedule_source}. "
                "These probabilities are calculated from the latest "
                "available team results."
            )
            table = prediction_display_table(
                fresh_predictions,
                "Current",
                "Calculated at (ET)",
            )
            st.dataframe(
                table,
                hide_index=True,
                alt="Current NBA matchup win probabilities",
            )

        if saved_predictions.empty:
            if not fresh_predictions.empty:
                st.caption("No original saved prediction exists for these games.")
        else:
            st.subheader("Original Saved Prediction")
            st.caption(
                "These are immutable pregame snapshots from SQLite; "
                "their probabilities are not replaced by current calculations."
            )
            table = prediction_display_table(
                saved_predictions,
                "Original saved",
                "Original saved at (ET)",
            )
            st.dataframe(
                table,
                hide_index=True,
                alt="Original saved pregame NBA predictions",
            )
            if saved_predictions["rating_source"].fillna("").str.contains(
                "previous-season|neutral", regex=True
            ).any():
                st.caption(
                    "The original saved prediction used prior-season or "
                    "neutral team ratings."
                )
            outdated_sources = sorted(
                source
                for source in saved_predictions["rating_source"]
                .fillna("")
                .astype(str)
                .unique()
                if "outdated current-season snapshot" in source
            )
            if outdated_sources:
                st.warning(
                    "These saved predictions use outdated team results: "
                    + "; ".join(outdated_sources)
                    + ". The original saved probabilities are unchanged."
                )
            try:
                report = build_prediction_pdf(
                    saved_predictions,
                    today,
                )
                st.download_button(
                    "Download PDF report",
                    data=report,
                    file_name=f"CourtVision_Pro_{today}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            except Exception as error:
                st.error(
                    "The PDF report could not be generated "
                    f"({type(error).__name__})."
                )
        st.stop()

    st.subheader("Prediction history")
    if st.button(
        "Refresh completed-game results",
        key="cv_refresh_prediction_history",
        icon=":material/refresh:",
    ):
        cached_team_game_logs.clear()
        st.rerun()
    current_results, results_warning = cached_team_game_logs()
    if results_warning:
        st.warning(results_warning)
    actual_results = extract_completed_results(current_results)
    update_prediction_results(actual_results)
    history = read_prediction_history()
    if history.empty:
        st.info(
            "No pregame predictions have been saved yet. Future games are "
            "recorded only before tipoff."
        )
    else:
        completed_count, accuracy = prediction_accuracy(history)
        with st.container(horizontal=True):
            st.metric("Predictions", len(history), border=True)
            st.metric(
                "Completed games",
                completed_count,
                border=True,
            )
            st.metric(
                "Winner accuracy",
                f"{accuracy:.1%}" if accuracy is not None else "—",
                border=True,
            )
        results = history.copy()
        results["Actual winner"] = results.apply(
            lambda row: (
                row["home_team"]
                if pd.notna(row["actual_winner"])
                and str(row["actual_winner"]) == str(row["home_team_id"])
                else row["away_team"]
                if pd.notna(row["actual_winner"])
                else "Pending"
            ),
            axis=1,
        )
        results["Correct"] = results.apply(
            lambda row: (
                "Yes" if row["predicted_winner"] == row["Actual winner"]
                else "No"
            ) if row["Actual winner"] != "Pending" else "Pending",
            axis=1,
        )
        results = results.rename(columns={
            "game_date": "Game date",
            "away_team": "Away",
            "home_team": "Home",
            "predicted_winner": "Predicted winner",
            "model_version": "Model version",
        })
        results["Home win probability"] = results[
            "home_win_probability"
        ].map(lambda value: f"{value:.1%}")
        results["Away win probability"] = results[
            "away_win_probability"
        ].map(lambda value: f"{value:.1%}")
        results["Prediction timestamp (ET)"] = results[
            "created_at"
        ].map(format_eastern_datetime)
        results["Game Tipoff (ET)"] = results[
            "tipoff_utc"
        ].map(format_eastern_datetime)
        st.dataframe(
            results[[
                "Game date",
                "Away",
                "Home",
                "Predicted winner",
                "Actual winner",
                "Correct",
                "Home win probability",
                "Away win probability",
                "Prediction timestamp (ET)",
                "Game Tipoff (ET)",
                "Model version",
            ]],
            hide_index=True,
            alt="Saved pregame predictions and completed game results",
        )
    st.caption(
        "Actual results update saved records only; they never alter the "
        "original prediction or its pregame inputs."
    )
    st.stop()

# Sidebar controls
st.sidebar.header("Player Analysis")

def load_recent_20_games(player_id, force_refresh=False):
    frames = []
    refresh_errors = []

    for season in ["2026-27", "2025-26"]:
        if frames:
            from nba_stats import read_cached_games
            if read_cached_games(player_id, season) is None:
                break
        try:
            games = get_player_games(
                player_id,
                season=season,
                force_refresh=force_refresh and season == "2026-27",
            )
        except RuntimeError as error:
            refresh_errors.append(str(error))
            if season == "2026-27":
                continue
            if not frames:
                raise
            break

        if games is None:
            refresh_errors.append(
                "NBA statistics could not be downloaded. Please retry."
            )
            if season == "2026-27":
                continue
            if not frames:
                raise RuntimeError(
                    "NBA statistics could not be downloaded. Please retry."
                )
            break
        if games.attrs.get("refresh_error"):
            refresh_errors.append(games.attrs["refresh_error"])
        if games is not None and not games.empty:
            frames.append(games)

        if frames:
            combined_count = pd.concat(frames, ignore_index=True)
            if "Game_ID" in combined_count.columns:
                combined_count = combined_count.drop_duplicates(
                    subset=["Game_ID"]
                )
            if len(combined_count) >= 20:
                break

    if not frames:
        if refresh_errors:
            raise RuntimeError(" ".join(dict.fromkeys(refresh_errors)))
        return None

    combined = pd.concat(frames, ignore_index=True)

    if "GAME_DATE" in combined.columns:
        combined["GAME_DATE"] = pd.to_datetime(
            combined["GAME_DATE"], errors="coerce"
        )
        combined = combined.sort_values(
            "GAME_DATE",
            ascending=False,
            na_position="last",
            kind="stable",
        )

    if "Game_ID" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["Game_ID"],
            keep="first",
        )

    combined = combined.head(20).reset_index(drop=True)
    combined.attrs["refresh_error"] = (
        " ".join(dict.fromkeys(refresh_errors)) if refresh_errors else None
    )
    return combined


selected_team = st.sidebar.selectbox(
    "Select NBA Team",
    ["All Teams"] + team_names,
    key="cv_team_select"
)

if selected_team == "All Teams":
    filtered_players = nba_players
else:
    filtered_players = nba_players[
        nba_players["team_name"] == selected_team
    ]

player_names = sorted(
    filtered_players["full_name"].dropna().unique()
)

player = st.sidebar.selectbox(
    "Select Player",
    player_names,
    key="cv_player_select"
)

stat = st.sidebar.selectbox(
    "Select Statistic",
    [
        "Points",
        "Rebounds",
        "Assists",
        "PRA"
    ]
)

# Requests are explicit so a slow statistics endpoint cannot block page startup.
player_id = get_player_id(player)
selection_key = (player, stat)
if st.session_state.get("cv_analysis_selection") != selection_key:
    st.session_state["cv_live_results"] = []
    st.session_state["cv_selected_event"] = None
    st.session_state["cv_analysis_selection"] = selection_key

if st.sidebar.button("Load Player Statistics", key="cv_load_statistics"):
    if player_id is None:
        st.sidebar.warning("Player ID not found.")
    else:
        checked_at = datetime.now(timezone.utc)
        try:
            with st.spinner(f"Loading recent NBA games for {player}..."):
                loaded_games = load_recent_20_games(
                    player_id,
                    force_refresh=False,
                )
            if loaded_games is not None:
                st.session_state[f"cv_games_{player_id}"] = loaded_games
            st.session_state[f"cv_games_refresh_error_{player_id}"] = (
                loaded_games.attrs.get("refresh_error")
                if loaded_games is not None else None
            )
            if loaded_games is None or loaded_games.empty:
                st.sidebar.warning("No recent games found for this player.")
        except RuntimeError as error:
            st.session_state[f"cv_games_refresh_error_{player_id}"] = str(error)
        finally:
            st.session_state[f"cv_games_checked_{player_id}"] = (
                checked_at.isoformat()
            )

@st.fragment(run_every="15m")
def refresh_loaded_player_stats(selected_player_id):
    games_key = f"cv_games_{selected_player_id}"
    previous_games = st.session_state.get(games_key)
    if previous_games is None:
        return

    checked_value = st.session_state.get(
        f"cv_games_checked_{selected_player_id}"
    )
    checked_at = (
        datetime.fromisoformat(checked_value)
        if checked_value else datetime.min.replace(tzinfo=timezone.utc)
    )
    if datetime.now(timezone.utc) - checked_at < timedelta(minutes=15):
        return

    previous_error = st.session_state.get(
        f"cv_games_refresh_error_{selected_player_id}"
    )
    try:
        refreshed_games = load_recent_20_games(selected_player_id)
        if refreshed_games is not None:
            st.session_state[games_key] = refreshed_games
            current_error = refreshed_games.attrs.get("refresh_error")
            st.session_state[
                f"cv_games_refresh_error_{selected_player_id}"
            ] = current_error
        else:
            current_error = previous_error
    except RuntimeError as error:
        current_error = str(error)
        st.session_state[
            f"cv_games_refresh_error_{selected_player_id}"
        ] = current_error
    finally:
        st.session_state[f"cv_games_checked_{selected_player_id}"] = (
            datetime.now(timezone.utc).isoformat()
        )

    current_games = st.session_state.get(games_key)
    if (
        previous_error != current_error
        or current_games is None
        or not previous_games.equals(current_games)
    ):
        st.rerun()


if player_id is not None:
    refresh_loaded_player_stats(player_id)

# Reuse the latest successful baseline between refresh checks.
games = st.session_state.get(f"cv_games_{player_id}")
if games is not None:
    games = games.copy()
statistics_available = games is not None and not games.empty
refresh_error = (
    st.session_state.get(f"cv_games_refresh_error_{player_id}")
    if player_id is not None else None
)
if refresh_error:
    st.sidebar.warning(refresh_error)
if statistics_available:
    latest_game = pd.to_datetime(
        games["GAME_DATE"], errors="coerce"
    ).max()

    if pd.notna(latest_game):
        st.sidebar.success(f"Latest available game: {latest_game:%b %d, %Y}")
        st.sidebar.caption(
            f"Games in projection sample: {min(len(games), 20)}"
        )
estimated_minutes = 35
if statistics_available:
    recent_minutes = pd.to_numeric(games["MIN"].head(10), errors="coerce").dropna()
    if not recent_minutes.empty:
        estimated_minutes = round(recent_minutes.mean())
else:
    st.info("Select a player, then choose Load Player Statistics in the sidebar. "
            "Live odds and manual odds can be used independently below.")

estimated_minutes = max(15, min(48, estimated_minutes))

minutes = st.sidebar.slider(
    "Projected Minutes",
    min_value=15,
    max_value=48,
    value=estimated_minutes,
    key=f"minutes_{player}_{estimated_minutes}"
)

# Only real downloaded statistics are used for projections.
points, rebounds, assists = 0.0, 0.0, 0.0
if statistics_available:
    points = games["PTS"].head(20).mean()
    rebounds = games["REB"].head(20).mean()
    assists = games["AST"].head(20).mean()

# Adjust statistics using actual historical playing time
historical_minutes = 35.0

if player_id is not None and games is not None and not games.empty:
    recent_games = games.head(20)

    valid_minutes = pd.to_numeric(
        recent_games["MIN"],
        errors="coerce"
    )

    valid_minutes = valid_minutes[
        valid_minutes > 0
    ]

    if not valid_minutes.empty:
        historical_minutes = float(valid_minutes.mean())

factor = minutes / historical_minutes

projected_points = points * factor
projected_rebounds = rebounds * factor
projected_assists = assists * factor

projected_pra = (
    projected_points
    + projected_rebounds
    + projected_assists
)


# Upcoming matchup from the locally saved NBA schedule
@st.cache_data(ttl=900, max_entries=1)
def load_nba_schedule():
    return pd.read_csv("data/nba_schedule_2026-27.csv")

st.subheader("🏀 Upcoming Matchup")

schedule = load_nba_schedule()
player_rows = nba_players[nba_players["full_name"] == player]
next_game = None

if not player_rows.empty:
    player_team = player_rows.iloc[0]["team_name"]

    schedule = schedule.copy()
    schedule["gameDateTimeUTC"] = pd.to_datetime(
        schedule["gameDateTimeUTC"], utc=True
    )

    schedule["home_team"] = (
        schedule["homeTeam_teamCity"] + " " +
        schedule["homeTeam_teamName"]
    )
    schedule["away_team"] = (
        schedule["awayTeam_teamCity"] + " " +
        schedule["awayTeam_teamName"]
    )

    upcoming = schedule[
        (
            (schedule["home_team"] == player_team) |
            (schedule["away_team"] == player_team)
        ) &
        (schedule["gameDateTimeUTC"] > pd.Timestamp.now(tz="UTC"))
    ].sort_values("gameDateTimeUTC")

    if not upcoming.empty:
        game = upcoming.iloc[0]
        is_home = game["home_team"] == player_team

        next_game = {
            "opponent": game["away_team"] if is_home else game["home_team"],
            "date": game["gameDateTimeUTC"].tz_convert(
                "America/New_York"
            ).strftime("%B %d, %Y at %I:%M %p ET"),
            "location": "Home" if is_home else "Away"
        }

        st.write(f"**{player} — vs. {next_game['opponent']}**")
        st.write(
            f"📅 {next_game['date']} | Location: {next_game['location']}"
        )

if next_game is None:
    st.info("No upcoming NBA matchup found.")

# Experimental opponent-adjusted projections
st.subheader("Opponent-Adjusted Projections")

if next_game and statistics_available and st.checkbox("Load opponent adjustment", value=False):
    try:
        opponent = next_game["opponent"]
        factors = get_defense_factors(opponent)

        if factors:
            adj_points = projected_points * factors["PTS"]
            adj_rebounds = projected_rebounds * factors["REB"]
            adj_assists = projected_assists * factors["AST"]
            adj_pra = adj_points + adj_rebounds + adj_assists

            st.write(f"**Upcoming opponent: {opponent}**")

            import pandas as pd

            comparison = pd.DataFrame({
                "Statistic": ["Points", "Rebounds", "Assists", "PRA"],
                "Original": [
                    projected_points,
                    projected_rebounds,
                    projected_assists,
                    projected_pra
                ],
                "Opponent Adjusted": [
                    adj_points,
                    adj_rebounds,
                    adj_assists,
                    adj_pra
                ]
            })

            st.dataframe(
                comparison.style.format({
                    "Original": "{:.1f}",
                    "Opponent Adjusted": "{:.1f}"
                }),
                hide_index=True,
                alt="Original and opponent-adjusted player projections",
            )

            st.caption(
                "Experimental adjustment using opponent team averages. "
                "Not validated for individual player predictions."
            )
        else:
            st.info("Opponent defensive statistics unavailable.")

    except Exception as error:
        st.error(f"Opponent adjustment error: {type(error).__name__}: {error}")
else:
    st.info("Load player statistics and live odds to enable the optional opponent adjustment.")

st.caption("The original projections below are not opponent-adjusted.")

# Display metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Points", f"{projected_points:.1f}" if statistics_available else "—")
col2.metric("Rebounds", f"{projected_rebounds:.1f}" if statistics_available else "—")
col3.metric("Assists", f"{projected_assists:.1f}" if statistics_available else "—")
col4.metric("PRA", f"{projected_pra:.1f}" if statistics_available else "—")

st.divider()

# Selected statistic
projections = {
    "Points": projected_points,
    "Rebounds": projected_rebounds,
    "Assists": projected_assists,
    "PRA": projected_pra
}

st.subheader(f"{player} — {stat} Projection")

st.metric(
    "Projected Value",
    f"{projections[stat]:.1f}" if statistics_available else "—"
)

st.caption("Projections are experimental and have not been backtested.")

# ==========================================
# COURTVISION PRO - REAL NBA STATISTICS
# ==========================================

st.divider()
st.header("Real NBA Player Statistics")

if not statistics_available:
    st.info("Use Load Player Statistics in the sidebar to populate game logs and projections.")
else:
    # The same recent games support the charts, projections, and EV calculators.
    st.success(f"Statistics loaded for {player}!")

    columns = [
        "GAME_DATE",
        "MATCHUP",
        "MIN",
        "PTS",
        "REB",
        "AST"
    ]

    st.dataframe(
        games[columns].head(20),
        hide_index=True,
        alt="Most recent player game log",
    )

    # Calculate actual NBA averages
    st.subheader("NBA Performance Averages")

    # Ensure newest games appear first
    games = games.sort_values(
        "GAME_DATE",
        ascending=False
    )

    # Calculate Points + Rebounds + Assists
    games["PRA"] = (
        games["PTS"]
        + games["REB"]
        + games["AST"]
    )

    stats_columns = ["PTS", "REB", "AST", "PRA"]

    averages = []

    for period in [5, 10, 20]:
        recent = games.head(period)

        row = {
            "Period": f"Last {period} Games ({len(recent)} available)"
        }

        for stat_name in stats_columns:
            row[stat_name] = round(
                recent[stat_name].mean(), 1
            )

        averages.append(row)

    averages_df = pd.DataFrame(averages)

    st.dataframe(
        averages_df,
        hide_index=True,
        alt="Player points, rebounds, assists, and PRA averages",
    )

    # Interactive NBA performance chart
    st.subheader("NBA Game-by-Game Performance")
    
    chart_stats = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST",
        "PRA": "PRA"
    }
    
    selected_column = chart_stats[stat]
    
    chart_data = games.copy()
    
    chart_data["GAME_DATE"] = pd.to_datetime(
        chart_data["GAME_DATE"]
    )
    
    chart_data = chart_data.sort_values("GAME_DATE")
    
    fig = px.line(
        chart_data,
        x="GAME_DATE",
        y=selected_column,
        markers=True,
        title=f"{player} — {stat} Performance"
    )
    
    fig.add_hline(
        y=chart_data[selected_column].mean(),
        line_dash="dash",
        annotation_text="Average"
    )
    
    fig.update_layout(
        xaxis_title="Game Date",
        yaxis_title=stat
    )
    
    st.plotly_chart(
        fig,
        width="stretch",
        config={"staticPlot": True, "responsive": True},
        alt=f"{player} {stat} game-by-game trend",
    )
    
    # Player Prop Analyzer
    st.divider()
    st.subheader("🏀 Player Prop Analyzer")
    
    prop_line = st.number_input(
        f"Enter {player}'s {stat} betting line",
        min_value=0.5,
        max_value=100.5,
        value=30.5,
        step=0.5,
        key="player_prop_line"
    )
    
    st.write(f"Selected Player: {player}")
    st.write(f"Statistic: {stat}")
    st.write(f"Betting Line: {prop_line}")
    
    # Calculate historical over/under results
    prop_results = games[selected_column].dropna()
    
    total_games = len(prop_results)
    
    over_games = (prop_results > prop_line).sum()
    under_games = (prop_results < prop_line).sum()
    push_games = (prop_results == prop_line).sum()
    
    if total_games > 0:
        over_pct = (over_games / total_games) * 100
        under_pct = (under_games / total_games) * 100
    
        st.subheader("Historical Over/Under Results")
    
        col1, col2, col3 = st.columns(3)
    
        col1.metric("Over", f"{over_pct:.1f}%")
        col2.metric("Under", f"{under_pct:.1f}%")
        col3.metric("Games Analyzed", total_games)
    
        st.write(
            f"Over: {over_games} | "
            f"Under: {under_games} | "
            f"Push: {push_games}"
        )
    
        st.caption(
            "Historical results only. These percentages "
            "are not predictions of future outcomes."
        )
    
        # Historical over/under performance chart
    st.subheader("Game-by-Game Over/Under Chart")
    
    chart_games = games.copy()
    chart_games["GAME_DATE"] = pd.to_datetime(
        chart_games["GAME_DATE"]
    )
    chart_games = chart_games.sort_values("GAME_DATE")
    
    chart_games["Result"] = chart_games[selected_column].apply(
        lambda value: "Over" if value > prop_line
        else "Under" if value < prop_line
        else "Push"
    )
    
    fig_ou = px.bar(
        chart_games,
        x="GAME_DATE",
        y=selected_column,
        color="Result",
        color_discrete_map={
            "Over": "green",
            "Under": "red",
            "Push": "gold"
        },
        title=f"{player} - {stat} vs Betting Line"
    )
    
    fig_ou.add_hline(
        y=prop_line,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Betting Line: {prop_line}"
    )
    
    fig_ou.update_layout(
        xaxis_title="Game Date",
        yaxis_title=stat,
        showlegend=False
    )
    
    st.plotly_chart(
        fig_ou,
        width="stretch",
        config={"staticPlot": True, "responsive": True},
        alt=f"{player} {stat} results compared with the betting line",
    )


# LIVE SPORTSBOOK COMPARISON
st.divider()
st.subheader("🏦 Live Sportsbook Comparison")

from sportsbook_config import SPORTSBOOKS
from sportsbook_ev import calculate_ev
from live_odds import (
    get_nba_events,
    get_player_props,
    compare_player,
    best_prices
)

market_options = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "PRA": "player_points_rebounds_assists"
}

selected_market = market_options[stat]


from live_odds import find_team_event

# Identify the selected player's team
player_rows = nba_players[
    nba_players["full_name"] == player
]

player_team = (
    player_rows.iloc[0]["team_name"]
    if not player_rows.empty
    else None
)

@st.cache_data(ttl=900, show_spinner=False)
def load_upcoming_nba_events():
    return get_nba_events()

@st.cache_data(ttl=60, show_spinner=False)
def load_cached_props(event_id, market):
    return get_player_props(event_id, market)

st.caption("Odds load on request. Refresh retrieves current player prices; "
           "changing a calculator input does not make another odds request.")
if player_team:
    st.info(f"Selected player: {player} | Team: {player_team}")
    if st.button("Load / Refresh Live Odds", key="cv_refresh_odds"):
        st.session_state["cv_live_results"] = []
        st.session_state["cv_selected_event"] = None
        try:
            with st.spinner("Finding the next sportsbook matchup..."):
                # Refresh this lookup so an event that has just started is not reused.
                load_upcoming_nba_events.clear()
                events = load_upcoming_nba_events()
                selected_event = find_team_event(player_team, events)
            if selected_event:
                st.session_state["cv_selected_event"] = selected_event
                event_id = selected_event["id"]
                with st.spinner("Retrieving player props..."):
                    load_cached_props.clear(event_id, selected_market)
                    data = load_cached_props(event_id, selected_market)
                st.session_state["cv_live_results"] = compare_player(player, data)
                st.rerun()
            else:
                st.warning(f"No upcoming sportsbook matchup available for {player_team}.")
        except Exception as error:
            # Never display a request URL: it can contain the API key.
            st.error("Live odds could not be loaded. "
                     "Check the Odds API key and quota, then try Refresh again.")
else:
    st.warning("Unable to identify the selected player's NBA team.")

if "cv_live_results" in st.session_state:
    results = st.session_state["cv_live_results"]

    if results:
        st.success(f"Found {len(results)} available player markets.")

        comparisons = best_prices(results)

        rows = []

        for line, prices in sorted(comparisons.items()):
            over = prices["Over"]
            under = prices["Under"]

            rows.append({
                "Line": line,
                "Best Over Book": over["Sportsbook"],
                "Over Odds": over["Over_Odds"],
                "Best Under Book": under["Sportsbook"],
                "Under Odds": under["Under_Odds"],
                "No-Vig Over %": over["No_Vig_Over"]
            })

        st.dataframe(
            rows,
            alt="Best available sportsbook prices for the selected player prop",
        )

    else:
        st.warning(
            f"No current {stat} markets found for {player}. "
            "Live sportsbook odds are unavailable for this selection. Your CourtVision projections remain available, and you can enter a sportsbook line manually in the EV calculator below."
        )



# MANUAL SPORTSBOOK ODDS ENTRY
st.divider()
st.subheader("✍️ Manual Sportsbook Odds Entry")

if "cv_manual_odds" not in st.session_state:
    st.session_state["cv_manual_odds"] = []

manual_book = st.selectbox(
    "Sportsbook",
    ["ProphetX", "DraftKings", "Hard Rock Bet"],
    key="cv_manual_book"
)

with st.form("manual_odds_form"):
    st.caption(f"Player: {player} | Statistic: {stat}")

    manual_line = st.number_input(
        "Betting Line",
        min_value=0.0,
        value=25.5,
        step=0.5
    )


    if manual_book == "ProphetX":
        st.caption("Enter ProphetX prices as percentages (0–100).")

        manual_over = st.number_input(
            "Over Price (%)",
            min_value=0.01,
            max_value=99.99,
            value=47.85,
            step=0.01
        )

        manual_under = st.number_input(
            "Under Price (%)",
            min_value=0.01,
            max_value=99.99,
            value=52.15,
            step=0.01
        )

    else:
        manual_over = st.number_input(
            "Over American Odds",
            value=-110,
            step=5
        )

        manual_under = st.number_input(
            "Under American Odds",
            value=-110,
            step=5
        )

    save_manual = st.form_submit_button("Save Odds")

    if save_manual:
        from prophetx_utils import percent_to_american

        if manual_book == "ProphetX":
            over_odds = percent_to_american(manual_over)
            under_odds = percent_to_american(manual_under)
            valid = True
        else:
            over_odds = manual_over
            under_odds = manual_under

            valid = (
                (over_odds >= 100 or over_odds <= -100)
                and (under_odds >= 100 or under_odds <= -100)
            )

        if not valid:
            st.error("Enter valid American odds.")
        else:
            entry = {
                "Sportsbook": manual_book,
                "Player": player,
                "Statistic": selected_market,
                "Line": manual_line,
                "Over_Odds": over_odds,
                "Under_Odds": under_odds,
                "Source": "Manual"
            }

            if manual_book == "ProphetX":
                entry["Over_Price_Percent"] = manual_over
                entry["Under_Price_Percent"] = manual_under

            st.session_state["cv_manual_odds"] = [
                old for old in st.session_state["cv_manual_odds"]
                if not (
                    old["Sportsbook"] == manual_book
                    and old["Player"] == player
                    and old["Statistic"] == selected_market
                    and old["Line"] == manual_line
                )
            ] + [entry]

            st.success("Manual odds saved!")

# THREE-SPORTSBOOK COMPARISON
st.divider()
st.subheader("🏀 Sportsbook Comparison")

from scipy.stats import norm

sportsbook_rows = []
live_results = (
    st.session_state.get("cv_live_results", [])
    + st.session_state.get("cv_manual_odds", [])
)

stat_column = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "PRA": "PRA"
}.get(stat)

comparison_sd = None
comparison_sample_size = 0
comparison_projection = float(projections.get(stat, 0.0))

if games is not None and stat_column is not None:
    comparison_games = games.copy()

    if stat_column == "PRA":
        if all(c in comparison_games.columns for c in ["PTS", "REB", "AST"]):
            comparison_games["PRA"] = (
                comparison_games["PTS"]
                + comparison_games["REB"]
                + comparison_games["AST"]
            )

    if stat_column in comparison_games.columns:
        comparison_values = pd.to_numeric(
            comparison_games[stat_column], errors="coerce"
        ).dropna()

        if len(comparison_values) >= 5 and comparison_values.std() > 0:
            comparison_sd = float(comparison_values.std())
            comparison_sample_size = len(comparison_values)

if 5 <= comparison_sample_size < 10:
    st.warning(
        f"Preliminary EV estimate: only {comparison_sample_size} games "
        "available. Small samples can produce unreliable probabilities."
    )

for book_key, book_name in SPORTSBOOKS.items():
    matches = [
        item for item in live_results
        if item["Sportsbook"].lower() == book_name.lower()
        and item["Player"] == player
        and item["Statistic"] == selected_market
    ]

    if matches:
        for item in matches:
            over_ev = "Insufficient data"
            under_ev = "Insufficient data"

            if comparison_sd is not None:
                over_result = calculate_ev(
                    comparison_projection,
                    float(item["Line"]),
                    comparison_sd,
                    float(item["Over_Odds"]),
                    "Over"
                )

                under_result = calculate_ev(
                    comparison_projection,
                    float(item["Line"]),
                    comparison_sd,
                    float(item["Under_Odds"]),
                    "Under"
                )

                over_ev = f'{over_result["EV"]:+.2f}%'
                under_ev = f'{under_result["EV"]:+.2f}%'

            sportsbook_rows.append({
                "Sportsbook": book_name,
                "Betting Line": item["Line"],
                "Over Odds": item["Over_Odds"],
                "Under Odds": item["Under_Odds"],
                "Over EV": over_ev,
                "Under EV": under_ev,
                "Data Connection": item.get("Source", "Live odds")
            })
    else:
        sportsbook_rows.append({
            "Sportsbook": book_name,
            "Betting Line": None,
            "Over Odds": None,
            "Under Odds": None,
            "Over EV": "Not available",
            "Under EV": "Not available",
            "Data Connection": (
                "No live odds loaded for this selection"
                if book_key == "draftkings"
                else "Not connected"
            )
        })

st.dataframe(
    pd.DataFrame(sportsbook_rows),
    hide_index=True,
    alt="Sportsbook prices and expected value for the selected player prop",
)

st.caption(
    "EV estimates use recent player statistics and a normal-distribution "
    "model. Probabilities are not yet backtested or calibrated. "
    "Sportsbook fees and betting limits are excluded."
)

# AUTOMATIC SPORTSBOOK EV CALCULATOR
st.divider()
st.subheader("🎯 Automatic Sportsbook EV")

if "cv_live_results" in st.session_state:
    live_results = st.session_state["cv_live_results"]

    player_markets = [
        item for item in live_results
        if item["Player"] == player
        and item["Statistic"] == selected_market
    ]

    if player_markets:
        from scipy.stats import norm

        auto_player_id = get_player_id(player)
        auto_games = (
            st.session_state.get(f"cv_games_{auto_player_id}")
            if auto_player_id is not None else None
        )

        stat_column = {
            "Points": "PTS",
            "Rebounds": "REB",
            "Assists": "AST",
            "PRA": "PRA"
        }[stat]

        if auto_games is not None:
            auto_games = auto_games.copy()

            if stat_column == "PRA":
                if all(
                    c in auto_games.columns
                    for c in ["PTS", "REB", "AST"]
                ):
                    auto_games["PRA"] = (
                        auto_games["PTS"]
                        + auto_games["REB"]
                        + auto_games["AST"]
                    )

        if auto_games is not None and stat_column in auto_games.columns:
            values = pd.to_numeric(
                auto_games[stat_column], errors="coerce"
            ).dropna()
        else:
            values = pd.Series(dtype=float)

        if len(values) >= 10 and values.std() > 0:
            sd = float(values.std())
            projection = float(projections.get(stat, 0.0))

            best = best_prices(player_markets)
            ev_rows = []

            for line, prices in sorted(best.items()):
                over = prices["Over"]
                under = prices["Under"]

                probability_over = float(
                    norm.sf(line, loc=projection, scale=sd)
                )
                probability_under = 1 - probability_over

                def calculate_legacy_ev(probability, american_odds):
                    if american_odds > 0:
                        profit_per_dollar = american_odds / 100
                    else:
                        profit_per_dollar = 100 / abs(american_odds)

                    return (
                        probability * profit_per_dollar
                        - (1 - probability)
                    ) * 100

                over_ev = calculate_legacy_ev(
                    probability_over, over["Over_Odds"]
                )
                under_ev = calculate_legacy_ev(
                    probability_under, under["Under_Odds"]
                )

                ev_rows.append({
                    "Line": line,
                    "Projection": round(projection, 1),
                    "Over Probability": round(probability_over * 100, 1),
                    "Best Over Book": over["Sportsbook"],
                    "Over Odds": over["Over_Odds"],
                    "Over EV %": round(over_ev, 2),
                    "Best Under Book": under["Sportsbook"],
                    "Under Odds": under["Under_Odds"],
                    "Under EV %": round(under_ev, 2)
                })

            ev_table = pd.DataFrame(ev_rows)

            st.dataframe(
                ev_table.style.format({
                    "Line": "{:.1f}",
                    "Projection": "{:.1f}",
                    "Over Probability": "{:.1f}%",
                    "Over EV %": "{:+.2f}%",
                    "Under EV %": "{:+.2f}%"
                }).map(
                    lambda x: (
                        "color: green" if x > 0
                        else "color: red"
                    ),
                    subset=["Over EV %", "Under EV %"]
                ),
                alt="Automatic expected-value estimates for available prop lines",
            )

            st.caption(
                "EV uses available sportsbook prices and a "
                "preliminary normal-distribution model. "
                "Probabilities are not yet backtested or calibrated. "
                "Fees and betting limits are excluded."
            )
        else:
            st.warning(
                "At least 10 valid recent games are needed "
                "for automatic EV estimates."
            )
    else:
        st.info(
            "Load live markets for the selected player "
            "and statistic to calculate EV."
        )
else:
    st.info("Load live NBA odds to activate automatic EV.")

# CourtVision Pro - Probability and EV Calculator
st.divider()
st.subheader("📊 Probability & EV Calculator")

from scipy.stats import norm

cv_projection = st.number_input(
    f"CourtVision Projected {stat}",
    min_value=0.0,
    value=round(float(projections.get(stat, 0.0)), 2),
    step=0.1,
    key=f"ev_projection_{player}_{stat}_{round(float(projections.get(stat, 0.0)), 2)}"
)

cv_line = st.number_input(
    "Prop Betting Line",
    min_value=0.0,
    value=0.0,
    step=0.5,
    key=f"ev_prop_line_{player}_{stat}"
)

cv_market = st.number_input(
    "ProphetX Market Probability (%)",
    min_value=0.0,
    max_value=99.99,
    value=0.0,
    step=0.1,
    key=f"ev_market_{player}_{stat}"
)

cv_player_id = get_player_id(player)
cv_games = st.session_state.get(f"cv_games_{cv_player_id}")

if cv_games is None or len(cv_games) < 20:
    cv_count = 0 if cv_games is None else len(cv_games)
    st.warning(
        f"Limited data: only {cv_count} games available. "
        "20 games are recommended for more stable projections."
    )

cv_stat = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST", "PRA": "PRA"}.get(stat, "PTS")
if cv_games is not None and cv_stat == "PRA" and all(c in cv_games.columns for c in ["PTS", "REB", "AST"]):
    cv_games = cv_games.copy()
    cv_games["PRA"] = cv_games["PTS"] + cv_games["REB"] + cv_games["AST"]

if cv_games is not None and cv_stat in cv_games.columns:
    cv_values = pd.to_numeric(
        cv_games[cv_stat], errors="coerce"
    ).dropna()
else:
    cv_values = pd.Series(dtype=float)

cv_valid_count = len(cv_values)

if cv_valid_count >= 5:
    cv_auto_sd = float(cv_values.std())
    if not pd.notna(cv_auto_sd) or cv_auto_sd <= 0:
        cv_auto_sd = None
else:
    cv_auto_sd = None

if cv_auto_sd is None:
    st.warning(
        f"Insufficient or invalid variability data "
        f"({cv_valid_count} valid games). "
        "At least 5 valid games with nonzero variability are required."
    )

if cv_auto_sd is None:
    st.info(
        "Probability and EV estimates are unavailable "
        "until sufficient valid game data is available."
    )
    st.stop()

cv_sd = st.number_input(
    "Player Statistic Standard Deviation",
    min_value=0.1,
    value=round(max(cv_auto_sd, 0.1), 2),
    step=0.1
)

if cv_line > 0 and cv_market > 0:
    cv_probability = norm.sf(cv_line, loc=cv_projection, scale=cv_sd)
    cv_price = cv_market / 100
    cv_ev = (cv_probability / cv_price - 1) * 100
else:
    st.info("Enter the correct prop line and ProphetX probability to calculate EV.")
    st.stop()

st.metric("Estimated Over Probability", f"{cv_probability:.1%}")
st.metric("Estimated EV Before Fees", f"{cv_ev:+.1f}%")

st.caption(
    "Preliminary normal-distribution estimate. "
    "Standard deviation is estimated from recent player game logs. "
    "EV assumes the market percentage equals the executable price "
    "of a $1 payout contract and excludes fees."
)

