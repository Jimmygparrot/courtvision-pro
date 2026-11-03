import streamlit as st
from nba_api.stats.endpoints import leaguedashteamstats


@st.cache_data(ttl=21600)
def get_defensive_stats(season="2026-27"):
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame",
        timeout=30
    )

    return stats.get_data_frames()[0]


def get_defense_factors(opponent, season="2026-27"):
    df = get_defensive_stats(season)

    row = df[df["TEAM_NAME"] == opponent]

    if row.empty:
        return None

    factors = {}

    for stat, column in {
        "PTS": "OPP_PTS",
        "REB": "OPP_REB",
        "AST": "OPP_AST",
    }.items():
        league_average = (
            (df[column] * df["GP"]).sum()
            / df["GP"].sum()
        )

        factors[stat] = float(
            row.iloc[0][column] / league_average
        )

    return factors
