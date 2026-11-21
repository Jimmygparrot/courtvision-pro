from pathlib import Path
import streamlit as st

# NBA Stats uses "LA Clippers"; the schedule and roster use "Los Angeles Clippers".
TEAM_NAME_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
}


@st.cache_data(ttl=300)
def _fetch_defensive_stats(season):
    import pandas as pd

    cache_file = Path(__file__).resolve().parent / "data" / f"defense_{season}.csv"

    if not cache_file.exists():
        return None

    df = pd.read_csv(cache_file)
    required = {"TEAM_NAME", "GP", "OPP_PTS", "OPP_REB", "OPP_AST"}

    if df.empty or not required.issubset(df.columns):
        return None

    return df


def get_defensive_stats(season="2026-27"):
    return _fetch_defensive_stats(season)


def get_defense_factors(opponent, season="2026-27"):
    df = get_defensive_stats(season)

    if df is None:
        return None

    lookup_name = TEAM_NAME_ALIASES.get(opponent, opponent)
    row = df[df["TEAM_NAME"] == lookup_name]

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
