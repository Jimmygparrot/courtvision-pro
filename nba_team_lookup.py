import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).with_name("nba_players_teams.csv")

def get_player_team(player_id):
    if player_id is None:
        return None

    try:
        players = pd.read_csv(DATA_FILE)

        match = players[
            players["id"] == int(player_id)
        ]

        if match.empty:
            return None

        team = match.iloc[0]["team_name"]

        if pd.isna(team):
            return None

        return str(team)

    except Exception as error:
        print(f"Local team lookup error: {error}")
        return None
