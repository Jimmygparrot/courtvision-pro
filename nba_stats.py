from nba_api.stats.endpoints import playergamelog


def get_player_games(player_id, season="2025-26"):
    """
    Retrieve NBA game statistics for a player.

    player_id: NBA's unique player identifier
    season: NBA season to retrieve
    """

    try:
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            timeout=30
        )

        games = game_log.get_data_frames()[0]

        if games.empty:
            print("No games found.")
            return None

        return games

    except Exception as error:
        print(f"NBA API error: {error}")
        return None
import pandas as pd



def get_player_id(player_name):
    """
    Find an NBA player ID, ignoring accents and capitalization.
    """
    import pandas as pd
    import unicodedata

    def normalize(name):
        name = unicodedata.normalize("NFKD", str(name))
        return "".join(
            char for char in name
            if not unicodedata.combining(char)
        ).casefold().strip()

    players_df = pd.read_csv("nba_players.csv")

    target = normalize(player_name)

    matches = players_df[
        players_df["full_name"].apply(normalize) == target
    ]

    if matches.empty:
        return None

    return int(matches.iloc[0]["id"])
