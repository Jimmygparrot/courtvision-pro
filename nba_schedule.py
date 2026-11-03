from datetime import date, timedelta

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams


def get_next_game(team_name, days_ahead=7):
    team_list = teams.get_teams()
    team = next(
        (t for t in team_list if t["full_name"] == team_name),
        None
    )

    if team is None:
        return None

    team_id = team["id"]
    team_names = {t["id"]: t["full_name"] for t in team_list}

    for offset in range(days_ahead + 1):
        game_date = date.today() + timedelta(days=offset)

        try:
            board = scoreboardv2.ScoreboardV2(
                game_date=game_date.strftime("%Y-%m-%d"),
                timeout=5
            )

            games = board.get_data_frames()[0].drop_duplicates(
                subset=["GAME_ID"]
            )

            for _, game in games.iterrows():
                home_id = int(game["HOME_TEAM_ID"])
                away_id = int(game["VISITOR_TEAM_ID"])

                if team_id == home_id:
                    return {
                        "date": str(game_date),
                        "opponent": team_names.get(away_id, "Unknown"),
                        "location": "Home"
                    }

                if team_id == away_id:
                    return {
                        "date": str(game_date),
                        "opponent": team_names.get(home_id, "Unknown"),
                        "location": "Away"
                    }

        except Exception as error:
            print(f"Schedule lookup failed for {game_date}: {error}")

    return None
