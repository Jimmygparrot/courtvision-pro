from nba_predictions import refresh_current_season_team_snapshot


if __name__ == "__main__":
    latest_game_date = refresh_current_season_team_snapshot()
    print(f"Current-season team results refreshed through {latest_game_date}.")
