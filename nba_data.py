from nba_api.stats.static import players

# CourtVision Pro - NBA Data Engine

# Retrieve the NBA player directory
all_players = players.get_players()

# Keep currently active players
active_players = [
    player for player in all_players
    if player["is_active"]
]

print("=== COURTVISION PRO ===")
print(f"Active NBA players: {len(active_players)}")

# Display the first 10 players
for player in active_players[:10]:
    print(player["full_name"])
import pandas as pd

# Convert NBA player directory into a DataFrame
players_df = pd.DataFrame(active_players)

# Save the player directory
players_df.to_csv("nba_players.csv", index=False)

print("NBA player database saved successfully!")
print(f"Total players saved: {len(players_df)}")

from nba_api.stats.endpoints import playergamelog

# Retrieve LeBron James's NBA player ID
lebron = players.find_players_by_full_name("LeBron James")

if lebron:
    player_id = lebron[0]["id"]

    print("\n=== LEBRON JAMES GAME LOG ===")

    try:
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season="2025-26"
        )

        games = game_log.get_data_frames()[0]

        print(games[["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]].head(10))
	
                # Calculate recent NBA performance averages
        print("\n=== PLAYER PERFORMANCE TRENDS ===")

        stats = ["MIN", "PTS", "REB", "AST"]

        for period in [5, 10, 20]:
            recent_games = games.head(period)

            averages = recent_games[stats].mean()

            print(f"\nLast {len(recent_games)} Games")
            print(f"Minutes: {averages['MIN']:.1f}")
            print(f"Points: {averages['PTS']:.1f}")
            print(f"Rebounds: {averages['REB']:.1f}")
            print(f"Assists: {averages['AST']:.1f}")

            pra = (
                averages["PTS"]
                + averages["REB"]
                + averages["AST"]
            )

            print(f"PRA: {pra:.1f}")
	
    except Exception as error:
        print(f"Unable to retrieve NBA statistics: {error}")


