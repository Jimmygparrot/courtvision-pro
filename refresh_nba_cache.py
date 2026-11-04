import time
from pathlib import Path

import pandas as pd
from nba_stats import get_player_games

SEASON = "2026-27"
CACHE_DIR = Path("data/nba_cache") / SEASON
CACHE_DIR.mkdir(parents=True, exist_ok=True)

players = pd.read_csv("nba_players.csv")
active_players = players[players["is_active"] == True].head(5)

for _, player in active_players.iterrows():
    player_id = int(player["id"])
    player_name = player["full_name"]

    print(f"Downloading {player_name}...", flush=True)

    try:
        games = get_player_games(player_id, SEASON)

        if games.empty:
            print(f"No games available for {player_name}")
        else:
            filename = CACHE_DIR / f"{player_id}.csv"
            games.to_csv(filename, index=False)
            print(f"Saved {len(games)} games for {player_name}")

    except Exception as error:
        print(f"Failed for {player_name}: {error}")

    time.sleep(2)

print("Cache test complete!")
