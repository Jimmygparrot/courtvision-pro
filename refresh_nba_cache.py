
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from nba_stats import get_player_games

SEASON = "2026-27"
CACHE_DIR = Path("data/nba_cache") / SEASON
STATE_FILE = CACHE_DIR / "refresh_progress.json"


def load_position():
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if state.get("season") == SEASON:
            return max(0, int(state.get("position", 0)))
    except (OSError, ValueError, TypeError):
        pass
    return 0


def save_position(position):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = STATE_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps({"season": SEASON, "position": position}),
        encoding="utf-8",
    )
    os.replace(temp_file, STATE_FILE)


def refresh_players(limit=20, delay=3):
    players = pd.read_csv("nba_players.csv")
    players = players[players["is_active"] == True].reset_index(drop=True)

    total = len(players)
    if total == 0:
        print("No active players found.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    start = load_position() % total
    count = min(max(0, limit), total)

    updated = 0
    unchanged = 0
    failed = 0

    print(f"Starting at player {start + 1} of {total}", flush=True)

    for step in range(count):
        position = (start + step) % total
        player = players.iloc[position]

        player_id = int(player["id"])
        player_name = player["full_name"]
        cache_file = CACHE_DIR / f"{player_id}.csv"

        before = cache_file.read_bytes() if cache_file.exists() else None

        print(f"Refreshing {player_name}...", flush=True)

        try:
            games = get_player_games(
                player_id,
                SEASON,
                force_refresh=True,
            )

            if games.attrs.get("refresh_error"):
                failed += 1
                print("  FAILED: NBA API unavailable")

            elif games.empty:
                unchanged += 1
                print("  EMPTY: No game data returned")

            elif (
                before is not None
                and cache_file.exists()
                and cache_file.read_bytes() == before
            ):
                unchanged += 1
                print("  UNCHANGED: Existing data preserved")

            else:
                updated += 1
                print(f"  UPDATED: {len(games)} games")

        except Exception as error:
            failed += 1
            print(f"  FAILED: {error}")

        save_position((position + 1) % total)

        if step < count - 1:
            time.sleep(delay)

    print("\nRefresh summary")
    print(f"Updated: {updated}")
    print(f"Unchanged/empty: {unchanged}")
    print(f"Failed: {failed}")
    print(f"Next player position: {load_position() + 1} of {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=3)
    args = parser.parse_args()

    refresh_players(args.limit, args.delay)
