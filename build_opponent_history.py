import pandas as pd
from nba_stats import get_player_id, get_player_games

df = pd.read_csv("minutes_model_backtest.csv")

opponent_records = []

players = sorted(df["Player"].unique())

print("=== BUILDING HISTORICAL OPPONENT DATABASE ===", flush=True)

for player in players:
    print(f"Processing {player}...", flush=True)

    try:
        player_id = get_player_id(player)
        games = get_player_games(player_id, season="2025-26")

        if games is None or games.empty:
            print("  No games returned", flush=True)
            continue

        games = games.copy()

        games["Date"] = pd.to_datetime(
            games["GAME_DATE"]
        ).dt.strftime("%Y-%m-%d")

        games["Opponent"] = games["MATCHUP"].str.extract(
            r"(?:@|vs\.)\s*([A-Z]{3})"
        )[0]

        games["Player"] = player

        opponent_records.append(
            games[["Player", "Date", "Opponent"]]
        )

    except Exception as error:
        print(f"  Error: {error}", flush=True)

if opponent_records:
    opponents = pd.concat(
        opponent_records,
        ignore_index=True
    ).drop_duplicates(["Player", "Date"])

    merged = df.merge(
        opponents,
        on=["Player", "Date"],
        how="left",
        validate="many_to_one"
    )

    merged.to_csv(
        "opponent_backtest_data.csv",
        index=False
    )

    print("\n=== RESULTS ===")
    print("Players:", merged["Player"].nunique())
    print("Historical games:", len(merged) // 4)
    print("Missing opponents:", merged["Opponent"].isna().sum())
    print("\nSaved: opponent_backtest_data.csv")
else:
    print("No opponent records retrieved.")
