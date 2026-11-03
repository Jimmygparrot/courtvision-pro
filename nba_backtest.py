import pandas as pd
from nba_stats import get_player_games

players = pd.read_csv("backtest_players.csv")
results = []

print("=== COURTVISION PRO BACKTEST ===")

for _, player in players.iterrows():
    name = player["full_name"]
    player_id = int(player["id"])

    print(f"Testing {name}...", flush=True)

    games = get_player_games(player_id, season="2025-26")

    if games is None or len(games) < 6:
        print("  Not enough games")
        continue

    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    games["PRA"] = games["PTS"] + games["REB"] + games["AST"]

    for stat in ["PTS", "REB", "AST", "PRA"]:
        actual = games[stat].astype(float)
        predicted = actual.shift(1).rolling(5).mean()

        valid = predicted.notna() & actual.notna()

        for date, pred, value in zip(
            games.loc[valid, "GAME_DATE"],
            predicted[valid],
            actual[valid]
        ):
            results.append({
                "Player": name,
                "Date": date,
                "Statistic": stat,
                "Projection": round(float(pred), 2),
                "Actual": float(value),
                "Absolute_Error": round(abs(pred - value), 2)
            })

    print(f"  {len(games)} games loaded")

if results:
    df = pd.DataFrame(results)
    df.to_csv("nba_backtest_results.csv", index=False)

    print("\n=== BACKTEST RESULTS ===")
    print("Players tested:", df["Player"].nunique())
    print("Predictions tested:", len(df))

    print("\nMean Absolute Error:")
    print(
        df.groupby("Statistic")["Absolute_Error"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\nSaved: nba_backtest_results.csv")
else:
    print("\nNo usable historical games returned.")
