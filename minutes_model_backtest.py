import pandas as pd
from nba_stats import get_player_games

players = pd.read_csv("backtest_players.csv")
results = []

print("=== COURTVISION MINUTES MODEL BACKTEST ===", flush=True)

for _, player in players.iterrows():
    name = player["full_name"]
    player_id = int(player["id"])

    print(f"Testing {name}...", flush=True)

    games = get_player_games(player_id, season="2025-26")

    if games is None or len(games) < 21:
        print("  Not enough games", flush=True)
        continue

    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    for col in ["PTS", "REB", "AST", "MIN"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")

    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    for i in range(20, len(games)):
        history = games.iloc[i-20:i]
        recent10 = games.iloc[i-10:i]
        actual_game = games.iloc[i]

        if history[["PTS", "REB", "AST", "MIN"]].isna().any().any():
            continue

        if actual_game[["PTS", "REB", "AST"]].isna().any():
            continue

        historical_minutes = history["MIN"].mean()
        projected_minutes = recent10["MIN"].mean()

        if historical_minutes <= 0:
            continue

        factor = projected_minutes / historical_minutes

        for stat in ["PTS", "REB", "AST", "PRA"]:
            if stat == "PRA":
                baseline = (
                    history["PTS"].mean()
                    + history["REB"].mean()
                    + history["AST"].mean()
                )
                actual = (
                    actual_game["PTS"]
                    + actual_game["REB"]
                    + actual_game["AST"]
                )
            else:
                baseline = history[stat].mean()
                actual = actual_game[stat]

            adjusted = baseline * factor

            results.append({
                "Player": name,
                "Date": actual_game["GAME_DATE"],
                "Statistic": stat,
                "Baseline_Error": abs(baseline - actual),
                "Minutes_Error": abs(adjusted - actual),
                "Baseline_Projection": baseline,
                "Minutes_Projection": adjusted,
                "Actual": actual,
            })

if results:
    df = pd.DataFrame(results)

    df.to_csv("minutes_model_backtest.csv", index=False)

    summary = df.groupby("Statistic")[
        ["Baseline_Error", "Minutes_Error"]
    ].mean().round(3)

    summary["Improvement"] = (
        summary["Baseline_Error"] - summary["Minutes_Error"]
    ).round(3)

    print("\n=== MODEL COMPARISON ===")
    print(summary.to_string())

    print("\nPlayers tested:", df["Player"].nunique())
    print("Games tested:", len(df) // 4)
    print("\nSaved: minutes_model_backtest.csv")
else:
    print("No usable historical games found.")
