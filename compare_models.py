import pandas as pd

df = pd.read_csv("nba_backtest_results.csv")
df["Date"] = pd.to_datetime(df["Date"])

results = []

for (player, stat), group in df.groupby(["Player", "Statistic"]):
    group = group.sort_values("Date").copy()

    # The first saved projection uses five earlier games.
    # Reconstruct those five values only when comparing
    # future observations with a full 20-game history.
    # Instead, compare on saved actuals after 20 more games.
    for window in [5, 10, 20]:
        group[f"MA{window}"] = (
            group["Actual"].shift(1).rolling(window).mean()
        )

    valid = group.dropna(subset=["MA5", "MA10", "MA20"])

    for window in [5, 10, 20]:
        for _, row in valid.iterrows():
            results.append({
                "Player": player,
                "Statistic": stat,
                "Window": window,
                "Absolute_Error": abs(row[f"MA{window}"] - row["Actual"])
            })

result = pd.DataFrame(results)

if result.empty:
    print("Not enough historical games for comparison.")
else:
    print("\n=== MODEL COMPARISON ===")
    print("Predictions per model:", len(result) // 3)

    summary = result.groupby(
        ["Statistic", "Window"]
    )["Absolute_Error"].mean().round(2)

    print(summary.to_string())

    result.to_csv("model_comparison.csv", index=False)
    print("\nSaved: model_comparison.csv")
