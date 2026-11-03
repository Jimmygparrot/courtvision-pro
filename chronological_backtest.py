import pandas as pd

df = pd.read_csv("nba_backtest_results.csv")
df["Date"] = pd.to_datetime(df["Date"])

results = []

for (player, stat), group in df.groupby(["Player", "Statistic"]):
    group = group.sort_values("Date").copy()

    for window in [5, 10, 20]:
        group[f"MA{window}"] = (
            group["Actual"].shift(1).rolling(window).mean()
        )

    valid = group.dropna(subset=["MA5", "MA10", "MA20"])

    for _, row in valid.iterrows():
        for window in [5, 10, 20]:
            results.append({
                "Player": player,
                "Date": row["Date"],
                "Statistic": stat,
                "Window": window,
                "Absolute_Error": abs(
                    row[f"MA{window}"] - row["Actual"]
                )
            })

result = pd.DataFrame(results)

if result.empty:
    print("Not enough data.")
else:
    result["Period"] = result["Date"].apply(
        lambda d: "Development" if d < pd.Timestamp("2026-02-01")
        else "Evaluation"
    )

    summary = (
        result.groupby(["Period", "Statistic", "Window"])
        ["Absolute_Error"]
        .mean()
        .round(2)
    )

    print("\n=== CHRONOLOGICAL MODEL COMPARISON ===")
    print(summary.to_string())

    print("\n=== SAMPLE SIZES ===")
    print(result.groupby(["Period", "Window"]).size())

    result.to_csv("chronological_backtest.csv", index=False)
    print("\nSaved: chronological_backtest.csv")
