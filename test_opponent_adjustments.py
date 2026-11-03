import pandas as pd
from nba_api.stats.static import teams

games = pd.read_csv("opponent_backtest_data.csv")
defense = pd.read_csv("historical_defense.csv")

team_map = {
    team["abbreviation"]: team["id"]
    for team in teams.get_teams()
}

games["TEAM_ID"] = games["Opponent"].map(team_map)

league = (
    defense.groupby("Game_Date")[
        ["OPP_PTS", "OPP_REB", "OPP_AST"]
    ]
    .mean()
    .reset_index()
)

league = league.rename(columns={
    "OPP_PTS": "League_PTS",
    "OPP_REB": "League_REB",
    "OPP_AST": "League_AST"
})

merged = games.merge(
    defense,
    left_on=["Date", "TEAM_ID"],
    right_on=["Game_Date", "TEAM_ID"],
    how="left",
    validate="many_to_one"
)

merged = merged.merge(
    league,
    on="Game_Date",
    how="left",
    validate="many_to_one"
)

for stat in ["PTS", "REB", "AST"]:
    merged[f"Factor_{stat}"] = (
        merged[f"OPP_{stat}"] /
        merged[f"League_{stat}"]
    )

factors = {
    "PTS": "Factor_PTS",
    "REB": "Factor_REB",
    "AST": "Factor_AST"
}

merged["Opponent_Projection"] = float("nan")

for stat, factor in factors.items():
    mask = merged["Statistic"] == stat

    merged.loc[mask, "Opponent_Projection"] = (
        merged.loc[mask, "Baseline_Projection"] *
        merged.loc[mask, factor]
    )

components = merged[
    merged["Statistic"].isin(["PTS", "REB", "AST"])
]

pra = (
    components.groupby(["Player", "Date"])[
        "Opponent_Projection"
    ]
    .sum()
    .rename("Adjusted_PRA")
    .reset_index()
)

merged = merged.merge(
    pra,
    on=["Player", "Date"],
    how="left",
    validate="many_to_one"
)

mask = merged["Statistic"] == "PRA"

merged.loc[mask, "Opponent_Projection"] = (
    merged.loc[mask, "Adjusted_PRA"]
)

merged["Opponent_Error"] = (
    merged["Opponent_Projection"] - merged["Actual"]
).abs()

merged.to_csv(
    "opponent_adjustment_backtest.csv",
    index=False
)

results = (
    merged.groupby("Statistic")[
        ["Baseline_Error", "Minutes_Error", "Opponent_Error"]
    ]
    .mean()
    .round(3)
)

print("\n=== COURTVISION MODEL COMPARISON ===")
print(results.to_string())

print("\nPrediction records:", len(merged))
print("Missing adjusted projections:",
      merged["Opponent_Projection"].isna().sum())

print("\nSaved: opponent_adjustment_backtest.csv")
