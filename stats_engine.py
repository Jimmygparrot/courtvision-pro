import pandas as pd

# Load our basketball statistics
df = pd.read_csv("player_stats.csv")

# Calculate Points + Rebounds + Assists
df["PRA"] = (
    df["Points"]
    + df["Rebounds"]
    + df["Assists"]
)

# Calculate averages for each player
averages = df.groupby("Player")[
    ["Minutes", "Points", "Rebounds", "Assists", "PRA"]
].mean()

# Display results
print("\n=== COURTVISION PRO ===")
print("NBA PLAYER STATISTICS ENGINE\n")

print(averages.round(1))

