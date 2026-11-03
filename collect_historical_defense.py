import pandas as pd
import time
from pathlib import Path
from nba_api.stats.endpoints import leaguedashteamstats

SOURCE = Path("opponent_backtest_data.csv")
OUTPUT = Path("historical_defense.csv")

REQUIRED = [
    "TEAM_ID", "TEAM_NAME",
    "OPP_PTS", "OPP_REB", "OPP_AST",
    "Game_Date", "Cutoff_Date"
]

games = pd.read_csv(SOURCE)

dates = (
    pd.to_datetime(games["Date"])
    .drop_duplicates()
    .sort_values()
)

if OUTPUT.exists():
    saved = pd.read_csv(OUTPUT)

    if not set(REQUIRED).issubset(saved.columns):
        raise ValueError("Existing CSV has missing columns.")
else:
    saved = pd.DataFrame(columns=REQUIRED)

def completed_dates(data):
    completed = set()

    if data.empty:
        return completed

    for date, group in data.groupby("Game_Date"):
        if (
            len(group) == 30
            and group["TEAM_ID"].nunique() == 30
            and group[REQUIRED].notna().all().all()
        ):
            completed.add(str(date))

    return completed

completed = completed_dates(saved)

print("=== COURTVISION DEFENSE COLLECTOR ===", flush=True)
print("Total dates:", len(dates), flush=True)
print("Already completed:", len(completed), flush=True)

for number, game_date in enumerate(dates, start=1):

    date_string = game_date.strftime("%Y-%m-%d")

    if date_string in completed:
        continue

    cutoff = game_date - pd.Timedelta(days=1)

    print(
        f"\n[{number}/{len(dates)}] Fetching {date_string}",
        flush=True
    )

    for attempt in range(1, 4):
        try:
            response = leaguedashteamstats.LeagueDashTeamStats(
                season="2025-26",
                measure_type_detailed_defense="Opponent",
                per_mode_detailed="PerGame",
                date_to_nullable=cutoff.strftime("%m/%d/%Y"),
                timeout=60
            )

            df = response.get_data_frames()[0]

            columns = REQUIRED[:5]

            if (
                len(df) != 30
                or not set(columns).issubset(df.columns)
            ):
                raise ValueError("Incomplete team data")

            df = df[columns].copy()

            if (
                df["TEAM_ID"].nunique() != 30
                or df[columns].isna().any().any()
            ):
                raise ValueError("Invalid team data")

            df["Game_Date"] = date_string
            df["Cutoff_Date"] = cutoff.strftime("%Y-%m-%d")

            saved = pd.concat(
                [
                    saved[saved["Game_Date"] != date_string],
                    df
                ],
                ignore_index=True
            )

            saved.to_csv(OUTPUT, index=False)

            completed.add(date_string)

            print("Saved: 30 teams", flush=True)
            break

        except Exception as error:
            print(
                f"Attempt {attempt}/3 failed: {error}",
                flush=True
            )

            if attempt < 3:
                time.sleep(5 * attempt)

    time.sleep(2)

print("\n=== FINAL RESULTS ===")
print("Dates completed:", len(completed))
print("Dates remaining:", len(dates) - len(completed))
print("Team records saved:", len(saved))
print("File:", OUTPUT)
