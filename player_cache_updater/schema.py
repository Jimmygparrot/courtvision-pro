"""Constants describing the cached player game-log schema."""

SEASON = "2026-27"
SEASON_START_YEAR = 2026
SOURCE_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/"
    "download/nba_stats_player_game_logs/player_game_logs_2027.csv"
)

# Column order of the existing data/nba_cache CSV files.
LOCAL_COLUMNS = [
    "SEASON_ID", "Player_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS", "VIDEO_AVAILABLE",
]

# The source uses the same names in lowercase.
SOURCE_TO_LOCAL = {column.lower(): column for column in LOCAL_COLUMNS}

COUNT_COLUMNS = [
    "SEASON_ID", "Player_ID", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM",
    "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS", "VIDEO_AVAILABLE",
]
PCT_COLUMNS = ["FG_PCT", "FG3_PCT", "FT_PCT"]
PCT_ATTEMPTS = {"FG_PCT": "FGA", "FG3_PCT": "FG3A", "FT_PCT": "FTA"}
MADE_ATTEMPTED = [("FGM", "FGA"), ("FG3M", "FG3A"), ("FTM", "FTA")]
KEY_COLUMNS = ["Player_ID", "Game_ID"]
COMPARE_COLUMNS = [c for c in LOCAL_COLUMNS if c not in KEY_COLUMNS]

PCT_TOLERANCE = 0.0005
MIN_SOURCE_ROWS = 100
