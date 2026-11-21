"""Validation and normalization of source and cached game-log rows."""

import pandas as pd

from .models import Failure, LocalFile, LocalFileError, NormalizedSource, SourceError
from .schema import (
    COUNT_COLUMNS,
    KEY_COLUMNS,
    LOCAL_COLUMNS,
    MADE_ATTEMPTED,
    PCT_ATTEMPTS,
    PCT_COLUMNS,
    SOURCE_TO_LOCAL,
)

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_IDENTITY_COUNTS = {"SEASON_ID", "Player_ID"}


def format_game_date(timestamp):
    """Format like nba_api: ``Nov 02, 2026`` (locale independent)."""
    return f"{MONTHS[timestamp.month - 1]} {timestamp.day:02d}, {timestamp.year}"


def normalize_game_ids(series):
    """Return (10-character ids, validity mask); invalid ids become NaN."""
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    valid = text.str.fullmatch(r"\d{1,10}").fillna(False).astype(bool)
    padded = text.where(valid).str.zfill(10).astype(object)
    return padded.where(valid), valid


def normalize_player_ids(series):
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & (numeric > 0) & (numeric % 1 == 0)
    return numeric.where(valid), valid


def _set_reason(reasons, mask, text):
    mask = mask.fillna(False).astype(bool) & reasons.isna()
    reasons.loc[mask] = text


def normalize_source_frame(raw, season_start_year, today):
    """Validate source rows and convert them to the cache schema.

    Rows that fail validation are excluded and reported. Missing shooting
    percentages are only filled (with 0.0) when there were no attempts.
    """
    frame = raw.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [name for name in SOURCE_TO_LOCAL if name not in frame.columns]
    if missing:
        raise SourceError(f"Source is missing required columns: {missing}")
    if frame.empty:
        raise SourceError("Source contains no rows.")

    reasons = pd.Series(pd.NA, index=frame.index, dtype=object)

    player_ids, player_ok = normalize_player_ids(frame["player_id"])
    _set_reason(reasons, ~player_ok, "invalid player_id")

    game_ids, game_ok = normalize_game_ids(frame["game_id"])
    _set_reason(reasons, ~game_ok, "invalid game_id")

    dates = pd.to_datetime(frame["game_date"], errors="coerce", format="mixed")
    _set_reason(reasons, dates.isna(), "invalid game_date")
    _set_reason(
        reasons,
        dates.dt.normalize() > pd.Timestamp(today),
        "game_date is in the future",
    )

    season = pd.to_numeric(frame["season_id"], errors="coerce")
    season_ok = (season % 1 == 0) & (season >= 10000) & (
        season % 10000 == season_start_year
    )
    _set_reason(reasons, ~season_ok, "season_id does not match the season")

    outcome = frame["wl"].astype("string").str.strip().str.upper()
    _set_reason(reasons, ~outcome.isin(["W", "L"]), "invalid wl")
    matchup = frame["matchup"].astype("string").str.strip()
    _set_reason(
        reasons,
        matchup.isna() | matchup.eq("") | matchup.str.lower().eq("nan"),
        "missing matchup",
    )

    numbers = {}
    for column in COUNT_COLUMNS:
        if column in _IDENTITY_COUNTS:
            continue
        values = pd.to_numeric(frame[column.lower()], errors="coerce")
        numbers[column] = values
        _set_reason(
            reasons,
            values.isna() | (values % 1 != 0),
            f"missing or non-integer {column}",
        )
        if column != "PLUS_MINUS":
            _set_reason(reasons, values < 0, f"negative {column}")

    for made, attempted in MADE_ATTEMPTED:
        _set_reason(
            reasons,
            numbers[made] > numbers[attempted],
            f"{made} exceeds {attempted}",
        )
    _set_reason(
        reasons,
        numbers["REB"] != numbers["OREB"] + numbers["DREB"],
        "REB does not equal OREB + DREB",
    )

    percentages = {}
    for column in PCT_COLUMNS:
        values = pd.to_numeric(frame[column.lower()], errors="coerce")
        percentages[column] = values
        attempts = numbers[PCT_ATTEMPTS[column]]
        _set_reason(
            reasons,
            values.isna() & (attempts > 0),
            f"missing {column} despite attempts",
        )
        _set_reason(
            reasons,
            (values < 0) | (values > 1),
            f"{column} outside 0-1",
        )

    failures = [
        Failure(
            "source",
            str(frame.at[index, "player_id"]),
            str(frame.at[index, "game_id"]),
            reason,
        )
        for index, reason in reasons.dropna().items()
    ]

    good = reasons.isna()
    rows = pd.DataFrame(index=frame.index[good])
    rows["SEASON_ID"] = season[good].astype("int64")
    rows["Player_ID"] = player_ids[good].astype("int64")
    rows["Game_ID"] = game_ids[good]
    rows["GAME_DATE"] = dates[good].map(format_game_date)
    rows["MATCHUP"] = matchup[good].astype(object)
    rows["WL"] = outcome[good].astype(object)
    for column in COUNT_COLUMNS:
        if column not in _IDENTITY_COUNTS:
            rows[column] = numbers[column][good].astype("int64")
    pct_filled = 0
    for column in PCT_COLUMNS:
        pct_filled += int(percentages[column][good].isna().sum())
        rows[column] = percentages[column][good].fillna(0.0).astype(float)
    rows = rows[LOCAL_COLUMNS]
    rows["_date"] = dates[good].dt.normalize()

    rows, conflict_failures, removed = _collapse_source_duplicates(rows)
    failures.extend(conflict_failures)
    return NormalizedSource(
        rows=rows.reset_index(drop=True),
        failures=failures,
        duplicates_removed=removed,
        pct_filled=pct_filled,
        raw_rows=len(frame),
    )


def _collapse_source_duplicates(rows):
    duplicated = rows.duplicated(KEY_COLUMNS, keep=False)
    if not duplicated.any():
        return rows, [], 0

    comparable = rows.drop(columns=["_date"])
    distinct = comparable[duplicated].drop_duplicates()
    conflicting_keys = set(
        map(tuple, distinct[distinct.duplicated(KEY_COLUMNS, keep=False)][
            KEY_COLUMNS
        ].to_numpy())
    )
    in_conflict = rows[KEY_COLUMNS].apply(tuple, axis=1).isin(conflicting_keys)
    failures = [
        Failure(
            "source",
            str(row.Player_ID),
            str(row.Game_ID),
            "conflicting duplicate source rows",
        )
        for row in rows[in_conflict].itertuples()
    ]
    kept = rows[~in_conflict]
    before = len(kept)
    kept = kept.drop_duplicates(KEY_COLUMNS, keep="first")
    return kept, failures, before - len(kept)


def normalize_local_frame(raw, path_label="cache file"):
    """Normalize an existing cache file; duplicates collapse to one row."""
    missing = [column for column in LOCAL_COLUMNS if column not in raw.columns]
    if missing:
        raise LocalFileError(f"{path_label} is missing columns: {missing}")

    rows = raw[LOCAL_COLUMNS].copy()
    original_ids = (
        rows["Game_ID"].astype("string").str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    game_ids, valid = normalize_game_ids(rows["Game_ID"])
    if not valid.all():
        raise LocalFileError(f"{path_label} has invalid Game_ID values")
    dates = pd.to_datetime(rows["GAME_DATE"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise LocalFileError(f"{path_label} has invalid GAME_DATE values")

    try:
        for column in COUNT_COLUMNS:
            rows[column] = pd.to_numeric(rows[column]).astype("int64")
        for column in PCT_COLUMNS:
            rows[column] = pd.to_numeric(rows[column]).fillna(0.0).astype(float)
    except (ValueError, TypeError) as error:
        raise LocalFileError(f"{path_label} has invalid numbers: {error}")

    rows["Game_ID"] = game_ids
    rows["GAME_DATE"] = dates.map(format_game_date)
    rows["_date"] = dates.dt.normalize()
    rows["_padded"] = original_ids.str.len().eq(10).to_numpy()

    before = len(rows)
    # Prefer the row already written with a full 10-character id.
    rows = rows.sort_values("_padded", kind="stable").drop_duplicates(
        KEY_COLUMNS, keep="last"
    )
    removed = before - len(rows)
    normalized = int((~rows["_padded"]).sum())
    rows = rows.drop(columns=["_padded"]).reset_index(drop=True)
    return LocalFile(rows=rows, duplicates_removed=removed, ids_normalized=normalized)
