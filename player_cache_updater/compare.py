"""Compare normalized source rows with an existing player cache file."""

import pandas as pd

from .models import PlayerPlan
from .schema import COMPARE_COLUMNS, LOCAL_COLUMNS, PCT_COLUMNS, PCT_TOLERANCE


def _differences(old_row, new_row):
    changes = {}
    for column in COMPARE_COLUMNS:
        old, new = old_row[column], new_row[column]
        if column in PCT_COLUMNS:
            same = abs(float(old) - float(new)) < PCT_TOLERANCE
        else:
            same = old == new
        if not same:
            changes[column] = [_plain(old), _plain(new)]
    return changes


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def _describe(row):
    return {
        "game_id": row["Game_ID"],
        "game_date": row["GAME_DATE"],
        "matchup": row["MATCHUP"],
    }


def plan_player(player_id, name, local, source_rows):
    """Return the update plan for one player.

    ``local`` is a ``LocalFile`` or ``None`` when no cache file exists.
    Source rows win over cached rows with the same game id.
    """
    plan = PlayerPlan(
        player_id=int(player_id), name=name, existing_file=local is not None
    )
    source = source_rows.set_index("Game_ID", drop=False)
    if local is None:
        plan.new_games = [_describe(row) for _, row in source.iterrows()]
        plan.proposed = _sorted(source_rows)
        return plan

    cached = local.rows.set_index("Game_ID", drop=False)
    plan.duplicates_removed = local.duplicates_removed
    plan.ids_normalized = local.ids_normalized

    replaced = []
    for game_id, new_row in source.iterrows():
        if game_id not in cached.index:
            plan.new_games.append(_describe(new_row))
            replaced.append(game_id)
            continue
        changes = _differences(cached.loc[game_id], new_row)
        if changes:
            plan.corrections.append({**_describe(new_row), "changes": changes})
            replaced.append(game_id)
        else:
            plan.unchanged += 1

    plan.local_only = int((~cached.index.isin(source.index)).sum())
    if plan.changed:
        kept = local.rows[~local.rows["Game_ID"].isin(replaced)]
        added = source_rows[source_rows["Game_ID"].isin(replaced)]
        plan.proposed = _sorted(pd.concat([kept, added], ignore_index=True))
    return plan


def _sorted(rows):
    ordered = rows.sort_values(
        ["_date", "Game_ID"], ascending=[False, False], kind="stable"
    )
    return ordered[LOCAL_COLUMNS + ["_date"]].reset_index(drop=True)
