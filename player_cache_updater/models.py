from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


class SourceError(Exception):
    """The source could not be downloaded or is unusable as a whole."""


class LocalFileError(Exception):
    """An existing cache file cannot be read or normalized."""


@dataclass
class Failure:
    scope: str
    player_id: str
    game_id: str
    reason: str


@dataclass
class NormalizedSource:
    rows: pd.DataFrame
    failures: list
    duplicates_removed: int = 0
    pct_filled: int = 0
    raw_rows: int = 0


@dataclass
class LocalFile:
    rows: pd.DataFrame
    duplicates_removed: int = 0
    ids_normalized: int = 0


@dataclass
class PlayerPlan:
    player_id: int
    name: str
    existing_file: bool
    new_games: list = field(default_factory=list)
    corrections: list = field(default_factory=list)
    unchanged: int = 0
    local_only: int = 0
    duplicates_removed: int = 0
    ids_normalized: int = 0
    proposed: Optional[pd.DataFrame] = None
    problems: list = field(default_factory=list)
    local_latest: Optional[pd.Timestamp] = None

    @property
    def changed(self):
        return bool(
            self.new_games
            or self.corrections
            or self.duplicates_removed
            or self.ids_normalized
        )
