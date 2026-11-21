from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).parent / ".github" / "workflows" / "update-nba-player-cache.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps(workflow):
    return {step["name"]: step for step in workflow["jobs"]["update"]["steps"]}


def run_scripts(workflow):
    return "\n".join(
        step.get("run", "") for step in workflow["jobs"]["update"]["steps"]
    )


def test_triggers_cover_11am_eastern_manual_runs_and_branch_dry_run(workflow):
    triggers = workflow[True]  # YAML reads the bare key "on" as True
    assert [item["cron"] for item in triggers["schedule"]] == [
        "0 15 * * *",  # 11:00 AM EDT (UTC-4)
        "0 16 * * *",  # 11:00 AM EST (UTC-5)
    ]
    assert triggers["workflow_dispatch"]["inputs"]["apply"]["default"] is False
    assert triggers["push"] == {
        "branches": ["nba-game-predictor"],
        "paths": [".github/workflows/update-nba-player-cache.yml"],
    }
    assert 'TZ=America/New_York date +%H' in run_scripts(workflow)
    assert '"11"' in steps(workflow)["Decide mode"]["run"]


def test_apply_mode_is_never_selected_for_push_events(workflow):
    decide = steps(workflow)["Decide mode"]["run"]
    assert "push" not in decide
    assert decide.count("mode=apply") == 2
    assert 'EVENT_NAME" = "schedule"' in decide
    assert 'EVENT_NAME" = "workflow_dispatch"' in decide
    commit = steps(workflow)["Commit and push validated player CSVs"]
    assert "steps.mode.outputs.mode == 'apply'" in commit["if"]


def test_job_targets_only_the_test_branch_with_minimal_scope(workflow):
    assert workflow["permissions"] == {"contents": "write"}
    assert workflow["env"]["TARGET_BRANCH"] == "nba-game-predictor"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    checkout = steps(workflow)["Check out the test branch"]
    assert checkout["with"]["ref"] == "${{ env.TARGET_BRANCH }}"
    scripts = run_scripts(workflow)
    assert "main" not in scripts
    assert 'HEAD:$TARGET_BRANCH' in scripts
    assert scripts.count("git push") == 1


def test_updater_runs_validated_without_new_players_and_with_guards(workflow):
    scripts = run_scripts(workflow)
    assert "--include-new-players" not in scripts
    updater = steps(workflow)["Download, validate and update player cache"]["run"]
    assert "--min-player-coverage 0.9" in updater
    assert "--max-corrected-records 300" in updater
    assert "--apply --backup-dir" in updater
    assert "pytest test_player_cache_updater.py" in scripts


def test_only_player_csvs_are_staged_and_checked_before_commit(workflow):
    commit = steps(workflow)["Commit and push validated player CSVs"]["run"]
    assert commit.count("git add") == 1
    assert "git add -- data/nba_cache/2026-27/*.csv" in commit
    guard = commit.index("python -m player_cache_updater.staging")
    assert guard < commit.index("git commit") < commit.index("git push")
    assert "git diff --cached --quiet" in commit
