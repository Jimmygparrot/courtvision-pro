<#
Publishes ONLY data/historical/nba_games_2026-27.csv to GitHub via a pull request.
Uses a temporary isolated worktree of origin/main, so the working folder and
its uncommitted changes are never touched. Never pushes to main.
#>
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$CsvPath,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Allowed = 'data/historical/nba_games_2026-27.csv'
if (-not $CsvPath) { $CsvPath = Join-Path $RepoRoot $Allowed }
if (-not (Test-Path $CsvPath)) { throw "Results CSV not found: $CsvPath" }

function Invoke-WorkGit { & git -C $script:Work @args; if ($LASTEXITCODE -ne 0) { throw "git $args failed" } }

& git -C $RepoRoot fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$branch = "results/update-$stamp"
$Work = Join-Path ([IO.Path]::GetTempPath()) "courtvision-publish-$stamp"
& git -C $RepoRoot worktree add -b $branch $Work origin/main
if ($LASTEXITCODE -ne 0) { throw 'worktree add failed' }

try {
    # Refuse to replace newer or more complete published results with older data.
    $published = Join-Path $Work $Allowed
    $oldRows = @(Import-Csv -LiteralPath $published)
    $newRows = @(Import-Csv -LiteralPath $CsvPath)
    if ($newRows.Count -eq 0) { throw 'Results CSV has no rows.' }
    $oldMax = ($oldRows | Measure-Object -Property GAME_DATE -Maximum).Maximum
    $newMax = ($newRows | Measure-Object -Property GAME_DATE -Maximum).Maximum
    if ([string]$newMax -lt [string]$oldMax) { throw "Refusing to publish older results (new $newMax < published $oldMax)." }
    $newKeys = [Collections.Generic.HashSet[string]]::new([string[]]@($newRows | ForEach-Object { "$($_.GAME_ID)|$($_.TEAM_ID)" }))
    $missing = @($oldRows | Where-Object { -not $newKeys.Contains("$($_.GAME_ID)|$($_.TEAM_ID)") })
    if ($missing.Count -gt 0) { throw "Refusing to publish: $($missing.Count) published team-game rows are missing from the new file." }

    Copy-Item -LiteralPath $CsvPath -Destination $published -Force

    $changed = @(Invoke-WorkGit status --porcelain | ForEach-Object { $_.Substring(3).Trim() })
    $unexpected = @($changed | Where-Object { $_ -ne $Allowed })
    if ($unexpected.Count -gt 0) { throw "Refusing to publish unapproved files: $($unexpected -join ', ')" }

    # Line-ending normalisation can make the file look modified while the content is identical.
    Invoke-WorkGit add -- $Allowed
    $staged = @(Invoke-WorkGit diff --cached --name-only)
    if ($staged.Count -eq 0) { Write-Output 'Results unchanged; nothing to publish.'; return }
    if ($staged.Count -ne 1 -or $staged[0] -ne $Allowed) { throw "Unexpected staged files: $($staged -join ', ')" }    Invoke-WorkGit diff --cached --stat

    if ($DryRun) { Write-Output "DRY RUN: would commit and push $branch (PR into main)."; return }

    Invoke-WorkGit -c user.name='CourtVision Results Bot' -c user.email='noreply@courtvision.invalid' commit -m 'Update current-season NBA team results' -m 'Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'
    Invoke-WorkGit push origin $branch
    $url = "https://github.com/Jimmygparrot/courtvision-pro/pull/new/$branch"
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        & gh pr create --repo Jimmygparrot/courtvision-pro --base main --head $branch `
            --title 'Update current-season NBA team results' `
            --body 'Automated refresh of data/historical/nba_games_2026-27.csv only.'
    } else {
        Write-Output "Open the pull request: $url"
    }
}
finally {
    & git -C $RepoRoot worktree remove --force $Work 2>$null
    & git -C $RepoRoot branch -D $branch 2>$null | Out-Null
}

