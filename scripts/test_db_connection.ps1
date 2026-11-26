<#
.SYNOPSIS
  Read-only connection test for the CourtVision Pro Supabase history database.

.DESCRIPTION
  Prompts for the courtvision_app password (hidden; never echoed, written to
  disk, or stored in command history), connects through the Supabase session
  pooler with sslmode=verify-full, and reads public.prediction_accuracy in a
  read-only transaction. Credentials exist only in this process's environment
  and are removed when the script ends.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\test_db_connection.ps1 -ProjectRef abcdefghijklmnop
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9]+$')]
    [string]$ProjectRef,
    [string]$DbHost = 'aws-1-us-east-1.pooler.supabase.com',
    [int]$Port = 5432,
    [string]$Database = 'postgres',
    [string]$CaCertificate = (Join-Path $env:USERPROFILE 'Downloads\prod-ca-2021.crt'),
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $CaCertificate -PathType Leaf)) {
    throw "CA certificate not found: $CaCertificate"
}
if (-not $Python) {
    $venv = Join-Path $repo '.venv\Scripts\python.exe'
    $Python = if (Test-Path $venv) { $venv } else { 'python' }
}

$secure = Read-Host -Prompt "Password for courtvision_app.$ProjectRef" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:COURTVISION_DB_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    $env:COURTVISION_DB_HOST = $DbHost
    $env:COURTVISION_DB_PORT = "$Port"
    $env:COURTVISION_DB_NAME = $Database
    $env:COURTVISION_DB_USER = "courtvision_app.$ProjectRef"
    $env:COURTVISION_DB_SSLROOTCERT = $CaCertificate

    & $Python (Join-Path $repo 'scripts\check_db_connection.py')
    exit $LASTEXITCODE
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $secure.Dispose()
    Remove-Item Env:COURTVISION_DB_PASSWORD, Env:COURTVISION_DB_HOST, Env:COURTVISION_DB_PORT, `
        Env:COURTVISION_DB_NAME, Env:COURTVISION_DB_USER, Env:COURTVISION_DB_SSLROOTCERT `
        -ErrorAction SilentlyContinue
}
