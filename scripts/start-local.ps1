<#
.SYNOPSIS
    Start KG Agent locally using Docker Compose

.DESCRIPTION
    This script starts all KG Agent services using Docker Compose.
    It checks for prerequisites, creates necessary directories, and manages the containers.

.PARAMETER Build
    Force rebuild of Docker images

.PARAMETER Detached
    Run containers in detached mode (background)

.PARAMETER Down
    Stop and remove all containers

.PARAMETER Logs
    Show logs from all containers

.PARAMETER Status
    Show status of all containers

.EXAMPLE
    .\start-local.ps1
    Starts all services in foreground mode

.EXAMPLE
    .\start-local.ps1 -Detached
    Starts all services in background

.EXAMPLE
    .\start-local.ps1 -Build -Detached
    Rebuilds images and starts in background

.EXAMPLE
    .\start-local.ps1 -Down
    Stops all services
#>

param(
    [switch]$Build,
    [switch]$Detached,
    [switch]$Down,
    [switch]$Logs,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$ComposeFile = "docker-compose.local.yml"

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "    KG Agent - Local Deployment" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# Check if Docker is running
function Test-Docker {
    Write-Info "Checking Docker..."
    try {
        $null = docker info 2>&1
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Err "Docker is not running. Please start Docker Desktop."
        return $false
    }
}

# Check for LM Studio
function Test-LMStudio {
    Write-Info "Checking for LM Studio on port 1234..."
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Success "LM Studio is running"
        return $true
    }
    catch {
        Write-Warn "LM Studio not detected on port 1234"
        Write-Host "  Agent/Chat features require LM Studio or another OpenAI-compatible LLM." -ForegroundColor Yellow
        Write-Host "  Download: https://lmstudio.ai/" -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
}

# Create required directories
function Initialize-Directories {
    Write-Info "Creating required directories..."
    $dirs = @(
        "storage",
        "storage/screenshots",
        "storage/pdfs",
        "data",
        "data/chroma_db",
        "data/chunks",
        "data/parsed",
        "data/raw",
        "models",
        "models/embeddings",
        "cache",
        "logs"
    )
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "  Created: $dir" -ForegroundColor Gray
        }
    }
    Write-Success "Directories ready"
}

# Create .env file if it doesn't exist
function Initialize-EnvFile {
    $envFile = ".env"
    $envExample = ".env.docker"

    if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
        Write-Info "Creating .env from template..."
        Copy-Item $envExample $envFile
        Write-Success "Created .env file - review and customize settings"
    }
}

# Show status
if ($Status) {
    Write-Info "Container Status:"
    docker compose -f $ComposeFile ps
    exit 0
}

# Show logs
if ($Logs) {
    Write-Info "Showing logs (Ctrl+C to exit)..."
    docker compose -f $ComposeFile logs -f
    exit 0
}

# Stop containers
if ($Down) {
    Write-Info "Stopping all services..."
    docker compose -f $ComposeFile down
    Write-Success "All services stopped"
    exit 0
}

# Prerequisites check
if (-not (Test-Docker)) {
    exit 1
}

Test-LMStudio
Initialize-Directories
Initialize-EnvFile

# Build arguments
$dockerArgs = @("-f", $ComposeFile, "up")

if ($Build) {
    $dockerArgs += "--build"
}

if ($Detached) {
    $dockerArgs += "-d"
}

# Start services
Write-Host ""
Write-Info "Starting KG Agent services..."
Write-Host ""

docker compose @dockerArgs

if ($Detached) {
    Write-Host ""
    Write-Success "Services started in background!"
    Write-Host ""
    Write-Host "  Dashboard:  http://localhost:3000" -ForegroundColor Cyan
    Write-Host "  API:        http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Commands:" -ForegroundColor Gray
    Write-Host "    .\scripts\start-local.ps1 -Logs      # View logs" -ForegroundColor Gray
    Write-Host "    .\scripts\start-local.ps1 -Status    # Check status" -ForegroundColor Gray
    Write-Host "    .\scripts\start-local.ps1 -Down      # Stop services" -ForegroundColor Gray
    Write-Host ""
}

