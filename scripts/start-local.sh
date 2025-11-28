#!/bin/bash
# ============================================================================
# KG Agent - Local Deployment Script (Linux/macOS)
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

COMPOSE_FILE="docker-compose.local.yml"

# Parse arguments
BUILD=false
DETACHED=false
DOWN=false
LOGS=false
STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build|-b) BUILD=true ;;
        --detached|-d) DETACHED=true ;;
        --down) DOWN=true ;;
        --logs|-l) LOGS=true ;;
        --status|-s) STATUS=true ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build, -b      Force rebuild of Docker images"
            echo "  --detached, -d   Run containers in background"
            echo "  --down           Stop and remove all containers"
            echo "  --logs, -l       Show logs from all containers"
            echo "  --status, -s     Show status of all containers"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo ""
echo -e "${MAGENTA}========================================"
echo "    KG Agent - Local Deployment"
echo -e "========================================${NC}"
echo ""

# Check Docker
check_docker() {
    info "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker."
        exit 1
    fi
    success "Docker is running"
}

# Check LM Studio
check_lmstudio() {
    info "Checking for LM Studio on port 1234..."
    if curl -s --max-time 2 http://localhost:1234/v1/models > /dev/null 2>&1; then
        success "LM Studio is running"
    else
        warn "LM Studio not detected on port 1234"
        echo "  Agent/Chat features require LM Studio or another OpenAI-compatible LLM."
        echo "  Download: https://lmstudio.ai/"
        echo ""
    fi
}

# Create directories
init_directories() {
    info "Creating required directories..."
    mkdir -p storage/{screenshots,pdfs}
    mkdir -p data/{chroma_db,chunks,parsed,raw}
    mkdir -p models/embeddings
    mkdir -p cache logs
    success "Directories ready"
}

# Create .env if needed
init_env() {
    if [[ ! -f ".env" && -f ".env.docker" ]]; then
        info "Creating .env from template..."
        cp .env.docker .env
        success "Created .env file - review and customize settings"
    fi
}

# Status
if $STATUS; then
    info "Container Status:"
    docker compose -f $COMPOSE_FILE ps
    exit 0
fi

# Logs
if $LOGS; then
    info "Showing logs (Ctrl+C to exit)..."
    docker compose -f $COMPOSE_FILE logs -f
    exit 0
fi

# Down
if $DOWN; then
    info "Stopping all services..."
    docker compose -f $COMPOSE_FILE down
    success "All services stopped"
    exit 0
fi

# Prerequisites
check_docker
check_lmstudio
init_directories
init_env

# Build command
DOCKER_ARGS="-f $COMPOSE_FILE up"

if $BUILD; then
    DOCKER_ARGS="$DOCKER_ARGS --build"
fi

if $DETACHED; then
    DOCKER_ARGS="$DOCKER_ARGS -d"
fi

# Start
echo ""
info "Starting KG Agent services..."
echo ""

docker compose $DOCKER_ARGS

if $DETACHED; then
    echo ""
    success "Services started in background!"
    echo ""
    echo -e "  Dashboard:  ${CYAN}http://localhost:3000${NC}"
    echo -e "  API:        ${CYAN}http://localhost:8000${NC}"
    echo -e "  API Docs:   ${CYAN}http://localhost:8000/docs${NC}"
    echo ""
    echo "  Commands:"
    echo "    ./scripts/start-local.sh --logs      # View logs"
    echo "    ./scripts/start-local.sh --status    # Check status"
    echo "    ./scripts/start-local.sh --down      # Stop services"
    echo ""
fi

