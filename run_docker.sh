#!/bin/bash
# Run Tensorus in Docker

# Set default action
ACTION=${1:-"up"}
SERVICES=${2:-""}

# Show usage
function show_usage {
    echo "Usage: $0 [action] [services]"
    echo ""
    echo "Actions:"
    echo "  up       - Start Tensorus services (default)"
    echo "  down     - Stop Tensorus services"
    echo "  build    - Build Tensorus services"
    echo "  logs     - View logs of Tensorus services"
    echo "  exec     - Execute a command in a container"
    echo "  test     - Run tests in a container"
    echo "  help     - Show this help message"
    echo ""
    echo "Services:"
    echo "  api      - Tensorus API service"
    echo "  dashboard - Tensorus Dashboard service"
    echo "  (empty)  - All services (default)"
    echo ""
    echo "Examples:"
    echo "  $0 up              # Start all services"
    echo "  $0 up api          # Start only the API service"
    echo "  $0 down            # Stop all services"
    echo "  $0 logs dashboard  # View logs of the dashboard service"
    echo "  $0 exec api bash   # Run bash in the API container"
    echo "  $0 test            # Run tests"
}

# Check for help flag
if [[ "$ACTION" == "help" ]]; then
    show_usage
    exit 0
fi

# Set compose file
COMPOSE_FILE="docker-compose.yml"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    if ! docker compose version &> /dev/null; then
        echo "Error: Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    else
        COMPOSE_CMD="docker compose"
    fi
else
    COMPOSE_CMD="docker-compose"
fi

# Check if compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Error: $COMPOSE_FILE not found."
    exit 1
fi

# Run action
case "$ACTION" in
    "up")
        echo "Starting Tensorus services..."
        if [[ -n "$SERVICES" ]]; then
            $COMPOSE_CMD -f $COMPOSE_FILE up -d $SERVICES
        else
            $COMPOSE_CMD -f $COMPOSE_FILE up -d
        fi
        
        # Wait for services to start
        sleep 3
        
        # Show service status
        $COMPOSE_CMD -f $COMPOSE_FILE ps
        
        # Show URLs
        echo ""
        echo "Tensorus services are running!"
        echo "API URL: http://localhost:8000"
        echo "Dashboard URL: http://localhost:8501"
        ;;
    "down")
        echo "Stopping Tensorus services..."
        $COMPOSE_CMD -f $COMPOSE_FILE down
        ;;
    "build")
        echo "Building Tensorus services..."
        if [[ -n "$SERVICES" ]]; then
            $COMPOSE_CMD -f $COMPOSE_FILE build $SERVICES
        else
            $COMPOSE_CMD -f $COMPOSE_FILE build
        fi
        ;;
    "logs")
        if [[ -n "$SERVICES" ]]; then
            $COMPOSE_CMD -f $COMPOSE_FILE logs -f $SERVICES
        else
            $COMPOSE_CMD -f $COMPOSE_FILE logs -f
        fi
        ;;
    "exec")
        if [[ -z "$SERVICES" ]]; then
            echo "Error: You must specify a service name."
            show_usage
            exit 1
        fi
        
        SERVICE_NAME="tensorus-$SERVICES"
        shift 2
        COMMAND=$@
        
        if [[ -z "$COMMAND" ]]; then
            COMMAND="bash"
        fi
        
        echo "Executing '$COMMAND' in $SERVICE_NAME container..."
        docker exec -it $SERVICE_NAME $COMMAND
        ;;
    "test")
        echo "Running tests in Docker container..."
        docker run --rm -v $(pwd):/app -w /app $(docker build -q .) python run_tests.py
        ;;
    *)
        echo "Error: Unknown action '$ACTION'"
        show_usage
        exit 1
        ;;
esac

exit 0 