
set -e

# Configuration
HEALTH_URL="http://localhost:8005/health"
TIMEOUT=10
MAX_RETRIES=3

echo "🔍 Running Voxtral Health Check..."
echo "================================="

# Function to check URL
check_url() {
    local url=$1
    local name=$2
    
    echo -n "Checking $name..."
    
    if curl -s -f --max-time $TIMEOUT "$url" > /dev/null; then
        echo " ✅ OK"
        return 0
    else
        echo " ❌ FAILED"
        return 1
    fi
}

# Function to get detailed health info
get_health_info() {
    echo "📊 Detailed Health Information:"
    echo "------------------------------"
    
    # Get health response
    if response=$(curl -s --max-time $TIMEOUT "$HEALTH_URL" 2>/dev/null); then
        echo "$response" | python -m json.tool
    else
        echo "❌ Could not retrieve health information"
        return 1
    fi
    
    echo ""
    echo "📈 System Resources:"
    echo "-------------------"
    
    # Memory usage
    echo "Memory Usage:"
    free -h
    echo ""
    
    # GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Usage:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        echo ""
    fi
    
    # Disk usage
    echo "Disk Usage:"
    df -h /
    echo ""
    
    # Process info
    echo "Process Information:"
    ps aux | grep -E "(python|uvicorn)" | grep -v grep
}

# Main health check
main() {
    local retries=0
    local success=false
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if check_url "$HEALTH_URL" "Health Endpoint"; then
            success=true
            break
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            echo "Retrying in 2 seconds... ($retries/$MAX_RETRIES)"
            sleep 2
        fi
    done
    
    if [ "$success" = true ]; then
        echo ""
        get_health_info
        echo "✅ Health check passed!"
        exit 0
    else
        echo ""
        echo "❌ Health check failed after $MAX_RETRIES attempts"
        
        # Show logs for debugging
        echo ""
        echo "📋 Recent logs:"
        echo "--------------"
        if [ -f "/app/logs/voxtral.log" ]; then
            tail -20 /app/logs/voxtral.log
        else
            echo "No log file found at /app/logs/voxtral.log"
        fi
        
        exit 1
    fi
}

# Run health check
main "$@"
