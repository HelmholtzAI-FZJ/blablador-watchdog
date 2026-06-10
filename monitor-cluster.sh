#!/bin/bash

# Cluster Usage Monitoring Script
# Monitors active users, their rights, used models, rate limits, and throttle status

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

REDIS_CLI="redis-cli -h $REDIS_HOST -p $REDIS_PORT"

# Check if redis-cli is installed
if ! command -v redis-cli &> /dev/null; then
    echo -e "${RED}Error: redis-cli is not installed${NC}"
    echo ""
    echo "Install redis-cli:"
    echo "  - macOS:  brew install redis"
    echo "  - Ubuntu: sudo apt-get install redis-tools"
    echo "  - CentOS: sudo yum install redis"
    echo "  - Docker: docker run -it --rm redis redis-cli --help"
    echo ""
    echo "Or use kubectl to access Redis directly:"
    echo "  kubectl exec -n app-ns -it <redis-pod-name> -- redis-cli"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Cluster Usage Monitor

Usage: $0 [command] [options]

Commands:
  summary          Quick summary of cluster usage (default)
  active-users     List all active users with their rights
  rate-limits      Show current rate limit usage per user/model
  throttled        List throttled users
  models           Show which models are being used
  banned-users     List all banned/blacklisted users
  all              Full cluster usage report
  help             Show this help message

Options:
  --limit N            Limit output to N users (default: 50)
  --format json        Output in JSON format
  --force-port-forward Restart the port-forward even if it's already running

Environment Variables:
  REDIS_HOST       Redis hostname (default: localhost)
  REDIS_PORT       Redis port (default: 6379)

Examples:
  $0
  $0 active-users --limit 10
  $0 rate-limits --format json
  $0 all

Kubernetes:
  # The script now auto-starts the port-forward if needed
  # Or explicitly start it:
  kubectl port-forward svc/redis-service 6379:6379 -n app-ns &
  export REDIS_HOST=localhost
  $0

  # Force restart the port-forward if needed:
  $0 --force-port-forward
  $0 all --force-port-forward

EOF
}

get_user_status() {
    local email="$1"
    if [ -z "$email" ] || [ "$email" = "(nil)" ]; then
        echo "external"
        return
    fi
    local result=$($REDIS_CLI HGET "access_control:users" "$email" 2>/dev/null)
    if [ "$result" = "(nil)" ] || [ -z "$result" ]; then
        if [[ "$email" == *"@fz-juelich.de" ]]; then
            echo "internal"
        else
            echo "external"
        fi
    else
        echo "$result"
    fi
}

get_user_email() {
    local token="$1"
    local result=$($REDIS_CLI GET "cache:auth:$token" 2>/dev/null)
    if [ "$result" = "(nil)" ] || [ -z "$result" ]; then
        echo ""
    else
        echo "$result"
    fi
}

format_status() {
    local status="$1"
    case "$status" in
        superuser)
            echo -e "${GREEN}superuser${NC}"
            ;;
        elevated)
            echo -e "${BLUE}elevated${NC}"
            ;;
        internal)
            echo -e "${GREEN}internal${NC}"
            ;;
        external)
            echo -e "${YELLOW}external${NC}"
            ;;
        punished)
            echo -e "${YELLOW}punished${NC}"
            ;;
        blacklisted)
            echo -e "${RED}blacklisted${NC}"
            ;;
        *)
            echo -e "${YELLOW}external${NC}"
            ;;
    esac
}

get_rate_limits() {
    local model="$1"
    local priority="$2"
    
    case "$priority" in
        superuser)
            echo "unlimited"
            ;;
        *)
            local limits=$($REDIS_CLI GET "rate_limits:$model:$priority" 2>/dev/null)
            if [ -n "$limits" ]; then
                echo "$limits"
            else
                echo "N/A"
            fi
            ;;
    esac
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Extract user tokens from rate limit keys
extract_rate_keys() {
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | sed 's/rate://g' | cut -d':' -f1 | sort -u
}

# Extract unique user/model combinations from rate keys
get_user_model_usage() {
    local limit="${1:-50}"
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | sed 's/rate://g' | while read -r key; do
        local user=$(echo "$key" | cut -d':' -f1)
        local model=$(echo "$key" | cut -d':' -f2- | rev | cut -d':' -f2- | rev)
        local current_count=$($REDIS_CLI GET "rate:$key" 2>/dev/null)
        if [ -n "$current_count" ] && [ "$current_count" != "(nil)" ]; then
            echo "$user|$model|$current_count"
        fi
    done | head -n "$limit" | sort -u
}

# Get rate limit threshold for a user/model combo
get_rate_limit() {
    local user="$1"
    local model="$2"
    
    # Try to get user priority from cache
    local priority=$($REDIS_CLI GET "user:priority:$user" 2>/dev/null)
    if [ "$priority" = "(nil)" ] || [ -z "$priority" ]; then
        # Fall back to access_control:users
        local email=$($REDIS_CLI GET "cache:auth:$user" 2>/dev/null)
        if [ -n "$email" ] && [ "$email" != "(nil)" ]; then
            priority=$($REDIS_CLI HGET "access_control:users" "$email" 2>/dev/null)
        fi
        if [ "$priority" = "(nil)" ] || [ -z "$priority" ]; then
            priority="external"
        fi
    fi
    
    # Superusers have no limit
    if [ "$priority" = "superuser" ]; then
        echo "-1"
        return
    fi
    
    # Get the rate limit for this model/priority
    local limit=$($REDIS_CLI GET "rate_limits:$model:$priority" 2>/dev/null)
    if [ -n "$limit" ] && [ "$limit" != "(nil)" ]; then
        echo "$limit"
    else
        # Try default priority
        limit=$($REDIS_CLI GET "rate_limits:$model:external" 2>/dev/null)
        if [ -n "$limit" ] && [ "$limit" != "(nil)" ]; then
            echo "$limit"
        else
            echo "0"
        fi
    fi
}

# Get throttled users — ALL users actually being throttled:
#   1. Users with active throttle:* keys (punished/blocked)
#   2. Users at or above their rate limit who are being blocked
get_throttled_users() {
    local limit="${1:-50}"
    local seen_users=""
    
    # Phase 1: Users with active throttle keys (explicitly blocked/punished)
    $REDIS_CLI KEYS "throttle:*" 2>/dev/null | while read -r key; do
        local expiry=$($REDIS_CLI GET "$key" 2>/dev/null)
        if [ -n "$expiry" ] && [ "$expiry" != "(nil)" ]; then
            local remaining=$(($expiry - $(date +%s)))
            if [ $remaining -gt 0 ]; then
                local user=$(echo "$key" | sed 's/throttle://g' | cut -d':' -f1)
                local email=$($REDIS_CLI GET "cache:auth:$user" 2>/dev/null)
                [ -z "$email" ] || [ "$email" = "(nil)" ] && email=""
                local model=$(echo "$key" | rev | cut -d':' -f1 | rev)
                echo "$user|$model|$remaining|$email|throttle_key"
            fi
        fi
    done
    
    # Phase 2: Users whose current rate count >= their limit
    # This catches users hitting rate limits even without a throttle:* key
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | while read -r key; do
        local raw=${key#rate:}
        # Extract user and model from rate:user:model:...key or rate:user:model
        local user=$(echo "$raw" | cut -d':' -f1)
        # Model is everything except the last field
        local model=$(echo "$raw" | rev | cut -d':' -f2- | rev)
        local current=$($REDIS_CLI GET "$key" 2>/dev/null)
        
        # Skip if not a number
        [[ "$current" =~ ^[0-9]+$ ]] || continue
        
        local limit=$(get_rate_limit "$user" "$model")
        # limit=-1 means unlimited (superuser)
        [ "$limit" = "-1" ] && continue
        
        # Check if user is at or above limit
        if [ -n "$current" ] && [ "$current" != "(nil)" ] && [ "$current" -ge "$limit" ] && [ "$limit" -gt 0 ]; then
            # Deduplicate by user (show once even if multiple models)
            echo "$user|$model|$current|$limit|rate_limited"
        fi
    done | sort -u -t'|' -k1,1 | head -n "$limit" | while IFS='|' read -r user model current limit reason; do
        local email=$($REDIS_CLI GET "cache:auth:$user" 2>/dev/null)
        [ -z "$email" ] || [ "$email" = "(nil)" ] && email=""
        # For rate-limited users without throttle keys, we don't know exact expiry
        # so we show "rate_limit" as the remaining info
        echo "$user|$model|$current|$email|$reason"
    done
}

# Get active models
get_active_models() {
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | sed 's/rate:[^:]*://g' | rev | cut -d':' -f2- | rev | sort -u | while read -r model; do
        if [ -n "$model" ] && [ "$model" != "default" ]; then
            local count=$($REDIS_CLI KEYS "rate:*:$model:*" 2>/dev/null | wc -l | tr -d ' ')
            echo "$model|$count"
        fi
    done | sort -t'|' -k2 -nr
}

# Get user priority from cache
get_user_priority() {
    local token="$1"
    local result=$($REDIS_CLI GET "user:priority:$token" 2>/dev/null)
    if [ "$result" = "(nil)" ] || [ -z "$result" ]; then
        echo "unknown"
    else
        echo "$result"
    fi
}

# Quick summary
cmd_summary() {
    print_header "Cluster Usage Summary"
    
    local timestamp=$(date)
    echo "Timestamp: $timestamp"
    echo ""
    
    # Active users count
    local rate_keys=$($REDIS_CLI KEYS "rate:*" 2>/dev/null | wc -l | tr -d ' ')
    local active_users=$rate_keys
    echo -e "Active users (requests in last window): ${GREEN}$active_users${NC}"
    
    # Throttled users count
    local throttled_count=$(get_throttled_users 9999 | wc -l | tr -d ' ')
    if [ -z "$throttled_count" ]; then
        throttled_count=0
    fi
    echo -e "Throttled users: ${RED}$throttled_count${NC}"
    echo "  (run: ./monitor-cluster.sh throttled  for full detail)"
    echo ""

    # Blacklisted + punished count
    local blacklisted_punished=$($REDIS_CLI HGETALL "access_control:users" 2>/dev/null | grep -E "blacklisted|punished" | wc -l | tr -d ' ')
    if [ -z "$blacklisted_punished" ]; then
        blacklisted_punished=0
    fi
    echo -e "Banned/Punished users: ${RED}$blacklisted_punished${NC}"

    # Show banned/punished users if any
    if [ "$blacklisted_punished" -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Banned/Punished users:${NC}"
        $REDIS_CLI HGETALL "access_control:users" 2>/dev/null | paste - - | while read -r email status; do
            if [ "$status" = "blacklisted" ] || [ "$status" = "punished" ]; then
                echo "  - $email ($status)"
            fi
        done
    fi
    
    # Models used
    local model_count=$(get_active_models | wc -l | tr -d ' ')
    if [ -z "$model_count" ]; then
        model_count=0
    fi
    echo -e "Models in use: ${BLUE}$model_count${NC}"
    
    echo ""
    print_header "Top 10 Users by Request Count"
    
    # Top users by request count
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | while read -r key; do
        local user=$(echo "$key" | sed 's/rate://g' | rev | cut -d':' -f3- | rev | cut -d':' -f1)
        local count=$($REDIS_CLI GET "$key" 2>/dev/null)
        if [ -n "$count" ] && [ "$count" != "(nil)" ]; then
            echo "$user $count"
        fi
    done | awk '{sum[$1]+=$2} END {for (user in sum) print user, sum[user]}' | sort -t' ' -k2 -nr | head -n 10 | while read -r user count; do
        local email=$(get_user_email "$user")
        if [ -n "$email" ] && [ "$email" != "(nil)" ]; then
            local status=$(get_user_status "$email")
            local email_display=$(printf "%-50s" "$email")
            echo -e "  ${BLUE}${email_display}${NC} | Status: $(format_status "$status") | Count: $count"
        fi
    done
    
    echo ""
    print_header "Active Models"
    
    get_active_models | head -n 10 | while IFS='|' read -r model count; do
        echo -e "  ${BLUE}$model${NC}: $count rate keys"
    done
}

# Active users list
cmd_active_users() {
    local limit="${1:-50}"
    
    print_header "Active Users (limit: $limit)"
    
    local email_width=50
    local status_width=12
    local count_width=10
    local model_width=60
    
    printf "${BLUE}%-*s${NC} | %-*s | %-*s | %-*s${NC}\n" "$email_width" "Email" "$status_width" "Status" "$count_width" "Count" "$model_width" "Model"
    printf "%$((email_width + status_width + count_width + model_width + 9))s\n" | tr ' ' '-'
    
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | sed 's/rate://g' | while read -r key; do
        local user=$(echo "$key" | cut -d':' -f1)
        local model=$(echo "$key" | cut -d':' -f2- | rev | cut -d':' -f2- | rev)
        local count=$($REDIS_CLI GET "rate:$key" 2>/dev/null)
        if [ -n "$count" ] && [ "$count" != "(nil)" ]; then
            echo "$user|$model|$count"
        fi
    done | sort -u | head -n "$limit" | while IFS='|' read -r user model count; do
        local email=$(get_user_email "$user")
        if [ -n "$email" ] && [ "$email" != "(nil)" ]; then
            local status=$(get_user_status "$email")
            local email_display=$(printf "%-${email_width}s" "$email")
            local status_formatted=$(format_status "$status")
            local status_padded=$(printf "%-${status_width}s" "$(echo "$status_formatted" | sed 's/\x1b\[[0-9;]*m//g')")
            printf "${BLUE}%-${email_width}s${NC} | %s | %-${count_width}s | %s\n" "$email_display" "$status_padded" "$count" "$(printf "%-${model_width}s" "$model")"
        fi
    done
}

# Rate limits display
cmd_rate_limits() {
    local limit="${1:-50}"
    local format="${2:-text}"
    
    print_header "Rate Limit Status"
    
    local email_width=50
    local status_width=12
    local count_width=10
    local model_width=60
    
    printf "${BLUE}%-*s${NC} | %-*s | %-*s | %-*s${NC}\n" "$email_width" "Email" "$status_width" "Status" "$count_width" "Count" "$model_width" "Model"
    printf "%$((email_width + status_width + count_width + model_width + 9))s\n" | tr ' ' '-'
    
    $REDIS_CLI KEYS "rate:*" 2>/dev/null | while read -r key; do
        local user=$(echo "$key" | sed 's/rate://g' | cut -d':' -f1)
        local model=$(echo "$key" | cut -d':' -f2- | rev | cut -d':' -f2- | rev)
        local count=$($REDIS_CLI GET "rate:$key" 2>/dev/null)
        if [ -n "$count" ] && [ "$count" != "(nil)" ]; then
            echo "$user|$model|$count"
        fi
    done | sort -u | head -n "$limit" | while IFS='|' read -r user model count; do
        local email=$(get_user_email "$user")
        if [ -n "$email" ] && [ "$email" != "(nil)" ]; then
            local status=$(get_user_status "$email")
            local throttle_count=$($REDIS_CLI GET "throttle_count:$user:$model" 2>/dev/null)
            local email_display=$(printf "%-${email_width}s" "$email")
            local status_formatted=$(format_status "$status")
            local status_padded=$(printf "%-${status_width}s" "$(echo "$status_formatted" | sed 's/\x1b\[[0-9;]*m//g')")
            
            if [ "$format" = "json" ]; then
                echo "{\"email\":\"${email}\",\"status\":\"$status\",\"model\":\"$model\",\"count\":$count,\"throttle_count\":${throttle_count:-0}}"
            else
    printf "${BLUE}%-${email_width}s${NC} | %s | %-${count_width}s | %s\n" "$email_display" "$status_padded" "$count" "$(printf "%-${model_width}s" "$model")"
                if [ -n "$throttle_count" ] && [ "$throttle_count" != "(nil)" ] && [ "$throttle_count" -gt 0 ]; then
                    echo -e "  ${YELLOW}Throttle violations:${NC} $throttle_count"
                fi
            fi
        fi
    done
    
    if [ "$format" = "json" ]; then
        echo "]"
    fi
}

cmd_throttled() {
    local limit="${1:-50}"
    
    print_header "Actually Throttled Users"
    echo "Includes: (1) punished users with active throttle keys, (2) users at their rate limit"
    echo ""
    
    local throttled=$(get_throttled_users "$limit")
    
    if [ -z "$throttled" ]; then
        echo "No throttled users"
        return
    fi
    
    local email_width=50
    local status_width=12
    local count_width=20
    local model_width=40
    
    printf "${BLUE}%-*s${NC} | %-*s | %-*s | %-*s | %s${NC}\n" "$email_width" "Email" "$status_width" "Status" "$count_width" "Remaining" "$model_width" "Model" "Reason"
    printf "%$((email_width + status_width + count_width + model_width + 12))s\n" | tr ' ' '-'
    
    echo "$throttled" | while IFS='|' read -r user model remaining email reason; do
        if [ -z "$email" ]; then
            email=$(get_user_email "$user")
        fi
        if [ -n "$email" ] && [ "$email" != "(nil)" ]; then
            local status=$(get_user_status "$email")
            local violations=$($REDIS_CLI GET "throttle_count:$user:$model" 2>/dev/null)
            local email_display=$(printf "%-${email_width}s" "$email")
            local status_formatted=$(format_status "$status")
            local status_padded=$(printf "%-${status_width}s" "$(echo "$status_formatted" | sed 's/\x1b\[[0-9;]*m//g')")
            
            # Format the remaining/info column based on reason
            if [ "$reason" = "throttle_key" ]; then
                local remaining_display=$(printf "%-${count_width}s" "${remaining}s remaining")
            else
                # Rate limited: remaining field actually has current count, limit is in field 4
                local limit_val=$(echo "$throttled" | grep "^$user|" | head -1 | cut -d'|' -f4)
                local remaining_display=$(printf "%-${count_width}s" "count=$remaining / limit=$limit_val")
            fi
            
            # Reason display
            case "$reason" in
                throttle_key)
                    local reason_display="${RED}blocked/punished${NC}"
                    ;;
                rate_limited)
                    local reason_display="${YELLOW}rate limit hit${NC}"
                    ;;
                *)
                    local reason_display="$reason"
                    ;;
            esac
            
            printf "${BLUE}%-${email_width}s${NC} | %s | %s | %s | %s\n" \
                "$email_display" \
                "$status_padded" \
                "$remaining_display" \
                "$(printf "%-${model_width}s" "$model")" \
                "$reason_display"
            
            if [ -n "$violations" ] && [ "$violations" != "(nil)" ]; then
                echo -e "  ${YELLOW}Throttle violations:${NC} $violations"
            fi
            echo ""
        fi
    done
}

# Models usage
cmd_models() {
    print_header "Active Models"
    
    echo "Model | Request Keys | Priority Distribution"
    echo "-------------------------------------------"
    
    get_active_models | while read -r model count; do
        if [ -n "$model" ]; then
            echo -e "${BLUE}$model${NC} | $count keys"
            
            # Show rate limit configuration
            # echo "  Rate limits:"
            # for priority in superuser elevated internal external punished; do
            #     case "$priority" in
            #         superuser)
            #             echo "    - $priority: unlimited"
            #             ;;
            #         *)
            #             echo "    - $priority: variable (configured per model)"
            #             ;;
            #     esac
            # done
            # echo ""
        fi
    done
}

# Banned/blacklisted users
cmd_banned_users() {
    local limit="${1:-50}"
    
    print_header "Banned (Blacklisted) Users"
    
    # Get all users from access_control:users hash
    local users=$($REDIS_CLI HGETALL "access_control:users" 2>/dev/null)
    
    if [ -z "$users" ]; then
        echo "No banned users found"
        return
    fi
    
    local email_width=50
    local status_width=12
    
    printf "${BLUE}%-*s${NC} | %-*s\n" "$email_width" "Email" "$status_width" "Status"
    printf "%$((email_width + status_width + 3))s\n" | tr ' ' '-'
    
    # Parse HGETALL output (alternating key-value pairs)
    echo "$users" | paste - - | while read -r email status; do
        if [ "$status" = "blacklisted" ] || [ "$status" = "punished" ]; then
            local email_display=$(printf "%-${email_width}s" "$email")
            local status_formatted=$(format_status "$status")
            printf "%s | %s\n" "$email_display" "$status_formatted"
        fi
    done | head -n "$limit"
}

# Full report
cmd_all() {
    local limit="${1:-50}"
    
    cmd_summary
    echo ""
    cmd_active_users "$limit"
    echo ""
    cmd_throttled "$limit"
    echo ""
    cmd_banned_users "$limit"
    echo ""
    # cmd_models
}

# Parse arguments
COMMAND="summary"
LIMIT=50
FORMAT="text"
FORCE_PORT_FORWARD=false

while [ $# -gt 0 ]; do
    case "$1" in
        summary|active-users|rate-limits|throttled|models|all|help)
            COMMAND="$1"
            shift
            ;;
        --limit)
            shift
            LIMIT="${1:-50}"
            shift
            ;;
        --format)
            shift
            FORMAT="${1:-text}"
            shift
            ;;
        --force-port-forward)
            FORCE_PORT_FORWARD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Auto-start port-forward if not running
start_port_forward() {
    echo "Starting port-forward for Redis..."
    kubectl port-forward svc/redis-service 6379:6379 -n app-ns > /tmp/pf-redis.log 2>&1 &
    PF_PID=$!
    echo "Port-forward started with PID $PF_PID"
    
    # Wait for port to be ready (max 10 seconds)
    for i in {1..20}; do
        sleep 0.5
        if lsof -i :$REDIS_PORT >/dev/null 2>&1; then
            echo "Port-forward is ready"
            return 0
        fi
    done
    
    echo "Error: Port-forward failed to start. Check /tmp/pf-redis.log"
    cat /tmp/pf-redis.log 2>/dev/null
    return 1
}

# Validate Redis connection
if ! $REDIS_CLI ping 2>/dev/null | grep -q PONG; then
    echo -e "${RED}Error: Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT${NC}"
    
    # Check if port-forward is running
    if ! lsof -i :$REDIS_PORT >/dev/null 2>&1; then
        echo "Port-forward is not running. Attempting to start it automatically..."
        if start_port_forward; then
            # Give Redis a moment to accept connections after port-forward starts
            sleep 1
            if $REDIS_CLI ping 2>/dev/null | grep -q PONG; then
                echo -e "${GREEN}Successfully connected to Redis!${NC}"
            else
                echo -e "${RED}Port-forward started but Redis still not responding.${NC}"
                echo ""
                echo "Diagnostics:"
                echo "  - Testing Redis pod directly:"
                kubectl exec -n app-ns -i $(kubectl get pod -n app-ns -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) -- redis-cli ping 2>/dev/null
                echo ""
                exit 1
            fi
        else
            exit 1
        fi
    else
        # Port-forward IS running but we still can't connect
        if [ "$FORCE_PORT_FORWARD" = true ]; then
            echo "Port-forward IS running but --force-port-forward specified. Restarting..."
            # Kill existing port-forward
            lsof -ti :$REDIS_PORT | xargs -r kill 2>/dev/null
            sleep 1
            if start_port_forward; then
                sleep 1
                if $REDIS_CLI ping 2>/dev/null | grep -q PONG; then
                    echo -e "${GREEN}Successfully connected to Redis after restart!${NC}"
                else
                    echo -e "${RED}Port-forward restarted but Redis still not responding.${NC}"
                    echo ""
                    echo "Diagnostics:"
                    echo "  - Testing Redis pod directly:"
                    kubectl exec -n app-ns -i $(kubectl get pod -n app-ns -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) -- redis-cli ping 2>/dev/null
                    echo ""
                    exit 1
                fi
            else
                exit 1
            fi
        else
            echo "Port-forward IS running (but connection still failing)"
            echo ""
            echo "Diagnostics:"
            echo "  - Testing Redis pod directly:"
            kubectl exec -n app-ns -i $(kubectl get pod -n app-ns -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) -- redis-cli ping 2>/dev/null
            echo ""
            echo "Hint: Use --force-port-forward to restart the port-forward"
            exit 1
        fi
    fi
fi

# Execute command
case "$COMMAND" in
    summary)
        cmd_summary
        ;;
    active-users)
        cmd_active_users "$LIMIT"
        ;;
    rate-limits)
        cmd_rate_limits "$LIMIT" "$FORMAT"
        ;;
    throttled)
        cmd_throttled "$LIMIT"
        ;;
    banned-users)
        cmd_banned_users "$LIMIT"
        ;;
    models)
        cmd_models
        ;;
    all)
        cmd_all "$LIMIT"
        ;;
    help)
        usage
        ;;
esac
