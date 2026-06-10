# Cluster Usage Monitor

A comprehensive monitoring script for tracking cluster usage, active users, rate limits, and throttling status. The script connects to Redis to fetch real-time usage data and provides detailed reports in both text and JSON formats.

## Features

- **Active users tracking** - List all users currently making requests
- **Rate limit monitoring** - Show current rate limit usage per user and model
- **Throttling detection** - Identify users who are currently throttled, including both punished users with active throttle keys AND users hitting their rate limit
- **Model usage analytics** - Track which models are being used most
- **User status classification** - Superuser, elevated, internal, external, punished, blacklisted
- **Automatic port-forwarding** - Automatically starts `kubectl port-forward` if needed
- **Multiple output formats** - Text (colored) and JSON support

## Requirements

- `redis-cli` (Redis command-line client)
- `kubectl` with access to the cluster
- `lsof` (for port detection)

### Install redis-cli

- **macOS**: `brew install redis`
- **Ubuntu/Debian**: `sudo apt-get install redis-tools`
- **CentOS/RHEL**: `sudo yum install redis`
- **Docker**: `docker run -it --rm redis redis-cli --help`

## Quick Start

```bash
# Run the script (auto-starts port-forward if needed)
./monitor-cluster.sh

# Get a quick summary
./monitor-cluster.sh summary

# View active users
./monitor-cluster.sh active-users --limit 10

# Check rate limits in JSON format
./monitor-cluster.sh rate-limits --format json

# Full cluster report
./monitor-cluster.sh all
```

## Usage

```bash
./monitor-cluster.sh [command] [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `summary` | Quick summary of cluster usage (default) |
| `active-users` | List all active users with their rights |
| `rate-limits` | Show current rate limit usage per user/model |
| `throttled` | List throttled users |
| `models` | Show which models are being used |
| `all` | Full cluster usage report |
| `help` | Show this help message |

### Options

| Option | Description |
|--------|-------------|
| `--limit N` | Limit output to N users (default: 50) |
| `--format json` | Output in JSON format |
| `--force-port-forward` | Restart the port-forward even if it's already running |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |

## Examples

```bash
# Get a quick summary (auto-starts port-forward if needed)
./monitor-cluster.sh

# View top 10 active users with their status
./monitor-cluster.sh active-users --limit 10

# Get rate limits in JSON format for further processing
./monitor-cluster.sh rate-limits --format json --limit 20

# See all throttled users
./monitor-cluster.sh throttled

# Get complete cluster usage report
./monitor-cluster.sh all

# Force restart the port-forward (useful if connection is stale)
./monitor-cluster.sh --force-port-forward

# Force restart and get full report
./monitor-cluster.sh all --force-port-forward
```

## Kubernetes Setup

The script handles port-forwarding automatically, but you can also manage it manually:

```bash
# Start port-forward manually
kubectl port-forward svc/redis-service 6379:6379 -n app-ns &

# Or explicitly start with the script
./monitor-cluster.sh
```

## User Status Types

The script classifies users based on their access level:

- **superuser** - Full access, unlimited rate limits (green)
- **elevated** - Enhanced privileges (blue)
- **internal** - Internal users from @fz-juelich.de (green)
- **external** - External users (yellow)
- **punished** - Users with reduced privileges (yellow)
- **blacklisted** - Blocked users (red)

## Architecture

### Port-Forwarding Logic

The script includes intelligent port-forwarding:

1. **Check Redis connectivity** - First attempt to ping Redis
2. **Auto-start if needed** - If connection fails and port-forward isn't running, start it automatically
3. **Wait for readiness** - Polls for 10 seconds to ensure the port is listening
4. **Fallback diagnostics** - If still failing, tests Redis directly in the pod
5. **Force restart option** - Use `--force-port-forward` to kill and restart

### Error Handling

Common error scenarios:

| Error | Cause | Solution |
|-------|-------|----------|
| `redis-cli not installed` | Missing Redis CLI | Install with `brew install redis` or `apt-get install redis-tools` |
| `Port-forward is not running` | Port-forward hasn't been started | Script auto-starts, or run manually with `kubectl port-forward` |
| `Cannot connect to Redis` | Redis not responding in pod | Check pod status: `kubectl get pod -n app-ns -l app=redis` |
| `Port already in use` | Stale port-forward process | Use `--force-port-forward` flag |

## Output Examples

### Summary Output
```
========================================
Cluster Usage Summary
========================================
Timestamp: 2026-03-24 19:00:00

Active users (requests in last window): 23
Throttled users: 2
Blacklisted users: 1
Models in use: 5

========================================
Top 10 Users by Request Count
========================================
  user1@fz-juelich.de                                    | Status: superuser  | Count: 1250
  user2@external.com                                     | Status: external   | Count: 892
```

### Rate Limits (JSON)
```json
{
  "email": "user1@fz-juelich.de",
  "status": "superuser",
  "model": "gpt-4",
  "count": 1250,
  "throttle_count": 0
}
```

## Monitoring Tips

1. **Regular monitoring**: Add to your shell profile or create an alias:
   ```bash
   alias cluster-stats='./monitor-cluster.sh summary'
   ```

2. **Background monitoring**: Use cron to run periodically:
   ```bash
   # Add to crontab
   0 * * * * /path/to/monitor-cluster.sh summary > /var/log/cluster-stats-$(date +\%Y\%m\%d).log
   ```

3. **Alerting**: Combine with other tools to create alerts for unusual activity:
   ```bash
   # Alert if more than 100 throttled users
   ./monitor-cluster.sh throttled | wc -l | awk '{if($1 > 100) print "ALERT: Too many throttled users!"}'
   ```

## Troubleshooting

### Port-forward not starting

Check the log file:
```bash
cat /tmp/pf-redis.log
```

### Redis pod not responding

Check pod status:
```bash
kubectl get pod -n app-ns -l app=redis
kubectl logs -n app-ns -l app=redis
```

### Connection refused

Verify the service exists:
```bash
kubectl get svc redis-service -n app-ns
```

## Contributing

When contributing improvements:
1. Ensure `redis-cli` detection is robust
2. Add proper error handling for all connection scenarios
3. Update this documentation for new features
4. Test with both text and JSON output formats

## License

Internal use - Juelich Supercomputing Centre
