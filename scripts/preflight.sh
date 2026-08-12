set -uo pipefail

fail=0
ok()  { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad() { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail+1)); }
warn(){ printf '  \033[33m!\033[0m %s\n' "$1"; }
hd()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

hd "1. Docker"
if command -v docker >/dev/null 2>&1; then
  ok "docker: $(docker --version | cut -d, -f1)"
  if docker compose version >/dev/null 2>&1; then
    ok "compose: $(docker compose version --short 2>/dev/null)"
  else
    bad "'docker compose' not available (v1 'docker-compose' is not supported here)"
  fi
  docker info >/dev/null 2>&1 && ok "daemon reachable" || bad "docker daemon not running"
else
  bad "docker not installed"
fi

hd "2. Ports the stack publishes"

check_port() {
  local port=$1 name=$2
  if command -v ss >/dev/null 2>&1; then
    listening=$(ss -ltn 2>/dev/null | awk -v p=":$port\$" '$4 ~ p {print $4}')
  else
    listening=$(netstat -ltn 2>/dev/null | awk -v p=":$port\$" '$4 ~ p {print $4}')
  fi
  if [ -n "${listening:-}" ]; then
    warn "$port ($name) ALREADY IN USE — you are running $name natively"
    echo "$port" >> /tmp/_busy_ports
  else
    ok "$port ($name) free"
  fi
}
rm -f /tmp/_busy_ports
check_port 5432  postgres
check_port 6379  redis
check_port 8123  clickhouse-http
check_port 9000  clickhouse-native
check_port 19092 redpanda-kafka
check_port 9644  redpanda-admin
check_port 8000  gateway

hd "3. Image tags actually resolve"
for img in \
  "postgres:16-alpine" \
  "redis:7-alpine" \
  "redpandadata/redpanda:latest" \
  "clickhouse/clickhouse-server:latest" \
  "python:3.12-slim"
do
  if docker manifest inspect "$img" >/dev/null 2>&1; then
    ok "$img"
  else
    bad "$img — cannot resolve (try: docker search / check the tag list)"
  fi
done

hd "4. Files bootstrap needs"
for f in sql/schema.sql sql/kafka_sink.sql sql/002_advertiser_side.sql \
         requirements.txt Dockerfile docker-compose.yml; do
  [ -f "$f" ] && ok "$f" || bad "$f missing"
done
[ -f .env ] && ok ".env present" || warn ".env missing — run: cp .env.example .env"
[ -d models ] && ok "models/ exists (bind mount target)" \
              || warn "models/ missing — run: mkdir -p models"

hd "Result"
if [ -s /tmp/_busy_ports ]; then
  cat <<'NOTE'
  Some ports are already in use. Pick ONE:

  A) Use the containers (recommended for a clean run)
     Stop your native services first:
       brew services stop postgresql redis        # macOS
       sudo systemctl stop postgresql redis        # Linux
     Then: make up

  B) Keep your native services and only containerise what you lack
     cp docker-compose.host.yml docker-compose.override.yml
     Edit it to comment out the services you already run, then: make up
     Your native Postgres must have an `adplatform` database and role.
NOTE
fi

if [ "$fail" -gt 0 ]; then
  printf '\n\033[31m%d blocking problem(s)\033[0m — fix these before make up\n' "$fail"
  exit 1
fi
printf '\n\033[32mPreflight clean.\033[0m Next: make up\n'
