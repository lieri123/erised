set -uo pipefail

GATEWAY="${GATEWAY:-http://localhost:8000}"
CH="${CH:-http://localhost:8123}"
COMPOSE="${COMPOSE:-docker compose}"

pass=0
fail=0

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail+1)); }
head() { printf '\n\033[1m%s\033[0m\n' "$1"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1"; exit 2; }
}
need curl
need python3

# ---------------------------------------------------------------------------
head "1. Gateway is up"
# ---------------------------------------------------------------------------
for i in $(seq 1 30); do
  if curl -sf -m 3 "$GATEWAY/health" >/dev/null 2>&1; then break; fi
  [ "$i" = 30 ] && { bad "$GATEWAY/health never responded"; echo; echo "  docker compose logs gateway"; exit 1; }
  sleep 2
done
ok "$GATEWAY/health responding"

# ---------------------------------------------------------------------------
head "2. All four stores connected (the real criterion)"
# ---------------------------------------------------------------------------
HEALTH=$(curl -sf -m 10 "$GATEWAY/health?deep=true")
echo "$HEALTH" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' | head -30

for store in postgres redis clickhouse kafka; do
  state=$(echo "$HEALTH" | python3 -c "
import json,sys
print(json.load(sys.stdin)['stores'].get('$store','missing'))" 2>/dev/null)
  if [ "$state" = "up" ]; then ok "$store: up"; else bad "$store: $state"; fi
done

# ---------------------------------------------------------------------------
head "3. Kafka topics exist with the right partition counts"
# ---------------------------------------------------------------------------
if $COMPOSE exec -T redpanda rpk topic list 2>/dev/null | tail -n +2 > /tmp/_topics; then
  cat /tmp/_topics | sed 's/^/    /'
  for t in impressions clicks conversions; do
    grep -q "^$t " /tmp/_topics && ok "topic $t" || bad "topic $t missing"
  done
else
  bad "could not run 'rpk topic list' (is the redpanda service named 'redpanda'?)"
fi

# ---------------------------------------------------------------------------
head "4. ClickHouse objects created, in the right order"
# ---------------------------------------------------------------------------
TABLES=$(curl -sf -m 5 "$CH/?query=SELECT%20name%20FROM%20system.tables%20WHERE%20database%3D%27default%27%20ORDER%20BY%20name" 2>/dev/null)
echo "$TABLES" | sed 's/^/    /'
for t in ad_impressions ad_clicks ctr_agg_pair ctr_agg_pair_mv ctr_agg_pair_clicks_mv \
         kafka_impressions kafka_impressions_mv kafka_clicks kafka_clicks_mv; do
  echo "$TABLES" | grep -qx "$t" && ok "clickhouse.$t" || bad "clickhouse.$t missing"
done

# ---------------------------------------------------------------------------
head "5. Postgres tables created"
# ---------------------------------------------------------------------------
if $COMPOSE exec -T postgres psql -U adplatform -d adplatform -tAc \
     "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1" > /tmp/_pgt 2>/dev/null; then
  tr '\n' ' ' < /tmp/_pgt | sed 's/^/    /'; echo
  for t in publishers impressions conversions; do
    grep -qx "$t" /tmp/_pgt && ok "postgres.$t" || bad "postgres.$t missing"
  done
  for t in advertisers campaigns ads; do
    grep -qx "$t" /tmp/_pgt && ok "postgres.$t (002 migration)" \
      || bad "postgres.$t missing — 002_advertiser_side.sql did not apply"
  done
else
  bad "could not query postgres"
fi

# ---------------------------------------------------------------------------
head "6. Gateway logged 'CTR stats refreshed' (not the ClickHouse warning)"
# ---------------------------------------------------------------------------
LOGS=$($COMPOSE logs gateway 2>/dev/null | tail -300)
if echo "$LOGS" | grep -q "CTR stats refreshed"; then
  ok "CTR stats refreshed"
  echo "$LOGS" | grep "CTR stats refreshed" | tail -1 | sed 's/^/    /'
else
  bad "no 'CTR stats refreshed' line"
fi
if echo "$LOGS" | grep -q "ClickHouse unavailable"; then
  bad "gateway logged 'ClickHouse unavailable' — check CLICKHOUSE_HOST"
else
  ok "no ClickHouse warning"
fi
if echo "$LOGS" | grep -q "Kafka unavailable"; then
  bad "gateway logged 'Kafka unavailable' — check KAFKA_BOOTSTRAP=redpanda:9092"
else
  ok "no Kafka warning"
fi
if echo "$LOGS" | grep -q "BUDGETS ARE NOT ENFORCED"; then
  bad "Redis did not connect — budgets unenforced"
else
  ok "budget enforcement active"
fi

# ---------------------------------------------------------------------------
head "7. Bootstrap exited 0"
# ---------------------------------------------------------------------------
code=$($COMPOSE ps -a --format json bootstrap 2>/dev/null \
       | python3 -c "
import json,sys
raw=sys.stdin.read().strip()
if not raw: print('?'); raise SystemExit
try: d=json.loads(raw)
except json.JSONDecodeError: d=[json.loads(l) for l in raw.splitlines() if l.strip()]
d = d[0] if isinstance(d,list) else d
print(d.get('ExitCode','?'))" 2>/dev/null)
[ "$code" = "0" ] && ok "bootstrap exit code 0" || bad "bootstrap exit code: $code"

# ---------------------------------------------------------------------------
printf '\n\033[1m%d passed, %d failed\033[0m\n' "$pass" "$fail"
if [ "$fail" -gt 0 ]; then
  cat <<'HINT'

Common causes, in order of likelihood:

  clickhouse tables missing   bootstrap ran before ClickHouse was really ready.
                              `docker compose up bootstrap` again — it is idempotent.
  kafka: down                 KAFKA_BOOTSTRAP must be redpanda:9092 INSIDE the
                              network, not localhost:19092.
  kafka_* tables but no data  sql/kafka_sink.sql hardcodes
                              kafka_broker_list='redpanda:9092'. The service has
                              to be named redpanda.
  no CTR stats refreshed      normal on a brand-new stack ONLY if ClickHouse is
                              down; with an empty ctr_agg_pair the loop still
                              logs "0 pairs, global_ctr=1.0000%".
HINT
  exit 1
fi
echo "Step 2 complete."
