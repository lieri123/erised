.PHONY: preflight up down logs verify seed reset bootstrap ps shell-ch shell-pg train publish model-status

preflight:         ## check docker, ports and image tags BEFORE starting
	./scripts/preflight.sh

up:                ## build and start the whole stack
	docker compose up -d --build
	@echo "waiting for gateway..."
	@until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 2; done
	@$(MAKE) --no-print-directory verify

verify:            ## run the step-2 acceptance checks
	./scripts/verify_stack.sh

bootstrap:         ## re-run schema + topic creation (idempotent)
	docker compose up bootstrap

seed:              ## fill advertisers/campaigns/ads so bids can fill
	docker compose run --rm bootstrap python -m scripts.seed_inventory --advertisers 12

logs:              ## follow gateway logs
	docker compose logs -f gateway

ps:
	docker compose ps -a

down:              ## stop, keep data
	docker compose down

reset:             ## stop and DELETE all volumes
	docker compose down -v

shell-ch:
	docker compose exec clickhouse clickhouse-client

shell-pg:
	docker compose exec postgres psql -U adplatform -d adplatform

train:             ## train a CTR model from ClickHouse into models/
	docker compose run --rm bootstrap python -m adplatform.ml.train_ctr --days 30 --out /app/models

publish:           ## upload models/current to S3 and flip the pointer
	python -m scripts.publish_model models/current

model-status:      ## show which model version is currently live
	python -m scripts.publish_model --show
