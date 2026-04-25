# Docker Deployment

## Single-service (minimal)
```bash
docker run -e OPENAI_API_KEY=sk-... \
  -p 8000:8000 -p 8001:8001 \
  ghcr.io/your-org/rag-financial-multimodal:2.0.0
```

## Full stack with docker compose
```bash
cp .env.example .env      # set OPENAI_API_KEY
docker compose up -d      # API + Redis

# With full observability (Prometheus, Grafana, Jaeger)
docker compose --profile observability up -d
```

## Build locally
```bash
docker build -t rag-financial-multimodal:local .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 rag-financial-multimodal:local
```

## Health check
```bash
curl http://localhost:8000/healthz   # liveness
curl http://localhost:8000/readyz    # readiness (checks all components)
```

## Ports
| Port | Service |
|---|---|
| 8000 | REST API + OpenAPI docs |
| 8001 | Prometheus metrics |
| 3000 | Grafana (observability profile) |
| 9090 | Prometheus (observability profile) |
| 16686 | Jaeger UI (observability profile) |

## Resource recommendations
| Scale | CPU | Memory |
|---|---|---|
| Dev/local | 0.5 CPU | 1 GB |
| Small prod | 2 CPU | 4 GB |
| Enterprise | 4+ CPU | 8+ GB |
