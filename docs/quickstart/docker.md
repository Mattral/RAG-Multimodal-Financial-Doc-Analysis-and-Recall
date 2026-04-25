# Docker Quickstart

```bash
git clone https://github.com/your-org/rag-financial-multimodal
cd rag-financial-multimodal
cp .env.example .env   # set OPENAI_API_KEY
docker compose up -d
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@your_10k.pdf" -F "tenant_id=demo"
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"query": "What was gross margin?", "tenant_id": "demo"}'
```

With observability: `docker compose --profile observability up -d`
- Grafana: http://localhost:3000  Prometheus: http://localhost:9090  Jaeger: http://localhost:16686
