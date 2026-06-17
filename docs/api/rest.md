# REST API Reference

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ingest` | Ingest a PDF document |
| `POST` | `/api/v1/query` | Query ingested documents |
| `POST` | `/api/v1/tenants` | Register a new tenant |
| `GET` | `/api/v1/tenants/{id}/usage` | Get tenant usage stats |
| `GET` | `/healthz` | K8s liveness probe |
| `GET` | `/readyz` | K8s readiness probe |

## Authentication
All endpoints (except health probes) require: `X-API-Key: your-key`
Set tenant with: `X-Tenant-ID: your-tenant`

See interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)
