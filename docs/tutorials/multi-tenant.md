# Multi-Tenant Setup

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-Tenant-ID: hedge_fund_a" -F "file=@goldman.pdf"
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-Tenant-ID: hedge_fund_b" \
  -d '{"query": "What was net income?"}'
curl http://localhost:8000/api/v1/tenants/hedge_fund_a/usage
```
Each tenant: isolated vector namespace, rate limits, token quotas, audit logs.
