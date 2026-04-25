# Helm Deployment

```bash
# Install
helm install rag-financial helm/rag-financial/ \
  --set secrets.openaiApiKey=$OPENAI_API_KEY \
  --set config.environment=production \
  --namespace rag-prod --create-namespace

# Upgrade
helm upgrade rag-financial helm/rag-financial/ \
  --set image.tag=2.1.0 -n rag-prod

# Check status
helm status rag-financial -n rag-prod
```

Key values: `replicaCount`, `resources`, `autoscaling`, `persistence`, `redis.enabled`.
See `helm/rag-financial/values.yaml` for all options.
