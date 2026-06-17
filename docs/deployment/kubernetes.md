# Kubernetes Deployment

```bash
# Apply base manifests
kubectl apply -k k8s/overlays/prod/

# Verify
kubectl get pods -n rag-prod
kubectl logs -f deploy/rag-financial -n rag-prod

# Scale manually
kubectl scale deploy/rag-financial --replicas=4 -n rag-prod
```

The HPA auto-scales 2-10 replicas based on CPU (70%) and memory (80%).
PodDisruptionBudget ensures at least 1 replica stays up during rollouts.
