# Security & Compliance

- **API Auth**: X-API-Key header, constant-time SHA-256 comparison
- **PII Redaction**: Presidio + CUSIP/ISIN/account number patterns before embedding
- **Guardrails**: Numeric grounding + prompt injection detection on every query
- **Audit Trail**: Append-only JSONL with SHA-256 tamper hash
- **Container**: Non-root user UID 1001, minimal runtime image
- **GDPR/CCPA**: Soft-delete endpoint + deletion audit events

See [SECURITY.md](../SECURITY.md) for the full policy.
