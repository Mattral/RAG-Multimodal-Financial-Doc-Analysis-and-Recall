# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | ✅ Active support  |
| 1.x     | ❌ End of life     |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues by emailing: **security@your-org.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if known)

We will acknowledge receipt within **48 hours** and provide a resolution timeline within **7 days**.

## Security Model

### API Authentication
- All API endpoints (except `/health`, `/healthz`, `/readyz`) require `X-API-Key` header.
- Keys are compared via constant-time SHA-256 hash comparison to prevent timing attacks.
- Set `RAG_API_MASTER_KEY` in environment — never hardcode in source.

### PII Handling
- All ingested text is passed through Presidio PII detection before embedding.
- PII entities (SSN, IBAN, account numbers, CUSIP, ISIN) are replaced with typed tokens.
- Raw PII never reaches the vector store.

### Secrets Management
- All API keys are loaded from environment variables via `pydantic-settings`.
- Keys are stored as `SecretStr` — never logged or serialized in plaintext.
- The `.env` file is in `.gitignore` — never commit it.

### Audit Trail
- Every ingest and query event is written to an append-only JSONL file.
- Each audit record includes a SHA-256 content hash for tamper detection.
- Queries are logged with hashed query text (not raw) for privacy.

### Sandboxed Code Execution
- Program-of-Thought code is AST-validated against an allowlist before execution.
- Blocked: `import`, `exec`, `eval`, `open`, `__builtins__`, `os`, `sys`, network calls.
- Hard 5-second execution timeout prevents denial-of-service via infinite loops.

### Prompt Injection
- Query-time prompt injection detection blocks known attack patterns.
- Detected injections are logged and return a 400 error — the pipeline is not executed.

### Container Security
- Docker image runs as non-root user (UID 1001).
- Multi-stage build minimizes attack surface (no build tools in runtime image).
- Trivy vulnerability scanning runs on every CI build.

### Data Deletion (GDPR/CCPA)
- The `DELETE /api/v1/documents/{doc_id}` endpoint triggers vector store deletion.
- All deletion events are logged to the audit trail with tenant ID and reason.
