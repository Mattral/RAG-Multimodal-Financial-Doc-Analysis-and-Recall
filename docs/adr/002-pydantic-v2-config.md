# ADR 002: Pydantic v2 BaseSettings for Configuration

**Status:** Accepted  
**Date:** 2024-06  
**Deciders:** Core team

## Context

The original system used scattered `os.getenv()` calls with no validation, no type coercion, and no structured defaults. This made configuration fragile, untestable, and hard to document.

## Decision

Use `pydantic-settings` with `BaseSettings` and nested `BaseModel` sub-configs. All settings are validated at startup. Nested configs use `env_nested_delimiter="__"` (e.g., `LLM_CONFIG__MODEL=gpt-4o`).

## Rationale

- **Single source of truth:** Every setting has a type, a default, and a description in one place.
- **Env-var override at any nesting level:** Works naturally with Kubernetes `ConfigMap` + `Secret` env injection.
- **Immutable sub-configs (`frozen=True`):** Prevents accidental runtime mutation of shared config objects.
- **`SecretStr` for keys:** API keys are never logged or serialized in plaintext.
- **Testable:** `reset_config()` singleton reset makes unit tests deterministic.

## Consequences

- **Positive:** Config bugs caught at startup, not at runtime. Full IDE autocomplete. Clean `--help` in CLI.
- **Negative:** Pydantic v2 is a breaking change from v1. Requires all models to use `model_dump()` instead of `.dict()`. Migration cost was ~2 days.

## Alternatives Considered

- **Dynaconf (rejected):** More features but heavier dependency, less Pythonic type integration.
- **Plain dataclasses (rejected):** No env-var loading, no validation, no `SecretStr`.
