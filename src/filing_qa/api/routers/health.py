"""Health, liveness, and readiness probe endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", summary="Liveness probe")
async def liveness():
    return {"status": "alive"}


@router.get("/healthz", summary="K8s liveness probe")
async def k8s_liveness():
    return {"status": "ok"}


@router.get("/readyz", summary="K8s readiness probe")
async def k8s_readiness(request: Request):
    shutdown_state = getattr(request.app.state, "shutdown", None)
    if shutdown_state is not None and shutdown_state.is_shutting_down:
        # Flip unready the instant shutdown begins so the Service/Ingress
        # stops routing new traffic here within one probe interval — this
        # is what makes connection draining effective, independent of the
        # actual in-flight request count or pipeline health.
        return JSONResponse(
            status_code=503,
            content={
                "status": "shutting_down",
                "in_flight_requests": shutdown_state.in_flight_requests,
            },
        )
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "pipeline_not_initialized"}
        )
    try:
        health = await pipeline.health_check()
        status_code = 200 if health["status"] == "healthy" else 503
        return JSONResponse(status_code=status_code, content=health)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})
