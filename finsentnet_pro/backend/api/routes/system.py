"""
FINSENT NET PRO - System Routes
Runtime settings and model profile management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/system", tags=["System"])
logger = logging.getLogger("finsent.routes.system")

_model_registry = None


def init(model_registry):
    global _model_registry
    _model_registry = model_registry


class ActiveModelRequest(BaseModel):
    profile_id: str = Field(..., min_length=3)


@router.get("/model-profiles")
async def get_model_profiles():
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not ready")
    return {
        "status": "ok",
        "model_registry": _model_registry.snapshot(),
    }


@router.post("/model-profiles/active")
async def set_active_model_profile(request: ActiveModelRequest):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not ready")

    updated = _model_registry.set_active(request.profile_id)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Unknown profile_id: {request.profile_id}")

    logger.info("Active model profile set to %s", request.profile_id)
    return {
        "status": "ok",
        "message": "Active model profile updated",
        "model_registry": _model_registry.snapshot(),
    }
