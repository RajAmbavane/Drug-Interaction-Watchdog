import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas.drug import IntakeImageRequest, IntakeResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/image", response_model=IntakeResponse)
async def intake_image(
    req: IntakeImageRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Submit a base64-encoded image of a prescription, pill packet, or lab report.
    Returns extracted drug names, medications, and lab values.
    """
    try:
        from agents.vision_intake_agent import VisionIntakeAgent

        agent = VisionIntakeAgent()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: agent.intake_from_b64(req.image_b64, req.mode)
        )

        return IntakeResponse(
            mode                = result.mode.value,
            success             = result.success,
            confidence          = result.confidence,
            drugs               = result.drugs or [],
            medications         = result.medications or [],
            lab_values          = result.lab_values if result.lab_values else None,
            prescriber          = result.prescriber,
            prescription_date   = result.prescription_date,
            patient_name_on_rx  = result.patient_name_on_rx,
            report_type         = result.report_type,
            patient_info        = result.patient_info or {},
            error               = result.error or "",
        )

    except Exception as exc:
        log.exception("Vision intake failed")
        raise HTTPException(
            status_code=500,
            detail="Image intake failed. Please try again with a clearer image.",
        )
