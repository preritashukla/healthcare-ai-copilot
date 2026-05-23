from fastapi import APIRouter, HTTPException, Query
from app.services.analytics_service import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.get("/vitals")
async def get_vitals_telemetry(patient_id: str = Query("Marcus Vance", description="Name of the patient to query vitals for")):
    """Retrieves patient vital sign telemetry data streams (HR, Temp) and aggregated recovery statistics."""
    try:
        summary = analytics_service.get_patient_vitals(patient_id)
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process telemetry: {str(e)}")
