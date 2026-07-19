from fastapi import APIRouter, HTTPException, status
from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.services.feedback_generator import feedback_generator
from app.utils.helpers import setup_logger

log = setup_logger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["Feedback"])

@router.post("/", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """
    Generate detailed feedback, career recommendations, learning resources,
    and priority actions based on a candidate's screening results.
    """
    try:
        # Generate feedback, incorporating LLM suggestions if configured
        feedback = await feedback_generator.generate_with_llm(
            screening=request.screening_result,
            focus_areas=request.focus_areas
        )
        return feedback
    except Exception as e:
        log.error(f"Error generating feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback generation error: {str(e)}"
        )
