from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models.database import get_db, ChatRecord
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_engine import rag_engine
from app.utils.helpers import setup_logger, generate_session_id

log = setup_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chatbot"])

@router.post("/", response_model=ChatResponse)
def chat_with_bot(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Interact with the RAG-powered resume screening assistant.
    """
    try:
        session_id = request.session_id or generate_session_id()

        # Save user message to database
        user_rec = ChatRecord(
            session_id=session_id,
            role="user",
            content=request.message
        )
        db.add(user_rec)
        db.commit()

        # Generate response using RAG engine
        response_text, returned_session_id, sources = rag_engine.generate(
            query=request.message,
            screening_context=request.screening_context,
            session_id=session_id
        )

        # Save assistant response to database
        assistant_rec = ChatRecord(
            session_id=returned_session_id,
            role="assistant",
            content=response_text
        )
        db.add(assistant_rec)
        db.commit()

        return ChatResponse(
            response=response_text,
            session_id=returned_session_id,
            sources=sources
        )

    except Exception as e:
        log.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline error: {str(e)}"
        )

@router.get("/history/{session_id}", response_model=List[dict])
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieve message history for a given chat session.
    """
    try:
        records = db.query(ChatRecord).filter(
            ChatRecord.session_id == session_id
        ).order_by(ChatRecord.created_at.asc()).all()

        return [
            {"role": rec.role, "content": rec.content, "timestamp": rec.created_at}
            for rec in records
        ]
    except Exception as e:
        log.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"History retrieval error: {str(e)}"
        )

@router.delete("/history/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    """
    Clear message history for a given chat session.
    """
    try:
        db.query(ChatRecord).filter(ChatRecord.session_id == session_id).delete()
        db.commit()
        rag_engine.clear_session(session_id)
        return
    except Exception as e:
        log.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"History deletion error: {str(e)}"
        )
