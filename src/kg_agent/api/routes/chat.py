"""API routes for chat history management."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.logging import logger
from ...services.chat_history import (
    get_chat_history,
    ChatHistoryService,
    Conversation,
    ChatMessage,
)

router = APIRouter()


# ==================== Request/Response Models ====================

class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    title: Optional[str] = "New Chat"
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationUpdate(BaseModel):
    """Request to update a conversation."""
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageCreate(BaseModel):
    """Request to add a message to a conversation."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Response for a conversation."""
    id: str
    title: str
    summary: Optional[str]
    message_count: int
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    last_message_at: Optional[str]


class MessageResponse(BaseModel):
    """Response for a message."""
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: Dict[str, Any]
    created_at: str


# ==================== Conversation Endpoints ====================

@router.get("/conversations", response_model=Dict[str, Any])
async def list_conversations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None)
):
    """
    List all conversations with optional search and pagination.
    """
    try:
        chat_history = get_chat_history()
        conversations = chat_history.list_conversations(
            limit=limit,
            offset=offset,
            search=search
        )

        return {
            "status": "success",
            "conversations": [conv.to_dict() for conv in conversations],
            "count": len(conversations),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(request: ConversationCreate):
    """
    Create a new conversation.
    """
    try:
        chat_history = get_chat_history()
        conversation = chat_history.create_conversation(
            title=request.title or "New Chat",
            summary=request.summary,
            metadata=request.metadata
        )

        return {
            "status": "success",
            "conversation": conversation.to_dict()
        }

    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conv_id}", response_model=Dict[str, Any])
async def get_conversation(conv_id: str):
    """
    Get a specific conversation by ID.
    """
    try:
        chat_history = get_chat_history()
        conversation = chat_history.get_conversation(conv_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "status": "success",
            "conversation": conversation.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/conversations/{conv_id}", response_model=Dict[str, Any])
async def update_conversation(conv_id: str, request: ConversationUpdate):
    """
    Update a conversation's title, summary, or metadata.
    """
    try:
        chat_history = get_chat_history()

        # Check if exists
        existing = chat_history.get_conversation(conv_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation = chat_history.update_conversation(
            conv_id=conv_id,
            title=request.title,
            summary=request.summary,
            metadata=request.metadata
        )

        return {
            "status": "success",
            "conversation": conversation.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conv_id}", response_model=Dict[str, Any])
async def delete_conversation(conv_id: str):
    """
    Delete a conversation and all its messages.
    """
    try:
        chat_history = get_chat_history()
        deleted = chat_history.delete_conversation(conv_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "status": "success",
            "message": f"Conversation {conv_id} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Message Endpoints ====================

@router.get("/conversations/{conv_id}/messages", response_model=Dict[str, Any])
async def get_messages(
    conv_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("asc", regex="^(asc|desc)$")
):
    """
    Get all messages for a conversation.
    """
    try:
        chat_history = get_chat_history()

        # Check if conversation exists
        conversation = chat_history.get_conversation(conv_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = chat_history.get_messages(
            conversation_id=conv_id,
            limit=limit,
            offset=offset,
            order=order
        )

        return {
            "status": "success",
            "conversation_id": conv_id,
            "messages": [msg.to_dict() for msg in messages],
            "count": len(messages)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conv_id}/messages", response_model=Dict[str, Any])
async def add_message(conv_id: str, request: MessageCreate):
    """
    Add a message to a conversation.
    """
    try:
        chat_history = get_chat_history()

        # Check if conversation exists
        conversation = chat_history.get_conversation(conv_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Validate role
        valid_roles = ["user", "assistant", "system"]
        if request.role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Must be one of: {valid_roles}"
            )

        message = chat_history.add_message(
            conversation_id=conv_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata
        )

        # Auto-update conversation title from first user message
        if conversation.message_count == 0 and request.role == "user":
            new_title = chat_history.generate_title_from_message(request.content)
            chat_history.update_conversation(conv_id, title=new_title)

        return {
            "status": "success",
            "message": message.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages/{msg_id}", response_model=Dict[str, Any])
async def delete_message(msg_id: str):
    """
    Delete a specific message.
    """
    try:
        chat_history = get_chat_history()
        deleted = chat_history.delete_message(msg_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Message not found")

        return {
            "status": "success",
            "message": f"Message {msg_id} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Utility Endpoints ====================

@router.get("/stats", response_model=Dict[str, Any])
async def get_chat_stats():
    """
    Get chat history statistics.
    """
    try:
        chat_history = get_chat_history()
        stats = chat_history.get_stats()

        return {
            "status": "success",
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting chat stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=Dict[str, Any])
async def clear_chat_history(
    confirm: bool = Query(False, description="Must be true to confirm deletion")
):
    """
    Clear all chat history. Requires confirm=true.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must pass confirm=true to clear all chat history"
        )

    try:
        chat_history = get_chat_history()
        result = chat_history.clear_all()

        return {
            "status": "success",
            "message": "Chat history cleared",
            **result
        }

    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

