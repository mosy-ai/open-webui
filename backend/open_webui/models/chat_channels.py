from pydantic import BaseModel
from typing import Optional

class UserMessage(BaseModel):
    """Creates a ``UserMessage`` object.

    Args:
        text: the message text content.
        page_id: the page ID (Facebook Page ID).
        sender_id: the message owner ID (user ID).
        input_channel: the name of the channel which received this message.
        metadata: additional metadata for this message.
    """
    text: str
    page_id: str
    sender_id: str
    input_channel: str
    metadata: Optional[dict] = None

class ChatChannelWebhookInfo(BaseModel):
    page_id: str
    chat_id: str
    message_id: str
    content: str
    timestamp: int
