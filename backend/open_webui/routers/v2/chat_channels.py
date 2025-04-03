import logging
import os
import uuid
import traceback
import time
import requests

from typing import Optional, Awaitable

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Query,
    BackgroundTasks,
)
from fastapi.responses import Response

from open_webui.env import SRC_LOG_LEVELS
from open_webui.chat_channels import facebook
from open_webui.models.chats import ChatForm
from open_webui.models.chat_channels import UserMessage, ChatChannelWebhookInfo
from open_webui.models.models import Models, ModelModel
from open_webui.utils.models import (
    get_all_models,
)
from open_webui.models.users import Users
from open_webui.models.chats import (
    Chats,
)
from open_webui.utils.chat import (
    generate_chat_completion,
)
from open_webui.utils.middleware import process_chat_payload

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["WEBHOOK"])

router = APIRouter()

FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
FACEBOOK_VERIFY_TOKEN = os.getenv("FACEBOOK_VERIFY_TOKEN")

PAGE_ID_TO_USER_ID = {"560161373845989": "89bd1078-76dc-4f40-9651-c98829fc8a86"}
PAGE_ID_TO_MODEL_ID = {"560161373845989": "vinh-cara-51-cskh-gpt-4o"}


def get_user_id_from_page_id(page_id: str) -> str:
    return PAGE_ID_TO_USER_ID[page_id]


def get_model_id_from_page_id(page_id: str) -> str:
    return PAGE_ID_TO_MODEL_ID[page_id]


def get_model_info_from_id(model_id: str) -> dict:
    model: ModelModel = Models.get_model_by_id(model_id)
    return model.model_dump()

def get_user_info(sender_id: str) -> dict:
    """Get the facebook user info from the sender id"""
    url = f"https://graph.facebook.com/{sender_id}"
    params = {
        "access_token": FACEBOOK_PAGE_ACCESS_TOKEN,
        "fields": "first_name,last_name,gender"
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(data)
    return data


def construct_chat_form(
    chat_id: str,
    message_id: str,
    model_id: str,
    user_message: str,
    timestamp: str,
) -> ChatForm:
    chat_data = {
        "id": "",
        "title": f"facebook_{chat_id}",
        "models": [model_id],
        "params": {},
        "history": {
            "messages": {
                message_id: {
                    "id": message_id,
                    "parentId": None,
                    "childrenIds": [],
                    "role": "user",
                    "content": user_message,
                    "timestamp": timestamp,
                    "models": [model_id],
                }
            },
            "currentId": message_id,
        },
        "messages": [
            {
                "id": message_id,
                "parentId": None,
                "childrenIds": [],
                "role": "user",
                "content": user_message,
                "timestamp": timestamp,
                "models": [model_id],
            }
        ],
        "tags": [],
        "timestamp": timestamp,
    }

    return ChatForm(chat=chat_data)


async def create_new_chat_session(
    user_id: str,
    chat_id: str,
    message_id: str,
    model_id: str,
    user_message: str,
    timestamp: str,
):
    """Create new chat session if not exists"""
    chat = Chats.get_chat_by_id(id=chat_id)
    if chat:
        # Get existing messages
        history_messages = chat.chat.get("history", {}).get("messages", {})
        if history_messages:
            # Get the last message
            previous_message_id = list(history_messages.keys())[-1]
            previous_message = history_messages[previous_message_id]

            # Create new user message with parent/child relationship
            user_message_data = {
                "id": message_id,
                "parentId": previous_message_id,
                "childrenIds": [],
                "role": "user",
                "content": user_message,
                "timestamp": timestamp,
                "models": [model_id],
            }

            # Update previous message's childrenIds
            previous_message["childrenIds"].append(message_id)
            Chats.upsert_message_to_chat_by_id_and_message_id(
                id=chat_id,
                message_id=previous_message_id,
                message=previous_message,
            )

            # Add new user message
            Chats.upsert_message_to_chat_by_id_and_message_id(
                id=chat_id,
                message_id=message_id,
                message=user_message_data,
            )
    else:
        form_data = construct_chat_form(
            chat_id=chat_id,
            message_id=message_id,
            model_id=model_id,
            user_message=user_message,
            timestamp=timestamp,
        )
        chat = Chats.insert_new_chat(
            user_id=user_id, form_data=form_data, chat_id=chat_id
        )
        log.info(f"New chat created: {chat.model_dump()}")
    return chat


def construct_input_form_data(
    model_id: str = None,
    messages: list = None,
    tool_ids: list = None,
    params: dict = {},
    session_id: str = None,
    chat_id: str = None,
    message_id: str = None,
) -> dict:
    """Construct input form data for chat completion
    This form data is mimics the form data of the /chat/completions endpoint
    It only works when we are using the Custom Model
    """
    return {
        "stream": False,
        "model": model_id,
        "messages": messages,
        "params": params,
        "tool_ids": tool_ids,
        "features": {"image_generation": False, "web_search": False},
        "session_id": session_id,
        "chat_id": chat_id,
        "id": message_id,
        "background_tasks": {"title_generation": False, "tags_generation": False},
    }


def postprocess_response(response: str):
    """Postprocess the message before sending it to the user"""

    # 1. Remove the reference to a file with pattern [file_name.extension]
    return response


def convert_history_messages_to_openai_format(history_messages: dict) -> list:
    """Convert the history messages to the OpenAI format"""
    return [
        {"role": message["role"], "content": message["content"]}
        for _, message in history_messages.items()
    ]


def save_message_to_chat_database(
    chat_id: str,
    model_info: dict,
    assistant_message_id: str,
    assistant_message: str,
    history_messages: dict,
):
    """Save the message to the chat database"""
    # Previous message will always be the user message
    previous_message_id = list(history_messages.keys())[-1]
    previous_message = history_messages[previous_message_id]
    # Add the assistant message to the childrenIds of the previous message
    previous_message["childrenIds"].append(assistant_message_id)
    # Update the history messages
    Chats.upsert_message_to_chat_by_id_and_message_id(
        id=chat_id,
        message_id=previous_message_id,
        message=previous_message,
    )

    # Add the assistant message to the chat history
    current_timestamp = int(time.time() * 1000)
    assistant_message = {
        "id": assistant_message_id,
        "parentId": previous_message_id,
        "childrenIds": [],
        "role": "assistant",
        "content": assistant_message,
        "timestamp": current_timestamp,
        "models": [previous_message["models"][0]],
        "model": model_info["id"],
        "modelName": model_info["name"],
        "modelIdx": 0
    }
    Chats.upsert_message_to_chat_by_id_and_message_id(
        id=chat_id,
        message_id=assistant_message_id,
        message=assistant_message,
    )


async def chat_completion_handler(request: Request, user_message: UserMessage):
    model_info = get_model_info_from_id(user_message.metadata["model_id"])
    user = Users.get_user_by_id(user_message.metadata["user_id"])
    log.info(f"User: {user}")
    log.info(f"User message metadata: {user_message.metadata}")
    # Get chat history from the database
    history_messages: dict = Chats.get_messages_by_chat_id(
        user_message.metadata["chat_id"]
    )
    log.info(f"History messages: {history_messages}")
    chat_history = convert_history_messages_to_openai_format(history_messages)
    chat_history.append({"role": "user", "content": user_message.text})

    # Construct the form data for the chat completion
    assistant_message_id = str(uuid.uuid4())
    form_data = construct_input_form_data(
        model_id=user_message.metadata["model_id"],
        messages=chat_history,
        tool_ids=model_info["meta"].get("toolIds", []),
        params={},
        session_id=user_message.metadata["chat_id"],
        chat_id=user_message.metadata["chat_id"],
        message_id=assistant_message_id,
    )
    
    # Add 

    tasks = form_data.pop("background_tasks", None)
    try:
        model_id = form_data.get("model", None)
        if model_id not in request.app.state.MODELS:
            raise Exception("Model not found")

        model = request.app.state.MODELS[model_id]

        metadata = {
            "user_id": user_message.metadata["user_id"],
            "chat_id": form_data.pop("chat_id", None),
            "message_id": form_data.pop("id", None),
            "session_id": form_data.pop("session_id", None),
            "tool_ids": form_data.get("tool_ids", None),
            "files": form_data.get("files", None),
            "features": form_data.get("features", None),
        }
        form_data["sender_info"] = user_message.metadata["sender_info"]

        form_data, metadata, events = await process_chat_payload(
            request=request, form_data=form_data, metadata=metadata, user=user, model=model
        )
    except Exception as e:
        log.error(f"Error processing chat completion: {e}")
        log.error(traceback.format_exc())
        return "Sorry, our system is currently experiencing issues. Please try again later."

    try:
        response = await generate_chat_completion(request, form_data, user)
        log.info(f"Chat completion response: {response}")

        # Get the content from the response
        content = response.get("choices", [])[0].get("message", {}).get("content")
        content = postprocess_response(content)

        # Save message in the database
        save_message_to_chat_database(
            chat_id=user_message.metadata["chat_id"],
            model_info=model_info,
            assistant_message_id=assistant_message_id,
            assistant_message=content,
            history_messages=history_messages,
        )

        return content

    except Exception as e:
        log.error(f"Error processing chat completion: {e}")
        return "Sorry, our system is currently experiencing issues. Please try again later."


@router.get("/")
async def health():
    return {"status": "ok"}


@router.get("/facebook/webhook")
async def facebook_token_verification(
    hub_verify_token: str = Query(..., alias="hub.verify_token"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
):
    if hub_verify_token == FACEBOOK_VERIFY_TOKEN:
        log.info(f"Facebook token verification successful: {hub_challenge}")
        return Response(content=hub_challenge, media_type="text/plain")
    else:
        log.warning(
            "Invalid fb verify token! Make sure this matches "
            "your webhook settings on the facebook app."
        )
        return "failure, invalid token"


@router.post("/facebook/webhook")
async def facebook_webhook(request: Request, background_tasks: BackgroundTasks):
    if not request.app.state.MODELS:
        await get_all_models(request)

    signature = request.headers.get("X-Hub-Signature", "")
    body = await request.body()
    log.info(f"Facebook webhook received: {body}")

    # Validate the webhook signature
    if not facebook.validate_hub_signature(FACEBOOK_APP_SECRET, body, signature):
        log.warning(
            "Wrong fb secret! Make sure this matches the "
            "secret in your facebook app settings"
        )
        return "not validated"

    # Get the webhook payload
    payload = await request.json()

    # Add message processing to background tasks
    background_tasks.add_task(process_facebook_message, request, payload)

    return {"success": True}


async def process_facebook_message(request: Request, payload: dict):
    try:
        chat_info: ChatChannelWebhookInfo = facebook.get_info_from_webhook(payload)
        log.info(
            f"Page ID: {chat_info.page_id}, Chat ID: {chat_info.chat_id}, Message ID: {chat_info.message_id}"
        )

        user_id = get_user_id_from_page_id(chat_info.page_id)
        model_id = get_model_id_from_page_id(chat_info.page_id)
        # Create new chat session if not exists
        await create_new_chat_session(
            user_id=user_id,
            chat_id=chat_info.chat_id,
            message_id=chat_info.message_id,
            model_id=model_id,
            user_message=chat_info.content,
            timestamp=chat_info.timestamp,
        )

        # Initialize the messenger for parsing and s
        messenger = facebook.Messenger(
            page_access_token=FACEBOOK_PAGE_ACCESS_TOKEN,
            on_new_message=chat_completion_handler,
        )
        
        # Get sender info
        ## Chat id is sender id
        sender_info = get_user_info(chat_info.chat_id)
        log.info(f"Sender info: {sender_info}")

        metadata = {
            "model_id": model_id,
            "user_id": user_id,
            "chat_id": chat_info.chat_id,
            "message_id": chat_info.message_id,
            "timestamp": chat_info.timestamp,
            "sender_info": sender_info,
        }
        # Handle the webhook and send a response back to the facebook user
        await messenger.handle(request=request, payload=payload, metadata=metadata)
    except Exception as e:
        log.error(f"Error processing Facebook message: {e}")
        traceback.print_exc()
