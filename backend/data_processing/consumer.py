import os
import sys
import json
import time
import logging
from aio_pika import IncomingMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_webui.models.files import (
    FileModelResponse,
    Files
)

log = logging.getLogger(__name__)

# Factory to get message model based on queue name
def get_message_model(queue_name: str):
    model_map = {
        "documents": FileModelResponse,
    }
    return model_map.get(queue_name)

# Consumer logic for processing messages
async def consume_message(queue_name: str, message: IncomingMessage):
    try:
        # Get the appropriate model for the queue
        message_model_cls = get_message_model(queue_name)
        if not message_model_cls:
            log.error(f"No message model defined for queue '{queue_name}'")
            await message.nack(requeue=False)
            return

        # Parse the message
        message_body = message.body.decode()
        message_data = json.loads(message_body)
        parsed_message = message_model_cls(**message_data)

        # Process the parsed message
        await process_message(queue_name, parsed_message)

        # Acknowledge the message
        await message.ack()
    except Exception as e:
        log.error(f"Error consuming message from queue '{queue_name}': {e}")
        await message.nack(requeue=False)

@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=2, min=1, max=10),  # Exponential backoff
    retry=retry_if_exception_type(Exception),  # Retry on exceptions
)
async def process_message_with_retry(queue_name: str, parsed_message):
    if queue_name == "documents":
        file_data = parsed_message.dict()
        file_id = file_data.get("id")

        # TODO: Index file into Vector DB if success update the status
        update_data = {
            "filename": file_data["filename"],
            "data": file_data.get("data", {}),
            "meta": file_data["meta"],
            "path": file_data["path"],
            "status": "success",
            "error_message": file_data["error_message"],
            "updated_at": int(time.time()),
        }

        # Attempt to update the file in the database
        updated_file = Files.update_file_data_by_id(file_id, update_data)
        if not updated_file:
            raise Exception(f"Failed to update file with ID: {file_id}")

        log.info(f"Successfully updated file with ID: {file_id}")

# Main function for processing the message with retry
async def process_message(queue_name: str, parsed_message):
    try:
        await process_message_with_retry(queue_name, parsed_message)
    except Exception as e:
        log.error(f"Final failure for message on queue '{queue_name}': {e}")