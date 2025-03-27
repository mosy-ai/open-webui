import aio_pika
import logging

from open_webui.env import (
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    SRC_LOG_LEVELS,
)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RABBITMQ"])

async def get_rabbitmq_channel():
    connection = await aio_pika.connect_robust(
        f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/",
    )
    channel = await connection.channel()
    return connection, channel

# Push file path to RabbitMQ queue
async def push_to_queue(queue_name: str, file_path: str):
    connection, channel = await get_rabbitmq_channel()
    try:
        # Declare a queue
        queue = await channel.declare_queue(queue_name, durable=True)

        # Send the message
        message = aio_pika.Message(body=file_path.encode())
        await channel.default_exchange.publish(message, routing_key=queue.name)

    finally:
        await connection.close()