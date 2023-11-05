import pika
import configparser
import logging
import sys
import os
import base64
import time

stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(log_formatter)

logger = logging.getLogger("pipeline")

def load_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

config = load_config()

rabbitmq_host = config.get("rabitmq", "host")
queue = config.get("rabitmq", "queue")


def callback(ch, method, properties, body):
    image_data = body.decode()
    image_bytes = base64.b64decode(image_data)
    # print(f"Received: {body}")
    timestamp = int(time.time())
    filename = f"received_image_{timestamp}.png"
    image_path = os.path.join('received_images', filename)  # You can change the file format as needed
    with open(image_path, 'wb') as image_file:
        image_file.write(image_bytes)

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
channel = connection.channel()

channel.queue_declare(queue=queue)

channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)

logger.info("Waiting for messages. To exit, press CTRL+C")
channel.start_consuming()
