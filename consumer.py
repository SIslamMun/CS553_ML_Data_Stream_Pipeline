import json
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
log_level = logging.INFO
logger = logging.getLogger("consumer")
logger.setLevel(log_level)
logger.addHandler(stream_handler)

def load_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

config = load_config()

rabbitmq_host = config.get("rabitmq", "host")
message_queue = config.get("rabitmq", "message_queue")
acknowledgment_queue = config.get("rabitmq", "acknowledgment_queue")


def callback(ch, method, properties, body):
    logger.info("Recieved a message from queue")
    message = body.decode() 
    message = json.loads(message)
    image_bytes = base64.b64decode(message['data'])
    timestamp = int(time.time())
    filename = f"received_image_{timestamp}.png"
    image_path = os.path.join('received_images', filename)  
    with open(image_path, 'wb') as image_file:
        logger.info(f"Saving image with filename {filename} in location {image_path}")
        image_file.write(image_bytes)
    
    cor = properties.correlation_id
    
    # Send reply back to producer
    logger.info(f"Pushing acknowledgement for the message with correlation_id {cor}")
    ch.basic_publish(exchange='',
                     routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id = cor),
                     body='Reply to '+cor) 
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
channel = connection.channel()

channel.queue_declare(queue=acknowledgment_queue)
channel.queue_declare(queue=message_queue)

channel.basic_consume(queue=message_queue, on_message_callback=callback)

logger.info("Waiting for messages. To exit, press CTRL+C")
channel.start_consuming()
