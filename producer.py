import base64
import time
import configparser
import logging
import pika
import sys
import uuid
import json
from aiohttp import web



stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(log_formatter)
log_level = logging.INFO #TODO later
logger = logging.getLogger("producer")
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

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
channel = connection.channel()

def createQueue(channel, queue):
    try:
        channel.queue_declare(queue=queue, passive=True)
        logger.info(f"Queue '{queue}' already exists.")
    except:
        # Queue does not exist, create it
        channel.queue_declare(queue=queue)
        logger.info(f"Queue '{queue}' created.")

async def producer_message(request):
    try:
        data = await request.post()
        # message = request.json['message']
        unique_id = str(uuid.uuid4())
        message = data.get('image')
        if message:
            image = message.file.read()
            image_data = base64.b64encode(image).decode()
            
            channel.basic_publish(exchange='', routing_key=message_queue, body=json.dumps({
                                                                        'data': image_data
                                                                    }), properties=pika.BasicProperties(reply_to = acknowledgment_queue, 
                                                          correlation_id = unique_id),)
            
            logger.info("{} Pushed image to the queue".format(unique_id))

            logger.info("{} waiting for response from consumer".format(unique_id))

           
            timeout = 10
            start_time = time.time()

            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    print(f"Timeout: Did not receive acknowledgment for unique ID: {unique_id}")
                    break

                method_frame, header_frame, body = channel.basic_get(queue=acknowledgment_queue, auto_ack=True)
                
                if method_frame:
                    # print(f"{method_frame}, {header_frame}, {body}")
                    if header_frame.correlation_id == unique_id:
                        logger.info(f"{unique_id} Received ack response from consumer")
                        break
                    else:
                        logger.info("different uniqueID recieved")                

            connection.close()
            return web.json_response({"status":"Success"})
        else:
            logger.info("Image not provided in the request input")
            return web.json_response({"status": "request failed"})

    except Exception as e:
        return web.json_response({"error": str(e)})


async def producer_status(request):
    try:
        return web.json_response({"Status":"RabitMQ is running"})
    except:
        return web.json_response({"Status":"RabitMQ is down"})

app = web.Application()
app.router.add_get('/', producer_status)
app.router.add_post('/pushImage', producer_message)

if __name__ == '__main__':
    createQueue(channel, message_queue)
    createQueue(channel, acknowledgment_queue)
    web.run_app(app)

